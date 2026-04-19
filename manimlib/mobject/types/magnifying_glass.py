from __future__ import annotations

from manimlib.mobject.mobject import Mobject, Group
from manimlib.scene.scene import Scene

import OpenGL.GL as gl
import numpy as np


class MagnifyingGlass(Mobject):
    """A circular magnifier overlay for the current scene.

    By default, this mobject re-renders the scene into the lens region using
    modified camera uniforms (no raster texture zoom), preserving vector detail.

    If ``rasterize=True``, it instead captures the already-rendered frame into
    an offscreen texture every frame and samples from it (legacy behavior).

    Notes
    -----
    - The magnifier is fixed in the frame by default (UI-style overlay).
    - Add it to the scene like any other mobject; Scene.render_groups is
      reordered so magnifiers render last.
    """

    shader_folder = "magnifying_glass"
    data_dtype = np.dtype([("point", np.float32, (3,))])

    # Used by Scene.assemble_render_groups to ensure these render last
    is_magnifying_glass = True

    def __init__(
        self,
        scene: Scene,
        radius: float = 1.0,
        magnification: float = 2.0,
        fixed_in_frame: bool = True,
        rasterize: bool = False,
        **kwargs,
    ):
        self.scene = scene
        self._radius = float(radius)
        self._magnification = float(magnification)
        self._rasterize = bool(rasterize)

        super().__init__(**kwargs)

        # Prevent instances from batching together (each has its own runtime texture).
        self.uniforms["_magnify_uid"] = float(id(self))

        # Shader branch for legacy (texture sampling) vs re-rendered magnification.
        self.uniforms["rasterize"] = 1.0 if self._rasterize else 0.0

        def init_gl_resources() -> None:
            ctx = self.scene.camera.ctx
            size = self.scene.camera.get_pixel_shape()
            self.scene_tex = ctx.texture(
                size, components=4, dtype="f1"
            )
            self.scene_fbo = ctx.framebuffer(
                color_attachments=self.scene_tex
            )

            # Offscreen target for non-rasterized magnification.
            # We re-render the scene with modified camera uniforms into this
            # framebuffer, then sample it inside the lens circle.
            self.zoom_tex = ctx.texture(
                size, components=4, dtype="f1"
            )
            self.zoom_depth = ctx.depth_renderbuffer(size)
            self.zoom_fbo = ctx.framebuffer(
                color_attachments=self.zoom_tex,
                depth_attachment=self.zoom_depth,
            )

        init_gl_resources()

        if fixed_in_frame:
            self.fix_in_frame()

        self.set_radius(self._radius)
        self.set_magnification(self._magnification)

        # Initialize lens_center; it will be refreshed every render() call.
        self.uniforms["lens_center"] = np.array(
            self.get_center(), dtype=float
        )

    def init_shader_wrapper(self, ctx):
        super().init_shader_wrapper(ctx)
        self.shader_wrapper.add_texture("Scene", self.scene_tex)

        # Used when rasterize=False: contains a re-rendered, zoomed view.
        self.shader_wrapper.add_texture("Zoomed", self.zoom_tex)

        # Capture the already-rendered scene into the lens texture right before
        # drawing the lens. This avoids relying on updater timing/order and
        # ensures the magnifier samples the exact current frame.
        original_pre_render = self.shader_wrapper.pre_render

        def pre_render_with_capture() -> None:
            if self.shader_wrapper is None:
                return
            if self.shader_wrapper.is_hidden():
                return

            camera = self.scene.camera

            # Resize runtime textures/fbos on window resize.
            self._ensure_runtime_texture_size(ctx)

            try:
                camera.blit(camera.fbo, self.scene_fbo)
            except gl.error.GLError:
                pass
            finally:
                # Restore draw target
                camera.fbo.use()

            original_pre_render()

        self.shader_wrapper.pre_render = pre_render_with_capture

    def _ensure_runtime_texture_size(self, ctx) -> None:
        camera = self.scene.camera
        size = camera.get_pixel_shape()
        if (
            getattr(self.scene_tex, "size", None) == size
            and getattr(self.zoom_tex, "size", None) == size
        ):
            return

        wrapper = getattr(self, "shader_wrapper", None)
        if wrapper is not None:
            # Releases old textures (Scene/Zoomed) so we don't double-release.
            wrapper.release_textures()

        # Release old framebuffer objects and depth attachment.
        if getattr(self, "scene_fbo", None) is not None:
            self.scene_fbo.release()
        if getattr(self, "zoom_fbo", None) is not None:
            self.zoom_fbo.release()
        if getattr(self, "zoom_depth", None) is not None:
            self.zoom_depth.release()

        # If the wrapper didn't own textures yet (unlikely during render), release them here.
        if wrapper is None:
            if getattr(self, "scene_tex", None) is not None:
                self.scene_tex.release()
            if getattr(self, "zoom_tex", None) is not None:
                self.zoom_tex.release()

        # Recreate runtime textures/fbos.
        self.scene_tex = ctx.texture(size, components=4, dtype="f1")
        self.scene_fbo = ctx.framebuffer(
            color_attachments=self.scene_tex
        )

        self.zoom_tex = ctx.texture(size, components=4, dtype="f1")
        self.zoom_depth = ctx.depth_renderbuffer(size)
        self.zoom_fbo = ctx.framebuffer(
            color_attachments=self.zoom_tex,
            depth_attachment=self.zoom_depth,
        )

        # Rebind textures in the wrapper (if it exists already).
        if wrapper is not None:
            wrapper.add_texture("Scene", self.scene_tex)
            wrapper.add_texture("Zoomed", self.zoom_tex)

    def _lens_center_ndc(self, camera_uniforms: dict) -> np.ndarray:
        """Compute the lens center in NDC coordinates (x,y in [-1, 1])."""
        point = np.array([*self.get_center(), 1.0], dtype=float)
        is_fixed = float(self.uniforms.get("is_fixed_in_frame", 0.0))

        if is_fixed < 0.5:
            view = self.scene.camera.frame.get_view_matrix()
            point = np.dot(view, point)

        fx, fy, fz = camera_uniforms.get(
            "frame_rescale_factors", (1.0, 1.0, 1.0)
        )
        point[0] *= fx
        point[1] *= fy
        point[2] *= fz

        w = 1.0 - point[2]
        if abs(w) < 1e-8:
            w = 1e-8
        return point[:2] / w

    def _iter_non_magnifier_groups(self):
        for group in getattr(self.scene, "render_groups", []):
            submobs = getattr(group, "submobjects", [])
            if any(
                getattr(m, "is_magnifying_glass", False)
                for m in submobs
            ):
                continue
            yield group

    def get_group_class(self):
        return MagnifyingGlassGroup

    def render(self, ctx, camera_uniforms: dict):
        # Keep lens_center uniform tightly in sync with transforms.
        # Relying on updater ordering can desync this value under threaded/parallel
        # updater execution, which would cause the shader to discard the whole quad.
        self.uniforms["lens_center"] = np.array(
            self.get_center(), dtype=float
        )

        # Preserve legacy behavior
        if self._rasterize:
            return super().render(ctx, camera_uniforms)

        camera = self.scene.camera
        self._ensure_runtime_texture_size(ctx)

        # Re-render the scene with a zoomed camera into an offscreen framebuffer.
        # This preserves vector detail and avoids relying on scissor with internal
        # offscreen passes (e.g., VMobject fill compositing).
        ndc = self._lens_center_ndc(camera_uniforms)
        zoom = max(self._magnification, 1e-6)
        zoom_uniforms = dict(camera_uniforms)
        zoom_uniforms["magnify_active"] = 1.0
        zoom_uniforms["magnify_center"] = (
            float(ndc[0]),
            float(ndc[1]),
        )
        zoom_uniforms["magnify_zoom"] = float(zoom)
        if "pixel_size" in zoom_uniforms:
            zoom_uniforms["pixel_size"] = (
                float(zoom_uniforms["pixel_size"]) / zoom
            )

        old_fbo = camera.fbo
        try:
            camera.fbo = self.zoom_fbo
            camera.fbo.use()
            self.zoom_fbo.clear(*camera.background_rgba)
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            for group in self._iter_non_magnifier_groups():
                group.render(ctx, zoom_uniforms)
        finally:
            camera.fbo = old_fbo
            camera.fbo.use()

        return super().render(ctx, camera_uniforms)

    def init_points(self):
        # Quad for GL_TRIANGLE_STRIP: UL, DL, UR, DR in local coordinates.
        # We size it to the desired radius via set_radius().
        from manimlib.constants import UL, DL, UR, DR

        self.set_points([UL, DL, UR, DR])

    def set_color(self, color, opacity=None, recurse=True):
        # This mobject is fully shader-driven; it doesn't use per-vertex rgba.
        return self

    def set_radius(self, radius: float) -> "MagnifyingGlass":
        self._radius = float(radius)
        diameter = 2.0 * self._radius
        self.set_height(diameter)
        self.set_width(diameter, stretch=True)
        self.set_uniform(lens_radius=self._radius)
        return self

    def get_radius(self) -> float:
        return self._radius

    def set_magnification(
        self, magnification: float
    ) -> "MagnifyingGlass":
        self._magnification = float(magnification)
        # Avoid division-by-zero in shader.
        self.set_uniform(magnification=max(self._magnification, 1e-6))
        return self

    def get_magnification(self) -> float:
        return self._magnification

    # Note: rasterize=True captures the current frame via ShaderWrapper.pre_render.
    # For rasterize=False, render() re-renders the scene into an offscreen target.


class MagnifyingGlassGroup(Group):
    is_magnifying_glass = True

    def __init__(self, *children: MagnifyingGlass, **kwargs):
        super().__init__(*children, **kwargs)

    def render(self, ctx, camera_uniforms: dict):
        for mob in self.submobjects:
            mob.render(ctx, camera_uniforms)
