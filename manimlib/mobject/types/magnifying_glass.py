from __future__ import annotations

from manimlib.mobject.mobject import Mobject
from manimlib.scene.scene import Scene

import OpenGL.GL as gl
import numpy as np


class MagnifyingGlass(Mobject):
    """A circular magnifier overlay for the current scene.

    This mobject renders the scene into an offscreen texture every frame, then
    displays a magnified version inside a circular region of a given radius.

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
        **kwargs,
    ):
        self.scene = scene
        self._radius = float(radius)
        self._magnification = float(magnification)

        super().__init__(**kwargs)

        # Prevent instances from batching together (each has its own runtime texture).
        self.uniforms["_magnify_uid"] = float(id(self))

        def init_gl_resources() -> None:
            ctx = self.scene.camera.ctx
            size = self.scene.camera.get_pixel_shape()
            self.scene_tex = ctx.texture(size, components=4, dtype="f1")
            self.scene_fbo = ctx.framebuffer(color_attachments=self.scene_tex)

        # In threaded preview mode, GL calls must happen on the main OpenGL thread.
        caller = getattr(self.scene, "_main_thread_caller", None)
        threaded_active = bool(getattr(self.scene, "_threaded_mode_active", False))
        if threaded_active and caller is not None and not caller.is_main_thread():
            caller.call(init_gl_resources)
        else:
            init_gl_resources()

        if fixed_in_frame:
            self.fix_in_frame()

        self.set_radius(self._radius)
        self.set_magnification(self._magnification)

        # Keep lens_center in sync with transforms; this must run before rendering
        # because uniforms are pushed to the GPU right before draw.
        def center_updater(mob: "MagnifyingGlass", dt: float = 0) -> None:
            mob.uniforms["lens_center"] = np.array(mob.get_center(), dtype=float)

        self.insert_updater(center_updater, index=0)

    def init_shader_wrapper(self, ctx):
        super().init_shader_wrapper(ctx)
        self.shader_wrapper.add_texture("Scene", self.scene_tex)

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

            # Resize runtime texture/fbo on window resize.
            try:
                size = camera.get_pixel_shape()
                if getattr(self.scene_tex, "size", None) != size:
                    # Recreate
                    self.scene_tex.release()
                    self.scene_fbo.release()
                    self.scene_tex = ctx.texture(size, components=4, dtype="f1")
                    self.scene_fbo = ctx.framebuffer(color_attachments=self.scene_tex)
                    # Rebind texture in the wrapper
                    self.shader_wrapper.release_textures()
                    self.shader_wrapper.add_texture("Scene", self.scene_tex)
            except Exception:
                pass

            try:
                camera.blit(camera.fbo, self.scene_fbo)
            except gl.error.GLError:
                pass
            finally:
                # Restore draw target
                camera.fbo.use()

            original_pre_render()

        self.shader_wrapper.pre_render = pre_render_with_capture

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

    def set_magnification(self, magnification: float) -> "MagnifyingGlass":
        self._magnification = float(magnification)
        # Avoid division-by-zero in shader.
        self.set_uniform(magnification=max(self._magnification, 1e-6))
        return self

    def get_magnification(self) -> float:
        return self._magnification

    # Note: no render() override. The lens capture happens in ShaderWrapper.pre_render,
    # while lens_center is maintained via an updater.
