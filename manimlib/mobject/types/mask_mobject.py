from manimlib.constants import (
    DL,
    DR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    UL,
    UR,
)
from manimlib.mobject.mobject import Mobject, Group
from manimlib.scene.scene import Scene
import OpenGL.GL as gl

import numpy as np


class MaskMobject(Mobject):
    shader_folder = "mask"
    data_dtype = np.dtype([("point", np.float32, (3,))])

    def __init__(
        self,
        scene: Scene,
        src_mobject: Mobject,
        mask_mobject: Mobject,
        height: float = FRAME_HEIGHT,
        **kwargs,
    ):
        self.scene = scene
        self.src_mobject = src_mobject
        self.mask_mobject = mask_mobject

        self.scene.camera.hide(self.src_mobject)
        self.scene.camera.hide(self.mask_mobject)

        super().__init__(**kwargs)

        # MaskMobject binds per-instance runtime textures.
        # Keep wrappers from being batched together across instances.
        self.uniforms["_mask_uid"] = float(id(self))

        def init_gl_resources() -> None:
            ctx = self.scene.camera.ctx
            size = self.scene.camera.get_pixel_shape()

            self.src_tex = ctx.texture(size, components=4, dtype="f1")
            self.mask_tex = ctx.texture(
                size, components=4, dtype="f1"
            )

            self.src_fbo = ctx.framebuffer(
                color_attachments=self.src_tex
            )
            self.mask_fbo = ctx.framebuffer(
                color_attachments=self.mask_tex
            )

        # In threaded preview mode, scene logic may run on a worker thread.
        # Creating textures/fbos must happen on the main OpenGL thread.
        caller = getattr(self.scene, "_main_thread_caller", None)
        threaded_active = bool(
            getattr(self.scene, "_threaded_mode_active", False)
        )
        if (
            threaded_active
            and caller is not None
            and not caller.is_main_thread()
        ):
            caller.call(init_gl_resources)
        else:
            init_gl_resources()

        self.fix_in_frame()
        self.set_height(height)
        self.set_width(FRAME_WIDTH, stretch=True)

        # Refresh textures every frame via an updater so this works under
        # parallel-updater execution (which may bypass Mobject.update overrides).
        def texture_updater(
            mob: "MaskMobject", dt: float = 0
        ) -> None:
            def update_textures() -> None:
                camera = mob.scene.camera
                old_fbo = camera.fbo
                try:
                    camera.fbo = mob.src_fbo
                    camera.show(mob.src_mobject, notify_render=False)
                    camera.capture(
                        mob.src_mobject,
                        clear_window=False,
                        transparent=True,
                    )
                    camera.hide(mob.src_mobject, notify_render=False)

                    camera.fbo = mob.mask_fbo
                    camera.show(mob.mask_mobject, notify_render=False)
                    camera.capture(
                        mob.mask_mobject,
                        clear_window=False,
                        transparent=True,
                    )
                    camera.hide(mob.mask_mobject, notify_render=False)
                except gl.error.GLError:
                    pass
                finally:
                    camera.fbo = old_fbo
                    camera.fbo.use()

            caller = getattr(mob.scene, "_main_thread_caller", None)
            threaded_active = bool(
                getattr(mob.scene, "_threaded_mode_active", False)
            )
            if (
                threaded_active
                and caller is not None
                and not caller.is_main_thread()
            ):
                caller.call(update_textures)
            else:
                update_textures()

        # Avoid calling the updater immediately from inside __init__.
        self.insert_updater(texture_updater, index=0)

    def get_group_class(self):
        # Ensure MaskMobject.render() is called (not batch-rendered via Group.render).
        # We need per-instance logic to refresh runtime textures under magnification.
        return MaskMobjectGroup

    def _capture_mobject_to_fbo(
        self,
        mobject: Mobject,
        fbo,
        ctx,
        camera_uniforms: dict,
    ) -> None:
        camera = self.scene.camera
        old_fbo = camera.fbo
        try:
            camera.fbo = fbo
            camera.fbo.use()
            camera.clear(clear_window=False, transparent=True)

            camera.show(mobject, notify_render=False)
            mobject.render(ctx, camera_uniforms)
            camera.hide(mobject, notify_render=False)
        finally:
            camera.fbo = old_fbo
            camera.fbo.use()

    def _refresh_textures_for_uniforms(
        self, ctx, camera_uniforms: dict
    ) -> None:
        """Refresh src/mask runtime textures using the given camera uniforms.

        This is required for MagnifyingGlass(rasterize=False), which re-renders
        the scene with magnifier uniforms. The default Camera.capture() path
        calls Camera.refresh_uniforms(), which resets magnify_* and would
        therefore produce un-magnified textures.
        """

        try:
            self._capture_mobject_to_fbo(
                self.src_mobject,
                self.src_fbo,
                ctx,
                camera_uniforms,
            )
            self._capture_mobject_to_fbo(
                self.mask_mobject,
                self.mask_fbo,
                ctx,
                camera_uniforms,
            )
        except gl.error.GLError:
            # If the GL state is transiently invalid (resize, context hiccup),
            # skip this frame rather than crashing.
            return

    def init_shader_wrapper(self, ctx):
        super().init_shader_wrapper(ctx)
        self.shader_wrapper.add_texture("Source", self.src_tex)
        self.shader_wrapper.add_texture("Mask", self.mask_tex)

    def render(self, ctx, camera_uniforms: dict):
        # If the scene is being re-rendered under magnifier uniforms (the
        # rasterize=False path of MagnifyingGlass), re-capture the internal
        # textures using those uniforms so the masked result magnifies too.
        if float(camera_uniforms.get("magnify_active", 0.0)) > 0.5:
            self._refresh_textures_for_uniforms(ctx, camera_uniforms)
        return super().render(ctx, camera_uniforms)

    def init_points(self):
        # Pequeño Fix: Orden correcto para GL_TRIANGLE_STRIP
        # Asegura que el quad en pantalla completa no deje triángulos vacíos.
        self.set_points([UL, DL, UR, DR])

    def set_color(self, color, opacity=None, recurse=True):
        pass

    # Note: No update() override. Texture refresh is handled by the updater.


class MaskMobjectGroup(Group):
    def render(self, ctx, camera_uniforms: dict):
        for mob in self.submobjects:
            mob.render(ctx, camera_uniforms)
