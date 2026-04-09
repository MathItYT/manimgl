from manimlib.constants import (
    DL,
    DR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    UL,
    UR,
)
from manimlib.mobject.mobject import Mobject
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
            self.mask_tex = ctx.texture(size, components=4, dtype="f1")

            self.src_fbo = ctx.framebuffer(color_attachments=self.src_tex)
            self.mask_fbo = ctx.framebuffer(
                color_attachments=self.mask_tex
            )

        # In threaded preview mode, scene logic may run on a worker thread.
        # Creating textures/fbos must happen on the main OpenGL thread.
        caller = getattr(self.scene, "_main_thread_caller", None)
        threaded_active = bool(getattr(self.scene, "_threaded_mode_active", False))
        if threaded_active and caller is not None and not caller.is_main_thread():
            caller.call(init_gl_resources)
        else:
            init_gl_resources()

        self.fix_in_frame()
        self.set_height(height)
        self.set_width(FRAME_WIDTH, stretch=True)

        # Refresh textures every frame via an updater so this works under
        # parallel-updater execution (which may bypass Mobject.update overrides).
        def texture_updater(mob: "MaskMobject", dt: float = 0) -> None:
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
            threaded_active = bool(getattr(mob.scene, "_threaded_mode_active", False))
            if threaded_active and caller is not None and not caller.is_main_thread():
                caller.call(update_textures)
            else:
                update_textures()

        # Avoid calling the updater immediately from inside __init__.
        self.insert_updater(texture_updater, index=0)

    def init_shader_wrapper(self, ctx):
        super().init_shader_wrapper(ctx)
        self.shader_wrapper.add_texture("Source", self.src_tex)
        self.shader_wrapper.add_texture("Mask", self.mask_tex)

    def init_points(self):
        # Pequeño Fix: Orden correcto para GL_TRIANGLE_STRIP
        # Asegura que el quad en pantalla completa no deje triángulos vacíos.
        self.set_points([UL, DL, UR, DR])

    def set_color(self, color, opacity=None, recurse=True):
        pass
    # Note: No update() override. Texture refresh is handled by the updater.
