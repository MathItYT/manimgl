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

    def update(self, dt=0, recurse=True):
        def update_textures() -> None:
            camera = self.scene.camera
            old_fbo = camera.fbo
            try:
                camera.fbo = self.src_fbo
                camera.show(self.src_mobject)
                camera.capture(
                    self.src_mobject, clear_window=False, transparent=True
                )
                camera.hide(self.src_mobject)

                camera.fbo = self.mask_fbo
                camera.show(self.mask_mobject)
                camera.capture(
                    self.mask_mobject,
                    clear_window=False,
                    transparent=True,
                )
                camera.hide(self.mask_mobject)
            except gl.error.GLError:
                pass
            finally:
                camera.fbo = old_fbo
                camera.fbo.use()

        caller = getattr(self.scene, "_main_thread_caller", None)
        threaded_active = bool(getattr(self.scene, "_threaded_mode_active", False))
        if threaded_active and caller is not None and not caller.is_main_thread():
            caller.call(update_textures)
        else:
            update_textures()

        super().update(dt, recurse)
        return self
