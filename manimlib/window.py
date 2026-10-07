from __future__ import annotations

import sys

import numpy as np
import wgpu

if sys.platform == "emscripten":
    from rendercanvas.pyodide import PyodideRenderCanvas
else:
    import glfw
    from rendercanvas.glfw import RenderCanvas

from manimlib.constants import ASPECT_RATIO
from manimlib.constants import DEFAULT_RESOLUTION
from manimlib.constants import FRAME_SHAPE
from manimlib.event_keys import Keys
from manimlib.event_keys import Mods
from manimlib.renderer.gpu import Gpu
from manimlib.renderer.shader_source import read_shader_file

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Sequence
    from manimlib.scene.scene import Scene


KEY_NAMES: dict[str, int] = {
    "Backspace": Keys.BACKSPACE,
    "Tab": Keys.TAB,
    "Enter": Keys.ENTER,
    "Escape": Keys.ESCAPE,
    "Delete": Keys.DELETE,
    "ArrowLeft": Keys.LEFT,
    "ArrowRight": Keys.RIGHT,
    "ArrowUp": Keys.UP,
    "ArrowDown": Keys.DOWN,
    "Shift": Keys.SHIFT,
    "Control": Keys.CTRL,
    "Alt": Keys.ALT,
    "Meta": Keys.CMD,
}
MOD_NAMES: dict[str, int] = {
    "Shift": Mods.SHIFT,
    "Control": Mods.CTRL,
    "Alt": Mods.ALT,
    "Meta": Mods.CMD,
}
WHEEL_NOTCH = 100.0
PRESENT_SHADER = "present.wgsl"
POSITION_STEPS = {"L": 0.0, "U": 0.0, "O": 0.5, "R": 1.0, "D": 1.0}


def to_key(name: str) -> Optional[int]:
    if name in KEY_NAMES:
        return KEY_NAMES[name]
    return ord(name.lower()) if len(name) == 1 else None


def to_mods(names: Sequence[str]) -> int:
    return sum(MOD_NAMES.get(name, 0) for name in names)


class Window(object):
    """
    Where a scene is previewed: somewhere to show a finished frame, and where mouse and key
    events come from.
    """

    def __init__(
        self,
        scene: Optional[Scene] = None,
        position_string: str = "UR",
        monitor_index: int = 1,
        full_screen: bool = False,
        size: Optional[tuple[int, int]] = None,
        position: Optional[tuple[int, int]] = None,
        canvas_id: str = "canvas",
        adapter=None,
        device=None,
    ):
        self.scene: Optional[Scene] = None
        self.canvas_id = canvas_id
        self.frame_view = None
        self.pressed_keys: set[int] = set()
        self.pointer_position = np.zeros(2)
        self.undrawn_event = True
        self.render_size = DEFAULT_RESOLUTION if sys.platform == "emscripten" else None

        if sys.platform == "emscripten":
            self.canvas = PyodideRenderCanvas(canvas_id, size=size, update_mode="manual")
            self.canvas.request_draw(self.draw)
            self.context = self.canvas.get_context("wgpu")
            self.gpu = Gpu(adapter=adapter, device=device)
            self.configure()
            self._install_browser_pointer_tracking()
        else:
            glfw.init()
            monitor = self.get_monitor(monitor_index)
            self.canvas = RenderCanvas(
                size=size or self.get_default_size(monitor, full_screen),
                update_mode="manual",
            )
            self.canvas.request_draw(self.draw)
            self.context = self.canvas.get_context("wgpu")
            self.gpu = Gpu()
            self.configure()
            glfw.set_window_pos(self.glfw_window, *(
                position or self.get_position(monitor, position_string)
            ))

        for event_type, handler in [
            ("pointer_move", self.on_pointer_move),
            ("pointer_down", self.on_pointer_down),
            ("pointer_up", self.on_pointer_up),
            ("wheel", self.on_wheel),
            ("key_down", self.on_key_down),
            ("key_up", self.on_key_up),
            ("resize", self.on_resize),
            ("close", self.on_close),
        ]:
            self.canvas.add_event_handler(handler, event_type)

        if scene:
            self.init_for_scene(scene)

    @classmethod
    async def create_for_pyodide(cls, canvas_id: str = "canvas", **kwargs):
        if sys.platform != "emscripten":
            raise RuntimeError("create_for_pyodide() is only available in Pyodide.")

        adapter = await wgpu.gpu.request_adapter_async(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError(
                "WebGPU no está disponible en este navegador. "
                "Se necesita un navegador compatible con WebGPU para ejecutar ManimGL."
            )

        device = await adapter.request_device_async()
        if device is None:
            raise RuntimeError(
                "WebGPU no pudo crear un dispositivo en este navegador. "
                "Comprueba que WebGPU esté habilitado y que el dispositivo sea compatible."
            )

        return cls(canvas_id=canvas_id, adapter=adapter, device=device, **kwargs)

    @property
    def glfw_window(self):
        if sys.platform == "emscripten":
            raise AttributeError("Pyodide windows do not have a GLFW window.")
        return self.canvas._window

    def init_for_scene(self, scene: Scene) -> None:
        self.pressed_keys.clear()
        self.undrawn_event = True
        self.scene = scene
        self.canvas.set_title(str(scene))

    def configure(self) -> None:
        self.device = self.gpu.device
        preferred = self.context.get_preferred_format(self.gpu.adapter)
        self.format = preferred.removesuffix("-srgb")
        self.context.configure(device=self.device, format=self.format)
        self.init_present_resources()

    def _install_browser_pointer_tracking(self) -> None:
        from js import eval

        eval(
            """
            (() => {
                const canvas = document.getElementById(%r);
                if (!canvas || canvas.__manimPointerTrackingInstalled) {
                    return;
                }

                const update = (event) => {
                    const rect = canvas.getBoundingClientRect();
                    canvas.__manimPointer = {
                        clientX: event.clientX,
                        clientY: event.clientY,
                        left: rect.left,
                        top: rect.top,
                        width: rect.width,
                        height: rect.height,
                    };
                };

                for (const type of ["pointermove", "pointerdown", "pointerup", "wheel"]) {
                    canvas.addEventListener(type, update, true);
                }
                canvas.__manimPointerTrackingInstalled = true;
            })();
            """ % self.canvas_id
        )

    def get_size(self) -> tuple[int, int]:
        return self.canvas.get_physical_size()

    def show(self, frame_view) -> None:
        self.frame_view = frame_view
        self.canvas.force_draw()
        self.undrawn_event = False
        self.poll_events()

    def draw(self) -> None:
        self.present(self.context.get_current_texture().create_view())

    def init_present_resources(self) -> None:
        self.present_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.float}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": wgpu.SamplerBindingType.filtering}},
        ])
        self.present_sampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear,
        )
        module = self.gpu.module(read_shader_file(PRESENT_SHADER))
        self.present_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self.present_layout],
            ),
            vertex={"module": module, "entry_point": "vs_main"},
            fragment={
                "module": module,
                "entry_point": "fs_main",
                "targets": [{"format": self.format}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

    def present(self, target_view) -> None:
        bind_group = self.device.create_bind_group(layout=self.present_layout, entries=[
            {"binding": 0, "resource": self.frame_view},
            {"binding": 1, "resource": self.present_sampler},
        ])
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0.0, 0.0, 0.0, 1.0),
        }])
        render_pass.set_pipeline(self.present_pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.draw(3)
        render_pass.end()
        self.gpu.queue.submit([encoder.finish()])

    def poll_events(self) -> None:
        self.canvas._process_events()

    @property
    def is_closing(self) -> bool:
        return self.canvas.get_closed()

    def has_undrawn_event(self) -> bool:
        return self.undrawn_event

    def is_key_pressed(self, key: int) -> bool:
        return key in self.pressed_keys

    def focus(self) -> None:
        if sys.platform == "emscripten":
            return
        glfw.focus_window(self.glfw_window)

    def destroy(self) -> None:
        if sys.platform != "emscripten":
            self.canvas.close()

    def get_monitor(self, index: int):
        monitors = glfw.get_monitors()
        return monitors[min(index, len(monitors) - 1)] if monitors else None

    def get_monitor_area(self, monitor) -> tuple[int, int, int, int]:
        if monitor is None:
            return (0, 0, 1920, 1080)
        return glfw.get_monitor_workarea(monitor)

    def get_default_size(self, monitor, full_screen: bool) -> tuple[int, int]:
        _, _, width, _ = self.get_monitor_area(monitor)
        if not full_screen:
            width //= 2
        return (width, int(width / ASPECT_RATIO))

    def get_position(self, monitor, position_string: str) -> tuple[int, int]:
        left, top, width, height = self.get_monitor_area(monitor)
        size = self.canvas.get_logical_size()
        return (
            int(left + POSITION_STEPS[position_string[1]] * (width - size[0])),
            int(top + POSITION_STEPS[position_string[0]] * (height - size[1])),
        )

    def note_event(self) -> None:
        self.undrawn_event = True

    def pixel_coords_to_space_coords(
        self,
        px: float,
        py: float,
        relative: bool = False
    ) -> np.ndarray:
        if self.scene is None or not hasattr(self.scene, "frame"):
            return np.zeros(3)

        pixel_shape = np.array(
            self.render_size if sys.platform == "emscripten" else self.canvas.get_logical_size()
        )
        fixed_frame_shape = np.array(FRAME_SHAPE)
        frame = self.scene.frame

        coords = np.zeros(3)
        coords[:2] = (fixed_frame_shape / pixel_shape) * np.array([px, py])
        if not relative:
            coords[:2] -= 0.5 * fixed_frame_shape
        return frame.from_fixed_frame_point(coords, relative)

    def event_position(self, event: dict) -> np.ndarray:
        if sys.platform == "emscripten":
            from js import document

            canvas = document.getElementById(self.canvas_id)
            if canvas is not None:
                pointer = getattr(canvas, "__manimPointer", None)
                if pointer is not None:
                    try:
                        client_y = float(pointer.clientY)
                    except (AttributeError, TypeError, ValueError):
                        client_y = None

                    if client_y is not None:
                        rect = canvas.getBoundingClientRect()
                        width, height = float(rect.width), float(rect.height)
                        render_width, render_height = self.render_size

                        if width > 0 and height > 0:
                            y = (
                                float(event["y"])
                            ) * render_height / height

                            x = (
                                float(event["x"])
                                * render_width
                                / width
                            )
                            return np.array([x, render_height - y])
        _, height = self.canvas.get_logical_size()
        return np.array([event["x"], height - event["y"]])

    def event_point(self, event: dict) -> np.ndarray:
        return self.pixel_coords_to_space_coords(*self.event_position(event))

    def on_pointer_move(self, event: dict) -> None:
        self.note_event()
        if self.scene is None:
            return
        position = self.event_position(event)
        movement = position - self.pointer_position
        self.pointer_position = position
        point = self.pixel_coords_to_space_coords(*position)
        d_point = self.pixel_coords_to_space_coords(*movement, relative=True)
        if event["buttons"]:
            self.scene.on_mouse_drag(
                point, d_point, event["buttons"], to_mods(event["modifiers"]),
            )
        else:
            self.scene.on_mouse_motion(point, d_point)

    def on_pointer_down(self, event: dict) -> None:
        self.note_event()
        if self.scene is None:
            return
        self.pointer_position = self.event_position(event)
        self.scene.on_mouse_press(
            self.event_point(event), event["button"], to_mods(event["modifiers"]),
        )

    def on_pointer_up(self, event: dict) -> None:
        self.note_event()
        if self.scene is None:
            return
        self.scene.on_mouse_release(
            self.event_point(event), event["button"], to_mods(event["modifiers"]),
        )

    def on_wheel(self, event: dict) -> None:
        self.note_event()
        if self.scene is None:
            return
        notches = np.array([event["dx"], event["dy"]]) / WHEEL_NOTCH
        self.scene.on_mouse_scroll(
            self.event_point(event),
            self.pixel_coords_to_space_coords(*notches, relative=True),
            *notches,
        )

    def on_key_down(self, event: dict) -> None:
        self.note_event()
        key = to_key(event["key"])
        if key is None:
            return
        self.pressed_keys.add(key)
        if self.scene:
            self.scene.on_key_press(key, to_mods(event["modifiers"]))

    def on_key_up(self, event: dict) -> None:
        self.note_event()
        key = to_key(event["key"])
        if key is None:
            return
        self.pressed_keys.discard(key)
        if self.scene:
            self.scene.on_key_release(key, to_mods(event["modifiers"]))

    def on_resize(self, event: dict) -> None:
        self.note_event()
        if self.scene:
            self.scene.on_resize(event["width"], event["height"])

    def on_close(self, event: dict) -> None:
        self.note_event()
        if self.scene:
            self.scene.on_close()


if sys.platform == "emscripten":
    async def run_scene_from_class(scene_class: type[Scene], canvas_id: str, size: tuple[int, int] = (1920, 1080)) -> Scene:
        import asyncio

        window = await Window.create_for_pyodide(canvas_id, size=size)
        scene = scene_class(window=window)

        await scene.build_async()
        await scene.update_frame_async(force_draw=True)

        scene._browser_interaction_task = asyncio.create_task(
            scene.browser_interaction_loop()
        )
        return scene
