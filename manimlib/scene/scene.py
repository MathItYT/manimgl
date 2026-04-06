from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import platform
import queue
import random
import threading
import time
from functools import wraps
from contextlib import contextmanager
from contextlib import ExitStack

import numpy as np
from tqdm.auto import tqdm as ProgressDisplay
from pyglet.window import key as PygletWindowKeys

from manimlib.animation.animation import prepare_animation
from manimlib.camera.camera import Camera
from manimlib.camera.camera_frame import CameraFrame
from manimlib.config import manim_config
from manimlib.event_handler import EVENT_DISPATCHER
from manimlib.event_handler.event_type import EventType
from manimlib.logger import log
from manimlib.mobject.mobject import _AnimationBuilder
from manimlib.mobject.mobject import Group
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.mobject import Point
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.scene.scene_embed import InteractiveSceneEmbed
from manimlib.scene.scene_embed import CheckpointManager
from manimlib.scene.scene_file_writer import SceneFileWriter
from manimlib.utils.dict_ops import merge_dicts_recursively
from manimlib.utils.family_ops import extract_mobject_family_members
from manimlib.utils.family_ops import recursive_mobject_remove
from manimlib.utils.iterables import batch_by_property
from manimlib.utils.sounds import play_sound
from manimlib.utils.color import color_to_rgba
from manimlib.window import Window

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Iterable, TypeVar, Optional
    from manimlib.typing import Vect3

    T = TypeVar('T')

    from PIL.Image import Image

    from manimlib.animation.animation import Animation


class _MainThreadCaller:
    """Run callables on the thread that created this instance.

    This is used to ensure OpenGL / pyglet calls stay on the main thread,
    while scene logic can run on a worker thread.
    """

    def __init__(self):
        self._main_thread_id = threading.get_ident()
        self._queue: queue.Queue[tuple[callable, tuple, dict, threading.Event, dict]] = queue.Queue()

    def is_main_thread(self) -> bool:
        return threading.get_ident() == self._main_thread_id

    def call(self, func, /, *args, **kwargs):
        if self.is_main_thread():
            return func(*args, **kwargs)

        done = threading.Event()
        holder: dict[str, object] = {"result": None, "exc": None}
        self._queue.put((func, args, kwargs, done, holder))
        done.wait()
        exc = holder.get("exc")
        if exc is not None:
            raise exc  # type: ignore[misc]
        return holder.get("result")

    def run_one(self, *, block: bool = False, timeout: float | None = None) -> bool:
        try:
            func, args, kwargs, done, holder = self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return False

        try:
            holder["result"] = func(*args, **kwargs)
        except BaseException as exc:
            holder["exc"] = exc
        finally:
            done.set()
            self._queue.task_done()
        return True

    def run_all(self, *, max_tasks: int | None = None) -> int:
        n = 0
        while max_tasks is None or n < max_tasks:
            if not self.run_one(block=False):
                break
            n += 1
        return n


_GLOBAL_MAIN_THREAD_CALLER: _MainThreadCaller | None = None


def get_main_thread_caller() -> _MainThreadCaller | None:
    """Return the active main-thread caller when threaded mode is running."""
    return _GLOBAL_MAIN_THREAD_CALLER


def _set_global_main_thread_caller(caller: _MainThreadCaller | None) -> None:
    global _GLOBAL_MAIN_THREAD_CALLER
    _GLOBAL_MAIN_THREAD_CALLER = caller


class Scene(object):
    random_seed: int = 0
    pan_sensitivity: float = 0.5
    scroll_sensitivity: float = 20
    drag_to_pan: bool = True
    max_num_saved_states: int = 50
    default_camera_config: dict = dict()
    default_file_writer_config: dict = dict()
    samples = 0
    # Euler angles, in degrees
    default_frame_orientation = (0, 0)

    def __init__(
        self,
        window: Optional[Window] = None,
        camera_config: dict = dict(),
        file_writer_config: dict = dict(),
        skip_animations: bool = False,
        always_update_mobjects: bool = False,
        start_at_animation_number: int | None = None,
        end_at_animation_number: int | None = None,
        show_animation_progress: bool = False,
        leave_progress_bars: bool = False,
        preview_while_skipping: bool = True,
        presenter_mode: bool = False,
        default_wait_time: float = 1.0,
        threaded: bool = False,
        parallel_animations: bool = False,
    ):
        self.updaters: list[Callable[[float], None]] = []
        self.skip_animations = skip_animations
        self.always_update_mobjects = always_update_mobjects
        self.start_at_animation_number = start_at_animation_number
        self.end_at_animation_number = end_at_animation_number
        self.show_animation_progress = show_animation_progress
        self.leave_progress_bars = leave_progress_bars
        self.preview_while_skipping = preview_while_skipping
        self.presenter_mode = presenter_mode
        self.default_wait_time = default_wait_time
        self.frame_sinks: list[object] = []

        # Threading
        self.threaded = threaded
        self.parallel_animations = parallel_animations
        self._threaded_mode_active: bool = False
        self._main_thread_caller: _MainThreadCaller | None = None
        self._threaded_window_event_queue: queue.Queue[tuple[str, tuple, dict]] | None = None
        self._threaded_wake_event: threading.Event | None = None
        self._threaded_stop_event: threading.Event | None = None
        self._threaded_worker_thread: threading.Thread | None = None

        # Threaded window event coalescing (high-frequency events)
        self._threaded_event_lock = threading.Lock()
        self._threaded_latest_mouse_motion: tuple[int, int, int, int] | None = None
        self._threaded_latest_mouse_drag: tuple[int, int, int, int, int, int] | None = None
        self._threaded_latest_mouse_scroll: tuple[int, int, float, float] | None = None

        # Threaded key state shadow (read from worker thread)
        self._threaded_pressed_keys: set[int] = set()

        self.camera_config = merge_dicts_recursively(
            manim_config.camera,         # Global default
            self.default_camera_config,  # Updated configuration that subclasses may specify
            camera_config,               # Updated configuration from instantiation
        )
        self.file_writer_config = merge_dicts_recursively(
            manim_config.file_writer,
            self.default_file_writer_config,
            file_writer_config,
        )

        self.window = window
        if self.window:
            self.window.init_for_scene(self)

        # Core state of the scene
        self.camera: Camera = Camera(
            window=self.window,
            samples=self.samples,
            **self.camera_config
        )
        self.frame: CameraFrame = self.camera.frame
        self.frame.reorient(*self.default_frame_orientation)
        self.frame.make_orientation_default()

        self.file_writer = SceneFileWriter(self, **self.file_writer_config)
        self.mobjects: list[Mobject] = [self.camera.frame]
        self.render_groups: list[Mobject] = []
        self.id_to_mobject_map: dict[int, Mobject] = dict()
        self.num_plays: int = 0
        self.time: float = 0
        self.skip_time: float = 0
        self.original_skipping_status: bool = self.skip_animations
        self.undo_stack = []
        self.redo_stack = []

        if self.start_at_animation_number is not None:
            self.skip_animations = True
        if self.file_writer.has_progress_display():
            self.show_animation_progress = False

        # Items associated with interaction
        self.mouse_point = Point()
        self.mouse_drag_point = Point()
        self.hold_on_wait = self.presenter_mode
        self.quit_interaction = False

        # Much nicer to work with deterministic scenes
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

    def __str__(self) -> str:
        return self.__class__.__name__

    def get_window(self) -> Window | None:
        return self.window

    def add_frame_sink(self, frame_sink: object) -> object:
        self.frame_sinks.append(frame_sink)
        return frame_sink

    def remove_frame_sink(self, frame_sink: object) -> None:
        if frame_sink in self.frame_sinks:
            self.frame_sinks.remove(frame_sink)

    def clear_frame_sinks(self) -> None:
        while self.frame_sinks:
            frame_sink = self.frame_sinks.pop()
            close = getattr(frame_sink, "close", None)
            if callable(close):
                close()

    def _emit_frame_sinks(self) -> None:
        for frame_sink in list(self.frame_sinks):
            try:
                push_frame = getattr(frame_sink, "push_frame", None)
                if callable(push_frame):
                    push_frame(self.camera)
            except Exception:
                log.exception("Frame sink failed while consuming a frame")

    def run(self) -> None:
        self.virtual_animation_start_time: float = 0
        self.real_animation_start_time: float = time.time()
        self.file_writer.begin()

        if self.window is not None and self.threaded:
            try:
                self._run_threaded()
            except EndScene:
                pass
            except KeyboardInterrupt:
                # Get rid keyboard interupt symbols
                print("", end="\r")
                self.file_writer.ended_with_interrupt = True
            self.tear_down()
            return

        self.setup()
        try:
            self.construct()
            self.interact()
        except EndScene:
            pass
        except KeyboardInterrupt:
            # Get rid keyboard interupt symbols
            print("", end="\r")
            self.file_writer.ended_with_interrupt = True
        self.tear_down()

    def _run_threaded(self) -> None:
        """Run scene logic on a worker thread; keep window + OpenGL on main.

        This is primarily for preview-mode responsiveness: the main thread keeps
        pumping the pyglet event loop, while the worker thread drives the scene.
        """
        assert self.window is not None

        self._threaded_mode_active = True
        self._main_thread_caller = _MainThreadCaller()
        _set_global_main_thread_caller(self._main_thread_caller)
        self._threaded_window_event_queue = queue.Queue()
        self._threaded_wake_event = threading.Event()
        self._threaded_stop_event = threading.Event()

        worker_exc: dict[str, BaseException | None] = {"exc": None}

        def worker() -> None:
            try:
                self.setup()
                self.construct()
                self.interact()
            except BaseException as exc:
                worker_exc["exc"] = exc
            finally:
                # Ensure the main loop wakes up and can exit
                if self._threaded_stop_event is not None:
                    self._threaded_stop_event.set()
                if self._threaded_wake_event is not None:
                    self._threaded_wake_event.set()

        self._threaded_worker_thread = threading.Thread(
            target=worker,
            name=f"ManimSceneWorker:{self}",
            daemon=True,
        )
        self._threaded_worker_thread.start()

        try:
            # Main loop: keep handling window events and servicing GL tasks.
            while self._threaded_worker_thread.is_alive():
                # Run one queued main-thread task (or wait briefly for one)
                # to keep latency low without spinning.
                if self._main_thread_caller is not None:
                    self._main_thread_caller.run_one(block=True, timeout=0.001)

                # Pump pyglet events (must be on main thread)
                if self.window is not None and getattr(self.window, "_window", None) is not None:
                    self.window._window.dispatch_events()

                if self.is_window_closing():
                    # Encourage the worker thread to unwind.
                    if self._threaded_stop_event is not None:
                        self._threaded_stop_event.set()
                    if self._threaded_wake_event is not None:
                        self._threaded_wake_event.set()

            # Drain any leftover main-thread calls to prevent deadlocks on shutdown
            if self._main_thread_caller is not None:
                self._main_thread_caller.run_all(max_tasks=1000)

        except KeyboardInterrupt:
            # Propagate to Scene.run() which handles file_writer flags.
            if self._threaded_stop_event is not None:
                self._threaded_stop_event.set()
            if self._threaded_wake_event is not None:
                self._threaded_wake_event.set()
            if self.window is not None:
                self.quit_interaction = True
                self.hold_on_wait = False
            # Best-effort join
            if self._threaded_worker_thread is not None:
                self._threaded_worker_thread.join(timeout=0.5)
            raise

        finally:
            if self._threaded_worker_thread is not None and self._threaded_worker_thread.is_alive():
                self._threaded_worker_thread.join(timeout=1.0)
            # Keep the wiring around until tear_down() completes (it may still
            # need main-thread execution), but mark inactive.
            self._threaded_mode_active = False
            _set_global_main_thread_caller(None)

        if worker_exc["exc"] is not None:
            raise worker_exc["exc"]

    def setup(self) -> None:
        """
        This is meant to be implement by any scenes which
        are comonly subclassed, and have some common setup
        involved before the construct method is called.
        """
        pass

    def construct(self) -> None:
        # Where all the animation happens
        # To be implemented in subclasses
        pass

    def tear_down(self) -> None:
        self.stop_skipping()
        self.file_writer.finish()
        self.clear_frame_sinks()
        if self.window:
            self.window.destroy()
            self.window = None

    def interact(self) -> None:
        """
        If there is a window, enter a loop
        which updates the frame while under
        the hood calling the pyglet event loop
        """
        if self.window is None:
            return
        log.info(
            "\nTips: Using the keys `d`, `f`, or `z` " +
            "you can interact with the scene. " +
            "Press `command + q` or `esc` to quit"
        )
        self.stop_skipping()
        while not self.is_window_closing():
            self.update_frame(1 / self.camera.fps)
            self.emit_frame()

    def embed(
        self,
        close_scene_on_exit: bool = True,
        show_animation_progress: bool = False,
    ) -> None:
        if not self.window:
            # Embed is only relevant for interactive development with a Window
            return
        self.show_animation_progress = show_animation_progress
        self.stop_skipping()
        self.update_frame(force_draw=True)

        InteractiveSceneEmbed(self).launch()

        # End scene when exiting an embed
        if close_scene_on_exit:
            raise EndScene()

    # Only these methods should touch the camera

    def get_image(self) -> Image:
        if (
            self._threaded_mode_active
            and self._main_thread_caller is not None
            and not self._main_thread_caller.is_main_thread()
        ):
            return self._main_thread_caller.call(self.get_image)
        if self.window is not None:
            self.camera.use_window_fbo(False)
            self.camera.capture(*self.render_groups, swap=False)
        image = self.camera.get_image()
        if self.window is not None:
            self.camera.use_window_fbo(True)
        return image

    def show(self) -> None:
        self.update_frame(force_draw=True)
        self.get_image().show()
    
    def update_self(self, dt: float) -> None:
        for updater in self.updaters:
            updater(dt)
    
    def add_updater(self, updater: Callable[[float], None]) -> None:
        self.updaters.append(updater)

    def update_frame(self, dt: float = 0, force_draw: bool = False) -> None:
        if (
            self._threaded_mode_active
            and self._main_thread_caller is not None
            and not self._main_thread_caller.is_main_thread()
        ):
            # Worker thread: update scene state, then ask main thread to render.
            if self._threaded_stop_event is not None and self._threaded_stop_event.is_set():
                raise EndScene()

            self.increment_time(dt)
            self.update_self(dt)
            self.update_mobjects(dt)

            # Apply any queued window events (enqueued from the main thread)
            self._process_queued_window_events()

            if self.skip_animations and not force_draw:
                return

            if self.is_window_closing():
                raise EndScene()

            if (
                self.window
                and dt == 0
                and not force_draw
                and not self.window.has_undrawn_event()
            ):
                # No need to redraw; the main thread already pumps events.
                return

            self._main_thread_caller.call(self._render_frame_main_thread)

            # Time-sync happens on the worker thread to keep main responsive.
            if self.window and not self.skip_animations:
                vt = self.time - self.virtual_animation_start_time
                rt = time.time() - self.real_animation_start_time
                delay = max(vt - rt, 0)
                if delay > 0 and self._threaded_wake_event is not None:
                    # Wake early on user input; we'll compensate in later frames.
                    self._threaded_wake_event.wait(timeout=delay)
                    self._threaded_wake_event.clear()
            return

        # Main thread / non-threaded mode: original behavior
        self.increment_time(dt)
        self.update_self(dt)
        self.update_mobjects(dt)
        if self.skip_animations and not force_draw:
            return

        if self.is_window_closing():
            raise EndScene()

        if self.window and dt == 0 and not self.window.has_undrawn_event() and not force_draw:
            # In this case, there's no need for new rendering, but we
            # shoudl still listen for new events
            self.window._window.dispatch_events()
            return

        self.camera.capture(*self.render_groups, swap=not self.file_writer.write_to_movie and len(self.frame_sinks) == 0)
        self._emit_frame_sinks()

        if self.window and not self.skip_animations:
            vt = self.time - self.virtual_animation_start_time
            rt = time.time() - self.real_animation_start_time
            time.sleep(max(vt - rt, 0))

    def _render_frame_main_thread(self) -> None:
        """Render a frame. Must be called on the main thread."""
        if self.is_window_closing():
            raise EndScene()
        self.camera.capture(
            *self.render_groups,
            swap=(not self.file_writer.write_to_movie and len(self.frame_sinks) == 0),
        )
        self._emit_frame_sinks()

    def emit_frame(self) -> None:
        if (
            self._threaded_mode_active
            and self._main_thread_caller is not None
            and not self._main_thread_caller.is_main_thread()
        ):
            return self._main_thread_caller.call(self.emit_frame)
        if not self.skip_animations:
            self.file_writer.write_frame(self.camera)

    def _dispatch_window_event(self, method_name: str, /, *args, **kwargs) -> None:
        """Entry point used by Window to deliver events.

        In threaded mode, events are queued to be executed by the worker thread.
        """
        # If no worker wiring exists yet, translate and call directly.
        if (
            not self._threaded_mode_active
            or self._threaded_window_event_queue is None
            or self._main_thread_caller is None
        ):
            self._handle_window_event(method_name, args, kwargs)
            return

        # If called from main thread (i.e. pyglet callbacks), enqueue.
        if self._main_thread_caller.is_main_thread():
            # Coalesce high-frequency events to avoid worker backlog when
            # the mouse moves quickly or a key is held down.
            if method_name == "on_mouse_motion" and len(args) >= 4:
                with self._threaded_event_lock:
                    x, y, dx, dy = args[0], args[1], args[2], args[3]
                    if self._threaded_latest_mouse_motion is None:
                        self._threaded_latest_mouse_motion = (x, y, dx, dy)
                    else:
                        _, _, acc_dx, acc_dy = self._threaded_latest_mouse_motion
                        self._threaded_latest_mouse_motion = (x, y, acc_dx + dx, acc_dy + dy)
            elif method_name == "on_mouse_drag" and len(args) >= 6:
                with self._threaded_event_lock:
                    x, y, dx, dy, buttons, modifiers = args[0], args[1], args[2], args[3], args[4], args[5]
                    if self._threaded_latest_mouse_drag is None:
                        self._threaded_latest_mouse_drag = (x, y, dx, dy, buttons, modifiers)
                    else:
                        _, _, acc_dx, acc_dy, _, _ = self._threaded_latest_mouse_drag
                        self._threaded_latest_mouse_drag = (x, y, acc_dx + dx, acc_dy + dy, buttons, modifiers)
            elif method_name == "on_mouse_scroll" and len(args) >= 4:
                with self._threaded_event_lock:
                    x, y, x_offset, y_offset = args[0], args[1], args[2], args[3]
                    if self._threaded_latest_mouse_scroll is None:
                        self._threaded_latest_mouse_scroll = (x, y, x_offset, y_offset)
                    else:
                        _, _, acc_x, acc_y = self._threaded_latest_mouse_scroll
                        self._threaded_latest_mouse_scroll = (x, y, acc_x + x_offset, acc_y + y_offset)
            else:
                self._threaded_window_event_queue.put((method_name, args, kwargs))
            if self._threaded_wake_event is not None:
                self._threaded_wake_event.set()
            return

        # Otherwise (e.g. worker thread), call directly.
        self._handle_window_event(method_name, args, kwargs)

    def _process_queued_window_events(self) -> None:
        if self._threaded_window_event_queue is None:
            return
        while True:
            try:
                method_name, args, kwargs = self._threaded_window_event_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._handle_window_event(method_name, args, kwargs)
            except EndScene:
                raise
            except Exception:
                log.exception(f"Exception while handling window event {method_name}")
            finally:
                self._threaded_window_event_queue.task_done()

        # Apply latest coalesced high-frequency events (at most one of each)
        with self._threaded_event_lock:
            motion = self._threaded_latest_mouse_motion
            drag = self._threaded_latest_mouse_drag
            scroll = self._threaded_latest_mouse_scroll
            self._threaded_latest_mouse_motion = None
            self._threaded_latest_mouse_drag = None
            self._threaded_latest_mouse_scroll = None

        if motion is not None:
            self._handle_window_event("on_mouse_motion", motion, {})
        if drag is not None:
            self._handle_window_event("on_mouse_drag", drag, {})
        if scroll is not None:
            self._handle_window_event("on_mouse_scroll", scroll, {})

    def _handle_window_event(self, method_name: str, args: tuple, kwargs: dict) -> None:
        """Translate raw window event arguments and call the appropriate handler."""
        # Maintain a worker-thread shadow of pressed keys.
        if method_name == "on_key_press" and len(args) >= 1:
            symbol = args[0]
            if isinstance(symbol, int):
                self._threaded_pressed_keys.add(symbol)
        elif method_name == "on_key_release" and len(args) >= 1:
            symbol = args[0]
            if isinstance(symbol, int):
                self._threaded_pressed_keys.discard(symbol)

        # Mouse events may be delivered as raw pixel coords (x, y, ...)
        if self.window is not None:
            if method_name == "on_mouse_motion" and len(args) >= 4:
                try:
                    x, y, dx, dy = args[:4]
                    point = self.window.pixel_coords_to_space_coords(int(x), int(y))
                    d_point = self.window.pixel_coords_to_space_coords(int(dx), int(dy), relative=True)
                except Exception:
                    log.exception("Failed to translate on_mouse_motion event")
                    return
                return getattr(self, method_name)(point, d_point)

            if method_name == "on_mouse_drag" and len(args) >= 6:
                try:
                    x, y, dx, dy, buttons, modifiers = args[:6]
                    point = self.window.pixel_coords_to_space_coords(int(x), int(y))
                    d_point = self.window.pixel_coords_to_space_coords(int(dx), int(dy), relative=True)
                except Exception:
                    log.exception("Failed to translate on_mouse_drag event")
                    return
                return getattr(self, method_name)(point, d_point, buttons, modifiers)

            if method_name in ("on_mouse_press", "on_mouse_release") and len(args) >= 4:
                try:
                    x, y, button, mods = args[:4]
                    point = self.window.pixel_coords_to_space_coords(int(x), int(y))
                except Exception:
                    log.exception(f"Failed to translate {method_name} event")
                    return
                return getattr(self, method_name)(point, button, mods)

            if method_name == "on_mouse_scroll" and len(args) >= 4 and not isinstance(args[0], np.ndarray):
                try:
                    x, y, x_offset, y_offset = args[:4]
                    point = self.window.pixel_coords_to_space_coords(int(x), int(y))
                    offset = self.window.pixel_coords_to_space_coords(x_offset, y_offset, relative=True)
                except Exception:
                    log.exception("Failed to translate on_mouse_scroll event")
                    return
                return getattr(self, method_name)(point, offset, x_offset, y_offset)

        return getattr(self, method_name)(*args, **kwargs)

    def is_key_pressed(self, symbol: int) -> bool:
        """Thread-safe key query.

        In threaded mode, window callbacks run on the main thread while scene
        logic runs on a worker thread; rely on a worker-side shadow set.
        """
        if self._threaded_mode_active and (self._main_thread_caller is not None) and (not self._main_thread_caller.is_main_thread()):
            return symbol in self._threaded_pressed_keys
        if self.window is None:
            return False
        return self.window.is_key_pressed(symbol)

    # Related to updating

    def update_mobjects(self, dt: float) -> None:
        for mobject in self.mobjects:
            mobject.update(dt)

    def should_update_mobjects(self) -> bool:
        return self.always_update_mobjects or any(
            mob.has_updaters() for mob in self.mobjects
        )

    # Related to time

    def get_time(self) -> float:
        return self.time

    def increment_time(self, dt: float) -> None:
        self.time += dt

    # Related to internal mobject organization

    def get_top_level_mobjects(self) -> list[Mobject]:
        # Return only those which are not in the family
        # of another mobject from the scene
        mobjects = self.get_mobjects()
        families = [m.get_family() for m in mobjects]

        def is_top_level(mobject):
            num_families = sum([
                (mobject in family)
                for family in families
            ])
            return num_families == 1
        return list(filter(is_top_level, mobjects))

    def get_mobject_family_members(self) -> list[Mobject]:
        return extract_mobject_family_members(self.mobjects)

    def assemble_render_groups(self):
        """
        Rendering can be more efficient when mobjects of the
        same type are grouped together, so this function creates
        Groups of all clusters of adjacent Mobjects in the scene
        """
        if (
            self._threaded_mode_active
            and self._main_thread_caller is not None
            and not self._main_thread_caller.is_main_thread()
        ):
            # This touches OpenGL via ShaderWrapper initialization, so it must
            # happen on the main thread.
            return self._main_thread_caller.call(self.assemble_render_groups)

        batches = batch_by_property(
            self.mobjects,
            lambda m: str(type(m)) + str(m.get_shader_wrapper(self.camera.ctx).get_id()) + str(m.z_index)
        )

        for group in self.render_groups:
            group.clear()
        self.render_groups = [
            batch[0].get_group_class()(*batch)
            for batch, key in batches
        ]

    @staticmethod
    def affects_mobject_list(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.assemble_render_groups()
            return self
        return wrapper

    @affects_mobject_list
    def add(self, *new_mobjects: Mobject):
        """
        Mobjects will be displayed, from background to
        foreground in the order with which they are added.
        """
        self.remove(*new_mobjects)
        self.mobjects += new_mobjects

        # Reorder based on z_index
        id_to_scene_order = {id(m): idx for idx, m in enumerate(self.mobjects)}
        self.mobjects.sort(key=lambda m: (m.z_index, id_to_scene_order[id(m)]))

        self.id_to_mobject_map.update({
            id(sm): sm
            for m in new_mobjects
            for sm in m.get_family()
        })
        return self

    def add_mobjects_among(self, values: Iterable):
        """
        This is meant mostly for quick prototyping,
        e.g. to add all mobjects defined up to a point,
        call self.add_mobjects_among(locals().values())
        """
        self.add(*filter(
            lambda m: isinstance(m, Mobject),
            values
        ))
        return self

    @affects_mobject_list
    def replace(self, mobject: Mobject, *replacements: Mobject):
        if mobject in self.mobjects:
            index = self.mobjects.index(mobject)
            self.mobjects = [
                *self.mobjects[:index],
                *replacements,
                *self.mobjects[index + 1:]
            ]
        return self

    @affects_mobject_list
    def remove(self, *mobjects_to_remove: Mobject):
        """
        Removes anything in mobjects from scenes mobject list, but in the event that one
        of the items to be removed is a member of the family of an item in mobject_list,
        the other family members are added back into the list.

        For example, if the scene includes Group(m1, m2, m3), and we call scene.remove(m1),
        the desired behavior is for the scene to then include m2 and m3 (ungrouped).
        """
        to_remove = set(extract_mobject_family_members(mobjects_to_remove))
        new_mobjects, _ = recursive_mobject_remove(self.mobjects, to_remove)
        self.mobjects = new_mobjects

    @affects_mobject_list
    def remove_all_except(self, *mobjects_to_keep : Mobject):
        self.clear()
        self.add(*mobjects_to_keep)

    def bring_to_front(self, *mobjects: Mobject):
        self.add(*mobjects)
        return self

    @affects_mobject_list
    def bring_to_back(self, *mobjects: Mobject):
        self.remove(*mobjects)
        self.mobjects = list(mobjects) + self.mobjects
        return self

    @affects_mobject_list
    def clear(self):
        self.mobjects = []
        return self

    def get_mobjects(self) -> list[Mobject]:
        return list(self.mobjects)

    def get_mobject_copies(self) -> list[Mobject]:
        return [m.copy() for m in self.mobjects]

    def point_to_mobject(
        self,
        point: np.ndarray,
        search_set: Iterable[Mobject] | None = None,
        buff: float = 0
    ) -> Mobject | None:
        """
        E.g. if clicking on the scene, this returns the top layer mobject
        under a given point
        """
        if search_set is None:
            search_set = self.mobjects
        for mobject in reversed(search_set):
            if mobject.is_point_touching(point, buff=buff):
                return mobject
        return None

    def get_group(self, *mobjects):
        if all(isinstance(m, VMobject) for m in mobjects):
            return VGroup(*mobjects)
        else:
            return Group(*mobjects)

    def id_to_mobject(self, id_value):
        return self.id_to_mobject_map[id_value]

    def ids_to_group(self, *id_values):
        return self.get_group(*filter(
            lambda x: x is not None,
            map(self.id_to_mobject, id_values)
        ))

    def i2g(self, *id_values):
        return self.ids_to_group(*id_values)

    def i2m(self, id_value):
        return self.id_to_mobject(id_value)

    # Related to skipping

    def update_skipping_status(self) -> None:
        if self.start_at_animation_number is not None:
            if self.num_plays == self.start_at_animation_number:
                self.skip_time = self.time
                if not self.original_skipping_status:
                    self.stop_skipping()
        if self.end_at_animation_number is not None:
            if self.num_plays >= self.end_at_animation_number:
                raise EndScene()

    def stop_skipping(self) -> None:
        self.virtual_animation_start_time = self.time
        self.real_animation_start_time = time.time()
        self.skip_animations = False

    # Methods associated with running animations

    def get_time_progression(
        self,
        run_time: float,
        n_iterations: int | None = None,
        desc: str = "",
        override_skip_animations: bool = False
    ) -> list[float] | np.ndarray | ProgressDisplay:
        if self.skip_animations and not override_skip_animations:
            return [run_time]

        times = np.arange(0, run_time, 1 / self.camera.fps) + 1 / self.camera.fps

        self.file_writer.set_progress_display_description(sub_desc=desc)

        if self.show_animation_progress:
            return ProgressDisplay(
                times,
                total=n_iterations,
                leave=self.leave_progress_bars,
                ascii=True if platform.system() == 'Windows' else None,
                desc=desc,
                bar_format="{l_bar} {n_fmt:3}/{total_fmt:3} {rate_fmt}{postfix}",
            )
        else:
            return times

    def get_run_time(self, animations: Iterable[Animation]) -> float:
        return np.max([animation.get_run_time() for animation in animations])

    def get_animation_time_progression(
        self,
        animations: Iterable[Animation]
    ) -> list[float] | np.ndarray | ProgressDisplay:
        animations = list(animations)
        run_time = self.get_run_time(animations)
        description = f"{self.num_plays} {animations[0]}"
        if len(animations) > 1:
            description += ", etc."
        time_progression = self.get_time_progression(run_time, desc=description)
        return time_progression

    def get_wait_time_progression(
        self,
        duration: float,
        stop_condition: Callable[[], bool] | None = None
    ) -> list[float] | np.ndarray | ProgressDisplay:
        kw = {"desc": f"{self.num_plays} Waiting"}
        if stop_condition is not None:
            kw["n_iterations"] = -1  # So it doesn't show % progress
            kw["override_skip_animations"] = True
        return self.get_time_progression(duration, **kw)

    def pre_play(self):
        if self.presenter_mode and self.num_plays == 0:
            self.hold_loop()

        self.update_skipping_status()

        if not self.skip_animations:
            self.file_writer.begin_animation()

        if self.window:
            self.virtual_animation_start_time = self.time
            self.real_animation_start_time = time.time()

    def post_play(self):
        if not self.skip_animations:
            self.file_writer.end_animation()

        if self.preview_while_skipping and self.skip_animations and self.window is not None:
            # Show some quick frames along the way
            self.update_frame(dt=0, force_draw=True)

        self.num_plays += 1

    def begin_animations(self, animations: Iterable[Animation]) -> None:
        all_mobjects = set(self.get_mobject_family_members())
        for animation in animations:
            animation.begin()
            # Anything animated that's not already in the
            # scene gets added to the scene.  Note, for
            # animated mobjects that are in the family of
            # those on screen, this can result in a restructuring
            # of the scene.mobjects list, which is usually desired.
            if animation.mobject not in all_mobjects:
                self.add(animation.mobject)
                all_mobjects = all_mobjects.union(animation.mobject.get_family())

    def progress_through_animations(self, animations: Iterable[Animation]) -> None:
        animations = list(animations)

        # Experimental: run disjoint animations in parallel (thread-per-animation)
        if self.parallel_animations and self._threaded_mode_active and len(animations) > 1:
            # Greedily batch animations into conflict-free groups while preserving
            # deterministic order for any animations that touch the same mobjects.
            family_id_sets: list[set[int]] = []
            for anim in animations:
                try:
                    family_id_sets.append({id(m) for m in anim.mobject.get_family()})
                except Exception:
                    family_id_sets.append({id(anim.mobject)})

            groups: list[list[Animation]] = []
            group_used_ids: list[set[int]] = []
            for anim, fam_ids in zip(animations, family_id_sets):
                placed = False
                for used_ids, group in zip(group_used_ids, groups):
                    if used_ids.isdisjoint(fam_ids):
                        group.append(anim)
                        used_ids.update(fam_ids)
                        placed = True
                        break
                if not placed:
                    groups.append([anim])
                    group_used_ids.append(set(fam_ids))

            def step(anim: Animation, dt: float, t: float) -> None:
                anim.update_mobjects(dt)
                alpha = t / anim.run_time
                anim.interpolate(alpha)

            last_t = 0
            with ThreadPoolExecutor(max_workers=len(animations)) as executor:
                for t in self.get_animation_time_progression(animations):
                    dt = t - last_t
                    last_t = t
                    for group in groups:
                        futures = [executor.submit(step, anim, dt, t) for anim in group]
                        for fut in futures:
                            fut.result()
                    self.update_frame(dt)
                    self.emit_frame()
            return

        last_t = 0
        for t in self.get_animation_time_progression(animations):
            dt = t - last_t
            last_t = t
            for animation in animations:
                animation.update_mobjects(dt)
                alpha = t / animation.run_time
                animation.interpolate(alpha)
            self.update_frame(dt)
            self.emit_frame()

    def finish_animations(self, animations: Iterable[Animation]) -> None:
        for animation in animations:
            animation.finish()
            animation.clean_up_from_scene(self)
        if self.skip_animations:
            self.update_mobjects(self.get_run_time(animations))
        else:
            self.update_mobjects(0)
    
    def start_mic_recording(
        self,
        rate: int = 44100,
        channels: int = 1,
        chunk: int = 1024,
        callback: Callable[[bytes, int, dict, int], None] | None = None,
    ) -> None:
        self.file_writer.start_mic_recording(rate, channels, chunk, callback)

    @affects_mobject_list
    def play(
        self,
        *proto_animations: Animation | _AnimationBuilder,
        run_time: float | None = None,
        rate_func: Callable[[float], float] | None = None,
        lag_ratio: float | None = None,
    ) -> None:
        if len(proto_animations) == 0:
            log.warning("Called Scene.play with no animations")
            return
        animations = list(map(prepare_animation, proto_animations))
        for anim in animations:
            anim.update_rate_info(run_time, rate_func, lag_ratio)
        self.pre_play()
        self.begin_animations(animations)
        self.progress_through_animations(animations)
        self.finish_animations(animations)
        self.post_play()

    def wait(
        self,
        duration: Optional[float] = None,
        stop_condition: Callable[[], bool] = None,
        note: str = None,
        ignore_presenter_mode: bool = False
    ):
        if duration is None:
            duration = self.default_wait_time
        self.pre_play()
        self.update_mobjects(dt=0)  # Any problems with this?
        if self.presenter_mode and not self.skip_animations and not ignore_presenter_mode:
            if note:
                log.info(note)
            self.hold_loop()
        else:
            time_progression = self.get_wait_time_progression(duration, stop_condition)
            last_t = 0
            for t in time_progression:
                dt = t - last_t
                last_t = t
                self.update_frame(dt)
                self.emit_frame()
                if stop_condition is not None and stop_condition():
                    break
        self.post_play()

    def hold_loop(self):
        while self.hold_on_wait:
            self.update_frame(dt=1 / self.camera.fps)
            self.emit_frame()
        self.hold_on_wait = True

    def wait_until(
        self,
        stop_condition: Callable[[], bool],
        max_time: float = 60
    ):
        self.wait(max_time, stop_condition=stop_condition)

    def force_skipping(self):
        self.original_skipping_status = self.skip_animations
        self.skip_animations = True
        return self

    def revert_to_original_skipping_status(self):
        if hasattr(self, "original_skipping_status"):
            self.skip_animations = self.original_skipping_status
        return self

    def add_sound(
        self,
        sound_file: str,
        time_offset: float = 0,
        gain: float | None = None,
        gain_to_background: float | None = None
    ):
        if self.skip_animations:
            return
        time = self.get_time() + time_offset
        self.file_writer.add_sound(sound_file, time, gain, gain_to_background)

    # Helpers for interactive development

    def get_state(self) -> SceneState:
        return SceneState(self)

    @affects_mobject_list
    def restore_state(self, scene_state: SceneState):
        scene_state.restore_scene(self)

    def save_state(self) -> None:
        state = self.get_state()
        if self.undo_stack and state.mobjects_match(self.undo_stack[-1]):
            return
        self.redo_stack = []
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_num_saved_states:
            self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.get_state())
            self.restore_state(self.undo_stack.pop())

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.get_state())
            self.restore_state(self.redo_stack.pop())

    @contextmanager
    def temp_skip(self):
        prev_status = self.skip_animations
        self.skip_animations = True
        try:
            yield
        finally:
            if not prev_status:
                self.stop_skipping()

    @contextmanager
    def temp_progress_bar(self):
        prev_progress = self.show_animation_progress
        self.show_animation_progress = True
        try:
            yield
        finally:
            self.show_animation_progress = prev_progress

    @contextmanager
    def temp_record(self):
        self.camera.use_window_fbo(False)
        self.file_writer.begin_insert()
        try:
            yield
        finally:
            self.file_writer.end_insert()
            self.camera.use_window_fbo(True)

    def temp_config_change(self, skip=False, record=False, progress_bar=False):
        stack = ExitStack()
        if skip:
            stack.enter_context(self.temp_skip())
        if record:
            stack.enter_context(self.temp_record())
        if progress_bar:
            stack.enter_context(self.temp_progress_bar())
        return stack

    def is_window_closing(self):
        return self.window and (self.window.is_closing or self.quit_interaction)

    # Event handling
    def set_floor_plane(self, plane: str = "xy"):
        if plane == "xy":
            self.frame.set_euler_axes("zxz")
        elif plane == "xz":
            self.frame.set_euler_axes("zxy")
        else:
            raise Exception("Only `xz` and `xy` are valid floor planes")

    def on_mouse_motion(
        self,
        point: Vect3,
        d_point: Vect3
    ) -> None:
        assert self.window is not None
        self.mouse_point.move_to(point)

        event_data = {"point": point, "d_point": d_point}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MouseMotionEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

        frame = self.camera.frame
        # Handle perspective changes
        if self.is_key_pressed(ord(manim_config.key_bindings.pan_3d)):
            ff_d_point = frame.to_fixed_frame_point(d_point, relative=True)
            ff_d_point *= self.pan_sensitivity
            frame.increment_theta(-ff_d_point[0])
            frame.increment_phi(ff_d_point[1])
        # Handle frame movements
        elif self.is_key_pressed(ord(manim_config.key_bindings.pan)):
            frame.shift(-d_point)

    def on_mouse_drag(
        self,
        point: Vect3,
        d_point: Vect3,
        buttons: int,
        modifiers: int
    ) -> None:
        self.mouse_drag_point.move_to(point)
        if self.drag_to_pan:
            self.frame.shift(-d_point)

        event_data = {"point": point, "d_point": d_point, "buttons": buttons, "modifiers": modifiers}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MouseDragEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_press(
        self,
        point: Vect3,
        button: int,
        mods: int
    ) -> None:
        self.mouse_drag_point.move_to(point)
        event_data = {"point": point, "button": button, "mods": mods}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MousePressEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_release(
        self,
        point: Vect3,
        button: int,
        mods: int
    ) -> None:
        event_data = {"point": point, "button": button, "mods": mods}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MouseReleaseEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_scroll(
        self,
        point: Vect3,
        offset: Vect3,
        x_pixel_offset: float,
        y_pixel_offset: float
    ) -> None:
        event_data = {"point": point, "offset": offset}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MouseScrollEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

        rel_offset = y_pixel_offset / self.camera.get_pixel_height()
        self.frame.scale(
            1 - self.scroll_sensitivity * rel_offset,
            about_point=point
        )

    def on_key_release(
        self,
        symbol: int,
        modifiers: int
    ) -> None:
        event_data = {"symbol": symbol, "modifiers": modifiers}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.KeyReleaseEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

    def on_key_press(
        self,
        symbol: int,
        modifiers: int
    ) -> None:
        # Some backends/devices can emit non-Unicode key symbols.
        # Ignore them silently to avoid warning spam during interaction.
        if symbol < 0 or symbol > 0x10FFFF:
            return
        try:
            char = chr(symbol)
        except (OverflowError, ValueError):
            return

        event_data = {"symbol": symbol, "modifiers": modifiers}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.KeyPressEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

        if char == manim_config.key_bindings.reset:
            self.play(self.camera.frame.animate.to_default_state())
        elif char == "y" and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL)):
            self.redo()
        elif char == "z" and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL)):
            self.undo()
        # command + q
        elif char == manim_config.key_bindings.quit and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL)):
            self.quit_interaction = True
        # Space or right arrow
        elif char == " " or symbol == PygletWindowKeys.RIGHT:
            if self.redo_stack:
                self.redo()
                return
            self.hold_on_wait = False
        elif symbol == PygletWindowKeys.LEFT:
            self.undo()

    def on_resize(self, width: int, height: int) -> None:
        pass

    def on_show(self) -> None:
        pass

    def on_hide(self) -> None:
        pass

    def on_close(self) -> None:
        """Exit hold_loop when window is closed."""
        self.hold_on_wait = False

    def focus(self) -> None:
        """
        Puts focus on the ManimGL window.
        """
        if not self.window:
            return
        self.window.focus()

    def set_background_color(self, background_color, background_opacity=1) -> None:
        self.camera.background_rgba = list(color_to_rgba(
            background_color, background_opacity
        ))


class SceneState():
    def __init__(self, scene: Scene, ignore: list[Mobject] | None = None, dont_modify: list[Mobject] | None = None):
        self.time = scene.time
        self.num_plays = scene.num_plays
        self.mobjects_to_copies = OrderedDict.fromkeys(scene.mobjects)
        if ignore:
            for mob in ignore:
                self.mobjects_to_copies.pop(mob, None)

        self.dont_modify = set(dont_modify) if dont_modify else set()

        last_m2c = scene.undo_stack[-1].mobjects_to_copies if scene.undo_stack else dict()
        for mob in self.mobjects_to_copies:
            # If it hasn't changed since the last state, just point to the
            # same copy as before
            if mob in last_m2c and last_m2c[mob].looks_identical(mob):
                self.mobjects_to_copies[mob] = last_m2c[mob]
            else:
                self.mobjects_to_copies[mob] = mob.copy()

    def __eq__(self, state: SceneState):
        return all((
            self.time == state.time,
            self.num_plays == state.num_plays,
            self.mobjects_to_copies == state.mobjects_to_copies
        ))

    def mobjects_match(self, state: SceneState):
        return self.mobjects_to_copies == state.mobjects_to_copies

    def n_changes(self, state: SceneState):
        m2c = state.mobjects_to_copies
        return sum(
            1 - int(mob in m2c and mob.looks_identical(m2c[mob]))
            for mob in self.mobjects_to_copies
        )

    def restore_scene(self, scene: Scene):
        scene.time = self.time
        scene.num_plays = self.num_plays
        scene.mobjects = [
            mob.become(mob_copy, match_updaters=True) if mob not in self.dont_modify else mob
            for mob, mob_copy in self.mobjects_to_copies.items()
        ]


class EndScene(Exception):
    pass


class ThreeDScene(Scene):
    samples = 4
    default_frame_orientation = (-30, 70)
    always_depth_test = True

    def add(self, *mobjects: Mobject, set_depth_test: bool = True, perp_stroke: bool = True):
        for mob in mobjects:
            if set_depth_test and not mob.is_fixed_in_frame() and self.always_depth_test:
                mob.apply_depth_test()
            if isinstance(mob, VMobject) and mob.has_stroke() and perp_stroke:
                mob.set_flat_stroke(False)
        super().add(*mobjects)
