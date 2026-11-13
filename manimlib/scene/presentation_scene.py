from __future__ import annotations

from abc import ABC
import copy
import inspect
import sys
import time

from manimlib.animation.animation import Animation
from manimlib.event_keys import Keys
from manimlib.event_keys import Mods
from manimlib.logger import log
from manimlib.mobject.mobject import _AnimationBuilder
from manimlib.scene.interactive_scene import InteractiveScene
from manimlib.scene.scene import EndScene

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, Optional, Union
    from manimlib.window import Window

    TransitionCallable = Callable[..., Any]
    TransitionInput = Union[
        TransitionCallable,
        Animation,
        _AnimationBuilder,
        Iterable[Union[Animation, _AnimationBuilder]],
        None,
    ]


_SLIDE_COUNTER = 0


def _next_slide_order() -> int:
    global _SLIDE_COUNTER
    _SLIDE_COUNTER += 1
    return _SLIDE_COUNTER


def _callable_accepts_scene(func: Callable[..., Any]) -> bool:
    """Return True if an unbound callable expects the scene (`self`) as its first argument."""
    if inspect.ismethod(func):
        return False
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    params = list(sig.parameters.values())
    if not params:
        return False

    first = params[0]
    if first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        # Required positional parameter (e.g. `self` or `scene`)
        if first.default is inspect.Parameter.empty:
            return True
        if first.name in ("self", "scene"):
            return True
    return False


def _extract_animations(
    value: Any,
    copy_animations: bool = False,
) -> list[Animation | _AnimationBuilder] | None:
    if isinstance(value, (Animation, _AnimationBuilder)):
        return [value.copy() if (copy_animations and isinstance(value, Animation)) else value]
    if isinstance(value, (list, tuple)):
        items = list(value)
        if items and all(isinstance(x, (Animation, _AnimationBuilder)) for x in items):
            if copy_animations:
                return [x.copy() if isinstance(x, Animation) else x for x in items]
            return items
    return None


class _BoundSlide:
    """Instance-bound wrapper around a Slide descriptor."""

    def __init__(self, slide_obj: Slide, scene: PresentationScene):
        self._slide = slide_obj
        self._scene = scene

    @property
    def name(self) -> str:
        return self._slide.name

    def __call__(self) -> None:
        self._slide.run_forward(self._scene)

    def forward(self) -> None:
        self._slide.run_forward(self._scene)

    def backward(self) -> None:
        self._slide.run_backward(self._scene)

    reverse = backward
    back = backward


class Slide:
    """Represents a single presentation step with user-defined forward and backward transitions."""

    def __init__(
        self,
        forward: TransitionInput = None,
        backward: TransitionInput = None,
        *,
        name: str | None = None,
    ):
        self.forward_spec: TransitionInput = forward
        self.backward_spec: TransitionInput = backward
        self.name: str = name or getattr(forward, "__name__", None) or "slide"
        self._order: int = _next_slide_order()

    def copy(self) -> Slide:
        cloned = Slide(
            forward=self.forward_spec,
            backward=self.backward_spec,
            name=self.name,
        )
        cloned._order = self._order
        return cloned

    def forward(self, func: TransitionInput) -> Slide:
        """Register or replace the forward transition for this slide."""
        self.forward_spec = func
        if self.name == "slide" and getattr(func, "__name__", None):
            self.name = func.__name__
        return self

    def backward(self, func: TransitionInput) -> Slide:
        """Register or replace the backward transition for this slide."""
        self.backward_spec = func
        return self

    reverse = backward
    back = backward

    def __get__(self, instance: PresentationScene | None, owner: type | None = None) -> Slide | _BoundSlide:
        if instance is None:
            return self
        return _BoundSlide(self, instance)

    def _invoke_spec(self, scene: PresentationScene, spec: TransitionInput) -> Any:
        if spec is None:
            return None

        direct_anims = _extract_animations(spec, copy_animations=True)
        if direct_anims is not None:
            scene.play(*direct_anims)
            return None

        if callable(spec):
            if _callable_accepts_scene(spec):
                result = spec(scene)
            else:
                result = spec()
            returned_anims = _extract_animations(result, copy_animations=False)
            if returned_anims is not None:
                scene.play(*returned_anims)
            return result

        raise TypeError(f"Unsupported slide transition specification: {spec!r}")

    async def _invoke_spec_async(self, scene: PresentationScene, spec: TransitionInput) -> Any:
        if spec is None:
            return None

        direct_anims = _extract_animations(spec, copy_animations=True)
        if direct_anims is not None:
            await scene.play_async(*direct_anims)
            return None

        if callable(spec):
            if _callable_accepts_scene(spec):
                result = spec(scene)
            else:
                result = spec()
            if inspect.isawaitable(result):
                result = await result
            returned_anims = _extract_animations(result, copy_animations=False)
            if returned_anims is not None:
                await scene.play_async(*returned_anims)
            return result

        raise TypeError(f"Unsupported slide transition specification: {spec!r}")

    def run_forward(self, scene: PresentationScene) -> None:
        self._invoke_spec(scene, self.forward_spec)

    def run_backward(self, scene: PresentationScene) -> None:
        self._invoke_spec(scene, self.backward_spec)

    async def run_forward_async(self, scene: PresentationScene) -> None:
        await self._invoke_spec_async(scene, self.forward_spec)

    async def run_backward_async(self, scene: PresentationScene) -> None:
        await self._invoke_spec_async(scene, self.backward_spec)

    def __repr__(self) -> str:
        has_fwd = self.forward_spec is not None
        has_bwd = self.backward_spec is not None
        return f"Slide(name={self.name!r}, forward={has_fwd}, backward={has_bwd})"


def slide(
    forward: TransitionInput = None,
    backward: TransitionInput = None,
    *,
    name: str | None = None,
) -> Slide | Callable[[TransitionCallable], Slide]:
    """Decorator or helper to define a Slide with forward and backward transitions.

    Usage as a method decorator on a PresentationScene subclass::

        class MyPresentation(PresentationScene):
            @slide
            def intro(self):
                self.play(FadeIn(self.title))

            @intro.backward
            def intro_back(self):
                self.play(FadeOut(self.title))
    """
    if forward is None and backward is None and name is not None:
        def decorator(func: TransitionCallable) -> Slide:
            return Slide(forward=func, backward=None, name=name)
        return decorator

    if forward is None and backward is not None:
        def decorator(func: TransitionCallable) -> Slide:
            return Slide(forward=func, backward=backward, name=name)
        return decorator

    return Slide(forward=forward, backward=backward, name=name)


class PresentationScene(InteractiveScene, ABC):
    """Abstract base scene for interactive slide presentations.

    Unlike `Scene`, which executes `construct()` linearly to completion and only
    then enters `interact()`, `PresentationScene` runs inside an interactive
    event-driven presentation loop where user-defined `forward` and `backward`
    slide transitions are triggered on demand without using `Scene.seek()`.
    """

    next_slide_keys: tuple[int, ...] = (Keys.RIGHT, Keys.SPACE)
    previous_slide_keys: tuple[int, ...] = (Keys.LEFT,)
    play_first_slide_on_start: bool = False

    def __init__(self, **kwargs):
        if type(self) is PresentationScene:
            raise TypeError(
                "PresentationScene is an abstract class and must be subclassed."
            )
        # Slide navigation manages pauses between transitions directly; disable
        # Scene's linear presenter_mode hold_loop.
        kwargs["presenter_mode"] = False
        super().__init__(**kwargs)
        self.presenter_mode = False
        self.hold_on_wait = False

        self.slides: list[Slide] = []
        self.current_slide_index: int = 0
        self.is_transitioning: bool = False
        self._in_interact_loop: bool = False
        self._pending_transition: str | None = None

    def _collect_class_slides(self) -> None:
        """Collect `@slide`-decorated methods defined on the subclass in definition order."""
        collected: dict[int, Slide] = {}
        for cls in reversed(type(self).__mro__):
            if cls in (PresentationScene, InteractiveScene, object):
                continue
            for value in cls.__dict__.values():
                if isinstance(value, Slide):
                    collected[id(value)] = value

        ordered = sorted(collected.values(), key=lambda s: s._order)
        existing_orders = {s._order for s in self.slides}
        for s in ordered:
            if s._order not in existing_orders:
                self.slides.append(s.copy())
                existing_orders.add(s._order)

    def setup(self):
        if sys.platform == "emscripten":
            return self.setup_async()
        super().setup()
        self._collect_class_slides()

    async def setup_async(self):
        await super().setup_async()
        self._collect_class_slides()

    # Slide registration API

    def add_slide(
        self,
        forward: Slide | TransitionInput = None,
        backward: TransitionInput = None,
        *,
        name: str | None = None,
    ) -> Slide | Callable[[TransitionCallable], Slide]:
        """Register a slide with forward and backward transitions.

        Can be called directly (`self.add_slide(forward_fn, backward_fn)`) or used
        as a decorator (`@self.slide` / `@self.add_slide`) followed by `@step.backward`.
        """
        if isinstance(forward, Slide):
            slide_obj = forward
            if backward is not None:
                slide_obj.backward(backward)
            if name is not None:
                slide_obj.name = name
            if slide_obj not in self.slides:
                self.slides.append(slide_obj)
            return slide_obj

        if forward is None and (backward is not None or name is not None):
            def decorator(func: TransitionCallable) -> Slide:
                s = Slide(forward=func, backward=backward, name=name)
                self.slides.append(s)
                return s
            return decorator

        slide_obj = Slide(forward=forward, backward=backward, name=name)
        self.slides.append(slide_obj)
        return slide_obj

    slide = add_slide

    def clear_slides(self) -> None:
        self.slides.clear()
        self.current_slide_index = 0
        self._pending_transition = None

    @property
    def current_slide(self) -> int:
        return self.current_slide_index

    @property
    def num_slides(self) -> int:
        return len(self.slides)

    def can_go_forward(self) -> bool:
        return not self.is_transitioning and self.current_slide_index < len(self.slides)

    def can_go_backward(self) -> bool:
        return not self.is_transitioning and self.current_slide_index > 0

    def _sync_realtime_clock(self) -> None:
        self.virtual_animation_start_time = self.time
        self.real_animation_start_time = time.time()

    # Slide transition execution

    def next_slide(self) -> bool:
        """Execute the forward transition for the next slide if available."""
        if not self.can_go_forward():
            return False

        slide_obj = self.slides[self.current_slide_index]
        self.is_transitioning = True
        self._pending_transition = None
        try:
            self._sync_realtime_clock()
            slide_obj.run_forward(self)
            self.current_slide_index += 1
            self.regenerate_selection_search_set()
            self._sync_realtime_clock()
            self.update_frame(dt=0, force_draw=True)
        finally:
            self.is_transitioning = False
            self._pending_transition = None
            self._sync_realtime_clock()
        return True

    def previous_slide(self) -> bool:
        """Execute the backward transition for the current slide if available."""
        if not self.can_go_backward():
            return False

        slide_obj = self.slides[self.current_slide_index - 1]
        if slide_obj.backward_spec is None:
            log.warning(f"Slide '{slide_obj.name}' has no backward transition defined.")
            return False

        self.is_transitioning = True
        self._pending_transition = None
        try:
            self._sync_realtime_clock()
            slide_obj.run_backward(self)
            self.current_slide_index -= 1
            self.regenerate_selection_search_set()
            self._sync_realtime_clock()
            self.update_frame(dt=0, force_draw=True)
        finally:
            self.is_transitioning = False
            self._pending_transition = None
            self._sync_realtime_clock()
        return True

    prev_slide = previous_slide

    def go_to_slide(self, target_index: int) -> bool:
        """Transition step by step to `target_index` (0 <= target_index <= num_slides)."""
        if self.is_transitioning:
            return False
        target_index = max(0, min(int(target_index), len(self.slides)))
        while self.current_slide_index < target_index:
            if not self.next_slide():
                return False
        while self.current_slide_index > target_index:
            if not self.previous_slide():
                return False
        return True

    async def next_slide_async(self) -> bool:
        if sys.platform != "emscripten":
            return self.next_slide()
        if not self.can_go_forward():
            return False

        slide_obj = self.slides[self.current_slide_index]
        self.is_transitioning = True
        self._pending_transition = None
        try:
            self._sync_realtime_clock()
            await slide_obj.run_forward_async(self)
            self.current_slide_index += 1
            self.regenerate_selection_search_set()
            self._sync_realtime_clock()
            await self.update_frame_async(dt=0, force_draw=True)
        finally:
            self.is_transitioning = False
            self._pending_transition = None
            self._sync_realtime_clock()
        return True

    async def previous_slide_async(self) -> bool:
        if sys.platform != "emscripten":
            return self.previous_slide()
        if not self.can_go_backward():
            return False

        slide_obj = self.slides[self.current_slide_index - 1]
        if slide_obj.backward_spec is None:
            log.warning(f"Slide '{slide_obj.name}' has no backward transition defined.")
            return False

        self.is_transitioning = True
        self._pending_transition = None
        try:
            self._sync_realtime_clock()
            await slide_obj.run_backward_async(self)
            self.current_slide_index -= 1
            self.regenerate_selection_search_set()
            self._sync_realtime_clock()
            await self.update_frame_async(dt=0, force_draw=True)
        finally:
            self.is_transitioning = False
            self._pending_transition = None
            self._sync_realtime_clock()
        return True

    prev_slide_async = previous_slide_async

    async def go_to_slide_async(self, target_index: int) -> bool:
        if sys.platform != "emscripten":
            return self.go_to_slide(target_index)
        if self.is_transitioning:
            return False
        target_index = max(0, min(int(target_index), len(self.slides)))
        while self.current_slide_index < target_index:
            if not await self.next_slide_async():
                return False
        while self.current_slide_index > target_index:
            if not await self.previous_slide_async():
                return False
        return True

    # Disable checkpoint registration & seek API

    def play(
        self,
        *proto_animations: Animation | _AnimationBuilder,
        run_time: float | None = None,
        rate_func: Callable[[float], float] | None = None,
        lag_ratio: float | None = None,
        register: bool = False,
    ) -> None:
        super().play(
            *proto_animations,
            run_time=run_time,
            rate_func=rate_func,
            lag_ratio=lag_ratio,
            register=False,
        )

    async def play_async(
        self,
        *proto_animations: Animation | _AnimationBuilder,
        run_time: float | None = None,
        rate_func: Callable[[float], float] | None = None,
        lag_ratio: float | None = None,
        register: bool = False,
    ) -> None:
        await super().play_async(
            *proto_animations,
            run_time=run_time,
            rate_func=rate_func,
            lag_ratio=lag_ratio,
            register=False,
        )

    def wait(
        self,
        duration: Optional[float] = None,
        stop_condition: Callable[[], bool] = None,
        note: str = None,
        ignore_presenter_mode: bool = True,
        register: bool = False,
    ) -> None:
        super().wait(
            duration=duration,
            stop_condition=stop_condition,
            note=note,
            ignore_presenter_mode=True,
            register=False,
        )

    async def wait_async(
        self,
        duration: float | None = None,
        stop_condition: Callable[[], bool] | None = None,
        note: str | None = None,
        ignore_presenter_mode: bool = True,
        register: bool = False,
    ) -> None:
        await super().wait_async(
            duration=duration,
            stop_condition=stop_condition,
            note=note,
            ignore_presenter_mode=True,
            register=False,
        )

    def seek(self, t: float, sync_audio: bool = True) -> None:
        raise RuntimeError(
            "PresentationScene does not use the seek API. "
            "Use next_slide() and previous_slide() instead."
        )

    async def seek_async(self, t: float, sync_audio: bool = True) -> None:
        raise RuntimeError(
            "PresentationScene does not use the seek API. "
            "Use next_slide_async() and previous_slide_async() instead."
        )

    # Non-linear presentation lifecycle

    def run(self) -> None:
        """Initialize the presentation and enter the interactive slide loop directly."""
        if sys.platform == "emscripten":
            raise RuntimeError(
                "Use await run_scene_from_class(...) for Pyodide scenes"
            )
        self._begin_build()
        try:
            self.construct()
            if self.window is None:
                # Headless mode (e.g. -w or -s): step through all slides forward
                if self.file_writer.write_to_movie or self.file_writer.save_last_frame or self.skip_animations:
                    while self.can_go_forward():
                        self.next_slide()
            else:
                if self.play_first_slide_on_start and self.slides and self.current_slide_index == 0:
                    self.next_slide()
                self.interact()
        except EndScene:
            pass
        except KeyboardInterrupt:
            print("", end="\r")
            self.file_writer.ended_with_interrupt = True
        finally:
            self.tear_down()

    def build(self):
        if sys.platform == "emscripten":
            raise RuntimeError("Use await build_async() in Pyodide")
        self._begin_build()
        try:
            self.construct()
            if self.window is None:
                if self.file_writer.write_to_movie or self.file_writer.save_last_frame or self.skip_animations:
                    while self.can_go_forward():
                        self.next_slide()
            elif self.play_first_slide_on_start and self.slides and self.current_slide_index == 0:
                self.next_slide()
        except EndScene:
            pass
        except KeyboardInterrupt:
            print("", end="\r")
            self.file_writer.ended_with_interrupt = True
        return self

    async def build_async(self):
        if sys.platform != "emscripten":
            return self.build()
        await self._begin_build_async()
        try:
            result = self.construct()
            if inspect.isawaitable(result):
                await result
            if self.play_first_slide_on_start and self.slides and self.current_slide_index == 0:
                await self.next_slide_async()
        except EndScene:
            pass
        except KeyboardInterrupt:
            print("", end="\r")
            self.file_writer.ended_with_interrupt = True
        return self

    def interact(self) -> None:
        """Main presentation loop driving frame updates and slide transitions."""
        if self.window is None:
            return
        if sys.platform == "emscripten":
            return
        log.info(
            "\nPresentation controls: Use `Right Arrow` or `Space` to advance, "
            "`Left Arrow` to go back.\n"
            "Tips: Using the keys `d`, `f`, or `z` you can interact with the scene. "
            "Press `command + q` or `esc` to quit"
        )
        self.skip_animations = False
        self._in_interact_loop = True
        self._sync_realtime_clock()
        self.update_frame(dt=0, force_draw=True)
        try:
            while not self.is_window_closing():
                if self._pending_transition is not None and not self.is_transitioning:
                    direction = self._pending_transition
                    self._pending_transition = None
                    if direction == "next":
                        self.next_slide()
                    elif direction == "prev":
                        self.previous_slide()
                    continue
                self.update_frame(1 / self.camera.fps)
        finally:
            self._in_interact_loop = False

    async def browser_interaction_loop(self) -> None:
        """Browser presentation loop handling slide transitions and interactive updates."""
        if sys.platform != "emscripten":
            raise RuntimeError("browser_interaction_loop() is only available in Pyodide")
        from manimlib.utils.browser_scheduler import next_animation_frame
        from js import window

        previous_time = await next_animation_frame()
        postprocess_settle_until = 0.0
        postprocess_settle_seconds = 0.5
        self._in_interact_loop = True

        try:
            while not self.is_window_closing():
                if self.window is None:
                    break
                if self._pending_transition is not None and not self.is_transitioning:
                    direction = self._pending_transition
                    self._pending_transition = None
                    if direction == "next":
                        await self.next_slide_async()
                    elif direction == "prev":
                        await self.previous_slide_async()
                    previous_time = await next_animation_frame()
                    continue

                current_time = await next_animation_frame()
                if self.is_window_closing() or self.window is None:
                    break
                dt = max(0.0, current_time - previous_time)
                previous_time = current_time

                self.window.poll_events()
                has_undrawn_event = self.window.has_undrawn_event()
                recording_active = bool(getattr(window, "__manimCaptureWebGPUFrame", None))

                if self.should_update_mobjects() or recording_active:
                    await self.update_frame_async(dt=dt, force_draw=True)
                elif has_undrawn_event:
                    postprocess_settle_until = current_time + postprocess_settle_seconds
                    await self.update_frame_async(force_draw=True)
                elif current_time < postprocess_settle_until:
                    await self.update_frame_async(dt=dt, force_draw=True)
        finally:
            self._in_interact_loop = False

    def _is_navigation_key(self, symbol: int) -> bool:
        return symbol in self.next_slide_keys or symbol in self.previous_slide_keys

    def _handle_slide_navigation_key(
        self,
        symbol: int,
        modifiers: int,
        is_async: bool = False,
    ) -> bool | Any:
        """Return True (or a coroutine in async mode) if the key press was handled as slide navigation."""
        if modifiers & Mods.ANY:
            return False

        # When mobjects are selected, let arrow keys nudge the selection as in InteractiveScene
        if len(self.selection) > 0 and symbol in (Keys.LEFT, Keys.RIGHT, Keys.UP, Keys.DOWN):
            return False

        if not self._is_navigation_key(symbol):
            return False

        # Ignore slide navigation inputs while a transition is currently running
        if self.is_transitioning or self._pending_transition is not None:
            return True

        if symbol in self.next_slide_keys:
            if self._in_interact_loop:
                if self.can_go_forward():
                    self._pending_transition = "next"
            else:
                if is_async:
                    return self.next_slide_async()
                self.next_slide()
            return True

        if symbol in self.previous_slide_keys:
            if self._in_interact_loop:
                if self.can_go_backward():
                    self._pending_transition = "prev"
            else:
                if is_async:
                    return self.previous_slide_async()
                self.previous_slide()
            return True

        return False

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if sys.platform == "emscripten":
            return self.on_key_press_async(symbol, modifiers)
        if self._handle_slide_navigation_key(symbol, modifiers, is_async=False):
            return
        if self.is_transitioning:
            # Allow quitting while transitioning, ignore other scene shortcuts
            try:
                char = chr(symbol)
            except (OverflowError, ValueError):
                char = ""
            if (char == "q" and (modifiers & Mods.CTRL_OR_CMD)) or symbol == Keys.ESCAPE:
                self.quit_interaction = True
            return
        super().on_key_press(symbol, modifiers)

    async def on_key_press_async(self, symbol: int, modifiers: int) -> None:
        handled = self._handle_slide_navigation_key(symbol, modifiers, is_async=True)
        if inspect.isawaitable(handled):
            await handled
            return
        if handled:
            return
        if self.is_transitioning:
            try:
                char = chr(symbol)
            except (OverflowError, ValueError):
                char = ""
            if (char == "q" and (modifiers & Mods.CTRL_OR_CMD)) or symbol == Keys.ESCAPE:
                self.quit_interaction = True
            return
        await super().on_key_press_async(symbol, modifiers)


__all__ = [
    "Slide",
    "slide",
    "PresentationScene",
]
