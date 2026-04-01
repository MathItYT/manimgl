from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from queue import Full
from queue import Queue
from typing import Callable
from urllib.request import urlretrieve

import cv2
import numpy as np

from manimlib.logger import log
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.scene.scene import Scene


@dataclass(frozen=True)
class HandMotionState:
    detected: bool
    x_norm: float
    y_norm: float
    dx: float
    dy: float
    pinch: bool
    gesture: str
    timestamp: float
    landmarks_norm: tuple[tuple[float, float], ...] = ()


class HandMesh(VMobject):
    """VMobject that draws a 2D hand mesh from normalized landmarks."""

    DEFAULT_CONNECTIONS: tuple[tuple[int, int], ...] = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )

    def __init__(
        self,
        *,
        reference_mobject: Mobject,
        connections: tuple[tuple[int, int], ...] | None = None,
        z_value: float = 0.0,
        hide_when_not_detected: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reference_mobject = reference_mobject
        self.connections = connections or self.DEFAULT_CONNECTIONS
        self.z_value = float(z_value)
        self.hide_when_not_detected = bool(hide_when_not_detected)
        self.set_points(np.array([reference_mobject.get_center()], dtype=float))

    def _to_scene_point(self, x_norm: float, y_norm: float) -> np.ndarray:
        left = self.reference_mobject.get_left()[0]
        top = self.reference_mobject.get_top()[1]
        x = left + (x_norm * self.reference_mobject.get_width())
        y = top - (y_norm * self.reference_mobject.get_height())
        return np.array([x, y, self.z_value], dtype=float)

    def update_from_state(self, state: HandMotionState) -> "HandMesh":
        if not state.detected or len(state.landmarks_norm) < 21:
            if self.hide_when_not_detected:
                self.set_stroke(opacity=0.0)
            return self

        points = [self._to_scene_point(x, y) for x, y in state.landmarks_norm]
        self.clear_points()
        for start_idx, end_idx in self.connections:
            if start_idx >= len(points) or end_idx >= len(points):
                continue
            self.start_new_path(points[start_idx])
            self.add_line_to(points[end_idx])
        self.set_stroke(opacity=1.0)
        return self


class HandMotionTracker:
    """Processes frames in a background thread and emits lightweight hand motion states."""

    def __init__(
        self,
        *,
        max_queue_frames: int = 3,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        pinch_threshold: float = 0.06,
        movement_threshold: float = 0.015,
        smoothing: float = 0.4,
        hand_landmarker_model_path: str | None = None,
        model_cache_dir: str | None = None,
        prefer_cuda: bool = True,
    ):
        self.max_num_hands = max(1, int(max_num_hands))
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.pinch_threshold = float(pinch_threshold)
        self.movement_threshold = float(movement_threshold)
        self.smoothing = min(0.99, max(0.0, float(smoothing)))
        self.hand_landmarker_model_path = hand_landmarker_model_path
        self.model_cache_dir = model_cache_dir
        self.prefer_cuda = bool(prefer_cuda)

        self._frame_queue: Queue[np.ndarray] = Queue(maxsize=max(1, int(max_queue_frames)))
        self._state_lock = threading.Lock()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_xy: tuple[float, float] | None = None
        self._state = HandMotionState(
            detected=False,
            x_norm=0.5,
            y_norm=0.5,
            dx=0.0,
            dy=0.0,
            pinch=False,
            gesture="none",
            timestamp=time.time(),
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="HandMotionTracker")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._running.clear()
        try:
            self._frame_queue.put_nowait(np.empty((0, 0, 0), dtype=np.uint8))
        except Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def on_frame(self, frame: np.ndarray) -> None:
        if not self._running.is_set():
            return
        if frame is None or not isinstance(frame, np.ndarray):
            return

        # Keep only the newest frame to reduce latency.
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                pass

        try:
            self._frame_queue.put_nowait(frame)
        except Full:
            pass

    def poll_latest_state(self) -> HandMotionState:
        with self._state_lock:
            return self._state

    def _set_state(self, state: HandMotionState) -> None:
        with self._state_lock:
            self._state = state

    def _set_no_hand_state(self) -> None:
        self._last_xy = None
        self._set_state(
            HandMotionState(
                detected=False,
                x_norm=self._state.x_norm,
                y_norm=self._state.y_norm,
                dx=0.0,
                dy=0.0,
                pinch=False,
                gesture="none",
                timestamp=time.time(),
                landmarks_norm=(),
            )
        )

    def _update_state_from_landmarks(self, hand_landmarks) -> None:
        if not hand_landmarks:
            self._set_no_hand_state()
            return

        hand = hand_landmarks[0]
        landmarks_norm = tuple((float(p.x), float(p.y)) for p in hand)

        index_tip = hand[8]
        thumb_tip = hand[4]

        x_raw = float(index_tip.x)
        y_raw = float(index_tip.y)

        if self._last_xy is None:
            x_smooth, y_smooth = x_raw, y_raw
            dx, dy = 0.0, 0.0
        else:
            last_x, last_y = self._last_xy
            alpha = self.smoothing
            x_smooth = (alpha * x_raw) + ((1.0 - alpha) * last_x)
            y_smooth = (alpha * y_raw) + ((1.0 - alpha) * last_y)
            dx = x_smooth - last_x
            dy = y_smooth - last_y

        self._last_xy = (x_smooth, y_smooth)

        pinch_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        pinch = pinch_distance < self.pinch_threshold

        if pinch:
            gesture = "pinch"
        elif abs(dx) > abs(dy) and abs(dx) > self.movement_threshold:
            gesture = "move_right" if dx > 0 else "move_left"
        elif abs(dy) > self.movement_threshold:
            gesture = "move_down" if dy > 0 else "move_up"
        else:
            gesture = "steady"

        self._set_state(
            HandMotionState(
                detected=True,
                x_norm=float(np.clip(x_smooth, 0.0, 1.0)),
                y_norm=float(np.clip(y_smooth, 0.0, 1.0)),
                dx=float(dx),
                dy=float(dy),
                pinch=pinch,
                gesture=gesture,
                timestamp=time.time(),
                landmarks_norm=landmarks_norm,
            )
        )

    def _load_mediapipe_backend(self):
        try:
            import mediapipe as mp
        except Exception as exc:
            raise RuntimeError(
                "mediapipe package is required. Install with: pip install \"manimgl[vision]\""
            ) from exc

        # Legacy/desktop backend.
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "hands"):
            return "solutions", mp, solutions.hands

        # Tasks API backend (available in newer/lightweight builds).
        try:
            from mediapipe.tasks.python import BaseOptions  # type: ignore[import-not-found]
            from mediapipe.tasks.python import vision  # type: ignore[import-not-found]
            return "tasks", mp, (BaseOptions, vision)
        except Exception as exc:
            raise RuntimeError(
                "Could not load any supported MediaPipe Hands backend. "
                "Expected either 'mediapipe.solutions.hands' or 'mediapipe.tasks.python.vision'."
            ) from exc

    def _resolve_hand_landmarker_model_path(self) -> str:
        candidates: list[str] = []

        if self.hand_landmarker_model_path:
            candidates.append(self.hand_landmarker_model_path)

        env_path = os.getenv("MANIMGL_HAND_LANDMARKER_MODEL_PATH")
        if env_path:
            candidates.append(env_path)

        candidates.extend([
            "hand_landmarker.task",
            str(Path("models") / "hand_landmarker.task"),
            str(Path("manimlib") / "extras" / "vision" / "models" / "hand_landmarker.task"),
        ])

        for raw_path in candidates:
            path = Path(raw_path).expanduser().resolve()
            if path.is_file():
                return str(path)

        if self.model_cache_dir:
            cache_dir = Path(self.model_cache_dir).expanduser().resolve()
        else:
            cache_dir = Path.home() / ".manimgl" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_model_path = cache_dir / "hand_landmarker.task"
        if cached_model_path.is_file():
            return str(cached_model_path)

        model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        try:
            log.info(f"Downloading MediaPipe hand landmarker model to {cached_model_path}")
            urlretrieve(model_url, str(cached_model_path))
            return str(cached_model_path)
        except Exception as exc:
            raise RuntimeError(
                "MediaPipe Tasks backend requires a hand landmarker model file (.task). "
                "Auto-download failed. Provide it via hand_landmarker_model_path or "
                "MANIMGL_HAND_LANDMARKER_MODEL_PATH. You can also set cache dir via "
                "model_cache_dir."
            ) from exc

        raise RuntimeError("Failed to resolve hand landmarker model path")

    def _worker(self) -> None:
        backend_name, mp, backend_payload = self._load_mediapipe_backend()

        if backend_name == "solutions":
            mp_hands = backend_payload
            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            ) as hands:
                while self._running.is_set():
                    try:
                        frame = self._frame_queue.get(timeout=0.1)
                    except Empty:
                        continue
                    if frame.size == 0:
                        continue

                    try:
                        self._process_frame_with_solutions(hands, frame)
                    except Exception:
                        log.exception("Hand tracking frame processing failed")
            return

        base_options_cls, vision = backend_payload
        model_path = self._resolve_hand_landmarker_model_path()

        def build_options(delegate):
            return vision.HandLandmarkerOptions(
                base_options=base_options_cls(
                    model_asset_path=model_path,
                    delegate=delegate,
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

        hand_landmarker = None
        if self.prefer_cuda and hasattr(base_options_cls, "Delegate"):
            try:
                hand_landmarker = vision.HandLandmarker.create_from_options(
                    build_options(base_options_cls.Delegate.GPU)
                )
                log.info("HandMotionTracker using MediaPipe GPU delegate")
            except Exception as exc:
                # GPU delegate typically unavailable because MediaPipe was compiled without CUDA support.
                # This is normal for pip-installed MediaPipe. CPU fallback (XNNPACK) is efficient.
                log.debug(f"MediaPipe GPU delegate not available (build compiled without GPU): {exc}")

        if hand_landmarker is None:
            cpu_delegate = None
            if hasattr(base_options_cls, "Delegate"):
                cpu_delegate = base_options_cls.Delegate.CPU
            hand_landmarker = vision.HandLandmarker.create_from_options(build_options(cpu_delegate))
            log.info("HandMotionTracker using MediaPipe CPU delegate (XNNPACK)")

        with hand_landmarker:
            while self._running.is_set():
                try:
                    frame = self._frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                if frame.size == 0:
                    continue

                try:
                    self._process_frame_with_tasks(mp, hand_landmarker, frame)
                except Exception:
                    log.exception("Hand tracking frame processing failed")

    def _process_frame_with_solutions(self, hands, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            return

        if frame.shape[2] == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)
        hand_landmarks = results.multi_hand_landmarks or []
        self._update_state_from_landmarks(hand_landmarks)

    def _process_frame_with_tasks(self, mp, hand_landmarker, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            return

        if frame.shape[2] == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hand_landmarker.detect(mp_image)
        hand_landmarks = result.hand_landmarks if result is not None else []
        self._update_state_from_landmarks(hand_landmarks)


def bind_hand_tracker_to_video(
    video_mobject,
    tracker: HandMotionTracker,
    enqueue_every_n_frames: int = 2,
    copy_frame: bool = True,
) -> None:
    """Taps frames from an existing VideoMobject iterator and sends them to tracker."""
    tracker.start()
    source_iterator = video_mobject.iterator
    step = max(1, int(enqueue_every_n_frames))

    def tapped_iterator():
        for frame_index, frame in enumerate(source_iterator):
            if frame_index % step == 0:
                tracker.on_frame(frame.copy() if copy_frame else frame)
            yield frame

    video_mobject.iterator = tapped_iterator()


def bind_hand_position_to_mobject(
    scene: Scene,
    target_mobject: Mobject,
    tracker: HandMotionTracker,
    reference_mobject: Mobject,
    *,
    update_fps: float = 30.0,
    z_value: float = 0.0,
    only_when_detected: bool = True,
):
    """Moves target_mobject according to normalized hand position over reference_mobject area."""
    period = 0.0 if update_fps <= 0 else (1.0 / update_fps)
    elapsed = 0.0

    def _updater(dt: float) -> None:
        nonlocal elapsed
        elapsed += dt
        if elapsed < period:
            return
        elapsed = 0.0

        state = tracker.poll_latest_state()
        if only_when_detected and not state.detected:
            return

        left = reference_mobject.get_left()[0]
        top = reference_mobject.get_top()[1]
        x = left + (state.x_norm * reference_mobject.get_width())
        y = top - (state.y_norm * reference_mobject.get_height())
        target_mobject.move_to(np.array([x, y, z_value], dtype=float))

    scene.add_updater(_updater)
    return _updater


def bind_hand_gesture_callback(
    scene: Scene,
    tracker: HandMotionTracker,
    callback: Callable[[HandMotionState], None],
    *,
    update_fps: float = 30.0,
):
    period = 0.0 if update_fps <= 0 else (1.0 / update_fps)
    elapsed = 0.0

    def _updater(dt: float) -> None:
        nonlocal elapsed
        elapsed += dt
        if elapsed < period:
            return
        elapsed = 0.0
        try:
            callback(tracker.poll_latest_state())
        except Exception:
            log.exception("Hand gesture callback failed")

    scene.add_updater(_updater)
    return _updater


def bind_hand_mesh_to_tracker(
    scene: Scene,
    hand_mesh: HandMesh,
    tracker: HandMotionTracker,
    *,
    update_fps: float = 30.0,
):
    period = 0.0 if update_fps <= 0 else (1.0 / update_fps)
    elapsed = 0.0

    def _updater(dt: float) -> None:
        nonlocal elapsed
        elapsed += dt
        if elapsed < period:
            return
        elapsed = 0.0
        hand_mesh.update_from_state(tracker.poll_latest_state())

    scene.add_updater(_updater)
    return _updater