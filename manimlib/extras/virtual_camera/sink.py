from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

try:
    import pyvirtualcam
except ImportError:  # pragma: no cover - exercised only when the extra is missing.
    pyvirtualcam = None

if TYPE_CHECKING:
    from manimlib.camera.camera import Camera
    from manimlib.scene.scene import Scene


class VirtualCameraSink:
    """Send rendered ManimGL frames to a system virtual camera.
    """

    def __init__(
        self,
        *,
        fps: int | None = None,
        device: str | None = None,
        backend: str | None = None,
        width: int | None = None,
        height: int | None = None,
        frame_stride: int = 1,
        block_until_next_frame: bool = False,
    ):
        if pyvirtualcam is None:
            raise ImportError(
                "pyvirtualcam is required for VirtualCameraSink. Install the optional virtual camera dependencies."
            )

        self.fps = fps
        self.device = device
        self.backend = backend
        self.width = width
        self.height = height
        self.frame_stride = max(1, int(frame_stride))
        self.block_until_next_frame = bool(block_until_next_frame)
        self._camera = None
        self._frame_count = 0

    def _ensure_camera(self, camera: "Camera") -> None:
        if self._camera is not None:
            return

        width = self.width or camera.get_pixel_width()
        height = self.height or camera.get_pixel_height()
        fps = self.fps or camera.fps

        self.width = width
        self.height = height

        self._camera = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=fps,
            device=self.device,
            backend=self.backend,
        )

    def push_frame(self, camera: "Camera") -> None:
        self._ensure_camera(camera)
        assert self._camera is not None

        self._frame_count += 1
        if self._frame_count % self.frame_stride != 0:
            return

        raw = camera.get_raw_fbo_data(dtype="f1", swap=False)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.height,
            self.width,
            camera.n_channels,
        )
        frame = np.ascontiguousarray(frame[::-1, :, :3])
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(
                "Virtual camera size does not match the current ManimGL camera frame size."
            )

        self._camera.send(frame)
        if self.block_until_next_frame:
            self._camera.sleep_until_next_frame()

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None


def bind_scene_to_virtual_camera(
    scene: "Scene",
    *,
    fps: int | None = None,
    device: str | None = None,
    backend: str | None = None,
    width: int | None = None,
    height: int | None = None,
    frame_stride: int = 1,
    block_until_next_frame: bool = False,
) -> VirtualCameraSink:
    sink = VirtualCameraSink(
        fps=fps,
        device=device,
        backend=backend,
        width=width,
        height=height,
        frame_stride=frame_stride,
        block_until_next_frame=block_until_next_frame,
    )
    scene.add_frame_sink(sink)
    return sink


def unbind_scene_from_virtual_camera(
    scene: "Scene",
    sink: VirtualCameraSink,
) -> None:
    scene.remove_frame_sink(sink)
    sink.close()