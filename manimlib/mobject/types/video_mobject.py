from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
import asyncio
import os
import platform
if __import__('sys').platform != 'emscripten':
    import threading
    import av
else:
    threading = None
    av = None
import numpy as np
from PIL import Image

from manimlib.mobject.types.image_mobject import ImageMobject
from manimlib.renderer.texture import LayeredPixels
from manimlib.renderer.uniform_block import COMMON_UNIFORMS
from manimlib.renderer.uniform_block import uniform_block_dtype
from manimlib.utils.images import get_full_video_path
from manimlib.utils.rate_functions import linear

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self, Tuple


# How many decoded frames one source keeps, where it has not read the whole video in
DEFAULT_CACHE_SIZE: int = 32

# Up to how many bytes of decoded frames a video is preloaded, held whole both in memory
# and on the gpu rather than read a frame at a time.
PRELOAD_LIMIT: int = 64_000_000


# Default FFmpeg input format for live video devices.
DEFAULT_DEVICE_FORMATS: dict[str, str] = {
    "Linux": "v4l2",
    "Darwin": "avfoundation",
    "Windows": "dshow",
}


class VideoSource(object):
    """
    One video source.

    A source can either represent a normal video file or a live capture
    device.

    Normal files are decoded on demand and may optionally be preloaded.

    Live devices are decoded continuously on a background thread. The
    render thread never waits for the device to produce another frame:
    only the most recently decoded frame is exposed.
    """

    _sources: dict[tuple, VideoSource] = {}

    @classmethod
    async def create_browser(cls, path: str) -> VideoSource:
        """Create a browser-backed video source from an Emscripten file."""
        if __import__("sys").platform != "emscripten":
            raise RuntimeError("create_browser() is only available in Pyodide.")
        from js import Blob, URL, Uint8Array, document, Promise
        mime_types = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".ogv": "video/ogg", ".m4v": "video/mp4"}
        mime = mime_types.get(os.path.splitext(str(path))[1].lower(), "application/octet-stream")
        with open(path, "rb") as file:
            data = file.read()
        buffer = Uint8Array.new(len(data))
        buffer.assign(data)
        blob = Blob.new([buffer], {"type": mime})
        url = URL.createObjectURL(blob)
        video = document.createElement("video")
        video.preload = "auto"
        video.muted = True
        video.playsInline = True
        video.src = url
        if video.readyState < 1:
            await Promise.new(lambda resolve, reject: video.addEventListener("loadedmetadata", resolve, {"once": True}))
        if video.readyState < 2:
            await Promise.new(lambda resolve, reject: video.addEventListener("loadeddata", resolve, {"once": True}))
        source = cls(str(path), preload=False, browser_video=video, browser_url=url)
        await source._detect_browser_frame_rate()
        source.num_frames = max(1, round(source.duration * float(source.frame_rate)))
        await source._load_browser_frame(0)
        return source
    @classmethod
    def get(
        cls,
        path: str,
        preload: bool | None = None,
        *,
        live: bool = False,
        device_format: str | None = None,
        device_options: dict | None = None,
    ) -> VideoSource:
        """
        Get a shared source.

        Multiple VideoMobjects referring to the same source share decoding.
        This is particularly important for live devices: only one capture
        thread is created for a given device configuration.
        """
        options = device_options or {}

        try:
            options_key = tuple(sorted(options.items()))
            hash(options_key)
        except (TypeError, ValueError):
            # Device options are normally strings, but don't require every
            # possible PyAV option value to be hashable.
            options_key = repr(sorted(options.items(), key=lambda item: item[0]))

        key = (
            str(path),
            bool(live),
            device_format,
            options_key,
        )

        source = cls._sources.get(key)

        if source is None:
            source = cls._sources[key] = cls(
                path,
                preload,
                live=live,
                device_format=device_format,
                device_options=device_options,
            )
        elif preload and not source.preloaded and not source.live:
            source.read_all()

        return source

    @classmethod
    def close_all(cls) -> None:
        """
        Stop every live source and close every underlying container.

        This is mainly useful when shutting down a renderer or test process.
        """
        for source in list(cls._sources.values()):
            source.close()

        cls._sources.clear()

    def __deepcopy__(self, memo: dict) -> VideoSource:
        """
        Sources are shared rather than copied.
        """
        return self

    def __init__(
        self,
        path: str,
        preload: bool | None = None,
        *,
        live: bool = False,
        device_format: str | None = None,
        device_options: dict | None = None,
        browser_video=None,
        browser_url: str | None = None,
    ):
        self.path = str(path)
        self.live = live

        self.container = None
        self.stream = None
        self.browser_video = browser_video
        self.browser_url = browser_url
        self.browser_canvas = None
        self.browser_context = None
        self._browser_latest_task = None
        self._browser_requested_index = -1

        # ------------------------------------------------------------------
        # Live capture state
        # ------------------------------------------------------------------

        self._capture_thread = None
        self._capture_stop = None
        self._frame_lock = None

        # These synchronization primitives only exist for native live capture.
        # Browser video files do not use Python threads.
        if self.live:
            if threading is None:
                raise RuntimeError(
                    "Live VideoMobject capture is not available in Pyodide."
                )
            self._capture_stop = threading.Event()
            self._frame_lock = threading.Lock()

        # Only the newest frame is retained.
        #
        # This is intentionally NOT a queue. If the camera produces frames
        # faster than Manim renders them, old frames are discarded rather
        # than accumulating latency.
        self._latest_frame: np.ndarray | None = None
        self._latest_index: int = -1

        self._capture_error: Exception | None = None
        self._closed = False

        # ------------------------------------------------------------------
        # File state
        # ------------------------------------------------------------------

        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.decoder = None
        self.next_index = 0

        self.all_frames: np.ndarray | None = None
        self.preloaded = False

        if self.live:
            self._init_live(
                device_format=device_format,
                device_options=device_options,
            )
        elif self.browser_video is not None:
            self._init_browser()
        else:
            self._init_file(preload)

    # ======================================================================
    # Normal video files
    # ======================================================================

    def _init_browser(self) -> None:
        """Initialize an HTMLVideoElement-backed browser video source."""
        self.width = int(self.browser_video.videoWidth)
        self.height = int(self.browser_video.videoHeight)
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError(f"Browser video {self.path!r} has no intrinsic dimensions.")
        # HTMLVideoElement does not expose the source frame rate directly.
        # It is detected asynchronously from requestVideoFrameCallback().
        self.frame_rate = Fraction(30, 1)
        duration = float(self.browser_video.duration)
        if not np.isfinite(duration) or duration <= 0:
            raise RuntimeError(f"Browser video {self.path!r} has an invalid duration: {duration!r}.")
        self.duration = duration
        self.num_frames = max(1, round(duration * float(self.frame_rate)))
        self.all_frames = None
        self.preloaded = False
        from js import document
        self.browser_canvas = document.createElement("canvas")
        self.browser_canvas.width = self.width
        self.browser_canvas.height = self.height
        self.browser_context = self.browser_canvas.getContext("2d")
        self._latest_frame = None
        self._latest_index = -1

    async def _detect_browser_frame_rate(self) -> None:
        """Estimate the source frame rate from decoded video frame timestamps."""
        video = self.browser_video

        if not hasattr(video, "requestVideoFrameCallback"):
            return

        from js import Promise

        original_time = float(video.currentTime)

        try:
            if abs(original_time) > 1e-6:
                video.currentTime = 0
                await Promise.new(
                    lambda resolve, reject: video.addEventListener(
                        "seeked", resolve, {"once": True}
                    )
                )

            def collect_frames(resolve, reject):
                times = []

                def callback(now, metadata):
                    media_time = float(metadata.mediaTime)
                    if not times or media_time > times[-1] + 1e-6:
                        times.append(media_time)

                    if len(times) >= 9:
                        resolve(times)
                    else:
                        video.requestVideoFrameCallback(callback)

                # Register the callback before starting playback.  Do not
                # await video.play(): the JS play() promise is not needed
                # here and awaiting it through Pyodide can stall the Python
                # coroutine even though the media element is playing.
                video.requestVideoFrameCallback(callback)
                video.play()

            samples = list(await Promise.new(collect_frames))

            deltas = np.diff(np.asarray(samples, dtype=float))
            deltas = deltas[deltas > 1e-5]
            if len(deltas):
                estimated_fps = 1.0 / float(np.median(deltas))
                if np.isfinite(estimated_fps) and 1.0 <= estimated_fps <= 240.0:
                    self.frame_rate = Fraction(estimated_fps).limit_denominator(1000)
        except Exception:
            self.frame_rate = Fraction(30, 1)
        finally:
            video.pause()
            video.currentTime = min(
                original_time,
                max(0.0, self.duration - 1e-6),
            )

    async def _load_browser_frame(self, index: int) -> None:
        """Seek the HTMLVideoElement and copy its current decoded frame."""
        index = int(np.clip(index, 0, self.num_frames - 1))
        target = min(index / float(self.frame_rate), max(0.0, self.duration - 1e-6))
        video = self.browser_video
        if abs(float(video.currentTime) - target) > 1e-6:
            from js import Promise
            video.currentTime = target
            await Promise.new(lambda resolve, reject: video.addEventListener("seeked", resolve, {"once": True}))
        self.browser_context.drawImage(video, 0, 0, self.width, self.height)
        image_data = self.browser_context.getImageData(0, 0, self.width, self.height)
        pixels = np.frombuffer(bytes(image_data.data.to_py()), dtype=np.uint8).reshape(self.height, self.width, 4).copy()
        self._latest_frame = pixels
        self._latest_index = index

    async def _browser_frame_worker(self) -> None:
        """Serialize browser seeks while keeping only the newest request."""
        while True:
            index = self._browser_requested_index
            if index < 0 or index == self._latest_index:
                return
            await self._load_browser_frame(index)
            if self._browser_requested_index == index:
                return

    def request_frame(self, index: int) -> None:
        """Schedule a browser decode without blocking Manim rendering."""
        if self.browser_video is None:
            return
        self._browser_requested_index = int(np.clip(index, 0, self.num_frames - 1))
        task = self._browser_latest_task
        if task is None or task.done():
            self._browser_latest_task = asyncio.create_task(self._browser_frame_worker())
    def _init_file(self, preload: bool | None) -> None:
        if av is None:
            raise RuntimeError(
                "VideoMobject file decoding via PyAV/FFmpeg is not available in Pyodide. "
                "Use a browser HTMLVideoElement-backed video source instead."
            )
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

        self.width = self.stream.codec_context.width
        self.height = self.stream.codec_context.height

        self.frame_rate = Fraction(
            self.stream.average_rate
            or self.stream.guessed_rate
            or 30
        )

        self.num_frames = self.get_num_frames()
        self.duration = float(self.num_frames / self.frame_rate)

        self.all_frames = None
        self.preloaded = False

        if preload is None:
            preload = self.fits_at_once()

        if preload:
            self.read_all()

    # ======================================================================
    # Live video devices
    # ======================================================================

    def _init_live(
        self,
        *,
        device_format: str | None,
        device_options: dict | None,
    ) -> None:
        """
        Initialize a live capture device.

        Opening the container itself happens here, but decoding does not.
        All blocking frame acquisition occurs exclusively on the capture
        thread.
        """
        if av is None or threading is None:
            raise RuntimeError("Live VideoMobject capture is native-only; browser capture must use MediaDevices.")
        if device_format is None:
            device_format = DEFAULT_DEVICE_FORMATS.get(
                platform.system()
            )

        if device_format is None:
            raise RuntimeError(
                "Could not determine the video-device format for "
                f"{platform.system()!r}. Specify device_format explicitly."
            )

        options = dict(device_options or {})

        self.container = av.open(
            self.path,
            format=device_format,
            options=options,
        )

        if not self.container.streams.video:
            self.container.close()
            raise RuntimeError(
                f"No video stream was found in device {self.path!r}."
            )

        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

        self.width = self.stream.codec_context.width
        self.height = self.stream.codec_context.height

        self.frame_rate = Fraction(
            self.stream.average_rate
            or self.stream.guessed_rate
            or 30
        )

        # A live device has no meaningful finite frame count or duration.
        self.num_frames = 1
        self.duration = float("inf")

        # Live sources are never preloaded.
        self.all_frames = None
        self.preloaded = False

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"VideoSource[{self.path}]",
            daemon=True,
        )

        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """
        Continuously decode the live source.

        This is the only method allowed to consume the device decoder.

        The important property is that there is no synchronization with the
        render loop. If the camera blocks waiting for its next frame, only
        this thread blocks.
        """
        index = 0

        try:
            for frame in self.container.decode(self.stream):
                if self._capture_stop.is_set():
                    break

                pixels = self.to_rgba(frame)

                # The conversion above may be relatively expensive, so the
                # lock is deliberately acquired only after conversion.
                with self._frame_lock:
                    self._latest_frame = pixels
                    self._latest_index = index

                index += 1

        except Exception as exc:
            if not self._capture_stop.is_set():
                self._capture_error = exc

        finally:
            self.decoder = None

    @property
    def latest_index(self) -> int:
        """
        Index of the most recently decoded frame.

        Browser-backed sources do not use the native capture lock.
        """
        if self.browser_video is not None:
            return self._latest_index

        with self._frame_lock:
            return self._latest_index

    def get_latest_frame(self) -> np.ndarray:
        """
        Return the newest frame immediately.

        This function NEVER waits for the capture device.

        Before the first frame arrives, a transparent blank frame is returned.
        """
        if self.browser_video is not None:
            pixels = self._latest_frame
        else:
            with self._frame_lock:
                pixels = self._latest_frame

        if pixels is None:
            return self.blank_frame()

        return pixels

    @property
    def capture_error(self) -> Exception | None:
        """
        Exception raised by the capture thread, if any.
        """
        return self._capture_error

    def close(self) -> None:
        """
        Stop capture and close the underlying PyAV container.

        This method is safe to call more than once.
        """
        if self._closed:
            return

        self._closed = True
        if self._capture_stop is not None:
            self._capture_stop.set()

        if self._browser_latest_task is not None and not self._browser_latest_task.done():
            self._browser_latest_task.cancel()

        thread = self._capture_thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=0.25)

        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
        if self.browser_url is not None:
            try:
                from js import URL
                URL.revokeObjectURL(self.browser_url)
            except Exception:
                pass
            self.browser_url = None
        self.container = None

    # ======================================================================
    # Common source API
    # ======================================================================

    def get_num_frames(self) -> int:
        """
        Determine the number of frames of a normal video file.
        """
        if self.stream.frames:
            return self.stream.frames

        if self.stream.duration and self.stream.time_base:
            seconds = float(
                self.stream.duration * self.stream.time_base
            )

            return max(
                1,
                round(seconds * float(self.frame_rate)),
            )

        if self.container.duration:
            seconds = self.container.duration / av.time_base

            return max(
                1,
                round(seconds * float(self.frame_rate)),
            )

        return sum(
            1
            for _ in self.container.decode(self.stream)
        )

    def fits_at_once(self) -> bool:
        """
        Whether the whole normal video can be held in memory.
        """
        return (
            self.num_frames
            * self.width
            * self.height
            * 4
            <= PRELOAD_LIMIT
        )

    def get_all_frames(self) -> np.ndarray:
        """
        Return every frame of a preloaded normal video.
        """
        if self.live:
            raise RuntimeError(
                "Live video sources cannot be preloaded."
            )

        if self.all_frames is None:
            self.read_all()

        return self.all_frames

    def read_all(self) -> None:
        """
        Decode an entire normal video into memory.
        """
        if self.live:
            raise RuntimeError(
                "Live video sources cannot be preloaded."
            )

        self.container.seek(
            0,
            stream=self.stream,
        )

        frames = [
            self.to_rgba(frame)
            for frame in self.container.decode(self.stream)
        ]

        # What was actually decoded is authoritative.
        self.num_frames = max(
            1,
            len(frames),
        )

        self.duration = float(
            self.num_frames / self.frame_rate
        )

        self.all_frames = np.stack(
            frames or [self.blank_frame()]
        )

        self.cache.clear()
        self.preloaded = True

    def blank_frame(self) -> np.ndarray:
        """
        Transparent frame with the source dimensions.
        """
        return np.zeros(
            (
                self.height,
                self.width,
                4,
            ),
            dtype=np.uint8,
        )

    def to_rgba(self, frame) -> np.ndarray:
        """
        Convert a decoded PyAV frame into straight RGBA bytes.
        """
        return frame.to_ndarray(format="rgba")

    # ======================================================================
    # Random/sequential decoding for normal files
    # ======================================================================

    def seek(self, index: int) -> None:
        """
        Seek to the keyframe at or before a requested frame.

        Not used by live sources.
        """
        if self.live:
            return

        time_base = (
            self.stream.time_base
            or Fraction(
                1,
                int(self.frame_rate),
            )
        )

        offset = int(
            index
            / self.frame_rate
            / time_base
        )

        self.container.seek(
            offset,
            stream=self.stream,
            backward=True,
            any_frame=False,
        )

        self.decoder = self.container.decode(
            self.stream
        )

        self.next_index = None

    def get_frame(self, index: int) -> np.ndarray:
        """
        Get one frame.

        For live sources the requested index is intentionally ignored:
        the newest frame is always returned.
        """
        if self.live or self.browser_video is not None:
            if self._latest_frame is None:
                return self.blank_frame()
            return self._latest_frame

        if self.preloaded:
            return self.get_all_frames()[
                int(
                    np.clip(
                        index,
                        0,
                        self.num_frames - 1,
                    )
                )
            ]

        while True:
            index = int(
                np.clip(
                    index,
                    0,
                    self.num_frames - 1,
                )
            )

            pixels = self.cache.get(index)

            if pixels is not None:
                self.cache.move_to_end(index)
                return pixels

            pixels = self.read_up_to(index)

            if pixels is not None:
                return pixels

            if index == 0:
                return self.blank_frame()

            # The container claimed more frames than were actually
            # decodable.
            self.num_frames = index
            index -= 1

    def read_up_to(
        self,
        index: int,
    ) -> np.ndarray | None:
        """
        Decode forward until a requested frame is reached.

        For live sources this is never allowed to block.
        """
        if self.live:
            return self.get_latest_frame()

        if (
            self.decoder is None
            or self.next_index is None
            or index < self.next_index
        ):
            self.seek(index)

        for frame in self.decoder:
            at = self.index_of(frame)

            self.next_index = at + 1

            if at >= index:
                pixels = self.to_rgba(frame)

                self.remember(
                    at,
                    pixels,
                )

                if at != index:
                    self.remember(
                        index,
                        pixels,
                    )

                return pixels

        self.decoder = None
        return None

    def index_of(self, frame) -> int:
        """
        Determine the frame number from its timestamp.
        """
        if (
            frame.pts is None
            or self.stream.time_base is None
        ):
            return self.next_index or 0

        seconds = float(
            frame.pts * self.stream.time_base
        )

        return round(
            seconds * float(self.frame_rate)
        )

    def remember(
        self,
        index: int,
        pixels: np.ndarray,
    ) -> None:
        """
        Put one decoded frame into the bounded cache.
        """
        self.cache[index] = pixels
        self.cache.move_to_end(index)

        while len(self.cache) > DEFAULT_CACHE_SIZE:
            self.cache.popitem(
                last=False
            )


class VideoFrames(LayeredPixels):
    """
    Frames belonging to a VideoMobject.

    A preloaded file has all frames on the GPU as layers.

    A normal streaming file keeps one GPU layer and replaces it when the
    requested frame changes.

    A live device also keeps one GPU layer, but its CPU-side frame comes from
    the latest frame published by VideoSource's capture thread.
    """

    def __init__(
        self,
        video: VideoSource,
        loaded: int = -1,
        layers: np.ndarray | None = None,
    ):
        self.video = video
        self.loaded = loaded

        if video.preloaded:
            super().__init__(
                video.get_all_frames(),
                key=video.path,
            )

        elif layers is not None:
            super().__init__(layers)

        else:
            super().__init__(
                video.blank_frame()[
                    np.newaxis
                ]
            )

    @property
    def preloaded(self) -> bool:
        """
        Whether all frames exist simultaneously as GPU layers.
        """
        return self.key is not None

    @property
    def live(self) -> bool:
        return self.video.live

    def copy(self) -> VideoFrames:
        """
        Preloaded stacks are shared.

        Streaming and live sources create a VideoFrames object referencing
        the same VideoSource but retain their own GPU layer.
        """
        if self.preloaded:
            return self

        return VideoFrames(
            self.video,
            self.loaded,
            self.layers,
        )

    def load(self, index: int) -> None:
        """
        Make a frame available for drawing.

        For live capture, the requested frame index is not authoritative.
        The device has its own clock, so we only upload when the capture
        thread has published a new frame.
        """
        if self.preloaded:
            return

        if self.live or self.video.browser_video is not None:
            latest_index = self.video.latest_index
            if latest_index < 0:
                return
            if latest_index == self.loaded:
                return
            pixels = self.video.get_frame(latest_index)
            self.loaded = latest_index
            self.set_layers(pixels[np.newaxis])
            return

        # Normal file streaming.
        if index == self.loaded:
            return

        self.loaded = index

        self.set_layers(
            self.video.get_frame(index)[
                np.newaxis
            ]
        )

    def get_pixels(self, index: int) -> np.ndarray:
        """
        Return straight RGBA pixels of the requested frame.
        """
        return self.layers[
            index
            if self.preloaded
            else 0
        ]


class VideoMobject(ImageMobject):
    """
    A video showing whichever frame set_time last selected.

    Normal files retain the original VideoMobject behavior.

    With live=True, filename identifies a capture device instead of a video
    file. Frames are acquired on a background thread and the render loop
    always uses the latest available frame.

    A live source therefore does NOT synchronize Manim's frame rate to the
    camera's frame rate.

    Example on Linux:

        VideoMobject(
            "/dev/video0",
            live=True,
            device_format="v4l2",
            device_options={
                "video_size": "1280x720",
                "framerate": "30",
            },
        )

    Example on macOS:

        VideoMobject(
            "0",
            live=True,
            device_format="avfoundation",
            device_options={
                "video_size": "1280x720",
                "framerate": "30",
            },
        )

    Example on Windows:

        VideoMobject(
            "video=Integrated Camera",
            live=True,
            device_format="dshow",
            device_options={
                "video_size": "1280x720",
                "framerate": "30",
            },
        )
    """

    shader_file: str = "video.wgsl"

    uniform_dtype: np.dtype = uniform_block_dtype(
        *COMMON_UNIFORMS,
        ("frame", 1),
    )

    def __init__(
        self,
        filename: str,
        height: float = 4.0,
        time: float = 0.0,
        loop: bool = False,
        preload: bool | None = None,
        *,
        live: bool = False,
        device_format: str | None = None,
        device_options: dict | None = None,
        _browser_source: VideoSource | None = None,
        **kwargs,
    ):
        self.loop = loop

        # Used by normal video files.
        self._preload = preload

        # Used by live capture.
        self.live = live
        self.device_format = device_format
        self.device_options = device_options
        self._browser_source = _browser_source

        super().__init__(
            filename,
            height=height,
            **kwargs,
        )

        self.set_time(time)

    @classmethod
    async def create(cls, filename: str, **kwargs):
        """Asynchronously construct a browser-backed VideoMobject."""
        import sys
        if sys.platform != "emscripten":
            return cls(filename, **kwargs)
        path = str(get_full_video_path(filename))
        source = await VideoSource.create_browser(path)
        return cls(filename, _browser_source=source, **kwargs)
    # ======================================================================
    # Texture initialization
    # ======================================================================

    def init_texture(
        self,
        filename: str,
    ) -> VideoFrames:
        """
        Create the VideoFrames backing this mobject.
        """
        if self.live:
            return VideoFrames(
                VideoSource.get(
                    str(filename),
                    preload=False,
                    live=True,
                    device_format=self.device_format,
                    device_options=self.device_options,
                )
            )

        path = str(get_full_video_path(filename))
        if self._browser_source is not None:
            return VideoFrames(self._browser_source)
        return VideoFrames(VideoSource.get(path, self._preload))

    # ======================================================================
    # Source properties
    # ======================================================================

    @property
    def frames(self) -> VideoFrames:
        return self.textures["Texture"]

    @property
    def source(self) -> VideoSource:
        return self.frames.video

    @property
    def video_path(self) -> str:
        return self.source.path

    @property
    def preloaded(self) -> bool:
        return self.frames.preloaded

    # ======================================================================
    # Current frame
    # ======================================================================

    @property
    def frame_index(self) -> int:
        """
        Which frame is currently displayed.

        For live sources this is the latest frame received from the device.
        """
        if self.source.live:
            return max(
                0,
                self.source.latest_index,
            )

        index = round(
            float(
                self.uniforms["frame"]
            )
        )

        if self.loop:
            return (
                index
                % self.source.num_frames
            )

        return int(
            np.clip(
                index,
                0,
                self.source.num_frames - 1,
            )
        )

    def get_source_size(
        self,
    ) -> Tuple[int, int]:
        return (
            self.source.width,
            self.source.height,
        )

    # ======================================================================
    # Time/frame control
    # ======================================================================

    def set_time(
        self,
        time: float,
    ):
        """
        Show the frame corresponding to a given time.

        For live sources, Manim's time does not control the device. Instead,
        this method simply refreshes the texture with the latest frame.
        """
        if self.live:
            self.uniforms["frame"] = max(
                0,
                self.source.latest_index,
            )

            self.frames.load(
                self.source.latest_index
            )

            return self

        return self.set_frame(
            time
            * float(
                self.source.frame_rate
            )
        )

    def increment_time(
        self,
        dt: float,
    ):
        """
        Advance a normal video.

        For live sources, dt is deliberately ignored because the capture
        device has its own clock.
        """
        if self.live:
            self.frames.load(
                self.source.latest_index
            )

            return self

        return self.set_time(
            self.get_time() + dt
        )

    def play_from(
        self,
        time: float = 0.0,
    ):
        """
        Start playback.

        For live sources this simply installs a refresh updater.
        """
        if self.live:
            return self.add_updater(
                lambda mob, dt: mob.increment_time(dt)
            )

        self.set_time(time)

        return self.add_updater(
            lambda mob, dt: mob.increment_time(dt)
        )

    def animate_set_time(
        self,
        time: float,
        run_time=None,
        rate_func=linear,
        **kwargs,
    ):
        """
        Animate a normal video to a specified time.

        A live source has no finite timeline, so animate_set_time is invalid
        for live VideoMobjects.
        """
        if self.live:
            raise ValueError(
                "animate_set_time() cannot be used "
                "with a live VideoMobject."
            )

        if run_time is None:
            run_time = abs(
                time - self.get_time()
            )

        return self.animate(
            run_time=run_time,
            rate_func=rate_func,
        ).set_time(time)

    def set_frame(
        self,
        index: float,
    ):
        """
        Show a frame by frame number.

        For live sources the frame number is controlled by the capture
        device, so this operation simply refreshes the latest frame.
        """
        if self.live:
            self.frames.load(
                self.source.latest_index
            )

            return self

        if not self.loop:
            index = np.clip(
                index,
                0,
                self.source.num_frames - 1,
            )

        self.uniforms["frame"] = index
        if self.source.browser_video is not None:
            self.source.request_frame(self.frame_index)

        self.frames.load(
            self.frame_index
        )

        return self

    def get_time(self) -> float:
        """
        Current video time.

        A live source has no meaningful timeline controlled by Manim.
        """
        if self.live:
            return 0.0

        return float(
            self.uniforms["frame"]
            / self.source.frame_rate
        )

    def get_duration(self) -> float:
        return self.source.duration

    def get_num_frames(self) -> int:
        if self.live:
            return max(
                0,
                self.source.latest_index + 1,
            )

        return self.source.num_frames

    def get_frame_rate(self) -> float:
        return float(
            self.source.frame_rate
        )

    # ======================================================================
    # Pixel access
    # ======================================================================

    def get_pixels(self) -> np.ndarray:
        """
        Straight RGBA pixels of the frame currently shown.
        """
        return self.frames.get_pixels(
            self.frame_index
        )

    def interpolate(
        self,
        mobject1,
        mobject2,
        alpha,
        *args,
        **kwargs,
    ) -> Self:
        """
        Blend as any Mobject does, then ensure that the texture corresponds
        to the resulting frame.

        For live sources this also ensures that interpolation never blocks
        waiting for the device.
        """
        super().interpolate(
            mobject1,
            mobject2,
            alpha,
            *args,
            **kwargs,
        )

        self.frames.load(
            self.frame_index
        )

        return self

    @property
    def image(self) -> Image.Image:
        """
        Current frame as a PIL image.
        """
        return Image.fromarray(
            self.get_pixels(),
            mode="RGBA",
        )


class Sprite(VideoMobject):
    """
    A VideoMobject read nearest pixel rather than blended, keeping pixel art
    crisp however far it is scaled up.
    """

    texture_filter: str = "nearest"