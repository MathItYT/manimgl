from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from queue import Empty
from queue import Full
from queue import Queue
from typing import Any
from typing import Callable
from urllib.request import urlopen

import cv2
import numpy as np

from manimlib.scene.scene import Scene
from manimlib.logger import log
from manimlib.mobject.geometry import Circle
from manimlib.mobject.mobject import Group, Mobject
from manimlib.mobject.svg.typst_mobject import MarkdownMobject
from manimlib.mobject.svg.text_mobject import Text
from manimlib.mobject.types.image_mobject import ImageMobject
from manimlib.mobject.types.mask_mobject import MaskMobject
from manimlib.constants import WHITE


YOUTUBE_VIDEO_ID_PATTERN = re.compile(r"(?:v=|youtu\.be/|/live/)([A-Za-z0-9_-]{11})")


@dataclass(frozen=True)
class YouTubeChatMessage:
    author: str
    text: str
    timestamp: float
    avatar_url: str | None = None
    avatar_image: np.ndarray | None = None


class YouTubeLiveChatClient:
    """Background reader for YouTube live chat messages.

    This client uses pytchat, which can attach directly to a live video id.
    """

    def __init__(
        self,
        video_id: str,
        *,
        max_queue_messages: int = 256,
        poll_interval_seconds: float = 0.2,
        avatar_size_px: int = 96,
        avatar_request_timeout_seconds: float = 2.0,
    ):
        self.video_id = self._normalize_video_id(video_id)
        self.max_queue_messages = max(1, int(max_queue_messages))
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self.avatar_size_px = max(16, int(avatar_size_px))
        self.avatar_request_timeout_seconds = max(0.5, float(avatar_request_timeout_seconds))

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._chat = None
        self._message_queue: Queue[YouTubeChatMessage] = Queue(maxsize=self.max_queue_messages)
        self._subscribers: list[Queue[YouTubeChatMessage]] = []
        self._subscribers_lock = threading.Lock()
        self._avatar_cache: dict[str, np.ndarray] = {}

    def _download_avatar_image(self, avatar_url: str) -> np.ndarray | None:
        try:
            with urlopen(avatar_url, timeout=self.avatar_request_timeout_seconds) as response:
                raw = response.read()
            encoded = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if image is None:
                return None

            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

            image = cv2.resize(
                image,
                (self.avatar_size_px, self.avatar_size_px),
                interpolation=cv2.INTER_AREA,
            )
            return image
        except Exception:
            return None

    def _get_avatar_image(self, avatar_url: str | None) -> np.ndarray | None:
        if not avatar_url:
            return None
        if avatar_url in self._avatar_cache:
            return self._avatar_cache[avatar_url]

        image = self._download_avatar_image(avatar_url)
        if image is None:
            return None

        self._avatar_cache[avatar_url] = image
        return image

    @staticmethod
    def _normalize_video_id(video_or_url: str) -> str:
        candidate = (video_or_url or "").strip()
        if len(candidate) == 11 and all(ch.isalnum() or ch in "-_" for ch in candidate):
            return candidate

        match = YOUTUBE_VIDEO_ID_PATTERN.search(candidate)
        if not match:
            raise ValueError(
                "video_id must be an 11-char YouTube video id or a URL containing one"
            )
        return match.group(1)

    @staticmethod
    def _queue_put_latest(queue_obj: Queue[YouTubeChatMessage], item: YouTubeChatMessage) -> None:
        try:
            queue_obj.put_nowait(item)
            return
        except Full:
            pass

        try:
            queue_obj.get_nowait()
        except Empty:
            pass

        try:
            queue_obj.put_nowait(item)
        except Full:
            pass

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        if self._chat is None:
            try:
                import pytchat  # type: ignore[import-not-found]
            except Exception as exc:
                log.error("pytchat package is required. Install with: pip install manimgl[youtube_chat]")
                raise RuntimeError("pytchat package is required") from exc

            # pytchat registers signal handlers, so this must happen on main thread.
            self._chat = pytchat.create(video_id=self.video_id)

        self._running.set()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="YouTubeLiveChatClient")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._running.clear()
        if timeout > 0 and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        if self._chat is not None:
            try:
                self._chat.terminate()
            except Exception:
                pass
            self._chat = None

    def subscribe_messages(self, max_queue_size: int = 128) -> Queue[YouTubeChatMessage]:
        message_queue: Queue[YouTubeChatMessage] = Queue(maxsize=max(1, int(max_queue_size)))
        with self._subscribers_lock:
            self._subscribers.append(message_queue)
        return message_queue

    def unsubscribe_messages(self, message_queue: Queue[YouTubeChatMessage]) -> None:
        with self._subscribers_lock:
            self._subscribers = [q for q in self._subscribers if q is not message_queue]

    def poll_latest_message(self) -> YouTubeChatMessage | None:
        latest: YouTubeChatMessage | None = None
        while True:
            try:
                latest = self._message_queue.get_nowait()
            except Empty:
                return latest

    def _publish_message(self, message: YouTubeChatMessage) -> None:
        self._queue_put_latest(self._message_queue, message)
        with self._subscribers_lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            self._queue_put_latest(subscriber, message)

    def _worker(self) -> None:
        chat = self._chat
        if chat is None:
            return

        try:
            while self._running.is_set() and chat.is_alive():
                data = chat.get()
                for item in data.sync_items():
                    text = (getattr(item, "message", "") or "").strip()
                    if not text:
                        continue
                    author_obj = getattr(item, "author", None)
                    avatar_url = getattr(author_obj, "imageUrl", None)
                    message = YouTubeChatMessage(
                        author=(getattr(author_obj, "name", "") or "anon").strip(),
                        text=text,
                        timestamp=time.time(),
                        avatar_url=avatar_url,
                        avatar_image=self._get_avatar_image(avatar_url),
                    )
                    self._publish_message(message)
                time.sleep(self.poll_interval_seconds)
        except Exception:
            log.exception("YouTube chat loop failed")


def bind_youtube_chat_callback(
    scene,
    chat_client: YouTubeLiveChatClient,
    callback: Callable[[YouTubeChatMessage], None],
    *,
    update_fps: float = 20.0,
):
    """Attach a scene updater that dispatches incoming chat messages to callback."""
    chat_client.start()
    message_queue = chat_client.subscribe_messages()

    period = 0.0 if update_fps <= 0 else (1.0 / update_fps)
    elapsed = 0.0

    def _dispatch_chat_events(dt: float) -> None:
        nonlocal elapsed
        elapsed += dt
        if elapsed < period:
            return
        elapsed = 0.0

        while True:
            try:
                message = message_queue.get_nowait()
            except queue.Empty:
                break

            try:
                callback(message)
            except Exception:
                log.exception("YouTube chat callback failed")

    scene.add_updater(_dispatch_chat_events)
    return _dispatch_chat_events


def bind_youtube_chat_to_text(
    scene,
    text_mobject: Text,
    chat_client: YouTubeLiveChatClient,
    *,
    max_lines: int = 8,
    update_fps: float = 8.0,
    show_author: bool = True,
    font_size: float | None = None,
):
    """Render a rolling window of live chat messages inside an existing Text mobject."""
    chat_client.start()
    message_queue = chat_client.subscribe_messages()

    period = 0.0 if update_fps <= 0 else (1.0 / update_fps)
    elapsed = 0.0
    lines: list[str] = []

    if font_size is None:
        font_size = float(getattr(text_mobject, "font_size", 24))

    def _update_chat_text(dt: float) -> None:
        nonlocal elapsed
        elapsed += dt
        if elapsed < period:
            return
        elapsed = 0.0

        changed = False
        while True:
            try:
                message = message_queue.get_nowait()
            except queue.Empty:
                break

            line = f"{message.author}: {message.text}" if show_author else message.text
            lines.append(line)
            if len(lines) > max(1, int(max_lines)):
                lines.pop(0)
            changed = True

        if not changed:
            return

        rendered_text = "\n".join(lines)
        new_text = Text(rendered_text, font_size=font_size)
        new_text.match_style(text_mobject)
        new_text.move_to(text_mobject)
        text_mobject.become(new_text)

    scene.add_updater(_update_chat_text)
    return _update_chat_text


def _escape_markdown(text: str) -> str:
    escaped = text
    for token in ("*", "_", "`", "[", "]", "(", ")"):
        escaped = escaped.replace(token, f"\\{token}")
    return escaped


def _default_avatar_array(size_px: int = 96) -> np.ndarray:
    # Neutral placeholder avatar used when YouTube avatar URL is missing.
    return np.full((size_px, size_px, 3), 120, dtype=np.uint8)


def _build_chat_row(
    message: YouTubeChatMessage,
    *,
    avatar_height: float,
    text_font_size: float,
    markdown_mobject_config: dict[str, Any] | None,
) -> Group:
    avatar_array = message.avatar_image if message.avatar_image is not None else _default_avatar_array()
    avatar_src = ImageMobject(avatar_array, height=avatar_height)
    avatar_mask = Circle(radius=0.5 * avatar_height)
    avatar_mask.set_fill(color=WHITE, opacity=1)
    avatar_mask.set_stroke(color=WHITE, width=0)
    avatar_src.fix_in_frame()
    avatar_mask.fix_in_frame()

    author = _escape_markdown(message.author)
    body = message.text
    markdown_config = dict(markdown_mobject_config or {})
    markdown_config.setdefault("font_size", int(text_font_size))
    content = MarkdownMobject(f"**{author}**: {body}", **markdown_config)
    content.fix_in_frame()
    result = Group(Group(avatar_src, avatar_mask), content).arrange(np.array([1.0, 0.0, 0.0]), buff=0.1, aligned_edge=np.array([0.0, 1.0, 0.0]))

    return result


def bind_youtube_chat_to_feed(
    scene: Scene,
    box: Mobject,
    chat_client: YouTubeLiveChatClient,
    *,
    max_messages: int = 6,
    update_fps: float = 8.0,
    avatar_height: float = 0.25,
    text_font_size: float = 20,
    line_buff: float = 0.18,
    box_buff: float = 0.2,
    markdown_mobject_config: dict[str, Any] | None = None,
):
    """Render a stacked live chat feed with circular avatars and markdown text.

    Each message row contains: masked circular avatar (left) + markdown text (right).
    All rows are stacked vertically (top to bottom), aligned to the left.
    """
    chat_client.start()
    message_queue = chat_client.subscribe_messages()

    period = 0.0 if update_fps <= 0 else (1.0 / update_fps)
    elapsed = 0.0
    messages: list[YouTubeChatMessage] = []
    active_all_rows: Group | None = None
    active_mask_mobjects: list[MaskMobject] = []

    def _update_chat_feed(dt: float) -> None:
        nonlocal elapsed, active_all_rows
        elapsed += dt
        if elapsed < period:
            return
        elapsed = 0.0

        changed = False
        while True:
            try:
                message = message_queue.get_nowait()
            except queue.Empty:
                break
            messages.append(message)
            if len(messages) > max(1, int(max_messages)):
                messages.pop(0)
            changed = True

        if not changed:
            return

        # Remove old composite
        if active_all_rows is not None:
            scene.remove(active_all_rows, *active_mask_mobjects)
            del active_all_rows  # Allow garbage collection of old rows and their textures.
            active_mask_mobjects.clear()

        # Create message rows: each row is [masked_avatar | markdown_text] arranged horizontally
        message_rows: Group = Group()
        for msg in messages:
            row = _build_chat_row(
                msg,
                avatar_height=avatar_height,
                text_font_size=text_font_size,
                markdown_mobject_config=markdown_mobject_config,
            )
            message_rows.add(row)
        message_rows.arrange(np.array([0.0, -1.0, 0.0]), buff=line_buff, aligned_edge=np.array([-1.0, 0.0, 0.0]))
        message_rows.set_max_width(box.get_width() - 2 * box_buff)
        message_rows.set_max_height(box.get_height() - 2 * box_buff)
        message_rows.move_to(box, aligned_edge=np.array([-1.0, 1.0, 0.0])).shift(box_buff * np.array([1.0, -1.0, 0.0]))
        for row in message_rows:
            src, mask = row[0]
            masked_avatar = MaskMobject(scene, src, mask)
            active_mask_mobjects.append(masked_avatar)
        active_all_rows = Group(*(row[1] for row in message_rows))
        scene.add(active_all_rows, *active_mask_mobjects)

    scene.add_updater(_update_chat_feed)
    return _update_chat_feed
