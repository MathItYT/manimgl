from __future__ import annotations

import asyncio
import base64
import time
import threading
from collections import deque
from queue import Empty
from queue import Full
from queue import Queue
from urllib.parse import urlencode

from manimlib.logger import log
from manimlib.mobject.svg.text_mobject import Text

try:
    import orjson as json_parser
except ImportError:
    import json as json_parser


class ElevenLabsRealtimeTranscriber:
    """Minimal ElevenLabs realtime STT client powered by websockets.

    Designed to receive PCM microphone chunks from SceneFileWriter callback.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model_id: str = "scribe_v2_realtime",
        sample_rate: int = 16000,
        audio_format: str = "pcm_16000",
        language_code: str | None = None,
        include_timestamps: bool = False,
        include_language_detection: bool = False,
        commit_strategy: str = "vad",
        vad_silence_threshold_secs: float = 1.5,
        vad_threshold: float = 0.4,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 100,
        reconnect_delay_seconds: float = 1.0,
        max_audio_queue_chunks: int = 24,
        chunks_per_enqueue: int = 2,
        reconnect_log_interval_seconds: float = 5.0,
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.language_code = language_code
        self.include_timestamps = include_timestamps
        self.include_language_detection = include_language_detection
        self.commit_strategy = commit_strategy
        self.vad_silence_threshold_secs = vad_silence_threshold_secs
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.max_audio_queue_chunks = max_audio_queue_chunks
        self.chunks_per_enqueue = max(1, chunks_per_enqueue)
        self.reconnect_log_interval_seconds = reconnect_log_interval_seconds

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._connected = threading.Event()
        self._audio_queue: asyncio.Queue[bytes] | None = None
        self._text_queue: Queue[tuple[str, str]] = Queue()
        self._pending_chunks: deque[bytes] = deque()
        self._last_reconnect_log_time: float = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ElevenLabsRealtimeTranscriber",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._connected.clear()
        self._pending_chunks.clear()
        
        # Enviar señal de fin (sentinel) para desbloquear el _sender de inmediato
        if self._loop is not None and self._audio_queue is not None:
            self._loop.call_soon_threadsafe(
                lambda: self._audio_queue.put_nowait(b"") if not self._audio_queue.full() else None
            )
            self._loop.call_soon_threadsafe(lambda: None)
            
        if self._thread is not None:
            self._thread.join(timeout=3)

    def on_mic_chunk(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: dict,
        status: int,
    ) -> None:
        del frame_count, time_info, status
        if not self._running.is_set():
            return
        if self._loop is None or self._audio_queue is None:
            return

        self._pending_chunks.append(in_data)
        if len(self._pending_chunks) < self.chunks_per_enqueue:
            return

        merged_chunk = b"".join(self._pending_chunks)
        self._pending_chunks.clear()
        self._loop.call_soon_threadsafe(self._enqueue_audio_chunk, merged_chunk)

    def _enqueue_audio_chunk(self, in_data: bytes) -> None:
        if self._audio_queue is None:
            return
        if self._audio_queue.full():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._audio_queue.put_nowait(in_data)
        except asyncio.QueueFull:
            pass

    def poll_latest_text(self) -> str | None:
        latest: str | None = None
        while True:
            try:
                _event_type, text = self._text_queue.get_nowait()
                latest = text
            except Empty:
                return latest

    def poll_latest_text_event(self) -> tuple[str, str] | None:
        latest: tuple[str, str] | None = None
        while True:
            try:
                latest = self._text_queue.get_nowait()
            except Empty:
                return latest

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_forever())
        except Exception:
            log.exception("Realtime transcription loop failed")
        finally:
            self._connected.clear()
            self._loop.close()
            self._loop = None

    def _build_uri(self) -> str:
        params = {
            "model_id": self.model_id,
            "audio_format": self.audio_format,
            "include_timestamps": str(self.include_timestamps).lower(),
            "include_language_detection": str(self.include_language_detection).lower(),
            "commit_strategy": self.commit_strategy,
        }
        if self.language_code:
            params["language_code"] = self.language_code
        if self.commit_strategy == "vad":
            params["vad_silence_threshold_secs"] = str(self.vad_silence_threshold_secs)
            params["vad_threshold"] = str(self.vad_threshold)
            params["min_speech_duration_ms"] = str(self.min_speech_duration_ms)
            params["min_silence_duration_ms"] = str(self.min_silence_duration_ms)
        return f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{urlencode(params)}"

    async def _run_forever(self) -> None:
        try:
            from websockets.asyncio.client import connect  # type: ignore[import-not-found]
        except Exception as exc:
            try:
                from websockets.client import connect  # type: ignore[import-not-found]
            except Exception:
                raise RuntimeError(
                    "websockets package is required. Install with: pip install manimgl[transcription]"
                ) from exc

        max_chunks = self.max_audio_queue_chunks if self.max_audio_queue_chunks > 0 else 24
        self._audio_queue = asyncio.Queue(maxsize=max_chunks)
        headers = {"xi-api-key": self.api_key}

        while self._running.is_set():
            try:
                async with connect(
                    self._build_uri(),
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as ws:
                    self._connected.set()
                    send_task = asyncio.create_task(self._sender(ws))
                    recv_task = asyncio.create_task(self._receiver(ws))
                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        exc = task.exception()
                        if exc is not None:
                            raise exc
            except Exception as exc:
                self._connected.clear()
                if not self._running.is_set():
                    break
                now = time.time()
                if now - self._last_reconnect_log_time >= self.reconnect_log_interval_seconds:
                    log.warning(f"Realtime transcription disconnected, retrying: {exc}")
                    self._last_reconnect_log_time = now
                await asyncio.sleep(self.reconnect_delay_seconds)

    async def _sender(self, ws) -> None:
        # Optimización: Pre-construimos la base del payload para no recrear el diccionario
        payload_template = {
            "message_type": "input_audio_chunk",
            "sample_rate": self.sample_rate,
        }
        
        while self._running.is_set():
            if self._audio_queue is None:
                await asyncio.sleep(0.05)
                continue
            
            try:
                # Nos ahorramos el bucle de timeout. Es bloqueante hasta que haya un chunk o se cancele
                audio_chunk = await self._audio_queue.get()
                
                # Sentinel para salir limpiamente
                if not audio_chunk:
                    break

                # Mutamos el diccionario existente (más rápido)
                payload_template["audio_base_64"] = base64.b64encode(audio_chunk).decode("ascii")
                
                payload = json_parser.dumps(payload_template)
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8")
                await ws.send(payload)
            except asyncio.CancelledError:
                break

    async def _receiver(self, ws) -> None:
        async for message in ws:
            try:
                event = json_parser.loads(message)
            except Exception:
                continue

            event_type = event.get("message_type")
            # Agrupamos las validaciones
            if event_type in ("partial_transcript", "committed_transcript", "committed_transcript_with_timestamps"):
                text = event.get("text", "").strip()
                if text:
                    event_kind = "partial" if event_type == "partial_transcript" else "committed"
                    self._text_queue.put((event_kind, text))
            elif event_type and event_type.endswith("error"):
                log.error(f"ElevenLabs realtime error: {event}")


def bind_transcriber_to_text(
    scene,
    text_mobject: Text,
    transcriber: ElevenLabsRealtimeTranscriber,
    update_fps: float = 8.0,
    partial_update_fps: float = 3.0,
    render_partial: bool = True,
    build_text_off_main_thread: bool = False,
    **text_kwargs,
):
    """Attach a scene updater that refreshes Text from latest transcript."""
    transcriber.start()

    committed_period = 0.0 if update_fps <= 0 else (1.0 / update_fps)
    partial_period = 0.0 if partial_update_fps <= 0 else (1.0 / partial_update_fps)
    elapsed_committed = 0.0
    elapsed_partial = 0.0
    pending_committed: str | None = None
    pending_partial: str | None = None
    last_applied_text: str | None = None

    build_request_lock = threading.Lock()
    build_request_text: str | None = None
    built_queue: Queue[tuple[str, Text]] = Queue(maxsize=1)

    def _enqueue_built(text: str, mobject: Text) -> None:
        try:
            built_queue.put_nowait((text, mobject))
        except Full:
            try:
                built_queue.get_nowait()
            except Empty:
                pass
            try:
                built_queue.put_nowait((text, mobject))
            except Full:
                pass

    if build_text_off_main_thread:
        def _builder_worker() -> None:
            worker_last_text: str | None = None
            while transcriber._running.is_set():
                with build_request_lock:
                    target = build_request_text
                if not target or target == worker_last_text:
                    time.sleep(0.01)
                    continue
                try:
                    built = Text(target, **text_kwargs)
                    _enqueue_built(target, built)
                    worker_last_text = target
                except Exception:
                    log.exception("Failed to build transcript text off main thread")

        threading.Thread(
            target=_builder_worker,
            daemon=True,
            name="TranscriptTextBuilder",
        ).start()

    def _update_transcript(dt: float) -> None:
        nonlocal elapsed_committed, elapsed_partial
        nonlocal pending_committed, pending_partial, last_applied_text
        nonlocal build_request_text

        elapsed_committed += dt
        elapsed_partial += dt

        while True:
            event = transcriber.poll_latest_text_event()
            if event is None:
                break
            event_kind, text = event
            if event_kind == "committed":
                pending_committed = text
            elif render_partial:
                pending_partial = text

        target_text: str | None = None
        if pending_committed and elapsed_committed >= committed_period:
            target_text = pending_committed
            pending_committed = None
            elapsed_committed = 0.0
            elapsed_partial = 0.0
        elif pending_partial and elapsed_partial >= partial_period:
            target_text = pending_partial
            pending_partial = None
            elapsed_partial = 0.0

        if not target_text or target_text == last_applied_text:
            target_text = None

        if target_text and build_text_off_main_thread:
            with build_request_lock:
                build_request_text = target_text
        elif target_text:
            try:
                replacement = Text(target_text, **text_kwargs)
                replacement.move_to(text_mobject)
                text_mobject.become(replacement)
                last_applied_text = target_text
            except Exception:
                log.exception("Failed to render transcript text")

        if build_text_off_main_thread:
            latest_built: tuple[str, Text] | None = None
            while True:
                try:
                    latest_built = built_queue.get_nowait()
                except Empty:
                    break
            if latest_built is not None:
                built_text, built_mobject = latest_built
                if built_text != last_applied_text:
                    built_mobject.move_to(text_mobject)
                    text_mobject.become(built_mobject)
                    last_applied_text = built_text

    scene.add_updater(_update_transcript)