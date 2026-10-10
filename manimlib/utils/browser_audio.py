"""Browser audio backend using HTMLAudioElement/Web APIs.

Audio is aligned to the scene clock by pausing whichever side is ahead. Active
audio is never hard-seeked to correct drift; its media clock is allowed to catch
up while scene rendering is held, or audio is paused while the scene catches up.
"""
from __future__ import annotations

from js import document, window


class BrowserAudio:
    SYNC_TOLERANCE = 0.035

    def __init__(self) -> None:
        self._players: dict[str, object] = {}
        self._sources: dict[str, str] = {}
        self._last_scene_time: float | None = None

    def _url(self, sound_file: str) -> str:
        if sound_file in self._sources:
            return self._sources[sound_file]

        # Browser assets live in Pyodide's MEMFS, not at an HTTP URL.
        from manimlib.utils.sounds import get_full_sound_file_path
        from js import Blob, URL, Uint8Array

        path = get_full_sound_file_path(sound_file)
        mime_types = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
        }
        mime = mime_types.get(
            __import__("os").path.splitext(str(path))[1].lower(),
            "application/octet-stream",
        )
        with open(path, "rb") as file:
            data = file.read()
        buffer = Uint8Array.new(len(data))
        buffer.assign(data)
        blob = Blob.new([buffer], {"type": mime})
        url = str(URL.createObjectURL(blob))
        self._sources[sound_file] = url
        return url

    def register(self, sound_file: str) -> None:
        if sound_file in self._players:
            return
        audio = document.createElement("audio")
        audio.preload = "auto"
        audio.src = self._url(sound_file)
        audio.load()
        try:
            attach_for_recording = getattr(
                window, "__manimAttachAudioForRecording", None
            )
            if attach_for_recording is not None:
                attach_for_recording(audio)
        except Exception:
            # Audio must still work if the editor has no recorder integration.
            pass
        self._players[sound_file] = audio

    def _get(self, sound_file: str):
        self.register(sound_file)
        return self._players[sound_file]

    @staticmethod
    def _play_at(audio, offset: float) -> None:
        """Start a newly activated scene sound at its one-time start offset."""
        audio.currentTime = max(0.0, float(offset))
        audio.playbackRate = 1.0
        promise = audio.play()
        if promise is not None:
            promise.catch(lambda error: None)

    @staticmethod
    def _resume(audio) -> None:
        """Resume an audio element without changing its current media position."""
        if bool(audio.ended):
            return
        promise = audio.play()
        if promise is not None:
            promise.catch(lambda error: None)

    @staticmethod
    def _sync_paused(audio) -> bool:
        return bool(getattr(audio, "manimSyncPaused", False))

    @staticmethod
    def _set_sync_paused(audio, value: bool) -> None:
        setattr(audio, "manimSyncPaused", bool(value))

    def _reset(self) -> None:
        for audio in self._players.values():
            audio.pause()
            audio.playbackRate = 1.0
            audio.currentTime = 0.0
            self._set_sync_paused(audio, False)
            setattr(audio, "manimEventKey", "")

    def _start_active(self, scene_time: float, events) -> None:
        for event_index, (event_time, sound_file, _duration) in enumerate(events):
            offset = float(scene_time) - float(event_time)
            if offset < 0.0:
                continue

            audio = self._get(sound_file)
            duration = float(audio.duration)
            if duration == duration and offset >= duration:
                continue
            if bool(audio.ended):
                continue

            # A player is positioned once when its scene event becomes active.
            # Repeated sync calls must not reset currentTime: that used to cut
            # speech and made the sound jump whenever the scene clock drifted.
            event_key = f"{event_index}:{float(event_time):.9f}:{sound_file}"
            active_key = str(getattr(audio, "manimEventKey", ""))
            if active_key != event_key:
                audio.pause()
                setattr(audio, "manimEventKey", event_key)
                self._set_sync_paused(audio, False)
                self._play_at(audio, offset)
            elif self._sync_paused(audio):
                # can_advance_scene() resumes this player when the scene catches
                # up to its media clock. Do not start it here and defeat the gate.
                continue
            elif bool(audio.paused):
                # Resume an unexpectedly paused, still-active element in place.
                self._resume(audio)

    def can_advance_scene(
        self,
        scene_time: float,
        proposed_time: float,
        events,
    ) -> bool:
        """Gate one timeline step so scene and active audio do not run ahead.

        Returns False while the scene is ahead of any active audio event. If
        audio is ahead, that player is paused until the scene catches up. This
        uses media-clock pausing, not Scene.seek/Scene.seek_async or time jumps.
        """
        scene_time = float(scene_time)
        proposed_time = float(proposed_time)
        self._start_active(proposed_time, events)
        scene_is_waiting = False
        tolerance = self.SYNC_TOLERANCE

        for event_time, sound_file, _duration in events:
            event_time = float(event_time)
            if proposed_time < event_time:
                continue

            audio = self._players.get(sound_file)
            if audio is None or bool(audio.ended):
                continue

            try:
                media_duration = float(audio.duration)
                media_time = float(audio.currentTime)
            except (TypeError, ValueError):
                continue

            offset = proposed_time - event_time
            if media_duration == media_duration and offset >= media_duration:
                continue

            drift = media_time - offset
            sync_paused = self._sync_paused(audio)

            if drift > tolerance:
                # Audio is ahead: hold only this audio player while the scene
                # continues, rather than seeking audio backwards.
                if not bool(audio.paused):
                    audio.pause()
                self._set_sync_paused(audio, True)
                continue

            if sync_paused:
                # The scene has caught up. Resume from the exact paused media
                # position; never assign currentTime to correct synchronization.
                self._set_sync_paused(audio, False)
                self._resume(audio)

            if drift < -tolerance:
                # The proposed scene frame would outrun this audio event. Keep
                # the scene clock/frame fixed and let the audio media clock catch up.
                if bool(audio.paused):
                    self._resume(audio)
                scene_is_waiting = True

        return not scene_is_waiting

    def play(self, sound_file: str, offset: float = 0.0) -> None:
        audio = self._get(sound_file)
        audio.pause()
        setattr(audio, "manimEventKey", "manual")
        self._set_sync_paused(audio, False)
        self._play_at(audio, offset)

    def seek(self, scene_time: float, events) -> None:
        """Explicitly reset audio for a user-requested timeline reposition."""
        self._reset()
        self._start_active(float(scene_time), events)
        self._last_scene_time = float(scene_time)

    def sync(self, scene_time: float, events) -> None:
        """Activate scene sounds without hard-resynchronizing active media."""
        scene_time = float(scene_time)
        if self._last_scene_time is not None and scene_time < self._last_scene_time:
            # A backwards timeline reset is explicit; no active sound may continue
            # from the old timeline position. The playback loop resets separately.
            self.stop_all()
        self._start_active(scene_time, events)
        self._last_scene_time = scene_time

    def stop_all(self) -> None:
        self._reset()
        self._last_scene_time = None


browser_audio = BrowserAudio()
