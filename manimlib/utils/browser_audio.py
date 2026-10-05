"""Browser audio backend using HTMLAudioElement/Web APIs.

This module is imported only under Pyodide. Audio playback is owned by JavaScript,
so Python never spawns native audio processes in the browser.
"""
from __future__ import annotations

from js import document, window


class BrowserAudio:
    def __init__(self) -> None:
        self._players: dict[str, object] = {}
        self._sources: dict[str, str] = {}
        self._last_scene_time: float | None = None

    def _url(self, sound_file: str) -> str:
        if sound_file in self._sources:
            return self._sources[sound_file]
        url = str(window.URL.new(sound_file, window.location.href))
        self._sources[sound_file] = url
        return url

    def register(self, sound_file: str) -> None:
        if sound_file in self._players:
            return
        audio = document.createElement("audio")
        audio.preload = "auto"
        audio.src = self._url(sound_file)
        audio.load()
        self._players[sound_file] = audio

    def _get(self, sound_file: str):
        self.register(sound_file)
        return self._players[sound_file]

    @staticmethod
    def _play(audio, offset: float) -> None:
        audio.currentTime = max(0.0, float(offset))
        promise = audio.play()
        if promise is not None:
            promise.catch(lambda error: None)

    def _reset(self) -> None:
        for audio in self._players.values():
            audio.pause()
            audio.currentTime = 0.0

    def _start_active(self, scene_time: float, events) -> None:
        for event_time, sound_file, _duration in events:
            offset = float(scene_time) - float(event_time)
            if offset < 0.0:
                continue
            audio = self._get(sound_file)
            # Do not restart an already-playing element on every animation frame.
            if not bool(audio.paused):
                continue
            self._play(audio, offset)

    def play(self, sound_file: str, offset: float = 0.0) -> None:
        audio = self._get(sound_file)
        audio.pause()
        self._play(audio, offset)

    def seek(self, scene_time: float, events) -> None:
        """Hard-resynchronize all audio after a timeline seek."""
        self._reset()
        self._start_active(float(scene_time), events)
        self._last_scene_time = float(scene_time)

    def sync(self, scene_time: float, events) -> None:
        """Advance audio with the timeline without restarting active sounds."""
        scene_time = float(scene_time)

        # A discontinuity means the editor jumped/scrubbed rather than advancing
        # normally. In that case a hard seek is required.
        if (
            self._last_scene_time is None
            or scene_time < self._last_scene_time
            or scene_time - self._last_scene_time > 0.25
        ):
            self.seek(scene_time, events)
            return

        self._start_active(scene_time, events)
        self._last_scene_time = scene_time

    def stop_all(self) -> None:
        self._reset()
        self._last_scene_time = None


browser_audio = BrowserAudio()
