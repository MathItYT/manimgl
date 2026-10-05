"""Browser audio backend using HTMLAudioElement/Web APIs.

This module is imported only under Pyodide. Audio playback is owned by JavaScript,
so Python never spawns native audio processes in the browser.
"""
from __future__ import annotations

import mimetypes
from pathlib import PurePosixPath

from js import document, window


class BrowserAudio:
    def __init__(self) -> None:
        self._players = []
        self._sources: dict[str, str] = {}

    def _url(self, sound_file: str) -> str:
        if sound_file in self._sources:
            return self._sources[sound_file]
        # Browser scenes should expose their assets through the page/server.
        # Resolve relative Manim paths against the document URL.
        url = str(window.URL.new(sound_file, window.location.href))
        self._sources[sound_file] = url
        return url

    def register(self, sound_file: str) -> None:
        if sound_file in self._sources:
            return
        url = self._url(sound_file)
        audio = document.createElement("audio")
        audio.preload = "auto"
        audio.src = url
        audio.load()
        self._players.append((sound_file, audio))

    def _get(self, sound_file: str):
        self.register(sound_file)
        for name, audio in self._players:
            if name == sound_file:
                return audio
        raise RuntimeError(f"Unable to create browser audio player: {sound_file}")

    def play(self, sound_file: str, offset: float = 0.0) -> None:
        audio = self._get(sound_file)
        audio.pause()
        audio.currentTime = max(0.0, float(offset))
        promise = audio.play()
        if promise is not None:
            # Avoid an unhandled JS rejection when autoplay is blocked.
            promise.catch(lambda error: None)

    def play_from(self, scene_time: float, events) -> None:
        self.stop_all()
        for event_time, sound_file, _duration in events:
            offset = float(scene_time) - float(event_time)
            if offset >= 0.0:
                self.play(sound_file, offset)

    def seek(self, scene_time: float, events) -> None:
        self.play_from(scene_time, events)

    def stop_all(self) -> None:
        for _name, audio in self._players:
            audio.pause()
            audio.currentTime = 0.0


browser_audio = BrowserAudio()
