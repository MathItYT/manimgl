from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class BrowserFont:
    name: str
    family: str
    path: str


class BrowserFontManager:
    def __init__(self):
        self._fonts: dict[str, BrowserFont] = {}

    async def register(self, file_or_buffer, file_name: str = "browser-font.ttf") -> BrowserFont:
        if sys.platform != "emscripten":
            raise RuntimeError("BrowserFontManager is only available in Pyodide")
        from js import window
        if not hasattr(window, "manimRegisterPangoFont"):
            raise RuntimeError("PangoCairo browser font API is not initialized")
        result = await window.manimRegisterPangoFont(file_or_buffer, file_name)
        font = BrowserFont(name=str(result.name), family=str(result.family), path=str(result.path))
        self._fonts[font.family] = font
        return font

    def get(self, family: str) -> BrowserFont | None:
        return self._fonts.get(family)

    def families(self) -> tuple[str, ...]:
        return tuple(self._fonts)


browser_font_manager = BrowserFontManager()


async def register_browser_font(file_or_buffer, file_name: str = "browser-font.ttf") -> BrowserFont:
    return await browser_font_manager.register(file_or_buffer, file_name)