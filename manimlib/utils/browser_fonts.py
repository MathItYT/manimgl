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

        from js import Uint8Array, window

        if not hasattr(window, "manimRegisterPangoFont"):
            raise RuntimeError("PangoCairo browser font API is not initialized")

        if hasattr(file_or_buffer, "to_py"):
            data = file_or_buffer.to_py()
        else:
            data = file_or_buffer

        result = await window.manimRegisterPangoFont(data, file_name)
        font = BrowserFont(
            name=str(result.name),
            family=str(result.family),
            path=str(result.path),
        )

        compiler = getattr(window, "manimTypstCompiler", None)
        if compiler is None:
            raise RuntimeError("Typst browser compiler is not initialized")

        buffer = Uint8Array.new(len(data))
        buffer.assign(data)
        await compiler.addFonts(buffer)

        self._fonts[font.family] = font
        return font

    async def register_path(self, path: str) -> BrowserFont:
        if sys.platform != "emscripten":
            raise RuntimeError("BrowserFontManager is only available in Pyodide")

        with open(path, "rb") as file:
            data = file.read()

        return await self.register(data, path.rsplit("/", 1)[-1])

    def get(self, family: str) -> BrowserFont | None:
        return self._fonts.get(family)

    def families(self) -> tuple[str, ...]:
        return tuple(self._fonts)


browser_font_manager = BrowserFontManager()


async def register_browser_font(file_or_buffer, file_name: str = "browser-font.ttf") -> BrowserFont:
    return await browser_font_manager.register(file_or_buffer, file_name)


async def register_browser_font_path(path: str) -> BrowserFont:
    return await browser_font_manager.register_path(path)
