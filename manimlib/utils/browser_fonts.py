from __future__ import annotations

import json
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class BrowserFont:
    name: str
    family: str
    path: str
    style: str = "Regular"
    weight: int = 400


def _font_metadata(path: str) -> dict[str, object]:
    from fontTools.ttLib import TTFont

    font = TTFont(path)
    names = font["name"].names

    def get_name(name_id: int, fallback: str = "") -> str:
        for record in names:
            if record.nameID != name_id:
                continue
            try:
                return record.toUnicode()
            except Exception:
                try:
                    return record.string.decode(record.getEncoding(), errors="replace")
                except Exception:
                    continue
        return fallback

    family = get_name(1) or get_name(16) or get_name(6) or path.rsplit("/", 1)[-1]
    subfamily = get_name(2) or get_name(17) or "Regular"
    full_name = get_name(4) or f"{family} {subfamily}".strip()

    weight = 400
    if "OS/2" in font:
        weight = int(getattr(font["OS/2"], "usWeightClass", 400) or 400)

    return {
        "name": full_name,
        "family": family,
        "style": subfamily,
        "weight": weight,
        "path": path,
    }


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
        pango_name = str(result.name)
        pango_family = str(result.family)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=file_name.rsplit(".", 1)[-1], delete=False) as tmp:
            tmp.write(bytes(data))
            metadata_path = tmp.name

        try:
            metadata = _font_metadata(metadata_path)
        finally:
            import os
            os.unlink(metadata_path)

        font = BrowserFont(
            name=str(metadata["name"]) or pango_name,
            family=str(metadata["family"]) or pango_family,
            path=str(metadata["path"]),
            style=str(metadata["style"]),
            weight=int(metadata["weight"]),
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

        font = await self.register(data, path.rsplit("/", 1)[-1])
        return BrowserFont(
            name=font.name,
            family=font.family,
            path=path,
            style=font.style,
            weight=font.weight,
        )

    def describe_path(self, path: str) -> dict[str, object]:
        return _font_metadata(path)

    def get(self, family: str) -> BrowserFont | None:
        return self._fonts.get(family)

    def families(self) -> tuple[str, ...]:
        return tuple(self._fonts)


browser_font_manager = BrowserFontManager()


async def register_browser_font(file_or_buffer, file_name: str = "browser-font.ttf") -> BrowserFont:
    return await browser_font_manager.register(file_or_buffer, file_name)


async def register_browser_font_path(path: str) -> BrowserFont:
    return await browser_font_manager.register_path(path)


def describe_browser_font_path(path: str) -> str:
    return json.dumps(browser_font_manager.describe_path(path), ensure_ascii=False)
