from __future__ import annotations

import sys

if sys.platform == "emscripten":
    from js import window


async def typst_to_svg_async(content: str) -> str:
    if sys.platform != "emscripten":
        from manimlib.utils.typst_file_writing import typst_to_svg
        return typst_to_svg(content)

    try:
        compiler = window.manimTypstCompiler
    except Exception as exc:
        raise RuntimeError("typst-wasm is not initialized. Call await initialize_browser_wasm() first.") from exc

    await compiler.addSource("main.typ", content)
    result = await compiler.compile({"main": "main.typ", "format": "svg"})
    output = result.output
    if output is None:
        raise RuntimeError("Typst produced no SVG output")
    return str(output)


async def initialize_browser_typst() -> None:
    if sys.platform != "emscripten":
        return
    try:
        window.manimTypstCompiler
    except Exception:
        await window.manimInitializeBrowserWasm()