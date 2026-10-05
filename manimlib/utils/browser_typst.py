from __future__ import annotations

import inspect
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

    # The browser bridge may expose either a native JS Promise or a Python
    # awaitable depending on how the compiler was initialized. Resolve both,
    # including a nested awaitable returned by result.output.
    source_result = compiler.addSource("main.typ", content)
    if inspect.isawaitable(source_result):
        await source_result

    result = compiler.compile({"main": "main.typ", "format": "svg"})
    if inspect.isawaitable(result):
        result = await result

    output = result.output
    if inspect.isawaitable(output):
        output = await output
    if output is None:
        raise RuntimeError("Typst produced no SVG output")

    svg = str(output)
    if inspect.iscoroutine(svg):
        raise RuntimeError("Typst compiler returned an unresolved coroutine instead of SVG text")
    return svg


async def initialize_browser_typst() -> None:
    if sys.platform != "emscripten":
        return
    try:
        window.manimTypstCompiler
    except Exception:
        await window.manimInitializeBrowserWasm()