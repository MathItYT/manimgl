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

    # typst-wasm 1.0 returns a document result whose SVG artifacts live on
    # individual pages: result.pages[0].output.
    try:
        pages = result.pages
    except AttributeError as exc:
        raise RuntimeError("Typst compiler returned no pages") from exc
    if inspect.isawaitable(pages):
        pages = await pages

    try:
        page_count = len(pages)
    except TypeError:
        page_count = 0
    if page_count == 0:
        raise RuntimeError("Typst produced no pages")

    page = pages[0]
    if inspect.isawaitable(page):
        page = await page

    try:
        output = page.output
    except AttributeError as exc:
        raise RuntimeError("Typst first page contains no SVG output") from exc
    if inspect.isawaitable(output):
        output = await output
    if output is None:
        raise RuntimeError("Typst first page produced no SVG output")

    # The API normally returns SVG text. Keep a small compatibility path for
    # Uint8Array-like values so Pyodide can consume either representation.
    if not isinstance(output, str):
        try:
            from js import TextDecoder
            if hasattr(output, "byteLength"):
                output = TextDecoder.new().decode(output)
        except Exception:
            pass

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