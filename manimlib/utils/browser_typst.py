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
        raise RuntimeError(
            "typst-wasm is not initialized. Call await initialize_browser_wasm() first."
        ) from exc

    await compiler.addSource("main.typ", content)
    await compiler.setMain("main.typ")
    result = await compiler.compile({"format": "svg"})
    pages = result.pages
    if not pages or pages.length == 0:
        raise RuntimeError("Typst produced no SVG pages")
    return str(pages[0].output)


async def initialize_browser_typst(
    worker_url: str = "https://cdn.jsdelivr.net/npm/typst-wasm@1.0.0/dist/worker/web-worker.js",
    core_url: str = "https://cdn.jsdelivr.net/npm/typst-wasm@1.0.0/dist/engine/engine.core.wasm",
    core2_url: str = "https://cdn.jsdelivr.net/npm/typst-wasm@1.0.0/dist/engine/engine.core2.wasm",
    core3_url: str = "https://cdn.jsdelivr.net/npm/typst-wasm@1.0.0/dist/engine/engine.core3.wasm",
) -> None:
    if sys.platform != "emscripten":
        return
    try:
        window.manimTypstCompiler
        return
    except Exception:
        pass
    await window.manimInitializeTypst(worker_url, core_url, core2_url, core3_url)
