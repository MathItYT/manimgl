from __future__ import annotations

import sys

if sys.platform == "emscripten":
    from js import window


async def next_animation_frame() -> float:
    if sys.platform != "emscripten":
        raise RuntimeError("next_animation_frame() is only available in Pyodide")
    next_frame = getattr(window, "__manimNextFrame", None)
    if next_frame is not None:
        timestamp = await next_frame()
    else:
        promise = window.Promise.new(lambda resolve, reject: window.requestAnimationFrame(resolve))
        timestamp = await promise
    return float(timestamp) / 1000.0
