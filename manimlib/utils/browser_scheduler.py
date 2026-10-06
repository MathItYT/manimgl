from __future__ import annotations

import sys

if sys.platform == "emscripten":
    from js import window


async def next_animation_frame() -> float:
    """Yield to the browser event loop and return a monotonic browser timestamp.

    requestAnimationFrame can remain unresolved while a Pyodide execution is
    waiting inside runPythonAsync on some browser/runtime combinations.  The
    editor only needs a browser-clock yield here, so use setTimeout as the
    scheduler primitive.  It reliably returns control to the JS event loop
    without depending on a painted frame.
    """
    if sys.platform != "emscripten":
        raise RuntimeError("next_animation_frame() is only available in Pyodide")

    promise = window.Promise.new(
        lambda resolve, reject: window.setTimeout(resolve, 16)
    )
    await promise
    return float(window.performance.now()) / 1000.0
