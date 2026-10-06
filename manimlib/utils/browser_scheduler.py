from __future__ import annotations

import asyncio
import sys

if sys.platform == "emscripten":
    from js import window


async def next_animation_frame() -> float:
    """Yield through Pyodide's browser asyncio loop and return browser time."""
    if sys.platform != "emscripten":
        raise RuntimeError("next_animation_frame() is only available in Pyodide")

    # Do not bridge a Python await through a hand-built JS Promise here.
    # Pyodide already provides an asyncio event loop integrated with the browser;
    # asyncio.sleep() schedules the coroutine back onto that loop reliably while
    # runPythonAsync is suspended.
    await asyncio.sleep(1 / 60)
    return float(window.performance.now()) / 1000.0
