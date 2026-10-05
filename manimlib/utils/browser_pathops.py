from __future__ import annotations

import sys

if sys.platform == "emscripten":
    from js import window


class BrowserPathOps:
    def __init__(self):
        self.pathkit = None

    def set_pathkit(self, pathkit) -> None:
        self.pathkit = pathkit

    def require(self):
        if self.pathkit is None:
            try:
                self.pathkit = window.PathKit
            except Exception:
                pass
        if self.pathkit is None:
            raise RuntimeError(
                "PathKit is not initialized. Load pathkit-wasm and call "
                "await initialize_browser_wasm() before constructing boolean mobjects."
            )
        return self.pathkit

    def combine(self, paths: list[str], operation: str) -> str:
        pk = self.require()
        if not paths:
            return ""
        result = pk.FromSVGString(paths[0])
        try:
            op = getattr(pk.PathOp, operation)
            for source in paths[1:]:
                other = pk.FromSVGString(source)
                try:
                    result.op(other, op)
                finally:
                    other.delete()
            return str(result.toSVGString())
        finally:
            result.delete()


browser_pathops = BrowserPathOps()
