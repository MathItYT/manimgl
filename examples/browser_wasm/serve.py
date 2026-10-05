#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import http.server
import os
from pathlib import Path


class BrowserWasmRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static server with headers required by threaded browser WASM."""

    def end_headers(self) -> None:
        # Development server: never let the browser cache Pyodide/WASM
        # assets or generated wheels. These resources keep the same URL
        # while their contents change during development.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the ManimGL browser WASM example."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory to serve (default: examples/).",
    )
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error(f"directory does not exist: {directory}")

    handler = functools.partial(
        BrowserWasmRequestHandler,
        directory=os.fspath(directory),
    )

    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Serving {directory}")
    print(f"http://{args.host}:{args.port}/pyodide_manim.html")
    print("Cache-Control: no-store")
    print("Cross-Origin-Opener-Policy: same-origin")
    print("Cross-Origin-Embedder-Policy: require-corp")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()