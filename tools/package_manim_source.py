#!/usr/bin/env python3
"""Package the real ManimGL Python source tree for the browser Pyright LSP."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "manimlib"
OUTPUT = ROOT / "examples" / "vendor" / "manimlib-source.zip"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".pyi"}:
            continue
        if "__pycache__" in path.parts:
            continue
        archive.write(path, path.relative_to(PACKAGE).as_posix())
print(f"Packaged real ManimGL source into {OUTPUT}")
