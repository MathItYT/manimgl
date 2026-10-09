#!/usr/bin/env python3
"""Mirror font files into the static site before GitHub Pages deployment."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import tempfile
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = REPO_ROOT / ".github" / "font-sources.json"
OUTPUT_DIR = REPO_ROOT / "examples" / "fonts"
FONT_DIR = OUTPUT_DIR / "files"
OUTPUT_MANIFEST = OUTPUT_DIR / "fonts.json"
TIMEOUT = 45
WORKERS = 16
MAX_RETRIES = 3

VALID_SIGNATURES = (b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1", b"wOFF", b"wOF2")

def is_valid_font_file(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 256:
            return False
        with open(path, "rb") as f:
            header = f.read(4)
        return header in VALID_SIGNATURES
    except Exception:
        return False

def download(entry):
    destination = OUTPUT_DIR / entry["file"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    if is_valid_font_file(destination):
        return {**entry, "size": destination.stat().st_size}, None

    request = urllib.request.Request(
        entry["source"],
        headers={"User-Agent": "MathItYT-ManimGL-font-mirror/1.0"},
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                data = response.read()
            if len(data) < 256:
                raise RuntimeError(f"archivo demasiado pequeño ({len(data)} bytes)")
            # Quick signature checks catch HTML error pages masquerading as fonts.
            if not (data[:4] in VALID_SIGNATURES):
                raise RuntimeError(f"la respuesta no parece ser un archivo de fuente (cabecera: {data[:4]!r})")
            fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".font-")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return {**entry, "size": len(data)}, None
        except Exception as exc:
            last_error = exc
            if destination.exists():
                destination.unlink()
            import time
            time.sleep(attempt * 0.5)

    return None, f'{entry["name"]}: {last_error}'

def main():
    entries = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(download, entry) for entry in entries]
        for future in as_completed(futures):
            ok, error = future.result()
            if ok:
                results.append(ok)
            else:
                failures.append(error)

    successful_files = {item["file"]: item for item in results}
    manifest = [
        {"name": entry["name"], "fileName": entry["name"], "url": "./" + entry["file"]}
        for entry in entries
        if entry["file"] in successful_files
    ]
    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Fuentes disponibles en el servidor: {len(results)}/{len(entries)}")
    if failures:
        print("Fuentes omitidas (no se alojaron correctamente):")
        for failure in sorted(failures):
            print(" - " + failure)
    if not results:
        raise SystemExit("No se pudo descargar ninguna fuente; se cancela el despliegue.")

if __name__ == "__main__":
    main()
