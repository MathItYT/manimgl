#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/examples/browser_wasm/dist/wheels"
CACHE="$ROOT/.cache"
WGPU_DIR="$CACHE/wgpu-py"
WGPU_REF="${WGPU_REF:-feature/pyodide-webgpu-backend}"

die() {
    echo "error: $*" >&2
    exit 1
}

command -v python >/dev/null 2>&1 || die "python was not found"
command -v git >/dev/null 2>&1 || die "git was not found"
python -m pip --version >/dev/null 2>&1 || die "python pip is required"

mkdir -p "$OUT" "$CACHE"

# Give every development wheel a random PEP 440 version. This changes the
# wheel filename on every build and prevents micropip from reusing an older
# wheel downloaded from the same URL.
BUILD_ID="$(python - <<'PY'
import secrets
print(secrets.randbelow(10**18))
PY
)"
MANIM_VERSION="1.7.2.dev${BUILD_ID}"
WGPU_VERSION="0.32.0.dev${BUILD_ID}"

echo "==> Browser build id: $BUILD_ID"
echo "    ManimGL version: $MANIM_VERSION"
echo "    wgpu version:    $WGPU_VERSION"


verify_wheel_version() {
    local wheel="$1"
    local expected="$2"
    WHEEL_PATH="$wheel" EXPECTED_VERSION="$expected" python - <<'PY'
import os
import zipfile

wheel = os.environ["WHEEL_PATH"]
expected = os.environ["EXPECTED_VERSION"]

with zipfile.ZipFile(wheel) as zf:
    metadata = next(
        name for name in zf.namelist()
        if name.endswith(".dist-info/METADATA")
    )
    text = zf.read(metadata).decode("utf-8")

actual = next(
    line.split(":", 1)[1].strip()
    for line in text.splitlines()
    if line.startswith("Version:")
)

if actual != expected:
    raise SystemExit(
        f"wheel version mismatch: expected {expected}, got {actual} ({wheel})"
    )
PY
}

echo "==> Building ManimGL wheel"
rm -rf -- "$ROOT/build" "$ROOT/manimgl.egg-info" "$ROOT/src/manimgl.egg-info"
find "$OUT" -maxdepth 1 -type f -name "manimgl-*.whl" -delete
SETUP_CFG_BACKUP="$CACHE/setup.cfg.pyodide-wheel-backup"
cp "$ROOT/setup.cfg" "$SETUP_CFG_BACKUP"
trap 'cp "$SETUP_CFG_BACKUP" "$ROOT/setup.cfg"; rm -f "$SETUP_CFG_BACKUP"' EXIT
sed -i -E "s/^version = .*/version = $MANIM_VERSION/" "$ROOT/setup.cfg"
python -m pip wheel "$ROOT" --no-deps --no-cache-dir --wheel-dir "$OUT"

MANIM_WHEEL="$(find "$OUT" -maxdepth 1 -type f -name 'manimgl-*.whl' -print -quit)"
[[ -n "$MANIM_WHEEL" ]] || die "ManimGL wheel was not produced"
verify_wheel_version "$MANIM_WHEEL" "$MANIM_VERSION"

echo "==> Preparing wgpu-py checkout"
if [[ ! -d "$WGPU_DIR/.git" ]]; then
    git clone https://github.com/MathItYT/wgpu-py.git "$WGPU_DIR"
fi

git -C "$WGPU_DIR" fetch origin "$WGPU_REF"
git -C "$WGPU_DIR" checkout "$WGPU_REF"
git -C "$WGPU_DIR" reset --hard "origin/$WGPU_REF"

echo "==> Building browser wgpu wheel"
find "$OUT" -maxdepth 1 -type f -name "wgpu-*.whl" -delete
WGPU_VERSION_FILE="$WGPU_DIR/wgpu/_version.py"
WGPU_VERSION_BACKUP="$CACHE/wgpu-version.pyodide-wheel-backup"
cp "$WGPU_VERSION_FILE" "$WGPU_VERSION_BACKUP"
trap 'cp "$WGPU_VERSION_BACKUP" "$WGPU_VERSION_FILE"; rm -f "$WGPU_VERSION_BACKUP"; cp "$SETUP_CFG_BACKUP" "$ROOT/setup.cfg" 2>/dev/null || true; rm -f "$SETUP_CFG_BACKUP"' EXIT
sed -i -E "s/^__version__ = .*/__version__ = \"$WGPU_VERSION\"/" "$WGPU_VERSION_FILE"

(
    cd "$WGPU_DIR"
    rm -rf -- build wgpu.egg-info
    WGPU_PY_BUILD_NOARCH=1 python -m pip wheel . --no-deps --no-cache-dir --wheel-dir "$OUT"
)

WGPU_WHEEL="$(find "$OUT" -maxdepth 1 -type f -name 'wgpu-*.whl' -print -quit)"
[[ -n "$WGPU_WHEEL" ]] || die "wgpu wheel was not produced"
verify_wheel_version "$WGPU_WHEEL" "$WGPU_VERSION"

echo "==> Writing wheel manifest"
MANIM_NAME="$(basename "$MANIM_WHEEL")"
WGPU_NAME="$(basename "$WGPU_WHEEL")"

cat > "$OUT/wheels.json" <<EOF
{
  "manimgl": "$MANIM_NAME",
  "wgpu": "$WGPU_NAME"
}
EOF

echo
echo "Wheels ready:"
echo "  $MANIM_NAME"
echo "  $WGPU_NAME"
echo "  wheels.json"
