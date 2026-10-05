#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/examples/browser_wasm/dist"
PANGO_CAIRO_WASM_DIR="${PANGO_CAIRO_WASM_DIR:-}"

die() {
    echo "error: $*" >&2
    exit 1
}

command -v emcc >/dev/null 2>&1 || die \
    "emcc was not found. Install/activate Emscripten first (source emsdk_env.sh)."

command -v pkg-config >/dev/null 2>&1 || die \
    "pkg-config was not found."

if [[ -z "$PANGO_CAIRO_WASM_DIR" ]]; then
    PANGO_CAIRO_WASM_DIR="$ROOT/.cache/pango-cairo-wasm"
    if [[ ! -d "$PANGO_CAIRO_WASM_DIR/.git" ]]; then
        mkdir -p "$(dirname "$PANGO_CAIRO_WASM_DIR")"
        echo "Cloning VitoVan/pango-cairo-wasm into $PANGO_CAIRO_WASM_DIR..."
        git clone --recurse-submodules \
            https://github.com/VitoVan/pango-cairo-wasm.git \
            "$PANGO_CAIRO_WASM_DIR"
    fi
fi

[[ -d "$PANGO_CAIRO_WASM_DIR/.git" ]] || die \
    "PANGO_CAIRO_WASM_DIR is not a git checkout: $PANGO_CAIRO_WASM_DIR"

# Normalize the checkout path before changing directory. PANGO_CAIRO_WASM_DIR
# is commonly supplied as a relative path from the ManimGL checkout; after
# cd-ing into the checkout, using that relative path again would incorrectly
# resolve it relative to itself.
PANGO_CAIRO_WASM_DIR="$(cd "$PANGO_CAIRO_WASM_DIR" && pwd)"

cd "$PANGO_CAIRO_WASM_DIR"

# VitoVan's env.sh expects magicdir to point at this checkout.
export magicdir="$PWD"

git submodule update --init --recursive

[[ -f env.sh ]] || die \
    "Missing upstream env.sh in $PANGO_CAIRO_WASM_DIR"

# env.sh is an upstream script from an older shell environment. It may invoke
# emsdk and probes unset variables, so temporarily disable nounset.
# shellcheck disable=SC1091
set +u
source ./env.sh
set -u

command -v emcc >/dev/null 2>&1 || die \
    "env.sh did not provide a working emcc."

# env.sh configures pkg-config to look exclusively in the WASM prefix.
# The upstream checkout contains env.sh even before its dependencies have been
# built, so its mere presence is NOT evidence that PangoCairo is installed.
#
# Build the vendored dependency stack when pangocairo.pc is missing. The
# upstream build.sh starts with Fedora-specific dnf commands; we deliberately
# remove only those host-package installation commands and let the local
# machine provide the native build tools. No Docker is used.
if ! pkg-config --exists pangocairo; then
    echo "pangocairo.pc is missing; building PangoCairo and its WASM dependencies..."
    echo "This can take 20+ minutes on the first build."

    LOCAL_BUILD="$PWD/.manim_build_pango.sh"

    # The upstream build script has two initial 'sudo dnf ...' commands.
    # Strip those host-package installation commands while preserving the
    # actual cross-compilation steps and its source ./env.sh.
    grep -vE '^[[:space:]]*sudo[[:space:]]+dnf([[:space:]]|$)' \
        ./build.sh > "$LOCAL_BUILD"
    chmod +x "$LOCAL_BUILD"

    bash "$LOCAL_BUILD"
    rm -f "$LOCAL_BUILD"
fi

# Refresh the check after the dependency build.
if ! pkg-config --exists pangocairo; then
    echo "pkg-config diagnostics:" >&2
    echo "  PKG_CONFIG_PATH=${PKG_CONFIG_PATH:-}" >&2
    echo "  PKG_CONFIG_LIBDIR=${PKG_CONFIG_LIBDIR:-}" >&2
    echo "  EM_PKG_CONFIG_PATH=${EM_PKG_CONFIG_PATH:-}" >&2
    echo "  EM_PKG_CONFIG_LIBDIR=${EM_PKG_CONFIG_LIBDIR:-}" >&2
    find "$PWD/build" -name 'pangocairo.pc' -o -name 'pango.pc' 2>/dev/null || true
    die "pangocairo is still not visible through pkg-config after building dependencies"
fi

echo "Using WASM pangocairo:"
pkg-config --modversion pangocairo
pkg-config --variable=prefix pangocairo

mkdir -p "$OUT"

# Fontconfig cannot use the host's /etc/fonts inside the browser. Bundle a
# small deterministic config together with a Unicode-capable sans font.
FONT_ASSETS="$ROOT/.cache/manim-browser-fonts"
rm -rf "$FONT_ASSETS"
mkdir -p "$FONT_ASSETS/fonts" "$FONT_ASSETS/etc/fonts"

command -v fc-match >/dev/null 2>&1 || die \
    "fc-match was not found; install Fontconfig utilities to bundle browser fonts."

BROWSER_FONT="$(fc-match -f '%{file}' 'sans:style=Regular' | head -n 1)"
[[ -f "$BROWSER_FONT" ]] || die \
    "Browser font was not found: $BROWSER_FONT"

cp "$BROWSER_FONT" "$FONT_ASSETS/fonts/"
FONT_FAMILY="$(fc-match -f '%{family}' 'sans:style=Regular' | head -n 1)"
[[ -n "$FONT_FAMILY" ]] || FONT_FAMILY="sans"

cat > "$FONT_ASSETS/etc/fonts/fonts.conf" <<EOF
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>/usr/share/fonts</dir>
  <dir>/usr/local/share/fonts</dir>
  <dir>/fonts</dir>
  <cachedir>/tmp/fontconfig</cachedir>
  <alias>
    <family>sans</family>
    <prefer><family>$FONT_FAMILY</family></prefer>
  </alias>
  <alias>
    <family>sans-serif</family>
    <prefer><family>$FONT_FAMILY</family></prefer>
  </alias>
</fontconfig>
EOF

PANGOCAIRO_FLAGS="$(
    pkg-config --libs --cflags \
        glib-2.0,gobject-2.0,cairo,pixman-1,freetype2,fontconfig,expat,harfbuzz,pangocairo
)"

# Pango's browser build uses pthreads. This requires a cross-origin-isolated
# page (COOP/COEP). The generated module is intentionally built with pthread
# support rather than pretending Pango is single-threaded.
emcc \
    $PANGOCAIRO_FLAGS \
    "$ROOT/examples/browser_wasm/pango_text.c" \
    -O3 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s EXPORT_NAME=createPangoModule \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE=4 \
    -s ASYNCIFY \
    -s EMULATE_FUNCTION_POINTER_CASTS=1 \
    -s EMULATE_FUNCTION_POINTER_CASTS=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS='["_manim_pango_text_to_svg","_manim_pango_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall","UTF8ToString"]' \
    -o "$OUT/pango_text.js"

cat > "$OUT/pango_text_loader.js" <<'EOF'
import createPangoModule from "./pango_text.js";

let modulePromise;

export async function initializePangoText() {
  if (!modulePromise) {
    modulePromise = createPangoModule({
      locateFile: (file) => new URL(file, import.meta.url).href,
    });
  }

  const Module = await modulePromise;

  window.PangoTextWasm = {
    textToSvg(markup, justify, indent, alignment, width) {
      const ptr = Module.ccall(
        "manim_pango_text_to_svg",
        "number",
        ["string", "number", "number", "number", "number"],
        [markup, justify ? 1 : 0, indent, alignment, width],
      );
      if (!ptr) throw new Error("PangoCairo failed to render text");
      const svg = Module.UTF8ToString(ptr);
      Module.ccall("manim_pango_free", null, ["number"], [ptr]);
      return svg;
    },
  };

  window.manimPangoTextToSvg = window.PangoTextWasm.textToSvg;
  return window.PangoTextWasm;
}

window.manimInitializePangoText = initializePangoText;
EOF

echo "PangoCairo browser WASM built successfully:"
echo "  $OUT/pango_text.js"
echo "  $OUT/pango_text_loader.js"
