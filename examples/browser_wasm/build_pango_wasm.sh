#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/examples/browser_wasm/dist"
PANGO_CAIRO_WASM_DIR="${PANGO_CAIRO_WASM_DIR:-}"

if [[ -z "$PANGO_CAIRO_WASM_DIR" ]]; then
    echo "Set PANGO_CAIRO_WASM_DIR to a checkout of VitoVan/pango-cairo-wasm." >&2
    exit 1
fi

mkdir -p "$OUT"

docker run --rm \
    -v "$ROOT:/app" \
    -v "$PANGO_CAIRO_WASM_DIR:/pango-cairo-wasm" \
    -w /app \
    vitovan/pango-cairo-wasm \
    bash -lc '
        if [ -f /pango-cairo-wasm/env.sh ]; then . /pango-cairo-wasm/env.sh; fi
        export PKG_CONFIG_PATH=/pango-cairo-wasm:$PKG_CONFIG_PATH
        export PANGOCAIRO_FLAGS="$(pkg-config --libs --cflags glib-2.0,gobject-2.0,cairo,pixman-1,freetype2,fontconfig,expat,harfbuzz,pangocairo) -s USE_PTHREADS=0 -s ASYNCIFY"
        emcc $PANGOCAIRO_FLAGS examples/browser_wasm/pango_text.c -O3 -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORT_NAME=createPangoModule -s EXPORTED_FUNCTIONS='["_manim_pango_text_to_svg","_manim_pango_free"]' -s EXPORTED_RUNTIME_METHODS='["ccall","UTF8ToString"]' -o examples/browser_wasm/dist/pango_text.js
    '

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
