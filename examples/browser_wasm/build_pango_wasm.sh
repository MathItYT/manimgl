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

    # Meson may leave behind a build.dat written by a different Meson
    # version. Capture the complete output so we can identify that specific
    # cache error and recover without deleting the whole dependency checkout.
    BUILD_LOG="$PWD/.manim_build_pango.log"

    if bash "$LOCAL_BUILD" 2>&1 | tee "$BUILD_LOG"; then
        rm -f "$LOCAL_BUILD" "$BUILD_LOG"
    else
        build_status=${PIPESTATUS[0]}
        STALE_BUILD_FILE="$(awk -F"'" '/ERROR: Build data file/ && /references functions or classes that don.t exist/ { print $2; exit }' "$BUILD_LOG")"
        STALE_BUILD_DIR="${STALE_BUILD_FILE%/meson-private/build.dat}"

        case "$STALE_BUILD_DIR" in
            "$PANGO_CAIRO_WASM_DIR"/*)
                if [[ -n "$STALE_BUILD_FILE" && -f "$STALE_BUILD_FILE" ]]; then
                    echo "Detected stale Meson metadata: $STALE_BUILD_FILE" >&2
                    echo "Removing only this configured build directory and retrying..." >&2
                    rm -rf -- "$STALE_BUILD_DIR"

                    if bash "$LOCAL_BUILD" 2>&1 | tee "$BUILD_LOG"; then
                        rm -f "$LOCAL_BUILD" "$BUILD_LOG"
                    else
                        build_status=${PIPESTATUS[0]}
                        rm -f "$LOCAL_BUILD"
                        echo "PangoCairo build failed again after Meson cache recovery; see $BUILD_LOG." >&2
                        exit "$build_status"
                    fi
                else
                    rm -f "$LOCAL_BUILD"
                    echo "PangoCairo build failed; see $BUILD_LOG." >&2
                    exit "$build_status"
                fi
                ;;
            *)
                rm -f "$LOCAL_BUILD"
                echo "PangoCairo build failed; see $BUILD_LOG." >&2
                exit "$build_status"
                ;;
        esac
    fi
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
    -s EXPORTED_FUNCTIONS='["_manim_pango_text_to_svg","_manim_pango_register_font","_manim_pango_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall","UTF8ToString","FS"]' \
    --preload-file "$FONT_ASSETS/etc/fonts@/etc/fonts" \
    --preload-file "$FONT_ASSETS/fonts@/fonts" \
    -o "$OUT/pango_text.js"

cat > "$OUT/pango_text_loader.js" <<'EOF'
import createPangoModule from "./pango_text.js";

let modulePromise;
let pangoWorker;
let pangoWorkerReady;
let nextWorkerRequestId = 1;
const workerRequests = new Map();

// Pango's synchronous ccall must not run on Pyodide's browser/UI thread.
// Keep the synchronous API for legacy callers, but route Text.create through
// this worker-backed Promise API.
function getPangoWorker() {
  if (pangoWorkerReady) return pangoWorkerReady;

  pangoWorker = new Worker(
    new URL("./pango_text_worker.js", import.meta.url),
    { type: "module" },
  );

  const worker = pangoWorker;
  pangoWorkerReady = new Promise((resolve, reject) => {
    let ready = false;
    let settled = false;

    const fail = (error) => {
      for (const request of workerRequests.values()) {
        request.reject(error);
      }
      workerRequests.clear();

      if (!settled) {
        settled = true;
        reject(error);
      } else if (pangoWorker === worker) {
        pangoWorker = null;
        pangoWorkerReady = null;
      }
    };

    worker.addEventListener("message", (event) => {
      const message = event.data || {};

      if (message.type === "ready") {
        ready = true;
        if (!settled) {
          settled = true;
          resolve(worker);
        }
        return;
      }

      if (message.type === "fatal") {
        fail(new Error(message.error || "Pango worker initialization failed"));
        return;
      }

      if (message.id == null) return;
      const request = workerRequests.get(message.id);
      if (!request) return;
      workerRequests.delete(message.id);

      if (message.ok) {
        request.resolve(message.value);
      } else {
        request.reject(new Error(message.error || "Pango worker request failed"));
      }
    });

    worker.addEventListener("error", (event) => {
      fail(new Error(event.message || "Pango worker crashed"));
    });

    worker.postMessage({ type: "initialize" });
  });

  return pangoWorkerReady;
}

async function requestPangoWorker(type, payload = {}, transfer = []) {
  const worker = await getPangoWorker();

  return new Promise((resolve, reject) => {
    const id = nextWorkerRequestId++;
    workerRequests.set(id, { resolve, reject });
    try {
      worker.postMessage({ id, type, ...payload }, transfer);
    } catch (error) {
      workerRequests.delete(id);
      reject(error);
    }
  });
}

export async function initializePangoText() {
  if (!modulePromise) {
    modulePromise = createPangoModule({
      locateFile: (file) => new URL(file, import.meta.url).href,
    });
  }

  const Module = await modulePromise;
  const fontCounter = { value: 0 };

  async function registerFont(fileOrBuffer, fileName = "browser-font.ttf") {
    let bytes;
    if (fileOrBuffer instanceof ArrayBuffer) {
      bytes = new Uint8Array(fileOrBuffer);
    } else if (ArrayBuffer.isView(fileOrBuffer)) {
      bytes = new Uint8Array(
        fileOrBuffer.buffer,
        fileOrBuffer.byteOffset,
        fileOrBuffer.byteLength,
      );
    } else if (fileOrBuffer && typeof fileOrBuffer.arrayBuffer === "function") {
      bytes = new Uint8Array(await fileOrBuffer.arrayBuffer());
      fileName = fileOrBuffer.name || fileName;
    } else {
      throw new TypeError("Expected a File, ArrayBuffer, or TypedArray");
    }

    const safeName = String(fileName).replace(/[^A-Za-z0-9._-]/g, "_");
    const path = `/fonts/${++fontCounter.value}-${safeName}`;

    // Register the font in the worker's separate Emscripten filesystem too.
    // Transfer a copy so the byte array remains available for the legacy API.
    const workerBytes = bytes.slice();
    await requestPangoWorker(
      "registerFont",
      { bytes: workerBytes.buffer, fileName },
      [workerBytes.buffer],
    );

    Module.FS.mkdirTree("/fonts");
    Module.FS.writeFile(path, bytes);
    const ptr = Module.ccall(
      "manim_pango_register_font",
      "number",
      ["string"],
      [path],
    );
    if (!ptr) {
      try { Module.FS.unlink(path); } catch (_) {}
      throw new Error(`PangoCairo could not register font: ${fileName}`);
    }

    try {
      return { name: fileName, family: Module.UTF8ToString(ptr), path };
    } finally {
      Module.ccall("manim_pango_free", null, ["number"], [ptr]);
    }
  }

  function textToSvg(markup, justify, indent, alignment, width) {
    const ptr = Module.ccall(
      "manim_pango_text_to_svg",
      "number",
      ["string", "number", "number", "number", "number"],
      [markup, justify ? 1 : 0, indent, alignment, width],
    );
    if (!ptr) throw new Error("PangoCairo failed to render text");
    try {
      return Module.UTF8ToString(ptr);
    } finally {
      Module.ccall("manim_pango_free", null, ["number"], [ptr]);
    }
  }

  function textToSvgAsync(markup, justify, indent, alignment, width) {
    return requestPangoWorker("render", {
      markup,
      justify: Boolean(justify),
      indent: Number(indent),
      alignment: Number(alignment),
      width: Number(width),
    });
  }

  window.PangoTextWasm = {
    textToSvg,
    textToSvgAsync,
    registerFont,
  };

  window.manimPangoTextToSvg = textToSvg;
  window.manimPangoTextToSvgAsync = textToSvgAsync;
  window.manimRegisterPangoFont = registerFont;
  return window.PangoTextWasm;
}

window.manimInitializePangoText = initializePangoText;
EOF

echo "PangoCairo browser WASM built successfully:"
echo "  $OUT/pango_text.js"
echo "  $OUT/pango_text_loader.js"
