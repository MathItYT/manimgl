# Browser WASM backends

The Pyodide build uses browser-native/WASM replacements for native-only ManimGL dependencies:

- typst-wasm for Typst SVG compilation.
- pathkit-wasm for boolean path operations.
- VitoVan/pango-cairo-wasm for text shaping and SVG generation.

## PathKit and Typst

`browser_wasm.js` initializes both automatically. It loads PathKit from npm and the Typst compiler/engine from typst-wasm.

Typst fonts are loaded from `@typst-wasm/fonts`.

## PangoCairo

PangoCairo is built locally with Emscripten. **Docker is not required.**

The build uses the upstream `VitoVan/pango-cairo-wasm` source tree and its vendored dependencies. The upstream project documents this as the source-build path; its `build.sh` currently targets Fedora, so on another Linux distribution you may need the equivalent native build tools installed.

### 1. Install and activate Emscripten

Install `emsdk` normally, then activate it in the shell used to build:

    source /path/to/emsdk/emsdk_env.sh

Verify:

    emcc --version
    pkg-config --version

Emscripten provides `emconfigure`, `emcmake` and `emmake` for configuring/building native build systems against `emcc`.

### 2. Build PangoCairo locally

The ManimGL script can clone the PangoCairo source automatically:

    ./examples/browser_wasm/build_pango_wasm.sh

It uses:

    .cache/pango-cairo-wasm/

Alternatively, use an existing checkout:

    PANGO_CAIRO_WASM_DIR=/path/to/pango-cairo-wasm \
        ./examples/browser_wasm/build_pango_wasm.sh

The script:

1. verifies local `emcc` and `pkg-config`;
2. clones the PangoCairo repository if necessary;
3. initializes its git submodules;
4. runs its local `build.sh` when `env.sh` does not exist;
5. sources the generated WASM environment;
6. links `pango_text.c` against the generated static Pango/Cairo stack;
7. generates `dist/pango_text.js`;
8. generates `dist/pango_text_loader.js`.

There is no `docker` invocation anywhere in this build path.

### 3. Browser headers

Pango's WASM build uses pthreads. The upstream project explicitly notes that Pango's pthreads require Web Workers and additional HTTP headers.

Therefore the browser application must be served as a cross-origin-isolated page:

    Cross-Origin-Opener-Policy: same-origin
    Cross-Origin-Embedder-Policy: require-corp

A local development server should provide these headers as well.

### 4. Python API

The Python API is asynchronous in Pyodide:

    title = await Text.create("Hello")
    formula = await Typst.create("x^2")

The synchronous native APIs remain unchanged.

## Why construction is asynchronous

Browser WASM compilers are promise-based and cannot block the browser event loop. Consequently browser scenes use:

    await scene.build_async()
    await scene.playback_async()

`build_async()` executes `construct()` and accepts an async `construct()` so WASM-backed text/Typst objects can be awaited. Playback is then driven by the timeline's `seek()` function.


## Monaco Manim editor

The branch also contains `examples/manim_editor.html`, a browser editor backed by Monaco and Pyodide. It accepts native synchronous ManimGL scene code and transforms it to the asynchronous browser API before executing it. Start the development server with:

    python examples/browser_wasm/serve.py

Then open `/manim_editor.html`.
