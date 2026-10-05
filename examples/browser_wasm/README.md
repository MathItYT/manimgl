# Browser WASM backends

The Pyodide build uses browser-native/WASM replacements for native-only ManimGL dependencies:

- typst-wasm for Typst SVG compilation.
- pathkit-wasm for boolean path operations.
- VitoVan/pango-cairo-wasm for text shaping and SVG generation.

## PathKit and Typst

browser_wasm.js initializes both automatically. It loads PathKit from npm and the Typst compiler/engine from typst-wasm.

Typst fonts are loaded from @typst-wasm/fonts.

## PangoCairo

PangoCairo is intentionally built from source rather than committed as a binary.

Clone https://github.com/VitoVan/pango-cairo-wasm, then run:

    PANGO_CAIRO_WASM_DIR=/path/to/pango-cairo-wasm ./examples/browser_wasm/build_pango_wasm.sh

The generated files are examples/browser_wasm/dist/pango_text.js and examples/browser_wasm/dist/pango_text_loader.js.

The Python API is asynchronous in Pyodide:

    title = await Text.create("Hello")
    formula = await Typst.create("x^2")

The synchronous native APIs remain unchanged.

## Why construction is asynchronous

Browser WASM compilers are promise-based and cannot block the browser event loop. Consequently browser scenes have:

    await scene.build_async()
    await scene.playback_async()

build_async() executes construct() and accepts an async construct() so WASM-backed text/Typst objects can be awaited. Playback is then driven by the timeline's seek() function.