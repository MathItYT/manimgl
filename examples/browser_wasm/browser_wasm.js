export async function initializeBrowserWasm(options = {}) {
  const pathkitScript = options.pathkitScript ||
    "https://cdn.jsdelivr.net/npm/pathkit-wasm@1.0.0/bin/pathkit.js";

  if (!window.PathKit) {
    await new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = pathkitScript;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  if (!window.PathKit) {
    const pathkit = await window.PathKitInit({
      locateFile: (file) => new URL(
        "https://cdn.jsdelivr.net/npm/pathkit-wasm@1.0.0/bin/" + file
      ).href,
    });
    window.PathKit = pathkit;
  }

  const typstModule = await import(
    options.typstModule || "https://cdn.jsdelivr.net/npm/typst-wasm@1.0.0/+esm"
  );
  const typstWorkerModule = await import(
    options.typstWorkerModule || "https://cdn.jsdelivr.net/npm/typst-wasm@1.0.0/dist/worker/browser.js"
  );

  if (!window.manimTypstCompiler) {
    const typstCdn = options.typstCdn || "https://cdn.jsdelivr.net/npm/typst-wasm@1.0.0/dist";
    const workerEntry = options.typstWorkerEntry || typstCdn + "/worker/web-worker.js";
    const workerUrl = options.typstWorkerUrl || URL.createObjectURL(
      new Blob([`import ${JSON.stringify(workerEntry)};`], {
        type: "text/javascript",
      })
    );

    const coreBase = options.typstCoreBase || typstCdn + "/engine/";
    const coreModules = {
      "engine.core.wasm": WebAssembly.compileStreaming(fetch(coreBase + "engine.core.wasm")),
      "engine.core2.wasm": WebAssembly.compileStreaming(fetch(coreBase + "engine.core2.wasm")),
      "engine.core3.wasm": WebAssembly.compileStreaming(fetch(coreBase + "engine.core3.wasm")),
    };

    window.manimTypstWorkerUrl = workerUrl;
    window.manimTypstCompiler = await typstModule.createTypstCompiler({
      backend: "auto",
      worker: () => typstWorkerModule.createWebWorker(workerUrl),
      coreModules,
    });

    const fontBase = options.typstFontBase || "https://cdn.jsdelivr.net/npm/@typst-wasm/fonts@1.0.0/dist/files/";
    const fontUrls = [
      fontBase + "NewCM10-Regular.otf",
      fontBase + "NewCMMath-Regular.otf",
    ];
    await window.manimTypstCompiler.addFonts(
      ...await Promise.all(fontUrls.map(async (url) =>
        new Uint8Array(await (await fetch(url)).arrayBuffer())
      ))
    );
  }

  if (!window.manimPangoTextToSvg && window.PangoTextWasm) {
    window.manimPangoTextToSvg = window.PangoTextWasm.textToSvg;
  }

  return {
    PathKit: window.PathKit,
    typst: window.manimTypstCompiler,
    pango: window.manimPangoTextToSvg || null,
  };
}

window.manimInitializeBrowserWasm = initializeBrowserWasm;