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
    options.typstModule ||
    "https://cdn.jsdelivr.net/npm/typst-wasm@1.0.0/dist/index.js"
  );
  const typstWorkerModule = await import(
    options.typstWorkerModule ||
    "https://cdn.jsdelivr.net/npm/typst-wasm@1.0.0/dist/worker/browser.js"
  );

  if (!window.manimTypstCompiler) {
    const workerUrl = options.typstWorkerUrl ||
      "https://cdn.jsdelivr.net/npm/typst-wasm@1.0.0/dist/worker/web-worker.js";
    const coreBase = options.typstCoreBase ||
      "https://cdn.jsdelivr.net/npm/typst-wasm@1.0.0/dist/engine/";

    const coreModules = {
      "engine.core.wasm": WebAssembly.compileStreaming(
        fetch(coreBase + "engine.core.wasm")
      ),
      "engine.core2.wasm": WebAssembly.compileStreaming(
        fetch(coreBase + "engine.core2.wasm")
      ),
      "engine.core3.wasm": WebAssembly.compileStreaming(
        fetch(coreBase + "engine.core3.wasm")
      ),
    };

    const worker = () => typstWorkerModule.createWebWorker(workerUrl);
    window.manimTypstCompiler = await typstModule.createTypstCompiler({
      backend: "worker",
      worker,
      coreModules,
    });
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
