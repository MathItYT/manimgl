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
