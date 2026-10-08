import createPangoModule from "./pango_text.js";

let modulePromise;
let asyncWorker;
let asyncWorkerReady;
let nextWorkerRequestId = 1;
const pendingWorkerRequests = new Map();

function ensureAsyncWorker() {
  if (asyncWorkerReady) return asyncWorkerReady;

  asyncWorker = new Worker(
    new URL("./pango_text_worker.js", import.meta.url),
    { type: "module" },
  );

  asyncWorkerReady = new Promise((resolve, reject) => {
    const fail = (error) => {
      const failure = error instanceof Error ? error : new Error(String(error));
      reject(failure);
      for (const pending of pendingWorkerRequests.values()) {
        pending.reject(failure);
      }
      pendingWorkerRequests.clear();
    };

    asyncWorker.addEventListener("message", (event) => {
      const message = event.data;
      if (message.type === "ready") {
        resolve();
        return;
      }
      if (message.type === "fatal") {
        fail(new Error(message.error || "Pango worker initialization failed"));
        return;
      }

      const pending = pendingWorkerRequests.get(message.id);
      if (!pending) return;
      pendingWorkerRequests.delete(message.id);
      if (message.ok) pending.resolve(message.value);
      else pending.reject(new Error(message.error || "Pango worker request failed"));
    });

    asyncWorker.addEventListener("error", (event) => {
      fail(event.error || new Error(event.message || "Pango worker failed"));
    }, { once: true });

    asyncWorker.postMessage({ type: "initialize" });
  });

  return asyncWorkerReady;
}

async function requestWorker(type, payload = {}, transfer = []) {
  await ensureAsyncWorker();
  const id = nextWorkerRequestId++;
  return new Promise((resolve, reject) => {
    pendingWorkerRequests.set(id, { resolve, reject });
    asyncWorker.postMessage({ id, type, ...payload }, transfer);
  });
}

function alignmentCode(alignment) {
  if (typeof alignment === "number") return alignment;
  const values = { LEFT: 0, CENTER: 1, RIGHT: 2 };
  return values[String(alignment || "CENTER").toUpperCase()] ?? 1;
}

export async function initializePangoText() {
  if (!modulePromise) {
    modulePromise = createPangoModule({
      locateFile: (file) => new URL(file, import.meta.url).href,
    });
  }

  const Module = await modulePromise;
  await ensureAsyncWorker();

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
    const path = "/fonts/" + (++fontCounter.value) + "-" + safeName;
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
      throw new Error("PangoCairo could not register font: " + fileName);
    }

    const family = Module.UTF8ToString(ptr);
    Module.ccall("manim_pango_free", null, ["number"], [ptr]);

    // Register the same font in the worker's independent WASM filesystem.
    // Transfer a copy so the main module can continue using its own bytes.
    const workerBytes = new Uint8Array(bytes).slice();
    const workerBuffer = workerBytes.buffer;
    const workerFont = await requestWorker(
      "registerFont",
      { bytes: workerBuffer, fileName },
      [workerBuffer],
    );
    return {
      name: fileName,
      family: workerFont.family || family,
      path: workerFont.path,
    };
  }

  window.PangoTextWasm = {
    textToSvg(markup, justify, indent, alignment, width) {
      const ptr = Module.ccall(
        "manim_pango_text_to_svg",
        "number",
        ["string", "number", "number", "number", "number"],
        [markup, justify ? 1 : 0, indent, alignmentCode(alignment), width],
      );
      if (!ptr) throw new Error("PangoCairo failed to render text");
      try {
        return Module.UTF8ToString(ptr);
      } finally {
        Module.ccall("manim_pango_free", null, ["number"], [ptr]);
      }
    },
    registerFont,
  };

  window.manimPangoTextToSvg = window.PangoTextWasm.textToSvg;
  window.manimPangoTextToSvgAsync = (
    markup, justify, indent, alignment, width,
  ) => requestWorker("render", {
    markup,
    justify: Boolean(justify),
    indent: Number(indent),
    alignment: alignmentCode(alignment),
    width: Number(width),
  });
  window.manimRegisterPangoFont = registerFont;
  return window.PangoTextWasm;
}

window.manimInitializePangoText = initializePangoText;
