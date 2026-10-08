import createPangoModule from "./pango_text.js";

let modulePromise;
let Module;
let fontCounter = 0;

async function initialize() {
  if (!modulePromise) {
    modulePromise = createPangoModule({
      locateFile: (file) => new URL(file, import.meta.url).href,
    });
  }
  Module = await modulePromise;
}

function registerFont(buffer, fileName = "browser-font.ttf") {
  const bytes = buffer instanceof ArrayBuffer
    ? new Uint8Array(buffer)
    : new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const safeName = String(fileName).replace(/[^A-Za-z0-9._-]/g, "_");
  const path = "/fonts/" + (++fontCounter) + "-" + safeName;

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

  try {
    return { name: fileName, family: Module.UTF8ToString(ptr), path };
  } finally {
    Module.ccall("manim_pango_free", null, ["number"], [ptr]);
  }
}

function renderText(markup, justify, indent, alignment, width) {
  const ptr = Module.ccall(
    "manim_pango_text_to_svg",
    "number",
    ["string", "number", "number", "number", "number"],
    [markup, justify ? 1 : 0, Number(indent), Number(alignment), Number(width)],
  );
  if (!ptr) throw new Error("PangoCairo failed to render text");
  try {
    return Module.UTF8ToString(ptr);
  } finally {
    Module.ccall("manim_pango_free", null, ["number"], [ptr]);
  }
}

self.addEventListener("message", async (event) => {
  const message = event.data || {};
  if (message.type === "initialize") {
    try {
      await initialize();
      self.postMessage({ type: "ready" });
    } catch (error) {
      self.postMessage({
        type: "fatal",
        error: String(error && (error.stack || error.message) || error),
      });
    }
    return;
  }

  try {
    if (!Module) throw new Error("Pango worker has not been initialized");
    let value;
    if (message.type === "registerFont") {
      value = registerFont(message.bytes, message.fileName);
    } else if (message.type === "render") {
      value = renderText(
        message.markup,
        message.justify,
        message.indent,
        message.alignment,
        message.width,
      );
    } else {
      throw new Error("Unknown Pango worker request: " + message.type);
    }
    self.postMessage({ id: message.id, ok: true, value });
  } catch (error) {
    self.postMessage({
      id: message.id,
      ok: false,
      error: String(error && (error.stack || error.message) || error),
    });
  }
});
