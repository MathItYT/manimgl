import createPangoModule from "./pango_text.js";

let modulePromise;

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

    const family = Module.UTF8ToString(ptr);
    Module.ccall("manim_pango_free", null, ["number"], [ptr]);
    return { name: fileName, family, path };
  }

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
    registerFont,
  };

  window.manimPangoTextToSvg = window.PangoTextWasm.textToSvg;
  window.manimRegisterPangoFont = registerFont;
  return window.PangoTextWasm;
}

window.manimInitializePangoText = initializePangoText;
