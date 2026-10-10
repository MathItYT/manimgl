import { MonacoPyrightProvider } from "monaco-pyright-lsp";

// Expose the provider as a browser global; Monaco itself is loaded separately
// by manim_editor.html and must not be duplicated inside this bundle.
globalThis.MonacoPyrightProvider = MonacoPyrightProvider;
