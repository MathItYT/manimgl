const fs = require("fs");
const path = require("path");

const source = require.resolve("monaco-pyright-lsp/dist/worker.js");
const outputDir = path.resolve(__dirname, "../vendor");
fs.mkdirSync(outputDir, { recursive: true });
fs.copyFileSync(source, path.join(outputDir, "pyright-worker.js"));
console.log("Copied Pyright worker to examples/vendor/pyright-worker.js");
