const fs = require("fs");
const path = require("path");

const source = require.resolve("monaco-pyright-lsp/dist/worker.js");
const outputDir = path.resolve(__dirname, "../vendor");
const output = path.join(outputDir, "pyright-worker.js");
fs.mkdirSync(outputDir, { recursive: true });
let worker = fs.readFileSync(source, "utf8");

// monaco-pyright-lsp normally exposes extra packages only as type stubs under
// /typings. Mount the supplied package a second time at / so Pyright can
// resolve and analyze its real .py files from the project root.
const marker = 'createUserFiles("/typings", msg.userFiles);';
if (!worker.includes(marker)) {
  throw new Error("Could not patch Pyright worker: expected user-file mount was not found.");
}
worker = worker.replace(marker, `${marker}\n    createUserFiles("/", msg.userFiles);`);
fs.writeFileSync(output, worker);
console.log("Patched and copied Pyright worker to examples/vendor/pyright-worker.js");
