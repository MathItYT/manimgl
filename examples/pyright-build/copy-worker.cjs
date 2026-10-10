const fs = require("fs");
const path = require("path");

const source = require.resolve("monaco-pyright-lsp/dist/worker.js");
const outputDir = path.resolve(__dirname, "../vendor");
const output = path.join(outputDir, "pyright-worker.js");
fs.mkdirSync(outputDir, { recursive: true });
let worker = fs.readFileSync(source, "utf8");

// monaco-pyright-lsp mounts typeStubs under /typings. The patched worker
// also mounts them at / so Pyright can resolve real Python sources as workspace
// imports. Runtime site-packages are marked separately: mounting those under
// /typings as well would duplicate a large amount of source in the worker FS.
const marker = 'createUserFiles("/typings", msg.userFiles);';
if (!worker.includes(marker)) {
  throw new Error("Could not patch Pyright worker: expected user-file mount was not found.");
}
const replacement = [
  'const runtimeSources = msg.userFiles.__pyright_runtime_sources__;',
  'delete msg.userFiles.__pyright_runtime_sources__;',
  marker,
  'createUserFiles("/", msg.userFiles);',
  'if (runtimeSources) createUserFiles("/", runtimeSources);',
  'self.addEventListener("message", (event) => {',
  '  const update = event.data;',
  '  if (update && update.type === "__manimgl_update_runtime_sources" && update.userFiles) {',
  '    createUserFiles("/", update.userFiles);',
  '  }',
  '});'
].join("\n");
worker = worker.replace(marker, replacement);
fs.writeFileSync(output, worker);
console.log("Patched and copied Pyright worker to examples/vendor/pyright-worker.js");
