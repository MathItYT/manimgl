const path = require("path");

module.exports = {
  mode: "production",
  target: "web",
  entry: "./provider-entry.js",
  output: {
    path: path.resolve(__dirname, "../vendor"),
    filename: "pyright-provider.js",
    library: { name: "PyrightLspBundle", type: "window" },
    globalObject: "globalThis",
    clean: false,
  },
  resolve: {
    conditionNames: ["browser", "import", "module", "default"],
    fallback: { fs: false, path: false, os: false, crypto: false },
  },
  performance: { hints: false },
};
