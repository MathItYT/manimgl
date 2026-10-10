#!/usr/bin/env python3
"""Generate a broad Pyright stub from the ManimGL source tree.

The browser language server cannot import the Pyodide runtime, so expose the
public names re-exported by manimlib.__init__ as permissive typed declarations.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "manimlib"
OUTPUT = ROOT / "examples" / "vendor" / "manimlib" / "__init__.pyi"

def module_path(module: str) -> Path:
    return PACKAGE.joinpath(*module.split(".")).with_suffix(".py")

def top_level_names(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    names = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.append((node.name, "class" if isinstance(node, ast.ClassDef) else "function"))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.append((target.id, "value"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if not name.startswith("_"):
                    names.append((name, "value"))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*" and not alias.name.startswith("_"):
                    names.append((alias.asname or alias.name, "value"))
    return names

init = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
modules = set()
explicit = []
for node in init.body:
    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("manimlib"):
        if any(alias.name == "*" for alias in node.names):
            modules.add(node.module)
        else:
            explicit.extend((alias.asname or alias.name, "value") for alias in node.names if not alias.name.startswith("_"))

exports = {}
for module in sorted(modules):
    path = module_path(module.removeprefix("manimlib."))
    if module == "manimlib":
        path = PACKAGE / "__init__.py"
    for name, kind in top_level_names(path):
        exports.setdefault(name, kind)
for name, kind in explicit:
    exports.setdefault(name, kind)

lines = [
    '"""Generated browser-facing declarations for ManimGL public exports."""',
    "from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, TypeVar",
    "Vect3 = Any",
    "ManimColor = Any",
    "Self = Any",
    "",
]
for name, kind in sorted(exports.items()):
    if name in {"annotations"}:
        continue
    if kind == "class":
        lines.extend([f"class {name}:", "    def __init__(self, *args: Any, **kwargs: Any) -> None: ...", ""])
    elif kind == "function":
        lines.append(f"def {name}(*args: Any, **kwargs: Any) -> Any: ...")
    else:
        lines.append(f"{name}: Any")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Generated {len(exports)} ManimGL names: {OUTPUT}")
