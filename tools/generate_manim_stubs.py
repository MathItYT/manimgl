#!/usr/bin/env python3
"""Generate browser-facing Pyright declarations for ManimGL exports.

Imported names are lower priority than declarations in their defining
modules. Otherwise a re-export can turn a real class (for example Circle)
into Circle: Any, which Pyright treats as a variable rather than a class.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "manimlib"
OUTPUT = ROOT / "examples" / "vendor" / "manimlib" / "__init__.pyi"
PRIORITY = {"import": 0, "value": 1, "function": 2, "class": 3}

def module_path(module: str) -> Path:
    return PACKAGE.joinpath(*module.split(".")).with_suffix(".py")

def top_level_names(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    names = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                names.append((node.name, "class"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.append((node.name, "function"))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.append((target.id, "value"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if not name.startswith("_"):
                    names.append((name, "import"))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*" and not alias.name.startswith("_"):
                    names.append((alias.asname or alias.name, "import"))
    return names

def add_export(exports, name, kind):
    if name.startswith("_"):
        return
    previous = exports.get(name)
    if previous is None or PRIORITY[kind] > PRIORITY[previous]:
        exports[name] = kind

init = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
modules = set()
explicit = []
for node in init.body:
    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("manimlib"):
        # Scan explicit imports too, to identify definitions in their modules.
        modules.add(node.module)
        if not any(alias.name == "*" for alias in node.names):
            explicit.extend((alias.asname or alias.name, "import") for alias in node.names)

exports = {}
for module in sorted(modules):
    relative = module.removeprefix("manimlib.")
    path = PACKAGE / "__init__.py" if module == "manimlib" else module_path(relative)
    for name, kind in top_level_names(path):
        add_export(exports, name, kind)
for name, kind in explicit:
    add_export(exports, name, kind)

lines = [
    """Generated browser-facing declarations for ManimGL public exports.""",
    "from typing import Any",
    "",
    "Vect3 = Any",
    "ManimColor = Any",
    "Self = Any",
    "",
]
for name, kind in sorted(exports.items()):
    if name == "annotations":
        continue
    if kind == "class":
        lines.extend([f"class {name}:", "    def __init__(self, *args: Any, **kwargs: Any) -> None: ...", ""])
    elif kind == "function":
        lines.append(f"def {name}(*args: Any, **kwargs: Any) -> Any: ...")
    else:
        lines.append(f"{name}: Any")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(f"Generated {len(exports)} ManimGL names: {OUTPUT}")
