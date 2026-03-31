from pathlib import Path
from typing import Dict, List

import setuptools


def parse_requirements(requirements_path: Path) -> List[str]:
    requirements: List[str] = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirements.append(stripped)
    return requirements


def get_dynamic_extras() -> Dict[str, List[str]]:
    extras_root = Path(__file__).parent / "manimlib" / "extras"
    if not extras_root.exists():
        return {}

    extras: Dict[str, List[str]] = {}
    for extra_dir in extras_root.iterdir():
        if not extra_dir.is_dir() or extra_dir.name.startswith("_"):
            continue
        requirements_file = extra_dir / "requirements.txt"
        extras[extra_dir.name] = (
            parse_requirements(requirements_file)
            if requirements_file.exists()
            else []
        )

    if extras:
        extras["all"] = sorted({dep for deps in extras.values() for dep in deps})

    return extras


setuptools.setup(extras_require=get_dynamic_extras())