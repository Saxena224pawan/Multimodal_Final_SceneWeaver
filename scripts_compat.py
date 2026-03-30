from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_validate_models_registry(project_root: Path) -> ModuleType:
    target = project_root / "scripts" / "00_pipeline" / "00_validate_models_registry.py"
    return _load_module("validate_models_registry", target)
