from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_model_links(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    modules = payload.get("modules")
    if not isinstance(modules, dict):
        return None
    return payload


def get_selected_model(payload: Optional[Dict[str, Any]], module: str) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    modules = payload.get("modules")
    if not isinstance(modules, dict):
        return None
    module_block = modules.get(module)
    if not isinstance(module_block, dict):
        return None
    selected = module_block.get("selected")
    if not isinstance(selected, dict):
        return None
    repo_id = selected.get("repo_id")
    local_path = selected.get("local_path")
    exists = selected.get("exists")
    if not isinstance(repo_id, str) or not isinstance(local_path, str):
        return None
    if not isinstance(exists, bool):
        return None
    return {
        "repo_id": repo_id,
        "local_path": local_path,
        "exists": exists,
        "key": selected.get("key"),
        "license": selected.get("license"),
    }
