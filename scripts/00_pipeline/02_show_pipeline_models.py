#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print linked pipeline models by module.")
    parser.add_argument(
        "--model-links",
        default="outputs/pipeline/model_links.json",
        help="Path to linked model manifest JSON.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    path = Path(args.model_links)
    if not path.exists():
        print(f"[ERROR] file not found: {path}")
        return 1

    payload = _load_json(path)
    modules = payload.get("modules", {})
    if not isinstance(modules, dict):
        print("[ERROR] invalid model links: missing modules object")
        return 1

    print(f"[INFO] linked model manifest: {path}")
    for module in sorted(modules.keys()):
        selected = modules[module].get("selected", {})
        repo_id = selected.get("repo_id", "<unknown>")
        exists = selected.get("exists", False)
        local_path = selected.get("local_path", "<unknown>")
        print(f"- {module}: {repo_id}")
        print(f"  exists={exists} path={local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
