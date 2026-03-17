from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HF_PLUGIN import HFPlugin

CANONICAL_MODULES = [
    "llm_model",
    "video_backbone",
    "caption_gen",
    "window_gen",
    "feedback_caption",
    "global_local_emb_feedback",
]

LEGACY_MODULE_ALIASES = {
    "LLM_MODEL": "llm_model",
    "VIDEO_GENERATIVE_BACKBONE": "video_backbone",
    "Caption_Gen": "caption_gen",
    "Window_Gen": "window_gen",
    "Feedback_Caption": "feedback_caption",
    "Globa_Local_Emb_Feedback": "global_local_emb_feedback",
}


def _normalize_module_name(module: str) -> str:
    if module == "all":
        return "all"
    return LEGACY_MODULE_ALIASES.get(module, module)


def _load_registry(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("datasets", [])
    if not isinstance(items, list):
        raise ValueError("datasets_registry.json must contain a 'datasets' list")
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if not item.get("repo_id") or not item.get("module") or not item.get("key"):
            continue
        module = str(item.get("module"))
        item = dict(item)
        item["module"] = _normalize_module_name(module)
        normalized.append(item)
    return normalized


def _select(
    items: List[Dict[str, Any]],
    module: str,
    keys: Optional[List[str]],
) -> List[Dict[str, Any]]:
    normalized_module = _normalize_module_name(module)
    selected = items
    if normalized_module != "all":
        selected = [it for it in selected if it.get("module") == normalized_module]
    if keys:
        key_set = set(keys)
        selected = [it for it in selected if it.get("key") in key_set]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Download curated storytelling datasets from datasets_registry.json")
    parser.add_argument("--registry", type=str, default="datasets_registry.json")
    parser.add_argument("--base_dir", type=str, default="datasets")
    parser.add_argument(
        "--module",
        type=str,
        default="all",
        choices=["all", *CANONICAL_MODULES, *sorted(LEGACY_MODULE_ALIASES.keys())],
    )
    parser.add_argument("--keys", type=str, nargs="*", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = root / registry_path
    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = root / base_dir

    datasets = _load_registry(registry_path)
    selected = _select(datasets, module=args.module, keys=args.keys)
    if not selected:
        print("No datasets selected.")
        return

    hf = HFPlugin()
    for item in selected:
        key = item["key"]
        repo_id = item["repo_id"]
        module = item["module"]
        local_dir = base_dir / module / key
        if args.dry_run:
            print(f"[DRY-RUN] {repo_id} -> {local_dir}")
            continue
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DOWNLOAD] {repo_id} -> {local_dir}")
        hf.download_dataset(repo_id=repo_id, local_dir=local_dir.as_posix())


if __name__ == "__main__":
    main()
