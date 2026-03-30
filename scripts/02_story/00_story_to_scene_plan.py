#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LLM_MODEL import build_scene_plan, validate_story_spec
from pipeline_runtime import get_selected_model, load_model_links


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate StorySpec JSON and build a window-level scene plan."
    )
    parser.add_argument(
        "--story-spec",
        default="LLM_MODEL/story_spec.template.json",
        help="Path to StorySpec JSON input.",
    )
    parser.add_argument(
        "--output",
        default="outputs/story/scene_plan.json",
        help="Output path for generated scene plan JSON.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=4,
        help="Length of each generation window in seconds.",
    )
    parser.add_argument(
        "--total-windows",
        type=int,
        default=None,
        help="Optional override for total windows. If omitted, derived from runtime/window length.",
    )
    parser.add_argument(
        "--json-report",
        default=None,
        help="Optional path to write machine-readable validation/build report.",
    )
    parser.add_argument(
        "--model-links",
        default="outputs/pipeline/model_links.json",
        help="Optional linked model manifest path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    story_spec_path = Path(args.story_spec)
    output_path = Path(args.output)
    report_path: Optional[Path] = Path(args.json_report) if args.json_report else None
    model_links_path = Path(args.model_links)

    if not story_spec_path.exists():
        error = {
            "ok": False,
            "errors": [{"path": "story_spec", "message": f"file not found: {story_spec_path}"}],
        }
        if report_path:
            _write_json(report_path, error)
        print(f"[ERROR] story spec not found: {story_spec_path}", file=sys.stderr)
        return 1

    try:
        story_spec = json.loads(story_spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        error = {
            "ok": False,
            "errors": [{"path": "story_spec", "message": f"invalid JSON: {exc}"}],
        }
        if report_path:
            _write_json(report_path, error)
        print(f"[ERROR] invalid JSON in {story_spec_path}: {exc}", file=sys.stderr)
        return 1

    validation_errors = validate_story_spec(story_spec)
    if validation_errors:
        payload = {"ok": False, "errors": validation_errors}
        if report_path:
            _write_json(report_path, payload)
        print("[ERROR] story spec validation failed:", file=sys.stderr)
        for error in validation_errors:
            print(f"  - {error['path']}: {error['message']}", file=sys.stderr)
        return 1

    try:
        scene_plan = build_scene_plan(
            story_spec,
            window_seconds=args.window_seconds,
            total_windows=args.total_windows,
        )
    except ValueError as exc:
        payload = {"ok": False, "errors": [{"path": "build_scene_plan", "message": str(exc)}]}
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] scene plan build failed: {exc}", file=sys.stderr)
        return 1

    model_links = load_model_links(model_links_path)
    selected_llm = get_selected_model(model_links, "llm_model")
    if selected_llm is not None:
        scene_plan["runtime_models"] = {"llm_model": selected_llm}

    _write_json(output_path, scene_plan)
    success_payload = {
        "ok": True,
        "errors": [],
        "output": str(output_path),
        "story_id": scene_plan["story_id"],
        "total_windows": scene_plan["total_windows"],
    }
    if report_path:
        _write_json(report_path, success_payload)

    print(
        f"[OK] scene plan generated: {output_path} "
        f"(story_id={scene_plan['story_id']}, windows={scene_plan['total_windows']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
