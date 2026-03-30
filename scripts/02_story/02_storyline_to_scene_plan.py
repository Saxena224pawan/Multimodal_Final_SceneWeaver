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

from LLM_MODEL import StorySpecBuilder, StorySpecBuilderConfig, build_scene_plan, validate_story_spec
from pipeline_runtime import get_selected_model, load_model_links


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _read_storyline(storyline: str, storyline_file: str) -> str:
    if storyline.strip():
        return storyline.strip()
    if storyline_file.strip():
        return Path(storyline_file).read_text(encoding="utf-8").strip()
    return ""


def _resolve_selected_model_id(selected: Optional[Dict[str, Any]]) -> Optional[str]:
    if not selected:
        return None
    local_path = selected.get("local_path")
    exists = selected.get("exists")
    repo_id = selected.get("repo_id")
    if isinstance(exists, bool) and exists and isinstance(local_path, str) and local_path.strip():
        return local_path
    if isinstance(repo_id, str) and repo_id.strip():
        return repo_id
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end: storyline -> StorySpec -> scene_plan windows/prompts."
    )
    parser.add_argument("--storyline", default="", help="Raw storyline text.")
    parser.add_argument("--storyline-file", default="", help="Optional storyline text file path.")
    parser.add_argument("--runtime-seconds", type=int, default=48, help="Target story runtime in seconds.")
    parser.add_argument("--window-seconds", type=int, default=4, help="Window length in seconds.")
    parser.add_argument("--target-beats", type=int, default=0, help="Optional explicit beat count.")
    parser.add_argument("--total-windows", type=int, default=0, help="Optional explicit window count.")
    parser.add_argument("--story-id", default="", help="Optional explicit story id.")
    parser.add_argument("--title", default="", help="Optional explicit story title.")

    parser.add_argument("--story-spec-output", default="outputs/story/story_spec.generated.json")
    parser.add_argument("--scene-plan-output", default="outputs/story/scene_plan.generated.json")
    parser.add_argument("--json-report", default="", help="Optional machine-readable report path.")

    parser.add_argument("--model-id", default="", help="Optional LLM model id/path for StorySpec generation.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable sampling in LLM generation mode.",
    )
    parser.add_argument(
        "--model-links",
        default="",
        help="Optional linked model manifest, used only when provided and --model-id is unset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    story_spec_output = Path(args.story_spec_output)
    scene_plan_output = Path(args.scene_plan_output)
    report_path: Optional[Path] = Path(args.json_report) if args.json_report else None

    try:
        storyline = _read_storyline(args.storyline, args.storyline_file)
    except OSError as exc:
        payload = {"ok": False, "errors": [{"path": "storyline", "message": str(exc)}]}
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] failed to read storyline: {exc}", file=sys.stderr)
        return 1

    if not storyline:
        payload = {
            "ok": False,
            "errors": [{"path": "storyline", "message": "provide --storyline or --storyline-file"}],
        }
        if report_path:
            _write_json(report_path, payload)
        print("[ERROR] empty storyline: provide --storyline or --storyline-file", file=sys.stderr)
        return 1

    model_id = args.model_id.strip()
    if not model_id and args.model_links.strip():
        model_links = load_model_links(Path(args.model_links))
        selected_llm = get_selected_model(model_links, "llm_model")
        resolved = _resolve_selected_model_id(selected_llm)
        if resolved:
            model_id = resolved

    builder = StorySpecBuilder(
        StorySpecBuilderConfig(
            model_id=model_id or None,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(args.do_sample),
        )
    )
    builder.load()

    try:
        story_spec = builder.build_from_storyline(
            storyline=storyline,
            runtime_seconds=args.runtime_seconds,
            window_seconds=args.window_seconds,
            story_id=args.story_id or None,
            title=args.title or None,
            target_beats=(args.target_beats if args.target_beats > 0 else None),
        )
    except ValueError as exc:
        payload = {"ok": False, "errors": [{"path": "build_story_spec", "message": str(exc)}]}
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] failed to build story spec: {exc}", file=sys.stderr)
        return 1

    spec_errors = validate_story_spec(story_spec)
    if spec_errors:
        payload = {"ok": False, "errors": spec_errors}
        if report_path:
            _write_json(report_path, payload)
        print("[ERROR] generated story spec failed validation", file=sys.stderr)
        for err in spec_errors:
            print(f"  - {err['path']}: {err['message']}", file=sys.stderr)
        return 1

    try:
        scene_plan = build_scene_plan(
            story_spec,
            window_seconds=args.window_seconds,
            total_windows=(args.total_windows if args.total_windows > 0 else None),
        )
    except ValueError as exc:
        payload = {"ok": False, "errors": [{"path": "build_scene_plan", "message": str(exc)}]}
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] failed to build scene plan: {exc}", file=sys.stderr)
        return 1

    _write_json(story_spec_output, story_spec)
    _write_json(scene_plan_output, scene_plan)

    payload = {
        "ok": True,
        "errors": [],
        "story_spec_output": str(story_spec_output),
        "scene_plan_output": str(scene_plan_output),
        "story_id": scene_plan["story_id"],
        "beats": len(story_spec.get("beats", [])),
        "windows": scene_plan["total_windows"],
        "build_mode": builder.last_build_mode,
        "llm_model_id": model_id,
        "llm_load_error": builder.last_load_error,
    }
    if report_path:
        _write_json(report_path, payload)

    print(
        f"[OK] scene plan generated: {scene_plan_output} "
        f"(story_id={scene_plan['story_id']}, windows={scene_plan['total_windows']}, mode={builder.last_build_mode})"
    )
    print(f"[OK] story spec saved: {story_spec_output}")
    if builder.last_load_error:
        print(f"[WARN] LLM builder load failed, used heuristic mode: {builder.last_load_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
