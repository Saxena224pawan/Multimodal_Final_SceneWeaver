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

from pipeline_runtime import get_selected_model, load_model_links
from VIDEO_GENERATIVE_BACKBONE import prepare_video_output_layout, validate_scene_plan_for_video


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare output layout and generation manifest for video windows."
    )
    parser.add_argument(
        "--scene-plan",
        default="outputs/story/scene_plan.json",
        help="Path to scene plan JSON.",
    )
    parser.add_argument(
        "--model-links",
        default="outputs/pipeline/model_links.json",
        help="Path to linked model manifest.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/video",
        help="Root output directory for video generation artifacts.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Target FPS for generated windows.",
    )
    parser.add_argument(
        "--container-ext",
        default="mp4",
        choices=["mp4", "mov", "webm"],
        help="Expected container extension for generated video windows.",
    )
    parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="Overwrite metadata files if they already exist.",
    )
    parser.add_argument(
        "--json-report",
        default=None,
        help="Optional machine-readable report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_plan_path = Path(args.scene_plan)
    model_links_path = Path(args.model_links)
    output_root = Path(args.output_root)
    report_path: Optional[Path] = Path(args.json_report) if args.json_report else None

    if not scene_plan_path.exists():
        payload = {
            "ok": False,
            "errors": [{"path": "scene_plan", "message": f"file not found: {scene_plan_path}"}],
        }
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] scene plan not found: {scene_plan_path}", file=sys.stderr)
        return 1

    try:
        scene_plan = json.loads(scene_plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        payload = {
            "ok": False,
            "errors": [{"path": "scene_plan", "message": f"invalid JSON: {exc}"}],
        }
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] invalid JSON in {scene_plan_path}: {exc}", file=sys.stderr)
        return 1

    errors = validate_scene_plan_for_video(scene_plan)
    if errors:
        payload = {"ok": False, "errors": errors}
        if report_path:
            _write_json(report_path, payload)
        print("[ERROR] scene plan validation failed for video layout:", file=sys.stderr)
        for err in errors:
            print(f"  - {err['path']}: {err['message']}", file=sys.stderr)
        return 1

    model_links = load_model_links(model_links_path)
    selected_video_model = get_selected_model(model_links, "video_backbone")

    try:
        manifest = prepare_video_output_layout(
            scene_plan,
            output_root=output_root,
            fps=args.fps,
            container_ext=args.container_ext,
            selected_video_model=selected_video_model,
            overwrite_metadata=args.overwrite_metadata,
        )
    except ValueError as exc:
        payload = {"ok": False, "errors": [{"path": "video_layout", "message": str(exc)}]}
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] failed to prepare video outputs: {exc}", file=sys.stderr)
        return 1

    manifest_path = output_root / "debug" / "generation_manifest.json"
    success = {
        "ok": True,
        "errors": [],
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "story_id": manifest["story_id"],
        "total_windows": manifest["total_windows"],
        "video_model": manifest.get("video_model", {}),
    }
    if report_path:
        _write_json(report_path, success)

    print(
        f"[OK] video output layout prepared: {output_root} "
        f"(story_id={manifest['story_id']}, windows={manifest['total_windows']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
