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

from Globa_Local_Emb_Feedback import (  # noqa: E402
    build_embedding_feedback,
    validate_scene_plan_for_embedding_feedback,
    validate_video_manifest_for_embedding_feedback,
)
from pipeline_runtime import get_selected_model, load_model_links  # noqa: E402


DEFAULT_DINOV2_MODEL = "facebook/dinov2-base"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_model_source(
    *,
    model_id_override: str,
    selected_model: Optional[Dict[str, Any]],
) -> str:
    if model_id_override.strip():
        return model_id_override.strip()

    if selected_model:
        local_path = selected_model.get("local_path")
        exists = selected_model.get("exists")
        if isinstance(local_path, str) and isinstance(exists, bool) and exists:
            return local_path

        repo_id = selected_model.get("repo_id")
        if isinstance(repo_id, str) and "dinov2" in repo_id.lower():
            return repo_id

    return DEFAULT_DINOV2_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute global/local DINOv2 embedding feedback across video windows."
    )
    parser.add_argument(
        "--scene-plan",
        default="outputs/story/scene_plan.json",
        help="Path to scene plan JSON.",
    )
    parser.add_argument(
        "--video-manifest",
        default="outputs/video/debug/generation_manifest.json",
        help="Path to video generation manifest JSON.",
    )
    parser.add_argument(
        "--model-links",
        default="outputs/pipeline/model_links.json",
        help="Path to linked model manifest JSON.",
    )
    parser.add_argument(
        "--model-id",
        default="",
        help="Optional DINOv2 model id/path override. If omitted, resolves from model links.",
    )
    parser.add_argument(
        "--output",
        default="outputs/embedding_feedback/global_local_embedding_feedback.json",
        help="Output JSON path for embedding feedback report.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=8,
        help="Max number of frames sampled per window from frames_dir.",
    )
    parser.add_argument(
        "--local-grid",
        type=int,
        default=3,
        help="Local pooled grid size per frame (e.g., 3 -> 3x3 local descriptors).",
    )
    parser.add_argument(
        "--global-min-sim",
        type=float,
        default=0.78,
        help="Minimum global embedding cosine similarity threshold.",
    )
    parser.add_argument(
        "--local-min-sim",
        type=float,
        default=0.72,
        help="Minimum local embedding cosine similarity threshold.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference: auto|cpu|cuda.",
    )
    parser.add_argument(
        "--fail-on-missing-frames",
        action="store_true",
        help="If set, windows without frame images are treated as errors.",
    )
    parser.add_argument(
        "--include-vectors",
        action="store_true",
        help="Include raw embedding vectors in output JSON (large files).",
    )
    parser.add_argument(
        "--json-report",
        default=None,
        help="Optional machine-readable execution report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    scene_plan_path = Path(args.scene_plan)
    video_manifest_path = Path(args.video_manifest)
    model_links_path = Path(args.model_links)
    output_path = Path(args.output)
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

    if not video_manifest_path.exists():
        payload = {
            "ok": False,
            "errors": [{"path": "video_manifest", "message": f"file not found: {video_manifest_path}"}],
        }
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] video manifest not found: {video_manifest_path}", file=sys.stderr)
        return 1

    try:
        scene_plan = _load_json(scene_plan_path)
    except json.JSONDecodeError as exc:
        payload = {
            "ok": False,
            "errors": [{"path": "scene_plan", "message": f"invalid JSON: {exc}"}],
        }
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] invalid JSON in {scene_plan_path}: {exc}", file=sys.stderr)
        return 1

    try:
        video_manifest = _load_json(video_manifest_path)
    except json.JSONDecodeError as exc:
        payload = {
            "ok": False,
            "errors": [{"path": "video_manifest", "message": f"invalid JSON: {exc}"}],
        }
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] invalid JSON in {video_manifest_path}: {exc}", file=sys.stderr)
        return 1

    scene_errors = validate_scene_plan_for_embedding_feedback(scene_plan)
    manifest_errors = validate_video_manifest_for_embedding_feedback(video_manifest)
    errors = scene_errors + manifest_errors
    if errors:
        payload = {"ok": False, "errors": errors}
        if report_path:
            _write_json(report_path, payload)
        print("[ERROR] embedding input validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err['path']}: {err['message']}", file=sys.stderr)
        return 1

    model_links = load_model_links(model_links_path)
    selected_embedding_model = get_selected_model(model_links, "global_local_emb_feedback")
    model_source = _resolve_model_source(
        model_id_override=args.model_id,
        selected_model=selected_embedding_model,
    )

    try:
        feedback = build_embedding_feedback(
            scene_plan,
            video_manifest,
            model_source=model_source,
            sample_frames=args.sample_frames,
            local_grid=args.local_grid,
            global_min_similarity=args.global_min_sim,
            local_min_similarity=args.local_min_sim,
            device=args.device,
            allow_missing_frames=not args.fail_on_missing_frames,
            include_vectors=args.include_vectors,
        )
    except ValueError as exc:
        payload = {"ok": False, "errors": [{"path": "embedding_feedback", "message": str(exc)}]}
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] embedding feedback build failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - explicit pipeline error surface
        payload = {"ok": False, "errors": [{"path": "embedding_runtime", "message": str(exc)}]}
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] embedding runtime failed: {exc}", file=sys.stderr)
        return 1

    if selected_embedding_model is not None:
        feedback.setdefault("runtime_models", {})[
            "global_local_emb_feedback"
        ] = selected_embedding_model

    _write_json(output_path, feedback)

    summary = feedback["summary"]
    ok = summary["error_windows"] == 0
    payload = {
        "ok": ok,
        "errors": [] if ok else [{"path": "rows", "message": "one or more windows failed"}],
        "output": str(output_path),
        "story_id": feedback["story_id"],
        "model_source": feedback["source"]["model_source"],
        "summary": summary,
    }
    if report_path:
        _write_json(report_path, payload)

    prefix = "[OK]" if ok else "[ERROR]"
    stream = sys.stdout if ok else sys.stderr
    print(
        f"{prefix} embedding feedback generated: {output_path} "
        f"(story_id={feedback['story_id']}, computed={summary['computed_windows']}, "
        f"compared={summary['compared_windows']}, pass={summary['pass_windows']}, "
        f"fail={summary['fail_windows']}, no_frames={summary['no_frames_windows']}, "
        f"errors={summary['error_windows']})",
        file=stream,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
