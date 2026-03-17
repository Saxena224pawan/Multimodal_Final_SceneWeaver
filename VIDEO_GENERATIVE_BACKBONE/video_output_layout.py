from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


WINDOW_ID_RE = re.compile(r"^w_[0-9]{3,}$")

REQUIRED_SCENE_PLAN_FIELDS = {
    "story_id",
    "title",
    "window_seconds",
    "total_windows",
    "windows",
}

REQUIRED_WINDOW_FIELDS = {
    "window_id",
    "window_index",
    "beat_id",
    "scene_prompt",
    "expected_caption",
    "emotion",
    "continuity_anchor",
}

REQUIRED_MANIFEST_FIELDS = {
    "version",
    "story_id",
    "title",
    "fps",
    "container_ext",
    "window_seconds",
    "total_windows",
    "jobs",
}

REQUIRED_JOB_FIELDS = {
    "job_id",
    "window_id",
    "window_index",
    "beat_id",
    "scene_prompt_file",
    "expected_caption_file",
    "output_video_path",
    "frames_dir",
    "status",
}


def _add_error(errors: List[Dict[str, str]], path: str, message: str) -> None:
    errors.append({"path": path, "message": message})


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def validate_scene_plan_for_video(payload: Any) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if not isinstance(payload, dict):
        _add_error(errors, "$", "scene plan must be a JSON object")
        return errors

    fields = set(payload.keys())
    for name in sorted(REQUIRED_SCENE_PLAN_FIELDS - fields):
        _add_error(errors, name, "missing required field")

    if "story_id" in payload and not _is_non_empty_string(payload.get("story_id")):
        _add_error(errors, "story_id", "must be a non-empty string")
    if "title" in payload and not _is_non_empty_string(payload.get("title")):
        _add_error(errors, "title", "must be a non-empty string")
    if "window_seconds" in payload:
        value = payload.get("window_seconds")
        if not isinstance(value, int) or value <= 0:
            _add_error(errors, "window_seconds", "must be a positive integer")
    if "total_windows" in payload:
        value = payload.get("total_windows")
        if not isinstance(value, int) or value <= 0:
            _add_error(errors, "total_windows", "must be a positive integer")

    windows = payload.get("windows")
    if "windows" in payload:
        if not isinstance(windows, list):
            _add_error(errors, "windows", "must be a list")
            windows = []
        elif not windows:
            _add_error(errors, "windows", "must contain at least one window")

    if isinstance(windows, list) and isinstance(payload.get("total_windows"), int):
        if len(windows) != payload["total_windows"]:
            _add_error(
                errors,
                "total_windows",
                f"declares {payload['total_windows']}, but windows has {len(windows)} items",
            )

    seen_window_ids: set[str] = set()
    seen_window_indices: set[int] = set()
    for idx, window in enumerate(windows or []):
        base = f"windows[{idx}]"
        if not isinstance(window, dict):
            _add_error(errors, base, "must be an object")
            continue

        for name in sorted(REQUIRED_WINDOW_FIELDS - set(window.keys())):
            _add_error(errors, f"{base}.{name}", "missing required field")

        window_id = window.get("window_id")
        if "window_id" in window:
            if not _is_non_empty_string(window_id):
                _add_error(errors, f"{base}.window_id", "must be a non-empty string")
            elif not WINDOW_ID_RE.match(window_id):
                _add_error(errors, f"{base}.window_id", "must match pattern w_000")
            elif window_id in seen_window_ids:
                _add_error(errors, f"{base}.window_id", f"duplicate window_id '{window_id}'")
            else:
                seen_window_ids.add(window_id)

        window_index = window.get("window_index")
        if "window_index" in window:
            if not isinstance(window_index, int) or window_index < 0:
                _add_error(errors, f"{base}.window_index", "must be an integer >= 0")
            elif window_index in seen_window_indices:
                _add_error(errors, f"{base}.window_index", f"duplicate window_index '{window_index}'")
            else:
                seen_window_indices.add(window_index)

        for field_name in ("beat_id", "scene_prompt", "expected_caption", "emotion"):
            if field_name in window and not _is_non_empty_string(window.get(field_name)):
                _add_error(errors, f"{base}.{field_name}", "must be a non-empty string")

        continuity = window.get("continuity_anchor")
        if "continuity_anchor" in window:
            if not isinstance(continuity, dict):
                _add_error(errors, f"{base}.continuity_anchor", "must be an object")
            else:
                world_anchor = continuity.get("world_anchor")
                if not _is_non_empty_string(world_anchor):
                    _add_error(errors, f"{base}.continuity_anchor.world_anchor", "must be a non-empty string")

    return errors


def validate_video_manifest(payload: Any) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if not isinstance(payload, dict):
        _add_error(errors, "$", "manifest must be a JSON object")
        return errors

    for name in sorted(REQUIRED_MANIFEST_FIELDS - set(payload.keys())):
        _add_error(errors, name, "missing required field")

    jobs = payload.get("jobs")
    if "jobs" in payload:
        if not isinstance(jobs, list):
            _add_error(errors, "jobs", "must be a list")
            jobs = []
        elif not jobs:
            _add_error(errors, "jobs", "must contain at least one job")

    seen_job_ids: set[str] = set()
    for idx, job in enumerate(jobs or []):
        base = f"jobs[{idx}]"
        if not isinstance(job, dict):
            _add_error(errors, base, "must be an object")
            continue
        for name in sorted(REQUIRED_JOB_FIELDS - set(job.keys())):
            _add_error(errors, f"{base}.{name}", "missing required field")

        job_id = job.get("job_id")
        if "job_id" in job:
            if not _is_non_empty_string(job_id):
                _add_error(errors, f"{base}.job_id", "must be a non-empty string")
            elif job_id in seen_job_ids:
                _add_error(errors, f"{base}.job_id", f"duplicate job_id '{job_id}'")
            else:
                seen_job_ids.add(job_id)

        if "status" in job and job.get("status") not in {"pending", "generated", "failed", "simulated", "skipped"}:
            _add_error(errors, f"{base}.status", "must be one of: pending, generated, failed, simulated, skipped")

    return errors


def prepare_video_output_layout(
    scene_plan: Dict[str, Any],
    *,
    output_root: Path,
    fps: int,
    container_ext: str,
    selected_video_model: Optional[Dict[str, Any]] = None,
    overwrite_metadata: bool = True,
) -> Dict[str, Any]:
    errors = validate_scene_plan_for_video(scene_plan)
    if errors:
        rendered = "\n".join(f"- {item['path']}: {item['message']}" for item in errors)
        raise ValueError(f"Invalid scene plan for video layout:\n{rendered}")

    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    if not _is_non_empty_string(container_ext):
        raise ValueError("container_ext must be a non-empty string")

    container_ext = container_ext.lstrip(".").lower()
    if container_ext not in {"mp4", "mov", "webm"}:
        raise ValueError("container_ext must be one of: mp4, mov, webm")

    windows_dir = output_root / "windows"
    final_dir = output_root / "final"
    debug_dir = output_root / "debug"
    metrics_dir = output_root / "metrics"
    for d in (windows_dir, final_dir, debug_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    sorted_windows = sorted(scene_plan["windows"], key=lambda w: int(w["window_index"]))
    jobs: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []

    for window in sorted_windows:
        window_id = window["window_id"]
        window_idx = int(window["window_index"])
        beat_id = window["beat_id"]
        window_dir = windows_dir / window_id
        frames_dir = window_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        prompt_file = window_dir / "prompt.txt"
        expected_caption_file = window_dir / "expected_caption.txt"
        metadata_file = window_dir / "window_job.json"
        output_video_file = window_dir / f"generated.{container_ext}"

        _write_text(prompt_file, window["scene_prompt"].strip() + "\n", overwrite=overwrite_metadata)
        _write_text(
            expected_caption_file,
            window["expected_caption"].strip() + "\n",
            overwrite=overwrite_metadata,
        )

        job = {
            "job_id": f"job_{window_id}",
            "window_id": window_id,
            "window_index": window_idx,
            "beat_id": beat_id,
            "status": "pending",
            "scene_prompt_file": str(prompt_file),
            "expected_caption_file": str(expected_caption_file),
            "output_video_path": str(output_video_file),
            "frames_dir": str(frames_dir),
            "scene_prompt": window["scene_prompt"],
            "expected_caption": window["expected_caption"],
            "emotion": window["emotion"],
            "window_seconds": scene_plan["window_seconds"],
            "fps": fps,
            "model": selected_video_model or {},
        }
        _write_json(metadata_file, job, overwrite=overwrite_metadata)
        jobs.append(job)
        status_rows.append(
            {
                "window_id": window_id,
                "window_index": window_idx,
                "status": "pending",
                "output_video_path": str(output_video_file),
            }
        )

    concat_file = final_dir / "windows.concat"
    concat_lines = ["ffconcat version 1.0"]
    final_dir_abs = final_dir.resolve()
    for job in jobs:
        video_path = Path(job["output_video_path"])
        video_abs = video_path if video_path.is_absolute() else (Path.cwd() / video_path).resolve()
        concat_ref = os.path.relpath(video_abs, start=final_dir_abs)
        concat_lines.append(f"file {concat_ref}")
    _write_text(concat_file, "\n".join(concat_lines) + "\n", overwrite=overwrite_metadata)

    final_assembly = {
        "version": 1,
        "story_id": scene_plan["story_id"],
        "title": scene_plan["title"],
        "concat_file": str(concat_file),
        "output_video_path": str(final_dir / f"{scene_plan['story_id']}_final.{container_ext}"),
        "window_count": len(jobs),
        "fps": fps,
        "container_ext": container_ext,
    }
    _write_json(final_dir / "final_assembly.json", final_assembly, overwrite=overwrite_metadata)

    manifest = {
        "version": 1,
        "story_id": scene_plan["story_id"],
        "title": scene_plan["title"],
        "fps": fps,
        "container_ext": container_ext,
        "window_seconds": scene_plan["window_seconds"],
        "total_windows": scene_plan["total_windows"],
        "video_model": selected_video_model or {},
        "paths": {
            "windows_dir": str(windows_dir),
            "final_dir": str(final_dir),
            "debug_dir": str(debug_dir),
            "metrics_dir": str(metrics_dir),
        },
        "jobs": jobs,
    }
    _write_json(debug_dir / "generation_manifest.json", manifest, overwrite=overwrite_metadata)

    generation_status = {
        "version": 1,
        "story_id": scene_plan["story_id"],
        "total_windows": scene_plan["total_windows"],
        "generated_windows": 0,
        "failed_windows": 0,
        "pending_windows": len(jobs),
        "rows": status_rows,
    }
    _write_json(metrics_dir / "generation_status.json", generation_status, overwrite=overwrite_metadata)

    runner_template = """#!/usr/bin/env bash
set -euo pipefail

# Replace this command template with your actual backend runtime.
# Example:
# python3 scripts/04_video/01_generate_video_windows.py \\
#   --manifest outputs/video/debug/generation_manifest.json \\
#   --mode command \\
#   --command-template 'python3 my_backend.py --prompt-file {q_prompt_file} --out {q_output_video} --model {q_model_path}'

python3 scripts/04_video/01_generate_video_windows.py \\
  --manifest outputs/video/debug/generation_manifest.json \\
  --mode dry-run
"""
    runner_script = debug_dir / "run_generate_windows.sh"
    _write_text(runner_script, runner_template, overwrite=overwrite_metadata)

    return manifest
