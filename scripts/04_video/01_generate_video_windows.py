#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from VIDEO_GENERATIVE_BACKBONE import validate_video_manifest


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate (or simulate) video windows using generation manifest."
    )
    parser.add_argument(
        "--manifest",
        default="outputs/video/debug/generation_manifest.json",
        help="Path to generation manifest JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "command"],
        default="dry-run",
        help="dry-run writes simulation markers; command executes command template per job.",
    )
    parser.add_argument(
        "--command-template",
        default=None,
        help=(
            "Command template used in command mode. Available placeholders: "
            "{window_id}, {beat_id}, {prompt_file}, {expected_caption_file}, {output_video}, "
            "{frames_dir}, {window_seconds}, {fps}, {model_repo_id}, {model_path}, "
            "and quoted variants with q_ prefix (e.g., {q_prompt_file})."
        ),
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Optional maximum number of jobs to process.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose output_video already exists.",
    )
    parser.add_argument(
        "--json-report",
        default=None,
        help="Optional machine-readable report path.",
    )
    return parser.parse_args()


def _build_context(job: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, str]:
    model = job.get("model", {})
    data = {
        "window_id": str(job["window_id"]),
        "beat_id": str(job["beat_id"]),
        "prompt_file": str(job["scene_prompt_file"]),
        "expected_caption_file": str(job["expected_caption_file"]),
        "output_video": str(job["output_video_path"]),
        "frames_dir": str(job["frames_dir"]),
        "window_seconds": str(job["window_seconds"]),
        "fps": str(job["fps"]),
        "model_repo_id": str(model.get("repo_id", "")),
        "model_path": str(model.get("local_path", "")),
        "story_id": str(manifest["story_id"]),
    }
    quoted = {f"q_{k}": shlex.quote(v) for k, v in data.items()}
    return {**data, **quoted}


def _build_full_status_rows(
    manifest: Dict[str, Any],
    processed_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_window_id = {row["window_id"]: row for row in processed_rows}
    full_rows: List[Dict[str, Any]] = []
    for job in sorted(manifest["jobs"], key=lambda item: int(item["window_index"])):
        processed = by_window_id.get(job["window_id"])
        if processed is not None:
            full_rows.append(
                {
                    "story_id": manifest["story_id"],
                    "window_id": job["window_id"],
                    "window_index": job["window_index"],
                    "beat_id": job["beat_id"],
                    "output_video_path": job["output_video_path"],
                    "status": processed["status"],
                }
            )
            continue
        full_rows.append(
            {
                "story_id": manifest["story_id"],
                "window_id": job["window_id"],
                "window_index": job["window_index"],
                "beat_id": job["beat_id"],
                "output_video_path": job["output_video_path"],
                "status": "pending",
            }
        )
    return full_rows


def _update_status_file(manifest_path: Path, manifest: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    metrics_path = manifest_path.parent.parent / "metrics" / "generation_status.json"
    full_rows = _build_full_status_rows(manifest, rows)
    generated = sum(1 for row in rows if row["status"] == "generated")
    failed = sum(1 for row in rows if row["status"] == "failed")
    simulated = sum(1 for row in full_rows if row["status"] == "simulated")
    skipped = sum(1 for row in full_rows if row["status"] == "skipped")
    pending = sum(1 for row in full_rows if row["status"] == "pending")
    payload = {
        "version": 1,
        "story_id": manifest["story_id"],
        "total_windows": manifest["total_windows"],
        "generated_windows": generated,
        "failed_windows": failed,
        "simulated_windows": simulated,
        "skipped_windows": skipped,
        "pending_windows": pending,
        "rows": full_rows,
    }
    _write_json(metrics_path, payload)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    report_path: Optional[Path] = Path(args.json_report) if args.json_report else None

    if not manifest_path.exists():
        payload = {
            "ok": False,
            "errors": [{"path": "manifest", "message": f"file not found: {manifest_path}"}],
        }
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        manifest = _load_json(manifest_path)
    except json.JSONDecodeError as exc:
        payload = {
            "ok": False,
            "errors": [{"path": "manifest", "message": f"invalid JSON: {exc}"}],
        }
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] invalid JSON in {manifest_path}: {exc}", file=sys.stderr)
        return 1

    errors = validate_video_manifest(manifest)
    if errors:
        payload = {"ok": False, "errors": errors}
        if report_path:
            _write_json(report_path, payload)
        print("[ERROR] video manifest validation failed:", file=sys.stderr)
        for item in errors:
            print(f"  - {item['path']}: {item['message']}", file=sys.stderr)
        return 1

    if args.mode == "command" and not args.command_template:
        payload = {
            "ok": False,
            "errors": [{"path": "command_template", "message": "required in command mode"}],
        }
        if report_path:
            _write_json(report_path, payload)
        print("[ERROR] --command-template is required in command mode", file=sys.stderr)
        return 1

    jobs = sorted(manifest["jobs"], key=lambda row: int(row["window_index"]))
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    run_rows: List[Dict[str, Any]] = []
    debug_dir = manifest_path.parent
    run_log_path = debug_dir / "last_generation_run.json"
    window_logs_dir = debug_dir / "window_logs"
    window_logs_dir.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        output_video = Path(job["output_video_path"])
        output_video.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = Path(job["frames_dir"])
        frames_dir.mkdir(parents=True, exist_ok=True)
        window_id = str(job["window_id"])
        stdout_log_path = window_logs_dir / f"{window_id}.stdout.log"
        stderr_log_path = window_logs_dir / f"{window_id}.stderr.log"

        row = {
            "story_id": manifest["story_id"],
            "window_id": window_id,
            "window_index": job["window_index"],
            "beat_id": job["beat_id"],
            "output_video_path": str(output_video),
            "status": "pending",
            "mode": args.mode,
            "started_at": _now_iso(),
            "ended_at": None,
            "command": None,
            "error": None,
            "stdout_log": str(stdout_log_path),
            "stderr_log": str(stderr_log_path),
        }

        if args.skip_existing and output_video.exists():
            row["status"] = "skipped"
            row["ended_at"] = _now_iso()
            run_rows.append(row)
            continue

        if args.mode == "dry-run":
            marker_path = output_video.with_suffix(output_video.suffix + ".dryrun.txt")
            marker_text = (
                "Dry-run marker: no real video generated.\n"
                f"window_id={job['window_id']}\n"
                f"beat_id={job['beat_id']}\n"
                f"prompt_file={job['scene_prompt_file']}\n"
                f"expected_output={output_video}\n"
            )
            marker_path.write_text(marker_text, encoding="utf-8")
            row["status"] = "simulated"
            row["ended_at"] = _now_iso()
            run_rows.append(row)
            continue

        ctx = _build_context(job, manifest)
        assert args.command_template is not None
        try:
            cmd = args.command_template.format(**ctx)
        except KeyError as exc:
            row["ended_at"] = _now_iso()
            row["status"] = "failed"
            row["error"] = f"invalid command template placeholder: {exc}"
            _write_text(stdout_log_path, "")
            _write_text(stderr_log_path, row["error"] + "\n")
            print(
                f"[FAILED] window={window_id} placeholder_error={exc} stderr_log={stderr_log_path}",
                file=sys.stderr,
            )
            run_rows.append(row)
            continue

        row["command"] = cmd
        proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        _write_text(stdout_log_path, proc.stdout or "")
        _write_text(stderr_log_path, proc.stderr or "")
        row["ended_at"] = _now_iso()
        if proc.returncode != 0:
            row["status"] = "failed"
            tail = (proc.stderr or proc.stdout or "").strip()
            tail = tail[-400:] if tail else ""
            row["error"] = (
                f"command exited with code {proc.returncode}; "
                f"stderr_log={stderr_log_path}; "
                f"tail={tail}"
            )
            print(
                f"[FAILED] window={window_id} rc={proc.returncode} stderr_log={stderr_log_path}",
                file=sys.stderr,
            )
        else:
            row["status"] = "generated" if output_video.exists() else "failed"
            if row["status"] == "failed":
                row["error"] = (
                    "command succeeded but output video not found; "
                    f"stdout_log={stdout_log_path}; stderr_log={stderr_log_path}"
                )
                print(
                    f"[FAILED] window={window_id} output_missing expected={output_video}",
                    file=sys.stderr,
                )
            else:
                print(f"[OK] window={window_id} generated={output_video}")
        run_rows.append(row)

    _update_status_file(manifest_path, manifest, run_rows)
    _write_json(
        run_log_path,
        {
            "version": 1,
            "manifest": str(manifest_path),
            "mode": args.mode,
            "rows": run_rows,
        },
    )

    summary = {
        "generated": sum(1 for row in run_rows if row["status"] == "generated"),
        "failed": sum(1 for row in run_rows if row["status"] == "failed"),
        "simulated": sum(1 for row in run_rows if row["status"] == "simulated"),
        "skipped": sum(1 for row in run_rows if row["status"] == "skipped"),
        "total": len(run_rows),
    }
    if report_path:
        _write_json(
            report_path,
            {
                "ok": summary["failed"] == 0,
                "errors": [] if summary["failed"] == 0 else [{"path": "jobs", "message": "one or more jobs failed"}],
                "summary": summary,
                "run_log": str(run_log_path),
            },
        )

    print(
        "[OK] video window run completed: "
        f"total={summary['total']} generated={summary['generated']} "
        f"simulated={summary['simulated']} skipped={summary['skipped']} failed={summary['failed']}"
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
