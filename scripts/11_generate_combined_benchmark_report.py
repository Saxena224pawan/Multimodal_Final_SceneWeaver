agentimport argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_ROOT = ROOT / "outputs" / "reports" / "vbench_all_metrics"


def load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt_seconds(value: float) -> str:
    sec = int(max(0.0, float(value)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def phase_status(enabled: bool, exit_code: int, summary: Optional[dict]) -> str:
    if not enabled:
        return "skipped"
    if summary is None:
        return "failed" if exit_code != 0 else "missing_summary"
    results = summary.get("results", [])
    if any(row.get("status") == "failed" for row in results):
        return "failed"
    if any(row.get("status") == "ok" for row in results):
        return "ok"
    if any(row.get("status") == "dry_run" for row in results):
        return "dry_run"
    return "missing_summary" if exit_code == 0 else "failed"


def summarize_results(results: List[dict]) -> Dict[str, int]:
    counts = {"ok": 0, "failed": 0, "dry_run": 0, "other": 0}
    for row in results:
        status = row.get("status")
        if status in counts:
            counts[status] += 1
        else:
            counts["other"] += 1
    return counts


def build_videobench_phase(enabled: bool, exit_code: int, summary_path: Optional[Path]) -> Dict[str, Any]:
    summary = load_json(summary_path)
    report_json = summary_path.parent / "interpretation_report.json" if summary_path else None
    report_md = summary_path.parent / "interpretation_report.md" if summary_path else None
    interpretation = load_json(report_json)
    results = summary.get("results", []) if summary else []
    return {
        "name": "videobench_window_prompt",
        "enabled": enabled,
        "exit_code": exit_code,
        "status": phase_status(enabled, exit_code, summary),
        "summary_path": (summary_path.as_posix() if summary_path and summary_path.exists() else None),
        "report_json_path": (report_json.as_posix() if report_json and report_json.exists() else None),
        "report_md_path": (report_md.as_posix() if report_md and report_md.exists() else None),
        "run_name": summary.get("run_name") if summary else None,
        "story_run_dir": summary.get("story_run_dir") if summary else None,
        "clips_dir": summary.get("clips_dir") if summary else None,
        "prompt_file": summary.get("prompt_file") if summary else None,
        "prompt_source": summary.get("prompt_source") if summary else None,
        "source_prompt_counts": summary.get("source_prompt_counts") if summary else None,
        "clip_count": summary.get("clip_count") if summary else None,
        "dimensions": summary.get("dimensions", []) if summary else [],
        "result_counts": summarize_results(results),
        "results": [
            {
                "dimension": row.get("dimension"),
                "status": row.get("status"),
                "mode": row.get("mode"),
                "score": row.get("score"),
                "score_available": row.get("score_available", False),
                "score_results_json": row.get("score_results_json"),
                "error_results_json": row.get("error_results_json"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "failure_marker_line": row.get("failure_marker_line", ""),
                "stdout_log": row.get("stdout_log"),
            }
            for row in results
        ],
        "sample_prompts": interpretation.get("sample_prompts", []) if interpretation else [],
        "total_elapsed_seconds": summary.get("total_elapsed_seconds") if summary else None,
    }


def build_continuity_phase(enabled: bool, exit_code: int, summary_path: Optional[Path]) -> Dict[str, Any]:
    summary = load_json(summary_path)
    report_json = summary_path.parent / "interpretation_report.json" if summary_path else None
    report_md = summary_path.parent / "interpretation_report.md" if summary_path else None
    interpretation = load_json(report_json)
    results = summary.get("results", []) if summary else []
    available_scores = interpretation.get("available_scores", {}) if interpretation else {}
    return {
        "name": "vbench_continuity",
        "enabled": enabled,
        "exit_code": exit_code,
        "status": phase_status(enabled, exit_code, summary),
        "summary_path": (summary_path.as_posix() if summary_path and summary_path.exists() else None),
        "report_json_path": (report_json.as_posix() if report_json and report_json.exists() else None),
        "report_md_path": (report_md.as_posix() if report_md and report_md.exists() else None),
        "run_name": summary.get("run_name") if summary else None,
        "videos_path": summary.get("videos_path") if summary else None,
        "source_videos_path": summary.get("source_videos_path") if summary else None,
        "sequence_mode": summary.get("sequence_mode") if summary else None,
        "combined_from_windows": summary.get("combined_from_windows") if summary else None,
        "source_window_count": summary.get("source_window_count") if summary else None,
        "input_videos": summary.get("input_videos") if summary else None,
        "dimensions": summary.get("dimensions", []) if summary else [],
        "result_counts": summarize_results(results),
        "available_scores": available_scores,
        "continuity_index_mean": interpretation.get("continuity_index_mean") if interpretation else None,
        "continuity_index_band": interpretation.get("continuity_index_band") if interpretation else None,
        "recommendations": interpretation.get("recommendations", []) if interpretation else [],
        "results": [
            {
                "dimension": row.get("dimension"),
                "status": row.get("status"),
                "score": available_scores.get(row.get("dimension")),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "failure_marker_line": row.get("failure_marker_line", ""),
            }
            for row in results
        ],
        "total_elapsed_seconds": summary.get("total_elapsed_seconds") if summary else None,
    }


def derive_overall_status(phases: List[Dict[str, Any]]) -> str:
    active = [phase for phase in phases if phase.get("enabled")]
    if not active:
        return "skipped"
    if any(phase.get("status") == "failed" for phase in active):
        return "failed"
    if any(phase.get("status") == "missing_summary" for phase in active):
        return "incomplete"
    if any(phase.get("status") == "ok" for phase in active):
        return "ok"
    if all(phase.get("status") == "dry_run" for phase in active):
        return "dry_run"
    return "skipped"


def derive_recommendations(videobench: Dict[str, Any], continuity: Dict[str, Any]) -> List[str]:
    items: List[str] = []
    if videobench.get("enabled") and videobench.get("status") == "failed":
        failed_dims = [row["dimension"] for row in videobench.get("results", []) if row.get("status") == "failed"]
        if failed_dims:
            items.append("Fix Video-Bench prompt metric failures first: {}.".format(", ".join(failed_dims)))
        else:
            items.append("Video-Bench prompt evaluation failed before results could be collected.")
    if continuity.get("enabled") and continuity.get("status") == "failed":
        failed_dims = [row["dimension"] for row in continuity.get("results", []) if row.get("status") == "failed"]
        if failed_dims:
            items.append("Fix VBench continuity failures first: {}.".format(", ".join(failed_dims)))
    for rec in continuity.get("recommendations", []):
        if rec not in items:
            items.append(rec)
    if videobench.get("enabled") and videobench.get("status") == "ok":
        items.append("Prompt-aligned metrics completed; review per-dimension stdout logs for qualitative failure modes.")
    if continuity.get("enabled") and continuity.get("status") == "ok" and not continuity.get("recommendations"):
        items.append("Continuity metrics completed cleanly; validate a few windows manually for narrative coherence.")
    if not items:
        items.append("No benchmark phase ran; verify launcher flags and inputs.")
    return items


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    videobench = payload["phases"][0]
    continuity = payload["phases"][1]
    lines: List[str] = []
    lines.append("# Combined Story Benchmark Report")
    lines.append("")
    lines.append(f"- Run: `{payload['run_name']}`")
    lines.append(f"- Story: `{payload['story_slug']}`")
    lines.append(f"- Eval target: `{payload['eval_target']}`")
    lines.append(f"- Overall status: `{payload['overall_status']}`")
    lines.append(f"- Created at: `{payload['created_at']}`")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    for phase in payload["phases"]:
        counts = phase.get("result_counts", {})
        lines.append(
            "- `{}`: status=`{}`, ok=`{}`, failed=`{}`, dry_run=`{}`, exit_code=`{}`".format(
                phase["name"],
                phase.get("status"),
                counts.get("ok", 0),
                counts.get("failed", 0),
                counts.get("dry_run", 0),
                phase.get("exit_code"),
            )
        )
        if phase.get("summary_path"):
            lines.append(f"  summary: `{phase['summary_path']}`")
        if phase.get("report_md_path"):
            lines.append(f"  report: `{phase['report_md_path']}`")
    lines.append("")
    lines.append("## Video-Bench Prompt Metrics")
    lines.append("")
    lines.append(f"- Status: `{videobench.get('status')}`")
    lines.append(f"- Story run dir: `{videobench.get('story_run_dir') or ''}`")
    lines.append(f"- Clips dir: `{videobench.get('clips_dir') or ''}`")
    lines.append(f"- Prompt source: `{videobench.get('prompt_source') or ''}`")
    lines.append(f"- Clip count: `{videobench.get('clip_count') if videobench.get('clip_count') is not None else 'n/a'}`")
    lines.append(f"- Source prompt counts: `{videobench.get('source_prompt_counts') or {}}`")
    lines.append("")
    for row in videobench.get("results", []):
        elapsed = row.get("elapsed_seconds")
        elapsed_txt = fmt_seconds(float(elapsed)) if elapsed is not None else "n/a"
        score = row.get("score")
        score_txt = f"{float(score):.4f}" if score is not None else "n/a"
        lines.append(
            f"- `{row.get('dimension')}`: status=`{row.get('status')}`, mode=`{row.get('mode')}`, score=`{score_txt}`, elapsed=`{elapsed_txt}`"
        )
        if row.get("score_results_json"):
            lines.append(f"  score json: `{row['score_results_json']}`")
        if not row.get("score_available", False):
            lines.append("  numeric score: `n/a` (upstream score file missing or empty)")
        if row.get("failure_marker_line"):
            lines.append(f"  failure: `{row['failure_marker_line']}`")
        if row.get("error_results_json"):
            lines.append(f"  error json: `{row['error_results_json']}`")
    if videobench.get("sample_prompts"):
        lines.append("")
        lines.append("Prompt samples:")
        for item in videobench.get("sample_prompts", [])[:3]:
            lines.append(
                f"- `window_{int(item.get('window_index', 0)):03d}` via `{item.get('prompt_source', '')}`: {str(item.get('prompt_text', ''))[:180]}"
            )
    lines.append("")
    lines.append("## VBench Continuity")
    lines.append("")
    lines.append(f"- Status: `{continuity.get('status')}`")
    lines.append(f"- Videos path: `{continuity.get('videos_path') or ''}`")
    lines.append(f"- Sequence mode: `{continuity.get('sequence_mode') or ''}`")
    lines.append(f"- Input videos: `{continuity.get('input_videos') if continuity.get('input_videos') is not None else 'n/a'}`")
    lines.append(f"- Combined from windows: `{continuity.get('combined_from_windows')}`")
    if continuity.get("continuity_index_mean") is not None:
        lines.append(
            f"- Continuity index: `{float(continuity['continuity_index_mean']):.4f}` ({continuity.get('continuity_index_band')})"
        )
    else:
        lines.append("- Continuity index: `n/a`")
    lines.append("")
    for row in continuity.get("results", []):
        elapsed = row.get("elapsed_seconds")
        elapsed_txt = fmt_seconds(float(elapsed)) if elapsed is not None else "n/a"
        score = row.get("score")
        score_txt = f"{float(score):.4f}" if score is not None else "n/a"
        lines.append(
            f"- `{row.get('dimension')}`: status=`{row.get('status')}`, score=`{score_txt}`, elapsed=`{elapsed_txt}`"
        )
        if row.get("failure_marker_line"):
            lines.append(f"  failure: `{row['failure_marker_line']}`")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    for item in payload.get("recommendations", []):
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a merged report across Video-Bench prompt metrics and VBench continuity.")
    parser.add_argument("--report_root", type=str, default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--story_slug", type=str, required=True)
    parser.add_argument("--run_name_base", type=str, default="")
    parser.add_argument("--eval_target", type=str, default="")
    parser.add_argument("--window_enabled", type=int, default=1)
    parser.add_argument("--continuity_enabled", type=int, default=1)
    parser.add_argument("--window_exit_code", type=int, default=0)
    parser.add_argument("--continuity_exit_code", type=int, default=0)
    parser.add_argument("--window_summary_path", type=str, default="")
    parser.add_argument("--continuity_summary_path", type=str, default="")
    args = parser.parse_args()

    report_root = Path(args.report_root).resolve()
    run_dir = report_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    window_summary_path = Path(args.window_summary_path).resolve() if args.window_summary_path else None
    continuity_summary_path = Path(args.continuity_summary_path).resolve() if args.continuity_summary_path else None

    videobench = build_videobench_phase(bool(args.window_enabled), int(args.window_exit_code), window_summary_path)
    continuity = build_continuity_phase(bool(args.continuity_enabled), int(args.continuity_exit_code), continuity_summary_path)
    phases = [videobench, continuity]

    payload = {
        "run_name": args.run_name,
        "run_name_base": args.run_name_base,
        "story_slug": args.story_slug,
        "eval_target": args.eval_target,
        "created_at": datetime.now().isoformat(),
        "overall_status": derive_overall_status(phases),
        "phases": phases,
    }
    payload["recommendations"] = derive_recommendations(videobench, continuity)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_md_path = run_dir / "comprehensive_report.md"
    write_markdown(report_md_path, payload)

    print(summary_path.as_posix())
    print(report_md_path.as_posix())

    if payload["overall_status"] in {"failed", "incomplete"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
