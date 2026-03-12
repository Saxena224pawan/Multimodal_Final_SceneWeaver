import argparse
import json
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_ROOT = ROOT / "outputs"
DEFAULT_REPORT_ROOT = ROOT / "outputs" / "reports" / "vbench_continuity"

# Per VBench custom_input support, temporal_flickering is not always available.
DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "temporal_flickering",
]
CUSTOM_INPUT_SUPPORTED = {
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
}


def parse_dimensions(raw: str) -> List[str]:
    dims = [d.strip() for d in raw.split(",") if d.strip()]
    if not dims:
        raise ValueError("No dimensions provided.")
    return dims


def find_latest_story_run(outputs_root: Path) -> Path:
    candidates = sorted(
        [p for p in outputs_root.glob("story_run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"No story runs found under {outputs_root.as_posix()}")
    latest = candidates[-1]
    clips_dir = latest / "clips"
    if clips_dir.is_dir():
        return clips_dir
    return latest


def ensure_videos_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"videos_path does not exist: {path.as_posix()}")
    if path.is_file() and path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise ValueError(f"Expected a video file for videos_path, got: {path.as_posix()}")
    return path


def build_vbench_command(
    vbench_bin: str,
    dimension: str,
    videos_path: Path,
    mode: str,
    ngpus: int,
    extra_args: Sequence[str],
) -> List[str]:
    cmd = [
        vbench_bin,
        "evaluate",
        "--dimension",
        dimension,
        "--videos_path",
        videos_path.as_posix(),
        "--mode",
        mode,
        "--ngpus",
        str(ngpus),
    ]
    cmd.extend(extra_args)
    return cmd


WINDOW_FILE_RE = re.compile(r"^window_(\d+)\.(mp4|mov|mkv|avi)$", re.IGNORECASE)


def discover_window_clips(videos_path: Path) -> List[Path]:
    if not videos_path.is_dir():
        return []
    windows = []
    for p in videos_path.iterdir():
        if not p.is_file():
            continue
        m = WINDOW_FILE_RE.match(p.name)
        if m is None:
            continue
        windows.append((int(m.group(1)), p))
    windows.sort(key=lambda x: x[0])
    return [p for _, p in windows]


def _write_concat_list(paths: Sequence[Path], list_path: Path) -> None:
    lines = []
    for p in paths:
        # ffmpeg concat format; escape single quotes for shell-safe parsing.
        escaped = p.resolve().as_posix().replace("'", "'\\''")
        lines.append("file '{}'\n".format(escaped))
    list_path.write_text("".join(lines), encoding="utf-8")


def _concat_with_imageio(paths: Sequence[Path], out_path: Path, log_path: Path) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is unavailable and Python imageio fallback could not be imported. "
            "Install imageio/imageio-ffmpeg or set --sequence_mode per_clip."
        ) from exc

    fps = 8.0
    first_reader = imageio.get_reader(paths[0].as_posix())
    try:
        meta = first_reader.get_meta_data()
        fps = float(meta.get("fps", fps))
    except Exception:
        fps = 8.0
    finally:
        first_reader.close()

    written = 0
    with imageio.get_writer(out_path.as_posix(), fps=fps) as writer:
        for clip_path in paths:
            reader = imageio.get_reader(clip_path.as_posix())
            try:
                for frame in reader:
                    writer.append_data(frame)
                    written += 1
            finally:
                reader.close()
    log_path.write_text(
        f"python_imageio_concat=1\noutput={out_path.as_posix()}\nclips={len(paths)}\nframes={written}\nfps={fps}\n",
        encoding="utf-8",
    )


def prepare_sequential_input(
    source_path: Path,
    run_dir: Path,
    sequence_mode: str,
    dry_run: bool,
) -> Tuple[Path, Dict[str, object]]:
    meta = {
        "combined": False,
        "source_video_count": count_input_videos(source_path),
        "source_window_count": 0,
    }
    if sequence_mode != "concat_windows":
        return source_path, meta
    if source_path.is_file():
        return source_path, meta
    if not source_path.is_dir():
        return source_path, meta

    window_paths = discover_window_clips(source_path)
    meta["source_window_count"] = len(window_paths)
    if len(window_paths) < 2:
        return source_path, meta

    prep_dir = run_dir / "_prepared_input"
    prep_dir.mkdir(parents=True, exist_ok=True)
    concat_list = prep_dir / "concat_windows.txt"
    combined_video = prep_dir / "sequential_video.mp4"
    _write_concat_list(window_paths, concat_list)
    meta["combined"] = True

    if dry_run:
        return combined_video, meta

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        py_concat_log = prep_dir / "python_concat.log"
        _concat_with_imageio(window_paths, combined_video, py_concat_log)
        return combined_video, meta

    copy_cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list.as_posix(),
        "-c",
        "copy",
        combined_video.as_posix(),
    ]
    copy_log = prep_dir / "ffmpeg_concat_copy.log"
    copy_proc = subprocess.run(
        copy_cmd,
        cwd=prep_dir.as_posix(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
    )
    copy_log.write_text(copy_proc.stdout, encoding="utf-8")
    if copy_proc.returncode == 0 and combined_video.exists():
        return combined_video, meta

    reencode_cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list.as_posix(),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        combined_video.as_posix(),
    ]
    reencode_log = prep_dir / "ffmpeg_concat_reencode.log"
    reencode_proc = subprocess.run(
        reencode_cmd,
        cwd=prep_dir.as_posix(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
    )
    reencode_log.write_text(reencode_proc.stdout, encoding="utf-8")
    if reencode_proc.returncode == 0 and combined_video.exists():
        return combined_video, meta

    py_concat_log = prep_dir / "python_concat_after_ffmpeg_failure.log"
    _concat_with_imageio(window_paths, combined_video, py_concat_log)
    if combined_video.exists():
        return combined_video, meta

    raise RuntimeError(
        "Failed to concatenate window clips into sequential video. "
        f"Check logs: {copy_log.as_posix()}, {reencode_log.as_posix()}, and {py_concat_log.as_posix()}"
    )


def count_input_videos(path: Path) -> int:
    if path.is_file():
        return 1
    exts = ("*.mp4", "*.mov", "*.mkv", "*.avi")
    count = 0
    for ext in exts:
        count += len(list(path.rglob(ext)))
    return count


def fmt_seconds(total: float) -> str:
    sec = int(max(0, total))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


FAILURE_MARKERS = (
    "Traceback (most recent call last):",
    "ChildFailedError",
    "_pickle.UnpicklingError",
    "NotImplementedError:",
    "No module named ",
)


def score_band(score: float) -> str:
    if score >= 0.95:
        return "excellent"
    if score >= 0.90:
        return "good"
    if score >= 0.80:
        return "fair"
    return "poor"


def dimension_hint(dimension: str) -> str:
    hints = {
        "subject_consistency": "Higher is better. Measures identity/subject stability over time.",
        "background_consistency": "Higher is better. Measures scene/layout consistency over time.",
        "motion_smoothness": "Higher is better. Measures temporal smoothness and reduced motion artifacts.",
        "temporal_flickering": "Higher is better. Measures lower frame-to-frame flicker.",
    }
    return hints.get(dimension, "Higher is better.")


def parse_dimension_eval(dim_dir: Path, dimension: str) -> tuple:
    eval_candidates = sorted(
        list((dim_dir / "evaluation_results").glob("*_eval_results.json")),
        key=lambda p: p.stat().st_mtime,
    )
    if not eval_candidates:
        return None, [], None

    eval_path = eval_candidates[-1]
    try:
        data = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception:
        return None, [], eval_path

    payload = data.get(dimension)
    if not isinstance(payload, list) or len(payload) == 0:
        return None, [], eval_path

    score = None
    try:
        score = float(payload[0])
    except Exception:
        score = None

    per_video = []
    if len(payload) > 1 and isinstance(payload[1], list):
        for item in payload[1]:
            if not isinstance(item, dict):
                continue
            try:
                vr = float(item.get("video_results"))
            except Exception:
                continue
            per_video.append(
                {
                    "video_path": str(item.get("video_path", "")),
                    "video_results": vr,
                }
            )
    per_video.sort(key=lambda x: x["video_results"])
    return score, per_video, eval_path


def generate_interpretation_report(run_dir: Path, summary: dict) -> tuple:
    continuity_dims = [
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "temporal_flickering",
    ]
    result_rows = []
    available_scores = {}
    failed_dims = []

    for row in summary.get("results", []):
        dimension = row.get("dimension", "")
        status = row.get("status", "unknown")
        dim_dir = run_dir / dimension
        score, per_video, eval_path = parse_dimension_eval(dim_dir=dim_dir, dimension=dimension)

        if status != "ok":
            failed_dims.append(dimension)

        if score is not None and dimension in continuity_dims:
            available_scores[dimension] = score

        result_rows.append(
            {
                "dimension": dimension,
                "status": status,
                "score": score,
                "score_band": (score_band(score) if score is not None else None),
                "hint": dimension_hint(dimension),
                "eval_results_json": (eval_path.as_posix() if eval_path is not None else None),
                "worst_videos": per_video[:3],
                "failure_marker_line": row.get("failure_marker_line", ""),
            }
        )

    continuity_index = None
    if available_scores:
        continuity_index = sum(available_scores.values()) / float(len(available_scores))

    recommendations = []
    if failed_dims:
        recommendations.append(
            f"Fix failed dimensions first: {', '.join(sorted(set(failed_dims)))}."
        )
    if available_scores.get("subject_consistency", 1.0) < 0.90:
        recommendations.append("Improve identity lock and cross-window character anchors.")
    if available_scores.get("background_consistency", 1.0) < 0.90:
        recommendations.append("Strengthen environment anchor and scene-layout constraints.")
    if available_scores.get("motion_smoothness", 1.0) < 0.95:
        recommendations.append("Increase temporal stability (fps/steps) and avoid abrupt motion jumps.")
    if available_scores.get("temporal_flickering", 1.0) < 0.95:
        recommendations.append("Tune denoising/seed continuity to reduce frame flicker.")
    if not recommendations:
        recommendations.append("Continuity metrics look strong; validate with qualitative spot checks.")

    report = {
        "run_name": summary.get("run_name"),
        "created_at": datetime.now().isoformat(),
        "videos_path": summary.get("videos_path"),
        "input_videos": summary.get("input_videos"),
        "continuity_dimensions": continuity_dims,
        "available_scores": available_scores,
        "continuity_index_mean": continuity_index,
        "continuity_index_band": (score_band(continuity_index) if continuity_index is not None else None),
        "results": result_rows,
        "recommendations": recommendations,
    }

    report_json_path = run_dir / "interpretation_report.json"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = []
    md_lines.append("# VBench Continuity Interpretation")
    md_lines.append("")
    md_lines.append(f"- Run: `{summary.get('run_name', '')}`")
    md_lines.append(f"- Videos: `{summary.get('videos_path', '')}`")
    md_lines.append(f"- Input videos: `{summary.get('input_videos', 0)}`")
    md_lines.append(
        f"- Continuity index (mean): `{continuity_index:.4f}` ({score_band(continuity_index)})"
        if continuity_index is not None
        else "- Continuity index (mean): `n/a`"
    )
    md_lines.append("")
    md_lines.append("## Metrics")
    md_lines.append("")
    for row in result_rows:
        score_txt = f"{row['score']:.4f}" if row["score"] is not None else "n/a"
        band_txt = row["score_band"] if row["score_band"] is not None else "n/a"
        md_lines.append(
            f"- `{row['dimension']}`: status=`{row['status']}`, score=`{score_txt}`, band=`{band_txt}`"
        )
        md_lines.append(f"  {row['hint']}")
        if row["failure_marker_line"]:
            md_lines.append(f"  Failure marker: `{row['failure_marker_line']}`")
        if row["worst_videos"]:
            worst = row["worst_videos"][0]
            md_lines.append(
                f"  Worst clip: `{worst.get('video_path', '')}` score=`{worst.get('video_results', 0.0):.4f}`"
            )
    md_lines.append("")
    md_lines.append("## Recommendations")
    md_lines.append("")
    for item in recommendations:
        md_lines.append(f"- {item}")
    md_lines.append("")

    report_md_path = run_dir / "interpretation_report.md"
    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return report_json_path, report_md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VBench continuity-focused evaluation on generated videos.")
    parser.add_argument(
        "--videos_path",
        type=str,
        default="",
        help="Directory containing generated videos, or a single video file. If empty, latest outputs/story_run_*/clips is used.",
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default=DEFAULT_OUTPUTS_ROOT.as_posix(),
        help="Root used when auto-selecting latest story run.",
    )
    parser.add_argument(
        "--report_root",
        type=str,
        default=DEFAULT_REPORT_ROOT.as_posix(),
        help="Directory to store command logs and copied VBench artifacts.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run label. Default: vbench_continuity_YYYYmmdd_HHMMSS",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        default=",".join(DEFAULT_DIMENSIONS),
        help="Comma-separated VBench dimensions to run.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="custom_input",
        choices=["custom_input", "vbench_standard"],
        help="VBench mode.",
    )
    parser.add_argument(
        "--sequence_mode",
        type=str,
        default="concat_windows",
        choices=["concat_windows", "per_clip"],
        help=(
            "How to treat input directory clips. "
            "'concat_windows' combines window_*.mp4 into one sequential video before evaluation."
        ),
    )
    parser.add_argument("--vbench_bin", type=str, default="vbench", help="VBench executable name/path.")
    parser.add_argument("--ngpus", type=int, default=1, help="GPU count passed to VBench.")
    parser.add_argument(
        "--extra_arg",
        action="append",
        default=[],
        help="Extra argument forwarded to VBench (repeatable). Example: --extra_arg=--imaging_quality_preprocessing_mode",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and write plan file only. Do not execute VBench.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).resolve()
    report_root = Path(args.report_root).resolve()
    run_name = args.run_name.strip() or f"vbench_continuity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = report_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dimensions = parse_dimensions(args.dimensions)
    if args.mode == "custom_input":
        unsupported = [d for d in dimensions if d not in CUSTOM_INPUT_SUPPORTED]
        if unsupported:
            raise ValueError(
                "Unsupported dimensions for custom_input mode: "
                f"{unsupported}. Either remove them or use --mode=vbench_standard."
            )

    if args.videos_path.strip():
        source_videos_path = ensure_videos_path(Path(args.videos_path).resolve())
    else:
        source_videos_path = ensure_videos_path(find_latest_story_run(outputs_root))

    videos_path, prep_meta = prepare_sequential_input(
        source_path=source_videos_path,
        run_dir=run_dir,
        sequence_mode=args.sequence_mode,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        videos_path = ensure_videos_path(videos_path)

    if not args.dry_run and shutil.which(args.vbench_bin) is None:
        raise RuntimeError(
            f"Could not find VBench executable '{args.vbench_bin}'. "
            "Install VBench and ensure the executable is in PATH."
        )

    extra_args: List[str] = []
    for item in args.extra_arg:
        extra_args.extend(shlex.split(item))

    input_video_count = count_input_videos(videos_path) if videos_path.exists() else 1
    source_video_count = count_input_videos(source_videos_path)
    print(f"run_name={run_name}")
    print(f"source_videos_path={source_videos_path.as_posix()}")
    print(f"evaluation_videos_path={videos_path.as_posix()}")
    print(f"sequence_mode={args.sequence_mode}")
    print(f"input_videos={input_video_count}")
    if prep_meta.get("combined"):
        print(f"source_window_count={prep_meta.get('source_window_count', 0)}")
        if args.dry_run:
            print("dry-run: skipped ffmpeg concat execution.")
    else:
        print(f"source_video_count={source_video_count}")
    print(f"dimensions={','.join(dimensions)}")
    if not args.dry_run:
        print("ETA will be updated after the first dimension finishes.")

    results = []
    run_start = time.monotonic()
    total_dims = len(dimensions)
    for idx, dimension in enumerate(dimensions, start=1):
        dim_dir = run_dir / dimension
        dim_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_vbench_command(
            vbench_bin=args.vbench_bin,
            dimension=dimension,
            videos_path=videos_path,
            mode=args.mode,
            ngpus=args.ngpus,
            extra_args=extra_args,
        )
        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        print(f"[{idx}/{total_dims}] start dimension={dimension}")

        if args.dry_run:
            result = {
                "dimension": dimension,
                "status": "dry_run",
                "command": cmd,
                "command_str": cmd_str,
                "workdir": dim_dir.as_posix(),
            }
            results.append(result)
            continue

        dim_start = time.monotonic()
        stdout_log_path = dim_dir / "stdout.log"
        stderr_log_path = dim_dir / "stderr.log"
        with stdout_log_path.open("w", encoding="utf-8") as stdout_log:
            proc = subprocess.Popen(
                cmd,
                cwd=dim_dir.as_posix(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            saw_failure_marker = False
            failure_line = ""
            for line in proc.stdout:
                stdout_log.write(line)
                # Stream line immediately so SLURM logs show progress.
                print(f"[{dimension}] {line}", end="")
                if (not saw_failure_marker) and any(marker in line for marker in FAILURE_MARKERS):
                    saw_failure_marker = True
                    failure_line = line.strip()
            returncode = proc.wait()

        stderr_log_path.write_text("stderr merged into stdout.log\n", encoding="utf-8")
        dim_elapsed = time.monotonic() - dim_start
        done_count = idx
        avg_dim = (time.monotonic() - run_start) / max(done_count, 1)
        eta_remaining = avg_dim * max(total_dims - done_count, 0)

        result = {
            "dimension": dimension,
            "status": "ok" if (returncode == 0 and not saw_failure_marker) else "failed",
            "returncode": returncode,
            "saw_failure_marker": saw_failure_marker,
            "failure_marker_line": failure_line,
            "elapsed_seconds": dim_elapsed,
            "command": cmd,
            "command_str": cmd_str,
            "workdir": dim_dir.as_posix(),
            "stdout_log": stdout_log_path.as_posix(),
            "stderr_log": stderr_log_path.as_posix(),
        }
        results.append(result)
        print(
            f"[{idx}/{total_dims}] done dimension={dimension} "
            f"status={result['status']} elapsed={fmt_seconds(dim_elapsed)} "
            f"eta_remaining={fmt_seconds(eta_remaining)}"
        )

    summary = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "videos_path": videos_path.as_posix(),
        "source_videos_path": source_videos_path.as_posix(),
        "sequence_mode": args.sequence_mode,
        "combined_from_windows": bool(prep_meta.get("combined")),
        "source_window_count": int(prep_meta.get("source_window_count", 0)),
        "mode": args.mode,
        "dimensions": dimensions,
        "input_videos": input_video_count,
        "results": results,
        "total_elapsed_seconds": time.monotonic() - run_start,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    failed_count = sum(1 for r in results if r.get("status") == "failed")
    if args.dry_run:
        print(f"[dry-run] planned {len(results)} VBench jobs")
        print(summary_path.as_posix())
        return

    report_json_path, report_md_path = generate_interpretation_report(run_dir=run_dir, summary=summary)
    total_elapsed = summary.get("total_elapsed_seconds", 0.0)
    print(
        f"completed_ok={ok_count}/{len(results)} failed={failed_count} "
        f"total_elapsed={fmt_seconds(float(total_elapsed))}"
    )
    print(summary_path.as_posix())
    print(report_json_path.as_posix())
    print(report_md_path.as_posix())
    if failed_count > 0 or ok_count == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
