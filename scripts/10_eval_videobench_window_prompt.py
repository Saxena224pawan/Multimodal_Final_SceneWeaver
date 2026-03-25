import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_ROOT = ROOT / "outputs"
DEFAULT_REPORT_ROOT = ROOT / "outputs" / "reports" / "videobench_window_prompt"
DEFAULT_DIMENSIONS = [
    "video-text consistency",
    "action",
    "scene",
    "object_class",
    "color",
]
NONSTATIC_DIMENSIONS = {
    "video-text consistency",
    "color",
    "object_class",
    "scene",
    "action",
    "temporal_consistency",
    "motion_effects",
}
STATIC_DIMENSIONS = {
    "aesthetic_quality",
    "imaging_quality",
}
SUPPORTED_DIMENSIONS = NONSTATIC_DIMENSIONS | STATIC_DIMENSIONS
WINDOW_FILE_RE = re.compile(r"^window_(\d+)\.(mp4|mov|mkv|avi)$", re.IGNORECASE)
FAILURE_MARKERS = (
    "Traceback (most recent call last):",
    "ModuleNotFoundError:",
    "AuthenticationError",
    "OpenAIError",
    "No module named ",
    "Error evaluating ",
    "Error during evaluation",
    "Connection error.",
    "An error occurred: Connection error.",
)
PROMPT_SOURCE_FIELDS = {
    "generation_prompt": "generation_prompt",
    "refined_prompt": "refined_prompt",
    "prompt_seed": "prompt_seed",
    "beat": "beat",
}
PROMPT_SOURCE_ORDER = (
    "generation_prompt",
    "refined_prompt",
    "prompt_seed",
    "beat",
)


def parse_dimensions(raw: str) -> List[str]:
    dims = [item.strip() for item in raw.split(",") if item.strip()]
    if not dims:
        raise ValueError("No dimensions provided.")
    unsupported = [dim for dim in dims if dim not in SUPPORTED_DIMENSIONS]
    if unsupported:
        raise ValueError(
            "Unsupported Video-Bench dimensions: "
            f"{unsupported}. Supported: {sorted(SUPPORTED_DIMENSIONS)}"
        )
    return dims


def slugify(text: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (text or "").strip()).strip("._").lower()
    return cleaned or fallback


def fmt_seconds(total: float) -> str:
    sec = int(max(0, total))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def find_latest_story_run(outputs_root: Path) -> Path:
    candidates = sorted(
        [p for p in outputs_root.glob("story_run*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    for candidate in reversed(candidates):
        clips_dir = candidate / "clips"
        run_log_path = candidate / "run_log.jsonl"
        if not clips_dir.is_dir() or not run_log_path.is_file():
            continue
        has_window_clip = any(clips_dir.glob("window_*.mp4")) or any(clips_dir.glob("window_*.mov")) or any(clips_dir.glob("window_*.mkv")) or any(clips_dir.glob("window_*.avi"))
        if has_window_clip:
            return candidate
    raise RuntimeError(
        f"No story run with non-empty window clips and run_log.jsonl found under {outputs_root.as_posix()}"
    )


def resolve_story_run_layout(path: Path) -> Tuple[Path, Path]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved.as_posix()}")
    if resolved.is_dir() and resolved.name == "clips":
        return resolved.parent, resolved
    clips_dir = resolved / "clips"
    if resolved.is_dir() and clips_dir.is_dir():
        return resolved, clips_dir
    raise ValueError(
        "Expected either a story run directory containing clips/ or the clips/ directory itself, "
        f"got: {resolved.as_posix()}"
    )


def discover_window_clips(clips_dir: Path) -> List[Tuple[int, Path]]:
    windows: List[Tuple[int, Path]] = []
    for path in clips_dir.iterdir():
        if not path.is_file():
            continue
        match = WINDOW_FILE_RE.match(path.name)
        if match is None:
            continue
        windows.append((int(match.group(1)), path))
    windows.sort(key=lambda item: item[0])
    return windows


def load_jsonl_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.as_posix()} line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path.as_posix()} line {line_no}")
            rows.append(payload)
    return rows


def load_agent_metadata_rows(metadata_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for metadata_path in sorted(metadata_dir.glob("window_*.json")):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {metadata_path.as_posix()}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {metadata_path.as_posix()}")
        try:
            window_index = int(payload.get("window_idx"))
        except Exception:
            continue
        rows.append(
            {
                "window_index": window_index,
                "generation_prompt": payload.get("final_prompt", ""),
                "refined_prompt": payload.get("final_prompt", ""),
                "prompt_seed": payload.get("narrative_beat", ""),
                "beat": payload.get("narrative_beat", ""),
                "_source_file": metadata_path.as_posix(),
            }
        )
    return rows


def pick_prompt(row: dict, prompt_source: str) -> Tuple[str, str]:
    if prompt_source == "auto":
        for source_name in PROMPT_SOURCE_ORDER:
            value = normalize_prompt(str(row.get(PROMPT_SOURCE_FIELDS[source_name], "")))
            if value:
                return value, source_name
        return "", ""

    field = PROMPT_SOURCE_FIELDS[prompt_source]
    value = normalize_prompt(str(row.get(field, "")))
    return value, prompt_source


def collect_window_records(
    *,
    run_dir: Path,
    clips_dir: Path,
    prompt_source: str,
    skip_missing_prompts: bool,
) -> Tuple[List[dict], Dict[str, object]]:
    window_clips = discover_window_clips(clips_dir)
    if not window_clips:
        raise RuntimeError(f"No window clips found in {clips_dir.as_posix()}")

    run_log_path = run_dir / "run_log.jsonl"
    metadata_dir = run_dir / "metadata"
    source_meta_path: Path
    if run_log_path.is_file():
        rows = load_jsonl_rows(run_log_path)
        source_meta_path = run_log_path
    elif metadata_dir.is_dir():
        rows = load_agent_metadata_rows(metadata_dir)
        source_meta_path = metadata_dir
    else:
        raise FileNotFoundError(
            f"Neither run_log.jsonl nor metadata/window_*.json found in {run_dir.as_posix()}; cannot recover per-window prompts."
        )

    by_index = {}
    for row in rows:
        try:
            window_index = int(row.get("window_index"))
        except Exception:
            continue
        by_index[window_index] = row

    prepared = []
    missing_prompt_windows = []
    missing_log_windows = []
    source_counter: Counter[str] = Counter()

    for window_index, clip_path in window_clips:
        row = by_index.get(window_index)
        if row is None:
            missing_log_windows.append(window_index)
            continue

        prompt_text, used_source = pick_prompt(row=row, prompt_source=prompt_source)
        if not prompt_text:
            missing_prompt_windows.append(window_index)
            if skip_missing_prompts:
                continue
            continue

        source_counter[used_source] += 1
        prepared.append(
            {
                "window_index": window_index,
                "clip_path": clip_path.resolve().as_posix(),
                "prompt_text": prompt_text,
                "prompt_source": used_source,
                "beat": normalize_prompt(str(row.get("beat", ""))),
            }
        )

    if missing_log_windows:
        raise RuntimeError(
            "Missing prompt rows for window indices: "
            f"{missing_log_windows}. Expected one prompt record per window clip from run_log.jsonl or metadata/window_*.json."
        )

    if missing_prompt_windows and not skip_missing_prompts:
        raise RuntimeError(
            "Missing prompt text for window indices: "
            f"{missing_prompt_windows}. Try --prompt_source=beat or --skip_missing_prompts."
        )

    if not prepared:
        raise RuntimeError("No window records available for Video-Bench preparation.")

    meta = {
        "run_log_path": run_log_path.as_posix() if run_log_path.is_file() else "",
        "prompt_metadata_source": source_meta_path.as_posix(),
        "source_prompt_counts": dict(sorted(source_counter.items())),
        "total_window_clips": len(window_clips),
        "prepared_windows": len(prepared),
        "skipped_missing_prompts": len(missing_prompt_windows) if skip_missing_prompts else 0,
    }
    return prepared, meta


def materialize_video(src: Path, dst: Path, link_mode: str) -> str:
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if link_mode in {"auto", "symlink"}:
        try:
            dst.symlink_to(src.resolve())
            return "symlink"
        except OSError:
            if link_mode == "symlink":
                raise

    if link_mode in {"auto", "hardlink"}:
        try:
            os.link(src.resolve().as_posix(), dst.as_posix())
            return "hardlink"
        except OSError:
            if link_mode == "hardlink":
                raise

    shutil.copy2(src.as_posix(), dst.as_posix())
    return "copy"


def prepare_videobench_dataset(
    *,
    records: Sequence[dict],
    run_dir: Path,
    model_name: str,
    link_mode: str,
) -> Tuple[Path, Path, Path, dict]:
    prepared_root = run_dir / "_videobench_input"
    model_dir = prepared_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    prompt_map = {}
    manifest_rows = []
    link_counter: Counter[str] = Counter()

    for prompt_idx, record in enumerate(records):
        clip_src = Path(record["clip_path"])
        ext = clip_src.suffix.lower()
        target_name = f"{prompt_idx:04d}_window_{int(record['window_index']):03d}{ext}"
        target_path = model_dir / target_name
        materialized_as = materialize_video(src=clip_src, dst=target_path, link_mode=link_mode)
        link_counter[materialized_as] += 1

        prompt_map[str(prompt_idx)] = str(record["prompt_text"])
        manifest_rows.append(
            {
                "prompt_index": prompt_idx,
                "window_index": int(record["window_index"]),
                "prompt_source": record["prompt_source"],
                "prompt_text": record["prompt_text"],
                "beat": record["beat"],
                "source_clip_path": clip_src.as_posix(),
                "prepared_clip_path": target_path.as_posix(),
                "materialized_as": materialized_as,
            }
        )

    prompt_file_path = prepared_root / "prompt_file.json"
    prompt_file_path.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_path = prepared_root / "manifest.json"
    manifest = {
        "created_at": datetime.now().isoformat(),
        "model_name": model_name,
        "prepared_root": prepared_root.as_posix(),
        "prompt_file": prompt_file_path.as_posix(),
        "clips": manifest_rows,
        "materialized_counts": dict(sorted(link_counter.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return prepared_root, model_dir, prompt_file_path, manifest


def resolve_mode(dimension: str, mode: str) -> str:
    if mode != "auto":
        return mode
    if dimension in STATIC_DIMENSIONS:
        return "custom_static"
    return "custom_nonstatic"


def build_videobench_command(
    *,
    videobench_bin: str,
    dimension: str,
    videos_path: Path,
    config_path: Path,
    model_name: str,
    prompt_file: Path,
    mode: str,
    extra_args: Sequence[str],
) -> List[str]:
    cmd = [
        videobench_bin,
        "--dimension",
        dimension,
        "--videos_path",
        videos_path.as_posix(),
        "--mode",
        mode,
        "--config_path",
        config_path.as_posix(),
        "--models",
        model_name,
        "--prompt_file",
        prompt_file.as_posix(),
    ]
    cmd.extend(extra_args)
    return cmd


def parse_extra_args(items: Sequence[str]) -> List[str]:
    extra = []
    for item in items:
        extra.extend(shlex.split(item))
    return extra

def parse_score_payload(path: Path) -> Tuple[Optional[float], dict, dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, {}, {}
    if not isinstance(payload, dict):
        return None, {}, {}
    average_scores = payload.get("average_scores") if isinstance(payload.get("average_scores"), dict) else {}
    scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
    numeric = []
    for value in average_scores.values():
        try:
            numeric.append(float(value))
        except Exception:
            continue
    aggregate_score = sum(numeric) / float(len(numeric)) if numeric else None
    return aggregate_score, average_scores, scores


def collect_dimension_artifacts(dim_dir: Path) -> dict:
    eval_dir = dim_dir / "evaluation_results"
    score_file = None
    error_file = None
    if eval_dir.is_dir():
        score_candidates = sorted(eval_dir.rglob("*_score_results.json"))
        error_candidates = sorted(eval_dir.rglob("*_error_results.json"))
        if score_candidates:
            score_file = score_candidates[0]
        if error_candidates:
            error_file = error_candidates[0]
    aggregate_score = None
    average_scores = {}
    scores = {}
    if score_file is not None:
        aggregate_score, average_scores, scores = parse_score_payload(score_file)
    return {
        "score": aggregate_score,
        "average_scores": average_scores,
        "scores": scores,
        "score_results_json": score_file.as_posix() if score_file is not None else None,
        "error_results_json": error_file.as_posix() if error_file is not None else None,
        "score_available": aggregate_score is not None,
    }


def generate_interpretation_report(run_dir: Path, summary: dict, manifest: dict) -> Tuple[Path, Path]:
    report = {
        "run_name": summary.get("run_name"),
        "created_at": datetime.now().isoformat(),
        "story_run_dir": summary.get("story_run_dir"),
        "clips_dir": summary.get("clips_dir"),
        "prepared_dataset_root": summary.get("prepared_dataset_root"),
        "prompt_file": summary.get("prompt_file"),
        "prompt_source": summary.get("prompt_source"),
        "source_prompt_counts": summary.get("source_prompt_counts"),
        "clip_count": summary.get("clip_count"),
        "dimensions": summary.get("dimensions"),
        "results": summary.get("results", []),
        "sample_prompts": manifest.get("clips", [])[:5],
    }
    report_json_path = run_dir / "interpretation_report.json"
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Video-Bench Window Prompt Evaluation")
    lines.append("")
    lines.append(f"- Run: `{summary.get('run_name', '')}`")
    lines.append(f"- Story run dir: `{summary.get('story_run_dir', '')}`")
    lines.append(f"- Clips dir: `{summary.get('clips_dir', '')}`")
    lines.append(f"- Prepared dataset root: `{summary.get('prepared_dataset_root', '')}`")
    lines.append(f"- Prompt source mode: `{summary.get('prompt_source', '')}`")
    lines.append(f"- Prompt file: `{summary.get('prompt_file', '')}`")
    lines.append(f"- Clip count: `{summary.get('clip_count', 0)}`")
    lines.append(f"- Source prompt counts: `{summary.get('source_prompt_counts', {})}`")
    lines.append("")
    lines.append("## Dimensions")
    lines.append("")
    for row in summary.get("results", []):
        score = row.get("score")
        score_txt = f"{float(score):.4f}" if score is not None else "n/a"
        lines.append(
            f"- `{row.get('dimension', '')}`: status=`{row.get('status', 'unknown')}`, "
            f"mode=`{row.get('mode', '')}`, score=`{score_txt}`, elapsed=`{fmt_seconds(float(row.get('elapsed_seconds', 0.0)))}`"
        )
        stdout_log = row.get("stdout_log")
        if stdout_log:
            lines.append(f"  stdout: `{stdout_log}`")
        score_json = row.get("score_results_json")
        if score_json:
            lines.append(f"  score json: `{score_json}`")
        if not row.get("score_available", False):
            lines.append("  numeric score: `n/a` (upstream score file missing or empty)")
        failure = row.get("failure_marker_line", "")
        if failure:
            lines.append(f"  failure marker: `{failure}`")
        error_json = row.get("error_results_json")
        if error_json:
            lines.append(f"  error json: `{error_json}`")
    lines.append("")
    lines.append("## Prompt Samples")
    lines.append("")
    for item in manifest.get("clips", [])[:5]:
        lines.append(
            f"- `window_{int(item.get('window_index', 0)):03d}` via `{item.get('prompt_source', '')}`: "
            f"{item.get('prompt_text', '')[:220]}"
        )
    lines.append("")

    report_md_path = run_dir / "interpretation_report.md"
    report_md_path.write_text("\n".join(lines), encoding="utf-8")
    return report_json_path, report_md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare per-window story clips plus prompts for Video-Bench and optionally run the benchmark."
    )
    parser.add_argument(
        "--videos_path",
        type=str,
        default="",
        help=(
            "Story run directory containing clips/ and run_log.jsonl, or the clips/ directory itself. "
            "If empty, the latest outputs/story_run*/ directory is used."
        ),
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default=DEFAULT_OUTPUTS_ROOT.as_posix(),
        help="Root used when auto-selecting the latest story run.",
    )
    parser.add_argument(
        "--report_root",
        type=str,
        default=DEFAULT_REPORT_ROOT.as_posix(),
        help="Directory to store prepared Video-Bench data, logs, and summaries.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run label. Default: videobench_window_prompt_YYYYmmdd_HHMMSS",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        default=",".join(DEFAULT_DIMENSIONS),
        help="Comma-separated Video-Bench dimensions to run.",
    )
    parser.add_argument(
        "--prompt_source",
        type=str,
        default="auto",
        choices=["auto", "generation_prompt", "refined_prompt", "prompt_seed", "beat"],
        help=(
            "Prompt source used for each window. "
            "auto prefers generation_prompt -> refined_prompt -> prompt_seed -> beat."
        ),
    )
    parser.add_argument(
        "--skip_missing_prompts",
        action="store_true",
        help="Skip windows that do not resolve a prompt instead of failing.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "custom_nonstatic", "custom_static"],
        help="Video-Bench mode. auto chooses custom_nonstatic or custom_static per dimension.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Single model label exposed to Video-Bench. Default derives from the story run directory name.",
    )
    parser.add_argument(
        "--link_mode",
        type=str,
        default="auto",
        choices=["auto", "symlink", "hardlink", "copy"],
        help="How to materialize clips into the prepared Video-Bench input directory.",
    )
    parser.add_argument(
        "--videobench_bin",
        type=str,
        default="videobench",
        help="Video-Bench executable name/path.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Video-Bench config.json path containing the required API credentials.",
    )
    parser.add_argument(
        "--extra_arg",
        action="append",
        default=[],
        help="Extra argument forwarded to Video-Bench (repeatable).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Prepare inputs and print commands only. Do not execute Video-Bench.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).resolve()
    report_root = Path(args.report_root).resolve()
    run_name = args.run_name.strip() or f"videobench_window_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = report_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dimensions = parse_dimensions(args.dimensions)
    extra_args = parse_extra_args(args.extra_arg)

    if args.videos_path.strip():
        story_run_dir, clips_dir = resolve_story_run_layout(Path(args.videos_path))
    else:
        story_run_dir = find_latest_story_run(outputs_root)
        _, clips_dir = resolve_story_run_layout(story_run_dir)

    records, prompt_meta = collect_window_records(
        run_dir=story_run_dir,
        clips_dir=clips_dir,
        prompt_source=args.prompt_source,
        skip_missing_prompts=bool(args.skip_missing_prompts),
    )

    model_name = slugify(args.model_name.strip() or story_run_dir.name, fallback="sceneweaver_windows")
    prepared_dataset_root, prepared_model_dir, prompt_file_path, manifest = prepare_videobench_dataset(
        records=records,
        run_dir=run_dir,
        model_name=model_name,
        link_mode=args.link_mode,
    )

    if not args.dry_run:
        config_path = Path(args.config_path).expanduser().resolve() if args.config_path.strip() else None
        if config_path is None or not config_path.is_file():
            raise FileNotFoundError(
                "Video-Bench config.json is required when not using --dry_run. "
                "Set --config_path or VIDEOBENCH_CONFIG_PATH in the launcher."
            )
        if shutil.which(args.videobench_bin) is None:
            raise RuntimeError(
                f"Could not find Video-Bench executable '{args.videobench_bin}'. "
                "Install Video-Bench and ensure the executable is in PATH."
            )
    else:
        config_path = Path(args.config_path).expanduser().resolve() if args.config_path.strip() else (run_dir / "config.json")

    print(f"run_name={run_name}")
    print(f"story_run_dir={story_run_dir.as_posix()}")
    print(f"clips_dir={clips_dir.as_posix()}")
    print(f"prepared_dataset_root={prepared_dataset_root.as_posix()}")
    print(f"prepared_model_dir={prepared_model_dir.as_posix()}")
    print(f"prompt_file={prompt_file_path.as_posix()}")
    print(f"model_name={model_name}")
    print(f"prompt_source={args.prompt_source}")
    print(f"source_prompt_counts={json.dumps(prompt_meta['source_prompt_counts'], sort_keys=True)}")
    print(f"clip_count={len(records)}")
    print(f"dimensions={','.join(dimensions)}")
    print(f"videobench_bin={args.videobench_bin}")
    print(f"config_path={config_path.as_posix()}")

    results = []
    run_start = time.monotonic()
    total_dims = len(dimensions)

    for idx, dimension in enumerate(dimensions, start=1):
        dim_slug = slugify(dimension, fallback=f"dim_{idx:02d}")
        dim_dir = run_dir / dim_slug
        dim_dir.mkdir(parents=True, exist_ok=True)
        resolved_mode = resolve_mode(dimension=dimension, mode=args.mode)
        cmd = build_videobench_command(
            videobench_bin=args.videobench_bin,
            dimension=dimension,
            videos_path=prepared_dataset_root,
            config_path=config_path,
            model_name=model_name,
            prompt_file=prompt_file_path,
            mode=resolved_mode,
            extra_args=extra_args,
        )
        cmd_str = " ".join(shlex.quote(item) for item in cmd)
        print(f"[{idx}/{total_dims}] start dimension={dimension} mode={resolved_mode}")

        if args.dry_run:
            results.append(
                {
                    "dimension": dimension,
                    "mode": resolved_mode,
                    "status": "dry_run",
                    "command": cmd,
                    "command_str": cmd_str,
                    "workdir": dim_dir.as_posix(),
                }
            )
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
                print(f"[{dimension}] {line}", end="")
                if (not saw_failure_marker) and any(marker in line for marker in FAILURE_MARKERS):
                    saw_failure_marker = True
                    failure_line = line.strip()
            returncode = proc.wait()

        stderr_log_path.write_text("stderr merged into stdout.log\n", encoding="utf-8")
        elapsed = time.monotonic() - dim_start
        avg_dim = (time.monotonic() - run_start) / max(idx, 1)
        eta_remaining = avg_dim * max(total_dims - idx, 0)
        artifacts = collect_dimension_artifacts(dim_dir)
        results.append(
            {
                "dimension": dimension,
                "mode": resolved_mode,
                "status": "ok" if (returncode == 0 and not saw_failure_marker) else "failed",
                "returncode": returncode,
                "elapsed_seconds": elapsed,
                "saw_failure_marker": saw_failure_marker,
                "failure_marker_line": failure_line,
                "command": cmd,
                "command_str": cmd_str,
                "workdir": dim_dir.as_posix(),
                "stdout_log": stdout_log_path.as_posix(),
                "stderr_log": stderr_log_path.as_posix(),
                **artifacts,
            }
        )
        print(
            f"[{idx}/{total_dims}] done dimension={dimension} "
            f"status={results[-1]['status']} elapsed={fmt_seconds(elapsed)} "
            f"eta_remaining={fmt_seconds(eta_remaining)}"
        )

    summary = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "story_run_dir": story_run_dir.as_posix(),
        "clips_dir": clips_dir.as_posix(),
        "prepared_dataset_root": prepared_dataset_root.as_posix(),
        "prepared_model_dir": prepared_model_dir.as_posix(),
        "prompt_file": prompt_file_path.as_posix(),
        "prompt_source": args.prompt_source,
        "source_prompt_counts": prompt_meta["source_prompt_counts"],
        "clip_count": len(records),
        "dimensions": dimensions,
        "model_name": model_name,
        "link_mode": args.link_mode,
        "videobench_bin": args.videobench_bin,
        "config_path": config_path.as_posix(),
        "results": results,
        "total_elapsed_seconds": time.monotonic() - run_start,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_json_path, report_md_path = generate_interpretation_report(
        run_dir=run_dir,
        summary=summary,
        manifest=manifest,
    )

    ok_count = sum(1 for row in results if row.get("status") == "ok")
    failed_count = sum(1 for row in results if row.get("status") == "failed")
    if args.dry_run:
        print(f"[dry-run] planned {len(results)} Video-Bench jobs")
        print(summary_path.as_posix())
        print(report_json_path.as_posix())
        print(report_md_path.as_posix())
        return

    print(
        f"completed_ok={ok_count}/{len(results)} failed={failed_count} "
        f"total_elapsed={fmt_seconds(float(summary['total_elapsed_seconds']))}"
    )
    print(summary_path.as_posix())
    print(report_json_path.as_posix())
    print(report_md_path.as_posix())
    if failed_count > 0 or ok_count == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()