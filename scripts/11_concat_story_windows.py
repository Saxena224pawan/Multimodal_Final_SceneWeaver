#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple


WINDOW_FILE_RE = re.compile(r"^window_(\d+)\.(mp4|mov|mkv|avi)$", re.IGNORECASE)


def discover_window_clips(clips_dir: Path) -> List[Path]:
    if not clips_dir.exists() or not clips_dir.is_dir():
        return []
    windows: List[Tuple[int, Path]] = []
    for clip_path in clips_dir.iterdir():
        if not clip_path.is_file():
            continue
        match = WINDOW_FILE_RE.match(clip_path.name)
        if match is None:
            continue
        windows.append((int(match.group(1)), clip_path))
    windows.sort(key=lambda item: item[0])
    return [path for _, path in windows]


def write_concat_list(paths: Sequence[Path], list_path: Path) -> None:
    lines = []
    for clip_path in paths:
        escaped = clip_path.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'\n")
    list_path.write_text("".join(lines), encoding="utf-8")


def concat_with_imageio(paths: Sequence[Path], out_path: Path, log_path: Path) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is unavailable and Python imageio fallback could not be imported. "
            "Install imageio/imageio-ffmpeg."
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


def concat_windows(paths: Sequence[Path], out_path: Path, work_dir: Path, dry_run: bool) -> str:
    if len(paths) == 1:
        if not dry_run:
            shutil.copy2(paths[0], out_path)
        return "single_clip_copy"

    concat_list = work_dir / "concat_windows.txt"
    write_concat_list(paths, concat_list)
    if dry_run:
        return "dry_run"

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        py_concat_log = work_dir / "python_concat.log"
        concat_with_imageio(paths, out_path, py_concat_log)
        return "python_imageio"

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
        out_path.as_posix(),
    ]
    copy_log = work_dir / "ffmpeg_concat_copy.log"
    copy_proc = subprocess.run(
        copy_cmd,
        cwd=work_dir.as_posix(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
    )
    copy_log.write_text(copy_proc.stdout, encoding="utf-8")
    if copy_proc.returncode == 0 and out_path.exists():
        return "ffmpeg_copy"

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
        out_path.as_posix(),
    ]
    reencode_log = work_dir / "ffmpeg_concat_reencode.log"
    reencode_proc = subprocess.run(
        reencode_cmd,
        cwd=work_dir.as_posix(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
    )
    reencode_log.write_text(reencode_proc.stdout, encoding="utf-8")
    if reencode_proc.returncode == 0 and out_path.exists():
        return "ffmpeg_reencode"

    py_concat_log = work_dir / "python_concat_after_ffmpeg_failure.log"
    concat_with_imageio(paths, out_path, py_concat_log)
    if out_path.exists():
        return "python_imageio_after_ffmpeg_failure"

    raise RuntimeError(
        "Failed to concatenate window clips into a full-story video. "
        f"Check logs in {work_dir.as_posix()}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate generated SceneWeaver window clips into one story video.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing clips/window_*.mp4.")
    parser.add_argument("--clips-dir", default="", help="Optional explicit clips directory. Defaults to <run-dir>/clips.")
    parser.add_argument("--output-name", default="full_story.mp4", help="Output filename to write inside the run directory.")
    parser.add_argument("--output-path", default="", help="Optional explicit output video path. Overrides --output-name.")
    parser.add_argument("--work-dir", default="", help="Optional working directory for concat lists and logs.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect inputs and planned output without writing video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    clips_dir = Path(args.clips_dir).expanduser().resolve() if args.clips_dir else run_dir / "clips"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    window_paths = discover_window_clips(clips_dir)
    if not window_paths:
        raise RuntimeError(f"No window clips found in {clips_dir}")

    out_path = Path(args.output_path).expanduser().resolve() if args.output_path else (run_dir / args.output_name)
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else (out_path.parent / "_concatenated_story")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.dry_run:
        out_path.unlink()

    method = concat_windows(window_paths, out_path, work_dir, args.dry_run)
    summary = {
        "run_dir": run_dir.as_posix(),
        "clips_dir": clips_dir.as_posix(),
        "output_video": out_path.as_posix(),
        "output_exists": out_path.exists(),
        "window_count": len(window_paths),
        "concat_method": method,
    }
    summary_path = work_dir / "concat_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
