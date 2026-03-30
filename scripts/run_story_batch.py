#!/usr/bin/env python3
"""Batch runner for generating up to 6 stories in one command."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _normalize_story_entry(item: Any, index: int) -> str:
    if isinstance(item, str):
        text = item.strip()
    elif isinstance(item, dict):
        text = ""
        for key in ("storyline", "story", "text", "prompt"):
            candidate = str(item.get(key, "")).strip()
            if candidate:
                text = candidate
                break
    else:
        text = ""
    if not text:
        raise ValueError(
            f"Invalid story entry at index {index}: expected a non-empty string or object with storyline text"
        )
    return text


def _load_storylines_json(path: str) -> List[str]:
    batch_path = Path(path)
    if not batch_path.is_file():
        raise FileNotFoundError(f"storylines json not found: {batch_path}")
    with batch_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("storylines json must be a JSON array")
    return [_normalize_story_entry(item, index) for index, item in enumerate(payload)]


def _load_storylines_file(path: str) -> List[str]:
    batch_path = Path(path)
    if not batch_path.is_file():
        raise FileNotFoundError(f"storylines file not found: {batch_path}")
    text = batch_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
    if len(blocks) > 1:
        return blocks
    return [line.strip() for line in text.splitlines() if line.strip()]


def _select_storylines(stories: List[str], max_stories: int) -> List[str]:
    if max_stories <= 0:
        raise ValueError("max_stories must be a positive integer")
    return stories[:max_stories]


def _story_slug(storyline: str, index: int) -> str:
    title = storyline.splitlines()[0].split(".")[0].strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", title).strip("_")
    fallback = f"story_{index + 1:02d}"
    return f"{index + 1:02d}_{(cleaned[:48] or fallback)}"


def _runner_script_path(runner: str) -> Path:
    if runner == "agents":
        return PROJECT_ROOT / "scripts" / "run_story_pipeline_with_agents.py"
    return PROJECT_ROOT / "scripts" / "run_story_pipeline.py"


def _output_flag(runner: str) -> str:
    return "--output-dir" if runner == "agents" else "--output_dir"


def _validate_forwarded_args(forwarded_args: List[str]) -> None:
    blocked = {
        "--storyline",
        "--output_dir",
        "--output-dir",
    }
    conflicting = [arg for arg in forwarded_args if arg in blocked]
    if conflicting:
        rendered = ", ".join(conflicting)
        raise ValueError(
            f"Do not pass {rendered} through the batch runner; it sets per-story storyline and output directory automatically."
        )


def _build_story_run_command(
    *,
    runner: str,
    runner_script: Path,
    storyline: str,
    output_dir: Path,
    forwarded_args: List[str],
) -> List[str]:
    return [
        sys.executable,
        runner_script.as_posix(),
        "--storyline",
        storyline,
        _output_flag(runner),
        output_dir.as_posix(),
        *forwarded_args,
    ]


def _parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run the story pipeline for up to 6 storylines in one batch command."
    )
    parser.add_argument(
        "--storylines-json",
        default="",
        help="JSON array of story strings or objects with a 'storyline' field.",
    )
    parser.add_argument(
        "--storylines-file",
        default="",
        help="Plain text file with one story per line or blank-line-separated story blocks.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/story_batch",
        help="Parent output directory for the batch run.",
    )
    parser.add_argument(
        "--runner",
        default="standard",
        choices=["standard", "agents"],
        help="Which underlying story pipeline to run.",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=6,
        help="Maximum number of stories to generate in this batch run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue generating remaining stories if one story fails.",
    )
    return parser.parse_known_args()


def _load_requested_stories(args: argparse.Namespace) -> List[str]:
    if args.storylines_json and args.storylines_file:
        raise ValueError("Provide only one of --storylines-json or --storylines-file")
    if args.storylines_json:
        stories = _load_storylines_json(args.storylines_json)
    elif args.storylines_file:
        stories = _load_storylines_file(args.storylines_file)
    else:
        raise ValueError("Provide --storylines-json or --storylines-file for batch generation")
    selected = _select_storylines(stories, int(args.max_stories))
    if not selected:
        raise ValueError("No usable storylines found in the batch input")
    return selected


def main() -> int:
    args, forwarded_args = _parse_args()
    _validate_forwarded_args(forwarded_args)
    stories = _load_requested_stories(args)
    runner_script = _runner_script_path(args.runner)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    failure_count = 0
    for story_index, storyline in enumerate(stories):
        story_slug = _story_slug(storyline, story_index)
        story_output_dir = output_root / story_slug
        story_output_dir.mkdir(parents=True, exist_ok=True)
        command = _build_story_run_command(
            runner=args.runner,
            runner_script=runner_script,
            storyline=storyline,
            output_dir=story_output_dir,
            forwarded_args=forwarded_args,
        )
        print(f"[batch] {story_index + 1}/{len(stories)} runner={args.runner} output={story_output_dir}")
        completed = subprocess.run(command, check=False)
        result = {
            "story_index": story_index,
            "story_slug": story_slug,
            "storyline_preview": storyline[:160],
            "output_dir": story_output_dir.as_posix(),
            "command": command,
            "exit_code": int(completed.returncode),
            "ok": completed.returncode == 0,
        }
        results.append(result)
        if completed.returncode != 0:
            failure_count += 1
            print(f"[batch] story {story_index + 1} failed with exit code {completed.returncode}")
            if not args.continue_on_error:
                break

    summary = {
        "runner": args.runner,
        "max_stories": int(args.max_stories),
        "stories_requested": len(stories),
        "stories_attempted": len(results),
        "stories_succeeded": sum(1 for item in results if item["ok"]),
        "stories_failed": failure_count,
        "continue_on_error": bool(args.continue_on_error),
        "forwarded_args": forwarded_args,
        "results": results,
    }
    summary_path = output_root / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[batch] summary: {summary_path}")
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
