from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_story_batch.py"
SPEC = importlib.util.spec_from_file_location("story_batch_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
story_batch_runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(story_batch_runner)


def test_story_batch_json_loader_accepts_strings_and_objects() -> None:
    with TemporaryDirectory() as tmpdir:
        batch_path = Path(tmpdir) / "stories.json"
        batch_path.write_text(
            json.dumps(
                [
                    "A fox steals the moonlight from a village.",
                    {"storyline": "A child rebuilds a clock tower before dawn."},
                ]
            ),
            encoding="utf-8",
        )
        stories = story_batch_runner._load_storylines_json(batch_path.as_posix())

    assert stories == [
        "A fox steals the moonlight from a village.",
        "A child rebuilds a clock tower before dawn.",
    ]


def test_story_batch_text_loader_splits_blank_line_blocks() -> None:
    with TemporaryDirectory() as tmpdir:
        batch_path = Path(tmpdir) / "stories.txt"
        batch_path.write_text(
            "Story one line a.\nStory one line b.\n\nStory two line a.\nStory two line b.",
            encoding="utf-8",
        )
        stories = story_batch_runner._load_storylines_file(batch_path.as_posix())

    assert stories == [
        "Story one line a.\nStory one line b.",
        "Story two line a.\nStory two line b.",
    ]


def test_story_batch_selection_and_command_builder() -> None:
    selected = story_batch_runner._select_storylines([f"story {idx}" for idx in range(8)], max_stories=6)
    standard_cmd = story_batch_runner._build_story_run_command(
        runner="standard",
        runner_script=Path("/tmp/run_story_pipeline.py"),
        storyline="A lantern guides two siblings home.",
        output_dir=Path("/tmp/out_standard"),
        forwarded_args=["--dry_run"],
    )
    agents_cmd = story_batch_runner._build_story_run_command(
        runner="agents",
        runner_script=Path("/tmp/run_story_pipeline_with_agents.py"),
        storyline="A lantern guides two siblings home.",
        output_dir=Path("/tmp/out_agents"),
        forwarded_args=["--reference-conditioning"],
    )

    assert len(selected) == 6
    assert "--output_dir" in standard_cmd
    assert "--output-dir" in agents_cmd
    assert standard_cmd[0]
    assert agents_cmd[0]
