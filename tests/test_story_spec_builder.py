from __future__ import annotations

from LLM_MODEL.story_spec_builder import StorySpecBuilder, StorySpecBuilderConfig
from LLM_MODEL.story_planner import validate_story_spec


def test_story_spec_builder_heuristic_mode_produces_valid_spec() -> None:
    builder = StorySpecBuilder(StorySpecBuilderConfig(model_id=None))
    builder.load()

    spec = builder.build_from_storyline(
        storyline=(
            "Mira rushes through the station with a sealed letter. "
            "At the departure board she finds Arjun waiting near the last train. "
            "They argue, then reconcile as the doors close."
        ),
        runtime_seconds=36,
        window_seconds=4,
    )

    errors = validate_story_spec(spec)
    assert errors == []
    assert builder.last_build_mode == "heuristic"
    assert len(spec["characters"]) >= 1
    assert len(spec["objects"]) >= 1
    assert len(spec["beats"]) >= 3
    assert "visual_features" in spec["characters"][0]
    assert spec["characters"][0]["size_bucket"] == "medium"
    assert "size_bucket" in spec["objects"][0]