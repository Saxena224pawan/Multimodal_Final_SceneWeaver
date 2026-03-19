from __future__ import annotations

from LLM_MODEL.story_planner import build_scene_plan, validate_story_spec


def _base_story_spec() -> dict:
    return {
        "version": 1,
        "story_id": "test_story",
        "title": "Test Story",
        "logline": "A courier must deliver a key package before departure.",
        "runtime_seconds": 24,
        "characters": [
            {
                "id": "lana",
                "name": "Lana",
                "role": "protagonist",
                "traits": ["determined", "focused"],
                "visual_features": ["short_curl_hair", "yellow_raincoat"],
                "size_bucket": "medium",
            }
        ],
        "objects": [
            {
                "id": "parcel",
                "name": "Parcel",
                "category": "prop",
                "visual_traits": ["brown_wrap", "sealed_wax"],
                "size_bucket": "small",
            },
            {
                "id": "ticket_gate",
                "name": "Ticket Gate",
                "category": "set_piece",
                "visual_traits": ["red_lights"],
                "size_bucket": "large",
            },
        ],
        "world": {
            "setting": "rainy station concourse",
            "anchors": ["station_concourse", "ticket_gate"],
        },
        "beats": [
            {
                "id": "setup",
                "summary": "Lana enters the station while holding the parcel.",
                "objective": "establish urgency",
                "emotion": "focused",
                "duration_hint_sec": 12,
                "must_include_characters": ["lana"],
                "must_include_objects": ["parcel", "ticket_gate"],
                "character_overrides": {
                    "lana": {
                        "size_bucket": "large",
                        "visual_features": ["rain_damp_hair", "yellow_raincoat"],
                    }
                },
                "object_overrides": {
                    "parcel": {
                        "size_bucket": "tiny",
                        "visual_features": ["wet_wrapping", "sealed_wax"],
                    }
                },
                "location": "station_concourse",
            },
            {
                "id": "handoff",
                "summary": "Lana reaches the gate and delivers the parcel.",
                "objective": "complete delivery",
                "emotion": "relieved",
                "duration_hint_sec": 12,
                "must_include_characters": ["lana"],
                "must_include_objects": ["parcel", "ticket_gate"],
                "location": "ticket_gate",
            },
        ],
    }


def test_validate_story_spec_accepts_scene_entity_contract_fields() -> None:
    spec = _base_story_spec()
    errors = validate_story_spec(spec)
    assert errors == []


def test_validate_story_spec_rejects_unknown_override_ids() -> None:
    spec = _base_story_spec()
    spec["beats"][0]["character_overrides"] = {"unknown_character": {"size_bucket": "small"}}
    errors = validate_story_spec(spec)

    assert any(error["path"].endswith("character_overrides.unknown_character") for error in errors)
    assert any("unknown character id" in error["message"] for error in errors)


def test_validate_story_spec_rejects_invalid_size_bucket() -> None:
    spec = _base_story_spec()
    spec["objects"][0]["size_bucket"] = "giant"
    errors = validate_story_spec(spec)

    assert any(error["path"].endswith("objects[0].size_bucket") for error in errors)


def test_build_scene_plan_includes_scene_entity_contract_and_prompt_enforcement() -> None:
    spec = _base_story_spec()
    spec["beats"][0]["story_phase"] = "setup"
    spec["beats"][0]["character_progression"] = "Lana hides her worry behind focused motion while protecting the parcel"
    spec["beats"][0]["relationship_dynamic"] = "The ticking gate and the parcel pressure Lana's choices even before she speaks"
    spec["beats"][0]["visible_change"] = "Show Lana gripping the parcel tighter, scanning the gate, and moving with contained urgency"
    scene_plan = build_scene_plan(spec, window_seconds=4)

    assert scene_plan["input_contract"]["object_count"] == 2
    assert scene_plan["total_windows"] == 6

    first = scene_plan["windows"][0]
    contract = first["scene_entity_contract"]

    assert contract["count_rule"] == "exact_anchors"
    assert contract["size_system"] == "relative_buckets_v1"
    assert contract["character_count_exact"] == 1
    assert contract["object_count_exact"] == 2

    assert contract["characters"][0]["id"] == "lana"
    assert contract["characters"][0]["size_bucket"] == "large"
    assert contract["characters"][0]["visual_features"] == ["rain damp hair", "yellow raincoat"]

    assert contract["objects"][0]["id"] == "parcel"
    assert contract["objects"][0]["size_bucket"] == "tiny"
    assert contract["objects"][0]["visual_features"] == ["wet wrapping", "sealed wax"]
    assert first["story_phase"] == "setup"
    assert first["character_progression"] == "Lana hides her worry behind focused motion while protecting the parcel"
    assert first["relationship_dynamic"] == "The ticking gate and the parcel pressure Lana's choices even before she speaks"
    assert first["visible_change"] == "Show Lana gripping the parcel tighter, scanning the gate, and moving with contained urgency"

    assert "Require exactly 1 anchor character and exactly 2 anchor objects" in first["scene_prompt"]
    assert "Do not add extra named anchor characters or objects in this window." in first["scene_prompt"]
    assert "Character progression: Lana hides her worry behind focused motion while protecting the parcel" in first["scene_prompt"]
    assert "Visible change: Show Lana gripping the parcel tighter, scanning the gate, and moving with contained urgency" in first["expected_caption"]
    assert "count_rule=exact_anchors" in first["expected_caption"]


def test_build_scene_plan_backward_compatible_defaults() -> None:
    spec = {
        "version": 1,
        "story_id": "legacy_spec",
        "title": "Legacy Spec",
        "logline": "A person crosses a room and opens a door.",
        "runtime_seconds": 8,
        "characters": [
            {
                "id": "lead",
                "name": "Lead",
                "role": "protagonist",
                "traits": ["calm", "steady"],
            }
        ],
        "objects": [
            {
                "id": "main_door",
                "name": "Main Door",
                "category": "set_piece",
                "visual_traits": ["blue_paint"],
            }
        ],
        "world": {"setting": "small hallway", "anchors": ["small_hallway"]},
        "beats": [
            {
                "id": "single_beat",
                "summary": "Lead moves toward the door.",
                "objective": "show steady movement",
                "emotion": "focused",
                "duration_hint_sec": 8,
                "must_include_characters": ["lead"],
                "must_include_objects": ["main_door"],
            }
        ],
    }

    errors = validate_story_spec(spec)
    assert errors == []

    scene_plan = build_scene_plan(spec, window_seconds=4)
    first = scene_plan["windows"][0]
    contract = first["scene_entity_contract"]

    assert contract["characters"][0]["size_bucket"] == "medium"
    assert contract["characters"][0]["visual_features"] == ["calm", "steady"]
    assert contract["objects"][0]["size_bucket"] == "large"
    assert contract["objects"][0]["visual_features"] == ["blue paint"]
    assert first["story_phase"] == "setup"
    assert "Character progression:" in first["scene_prompt"]
    assert "Visible change:" in first["expected_caption"]
