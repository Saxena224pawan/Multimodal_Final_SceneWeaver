from __future__ import annotations

from director_llm import SceneDirector, SceneDirectorConfig
from scripts.run_story_pipeline import _extract_character_names, build_generation_prompt


def test_extract_character_names_ignores_location_names() -> None:
    story = (
        "A brave warrior searches for the legendary Crystal Mountain across treacherous valleys "
        "and dark forests."
    )
    beat = "Start this beat clearly: A brave warrior searches for the legendary Crystal Mountain."

    assert _extract_character_names(story, beat) == ["Protagonist"]


def test_build_generation_prompt_skips_dialogue_staging_for_self_talk() -> None:
    prompt = build_generation_prompt(
        refined_prompt="Shot type: medium. Action: Warrior looks ahead.",
        beat="A brave warrior searches for the legendary Crystal Mountain.",
        style_prefix="cinematic realism",
        character_lock="",
        previous_environment_anchor="",
        current_environment_anchor="dark forest path",
        scene_change_requested=False,
        story_state_hint="Required now: the warrior searches.",
        scene_conversation="She whispers to herself, 'I must find it before nightfall.'",
        previous_scene_conversation="",
        conversation_progress_instruction="Establish the first meaningful exchange for this part of the story.",
        story_progress_instruction="Establish the first beat clearly with a readable action and objective.",
        dialogue_scene=False,
    )

    lowered = prompt.lower()
    assert "two-shot" not in lowered
    assert "both speakers" not in lowered
    assert "do not invent an extra speaker" in lowered


def test_scene_director_skips_scene_conversation_for_solo_exploration() -> None:
    director = SceneDirector(SceneDirectorConfig(model_id=None), window_seconds=4)
    storyline = (
        "A brave warrior searches for the legendary Crystal Mountain across treacherous valleys "
        "and dark forests."
    )
    window = director.plan_windows(
        storyline=storyline,
        total_minutes=0.2,
        beats_override=[
            {"beat": "A brave warrior searches for the legendary Crystal Mountain across treacherous valleys and dark forests."}
        ],
    )[0]

    bundle = director.refine_prompt(
        storyline=storyline,
        window=window,
        previous_prompt="Shot type: wide. Action: The warrior surveys the valley.",
        previous_scene_conversation="",
        memory_feedback=None,
    )

    assert bundle.scene_conversation == ""
