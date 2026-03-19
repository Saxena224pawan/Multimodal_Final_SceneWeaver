from __future__ import annotations

from director_llm import SceneDirector, SceneDirectorConfig


def test_scene_director_derives_character_evolution_fields_for_windows() -> None:
    director = SceneDirector(SceneDirectorConfig(model_id=None), window_seconds=4)
    windows = director.plan_windows(
        storyline=(
            "Mira rushes through the station toward the last train. "
            "Arjun steps in front of her and asks her to stay. "
            "They argue beside the platform doors, then reconcile as the train leaves."
        ),
        total_minutes=0.4,
        beats_override=[
            {"beat": "Mira rushes through the station toward the last train."},
            {"beat": "Arjun steps in front of Mira and asks her to stay."},
            {"beat": "Mira and Arjun argue beside the closing train doors."},
            {"beat": "Mira and Arjun reconcile and stand together as the train leaves."},
        ],
    )

    assert windows[0].story_phase == "setup"
    assert windows[0].character_progression
    assert windows[1].relationship_dynamic
    assert windows[-1].visible_change


def test_scene_director_heuristic_prompt_includes_character_evolution_guidance() -> None:
    director = SceneDirector(SceneDirectorConfig(model_id=None), window_seconds=4)
    storyline = (
        "Mira rushes through the station toward the last train. "
        "Arjun steps in front of her and asks her to stay. "
        "They argue beside the platform doors, then reconcile as the train leaves."
    )
    window = director.plan_windows(
        storyline=storyline,
        total_minutes=0.4,
        beats_override=[
            {"beat": "Mira and Arjun argue beside the closing train doors."},
            {"beat": "Mira and Arjun reconcile and stand together as the train leaves."},
        ],
    )[0]

    bundle = director.refine_prompt(
        storyline=storyline,
        window=window,
        previous_prompt="Shot type: wide establishing. Action: Mira rushes through the station.",
        previous_scene_conversation="Arjun asks Mira to stop for one moment.",
        memory_feedback=None,
    )

    assert "Character progression:" in bundle.prompt_text
    assert "Relationship dynamic:" in bundle.prompt_text
    assert "Visible change:" in bundle.prompt_text
    assert bundle.shot_plan.subject_blocking


def test_scene_director_dynamic_window_generation_scales_with_story_length() -> None:
    director = SceneDirector(
        SceneDirectorConfig(
            model_id=None,
            window_count_mode="dynamic",
            target_words_per_window=10,
            min_dynamic_windows=1,
            max_dynamic_windows=12,
        ),
        window_seconds=4,
    )
    short_windows = director.plan_windows(
        storyline="Mira waits by the gate. Arjun arrives. They leave together.",
        total_minutes=2.0,
    )
    long_windows = director.plan_windows(
        storyline=(
            "Mira studies the departure board while clutching her ticket and scanning the crowd for Arjun. "
            "Arjun pushes through the station with an apology, but Mira refuses to slow down and keeps moving toward the platform. "
            "They argue across the escalator, force each other to confront what the trip means, and nearly separate at the security gate. "
            "A sudden announcement changes the departure, so Mira pauses, turns back, and decides to hear Arjun out before boarding. "
            "They reconcile beside the train doors and step onto the carriage together with a calmer, shared resolve."
        ),
        total_minutes=0.2,
    )

    assert len(short_windows) == 3
    assert len(long_windows) > len(short_windows)
    assert len(long_windows) <= 12


def test_scene_director_fixed_window_generation_still_available() -> None:
    director = SceneDirector(
        SceneDirectorConfig(model_id=None, window_count_mode="fixed"),
        window_seconds=5,
    )
    windows = director.plan_windows(
        storyline="Mira waits at the station for Arjun.",
        total_minutes=0.5,
    )

    assert len(windows) == 6

