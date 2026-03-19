from __future__ import annotations

import importlib.util
from pathlib import Path

from multi_agent_refinement.agent_base import AgentResult
from multi_agent_refinement.refinement_engine import RefinementEngine


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_story_pipeline.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("story_pipeline_runner", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
story_pipeline_runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(story_pipeline_runner)


class DummyVideoModel:
    def __init__(self) -> None:
        self.calls = []

    def generate_clip(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return [f"frame_{len(self.calls)}"]


class DummyCaptioner:
    is_stub = False

    def caption_frames(self, frames, sample_count=4):
        return ["caption"] * max(1, int(sample_count)), "summary", False


class SequenceAgent:
    def __init__(self, name: str, scores):
        self.name = name
        self.scores = list(scores)
        self.calls = 0

    def evaluate(self, **kwargs):
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return AgentResult(
            score=float(score),
            feedback=f"{self.name}:{score}",
            suggestions=["tighten the prompt"],
            metadata={"issues": [f"{self.name}_issue"]},
        )


class DummyPromptOptimizer:
    def optimize(self, base_prompt: str, agent_feedback, iteration_count: int, narrative_beat: str) -> str:
        return f"{base_prompt} retry_{iteration_count + 1}".strip()


def _make_engine(*, scores, quality_threshold: float, patience: int, tolerance: float):
    video_model = DummyVideoModel()
    engine = RefinementEngine(
        video_model=video_model,
        captioner=DummyCaptioner(),
        embedding_model=None,
        llm_model=None,
        config={
            "max_iterations": 5,
            "quality_threshold": quality_threshold,
            "convergence_patience": patience,
            "convergence_tolerance": tolerance,
            "progressive_tightening": True,
            "tightening_strength": 0.8,
        },
    )
    engine.continuity_auditor = SequenceAgent("continuity", scores)
    engine.storybeats_checker = SequenceAgent("storybeats", scores)
    engine.physics_validator = SequenceAgent("physics", scores)
    engine.prompt_optimizer = DummyPromptOptimizer()
    return engine, video_model


def test_standard_retry_tightening_strengthens_retry_settings() -> None:
    base_reference_strength = story_pipeline_runner._reference_strength_for_window(
        base_strength=0.68,
        scene_change_requested=False,
        reference_source="previous_window_tail",
        same_scene_as_previous=True,
    )
    first = story_pipeline_runner._tightened_retry_settings(
        base_guidance_scale=6.0,
        base_negative_prompt="blurry, flicker",
        base_reference_strength=base_reference_strength,
        base_noise_blend_strength=0.2,
        attempt_idx=0,
        max_attempts=4,
        progressive_tightening=True,
        tightening_strength=0.8,
    )
    last = story_pipeline_runner._tightened_retry_settings(
        base_guidance_scale=6.0,
        base_negative_prompt="blurry, flicker",
        base_reference_strength=base_reference_strength,
        base_noise_blend_strength=0.2,
        attempt_idx=3,
        max_attempts=4,
        progressive_tightening=True,
        tightening_strength=0.8,
    )

    assert last["guidance_scale"] > first["guidance_scale"]
    assert last["reference_strength"] >= first["reference_strength"]
    assert last["noise_blend_strength"] >= first["noise_blend_strength"]
    assert "identity drift" in last["negative_prompt"]
    assert last["repair_constraint"]



def test_standard_retry_convergence_status_detects_threshold_and_plateau() -> None:
    threshold_hit = story_pipeline_runner._retry_convergence_status(
        [0.62, 0.74],
        threshold=0.72,
        patience=2,
        tolerance=0.01,
    )
    plateau_hit = story_pipeline_runner._retry_convergence_status(
        [0.61, 0.615, 0.618],
        threshold=0.8,
        patience=2,
        tolerance=0.01,
    )

    assert threshold_hit == (True, "threshold_met")
    assert plateau_hit == (True, "score_plateau")



def test_refinement_engine_converges_on_threshold_with_progressive_tightening() -> None:
    engine, video_model = _make_engine(
        scores=[0.55, 0.68, 0.84],
        quality_threshold=0.8,
        patience=2,
        tolerance=0.01,
    )
    frames, metadata = engine.refine_window(
        base_prompt="cinematic shot of Mira at the station",
        narrative_beat="Mira decides whether to board the train.",
        window_idx=0,
        generation_kwargs={
            "guidance_scale": 5.4,
            "negative_prompt": "blurry, flicker",
            "reference_strength": 0.62,
            "noise_blend_strength": 0.2,
            "disable_random_generation": True,
        },
    )

    assert frames
    assert metadata.total_iterations == 3
    assert metadata.converged is True
    assert metadata.convergence_reason == "threshold_met"
    assert metadata.best_score >= 0.84
    assert video_model.calls[-1]["kwargs"]["guidance_scale"] > video_model.calls[0]["kwargs"]["guidance_scale"]
    assert "identity drift" in video_model.calls[-1]["kwargs"]["negative_prompt"]



def test_refinement_engine_stops_on_score_plateau() -> None:
    engine, _video_model = _make_engine(
        scores=[0.60, 0.605, 0.607, 0.608],
        quality_threshold=0.9,
        patience=2,
        tolerance=0.01,
    )
    _frames, metadata = engine.refine_window(
        base_prompt="cinematic shot of Arjun waiting",
        narrative_beat="Arjun waits for Mira and looks unsettled.",
        window_idx=1,
        generation_kwargs={
            "guidance_scale": 5.0,
            "negative_prompt": "blurry",
            "reference_strength": 0.58,
            "disable_random_generation": True,
        },
    )

    assert metadata.total_iterations == 3
    assert metadata.converged is True
    assert metadata.convergence_reason == "score_plateau"
