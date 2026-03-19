from __future__ import annotations

from multi_agent_refinement.agent_base import AgentResult
from multi_agent_refinement.agents import ContinuityAuditor, PhysicsValidator, PromptOptimizer, StorybeatsChecker
from tests.multi_agent_support import MockEmbedder, MockLLM


def _sample_frames():
    import numpy as np

    base_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    base_frame[..., 0] = 95
    base_frame[..., 1] = 125
    base_frame[..., 2] = 80
    return [base_frame.copy() for _ in range(3)]


def test_agent_result() -> None:
    result = AgentResult(
        score=0.85,
        feedback="Test feedback",
        suggestions=["suggestion 1"],
        metadata={"key": "value"},
    )

    assert result.score == 0.85
    assert result.feedback == "Test feedback"
    assert result.suggestions == ["suggestion 1"]
    assert result.metadata == {"key": "value"}


def test_continuity_auditor() -> None:
    frames = _sample_frames()
    auditor = ContinuityAuditor(MockEmbedder(), weight=0.35)

    result = auditor.evaluate(
        current_frames=frames,
        previous_frames=frames,
        character_names=["Alice", "Bob"],
        scene_anchor_frames=frames[:1],
        scene_location="same forest clearing",
    )

    assert isinstance(result, AgentResult)
    assert 0 <= result.score <= 1
    assert "background_anchor" in result.metadata.get("scores", {})
    assert "style_consistency" in result.metadata.get("scores", {})


def test_storybeats_checker() -> None:
    checker = StorybeatsChecker(MockLLM(), weight=0.40)

    result = checker.evaluate(
        window_beat="Hero discovers the artifact",
        generated_captions=["Frame 0: hero in forest", "Frame 1: hero finds gold"],
    )

    assert isinstance(result, AgentResult)
    assert 0 <= result.score <= 1


def test_physics_validator() -> None:
    validator = PhysicsValidator(MockLLM(), weight=0.25)

    result = validator.evaluate(
        generated_captions=["Frame 0: hero walks", "Frame 1: hero standing"],
        scene_constraints="Outdoor forest",
    )

    assert isinstance(result, AgentResult)
    assert 0 <= result.score <= 1


def test_prompt_optimizer() -> None:
    optimizer = PromptOptimizer(MockLLM())
    improved = optimizer.optimize(
        base_prompt="Original prompt",
        agent_feedback={
            "continuity": AgentResult(
                score=0.65,
                feedback="Character changed",
                suggestions=["Fix character"],
                metadata={"issues": ["character drift"]},
            )
        },
        iteration_count=0,
        narrative_beat="Finding the artifact",
    )

    assert isinstance(improved, str)
    assert len(improved) > 0


def test_prompt_optimizer_guardrails() -> None:
    class LongResponseLLM:
        def generate(self, prompt: str) -> str:
            return "x" * 1200

    optimizer = PromptOptimizer(LongResponseLLM())
    improved = optimizer.optimize(
        base_prompt="Original prompt",
        agent_feedback={
            "continuity": AgentResult(
                score=0.5,
                feedback="Too much drift",
                suggestions=["Tighten background"],
                metadata={"issues": ["drift"]},
            )
        },
        iteration_count=1,
        narrative_beat="Crow reaches the pot",
    )

    assert improved == "Original prompt"


def main() -> None:
    test_agent_result()
    test_continuity_auditor()
    test_storybeats_checker()
    test_physics_validator()
    test_prompt_optimizer()
    test_prompt_optimizer_guardrails()
    print("multi-agent component smoke tests passed")


if __name__ == "__main__":
    main()
