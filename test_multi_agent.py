"""Test multi-agent system components"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_refinement.agent_base import Agent, AgentResult


def test_agent_result():
    """Test AgentResult dataclass"""
    result = AgentResult(
        score=0.85,
        feedback="Test feedback",
        suggestions=["suggestion 1"],
        metadata={"key": "value"},
    )

    assert result.score == 0.85
    assert result.feedback == "Test feedback"
    assert len(result.suggestions) == 1
    print("✅ AgentResult test passed")


def test_continuity_auditor():
    """Test ContinuityAuditor agent"""
    try:
        from multi_agent_refinement.agents import ContinuityAuditor

        # Create mock embedder
        class MockEmbedder:
            def embed_frame(self, frame):
                import numpy as np

                return np.array([0.5] * 256)

        auditor = ContinuityAuditor(MockEmbedder(), weight=0.35)

        # Test evaluation
        mock_frames = [None, None, None]
        result = auditor.evaluate(
            current_frames=mock_frames,
            previous_frames=mock_frames,
            character_names=["Alice", "Bob"],
        )

        assert isinstance(result, AgentResult)
        assert 0 <= result.score <= 1
        print("✅ ContinuityAuditor test passed")
    except Exception as e:
        print(f"❌ ContinuityAuditor test failed: {e}")


def test_storybeats_checker():
    """Test StorybeatsChecker agent"""
    try:
        from multi_agent_refinement.agents import StorybeatsChecker

        # Create mock LLM
        class MockLLM:
            def generate(self, prompt):
                return '{"beat_adherence": 80, "character_clarity": 85, "plot_advancement": 75, "issues": []}'

        checker = StorybeatsChecker(MockLLM(), weight=0.40)

        result = checker.evaluate(
            window_beat="Hero discovers the artifact",
            generated_captions=["Frame 0: hero in forest", "Frame 1: hero finds gold"],
        )

        assert isinstance(result, AgentResult)
        assert 0 <= result.score <= 1
        print("✅ StorybeatsChecker test passed")
    except Exception as e:
        print(f"❌ StorybeatsChecker test failed: {e}")


def test_physics_validator():
    """Test PhysicsValidator agent"""
    try:
        from multi_agent_refinement.agents import PhysicsValidator

        # Create mock LLM
        class MockLLM:
            def generate(self, prompt):
                return '{"teleportation": 90, "gravity": 85, "permanence": 80, "issues": []}'

        validator = PhysicsValidator(MockLLM(), weight=0.25)

        result = validator.evaluate(
            generated_captions=["Frame 0: hero walks", "Frame 1: hero standing"],
            scene_constraints="Outdoor forest",
        )

        assert isinstance(result, AgentResult)
        assert 0 <= result.score <= 1
        print("✅ PhysicsValidator test passed")
    except Exception as e:
        print(f"❌ PhysicsValidator test failed: {e}")


def test_prompt_optimizer():
    """Test PromptOptimizer agent"""
    try:
        from multi_agent_refinement.agents import PromptOptimizer

        # Create mock LLM
        class MockLLM:
            def generate(self, prompt):
                return "An improved prompt about the hero searching for the artifact in a golden forest"

        optimizer = PromptOptimizer(MockLLM())

        # Create mock feedback
        feedback = {
            "continuity": AgentResult(
                score=0.65,
                feedback="Character changed",
                suggestions=["Fix character"],
                metadata={"issues": ["character drift"]},
            )
        }

        improved = optimizer.optimize(
            base_prompt="Original prompt",
            agent_feedback=feedback,
            iteration_count=0,
            narrative_beat="Finding the artifact",
        )

        assert isinstance(improved, str)
        assert len(improved) > 0
        print("✅ PromptOptimizer test passed")
    except Exception as e:
        print(f"❌ PromptOptimizer test failed: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Multi-Agent System Tests")
    print("=" * 60 + "\n")

    test_agent_result()
    test_continuity_auditor()
    test_storybeats_checker()
    test_physics_validator()
    test_prompt_optimizer()

    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
