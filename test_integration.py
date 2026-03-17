"""Test harness for multi-agent system integration testing"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock models for testing without heavy dependencies
class MockVideoModel:
    """Mock video model that generates dummy frames"""

    def generate(self, prompt: str) -> List[Any]:
        """Generate mock frames"""
        try:
            import numpy as np
            # Return 10 mock frames (480x832 RGB)
            return [np.zeros((480, 832, 3), dtype=np.uint8) for _ in range(10)]
        except ImportError:
            # Fallback: return simple dict-based frames for testing
            return [{"type": "mock_frame", "size": (480, 832, 3)} for _ in range(10)]


class MockCaptioner:
    """Mock captioner that returns static captions"""

    def caption(self, frame: Any) -> str:
        """Generate mock caption"""
        return "A scene in a story setting with characters and action"


class MockEmbedder:
    """Mock embedder that returns static embeddings"""

    def embed_frame(self, frame: Any) -> List[float]:
        """Generate mock embedding"""
        # Simulate CLIP embedding (768 dims)
        return [0.5] * 768

    def embed_text(self, text: str) -> List[float]:
        """Generate mock text embedding"""
        return [0.5] * 768


class MockLLM:
    """Mock LLM that returns simulated agent responses"""

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        """Generate mock LLM response"""
        self.call_count += 1

        # Determine what type of response based on prompt content
        if "beat_adherence" in prompt:
            # StorybeatsChecker response
            return json.dumps(
                {
                    "beat_adherence": 85,
                    "character_clarity": 82,
                    "plot_advancement": 78,
                    "issues": [],
                }
            )
        elif "teleportation" in prompt:
            # PhysicsValidator response
            return json.dumps(
                {
                    "teleportation": 90,
                    "gravity": 88,
                    "permanence": 85,
                    "issues": [],
                }
            )
        else:
            # PromptOptimizer response
            return "Refined prompt: A hero discovers a magical artifact in an ancient forest with mysterious glowing ruins and guardian creatures"


class MockPipeline:
    """Mock SceneWeaver pipeline for testing"""

    def __init__(self):
        self.video_model = MockVideoModel()
        self.captioner = MockCaptioner()
        self.embedder = MockEmbedder()
        self.llm_model = MockLLM()

    def run_integration_test(
        self,
        num_windows: int = 3,
        max_iterations: int = 2,
        output_dir: str = "outputs/test_integration",
    ) -> Dict[str, Any]:
        """Run integration test with mock components"""

        from multi_agent_refinement.integrator import MultiAgentPipelineIntegrator

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("Multi-Agent Integration Test")
        print("=" * 70 + "\n")

        print(f"Test Configuration:")
        print(f"  Windows: {num_windows}")
        print(f"  Max iterations/window: {max_iterations}")
        print(f"  Output: {output_dir}\n")

        # Initialize integrator
        integrator = MultiAgentPipelineIntegrator(
            video_model=self.video_model,
            captioner=self.captioner,
            embedding_model=self.embedder,
            llm_model=self.llm_model,
            output_dir=output_dir,
            max_iterations=max_iterations,
            quality_threshold=0.70,
        )

        # Simulate window generation
        test_windows = [
            {
                "idx": 0,
                "beat": "The hero enters the enchanted forest and discovers ancient ruins",
                "characters": ["Hero"],
                "location": "Mystical forest with glowing ruins",
            },
            {
                "idx": 1,
                "beat": "The hero finds a magical artifact guarded by mysterious creatures",
                "characters": ["Hero", "Guardian Creatures"],
                "location": "Ancient temple chamber",
            },
            {
                "idx": 2,
                "beat": "The hero escapes with the artifact as the temple collapses",
                "characters": ["Hero"],
                "location": "Collapsing temple with falling stones",
            },
        ]

        # Limit to num_windows
        test_windows = test_windows[:num_windows]

        print("Generating windows with multi-agent refinement...")
        print("-" * 70 + "\n")

        previous_frames = None

        for window in test_windows:
            frames, metadata = integrator.generate_window(
                base_prompt=f"A video of: {window['beat']}",
                narrative_beat=window["beat"],
                window_idx=window["idx"],
                previous_frames=previous_frames,
                character_names=window["characters"],
                scene_location=window["location"],
            )

            if frames is None:
                print(f"Window {window['idx']}: ❌ Generation failed")
                continue

            integrator.save_metadata(metadata)
            previous_frames = frames

            score = metadata.scores_history[-1] if metadata.scores_history else 0.0
            print(f"Window {window['idx']}: {metadata.total_iterations} iteration(s), score: {score:.3f}")

        # Get summary
        integrator.save_summary()
        stats = integrator.get_convergence_stats()

        print("\n" + "-" * 70)
        print("\n📊 Test Results:")
        print(f"  Total windows: {stats['total_windows']}")
        print(f"  Avg iterations: {stats['avg_iterations']:.2f}")
        print(f"  Avg final score: {stats['avg_final_score']:.3f}")
        print(f"  Windows at threshold: {stats['windows_at_threshold']}/{stats['total_windows']}")

        print(f"\n✅ Test completed!")
        print(f"   LLM was called {self.llm_model.call_count} times")
        print(f"   Results saved to: {output_dir}\n")

        return stats


def test_basic_integration():
    """Test basic integration with mock components"""
    pipeline = MockPipeline()
    stats = pipeline.run_integration_test(
        num_windows=2,
        max_iterations=2,
        output_dir="outputs/test_basic_integration",
    )

    assert stats["total_windows"] == 2
    assert stats["avg_iterations"] >= 1.0
    assert stats["avg_final_score"] > 0.0

    print("✅ Basic integration test passed!\n")


def test_ablation_setup():
    """Test that ablation study data can be collected"""
    from multi_agent_refinement.integrator import MultiAgentPipelineIntegrator

    pipeline = MockPipeline()
    output_dir = "outputs/test_ablation_setup"

    print("\n" + "=" * 70)
    print("Testing Ablation Study Setup")
    print("=" * 70 + "\n")

    integrator = MultiAgentPipelineIntegrator(
        video_model=pipeline.video_model,
        captioner=pipeline.captioner,
        embedding_model=pipeline.embedder,
        llm_model=pipeline.llm_model,
        output_dir=output_dir,
        max_iterations=2,
        quality_threshold=0.70,
    )

    # Generate mock window
    frames, metadata = integrator.generate_window(
        base_prompt="Test prompt",
        narrative_beat="Test beat",
        window_idx=0,
        character_names=["Character1"],
        scene_location="Test location",
    )

    # Verify metadata structure
    meta_dict = metadata.to_dict()
    assert "window_idx" in meta_dict
    assert "total_iterations" in meta_dict
    assert "scores_history" in meta_dict
    assert "agents_history" in meta_dict

    print("✅ Ablation setup test passed!")
    print(f"   Metadata structure verified")
    print(f"   Window metadata: {output_dir}/metadata/window_000.json\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Multi-Agent System Integration Tests")
    print("=" * 70 + "\n")

    try:
        test_basic_integration()
        test_ablation_setup()

        print("=" * 70)
        print("🎉 All integration tests passed!")
        print("=" * 70 + "\n")

        print("Next steps:")
        print("1. Run with real models: python scripts/run_story_pipeline_with_agents.py")
        print("2. Collect data on 10-20 stories")
        print("3. Run ablation studies")
        print("4. Create visualizations\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
