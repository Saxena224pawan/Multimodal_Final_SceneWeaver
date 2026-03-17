"""
Example: Using Multi-Agent Refinement with SceneWeaver Pipeline

This script demonstrates how to integrate the multi-agent system into
an existing story pipeline run.
"""

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_refinement.integrator import MultiAgentPipelineIntegrator


def example_with_existing_pipeline():
    """
    Example integration with the existing run_story_pipeline.py

    To use this in your actual pipeline, replace the window generation loop:

    BEFORE:
    -------
    for window in windows:
        frames = video_model.generate(window.prompt)
        # Direct generation

    AFTER:
    ------
    integrator = MultiAgentPipelineIntegrator(
        video_model=video_model,
        captioner=captioner,
        embedding_model=embedding_model,
        llm_model=llm_model,
        output_dir=output_dir
    )

    for i, window in enumerate(windows):
        frames, metadata = integrator.generate_window(
            base_prompt=window.prompt,
            narrative_beat=window.beat,
            window_idx=i,
            previous_frames=previous_frames[-1:] if previous_frames else None,
            character_names=window.character_names,
            scene_location=window.location
        )
        integrator.save_metadata(metadata)
        previous_frames = frames

    # Save summary
    integrator.save_summary()
    stats = integrator.get_convergence_stats()
    print(f"Average iterations: {stats['avg_iterations']:.2f}")
    print(f"Avg final score: {stats['avg_final_score']:.3f}")
    """

    print(__doc__)


def example_minimal_test():
    """Minimal test of the multi-agent system"""
    print("\n" + "=" * 70)
    print("Multi-Agent Refinement System - Minimal Test")
    print("=" * 70 + "\n")

    # Create mock components
    class MockVideoModel:
        def generate(self, prompt):
            import numpy as np

            # Return 10 mock frames (empty arrays)
            return [np.zeros((480, 832, 3), dtype=np.uint8) for _ in range(10)]

    class MockCaptioner:
        def caption(self, frame):
            return "A scene in a forest with a hero character"

    class MockEmbedder:
        def embed_frame(self, frame):
            import numpy as np

            return np.array([0.5] * 256)

    class MockLLM:
        def generate(self, prompt):
            if "beat_adherence" in prompt:
                return '{"beat_adherence": 85, "character_clarity": 80, "plot_advancement": 75, "issues": []}'
            elif "teleportation" in prompt:
                return '{"teleportation": 90, "gravity": 85, "permanence": 80, "issues": []}'
            else:
                return "An improved prompt focusing on the hero's quest in the forest"

    # Initialize integrator
    output_dir = "outputs/test_multi_agent"
    integrator = MultiAgentPipelineIntegrator(
        video_model=MockVideoModel(),
        captioner=MockCaptioner(),
        embedding_model=MockEmbedder(),
        llm_model=MockLLM(),
        output_dir=output_dir,
        max_iterations=3,
        quality_threshold=0.70,
    )

    print("Testing multi-agent system with mock components...")
    print(f"Output directory: {output_dir}\n")

    # Simulate 2 window generation
    windows = [
        {
            "idx": 0,
            "beat": "The hero enters the enchanted forest",
            "characters": ["Hero"],
            "location": "Dark forest",
        },
        {
            "idx": 1,
            "beat": "The hero discovers a magical artifact",
            "characters": ["Hero"],
            "location": "Forest clearing with ancient ruins",
        },
    ]

    previous_frames = None

    for window in windows:
        print(f"\n🎬 Generating Window {window['idx']}: {window['beat']}")
        print("-" * 70)

        frames, metadata = integrator.generate_window(
            base_prompt=f"A video of: {window['beat']}",
            narrative_beat=window["beat"],
            window_idx=window["idx"],
            previous_frames=previous_frames,
            character_names=window["characters"],
            scene_location=window["location"],
        )

        integrator.save_metadata(metadata)
        previous_frames = frames

        print(f"\n  Completed in {metadata.total_iterations} iteration(s)")
        print(f"  Final score: {metadata.scores_history[-1]:.3f}")

    # Save summary
    integrator.save_summary()

    # Print stats
    stats = integrator.get_convergence_stats()
    print("\n" + "=" * 70)
    print("📊 Convergence Statistics")
    print("=" * 70)
    print(f"  Total windows: {stats['total_windows']}")
    print(f"  Avg iterations per window: {stats['avg_iterations']:.2f}")
    print(f"  Max iterations: {stats['max_iterations']}")
    print(f"  Avg final score: {stats['avg_final_score']:.3f}")
    print(f"  Windows at threshold (0.70+): {stats['windows_at_threshold']}/{stats['total_windows']}")
    print("=" * 70 + "\n")

    print("✅ Test completed successfully!")
    print(f"✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    # Run minimal test
    try:
        example_minimal_test()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
