from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from multi_agent_refinement.integrator import MultiAgentPipelineIntegrator
from tests.multi_agent_support import (
    MockArrayVideoModel,
    MockCaptioner,
    MockEmbedder,
    MockLLM,
    MockPipeline,
)


def test_basic_integration() -> None:
    pipeline = MockPipeline()
    with TemporaryDirectory() as tmpdir:
        stats = pipeline.run_integration_test(
            output_dir=tmpdir,
            num_windows=2,
            max_iterations=2,
        )
        assert (Path(tmpdir) / "summary.json").exists()

    assert stats["total_windows"] == 2
    assert stats["avg_iterations"] >= 1.0
    assert stats["avg_final_score"] > 0.0


def test_ndarray_frame_outputs() -> None:
    llm = MockLLM()
    with TemporaryDirectory() as tmpdir:
        integrator = MultiAgentPipelineIntegrator(
            video_model=MockArrayVideoModel(),
            captioner=MockCaptioner(),
            embedding_model=MockEmbedder(),
            llm_model=llm,
            output_dir=tmpdir,
            max_iterations=2,
            quality_threshold=0.70,
        )

        frames, metadata = integrator.generate_window(
            base_prompt="A crow approaches a water pot",
            narrative_beat="The crow spots the pot and studies the water level",
            window_idx=0,
            previous_frames=None,
            character_names=["Crow"],
            scene_location="Dusty courtyard",
        )

    assert frames is not None
    assert len(frames) == 10
    assert metadata.total_iterations >= 1


def test_ablation_setup() -> None:
    pipeline = MockPipeline()
    with TemporaryDirectory() as tmpdir:
        integrator = MultiAgentPipelineIntegrator(
            video_model=pipeline.video_model,
            captioner=pipeline.captioner,
            embedding_model=pipeline.embedder,
            llm_model=pipeline.llm_model,
            output_dir=tmpdir,
            max_iterations=2,
            quality_threshold=0.70,
        )

        frames, metadata = integrator.generate_window(
            base_prompt="Test prompt",
            narrative_beat="Test beat",
            window_idx=0,
            character_names=["Character1"],
            scene_location="Test location",
        )
        assert frames is not None
        integrator.save_metadata(metadata)

        metadata_path = Path(tmpdir) / "metadata" / "window_000.json"
        assert metadata_path.exists()
        meta_dict = metadata.to_dict()

    assert "window_idx" in meta_dict
    assert "total_iterations" in meta_dict
    assert "scores_history" in meta_dict
    assert "agents_history" in meta_dict


def test_ablation_agent_toggles() -> None:
    pipeline = MockPipeline()
    with TemporaryDirectory() as baseline_dir, TemporaryDirectory() as ablated_dir:
        baseline = pipeline.create_integrator(output_dir=baseline_dir)
        baseline_frames, baseline_metadata = baseline.generate_window(
            base_prompt="A courier runs through the station",
            narrative_beat="The courier races for the last train.",
            window_idx=0,
            character_names=["Courier"],
            scene_location="Station platform",
        )
        assert baseline_frames is not None

        no_continuity = pipeline.create_integrator(
            output_dir=ablated_dir,
            enable_continuity=False,
        )
        ablated_frames, ablated_metadata = no_continuity.generate_window(
            base_prompt="A courier runs through the station",
            narrative_beat="The courier races for the last train.",
            window_idx=0,
            character_names=["Courier"],
            scene_location="Station platform",
        )
        assert ablated_frames is not None

    baseline_scores = baseline_metadata.agents_history[0]["scores"]
    ablated_scores = ablated_metadata.agents_history[0]["scores"]
    enabled_agents = ablated_metadata.agents_history[0]["enabled_agents"]

    assert "continuity" in baseline_scores
    assert "continuity" not in ablated_scores
    assert set(ablated_scores) == {"storybeats", "physics"}
    assert enabled_agents == ["storybeats", "physics"]


def main() -> None:
    test_basic_integration()
    test_ndarray_frame_outputs()
    test_ablation_setup()
    test_ablation_agent_toggles()
    print("multi-agent integration smoke tests passed")


if __name__ == "__main__":
    main()
