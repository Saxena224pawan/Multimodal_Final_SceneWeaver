from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class MockVideoModel:
    def generate(self, prompt: str) -> List[Any]:
        try:
            import numpy as np

            return [np.zeros((480, 832, 3), dtype=np.uint8) for _ in range(10)]
        except ImportError:
            return [{"type": "mock_frame", "size": (480, 832, 3)} for _ in range(10)]


class MockArrayVideoModel:
    def generate(self, prompt: str):
        import numpy as np

        return np.zeros((10, 480, 832, 3), dtype=np.uint8)


class MockCaptioner:
    is_stub = False

    def caption(self, frame: Any) -> str:
        return "A scene in a story setting with characters and action"

    def caption_frames(self, frames, sample_count: int = 4):
        count = max(1, int(sample_count))
        return [self.caption(None)] * count, "story summary", False


class MockEmbedder:
    def embed_frame(self, frame: Any) -> List[float]:
        return [0.5] * 768

    def embed_text(self, text: str) -> List[float]:
        return [0.5] * 768


class MockLLM:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        if "beat_adherence" in prompt:
            return json.dumps(
                {
                    "beat_adherence": 85,
                    "character_clarity": 82,
                    "plot_advancement": 78,
                    "issues": [],
                }
            )
        if "teleportation" in prompt:
            return json.dumps(
                {
                    "teleportation": 90,
                    "gravity": 88,
                    "permanence": 85,
                    "issues": [],
                }
            )
        return (
            "Refined prompt: A hero discovers a magical artifact in an ancient forest "
            "with mysterious glowing ruins and guardian creatures"
        )


def build_test_windows() -> List[Dict[str, Any]]:
    return [
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


class MockPipeline:
    def __init__(self) -> None:
        self.video_model = MockVideoModel()
        self.captioner = MockCaptioner()
        self.embedder = MockEmbedder()
        self.llm_model = MockLLM()

    def create_integrator(
        self,
        output_dir: str,
        *,
        max_iterations: int = 2,
        quality_threshold: float = 0.70,
        **integrator_kwargs,
    ):
        from multi_agent_refinement.integrator import MultiAgentPipelineIntegrator

        return MultiAgentPipelineIntegrator(
            video_model=self.video_model,
            captioner=self.captioner,
            embedding_model=self.embedder,
            llm_model=self.llm_model,
            output_dir=output_dir,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            **integrator_kwargs,
        )

    def run_integration_test(
        self,
        *,
        output_dir: str,
        num_windows: int = 3,
        max_iterations: int = 2,
    ) -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        integrator = self.create_integrator(
            output_dir=output_path.as_posix(),
            max_iterations=max_iterations,
        )

        previous_frames = None
        for window in build_test_windows()[:num_windows]:
            frames, metadata = integrator.generate_window(
                base_prompt=f"A video of: {window['beat']}",
                narrative_beat=window["beat"],
                window_idx=window["idx"],
                previous_frames=previous_frames,
                character_names=window["characters"],
                scene_location=window["location"],
            )
            if frames is None:
                raise AssertionError(f"mock integration produced no frames for window {window['idx']}")
            integrator.save_metadata(metadata)
            previous_frames = frames

        integrator.save_summary()
        return integrator.get_convergence_stats()
