"""Integration module - Hook the multi-agent system into existing pipeline"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from datetime import datetime

from .refinement_engine import RefinementEngine, RefinementMetadata


class MultiAgentPipelineIntegrator:
    """
    Wrapper to integrate multi-agent refinement into existing SceneWeaver pipeline.

    Usage:
        integrator = MultiAgentPipelineIntegrator(
            video_model=wan_backbone,
            captioner=caption_model,
            embedding_model=embedder,
            llm_model=llm,
            output_dir="outputs/my_run"
        )

        for window in windows:
            frames, metadata = integrator.generate_window(
                base_prompt=window.prompt,
                narrative_beat=window.beat,
                window_idx=i,
                character_names=window.characters
            )
            integrator.save_metadata(metadata)
    """

    def __init__(
        self,
        video_model: Any,
        captioner: Any,
        embedding_model: Any,
        llm_model: Any,
        output_dir: str = "outputs/multi_agent",
        max_iterations: int = 3,
        quality_threshold: float = 0.70,
    ):
        """
        Initialize integrator.

        Args:
            video_model: WanBackbone instance
            captioner: Captioner instance
            embedding_model: VisionEmbedder instance
            llm_model: Language model instance
            output_dir: Directory to save metadata
            max_iterations: Max refinement iterations per window
            quality_threshold: Score threshold to accept window
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata directory
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Initialize refinement engine
        config = {
            "max_iterations": max_iterations,
            "quality_threshold": quality_threshold,
        }

        self.engine = RefinementEngine(
            video_model=video_model,
            captioner=captioner,
            embedding_model=embedding_model,
            llm_model=llm_model,
            config=config,
        )

        # Track all windows
        self.all_metadata = []

    def generate_window(
        self,
        base_prompt: str,
        narrative_beat: str,
        window_idx: int,
        previous_frames: Optional[List[Any]] = None,
        character_names: Optional[List[str]] = None,
        scene_location: Optional[str] = None,
    ) -> tuple:
        """
        Generate a window using multi-agent refinement.

        Returns:
            Tuple of (frames, metadata)
        """
        frames, metadata = self.engine.refine_window(
            base_prompt=base_prompt,
            narrative_beat=narrative_beat,
            window_idx=window_idx,
            previous_frames=previous_frames,
            character_names=character_names,
            scene_location=scene_location,
        )

        # Track metadata
        self.all_metadata.append(metadata)

        return frames, metadata

    def save_metadata(self, metadata: RefinementMetadata) -> None:
        """Save window metadata to JSON"""
        filename = self.metadata_dir / f"window_{metadata.window_idx:03d}.json"
        with open(filename, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def save_summary(self) -> None:
        """Save summary of all windows"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_windows": len(self.all_metadata),
            "average_iterations": (
                sum(m.total_iterations for m in self.all_metadata) / len(self.all_metadata)
                if self.all_metadata
                else 0
            ),
            "windows": [m.to_dict() for m in self.all_metadata],
        }

        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Saved summary to {summary_file}")

    def get_convergence_stats(self) -> Dict:
        """Get convergence statistics"""
        if not self.all_metadata:
            return {}

        iterations = [m.total_iterations for m in self.all_metadata]
        scores = [m.scores_history[-1] if m.scores_history else 0 for m in self.all_metadata]

        return {
            "total_windows": len(self.all_metadata),
            "avg_iterations": sum(iterations) / len(iterations),
            "max_iterations": max(iterations),
            "avg_final_score": sum(scores) / len(scores),
            "windows_at_threshold": sum(1 for s in scores if s >= 0.70),
        }
