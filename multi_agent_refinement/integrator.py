"""Integration module - Hook the multi-agent system into existing pipeline"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .refinement_engine import RefinementEngine, RefinementMetadata


class MultiAgentPipelineIntegrator:
    """Wrapper to integrate multi-agent refinement into the SceneWeaver pipeline."""

    def __init__(
        self,
        video_model: Any,
        captioner: Any,
        embedding_model: Any,
        llm_model: Any,
        output_dir: str = "outputs/multi_agent",
        max_iterations: int = 5,
        quality_threshold: float = 0.70,
        convergence_patience: int = 2,
        convergence_tolerance: float = 0.015,
        progressive_tightening: bool = True,
        tightening_strength: float = 0.8,
        enable_continuity: bool = True,
        enable_storybeats: bool = True,
        enable_physics: bool = True,
        agent_weights: Optional[Dict[str, float]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.quality_threshold = float(quality_threshold)
        self.convergence_patience = int(convergence_patience)
        self.convergence_tolerance = float(convergence_tolerance)
        self.progressive_tightening = bool(progressive_tightening)
        self.tightening_strength = float(tightening_strength)
        self.enable_continuity = bool(enable_continuity)
        self.enable_storybeats = bool(enable_storybeats)
        self.enable_physics = bool(enable_physics)
        self.agent_weights = dict(agent_weights or {})

        config = {
            "max_iterations": int(max_iterations),
            "quality_threshold": float(quality_threshold),
            "convergence_patience": int(convergence_patience),
            "convergence_tolerance": float(convergence_tolerance),
            "progressive_tightening": bool(progressive_tightening),
            "tightening_strength": float(tightening_strength),
            "enable_continuity": self.enable_continuity,
            "enable_storybeats": self.enable_storybeats,
            "enable_physics": self.enable_physics,
            "agent_weights": self.agent_weights,
        }
        self.engine = RefinementEngine(
            video_model=video_model,
            captioner=captioner,
            embedding_model=embedding_model,
            llm_model=llm_model,
            config=config,
        )
        self.all_metadata: List[RefinementMetadata] = []

    def generate_window(
        self,
        base_prompt: str,
        narrative_beat: str,
        window_idx: int,
        previous_frames: Optional[List[Any]] = None,
        character_names: Optional[List[str]] = None,
        scene_location: Optional[str] = None,
        scene_anchor_frames: Optional[List[Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[List[Any]], RefinementMetadata]:
        frames, metadata = self.engine.refine_window(
            base_prompt=base_prompt,
            narrative_beat=narrative_beat,
            window_idx=window_idx,
            previous_frames=previous_frames,
            character_names=character_names,
            scene_location=scene_location,
            scene_anchor_frames=scene_anchor_frames,
            generation_kwargs=generation_kwargs,
        )
        self.all_metadata.append(metadata)
        return frames, metadata

    def save_metadata(self, metadata: RefinementMetadata) -> None:
        filename = self.metadata_dir / f"window_{metadata.window_idx:03d}.json"
        with filename.open("w", encoding="utf-8") as handle:
            json.dump(metadata.to_dict(), handle, indent=2)

    def save_summary(self) -> Dict[str, Any]:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_windows": len(self.all_metadata),
            "quality_threshold": self.quality_threshold,
            "convergence_patience": self.convergence_patience,
            "convergence_tolerance": self.convergence_tolerance,
            "progressive_tightening": self.progressive_tightening,
            "tightening_strength": self.tightening_strength,
            "enabled_agents": {
                "continuity": self.enable_continuity,
                "storybeats": self.enable_storybeats,
                "physics": self.enable_physics,
            },
            "agent_weights": self.agent_weights,
            "average_iterations": (
                sum(m.total_iterations for m in self.all_metadata) / len(self.all_metadata)
                if self.all_metadata
                else 0.0
            ),
            "converged_windows": sum(1 for m in self.all_metadata if m.converged),
            "windows": [m.to_dict() for m in self.all_metadata],
        }
        summary_file = self.output_dir / "summary.json"
        with summary_file.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\n✅ Saved summary to {summary_file}")
        return summary

    def get_convergence_stats(self) -> Dict[str, Any]:
        if not self.all_metadata:
            return {
                "total_windows": 0,
                "avg_iterations": 0.0,
                "max_iterations": 0,
                "avg_final_score": 0.0,
                "windows_at_threshold": 0,
                "windows_converged": 0,
            }

        iterations = [m.total_iterations for m in self.all_metadata]
        scores = [m.scores_history[-1] if m.scores_history else 0.0 for m in self.all_metadata]
        return {
            "total_windows": len(self.all_metadata),
            "avg_iterations": sum(iterations) / len(iterations),
            "max_iterations": max(iterations),
            "avg_final_score": sum(scores) / len(scores),
            "windows_at_threshold": sum(1 for score in scores if score >= self.quality_threshold),
            "windows_converged": sum(1 for item in self.all_metadata if item.converged),
        }
