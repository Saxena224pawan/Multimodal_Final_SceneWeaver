"""RefinementEngine - Main orchestrator for multi-agent iterative refinement"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent_base import Agent, AgentResult
from .agents import ContinuityAuditor, PhysicsValidator, PromptOptimizer, StorybeatsChecker


@dataclass
class RefinementMetadata:
    """Track iteration details for analysis"""

    window_idx: int
    narrative_beat: str
    total_iterations: int
    scores_history: List[float] = field(default_factory=list)
    agents_history: List[Dict] = field(default_factory=list)
    final_prompt: str = ""
    generation_time: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "window_idx": self.window_idx,
            "narrative_beat": self.narrative_beat,
            "total_iterations": self.total_iterations,
            "scores_history": self.scores_history,
            "agents_history": self.agents_history,
            "final_prompt": self.final_prompt,
            "generation_time": self.generation_time,
            "timestamp": self.timestamp,
        }


class RefinementEngine:
    """
    Orchestrates multi-agent iterative refinement loop.

    Workflow:
    1. Generate video with prompt
    2. Run all agents in parallel
    3. Compute aggregate score
    4. If score >= threshold: ACCEPT
    5. Else if iterations < max: optimize prompt and retry
    6. Else: accept with lower score
    """

    def __init__(
        self,
        video_model: Any,
        captioner: Any,
        embedding_model: Any,
        llm_model: Any,
        config: Optional[Dict] = None,
    ):
        """
        Initialize RefinementEngine.

        Args:
            video_model: WanBackbone or similar (generates frames)
            captioner: Captioner (generates per-frame descriptions)
            embedding_model: VisionEmbedder (extracts embeddings)
            llm_model: Language model (for LLM agents)
            config: Configuration dict with keys:
                - max_iterations: max retries (default: 3)
                - quality_threshold: score to accept (default: 0.70)
                - agent_weights: dict of agent names to weights
        """
        self.video_model = video_model
        self.captioner = captioner
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Config with defaults
        self.config = config or {}
        self.max_iterations = self.config.get("max_iterations", 3)
        self.quality_threshold = self.config.get("quality_threshold", 0.70)

        # Initialize agents
        self.continuity_auditor = ContinuityAuditor(embedding_model, weight=0.35)
        self.storybeats_checker = StorybeatsChecker(llm_model, weight=0.40)
        self.physics_validator = PhysicsValidator(llm_model, weight=0.25)
        self.prompt_optimizer = PromptOptimizer(llm_model)

        self.agents = {
            "continuity": self.continuity_auditor,
            "storybeats": self.storybeats_checker,
            "physics": self.physics_validator,
        }

    def refine_window(
        self,
        base_prompt: str,
        narrative_beat: str,
        window_idx: int,
        previous_frames: Optional[List[Any]] = None,
        character_names: Optional[List[str]] = None,
        scene_location: Optional[str] = None,
    ) -> Tuple[Optional[List[Any]], RefinementMetadata]:
        """
        Run iterative refinement for a single window.

        Args:
            base_prompt: Initial prompt for video generation
            narrative_beat: Story beat this window should represent
            window_idx: Index of this window (for metadata)
            previous_frames: Last frames from previous window (for continuity)
            character_names: Characters in scene
            scene_location: Scene location/description

        Returns:
            Tuple of (generated_frames, metadata)
        """
        from datetime import datetime

        metadata = RefinementMetadata(
            window_idx=window_idx,
            narrative_beat=narrative_beat,
            total_iterations=0,
            timestamp=datetime.now().isoformat(),
        )

        current_prompt = base_prompt
        current_frames = None
        start_time = time.time()

        print(f"\n{'='*70}")
        print(f"Window {window_idx:03d} | {narrative_beat[:40]}")
        print(f"{'='*70}")

        for iteration in range(self.max_iterations):
            print(f"\n[Iteration {iteration + 1}/{self.max_iterations}]")

            # STEP 1: Generate video
            print("  Generating video...")
            try:
                current_frames = self.video_model.generate(current_prompt)
            except Exception as e:
                print(f"  ❌ Generation failed: {e}")
                return None, metadata

            if current_frames is None or len(current_frames) == 0:
                print("  ❌ No frames generated")
                return None, metadata

            # STEP 2: Generate captions
            print("  Generating captions...")
            try:
                captions = [self.captioner.caption(f) for f in current_frames[:5]]  # Sample 5
            except Exception as e:
                print(f"  ⚠️  Caption generation failed: {e}")
                captions = [""] * len(current_frames)

            # STEP 3: Run all agents
            print("  Running agents...")
            agent_scores = {}
            agent_results = {}

            # Continuity Auditor
            try:
                continuity_result = self.continuity_auditor.evaluate(
                    current_frames=current_frames,
                    previous_frames=previous_frames,
                    character_names=character_names,
                    scene_location=scene_location,
                )
                agent_results["continuity"] = continuity_result
                agent_scores["continuity"] = continuity_result.score
                print(f"    ✓ ContinuityAuditor: {continuity_result.score:.3f}")
            except Exception as e:
                print(f"    ✗ ContinuityAuditor failed: {e}")
                agent_scores["continuity"] = 0.70

            # StorybeatsChecker
            try:
                storybeats_result = self.storybeats_checker.evaluate(
                    window_beat=narrative_beat,
                    generated_captions=captions,
                    previous_beat=None,
                    character_constraints=", ".join(character_names)
                    if character_names
                    else None,
                )
                agent_results["storybeats"] = storybeats_result
                agent_scores["storybeats"] = storybeats_result.score
                print(f"    ✓ StorybeatsChecker: {storybeats_result.score:.3f}")
            except Exception as e:
                print(f"    ✗ StorybeatsChecker failed: {e}")
                agent_scores["storybeats"] = 0.70

            # PhysicsValidator
            try:
                physics_result = self.physics_validator.evaluate(
                    generated_captions=captions,
                    scene_constraints=scene_location,
                    character_positions=(
                        {name: "present" for name in character_names}
                        if character_names
                        else None
                    ),
                )
                agent_results["physics"] = physics_result
                agent_scores["physics"] = physics_result.score
                print(f"    ✓ PhysicsValidator: {physics_result.score:.3f}")
            except Exception as e:
                print(f"    ✗ PhysicsValidator failed: {e}")
                agent_scores["physics"] = 0.70

            # STEP 4: Compute aggregate score
            aggregate_score = self._compute_aggregate_score(agent_scores)
            print(f"\n  📊 Aggregate Score: {aggregate_score:.3f}")

            metadata.scores_history.append(aggregate_score)
            metadata.agents_history.append({
                "iteration": iteration + 1,
                "scores": agent_scores,
                "aggregate": aggregate_score,
            })

            # STEP 5: Check if quality is acceptable
            if aggregate_score >= self.quality_threshold:
                print(f"  ✅ THRESHOLD MET ({aggregate_score:.3f} >= {self.quality_threshold})")
                metadata.total_iterations = iteration + 1
                metadata.final_prompt = current_prompt
                break

            # STEP 6: Check if we should retry
            if iteration == self.max_iterations - 1:
                print(f"  ⚠️  MAX ITERATIONS REACHED")
                metadata.total_iterations = iteration + 1
                metadata.final_prompt = current_prompt
                break

            # STEP 7: Optimize prompt and retry
            print(f"\n  🔧 Optimizing prompt...")
            try:
                current_prompt = self.prompt_optimizer.optimize(
                    base_prompt=current_prompt,
                    agent_feedback=agent_results,
                    iteration_count=iteration,
                    narrative_beat=narrative_beat,
                )
                print(f"  New prompt length: {len(current_prompt)} chars")
            except Exception as e:
                print(f"  ❌ Optimization failed: {e}")
                break

        metadata.generation_time = time.time() - start_time
        print(f"\n  ⏱️  Window generation time: {metadata.generation_time:.1f}s")
        print(f"{'='*70}")

        return current_frames, metadata

    def _compute_aggregate_score(self, agent_scores: Dict[str, float]) -> float:
        """Weighted average of agent scores"""
        if not agent_scores:
            return 0.5

        weights = {
            "continuity": 0.35,
            "storybeats": 0.40,
            "physics": 0.25,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for agent_name, score in agent_scores.items():
            weight = weights.get(agent_name, 1.0)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5
