"""RefinementEngine - Main orchestrator for multi-agent iterative refinement"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .agent_base import AgentResult
from .agents import ContinuityAuditor, PhysicsValidator, PromptOptimizer, StorybeatsChecker


@dataclass
class RefinementMetadata:
    """Track iteration details for analysis"""

    window_idx: int
    narrative_beat: str
    total_iterations: int
    scores_history: List[float] = field(default_factory=list)
    agents_history: List[Dict[str, Any]] = field(default_factory=list)
    final_prompt: str = ""
    generation_time: float = 0.0
    timestamp: str = ""
    converged: bool = False
    convergence_reason: str = ""
    best_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_idx": self.window_idx,
            "narrative_beat": self.narrative_beat,
            "total_iterations": self.total_iterations,
            "scores_history": self.scores_history,
            "agents_history": self.agents_history,
            "final_prompt": self.final_prompt,
            "generation_time": self.generation_time,
            "timestamp": self.timestamp,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "best_score": self.best_score,
        }


class RefinementEngine:
    """Orchestrates multi-agent iterative refinement loop."""

    def __init__(
        self,
        video_model: Any,
        captioner: Any,
        embedding_model: Any,
        llm_model: Any,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.video_model = video_model
        self.captioner = captioner
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        self.config = config or {}
        self.max_iterations = int(self.config.get("max_iterations", 5))
        self.quality_threshold = float(self.config.get("quality_threshold", 0.70))
        self.caption_sample_count = int(self.config.get("caption_sample_count", 4))
        self.convergence_patience = int(self.config.get("convergence_patience", 2))
        self.convergence_tolerance = float(self.config.get("convergence_tolerance", 0.015))
        self.progressive_tightening = bool(self.config.get("progressive_tightening", True))
        self.tightening_strength = float(self.config.get("tightening_strength", 0.8))
        self.enable_continuity = bool(self.config.get("enable_continuity", True))
        self.enable_storybeats = bool(self.config.get("enable_storybeats", True))
        self.enable_physics = bool(self.config.get("enable_physics", True))
        self.fallback_agent_score = float(self.config.get("fallback_agent_score", 0.70))

        configured_weights = self.config.get("agent_weights", {})
        if not isinstance(configured_weights, dict):
            configured_weights = {}
        self.agent_weights = {
            "continuity": float(configured_weights.get("continuity", 0.35)),
            "storybeats": float(configured_weights.get("storybeats", 0.40)),
            "physics": float(configured_weights.get("physics", 0.25)),
        }
        self.enabled_agents = [
            agent_name
            for agent_name, enabled in (
                ("continuity", self.enable_continuity),
                ("storybeats", self.enable_storybeats),
                ("physics", self.enable_physics),
            )
            if enabled
        ]
        if not self.enabled_agents:
            raise ValueError("At least one refinement agent must be enabled.")

        self.continuity_auditor = (
            ContinuityAuditor(embedding_model, weight=self.agent_weights["continuity"])
            if self.enable_continuity
            else None
        )
        self.storybeats_checker = (
            StorybeatsChecker(llm_model, weight=self.agent_weights["storybeats"])
            if self.enable_storybeats
            else None
        )
        self.physics_validator = (
            PhysicsValidator(llm_model, weight=self.agent_weights["physics"])
            if self.enable_physics
            else None
        )
        self.prompt_optimizer = PromptOptimizer(llm_model)

    @staticmethod
    def _attempt_progress(iteration: int, max_iterations: int) -> float:
        if max_iterations <= 1:
            return 0.0
        return max(0.0, min(1.0, iteration / max(1, max_iterations - 1)))

    @staticmethod
    def _merge_negative_prompt_terms(base_negative_prompt: str, extra_terms: List[str]) -> str:
        terms = [term.strip() for term in str(base_negative_prompt or "").split(",") if term.strip()]
        normalized = {term.lower() for term in terms}
        for term in extra_terms:
            cleaned = str(term or "").strip()
            if cleaned and cleaned.lower() not in normalized:
                terms.append(cleaned)
                normalized.add(cleaned.lower())
        return ", ".join(terms)

    def _tightening_state(
        self,
        generation_kwargs: Optional[Dict[str, Any]],
        iteration: int,
        max_iterations: int,
    ) -> Dict[str, Any]:
        kwargs = dict(generation_kwargs or {})
        progress = self._attempt_progress(iteration, max_iterations) if self.progressive_tightening else 0.0
        strength = max(0.0, self.tightening_strength)
        guidance_scale = float(kwargs.get("guidance_scale", 0.0))
        guidance_scale = min(14.0, max(0.0, guidance_scale + (progress * strength * 1.1)))

        reference_strength = kwargs.get("reference_strength")
        if reference_strength is not None:
            reference_strength = min(0.95, max(0.0, float(reference_strength) + (progress * strength * 0.10)))

        noise_blend_strength = kwargs.get("noise_blend_strength")
        if noise_blend_strength is not None:
            noise_blend_strength = min(0.65, max(0.0, float(noise_blend_strength) + (progress * strength * 0.05)))

        extra_terms: List[str] = []
        prompt_constraints: List[str] = []
        if progress > 0.0:
            extra_terms.extend([
                "identity drift",
                "background drift",
                "beat confusion",
                "unclear action",
            ])
            prompt_constraints.append(
                "Stronger retry: preserve exact character identity, stable background layout, and a clearly readable story beat."
            )
        if progress >= 0.45:
            extra_terms.extend([
                "pose reset",
                "emotion reset",
                "duplicate subjects",
                "scene discontinuity",
            ])
            prompt_constraints.append(
                "No pose reset, no emotion reset, no duplicate subjects, and no weakened action clarity."
            )
        if progress >= 0.8:
            extra_terms.extend([
                "prop inconsistency",
                "weak eyelines",
                "staging drift",
            ])
            prompt_constraints.append(
                "Make the beat unmistakable with tighter staging, stronger eyelines, and cleaner prop continuity."
            )

        negative_prompt = self._merge_negative_prompt_terms(str(kwargs.get("negative_prompt", "")), extra_terms)
        return {
            "attempt_progress": round(progress, 4),
            "guidance_scale": round(guidance_scale, 4),
            "reference_strength": None if reference_strength is None else round(reference_strength, 4),
            "noise_blend_strength": None if noise_blend_strength is None else round(noise_blend_strength, 4),
            "negative_prompt": negative_prompt,
            "prompt_constraint": " ".join(prompt_constraints).strip(),
        }

    def _tightened_generation_kwargs(
        self,
        generation_kwargs: Optional[Dict[str, Any]],
        iteration: int,
        max_iterations: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        kwargs = dict(generation_kwargs or {})
        state = self._tightening_state(kwargs, iteration, max_iterations)
        kwargs["guidance_scale"] = state["guidance_scale"]
        kwargs["negative_prompt"] = state["negative_prompt"]
        if state["reference_strength"] is not None:
            kwargs["reference_strength"] = state["reference_strength"]
        if state["noise_blend_strength"] is not None:
            kwargs["noise_blend_strength"] = state["noise_blend_strength"]
        return kwargs, state

    def _convergence_status(self, score_history: List[float]) -> Tuple[bool, str]:
        if not score_history:
            return False, ""
        latest = float(score_history[-1])
        if latest >= self.quality_threshold:
            return True, "threshold_met"
        if self.convergence_patience <= 0 or len(score_history) < self.convergence_patience + 1:
            return False, ""
        recent = [float(score) for score in score_history[-(self.convergence_patience + 1) :]]
        if max(recent) - min(recent) <= max(0.0, self.convergence_tolerance):
            return True, "score_plateau"
        return False, ""

    def _generate_video(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]],
        iteration: int,
    ) -> List[Any]:
        kwargs = dict(generation_kwargs or {})
        seed = kwargs.get("seed")
        if seed is not None and not bool(kwargs.get("disable_random_generation", False)):
            kwargs["seed"] = int(seed) + iteration

        if hasattr(self.video_model, "generate_clip"):
            return self.video_model.generate_clip(prompt=prompt, **kwargs)
        if hasattr(self.video_model, "generate"):
            return self.video_model.generate(prompt)
        if callable(self.video_model):
            return self.video_model(prompt, **kwargs)
        raise AttributeError("video_model must provide generate_clip(prompt=...) or generate(prompt)")

    def _caption_generated_frames(self, frames: List[Any]) -> List[str]:
        if self.captioner is None:
            return []

        frame_count = len(frames) if frames is not None else 0
        sample_count = max(1, min(self.caption_sample_count, frame_count)) if frame_count else 0
        if sample_count <= 0:
            return []

        if hasattr(self.captioner, "caption_frames"):
            captions, _summary, _dupes = self.captioner.caption_frames(frames, sample_count=sample_count)
            if not captions:
                return []
            if len(captions) <= sample_count:
                return captions
            stride = max(1, len(captions) // sample_count)
            return captions[::stride][:sample_count]

        if hasattr(self.captioner, "caption"):
            return [self.captioner.caption(frame) for frame in frames[:sample_count]]

        return []

    def refine_window(
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
        from datetime import datetime

        metadata = RefinementMetadata(
            window_idx=window_idx,
            narrative_beat=narrative_beat,
            total_iterations=0,
            timestamp=datetime.now().isoformat(),
        )

        current_prompt = (base_prompt or narrative_beat or "").strip()
        current_frames: Optional[List[Any]] = None
        best_frames: Optional[List[Any]] = None
        best_prompt = current_prompt
        best_score = -1.0
        start_time = time.time()
        effective_max_iterations = max(1, self.max_iterations)

        print(f"\n{'=' * 70}")
        print(f"Window {window_idx:03d} | {narrative_beat[:40]}")
        print(f"{'=' * 70}")

        for iteration in range(effective_max_iterations):
            print(f"\n[Iteration {iteration + 1}/{effective_max_iterations}]")
            tightened_kwargs, tightening_state = self._tightened_generation_kwargs(
                generation_kwargs=generation_kwargs,
                iteration=iteration,
                max_iterations=effective_max_iterations,
            )
            prompt_for_generation = current_prompt
            if tightening_state.get("prompt_constraint"):
                prompt_for_generation = f"{current_prompt} {tightening_state['prompt_constraint']}".strip()

            print("  Generating video...")
            try:
                current_frames = self._generate_video(
                    prompt=prompt_for_generation,
                    generation_kwargs=tightened_kwargs,
                    iteration=iteration,
                )
            except Exception as exc:
                print(f"  ❌ Generation failed: {exc}")
                metadata.convergence_reason = "generation_failed"
                break

            if current_frames is None or len(current_frames) == 0:
                print("  ❌ No frames generated")
                metadata.convergence_reason = "generation_failed"
                break

            print("  Generating captions...")
            try:
                captions = self._caption_generated_frames(current_frames)
            except Exception as exc:
                print(f"  ⚠️  Caption generation failed: {exc}")
                captions = []

            print("  Running agents...")
            agent_scores: Dict[str, float] = {}
            agent_results: Dict[str, AgentResult] = {}

            if self.continuity_auditor is None:
                print("    - ContinuityAuditor: disabled")
            else:
                try:
                    continuity_result = self.continuity_auditor.evaluate(
                        current_frames=current_frames,
                        previous_frames=previous_frames,
                        character_names=character_names,
                        scene_location=scene_location,
                        scene_anchor_frames=scene_anchor_frames,
                    )
                    agent_results["continuity"] = continuity_result
                    agent_scores["continuity"] = continuity_result.score
                    print(f"    ✓ ContinuityAuditor: {continuity_result.score:.3f}")
                except Exception as exc:
                    print(f"    ✗ ContinuityAuditor failed: {exc}")
                    agent_scores["continuity"] = self.fallback_agent_score

            if self.storybeats_checker is None:
                print("    - StorybeatsChecker: disabled")
            else:
                try:
                    storybeats_result = self.storybeats_checker.evaluate(
                        window_beat=narrative_beat,
                        generated_captions=captions,
                        previous_beat=None,
                        character_constraints=", ".join(character_names) if character_names else None,
                    )
                    agent_results["storybeats"] = storybeats_result
                    agent_scores["storybeats"] = storybeats_result.score
                    print(f"    ✓ StorybeatsChecker: {storybeats_result.score:.3f}")
                except Exception as exc:
                    print(f"    ✗ StorybeatsChecker failed: {exc}")
                    agent_scores["storybeats"] = self.fallback_agent_score

            if self.physics_validator is None:
                print("    - PhysicsValidator: disabled")
            else:
                try:
                    physics_result = self.physics_validator.evaluate(
                        generated_captions=captions,
                        scene_constraints=scene_location,
                        character_positions={name: "present" for name in character_names} if character_names else None,
                    )
                    agent_results["physics"] = physics_result
                    agent_scores["physics"] = physics_result.score
                    print(f"    ✓ PhysicsValidator: {physics_result.score:.3f}")
                except Exception as exc:
                    print(f"    ✗ PhysicsValidator failed: {exc}")
                    agent_scores["physics"] = self.fallback_agent_score

            aggregate_score = self._compute_aggregate_score(agent_scores)
            print(f"\n  📊 Aggregate Score: {aggregate_score:.3f}")

            metadata.scores_history.append(aggregate_score)
            metadata.agents_history.append(
                {
                    "iteration": iteration + 1,
                    "scores": agent_scores,
                    "enabled_agents": list(self.enabled_agents),
                    "aggregate": aggregate_score,
                    "prompt_excerpt": prompt_for_generation[:240],
                    "tightening": tightening_state,
                }
            )
            metadata.total_iterations = iteration + 1

            if aggregate_score > best_score or best_frames is None:
                best_score = aggregate_score
                best_frames = current_frames
                best_prompt = prompt_for_generation

            converged, convergence_reason = self._convergence_status(metadata.scores_history)
            if converged:
                metadata.converged = True
                metadata.convergence_reason = convergence_reason
                if convergence_reason == "threshold_met":
                    print(f"  ✅ THRESHOLD MET ({aggregate_score:.3f} >= {self.quality_threshold})")
                else:
                    print(f"  ✅ SCORE PLATEAU DETECTED ({aggregate_score:.3f})")
                break

            if iteration == effective_max_iterations - 1:
                print("  ⚠️  MAX ITERATIONS REACHED")
                metadata.convergence_reason = "max_iterations"
                break

            print("\n  🔧 Optimizing prompt...")
            try:
                improved_prompt = self.prompt_optimizer.optimize(
                    base_prompt=current_prompt,
                    agent_feedback=agent_results,
                    iteration_count=iteration,
                    narrative_beat=narrative_beat,
                )
                current_prompt = improved_prompt or current_prompt
                print(f"  New prompt length: {len(current_prompt)} chars")
            except Exception as exc:
                print(f"  ❌ Optimization failed: {exc}")
                metadata.convergence_reason = "optimizer_failed"
                break

        metadata.final_prompt = best_prompt
        metadata.best_score = max(0.0, best_score)
        metadata.generation_time = time.time() - start_time
        if not metadata.convergence_reason:
            metadata.convergence_reason = "generation_failed" if not metadata.scores_history else "max_iterations"
        metadata.converged = metadata.convergence_reason in {"threshold_met", "score_plateau"}
        print(f"\n  ⏱️  Window generation time: {metadata.generation_time:.1f}s")
        print(f"{'=' * 70}")

        selected_frames = best_frames if best_frames is not None else current_frames
        return selected_frames, metadata

    def _compute_aggregate_score(self, agent_scores: Dict[str, float]) -> float:
        if not agent_scores:
            return 0.5

        weights = self.agent_weights

        weighted_sum = 0.0
        total_weight = 0.0
        for agent_name, score in agent_scores.items():
            weight = weights.get(agent_name, 1.0)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5
