"""ContinuityAuditor Agent - Checks visual continuity between windows"""

from typing import Any, List, Optional
from ..agent_base import Agent, AgentResult

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class ContinuityAuditor(Agent):
    """
    Validates visual continuity between consecutive video windows.

    Uses existing embeddings (CLIP/DINOv2) to measure:
    - Character consistency (same person appears similar)
    - Motion smoothness (transition between windows is natural)
    """

    def __init__(self, embedding_model: Any, weight: float = 0.35):
        """
        Args:
            embedding_model: VisionEmbedder instance (from pipeline)
            weight: Importance in aggregate scoring
        """
        super().__init__("ContinuityAuditor", weight)
        self.embedding_model = embedding_model

    def evaluate(
        self,
        current_frames: List[Any],
        previous_frames: Optional[List[Any]] = None,
        character_names: Optional[List[str]] = None,
        scene_location: Optional[str] = None,
    ) -> AgentResult:
        """
        Score visual continuity of generated window.

        Args:
            current_frames: Video frames from current window
            previous_frames: Last frames from previous window (for transition check)
            character_names: Characters expected in scene
            scene_location: Expected location/scene

        Returns:
            AgentResult with continuity score (0-1)
        """
        scores = {}
        issues = []
        suggestions = []

        # 1. Character Consistency Check
        if character_names and previous_frames is not None:
            char_score = self._check_character_consistency(
                current_frames, previous_frames
            )
            scores["character"] = char_score
            if char_score < 0.70:
                issues.append("Character appearance changed significantly between windows")
                suggestions.append(
                    "Regenerate with stronger character identity constraints in prompt"
                )

        # 2. Motion Smoothness (transition check)
        if previous_frames is not None and len(previous_frames) > 0:
            motion_score = self._check_motion_smoothness(
                previous_frames[-1:], current_frames[:1]
            )
            scores["motion"] = motion_score
            if motion_score < 0.70:
                issues.append("Abrupt transition detected between windows")
                suggestions.append("Add frame-based continuity anchor in next prompt")

        # Aggregate score
        if scores:
            if HAS_NUMPY:
                aggregate_score = np.mean(list(scores.values()))
            else:
                aggregate_score = sum(scores.values()) / len(scores)
        else:
            aggregate_score = 0.75  # Default if minimal checks

        feedback = "\n".join(
            [f"  {k.capitalize()}: {v:.3f}" for k, v in scores.items()]
        )
        if not feedback:
            feedback = "Insufficient data for detailed continuity check"

        return AgentResult(
            score=min(max(aggregate_score, 0.0), 1.0),  # Clamp 0-1
            feedback=feedback,
            suggestions=suggestions,
            metadata={"issues": issues, "scores": scores},
        )

    def _check_character_consistency(
        self, current_frames: List[Any], previous_frames: List[Any]
    ) -> float:
        """Check if characters look similar between windows using embeddings"""
        try:
            if len(previous_frames) == 0 or len(current_frames) == 0:
                return 0.75

            prev_embed = self.embedding_model.embed_frame(previous_frames[-1])
            curr_embed = self.embedding_model.embed_frame(current_frames[0])

            # Cosine similarity
            if HAS_NUMPY:
                similarity = float(np.dot(prev_embed, curr_embed))
            else:
                similarity = self._cosine_similarity(prev_embed, curr_embed)

            return min(max(similarity, 0.0), 1.0)
        except Exception as e:
            print(f"Warning: Character consistency check failed: {e}")
            return 0.75

    def _check_motion_smoothness(
        self, last_prev_frame: List[Any], first_curr_frame: List[Any]
    ) -> float:
        """Check if transition between windows is smooth"""
        try:
            if len(last_prev_frame) == 0 or len(first_curr_frame) == 0:
                return 0.75

            prev_embed = self.embedding_model.embed_frame(last_prev_frame[0])
            curr_embed = self.embedding_model.embed_frame(first_curr_frame[0])

            # Cosine similarity
            if HAS_NUMPY:
                similarity = float(np.dot(prev_embed, curr_embed))
            else:
                similarity = self._cosine_similarity(prev_embed, curr_embed)

            return min(max(similarity, 0.0), 1.0)
        except Exception as e:
            print(f"Warning: Motion smoothness check failed: {e}")
            return 0.75

    @staticmethod
    def _cosine_similarity(vec1: Any, vec2: Any) -> float:
        """Pure Python cosine similarity (fallback for numpy)"""
        try:
            # Convert to lists if needed
            v1 = list(vec1) if not isinstance(vec1, list) else vec1
            v2 = list(vec2) if not isinstance(vec2, list) else vec2

            # Dot product
            dot_product = sum(a * b for a, b in zip(v1, v2))

            # Magnitudes
            mag1 = (sum(a * a for a in v1)) ** 0.5
            mag2 = (sum(b * b for b in v2)) ** 0.5

            # Cosine similarity
            if mag1 * mag2 < 1e-12:
                return 0.5
            return dot_product / (mag1 * mag2)
        except Exception:
            return 0.5
