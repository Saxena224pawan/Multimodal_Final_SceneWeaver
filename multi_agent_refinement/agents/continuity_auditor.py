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
    - Scene-anchor stability (background/layout stays grounded to the same scene)
    - Style consistency (palette, lighting mood, and texture stay coherent)
    """

    def __init__(self, embedding_model: Any, weight: float = 0.35):
        super().__init__("ContinuityAuditor", weight)
        self.embedding_model = embedding_model

    def evaluate(
        self,
        current_frames: List[Any],
        previous_frames: Optional[List[Any]] = None,
        character_names: Optional[List[str]] = None,
        scene_location: Optional[str] = None,
        scene_anchor_frames: Optional[List[Any]] = None,
    ) -> AgentResult:
        scores = {}
        issues = []
        suggestions = []

        if character_names and previous_frames is not None:
            char_score = self._check_character_consistency(current_frames, previous_frames)
            scores["character"] = char_score
            if char_score < 0.70:
                issues.append("Character appearance changed significantly between windows")
                suggestions.append("Regenerate with stronger character identity constraints in prompt")

        if previous_frames is not None and len(previous_frames) > 0:
            motion_score = self._check_motion_smoothness(previous_frames[-1:], current_frames[:1])
            scores["motion"] = motion_score
            if motion_score < 0.70:
                issues.append("Abrupt transition detected between windows")
                suggestions.append("Add frame-based continuity anchor in next prompt")

        if scene_anchor_frames is not None and len(scene_anchor_frames) > 0:
            background_score = self._check_scene_anchor_consistency(scene_anchor_frames[:1], current_frames[:1])
            scores["background_anchor"] = background_score
            if background_score < 0.72:
                issues.append("Background layout drifted away from the established scene anchor")
                if scene_location:
                    suggestions.append(
                        f"Keep the exact same background landmarks, prop placement, and lighting for {scene_location}"
                    )
                else:
                    suggestions.append(
                        "Keep the same background landmarks, prop placement, horizon line, and lighting as the original scene anchor"
                    )

        current_first_frame = current_frames[:1] if self._has_frames(current_frames) else []
        style_reference_frames = None
        if self._has_frames(scene_anchor_frames):
            style_reference_frames = scene_anchor_frames[:1]
        elif self._has_frames(previous_frames):
            style_reference_frames = previous_frames[-1:]
        style_score = self._check_style_consistency(style_reference_frames, current_first_frame)
        if style_score is not None:
            scores["style_consistency"] = style_score
            if style_score < 0.74:
                issues.append("Visual style drifted away from the established palette or lighting mood")
                if scene_location:
                    suggestions.append(
                        f"Preserve the same palette, contrast, lighting mood, and texture level for {scene_location}"
                    )
                else:
                    suggestions.append(
                        "Preserve the same palette, contrast, lighting mood, and texture level as the earlier window"
                    )

        if scores:
            if HAS_NUMPY:
                aggregate_score = float(np.mean(list(scores.values())))
            else:
                aggregate_score = sum(scores.values()) / len(scores)
        else:
            aggregate_score = 0.75

        feedback = "\n".join([f"  {k.capitalize()}: {v:.3f}" for k, v in scores.items()])
        if not feedback:
            feedback = "Insufficient data for detailed continuity check"

        return AgentResult(
            score=min(max(aggregate_score, 0.0), 1.0),
            feedback=feedback,
            suggestions=suggestions,
            metadata={"issues": issues, "scores": scores},
        )

    def _check_character_consistency(self, current_frames: List[Any], previous_frames: List[Any]) -> float:
        if not self._has_frames(previous_frames) or not self._has_frames(current_frames):
            return 0.75
        return self._embed_similarity(previous_frames[-1], current_frames[0])

    def _check_motion_smoothness(self, last_prev_frame: List[Any], first_curr_frame: List[Any]) -> float:
        if not self._has_frames(last_prev_frame) or not self._has_frames(first_curr_frame):
            return 0.75
        return self._embed_similarity(last_prev_frame[0], first_curr_frame[0])

    def _check_scene_anchor_consistency(self, scene_anchor_frames: List[Any], current_frames: List[Any]) -> float:
        if not self._has_frames(scene_anchor_frames) or not self._has_frames(current_frames):
            return 0.75
        return self._embed_similarity(scene_anchor_frames[0], current_frames[0])

    def _check_style_consistency(self, reference_frames: Optional[List[Any]], current_frames: List[Any]) -> Optional[float]:
        if not self._has_frames(reference_frames) or not self._has_frames(current_frames):
            return None
        reference_signature = self._style_signature(reference_frames[0])
        current_signature = self._style_signature(current_frames[0])
        if reference_signature is None or current_signature is None:
            return None
        diffs = [abs(reference_signature[key] - current_signature[key]) for key in reference_signature.keys()]
        if not diffs:
            return None
        score = 1.0 - (sum(diffs) / len(diffs))
        return min(max(float(score), 0.0), 1.0)

    @staticmethod
    def _has_frames(frames: Optional[List[Any]]) -> bool:
        if frames is None:
            return False
        try:
            return len(frames) > 0
        except TypeError:
            return False

    def _style_signature(self, frame: Any) -> Optional[dict[str, float]]:
        if not HAS_NUMPY:
            return None
        try:
            arr = np.asarray(frame)
            if arr.ndim == 4:
                arr = arr[-1]
            if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim not in (2, 3):
                return None
            arr = arr.astype(np.float32)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return None
            min_val = float(finite.min())
            max_val = float(finite.max())
            if min_val >= 0.0 and max_val <= 1.0:
                arr = arr * 255.0
            elif min_val >= -1.0 and max_val <= 1.0:
                arr = (arr + 1.0) * 127.5
            arr = np.clip(arr, 0.0, 255.0)
            if arr.ndim == 2:
                rgb = np.stack([arr, arr, arr], axis=2)
            else:
                if arr.shape[2] == 1:
                    rgb = np.repeat(arr, 3, axis=2)
                else:
                    rgb = arr[..., :3]
            gray = rgb.mean(axis=2)
            mean_rgb = rgb.reshape(-1, 3).mean(axis=0)
            channel_delta = np.abs(rgb - gray[..., None]).mean(axis=2)
            grad_x = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
            grad_y = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
            return {
                "mean_r": float(mean_rgb[0] / 255.0),
                "mean_g": float(mean_rgb[1] / 255.0),
                "mean_b": float(mean_rgb[2] / 255.0),
                "brightness": float(gray.mean() / 255.0),
                "contrast": float(min(max(np.std(gray) / 64.0, 0.0), 1.0)),
                "saturation": float(min(max(channel_delta.mean() / 85.0, 0.0), 1.0)),
                "texture": float(min(max((grad_x + grad_y) / 26.0, 0.0), 1.0)),
            }
        except Exception:
            return None

    def _embed_similarity(self, left_frame: Any, right_frame: Any) -> float:
        try:
            if self.embedding_model is None:
                return 0.75

            left_embed = self.embedding_model.embed_frame(left_frame)
            right_embed = self.embedding_model.embed_frame(right_frame)

            if HAS_NUMPY:
                similarity = float(np.dot(left_embed, right_embed))
            else:
                similarity = self._cosine_similarity(left_embed, right_embed)

            return min(max(similarity, 0.0), 1.0)
        except Exception as exc:
            print(f"Warning: Continuity embedding check failed: {exc}")
            return 0.75

    @staticmethod
    def _cosine_similarity(vec1: Any, vec2: Any) -> float:
        try:
            v1 = list(vec1) if not isinstance(vec1, list) else vec1
            v2 = list(vec2) if not isinstance(vec2, list) else vec2
            dot_product = sum(a * b for a, b in zip(v1, v2))
            mag1 = (sum(a * a for a in v1)) ** 0.5
            mag2 = (sum(b * b for b in v2)) ** 0.5
            if mag1 * mag2 < 1e-12:
                return 0.5
            return dot_product / (mag1 * mag2)
        except Exception:
            return 0.5
