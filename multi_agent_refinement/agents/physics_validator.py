"""PhysicsValidator Agent - Checks physical plausibility"""

import json
from typing import Any, List, Optional
from ..agent_base import Agent, AgentResult


class PhysicsValidator(Agent):
    """
    Validates physical plausibility of generated video.

    Checks:
    - No character teleportation
    - Gravity obeyed
    - Object permanence
    """

    def __init__(self, llm_model: Any, weight: float = 0.25):
        """
        Args:
            llm_model: Language model for evaluation
            weight: Importance in aggregate scoring
        """
        super().__init__("PhysicsValidator", weight)
        self.llm_model = llm_model

    def evaluate(
        self,
        generated_captions: List[str],
        scene_constraints: Optional[str] = None,
        character_positions: Optional[dict] = None,
    ) -> AgentResult:
        """
        Score physical plausibility.

        Args:
            generated_captions: Per-frame descriptions
            scene_constraints: Constraints like "Indoor, no flying"
            character_positions: Expected character locations

        Returns:
            AgentResult with physics score (0-1)
        """
        captions_text = "\n".join(
            [f"Frame {i}: {cap}" for i, cap in enumerate(generated_captions)]
        )

        eval_prompt = f"""You are a physics checker for video generation.

Scene Constraints: {scene_constraints or "None specified"}
{f"Expected Characters: {', '.join(character_positions.keys())}" if character_positions else ""}

Video Frame Captions:
{captions_text}

Rate physicality 0-100 for:
1. TELEPORTATION: Are characters teleporting? (0=yes, 100=no)
2. GRAVITY: Is gravity obeyed? (0=violated, 100=perfect)
3. OBJECT_PERMANENCE: Do objects stay present? (0=disappear, 100=constant)

Respond ONLY with JSON:
{{"teleportation": <0-100>, "gravity": <0-100>, "permanence": <0-100>, "issues": ["list"]}}"""

        try:
            response = self.llm_model.generate(eval_prompt)

            # Extract JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result_dict = json.loads(json_str)
            else:
                result_dict = {
                    "teleportation": 80,
                    "gravity": 80,
                    "permanence": 80,
                    "issues": ["Failed to parse response"],
                }

            # Normalize scores
            teleport_score = min(max(result_dict.get("teleportation", 80) / 100.0, 0.0), 1.0)
            gravity_score = min(max(result_dict.get("gravity", 80) / 100.0, 0.0), 1.0)
            permanence_score = min(max(result_dict.get("permanence", 80) / 100.0, 0.0), 1.0)

            # Aggregate
            score = (
                teleport_score * 0.35 +
                gravity_score * 0.35 +
                permanence_score * 0.30
            )

            feedback = (
                f"  Teleportation: {teleport_score:.3f}\n"
                f"  Gravity: {gravity_score:.3f}\n"
                f"  Permanence: {permanence_score:.3f}"
            )

            issues = result_dict.get("issues", [])
            suggestions = []
            if teleport_score < 0.7:
                suggestions.append("Add motion continuity constraints")
            if gravity_score < 0.7:
                suggestions.append("Emphasize ground/floor anchors")
            if permanence_score < 0.7:
                suggestions.append("Add object persistence constraints")

            return AgentResult(
                score=min(max(score, 0.0), 1.0),
                feedback=feedback,
                suggestions=suggestions,
                metadata={
                    "teleportation": teleport_score,
                    "gravity": gravity_score,
                    "permanence": permanence_score,
                    "issues": issues,
                },
            )

        except Exception as e:
            print(f"Warning: PhysicsValidator evaluation failed: {e}")
            return AgentResult(
                score=0.70,
                feedback=f"Evaluation error: {str(e)[:50]}",
                suggestions=["Retry with clearer scene description"],
                metadata={"error": str(e)},
            )
