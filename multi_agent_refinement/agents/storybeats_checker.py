"""StorybeatsChecker Agent - Validates narrative coherence"""

import json
from typing import Any, List, Optional
from ..agent_base import Agent, AgentResult, extract_first_json_object


class StorybeatsChecker(Agent):
    """
    Checks if generated video advances the narrative beat properly.

    Uses LLM to analyze:
    - Does the video match the narrative beat?
    - Is the character action clear?
    - Does it advance the plot?
    """

    def __init__(self, llm_model: Any, weight: float = 0.40):
        """
        Args:
            llm_model: Language model for evaluation (from pipeline)
            weight: Importance in aggregate scoring
        """
        super().__init__("StorybeatsChecker", weight)
        self.llm_model = llm_model

    def evaluate(
        self,
        window_beat: str,
        generated_captions: List[str],
        previous_beat: Optional[str] = None,
        character_constraints: Optional[str] = None,
    ) -> AgentResult:
        """
        Score narrative coherence.

        Args:
            window_beat: Target story beat for this window
            generated_captions: Per-frame captions from generated video
            previous_beat: Previous beat (for continuity context)
            character_constraints: Character constraints/roles

        Returns:
            AgentResult with narrative score (0-1)
        """
        # Build evaluation prompt for LLM
        captions_text = "\n".join(
            [f"Frame {i}: {cap}" for i, cap in enumerate(generated_captions)]
        )

        eval_prompt = f"""You are a narrative critic evaluating generated video.

Target Story Beat: {window_beat}
{f"Previous Beat: {previous_beat}" if previous_beat else ""}
{f"Character Context: {character_constraints}" if character_constraints else ""}

Generated Video Frame Captions:
{captions_text}

Rate these aspects 0-100:
1. BEAT_ADHERENCE: Does the video show this narrative beat?
2. CHARACTER_CLARITY: Are characters and their actions clear?
3. PLOT_ADVANCEMENT: Does it advance the story forward?

Respond ONLY with JSON:
{{"beat_adherence": <0-100>, "character_clarity": <0-100>, "plot_advancement": <0-100>, "issues": ["list", "of", "issues"]}}"""

        try:
            # Call LLM
            response = self.llm_model.generate(eval_prompt)

            result_dict = extract_first_json_object(response)
            if result_dict is None:
                result_dict = {
                    "beat_adherence": 70,
                    "character_clarity": 70,
                    "plot_advancement": 70,
                    "issues": ["Failed to parse LLM response"],
                }

            # Normalize scores to 0-1
            beat_adherence = min(max(result_dict.get("beat_adherence", 70) / 100.0, 0.0), 1.0)
            character_clarity = min(max(result_dict.get("character_clarity", 70) / 100.0, 0.0), 1.0)
            plot_advancement = min(max(result_dict.get("plot_advancement", 70) / 100.0, 0.0), 1.0)

            # Weighted aggregate
            score = (
                beat_adherence * 0.40 +
                character_clarity * 0.35 +
                plot_advancement * 0.25
            )

            # Format feedback
            feedback = (
                f"  Beat Adherence: {beat_adherence:.3f}\n"
                f"  Character Clarity: {character_clarity:.3f}\n"
                f"  Plot Advancement: {plot_advancement:.3f}"
            )

            issues = result_dict.get("issues", [])
            suggestions = []
            if beat_adherence < 0.7:
                suggestions.append("Strengthen beat description in next prompt")
            if character_clarity < 0.7:
                suggestions.append("Add more character action details")
            if plot_advancement < 0.7:
                suggestions.append("Emphasize plot progression in prompt")

            return AgentResult(
                score=min(max(score, 0.0), 1.0),
                feedback=feedback,
                suggestions=suggestions,
                metadata={
                    "beat_adherence": beat_adherence,
                    "character_clarity": character_clarity,
                    "plot_advancement": plot_advancement,
                    "issues": issues,
                },
            )

        except Exception as e:
            print(f"Warning: StorybeatsChecker evaluation failed: {e}")
            return AgentResult(
                score=0.65,  # Neutral score on error
                feedback=f"Evaluation error: {str(e)[:50]}",
                suggestions=["Retry with clearer captions"],
                metadata={"error": str(e)},
            )
