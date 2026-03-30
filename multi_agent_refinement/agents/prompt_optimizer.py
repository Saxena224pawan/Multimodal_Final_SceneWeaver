"""PromptOptimizer Agent - Refines prompts based on feedback"""

from typing import Any, Dict
from ..agent_base import AgentResult, Agent


class PromptOptimizer(Agent):
    """
    Meta-agent that analyzes failure feedback and generates improved prompts.

    Uses LLM to rewrite prompts based on agent feedback.
    """

    def __init__(self, llm_model: Any, weight: float = 1.0):
        """
        Args:
            llm_model: Language model for prompt optimization
            weight: Not used in aggregate (meta-agent)
        """
        super().__init__("PromptOptimizer", weight)
        self.llm_model = llm_model

    def optimize(
        self,
        base_prompt: str,
        agent_feedback: Dict[str, AgentResult],
        iteration_count: int,
        narrative_beat: str,
    ) -> str:
        """
        Generate improved prompt based on agent feedback.

        Args:
            base_prompt: Current/original prompt
            agent_feedback: Feedback from StorybeatsChecker, ContinuityAuditor, etc.
            iteration_count: How many attempts so far (0, 1, 2...)
            narrative_beat: Current story beat target

        Returns:
            Refined prompt for next generation attempt
        """
        # Summarize feedback from all agents
        feedback_summary = self._summarize_feedback(agent_feedback)

        # Determine severity
        avg_score = sum(f.score for f in agent_feedback.values()) / len(agent_feedback)
        severity = "critical" if avg_score < 0.5 else "moderate" if avg_score < 0.7 else "minor"

        # Build optimization prompt
        opt_prompt = f"""You are a prompt optimization expert for video generation.

OBJECTIVE: Rewrite the prompt to address specific failures.

CURRENT STATUS:
- Target Narrative Beat: {narrative_beat}
- Iteration #{iteration_count + 1}
- Current Quality Score: {avg_score:.2f}/1.0 ({severity} issues)

ORIGINAL PROMPT:
{base_prompt}

FEEDBACK FROM AGENTS:
{feedback_summary}

YOUR TASK:
Rewrite the prompt to fix the identified issues. Be specific and concrete.

KEY GUIDELINES:
- Keep the core narrative intent
- Address each identified issue
- Be specific: "add 3 forest trees" not "more details"
- Add constraints that prevent observed failures
- Keep prompt under 200 words

{f"NOTE: This is attempt #{iteration_count + 1}. Make SUBSTANTIAL changes." if iteration_count > 0 else ""}

IMPROVED PROMPT:
"""

        try:
            improved_prompt = self.llm_model.generate(opt_prompt)
            improved_prompt = " ".join((improved_prompt or "").strip().split())
            if len(improved_prompt) == 0:
                return base_prompt
            if len(improved_prompt) > 900:
                return base_prompt
            return improved_prompt
        except Exception as e:
            print(f"Warning: Prompt optimization failed: {e}")
            return base_prompt

    def _summarize_feedback(self, agent_feedback: Dict[str, AgentResult]) -> str:
        """Format feedback from all agents into readable summary"""
        summary = []
        for agent_name, result in agent_feedback.items():
            issues_str = (
                ", ".join(result.metadata.get("issues", []))
                if result.metadata.get("issues")
                else "None detected"
            )
            suggestions_str = ", ".join(result.suggestions) if result.suggestions else "None"

            summary.append(
                f"""{agent_name} (Score: {result.score:.2f}/1.0):
    Feedback: {result.feedback}
    Issues: {issues_str}
    Suggestions: {suggestions_str}"""
            )
        return "\n".join(summary)

    def evaluate(self, **kwargs) -> AgentResult:
        """Not used - PromptOptimizer.optimize() is called directly"""
        raise NotImplementedError("Use optimize() method instead")
