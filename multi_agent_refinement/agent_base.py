"""Base classes and interfaces for multi-agent system"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentResult:
    """Standard output format for all agents"""
    score: float  # 0-1, higher is better
    feedback: str  # Natural language explanation
    suggestions: List[str] = field(default_factory=list)  # Actionable improvements
    metadata: Dict[str, Any] = field(default_factory=dict)  # Agent-specific data

    def __repr__(self) -> str:
        return f"AgentResult(score={self.score:.3f}, feedback='{self.feedback[:50]}...')"


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    payload = str(text or "")
    decoder = json.JSONDecoder()
    for idx, char in enumerate(payload):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(payload[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


class Agent(ABC):
    """Base class for all refinement agents"""

    def __init__(self, name: str, weight: float = 1.0):
        """
        Args:
            name: Agent identifier
            weight: Importance weight in aggregate scoring (0-1)
        """
        self.name = name
        self.weight = weight

    @abstractmethod
    def evaluate(self, **kwargs) -> AgentResult:
        """
        Evaluate generated window and return quality score.

        Must be implemented by subclasses.

        Returns:
            AgentResult with score, feedback, and suggestions
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}(weight={self.weight})"
