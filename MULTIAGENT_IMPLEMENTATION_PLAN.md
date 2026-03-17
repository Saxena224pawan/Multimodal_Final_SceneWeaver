# Multi-Agent Iterative Refinement Implementation Plan

## Vision
Transform SceneWeaver from single-pass generation → iterative multi-agent refinement pipeline
- **Goal**: Each window passes through agent checks; regenerate if quality threshold not met
- **Paper Title**: "Multi-Agent Iterative Refinement for Coherent Story-to-Video Generation"
- **Timeline**: 4-6 weeks to publishable prototype

---

## Architecture Overview

```
CURRENT PIPELINE:
Storyline → SceneDirector → LLM Prompt → Video Generation → Store Output

NEW PIPELINE WITH MULTI-AGENT:
Storyline → SceneDirector
         ↓
    Base Prompt (v0)
         ↓
    [ITERATIVE REFINEMENT LOOP]
    ├─ Generate Video (v_current)
    ├─ Agent 1: StorybeatsChecker ──→ score_narrative
    ├─ Agent 2: ContinuityAuditor ──→ score_continuity
    ├─ Agent 3: PhysicsValidator ──→ score_physics
    ├─ Aggregate Score = w1*score_narrative + w2*score_continuity + w3*score_physics
    │
    ├─ IF aggregate_score >= THRESHOLD → ACCEPT (go to next window)
    └─ ELSE → Agent 4: PromptOptimizer ──→ refined_prompt (v_current+1)
              Loop back to Generate Video
         ↓
    Store Output + Metadata (iteration count, scores history)
```

---

## Directory Structure (New Files)

```
multi_agent_refinement/
├── __init__.py
├── agent_base.py                 # Base agent class
├── agents/
│   ├── __init__.py
│   ├── storybeats_checker.py     # Agent 1
│   ├── continuity_auditor.py     # Agent 2
│   ├── physics_validator.py      # Agent 3
│   └── prompt_optimizer.py       # Agent 4
├── refinement_engine.py          # Orchestrator
├── evaluation_metrics.py          # Scoring functions
└── configs/
    └── agent_config.yaml         # Agent hyperparameters
```

---

## Step 1: Create Agent Base Class

**File**: `multi_agent_refinement/agent_base.py`

### Design
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass
class AgentResult:
    """Standard output from any agent"""
    score: float                    # 0-1, higher is better
    feedback: str                   # Natural language feedback
    suggestions: List[str]          # Actionable suggestions
    metadata: Dict[str, Any]        # Agent-specific metadata

class Agent(ABC):
    """Base class for all agents"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight        # Used in aggregation

    @abstractmethod
    def evaluate(self, **kwargs) -> AgentResult:
        """
        Evaluate a generated window.
        Must be implemented by subclasses.
        """
        pass

    def __repr__(self):
        return f"{self.name}(weight={self.weight})"
```

---

## Step 2: Implement Agent 1 - StorybeatsChecker

**File**: `multi_agent_refinement/agents/storybeats_checker.py`

### Purpose
Ensures generated window advances the narrative

### Implementation
```python
from agent_base import Agent, AgentResult
from typing import List, Dict
import json

class StorybeatsChecker(Agent):
    """
    Checks if generated video advances the story beats.

    Uses LLM to analyze:
    - Does the video match the narrative beat?
    - Is the character action clear?
    - Does it advance the plot?
    """

    def __init__(self, llm_model, weight: float = 0.4):
        super().__init__("StorybeatsChecker", weight)
        self.llm_model = llm_model  # Qwen or similar

    def evaluate(
        self,
        window_beat: str,           # "Hero finds the artifact in the forest"
        generated_captions: List[str],  # Per-frame captions from captioner
        previous_beat: str = None,  # Previous beat context
        character_constraints: str = None
    ) -> AgentResult:
        """
        Score narrative coherence
        """

        # Construct prompt for LLM
        evaluation_prompt = f"""
You are a narrative critic for story-to-video generation.

Target Story Beat: {window_beat}
Previous Beat: {previous_beat}
Character Constraints: {character_constraints}

Generated Video Captions (per-frame):
{json.dumps(generated_captions, indent=2)}

Evaluate the generated video on these criteria:
1. **Beat Adherence** (0-1): Does the video show this narrative beat?
2. **Character Consistency** (0-1): Are characters acting as expected?
3. **Plot Advancement** (0-1): Does it advance the story forward?
4. **Clarity** (0-1): Is the intended action clear to viewers?

Respond in JSON format:
{{
  "beat_adherence": 0.85,
  "character_consistency": 0.9,
  "plot_advancement": 0.75,
  "clarity": 0.8,
  "issues": ["List of specific issues"],
  "suggestions": ["How to improve"]
}}
"""

        # Call LLM
        response = self.llm_model.generate(evaluation_prompt)
        result_dict = json.loads(response)

        # Compute aggregate score
        score = (
            result_dict["beat_adherence"] * 0.4 +
            result_dict["character_consistency"] * 0.25 +
            result_dict["plot_advancement"] * 0.25 +
            result_dict["clarity"] * 0.1
        )

        feedback = f"""
Beat Adherence: {result_dict['beat_adherence']:.2f}
Character Consistency: {result_dict['character_consistency']:.2f}
Plot Advancement: {result_dict['plot_advancement']:.2f}
Clarity: {result_dict['clarity']:.2f}

Issues: {', '.join(result_dict['issues'])}
        """

        return AgentResult(
            score=score,
            feedback=feedback,
            suggestions=result_dict["suggestions"],
            metadata=result_dict
        )
```

---

## Step 3: Implement Agent 2 - ContinuityAuditor

**File**: `multi_agent_refinement/agents/continuity_auditor.py`

### Purpose
Ensures visual consistency between windows

### Implementation
```python
from agent_base import Agent, AgentResult
import torch
from typing import Optional, Tuple
import numpy as np

class ContinuityAuditor(Agent):
    """
    Checks visual continuity between consecutive windows.

    Uses:
    - CLIP embeddings (existing in pipeline)
    - Character tracking
    - Environment anchors
    """

    def __init__(self, embedding_model, weight: float = 0.35):
        super().__init__("ContinuityAuditor", weight)
        self.embedding_model = embedding_model  # CLIP or DINOv2

    def evaluate(
        self,
        current_frames: torch.Tensor,      # Shape: (N, 3, H, W), N frames from current window
        previous_frames: Optional[torch.Tensor] = None,  # Last frames from previous window
        character_names: Optional[list] = None,
        scene_location: str = None
    ) -> AgentResult:
        """
        Score visual continuity
        """

        scores = {}
        issues = []
        suggestions = []

        # 1. Character Consistency Check
        if character_names and previous_frames is not None:
            char_score = self._check_character_consistency(
                current_frames,
                previous_frames,
                character_names
            )
            scores['character'] = char_score
            if char_score < 0.7:
                issues.append("Character appearance changed significantly")
                suggestions.append("Regenerate with stronger character constraints")

        # 2. Scene Continuity Check
        if scene_location:
            scene_score = self._check_scene_consistency(
                current_frames,
                scene_location
            )
            scores['scene'] = scene_score
            if scene_score < 0.7:
                issues.append("Scene/location inconsistent")
                suggestions.append("Anchor location descriptor in prompt")

        # 3. Motion Smoothness Check
        if previous_frames is not None:
            motion_score = self._check_motion_smoothness(
                previous_frames[-1:],  # Last frame of previous
                current_frames[:1]     # First frame of current
            )
            scores['motion'] = motion_score
            if motion_score < 0.7:
                issues.append("Abrupt transition between windows")
                suggestions.append("Add transition frame guidance")

        # Aggregate
        aggregate_score = np.mean(list(scores.values()))

        feedback = "\n".join([f"{k}: {v:.2f}" for k, v in scores.items()])

        return AgentResult(
            score=aggregate_score,
            feedback=feedback,
            suggestions=suggestions,
            metadata=scores
        )

    def _check_character_consistency(self, current, previous, names):
        """Use CLIP to check character embeddings are similar"""
        with torch.no_grad():
            prev_emb = self.embedding_model(previous)  # (N, 512)
            curr_emb = self.embedding_model(current)   # (M, 512)

            # Compare last frame of previous to first frame of current
            similarity = torch.cosine_similarity(
                prev_emb[-1:],  # Shape: (1, 512)
                curr_emb[:1]    # Shape: (1, 512)
            )
        return float(similarity.item())

    def _check_scene_consistency(self, frames, scene_desc):
        """Verify scene descriptor matches frames"""
        # Generate caption with descriptive prompts
        with torch.no_grad():
            emb = self.embedding_model(frames)  # (N, 512)
            scene_emb = self.embedding_model.encode_text(scene_desc)  # (512,)

            # Average similarity across frames
            similarity = torch.nn.functional.cosine_similarity(
                emb,
                scene_emb.unsqueeze(0)
            ).mean()
        return float(similarity.item())

    def _check_motion_smoothness(self, last_prev, first_curr):
        """Check if transition is smooth"""
        with torch.no_grad():
            emb_prev = self.embedding_model(last_prev)
            emb_curr = self.embedding_model(first_curr)
            similarity = torch.cosine_similarity(emb_prev, emb_curr)
        return float(similarity.item())
```

---

## Step 4: Implement Agent 3 - PhysicsValidator

**File**: `multi_agent_refinement/agents/physics_validator.py`

### Purpose
Checks plausibility: no teleports, gravity, object permanence

### Implementation
```python
from agent_base import Agent, AgentResult
from typing import List, Dict
import json

class PhysicsValidator(Agent):
    """
    Validates physical plausibility of generated video.

    Checks:
    - No character teleportation
    - Gravity obeyed
    - Object permanence
    - Camera smoothness
    """

    def __init__(self, caption_model, llm_model, weight: float = 0.25):
        super().__init__("PhysicsValidator", weight)
        self.caption_model = caption_model  # For frame descriptions
        self.llm_model = llm_model

    def evaluate(
        self,
        frames: List,                   # Video frames
        scene_constraints: str = None   # "Indoor, no flying"
        character_positions: Dict = None  # {char_name: expected_position}
    ) -> AgentResult:
        """
        Score physical plausibility
        """

        issues = []
        suggestions = []

        # Generate captions for analysis
        captions = [self.caption_model.caption(f) for f in frames]

        # 1. Teleportation Check
        teleport_score = self._check_no_teleportation(captions)
        if teleport_score < 0.8:
            issues.append("Possible character teleportation detected")
            suggestions.append("Add motion continuity constraints")

        # 2. Gravity Check
        gravity_score = self._check_gravity_plausibility(captions)
        if gravity_score < 0.8:
            issues.append("Gravity violation detected")
            suggestions.append("Emphasize ground/floor in prompt")

        # 3. Object Permanence
        if character_positions:
            permanence_score = self._check_object_permanence(captions, character_positions)
        else:
            permanence_score = 0.9  # Skip if no tracking data

        if permanence_score < 0.7:
            issues.append("Object/character disappeared unexpectedly")
            suggestions.append("Add persistence constraints")

        # Use LLM for final veto
        llm_veto_score = self._llm_physics_check(captions, scene_constraints)

        # Aggregate
        aggregate_score = (
            teleport_score * 0.3 +
            gravity_score * 0.3 +
            permanence_score * 0.2 +
            llm_veto_score * 0.2
        )

        return AgentResult(
            score=aggregate_score,
            feedback=f"Teleport: {teleport_score:.2f}, Gravity: {gravity_score:.2f}, Permanence: {permanence_score:.2f}, LLM-Veto: {llm_veto_score:.2f}",
            suggestions=suggestions,
            metadata={"issues": issues}
        )

    def _check_no_teleportation(self, captions: List[str]) -> float:
        """Analyze captions for sudden position changes"""
        prompt = f"""
Analyze these video frame captions for character teleportation (sudden position change):
{json.dumps(captions, indent=2)}

Rate 0-1: How physically plausible are the positions? (1=no teleportation, 0=obvious jumps)
Return ONLY a number.
        """
        response = float(self.llm_model.generate(prompt).strip())
        return min(max(response, 0), 1)  # Clamp 0-1

    def _check_gravity_plausibility(self, captions: List[str]) -> float:
        """Check if gravity is respected"""
        prompt = f"""
Do these captions describe physically plausible motion given gravity?
{json.dumps(captions, indent=2)}

Rate 0-1: How well is gravity respected? (1=realistic, 0=defies physics)
Return ONLY a number.
        """
        response = float(self.llm_model.generate(prompt).strip())
        return min(max(response, 0), 1)

    def _check_object_permanence(self, captions: List[str], tracking: Dict) -> float:
        """Check if objects stay present"""
        prompt = f"""
Expected objects to track: {json.dumps(tracking)}

Do the following captions show these objects consistently?
{json.dumps(captions, indent=2)}

Rate 0-1: How consistently are tracked objects present? (1=always, 0=keeps disappearing)
Return ONLY a number.
        """
        response = float(self.llm_model.generate(prompt).strip())
        return min(max(response, 0), 1)

    def _llm_physics_check(self, captions: List[str], constraints: str = None) -> float:
        """Final LLM veto check"""
        prompt = f"""
Scene constraints: {constraints or 'None specified'}

Frame sequence captions:
{json.dumps(captions, indent=2)}

Rate 0-1: Overall physical plausibility? (1=realistic, 0=absurd)
Return ONLY a number.
        """
        response = float(self.llm_model.generate(prompt).strip())
        return min(max(response, 0), 1)
```

---

## Step 5: Implement Agent 4 - PromptOptimizer

**File**: `multi_agent_refinement/agents/prompt_optimizer.py`

### Purpose
Analyzes failures and produces improved prompts

### Implementation
```python
from agent_base import Agent, AgentResult
from typing import List, Dict
import json

class PromptOptimizer(Agent):
    """
    Takes feedback from all agents and generates improved prompts.

    This is a meta-agent: uses LLM to interpret feedback and rewrite prompt.
    """

    def __init__(self, llm_model, weight: float = 1.0):
        super().__init__("PromptOptimizer", weight)
        self.llm_model = llm_model

    def optimize(
        self,
        base_prompt: str,
        agent_feedback: Dict[str, 'AgentResult'],  # {agent_name: AgentResult}
        iteration_count: int,
        narrative_beat: str
    ) -> str:
        """
        Generate improved prompt based on feedback

        Args:
            base_prompt: Original/current prompt
            agent_feedback: Feedbacks from StorybeatsChecker, ContinuityAuditor, etc.
            iteration_count: How many times we've tried (to avoid over-optimization)
            narrative_beat: Current story beat target

        Returns:
            Improved prompt for next generation attempt
        """

        # Summarize feedback
        feedback_summary = self._summarize_feedback(agent_feedback)

        # Check if we should take more aggressive action
        avg_score = sum(f.score for f in agent_feedback.values()) / len(agent_feedback)
        severity = "high" if avg_score < 0.5 else "medium" if avg_score < 0.7 else "low"

        # Construct optimization prompt
        optimization_prompt = f"""
You are a prompt optimization expert for video generation.

CURRENT SITUATION:
- Target Narrative Beat: {narrative_beat}
- Iteration: {iteration_count}
- Overall Quality Score: {avg_score:.2f}/1.0 (severity: {severity})

ORIGINAL PROMPT:
{base_prompt}

AGENT FEEDBACK:
{feedback_summary}

YOUR TASK:
Rewrite the prompt to address the specific failures noted above.

CRITICAL GUIDELINES:
1. Keep the narrative intent of the original beat
2. Be SPECIFIC: Instead of "more detail", say "add 3 more trees"
3. Strengthen weak areas: If continuity is failing, add character position constraints
4. Add constraints that prevent the observed failures
5. Be concise: Prompts should stay under 200 words

{"" if iteration_count < 3 else "NOTE: This is attempt #" + str(iteration_count) + ". Make SUBSTANTIAL changes."}

IMPROVED PROMPT:
"""

        improved_prompt = self.llm_model.generate(optimization_prompt)
        return improved_prompt

    def _summarize_feedback(self, agent_feedback: Dict) -> str:
        """Format feedback from all agents into readable summary"""
        summary = []
        for agent_name, result in agent_feedback.items():
            summary.append(f"""
{agent_name} (Score: {result.score:.2f}/1.0):
  Feedback: {result.feedback}
  Suggestions: {', '.join(result.suggestions)}
            """)
        return "\n".join(summary)
```

---

## Step 6: Build Refinement Engine

**File**: `multi_agent_refinement/refinement_engine.py`

### Purpose
Main orchestrator that runs the loop

### Implementation
```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from datetime import datetime

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


class RefinementEngine:
    """
    Orchestrates multi-agent iterative refinement loop.
    """

    def __init__(
        self,
        agents: Dict,  # {agent_name: Agent instance}
        video_model,
        caption_model,
        embedding_model,
        config: Dict
    ):
        self.agents = agents
        self.video_model = video_model
        self.caption_model = caption_model
        self.embedding_model = embedding_model

        # Config
        self.max_iterations = config.get("max_iterations", 5)
        self.quality_threshold = config.get("quality_threshold", 0.75)
        self.weights = config.get("agent_weights", {})

    def refine_window(
        self,
        base_prompt: str,
        narrative_beat: str,
        window_idx: int,
        previous_frames: Optional[list] = None,
        character_names: Optional[list] = None,
        scene_location: str = None
    ) -> tuple:
        """
        Run iterative refinement for a single window.

        Returns:
            (generated_frames, metadata)
        """

        metadata = RefinementMetadata(
            window_idx=window_idx,
            narrative_beat=narrative_beat,
            total_iterations=0
        )

        current_prompt = base_prompt
        current_frames = None

        for iteration in range(self.max_iterations):
            print(f"\n{'='*60}")
            print(f"Window {window_idx} | Iteration {iteration + 1}/{self.max_iterations}")
            print(f"{'='*60}")

            # STEP 1: Generate video
            print(f"Generating video with prompt...")
            current_frames = self.video_model.generate(current_prompt)
            captions = self._get_captions(current_frames)

            # STEP 2: Run all agents
            print(f"Running agents...")
            agent_scores = {}
            agent_results = {}

            # Agent 1: StorybeatsChecker
            story_result = self.agents["storybeats_checker"].evaluate(
                window_beat=narrative_beat,
                generated_captions=captions,
                previous_beat=None
            )
            agent_results["storybeats"] = story_result
            agent_scores["storybeats"] = story_result.score
            print(f"  StorybeatsChecker: {story_result.score:.3f}")

            # Agent 2: ContinuityAuditor
            continuity_result = self.agents["continuity_auditor"].evaluate(
                current_frames=current_frames,
                previous_frames=previous_frames,
                character_names=character_names,
                scene_location=scene_location
            )
            agent_results["continuity"] = continuity_result
            agent_scores["continuity"] = continuity_result.score
            print(f"  ContinuityAuditor: {continuity_result.score:.3f}")

            # Agent 3: PhysicsValidator
            physics_result = self.agents["physics_validator"].evaluate(
                frames=current_frames,
                character_positions={c: "present" for c in (character_names or [])}
            )
            agent_results["physics"] = physics_result
            agent_scores["physics"] = physics_result.score
            print(f"  PhysicsValidator: {physics_result.score:.3f}")

            # STEP 3: Compute aggregate score
            aggregate_score = self._compute_aggregate_score(agent_scores)
            print(f"\nAggregate Score: {aggregate_score:.3f}")

            metadata.scores_history.append(aggregate_score)
            metadata.agents_history.append({
                "iteration": iteration + 1,
                "scores": agent_scores,
                "aggregate": aggregate_score
            })

            # STEP 4: Check threshold
            if aggregate_score >= self.quality_threshold:
                print(f"\n✅ QUALITY THRESHOLD MET ({aggregate_score:.3f} >= {self.quality_threshold})")
                metadata.total_iterations = iteration + 1
                metadata.final_prompt = current_prompt
                break

            # STEP 5: Continue? Or give up?
            if iteration == self.max_iterations - 1:
                print(f"\n⚠️  MAX ITERATIONS REACHED")
                metadata.total_iterations = iteration + 1
                metadata.final_prompt = current_prompt
                break

            # STEP 6: Optimize prompt
            print(f"\nOptimizing prompt (iteration {iteration + 2})...")
            optimizer = self.agents["prompt_optimizer"]
            current_prompt = optimizer.optimize(
                base_prompt=current_prompt,
                agent_feedback=agent_results,
                iteration_count=iteration + 1,
                narrative_beat=narrative_beat
            )
            print(f"New prompt:\n{current_prompt[:150]}...\n")

        return current_frames, metadata

    def _get_captions(self, frames: list) -> List[str]:
        """Get captions for frames"""
        return [self.caption_model.caption(f) for f in frames]

    def _compute_aggregate_score(self, agent_scores: Dict[str, float]) -> float:
        """Weighted average of agent scores"""
        weighted_sum = 0
        total_weight = 0

        for agent_name, score in agent_scores.items():
            weight = self.weights.get(agent_name, 1.0)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5
```

---

## Step 7: Integration with Existing Pipeline

**File**: `scripts/run_story_pipeline_with_agents.py`

### Modify existing pipeline to use refinement engine

```python
# New top-level script
from multi_agent_refinement.refinement_engine import RefinementEngine
from multi_agent_refinement.agents.storybeats_checker import StorybeatsChecker
from multi_agent_refinement.agents.continuity_auditor import ContinuityAuditor
from multi_agent_refinement.agents.physics_validator import PhysicsValidator
from multi_agent_refinement.agents.prompt_optimizer import PromptOptimizer

def main():
    # Load models (existing code)
    video_model = load_wan_model(model_id)
    director = load_director(director_model_id)
    embedding_model = load_embedder("clip")
    caption_model = load_captioner("blip2")
    llm_model = load_llm("qwen-3b")

    # Initialize agents
    agents = {
        "storybeats_checker": StorybeatsChecker(llm_model, weight=0.4),
        "continuity_auditor": ContinuityAuditor(embedding_model, weight=0.35),
        "physics_validator": PhysicsValidator(caption_model, llm_model, weight=0.25),
        "prompt_optimizer": PromptOptimizer(llm_model)
    }

    # Load config
    agent_config = {
        "max_iterations": 5,
        "quality_threshold": 0.75,
        "agent_weights": {"storybeats": 0.4, "continuity": 0.35, "physics": 0.25}
    }

    # Initialize refinement engine
    refinement_engine = RefinementEngine(
        agents=agents,
        video_model=video_model,
        caption_model=caption_model,
        embedding_model=embedding_model,
        config=agent_config
    )

    # Load story (existing)
    storyline = load_storyline(args.storyline)
    windows = director.plan_windows(storyline)

    # NEW: Use refinement engine instead of direct generation
    outputs = []
    for window_idx, window in enumerate(windows):
        frames, metadata = refinement_engine.refine_window(
            base_prompt=window.prompt,
            narrative_beat=window.beat,
            window_idx=window_idx,
            previous_frames=outputs[-1]["frames"] if outputs else None,
            character_names=window.characters,
            scene_location=window.location
        )

        outputs.append({
            "frames": frames,
            "metadata": metadata,
            "beat": window.beat
        })

        # Save iteration metadata to JSON for analysis
        save_metadata(metadata, f"window_{window_idx:03d}_metadata.json")

    print(f"\n✅ Pipeline complete!")
```

---

## Step 8: Evaluation & Metrics

**File**: `multi_agent_refinement/evaluation_metrics.py`

```python
import numpy as np
from typing import List, Dict

class EvaluationMetrics:
    """Compute metrics for multi-agent refinement"""

    @staticmethod
    def convergence_speed(metadata_history: List) -> Dict:
        """How quickly does quality improve?"""
        scores = [m.scores_history for m in metadata_history]

        return {
            "avg_iterations": np.mean([m.total_iterations for m in metadata_history]),
            "max_iterations": np.max([m.total_iterations for m in metadata_history]),
            "converged_ratio": sum(1 for m in metadata_history if m.total_iterations < 5) / len(metadata_history)
        }

    @staticmethod
    def quality_improvement(metadata: object) -> float:
        """Score improvement from first to last iteration"""
        if len(metadata.scores_history) < 2:
            return 0.0
        return metadata.scores_history[-1] - metadata.scores_history[0]

    @staticmethod
    def agent_importance(agent_feedback_list: List[Dict]) -> Dict[str, float]:
        """Which agents contribute most to refinement decisions?"""
        agent_vote_counts = {}

        for feedback in agent_feedback_list:
            min_score_agent = min(feedback.scores.items(), key=lambda x: x[1])
            agent_name = min_score_agent[0]
            agent_vote_counts[agent_name] = agent_vote_counts.get(agent_name, 0) + 1

        total = sum(agent_vote_counts.values())
        return {k: v/total for k, v in agent_vote_counts.items()}

    @staticmethod
    def threshold_efficiency(metadatas: List) -> float:
        """What fraction hit threshold? (higher = better pipeline)"""
        hit_threshold = sum(1 for m in metadatas if m.scores_history[-1] >= 0.75)
        return hit_threshold / len(metadatas) if metadatas else 0.0
```

---

## Implementation Timeline

### Week 1: Foundation
- [ ] Create agent base class
- [ ] Implement StorybeatsChecker & ContinuityAuditor
- [ ] Basic refinement loop integration

### Week 2: Complete Agents
- [ ] Implement PhysicsValidator
- [ ] Implement PromptOptimizer
- [ ] Build RefinementEngine orchestrator

### Week 3: Integration & Testing
- [ ] Integrate with existing pipeline
- [ ] Run on 5-10 test stories
- [ ] Collect iteration metadata

### Week 4: Evaluation & Analysis
- [ ] Compute convergence metrics
- [ ] Run ablation studies (disable each agent → see impact)
- [ ] Compare single-pass vs. multi-agent

### Week 5-6: Paper Preparation
- [ ] Write paper structure
- [ ] Create visualizations (score progression, agent voting)
- [ ] Submit to ACL/AAAI

---

## Config File

**File**: `multi_agent_refinement/configs/agent_config.yaml`

```yaml
refinement:
  max_iterations: 5
  quality_threshold: 0.75
  early_stopping: true

agents:
  storybeats_checker:
    enabled: true
    weight: 0.4
    llm_model: "qwen-3b"

  continuity_auditor:
    enabled: true
    weight: 0.35
    embedding_backend: "clip"  # or "dinov2"

  physics_validator:
    enabled: true
    weight: 0.25
    llm_model: "qwen-3b"

  prompt_optimizer:
    enabled: true
    optimization_strategy: "llm"

logging:
  save_metadata: true
  save_iteration_frames: false  # Too expensive
  verbose: true
```

---

## Success Metrics for Research

### Metric 1: Convergence Speed ✅
```
Target: Avg 2.5 iterations to quality threshold
Current: Not applicable (single-pass)
Expected: 60% of windows hit threshold by iteration 3
```

### Metric 2: Quality Improvement
```
Target: Multi-agent avg score: 0.80 (vs. baseline single-pass: 0.65)
Measure: Across 100 test windows
```

### Metric 3: Agent Contribution
```
Show ablation: Remove each agent → score drop
Example: Remove PhysicsValidator → score drops 0.08
```

### Metric 4: Cost-Benefit
```
Ask: Is 2 extra iterations worth 15% quality boost?
Cost: 3x compute per window
Benefit: Better narrative + continuity + physics
ROI: Yes if human eval prefers refined outputs 80%+ of time
```

---

## What This Gives You (Paper-wise)

✅ **Novel Framework**: First multi-agent iterative refinement for story-to-video
✅ **Methodology**: Detailed agent design + scoring
✅ **Experiments**: Convergence analysis, ablation studies, user evaluation
✅ **Contribution**: Published pipeline + code released
✅ **Impact**: 15-20% quality improvement over single-pass

---

**Ready to code?** Want me to start implementing these agents step-by-step?
