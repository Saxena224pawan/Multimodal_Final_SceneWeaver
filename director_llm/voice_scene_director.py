"""Voice-aware scene director - extends SceneDirector for audio-driven generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import from existing SceneDirector
from director_llm import SceneDirector, SceneDirectorConfig, SceneWindow


@dataclass
class VoiceSceneConfig:
    """Configuration for voice-driven scene generation."""
    base_director_config: Optional[SceneDirectorConfig] = None
    audio_duration_sec: float = 60.0
    allow_silence_ms: int = 500
    dialogue_window_multiplier: float = 1.2  # Longer windows for dialogue
    emotional_pause_ms: int = 800  # Pause duration for emotional beats
    sync_precision_ms: int = 100  # Precision for audio-visual sync


class VoiceSceneDirector:
    """
    Extends SceneDirector to handle voice script input.
    Maps audio timestamps to visual prompts with continuity.
    """

    def __init__(self, config: Optional[VoiceSceneConfig] = None):
        self.config = config or VoiceSceneConfig()
        if self.config.base_director_config is None:
            self.config.base_director_config = SceneDirectorConfig()
        self.base_director = SceneDirector(
            self.config.base_director_config,
            window_seconds=10
        )

    def plan_windows_from_voice_beats(
        self,
        voice_beats: List[Any],
        total_duration_sec: float,
    ) -> List[SceneWindow]:
        """
        Generate scene windows from parsed voice beats.
        
        Args:
            voice_beats: List of ParsedBeat objects with timing and text
            total_duration_sec: Total audio duration in seconds
            
        Returns:
            List of SceneWindow objects synchronized to audio
        """
        windows = []
        current_window_idx = 0

        # Sort beats by timing
        sorted_beats = sorted(voice_beats, key=lambda b: b.start_time_ms)

        for beat_idx, beat in enumerate(sorted_beats):
            # Calculate timing
            start_sec = beat.start_time_ms / 1000.0
            end_sec = beat.end_time_ms / 1000.0
            duration_sec = end_sec - start_sec

            # Adjust window duration based on beat type
            if beat.narrative_type == "dialogue":
                window_duration = duration_sec * self.config.dialogue_window_multiplier
            elif beat.narrative_type == "emotional":
                window_duration = duration_sec + (self.config.emotional_pause_ms / 1000.0)
            else:
                window_duration = duration_sec

            # Create environment anchor from beat context
            environment_anchor = self._build_environment_anchor(beat, beat_idx, sorted_beats)

            # Create character continuity lock
            character_lock = self._build_character_lock(beat)

            # Determine if scene change needed
            scene_change = self._detect_scene_change(beat, beat_idx, sorted_beats)

            # Generate refined prompt from beat
            prompt_seed = self._generate_prompt_from_beat(beat, sorted_beats, beat_idx)

            window = SceneWindow(
                index=current_window_idx,
                start_sec=int(start_sec),
                end_sec=int(end_sec),
                beat=beat.beat_text,
                prompt_seed=prompt_seed,
                scene_id=beat.scene_id,
                environment_anchor=environment_anchor,
                character_lock=character_lock,
                scene_change=scene_change,
                story_phase=self._get_story_phase(beat_idx, len(sorted_beats)),
            )
            windows.append(window)
            current_window_idx += 1

        return windows

    def _build_environment_anchor(
        self, beat: Any, beat_idx: int, all_beats: List[Any]
    ) -> str:
        """Build environment continuity anchor from beat text."""
        # Extract location keywords
        location_keywords = [
            "vineyard", "office", "stage", "screen", "studio",
            "గుండెల్లో", "సినిమా", "రంగస్థలం"
        ]

        environment = ""
        for keyword in location_keywords:
            if keyword.lower() in beat.beat_text.lower():
                environment = keyword
                break

        if not environment:
            environment = "documentary setting"

        # Add lighting context based on beat type
        if beat.narrative_type == "emotional":
            lighting = "soft, introspective lighting"
        elif beat.narrative_type == "action":
            lighting = "bright, clear lighting"
        else:
            lighting = "natural documentary lighting"

        return f"Consistent {environment}. {lighting}. Maintain visual continuity."

    def _build_character_lock(self, beat: Any) -> str:
        """Build character continuity constraints from beat."""
        # Main subject: N.T. Rama Rao or the concept/legacy
        character_lock = "Keep N.T. Rama Rao as central figure or representative"

        if beat.speaker_info:
            character_lock += f". Speaker: {beat.speaker_info}"

        # Add emotional state constraint
        if beat.narrative_type == "emotional":
            character_lock += ". Convey emotional depth and sincerity."
        elif beat.narrative_type == "action":
            character_lock += ". Show purposeful, disciplined movement."

        return character_lock

    def _detect_scene_change(
        self, beat: Any, beat_idx: int, all_beats: List[Any]
    ) -> bool:
        """Detect if scene should change (location shift)."""
        scene_change_keywords = ["office", "stage", "studio", "setting", "location", "രംഗം"]

        if any(kw in beat.beat_text.lower() for kw in scene_change_keywords):
            if beat_idx > 0:
                prev_beat = all_beats[beat_idx - 1]
                if prev_beat.scene_id != beat.scene_id:
                    return True

        return False

    def _generate_prompt_from_beat(
        self, beat: Any, all_beats: List[Any], beat_idx: int
    ) -> str:
        """Generate visual prompt from voice beat text."""
        # Start with beat narrative
        prompt = beat.beat_text

        # Add context from beat classification
        if beat.narrative_type == "emotional":
            prompt += " Show introspection, dignity, and emotional resonance."
        elif beat.narrative_type == "action":
            prompt += " Show purposeful action and physical discipline."
        elif beat.narrative_type == "dialogue":
            prompt += " Show natural speaking and engaged communication."

        # Add visual cue if extracted
        if beat.visual_cue:
            prompt += f" {beat.visual_cue}"

        # Add continuity hint from previous beat
        if beat_idx > 0:
            prev_beat = all_beats[beat_idx - 1]
            if prev_beat.scene_id == beat.scene_id:
                prompt += " Maintain visual continuity from previous moment."

        return prompt

    def _get_story_phase(self, beat_idx: int, total_beats: int) -> str:
        """Determine story phase (intro, rising, climax, resolution)."""
        if total_beats == 0:
            return "standalone"

        ratio = beat_idx / max(total_beats, 1)

        if ratio < 0.2:
            return "introduction"
        elif ratio < 0.5:
            return "rising_action"
        elif ratio < 0.8:
            return "climax"
        else:
            return "resolution"

    def export_scene_plan_json(
        self, windows: List[SceneWindow], output_path: str | Path
    ) -> None:
        """Export scene plan to JSON compatible with existing pipeline."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict format matching existing configs
        plan_data = []
        for window in windows:
            plan_data.append({
                "beat": window.beat,
                "scene_id": window.scene_id,
                "environment_anchor": window.environment_anchor,
                "character_lock": window.character_lock,
                "scene_change": window.scene_change,
                "story_phase": window.story_phase,
                "start_sec": window.start_sec,
                "end_sec": window.end_sec,
                "prompt_seed": window.prompt_seed,
            })

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(plan_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_scene_plan_json(path: str | Path) -> List[Dict[str, Any]]:
        """Load existing scene plan JSON."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
