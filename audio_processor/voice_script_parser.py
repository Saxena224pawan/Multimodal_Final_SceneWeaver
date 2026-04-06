"""Voice script parser - converts transcribed text into story beats and scenes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ParsedBeat:
    """Single narrative beat extracted from voice script."""
    index: int
    beat_text: str
    start_time_ms: int
    end_time_ms: int
    scene_id: str = ""
    is_dialogue: bool = False
    speaker_info: Optional[str] = None
    narrative_type: str = "action"  # action | dialogue | emotional | context
    visual_cue: str = ""  # Optional visual direction


@dataclass
class VoiceScriptParserConfig:
    """Configuration for voice script parsing."""
    min_beat_length: int = 50
    max_beat_length: int = 300
    detect_speakers: bool = True
    extract_visual_cues: bool = True
    language: str = "te"  # Telugu


class VoiceScriptParser:
    """
    Parses raw transcribed text from voice/audio into structured story beats.
    Handles Telugu + English mixed content, detects dialogue vs narration.
    """

    def __init__(self, config: Optional[VoiceScriptParserConfig] = None):
        self.config = config or VoiceScriptParserConfig()
        self._beat_separators = {
            "తెలుగు": ["।", "।।", "।"],  # Telugu punctuation
            "english": [".", "!", "?"],
        }

    def parse_transcript(
        self, raw_text: str, audio_segments: Optional[List[Any]] = None
    ) -> List[ParsedBeat]:
        """
        Parse raw transcribed text into story beats.
        
        Args:
            raw_text: Full transcribed text from audio
            audio_segments: Optional list of AudioSegment objects with timing
            
        Returns:
            List of ParsedBeat objects
        """
        # Clean and normalize text
        clean_text = self._normalize_text(raw_text)

        # Split into logical beats
        beat_texts = self._split_into_beats(clean_text)

        # Convert to ParsedBeat objects with timing
        beats = []
        for i, beat_text in enumerate(beat_texts):
            beat = self._create_parsed_beat(i, beat_text, audio_segments)
            if beat:
                beats.append(beat)

        return beats

    def _normalize_text(self, text: str) -> str:
        """Normalize and clean transcribed text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Normalize common variations
        text = text.replace("NTR", "N.T. Rama Rao")
        text = text.replace("ntrgaru", "N.T. Rama Rao")

        return text

    def _split_into_beats(self, text: str) -> List[str]:
        """Split text into narrative beats based on punctuation and content."""
        # Split by sentence boundaries (both English and Telugu)
        separators = r"(?<=[।.!?])\s+|(?<=,)\s+"
        raw_sentences = re.split(separators, text)

        # Group sentences into logical beats
        beats = []
        current_beat = []
        current_length = 0

        for sent in raw_sentences:
            if not sent.strip():
                continue

            sent_len = len(sent)

            # Start new beat if current one is getting too long
            if (
                current_length + sent_len > self.config.max_beat_length
                and current_beat
            ):
                beats.append(" ".join(current_beat))
                current_beat = [sent]
                current_length = sent_len
            else:
                current_beat.append(sent)
                current_length += sent_len

        if current_beat:
            beats.append(" ".join(current_beat))

        # Filter out beats that are too short (but be lenient for short inputs)
        result_beats = [b.strip() for b in beats if len(b.strip()) >= self.config.min_beat_length]
        
        # If filtering removed everything, return as-is (don't lose content)
        if not result_beats and beats:
            result_beats = [b.strip() for b in beats if b.strip()]
        
        return result_beats

    def _create_parsed_beat(
        self, index: int, beat_text: str, audio_segments: Optional[List[Any]] = None
    ) -> Optional[ParsedBeat]:
        """Convert beat text into ParsedBeat with metadata."""
        if not beat_text.strip():
            return None

        # Detect if beat is dialogue or narration
        is_dialogue = self._is_dialogue(beat_text)

        # Extract speaker info if present
        speaker_info = self._extract_speaker(beat_text) if is_dialogue else None

        # Determine narrative type
        narrative_type = self._classify_narrative_type(beat_text)

        # Extract visual cues from beat text
        visual_cue = self._extract_visual_cue(beat_text) if self.config.extract_visual_cues else ""

        # Generate scene ID
        scene_id = self._generate_scene_id(beat_text, narrative_type)

        # Estimate timing if segments provided
        start_ms, end_ms = self._estimate_timing(index, beat_text, audio_segments)

        return ParsedBeat(
            index=index,
            beat_text=beat_text,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            scene_id=scene_id,
            is_dialogue=is_dialogue,
            speaker_info=speaker_info,
            narrative_type=narrative_type,
            visual_cue=visual_cue,
        )

    def _is_dialogue(self, text: str) -> bool:
        """Detect if beat contains dialogue or is purely narration."""
        # Check for common dialogue markers
        dialogue_markers = [
            r'ఆయన.*చెప్',  # Telugu: he/she said
            r'నాకు.*గారు',  # To me garu
            r'".*"',  # Quoted speech
            r"'.*'",  # Single quoted
            r"said|speaks|tells|asks",  # English dialogue verbs
        ]

        for marker in dialogue_markers:
            if re.search(marker, text, re.IGNORECASE):
                return True

        return False

    def _extract_speaker(self, text: str) -> Optional[str]:
        """Extract speaker name/role from dialogue text."""
        # Simple regex to find common speaker patterns
        patterns = [
            r"అతను.*చెప్|అతను.*అన్న",
            r"ఆమె.*చెప్|ఆమె.*అన్న",
            r"నటుడు|నటిక|నేను|నీవు",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

        return None

    def _classify_narrative_type(self, text: str) -> str:
        """Classify beat as action, dialogue, emotional, or context."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["jump", "leap", "run", "walk", "act"]):
            return "action"
        elif any(word in text_lower for word in ["feel", "pain", "sad", "happy", "ఆ బాధ"]):
            return "emotional"
        elif any(word in text_lower for word in ["said", "speak", "talk", "చెప్", "అన్న"]):
            return "dialogue"
        else:
            return "context"

    def _extract_visual_cue(self, text: str) -> str:
        """Extract visual direction hints from beat text."""
        # Look for descriptive adjectives and actions
        visual_keywords = [
            "vineyard", "office", "screen", "stage", "scene",
            "సినిమా", "దృశ్యం", "రంగస్థలం", "పాత్ర"
        ]

        for keyword in visual_keywords:
            if keyword.lower() in text.lower():
                return f"Scene includes: {keyword}"

        return ""

    def _generate_scene_id(self, text: str, narrative_type: str) -> str:
        """Generate unique scene identifier from beat text."""
        # Extract first few significant words
        words = re.findall(r"\b[ఆ-హ్ఎ-ఔa-zA-Z]{3,}\b", text)[:3]
        scene_id = "_".join(w.lower() for w in words)
        return scene_id or f"scene_{narrative_type}"

    def _estimate_timing(
        self, index: int, beat_text: str, audio_segments: Optional[List[Any]] = None
    ) -> tuple[int, int]:
        """Estimate start and end timing for beat."""
        if audio_segments and index < len(audio_segments):
            seg = audio_segments[index]
            return seg.start_time_ms, seg.end_time_ms

        # Fallback: estimate based on text length
        # Assume average speaking rate: ~140 words/min = 2.33 words/sec
        word_count = len(beat_text.split())
        duration_ms = int(word_count / 2.33 * 1000)
        start_ms = index * 4000  # Rough estimate
        end_ms = start_ms + max(duration_ms, 1000)

        return start_ms, end_ms

    def save_beats_json(self, beats: List[ParsedBeat], output_path: str | Path) -> None:
        """Save parsed beats to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "beats": [asdict(b) for b in beats],
            "total_beats": len(beats),
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_beats_json(self, path: str | Path) -> List[ParsedBeat]:
        """Load parsed beats from JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return [ParsedBeat(**b) for b in data["beats"]]
