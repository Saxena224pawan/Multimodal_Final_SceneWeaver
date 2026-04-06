"""Voice script extractor - handles ASR and audio processing."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AudioSegment:
    """Single audio segment with extracted text and timing."""
    start_time_ms: int
    end_time_ms: int
    text: str
    speaker: Optional[str] = None
    confidence: float = 1.0


@dataclass
class AudioProcessorConfig:
    """Configuration for audio processing."""
    model_id: str = "openai/whisper-base"  # or "whisper-small", "whisper-medium"
    device: str = "cuda"
    language: str = "te"  # Telugu
    segment_min_duration_ms: int = 500
    segment_max_duration_ms: int = 5000


class AudioProcessor:
    """
    Extracts text and timestamps from voice files using Whisper or similar ASR.
    Supports multiple audio formats (.wav, .mp3, .m4a, .flac, etc.)
    """

    def __init__(self, config: AudioProcessorConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._torch = None

    def load(self) -> None:
        """Load Whisper model and processor."""
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers required for AudioProcessor. Install: pip install transformers"
            ) from exc

        self._torch = torch
        device = self.config.device if torch.cuda.is_available() else "cpu"
        
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(self.config.model_id)
        self._model.to(device)
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)

    def extract_from_file(
        self, audio_path: str | Path, return_segments: bool = True
    ) -> tuple[str, List[AudioSegment]] | str:
        """
        Extract text from audio file using Whisper.
        
        Args:
            audio_path: Path to audio file
            return_segments: If True, return per-segment timing; else return full text
            
        Returns:
            Tuple of (full_text, segments) or just full_text
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            import librosa
        except ImportError as exc:
            raise ImportError(
                "librosa required for audio loading. Install: pip install librosa"
            ) from exc

        # Load audio with librosa
        audio, sr = librosa.load(str(audio_path), sr=16000)

        # Prepare input for Whisper
        inputs = self._processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate transcription
        with self._torch.no_grad():
            generated_ids = self._model.generate(**inputs, language=self.config.language)

        # Decode to text
        full_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        if not return_segments:
            return full_text

        # Split into segments based on duration and sentence boundaries
        segments = self._split_into_segments(full_text, audio, sr)
        return full_text, segments

    def _split_into_segments(
        self, text: str, audio: Any, sr: int
    ) -> List[AudioSegment]:
        """Split text into segments with timestamps."""
        # Naive segmentation based on sentence boundaries
        import re

        sentences = re.split(r"(?<=[।.!?])\s+", text)
        total_duration_ms = len(audio) / sr * 1000

        segments = []
        current_time = 0.0
        time_per_char = total_duration_ms / max(len(text), 1)

        for sent in sentences:
            if not sent.strip():
                continue

            sent_duration = len(sent) * time_per_char
            start_ms = int(current_time)
            end_ms = int(current_time + sent_duration)

            # Respect min/max duration constraints
            if end_ms - start_ms >= self.config.segment_min_duration_ms:
                segments.append(
                    AudioSegment(
                        start_time_ms=start_ms,
                        end_time_ms=end_ms,
                        text=sent.strip(),
                    )
                )
            current_time += sent_duration

        return segments

    def save_segments_json(
        self, segments: List[AudioSegment], output_path: str | Path
    ) -> None:
        """Save extracted segments to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "segments": [asdict(seg) for seg in segments],
            "total_segments": len(segments),
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_segments_json(self, path: str | Path) -> List[AudioSegment]:
        """Load segments from JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return [AudioSegment(**seg) for seg in data["segments"]]
