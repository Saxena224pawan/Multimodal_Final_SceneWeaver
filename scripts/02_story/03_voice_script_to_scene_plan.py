#!/usr/bin/env python3
"""
Voice Script to Scene Plan Converter

Converts audio voice script (or raw text) into visual scene plan JSON.
Pipeline: Audio → ASR (optional) → Parse Beats → Generate Windows → Scene Plan

Example:
  python scripts/02_story/03_voice_script_to_scene_plan.py \\
    --voice_script path/to/script.txt \\
    --total_duration_sec 180 \\
    --output outputs/ntr_scene_plan.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_processor import VoiceScriptParser, VoiceScriptParserConfig
from director_llm.voice_scene_director import VoiceSceneDirector, VoiceSceneConfig
from director_llm import SceneDirectorConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert voice script to visual scene plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From raw text script
  python scripts/02_story/03_voice_script_to_scene_plan.py \\
    --voice_script path/to/script.txt \\
    --total_duration_sec 180 \\
    --output outputs/scene_plan.json

  # Custom window duration and narrative type
  python scripts/02_story/03_voice_script_to_scene_plan.py \\
    --voice_script path/to/script.txt \\
    --total_duration_sec 240 \\
    --output outputs/scene_plan.json \\
    --window_seconds 8 \\
    --language te
        """,
    )

    # Required arguments
    parser.add_argument(
        "--voice_script",
        type=str,
        required=True,
        help="Path to voice script file (plain text) or audio file (will use ASR)",
    )
    parser.add_argument(
        "--total_duration_sec",
        type=float,
        default=180.0,
        help="Total duration of voice/story in seconds (default: 180)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for scene plan JSON",
    )

    # Optional arguments
    parser.add_argument(
        "--window_seconds",
        type=int,
        default=10,
        help="Default window duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="te",
        choices=["te", "ta", "en"],
        help="Language of voice script (default: te for Telugu)",
    )
    parser.add_argument(
        "--min_beat_length",
        type=int,
        default=50,
        help="Minimum beat length in characters (default: 50)",
    )
    parser.add_argument(
        "--max_beat_length",
        type=int,
        default=300,
        help="Maximum beat length in characters (default: 300)",
    )
    parser.add_argument(
        "--use_asr",
        action="store_true",
        help="Use ASR to extract text from audio file (requires Whisper)",
    )
    parser.add_argument(
        "--asr_model",
        type=str,
        default="openai/whisper-base",
        help="Whisper model ID for ASR (default: openai/whisper-base)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for ASR model (default: cuda)",
    )

    return parser.parse_args()


def load_voice_script(path: str) -> str:
    """Load voice script from file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Script file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_audio(
    audio_path: str,
    asr_model: str = "openai/whisper-base",
    device: str = "cuda",
    language: str = "te",
) -> str:
    """
    Extract text from audio using Whisper ASR.
    Requires: pip install openai-whisper librosa
    """
    try:
        from audio_processor import AudioProcessor, AudioProcessorConfig
    except ImportError:
        raise ImportError(
            "AudioProcessor requires transformers and librosa. "
            "Install: pip install transformers librosa torch"
        )

    print(f"Loading ASR model: {asr_model}")
    config = AudioProcessorConfig(
        model_id=asr_model,
        device=device,
        language=language,
    )
    processor = AudioProcessor(config)
    processor.load()

    print(f"Extracting text from audio: {audio_path}")
    text, _ = processor.extract_from_file(audio_path, return_segments=True)
    return text


def parse_voice_beats(
    raw_text: str,
    language: str = "te",
    min_beat_length: int = 50,
    max_beat_length: int = 300,
) -> List[Any]:
    """Parse raw voice script into structured beats."""
    print("Parsing voice script into beats...")

    config = VoiceScriptParserConfig(
        min_beat_length=min_beat_length,
        max_beat_length=max_beat_length,
        language=language,
    )
    parser = VoiceScriptParser(config)
    beats = parser.parse_transcript(raw_text)

    print(f"Extracted {len(beats)} beats from script")
    for i, beat in enumerate(beats):
        print(f"  Beat {i}: {beat.beat_text[:60]}...")

    return beats


def generate_scene_plan(
    beats: List[Any],
    total_duration_sec: float,
    window_seconds: int = 10,
) -> List[Dict[str, Any]]:
    """Generate scene plan from voice beats."""
    print("Generating scene plan from voice beats...")

    director_config = SceneDirectorConfig(
        window_count_mode="dynamic",
        target_words_per_window=28,
    )
    voice_config = VoiceSceneConfig(
        base_director_config=director_config,
        audio_duration_sec=total_duration_sec,
    )
    director = VoiceSceneDirector(voice_config)

    windows = director.plan_windows_from_voice_beats(beats, total_duration_sec)

    print(f"Generated {len(windows)} scene windows")
    for w in windows:
        print(f"  Window {w.index}: {w.beat[:50]}... ({w.start_sec}s-{w.end_sec}s)")

    return windows


def export_plan(
    windows: List[Any],
    output_path: str,
) -> None:
    """Export scene plan to JSON."""
    from director_llm.voice_scene_director import VoiceSceneDirector

    output_path = Path(output_path)
    print(f"Exporting scene plan to: {output_path}")

    director = VoiceSceneDirector()
    director.export_scene_plan_json(windows, output_path)
    print("✓ Scene plan exported successfully")


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Step 1: Load or extract voice script text
        script_path = Path(args.voice_script)
        if args.use_asr or script_path.suffix in [".mp3", ".wav", ".m4a", ".flac"]:
            print(f"[1/4] Extracting text from audio using ASR...")
            raw_text = extract_text_from_audio(
                args.voice_script,
                asr_model=args.asr_model,
                device=args.device,
                language=args.language,
            )
        else:
            print(f"[1/4] Loading voice script from: {args.voice_script}")
            raw_text = load_voice_script(args.voice_script)

        print(f"Script text length: {len(raw_text)} characters\n")

        # Step 2: Parse beats from script
        print(f"[2/4] Parsing voice beats...")
        beats = parse_voice_beats(
            raw_text,
            language=args.language,
            min_beat_length=args.min_beat_length,
            max_beat_length=args.max_beat_length,
        )
        print()

        # Step 3: Generate scene windows
        print(f"[3/4] Generating scene plan from beats...")
        windows = generate_scene_plan(
            beats,
            total_duration_sec=args.total_duration_sec,
            window_seconds=args.window_seconds,
        )
        print()

        # Step 4: Export to JSON
        print(f"[4/4] Exporting scene plan...")
        export_plan(windows, args.output)

        print("\n✓ Pipeline complete!")
        print(f"  Input: {args.voice_script}")
        print(f"  Duration: {args.total_duration_sec}s")
        print(f"  Beats: {len(beats)}")
        print(f"  Windows: {len(windows)}")
        print(f"  Output: {args.output}")

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
