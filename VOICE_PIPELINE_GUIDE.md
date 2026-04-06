# Voice Script to Visual Generation Pipeline

## Overview

SceneWeaver now supports **Voice-to-Visual** generation alongside the existing text-based pipeline. Convert voice scripts (or raw text narration) directly into synchronized video stories.

```
Voice Script (MP3/WAV/Text)
    ↓
[Speech Recognition - Optional ASR]
    ↓
Text Transcription
    ↓
[Beat Parser - Extract narrative segments]
    ↓
Structured Beats with Timing
    ↓
[Voice Scene Director - Generate visual prompts]
    ↓
Scene Plan (Compatible with existing pipeline)
    ↓
[Video Backbone - Generate frames]
    ↓
Final Video (Synced to audio)
```

---

## Quick Start

### 1. From Text Script (No ASR needed)

```bash
python scripts/02_story/03_voice_script_to_scene_plan.py \
  --voice_script path/to/script.txt \
  --total_duration_sec 180 \
  --output outputs/scene_plan.json
```

### 2. From Audio File (Requires Whisper)

```bash
pip install openai-whisper librosa

python scripts/02_story/03_voice_script_to_scene_plan.py \
  --voice_script path/to/audio.mp3 \
  --use_asr \
  --total_duration_sec 180 \
  --output outputs/scene_plan.json
```

### 3. Full Pipeline (Voice → Video)

```bash
# Generate scene plan
python scripts/02_story/03_voice_script_to_scene_plan.py \
  --voice_script minitalks_ntr.txt \
  --total_duration_sec 180 \
  --output outputs/ntr_scene_plan.json

# Generate video (using existing pipeline)
python scripts/run_story_pipeline.py \
  --window_plan_json outputs/ntr_scene_plan.json \
  --output_dir outputs/ntr_video \
  --video_model_id "wan"
```

---

## New Modules

### Audio Processor (`audio_processor/`)

#### `AudioProcessor` - ASR & Audio Extraction
```python
from audio_processor import AudioProcessor, AudioProcessorConfig

config = AudioProcessorConfig(
    model_id="openai/whisper-base",
    device="cuda",
    language="te"  # Telugu
)
processor = AudioProcessor(config)
processor.load()

# Extract from audio file
text, segments = processor.extract_from_file("script.mp3", return_segments=True)

# Save segments
processor.save_segments_json(segments, "segments.json")
```

**Supports:**
- `.wav`, `.mp3`, `.m4a`, `.flac` audio formats
- Telugu, Tamil, English (configurable)
- Segment-level timing and confidence scores
- GPU acceleration (CUDA) or CPU fallback

---

#### `VoiceScriptParser` - Text to Beats
```python
from audio_processor import VoiceScriptParser, VoiceScriptParserConfig

config = VoiceScriptParserConfig(
    min_beat_length=50,
    max_beat_length=300,
    language="te"
)
parser = VoiceScriptParser(config)

beats = parser.parse_transcript(raw_text)

# Export beats
parser.save_beats_json(beats, "beats.json")
```

**Features:**
- Detects dialogue vs. narration
- Extracts speaker information
- Classifies beat type: action, dialogue, emotional, context
- Estimates timing from text + optional audio segments
- Mixed-language support (Telugu + English)

---

### Voice Scene Director (`director_llm/voice_scene_director.py`)

Extends `SceneDirector` for audio-driven generation:

```python
from director_llm.voice_scene_director import VoiceSceneDirector, VoiceSceneConfig
from director_llm import SceneDirectorConfig

director_config = SceneDirectorConfig(window_count_mode="dynamic")
voice_config = VoiceSceneConfig(
    base_director_config=director_config,
    audio_duration_sec=180.0,
    dialogue_window_multiplier=1.2
)

director = VoiceSceneDirector(voice_config)
windows = director.plan_windows_from_voice_beats(beats, total_duration_sec=180.0)

# Export as JSON (compatible with existing pipeline)
director.export_scene_plan_json(windows, "scene_plan.json")
```

**Capabilities:**
- Maps audio timestamps to scene windows
- Adaptive window duration based on beat type
- Automatic environment & character continuity
- Story phase detection (intro → rising → climax → resolution)
- Scene change detection
- Visual prompt generation from narrative beats

---

## Scene Plan Output Format

Generated JSON format (compatible with existing pipeline):

```json
[
  {
    "beat": "Narrative text from voice script",
    "scene_id": "unique_scene_identifier",
    "environment_anchor": "Visual setting description for continuity",
    "character_lock": "Character consistency constraints",
    "scene_change": false,
    "story_phase": "introduction|rising_action|climax|resolution",
    "start_sec": 0,
    "end_sec": 14,
    "prompt_seed": "Full visual prompt derived from voice beat"
  },
  ...
]
```

Example from NTR script:
```json
{
  "beat": "తన భార్య Basavatarakam గారిని cancer వల్ల కోల్పోయారు...",
  "scene_id": "basavatarakam_cancer",
  "environment_anchor": "Consistent screen. soft, introspective lighting.",
  "character_lock": "Keep N.T. Rama Rao as central figure. Convey emotional depth.",
  "scene_change": false,
  "story_phase": "climax",
  "start_sec": 16,
  "end_sec": 30,
  "prompt_seed": "తన భార్య Basavatarakam గారిని cancer వల్ల... Show introspection, dignity, emotional resonance."
}
```

---

## Configuration Options

### `03_voice_script_to_scene_plan.py` CLI Arguments

```bash
--voice_script PATH          # Path to script file or audio file (required)
--total_duration_sec FLOAT   # Total duration in seconds (default: 180)
--output PATH               # Output scene plan JSON path (required)
--window_seconds INT        # Default window duration (default: 10)
--language LANG             # Language: te|ta|en (default: te)
--min_beat_length INT       # Minimum beat chars (default: 50)
--max_beat_length INT       # Maximum beat chars (default: 300)
--use_asr                   # Enable ASR for audio files
--asr_model ID              # Whisper model ID (default: openai/whisper-base)
--device DEVICE             # cuda or cpu (default: cuda)
```

### VoiceSceneConfig

```python
@dataclass
class VoiceSceneConfig:
    base_director_config: Optional[SceneDirectorConfig] = None
    audio_duration_sec: float = 60.0
    allow_silence_ms: int = 500
    dialogue_window_multiplier: float = 1.2  # 20% longer for dialogue
    emotional_pause_ms: int = 800            # Extra pause for emotional beats
    sync_precision_ms: int = 100              # Frame-sync precision
```

---

## Example: NTR Legacy Documentary

**Input Script** (`minitalks_ntr.txt`):
```
Hello Buddies, Welcome to Minitalks-Classics
ఈ రోజు మనం ఒక లెజెండ్ గురించి మాట్లాడుకుందాం. 
ఆయన పేరు చెప్పగానే తెలుగు ప్రజల గుండెల్లో ఒక గౌరవం...
[NTR's discipline and legacy...]
```

**Processing:**
```bash
python scripts/02_story/03_voice_script_to_scene_plan.py \
  --voice_script minitalks_ntr.txt \
  --total_duration_sec 180 \
  --language te \
  --output ntr_scene_plan.json
```

**Output:** 6 narrative beats → 6 scene windows with:
- Introduction phase (greeting + introduction)
- Rising action (NTR's journey)
- Climax (his roles and personal struggles)
- Resolution (his legacy)

**Generated Scenes:**
- Scene 0: Documentary intro with welcoming tone
- Scene 1-2: NTR's life story and discipline
- Scene 3: His iconic acting roles
- Scene 4: Emotional moment (wife's passing)
- Scene 5: Legacy and conclusion

---

## Integration with Existing Pipeline

The generated scene plan is **100% compatible** with:
- `scripts/run_story_pipeline.py`
- `video_backbone/wan_backbone.py`
- Existing memory embeddings & continuity logic
- Multi-agent refinement system

**No changes needed** to existing video generation code.

---

## Supported Languages

Currently optimized for:
- **Telugu (te)** - Full support with Whisper
- **Tamil (ta)** - Basic support
- **English (en)** - Full support

**To add more languages:** Update `VoiceScriptParserConfig.language` and Whisper model settings.

---

## Advanced Usage

### Custom Beat Classification

Override `_classify_narrative_type()` in `VoiceScriptParser`:

```python
class CustomParser(VoiceScriptParser):
    def _classify_narrative_type(self, text):
        if "mythology" in text.lower():
            return "mythological"
        return super()._classify_narrative_type(text)
```

### Dynamic Timing Adjustment

Modify `VoiceSceneConfig` for different pacing:

```python
# Fast-paced dialogue sequences
config = VoiceSceneConfig(
    dialogue_window_multiplier=1.5,
    emotional_pause_ms=500
)

# Slow, contemplative scenes
config = VoiceSceneConfig(
    dialogue_window_multiplier=0.9,
    emotional_pause_ms=1500
)
```

### Manual Beat Editing

Edit the generated beats JSON before scene plan generation:

```bash
# Generate beats
python -c "
from audio_processor import VoiceScriptParser
...
beats = parser.parse_transcript(text)
parser.save_beats_json(beats, 'beats.json')
"

# Edit beats.json manually

# Convert to scene plan
python -c "
from director_llm.voice_scene_director import VoiceSceneDirector
...
beats = parser.load_beats_json('beats.json')
windows = director.plan_windows_from_voice_beats(beats, 180)
"
```

---

## Dependencies

### Required
```bash
pip install transformers torch librosa
```

### For ASR (optional)
```bash
pip install openai-whisper
# or: pip install transformers[torch]
```

### For GPU acceleration
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Troubleshooting

### **Q: "Module not found" error**
Make sure you're running from project root:
```bash
cd /path/to/Multimodal_Final_SceneWeaver-1
python scripts/02_story/03_voice_script_to_scene_plan.py ...
```

### **Q: ASR is slow**
- Use `--asr_model openai/whisper-small` (faster, less accurate)
- Or `--device cpu` if CUDA memory is limited
- Pre-process audio to 16kHz mono for faster loading

### **Q: Beats are too fragmented**
- Increase `--max_beat_length` (default: 300)
- Decrease `--min_beat_length` (default: 50)

### **Q: Scene plan doesn't match audio timing**
- Provide actual audio file with `--use_asr` for accurate timestamps
- Text-based estimation assumes ~140 words/min speaking rate

---

## Next Steps

1. **Audio-Visual Sync Module** - Fine-tune frame timing to match audio
2. **Advanced Dialogue Handling** - Multi-speaker support with speaker diarization
3. **Custom Vision Models** - Train models on dubbed/subtitled content
4. **Cloud Integration** - Streaming ASR for large files

---

## Contributing

To extend the voice pipeline:

1. Add new beat types in `voice_script_parser.py`
2. Extend `VoiceSceneDirector` for specialized prompt generation
3. Submit tests in `tests/test_voice_pipeline.py`

---

## References

- **Original Repository:** SceneWeaver - Multimodal Story-to-Video
- **SceneDirector:** `director_llm/scene_director.py`
- **Video Backbone:** `video_backbone/wan_backbone.py`
- **Whisper:** https://github.com/openai/whisper

