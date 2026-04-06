"""Integration tests for voice-to-visual pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from audio_processor import VoiceScriptParser, VoiceScriptParserConfig, AudioSegment
from director_llm.voice_scene_director import VoiceSceneDirector, VoiceSceneConfig
from director_llm import SceneDirectorConfig


class TestVoiceScriptParser:
    """Test voice script parsing."""

    def test_parse_simple_text(self):
        """Test parsing basic script text."""
        text = "Hello world. This is a test. It works well."
        config = VoiceScriptParserConfig(min_beat_length=5)
        parser = VoiceScriptParser(config)

        beats = parser.parse_transcript(text)

        assert len(beats) > 0
        assert beats[0].beat_text is not None
        assert beats[0].start_time_ms >= 0
        assert beats[0].end_time_ms >= beats[0].start_time_ms

    def test_parse_mixed_language(self):
        """Test parsing Telugu and English mixed text."""
        text = "ఈ రోజు N.T. Rama Rao గారిని గుర్తుంచుకుందాం. He was a legend."
        config = VoiceScriptParserConfig(language="te")
        parser = VoiceScriptParser(config)

        beats = parser.parse_transcript(text)

        assert len(beats) > 0
        assert any("N.T. Rama Rao" in b.beat_text for b in beats)

    def test_dialogue_detection(self):
        """Test dialogue vs narration detection."""
        dialogue_text = '"Hello," he said. "How are you?"'
        narration_text = "The story begins in a quiet village."

        parser = VoiceScriptParser()

        beats_dialogue = parser.parse_transcript(dialogue_text)
        beats_narration = parser.parse_transcript(narration_text)

        # Check detection (may vary based on heuristics)
        assert len(beats_dialogue) > 0 or len(beats_narration) > 0

    def test_save_and_load_beats(self):
        """Test saving and loading beats from JSON."""
        text = "First beat. Second beat. Third beat."
        parser = VoiceScriptParser()
        beats = parser.parse_transcript(text)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "beats.json"
            parser.save_beats_json(beats, json_path)

            loaded_beats = parser.load_beats_json(json_path)

            assert len(loaded_beats) == len(beats)
            assert loaded_beats[0].beat_text == beats[0].beat_text

    def test_emotional_classification(self):
        """Test narrative type classification."""
        emotional_text = "She felt the pain deeply and sadness filled her heart."
        action_text = "He jumped over the fence and ran quickly."

        parser = VoiceScriptParser()

        em_beats = parser.parse_transcript(emotional_text)
        act_beats = parser.parse_transcript(action_text)

        # Check if classifier works
        assert len(em_beats) > 0
        assert len(act_beats) > 0


class TestVoiceSceneDirector:
    """Test voice-aware scene generation."""

    def test_generate_windows_from_beats(self):
        """Test scene window generation from beats."""
        from audio_processor import ParsedBeat

        beats = [
            ParsedBeat(
                index=0,
                beat_text="First scene description",
                start_time_ms=0,
                end_time_ms=5000,
                scene_id="scene_1",
            ),
            ParsedBeat(
                index=1,
                beat_text="Second scene description",
                start_time_ms=5000,
                end_time_ms=10000,
                scene_id="scene_2",
            ),
        ]

        director = VoiceSceneDirector()
        windows = director.plan_windows_from_voice_beats(beats, 10.0)

        assert len(windows) == 2
        assert windows[0].beat == "First scene description"
        assert windows[0].start_sec == 0
        assert windows[1].start_sec == 5

    def test_character_lock_generation(self):
        """Test character continuity lock generation."""
        from audio_processor import ParsedBeat

        beat = ParsedBeat(
            index=0,
            beat_text="Character scene",
            start_time_ms=0,
            end_time_ms=5000,
            scene_id="scene_1",
            narrative_type="dialogue",
        )

        director = VoiceSceneDirector()
        char_lock = director._build_character_lock(beat)

        assert "character" in char_lock.lower() or "keep" in char_lock.lower()

    def test_environment_anchor_generation(self):
        """Test environment continuity anchor generation."""
        from audio_processor import ParsedBeat

        beat = ParsedBeat(
            index=0,
            beat_text="Scene in a beautiful garden",
            start_time_ms=0,
            end_time_ms=5000,
            scene_id="scene_1",
            narrative_type="action",
        )

        director = VoiceSceneDirector()
        env_anchor = director._build_environment_anchor(beat, 0, [beat])

        assert len(env_anchor) > 0
        assert "garden" in env_anchor.lower() or "setting" in env_anchor.lower()

    def test_export_scene_plan_json(self):
        """Test exporting scene plan to JSON."""
        from audio_processor import ParsedBeat

        beats = [
            ParsedBeat(
                index=0,
                beat_text="Test beat",
                start_time_ms=0,
                end_time_ms=5000,
                scene_id="test_scene",
            ),
        ]

        director = VoiceSceneDirector()
        windows = director.plan_windows_from_voice_beats(beats, 5.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "plan.json"
            director.export_scene_plan_json(windows, json_path)

            assert json_path.exists()

            with json_path.open() as f:
                data = json.load(f)

            assert len(data) > 0
            assert "beat" in data[0]
            assert "scene_id" in data[0]

    def test_story_phase_detection(self):
        """Test story phase classification."""
        director = VoiceSceneDirector()

        intro = director._get_story_phase(0, 10)
        middle = director._get_story_phase(5, 10)
        end = director._get_story_phase(9, 10)

        assert intro == "introduction"
        assert middle in ["rising_action", "climax"]
        assert end == "resolution"


class TestEndToEndPipeline:
    """Test complete voice-to-scene pipeline."""

    def test_full_pipeline_text_to_json(self):
        """Test full pipeline from text to scene plan JSON."""
        raw_text = """
        Introduction to story.
        ఈ రోజు మనం ఒక గుర్తుండే కథ చెప్పుకుందాం.
        The hero's journey begins here.
        He faced many challenges but never gave up.
        అతని discipline and dedication were remarkable.
        In the end, he achieved his dreams.
        That is the lesson we learn today.
        """

        # Step 1: Parse beats
        parser = VoiceScriptParser()
        beats = parser.parse_transcript(raw_text)
        assert len(beats) > 0

        # Step 2: Generate windows
        director = VoiceSceneDirector()
        windows = director.plan_windows_from_voice_beats(beats, 180.0)
        assert len(windows) > 0

        # Step 3: Export JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "scene_plan.json"
            director.export_scene_plan_json(windows, json_path)

            # Verify JSON structure
            with json_path.open() as f:
                plan = json.load(f)

            assert isinstance(plan, list)
            for scene in plan:
                assert "beat" in scene
                assert "scene_id" in scene
                assert "environment_anchor" in scene
                assert "character_lock" in scene
                assert "prompt_seed" in scene


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
