from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .scene_director import PromptBundle, SceneDirector, SceneDirectorConfig, SceneWindow, ShotPlan


@dataclass
class CharacterState:
    face_id: str
    hair: str
    outfit: str
    emotion: str


@dataclass
class LocationState:
    place: str
    time: str
    weather: str
    lighting: str


@dataclass
class CameraState:
    lens: str
    height: str
    movement: str


@dataclass
class ContinuityState:
    previous_action: str
    must_preserve: List[str] = field(default_factory=list)


@dataclass
class WindowState:
    characters: Dict[str, CharacterState]
    location: LocationState
    camera: CameraState
    continuity: ContinuityState

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StatefulPromptBundle(PromptBundle):
    window_state: WindowState


class StatefulSceneDirector(SceneDirector):
    def refine_prompt(
        self,
        storyline: str,
        window: SceneWindow,
        previous_prompt: str,
        memory_feedback: Optional[Dict[str, Any]],
        previous_window_state: Optional[WindowState] = None,
    ) -> StatefulPromptBundle:
        if self._model is None or self._tokenizer is None or self._torch is None:
            return self._heuristic_refine_prompt(
                storyline=storyline,
                window=window,
                previous_prompt=previous_prompt,
                memory_feedback=memory_feedback,
                previous_window_state=previous_window_state,
            )

        compact_prev = self._compact_previous_prompt(previous_prompt)
        compact_storyline = self._compact_storyline(storyline)
        memory_text = self._memory_feedback_text(memory_feedback)
        context = {
            "storyline": compact_storyline,
            "window_index": window.index,
            "window_time": f"{window.start_sec}-{window.end_sec}s",
            "beat": window.beat,
            "previous_prompt": compact_prev,
            "previous_window_state": previous_window_state.to_dict() if previous_window_state else None,
            "memory_feedback": memory_text,
        }
        prompt = (
            "You are a strict scene director for text-to-video generation.\n"
            "Task: produce ONE continuity state JSON object for the CURRENT window only.\n"
            "Rules:\n"
            "1) Keep character identity stable across windows using the same names and face_id values.\n"
            "2) Keep location, wardrobe, props, and camera continuity unless the current beat requires a change.\n"
            "3) Put required carry-over constraints into continuity.must_preserve as short visual items.\n"
            "4) Keep camera practical and concrete.\n"
            "5) Return JSON only. No markdown.\n"
            "Required JSON shape:\n"
            "{\"characters\":{\"Name\":{\"face_id\":\"ref_char_01\",\"hair\":\"...\",\"outfit\":\"...\",\"emotion\":\"...\"}},"
            "\"location\":{\"place\":\"...\",\"time\":\"...\",\"weather\":\"...\",\"lighting\":\"...\"},"
            "\"camera\":{\"lens\":\"...\",\"height\":\"...\",\"movement\":\"...\"},"
            "\"continuity\":{\"previous_action\":\"...\",\"must_preserve\":[\"...\",\"...\"]}}\n"
            f"Context:\n{json.dumps(context, ensure_ascii=False)}\n"
        )

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature

        tokenized = self._tokenizer(prompt, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        tokenized = {k: v.to(model_device) for k, v in tokenized.items()}
        with self._torch.no_grad():
            output_ids = self._model.generate(**tokenized, **gen_kwargs)
        new_token_ids = output_ids[0][tokenized["input_ids"].shape[-1] :]
        out = self._tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        parsed = self._extract_json_object(out)
        if parsed:
            window_state = self._to_window_state(parsed, window, previous_window_state)
            shot_plan = self._window_state_to_shot_plan(window_state=window_state, window=window)
            prompt_text = self._prompt_from_window_state(
                window_state=window_state,
                window=window,
                continuity_note=self._continuity_note(memory_feedback),
                previous_context=self._previous_context(previous_prompt),
            )
            cleaned = self._normalize_prompt(prompt_text)
            return StatefulPromptBundle(prompt_text=cleaned, shot_plan=shot_plan, window_state=window_state)

        return self._heuristic_refine_prompt(
            storyline=storyline,
            window=window,
            previous_prompt=previous_prompt,
            memory_feedback=memory_feedback,
            previous_window_state=previous_window_state,
        )

    def _heuristic_refine_prompt(
        self,
        storyline: str,
        window: SceneWindow,
        previous_prompt: str,
        memory_feedback: Optional[Dict[str, Any]],
        previous_window_state: Optional[WindowState],
    ) -> StatefulPromptBundle:
        window_state = self._heuristic_window_state(
            storyline=storyline,
            window=window,
            previous_window_state=previous_window_state,
        )
        shot_plan = self._window_state_to_shot_plan(window_state=window_state, window=window)
        prompt_text = self._prompt_from_window_state(
            window_state=window_state,
            window=window,
            continuity_note=self._continuity_note(memory_feedback),
            previous_context=self._previous_context(previous_prompt),
        )
        cleaned = self._normalize_prompt(prompt_text)
        return StatefulPromptBundle(prompt_text=cleaned, shot_plan=shot_plan, window_state=window_state)

    def _heuristic_window_state(
        self,
        storyline: str,
        window: SceneWindow,
        previous_window_state: Optional[WindowState],
    ) -> WindowState:
        if previous_window_state is not None:
            state = copy.deepcopy(previous_window_state)
        else:
            protagonist = self._infer_character_names(storyline=storyline, beat=window.beat)[0]
            state = WindowState(
                characters={
                    protagonist: CharacterState(
                        face_id="ref_char_01",
                        hair="consistent previous hairstyle",
                        outfit="consistent previous outfit",
                        emotion="focused",
                    )
                },
                location=LocationState(
                    place=self._infer_place(window.beat),
                    time=self._infer_time(window.beat),
                    weather=self._infer_weather(window.beat),
                    lighting=self._infer_lighting(window.beat),
                ),
                camera=CameraState(
                    lens=self._default_lens(window.index),
                    height=self._default_camera_height(window.index),
                    movement=self._default_camera_movement(window.index),
                ),
                continuity=ContinuityState(previous_action="", must_preserve=[]),
            )

        inferred_names = self._infer_character_names(storyline=storyline, beat=window.beat)
        for idx, name in enumerate(inferred_names, start=1):
            if name not in state.characters:
                state.characters[name] = CharacterState(
                    face_id=f"ref_char_{idx:02d}",
                    hair="consistent previous hairstyle",
                    outfit="consistent previous outfit",
                    emotion=self._infer_emotion(window.beat),
                )

        for character in state.characters.values():
            character.emotion = self._infer_emotion(window.beat)

        if self._beat_requests_scene_change(window.beat):
            state.location.place = self._infer_place(window.beat)
        state.location.time = self._infer_time(window.beat, fallback=state.location.time)
        state.location.weather = self._infer_weather(window.beat, fallback=state.location.weather)
        state.location.lighting = self._infer_lighting(window.beat, fallback=state.location.lighting)
        if previous_window_state is None:
            state.camera = CameraState(
                lens=self._default_lens(window.index),
                height=self._default_camera_height(window.index),
                movement=self._default_camera_movement(window.index),
            )
        state.continuity.previous_action = self._previous_action_text(previous_window_state, window)
        state.continuity.must_preserve = self._build_preserve_list(state=state)
        return state

    def _to_window_state(
        self,
        parsed: Dict[str, Any],
        window: SceneWindow,
        previous_window_state: Optional[WindowState],
    ) -> WindowState:
        prior = copy.deepcopy(previous_window_state) if previous_window_state is not None else None
        default_names = self._infer_character_names(storyline="", beat=window.beat)
        source_characters = parsed.get("characters")
        characters: Dict[str, CharacterState] = {}
        if isinstance(source_characters, dict):
            for idx, (name, payload) in enumerate(source_characters.items(), start=1):
                payload = payload if isinstance(payload, dict) else {}
                prior_character = prior.characters.get(name) if prior is not None else None
                characters[name] = CharacterState(
                    face_id=self._string_field(
                        payload,
                        "face_id",
                        prior_character.face_id if prior_character else f"ref_char_{idx:02d}",
                    ),
                    hair=self._string_field(
                        payload,
                        "hair",
                        prior_character.hair if prior_character else "consistent previous hairstyle",
                    ),
                    outfit=self._string_field(
                        payload,
                        "outfit",
                        prior_character.outfit if prior_character else "consistent previous outfit",
                    ),
                    emotion=self._string_field(
                        payload,
                        "emotion",
                        prior_character.emotion if prior_character else self._infer_emotion(window.beat),
                    ),
                )
        if not characters:
            fallback_name = default_names[0]
            if prior is not None and prior.characters:
                characters = copy.deepcopy(prior.characters)
            else:
                characters[fallback_name] = CharacterState(
                    face_id="ref_char_01",
                    hair="consistent previous hairstyle",
                    outfit="consistent previous outfit",
                    emotion=self._infer_emotion(window.beat),
                )

        location_payload = parsed.get("location") if isinstance(parsed.get("location"), dict) else {}
        prior_location = prior.location if prior is not None else None
        location = LocationState(
            place=self._string_field(location_payload, "place", prior_location.place if prior_location else self._infer_place(window.beat)),
            time=self._string_field(location_payload, "time", prior_location.time if prior_location else self._infer_time(window.beat)),
            weather=self._string_field(location_payload, "weather", prior_location.weather if prior_location else self._infer_weather(window.beat)),
            lighting=self._string_field(location_payload, "lighting", prior_location.lighting if prior_location else self._infer_lighting(window.beat)),
        )

        camera_payload = parsed.get("camera") if isinstance(parsed.get("camera"), dict) else {}
        prior_camera = prior.camera if prior is not None else None
        camera = CameraState(
            lens=self._string_field(camera_payload, "lens", prior_camera.lens if prior_camera else self._default_lens(window.index)),
            height=self._string_field(camera_payload, "height", prior_camera.height if prior_camera else self._default_camera_height(window.index)),
            movement=self._string_field(camera_payload, "movement", prior_camera.movement if prior_camera else self._default_camera_movement(window.index)),
        )

        continuity_payload = parsed.get("continuity") if isinstance(parsed.get("continuity"), dict) else {}
        preserve_raw = continuity_payload.get("must_preserve")
        must_preserve: List[str] = []
        if isinstance(preserve_raw, list):
            for item in preserve_raw:
                if isinstance(item, str):
                    cleaned = " ".join(item.split()).strip()
                    if cleaned:
                        must_preserve.append(cleaned)
        provisional_state = WindowState(
            characters=characters,
            location=location,
            camera=camera,
            continuity=ContinuityState(previous_action="", must_preserve=[]),
        )
        if not must_preserve:
            must_preserve = self._build_preserve_list(state=provisional_state)
        continuity = ContinuityState(
            previous_action=self._string_field(
                continuity_payload,
                "previous_action",
                self._previous_action_text(previous_window_state, window),
            ),
            must_preserve=must_preserve[:8],
        )
        return WindowState(characters=characters, location=location, camera=camera, continuity=continuity)

    def _window_state_to_shot_plan(self, window_state: WindowState, window: SceneWindow) -> ShotPlan:
        names = list(window_state.characters.keys())
        subject_blocking = (
            f"keep {', '.join(names)} framed consistently" if names else "keep main subjects consistent and centered"
        )
        continuity_anchor = "; ".join(window_state.continuity.must_preserve[:4]) or window_state.location.place
        return ShotPlan(
            shot_type=self._lens_to_shot_type(window_state.camera.lens, window.index),
            camera_angle=window_state.camera.height,
            camera_motion=window_state.camera.movement,
            subject_blocking=subject_blocking,
            continuity_anchor=continuity_anchor,
            action=window.beat,
        )

    def _prompt_from_window_state(
        self,
        window_state: WindowState,
        window: SceneWindow,
        continuity_note: str,
        previous_context: str,
    ) -> str:
        character_parts = []
        for name, char in window_state.characters.items():
            character_parts.append(
                f"{name} ({char.face_id}): hair {char.hair}, outfit {char.outfit}, emotion {char.emotion}."
            )
        preserve_text = ", ".join(window_state.continuity.must_preserve) or "preserve established visual continuity"
        base = (
            f"Characters: {' '.join(character_parts)} "
            f"Location: {window_state.location.place}, time {window_state.location.time}, weather {window_state.location.weather}, lighting {window_state.location.lighting}. "
            f"Camera: lens {window_state.camera.lens}, height {window_state.camera.height}, movement {window_state.camera.movement}. "
            f"Current beat: {window.beat}. "
            f"Previous action: {window_state.continuity.previous_action}. "
            f"Must preserve: {preserve_text}. "
            f"Time window {window.start_sec}-{window.end_sec}s, {continuity_note}.{previous_context}"
        )
        return self._normalize_prompt(base)

    @staticmethod
    def _string_field(payload: Dict[str, Any], key: str, fallback: str) -> str:
        value = payload.get(key)
        if isinstance(value, str):
            cleaned = " ".join(value.split()).strip()
            if cleaned:
                return cleaned
        return fallback

    @staticmethod
    def _infer_character_names(storyline: str, beat: str) -> List[str]:
        tokens = re.findall(r"\b[A-Z][a-zA-Z0-9_'-]+\b", f"{storyline} {beat}")
        filtered: List[str] = []
        for token in tokens:
            if token.lower() in {"the", "then", "after", "when", "night", "day"}:
                continue
            if token not in filtered:
                filtered.append(token)
        return filtered[:3] or ["Protagonist"]

    @staticmethod
    def _infer_emotion(beat: str) -> str:
        text = (beat or "").lower()
        mapping = [
            ("fight", "determined"),
            ("chase", "urgent"),
            ("run", "focused"),
            ("cry", "distressed"),
            ("smile", "hopeful"),
            ("laugh", "joyful"),
            ("hide", "tense but controlled"),
            ("search", "alert"),
            ("discover", "surprised"),
            ("rest", "calm"),
        ]
        for token, emotion in mapping:
            if token in text:
                return emotion
        return "focused"

    @staticmethod
    def _infer_place(beat: str) -> str:
        text = (beat or "").strip()
        if not text:
            return "same established location"
        lower = text.lower()
        for marker in (" in ", " at ", " inside ", " outside ", " near "):
            pos = lower.find(marker)
            if pos != -1:
                place = text[pos + len(marker) :].strip(" .,;")
                if place:
                    return place[:80]
        return "same established location"

    @staticmethod
    def _infer_time(beat: str, fallback: str = "unspecified") -> str:
        text = (beat or "").lower()
        if "night" in text:
            return "night"
        if "sunset" in text or "dusk" in text:
            return "sunset"
        if "dawn" in text or "sunrise" in text:
            return "dawn"
        if "day" in text or "morning" in text or "noon" in text:
            return "day"
        return fallback

    @staticmethod
    def _infer_weather(beat: str, fallback: str = "clear") -> str:
        text = (beat or "").lower()
        if "rain" in text:
            return "rain"
        if "snow" in text:
            return "snow"
        if "fog" in text or "mist" in text:
            return "fog"
        if "storm" in text:
            return "storm"
        return fallback

    @staticmethod
    def _infer_lighting(beat: str, fallback: str = "natural motivated light") -> str:
        text = (beat or "").lower()
        if "neon" in text:
            return "neon side light"
        if "candle" in text:
            return "warm candle light"
        if "sunset" in text:
            return "warm low-angle sunset light"
        if "night" in text:
            return "low-key night lighting"
        return fallback

    def _default_lens(self, index: int) -> str:
        lenses = ["24mm", "35mm", "50mm", "85mm"]
        return lenses[index % len(lenses)]

    def _default_camera_height(self, index: int) -> str:
        heights = ["eye level", "eye level", "slightly low", "shoulder height"]
        return heights[index % len(heights)]

    def _default_camera_movement(self, index: int) -> str:
        motions = ["locked-off", "slow dolly in", "gentle tracking", "locked-off"]
        return motions[index % len(motions)]

    @staticmethod
    def _lens_to_shot_type(lens: str, index: int) -> str:
        value = (lens or "").lower()
        if "24" in value or "28" in value:
            return "wide establishing"
        if "35" in value or "50" in value:
            return "medium shot"
        if "85" in value:
            return "close-up"
        presets = ["wide establishing", "medium shot", "close-up", "profile detail"]
        return presets[index % len(presets)]

    @staticmethod
    def _previous_action_text(previous_window_state: Optional[WindowState], window: SceneWindow) -> str:
        if previous_window_state is not None:
            prev_items = previous_window_state.continuity.must_preserve[:2]
            if prev_items:
                return f"continue after preserving {', '.join(prev_items)}"
        if window.index == 0:
            return "story opening"
        return f"continue from window {window.index - 1}"

    @staticmethod
    def _build_preserve_list(state: WindowState) -> List[str]:
        preserve: List[str] = []
        for name, char in state.characters.items():
            preserve.append(f"{name} face_id {char.face_id}")
            preserve.append(char.outfit)
        preserve.append(state.location.place)
        preserve.append(state.location.lighting)
        preserve.append(state.camera.lens)
        preserve.append(state.camera.height)
        preserve.append(state.camera.movement)
        deduped: List[str] = []
        for item in preserve:
            cleaned = " ".join((item or "").split()).strip()
            if cleaned and cleaned not in deduped:
                deduped.append(cleaned)
        return deduped[:8]

    @staticmethod
    def _beat_requests_scene_change(beat: str) -> bool:
        text = (beat or "").lower()
        hints = (
            "new location",
            "cut to",
            "arrive",
            "arrives",
            "enter",
            "enters",
            "exit",
            "leave",
            "leaves",
            "move to",
            "moves to",
            "travel",
            "travels",
            "inside",
            "outside",
            "indoors",
            "outdoors",
            "back at",
        )
        return any(token in text for token in hints)
