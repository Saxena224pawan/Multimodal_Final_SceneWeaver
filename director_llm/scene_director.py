from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SceneWindow:
    index: int
    start_sec: int
    end_sec: int
    beat: str
    prompt_seed: str
    scene_id: str = ""
    environment_anchor: str = ""
    character_lock: str = ""
    scene_change: Optional[bool] = None


@dataclass
class ShotPlan:
    shot_type: str
    camera_angle: str
    camera_motion: str
    subject_blocking: str
    continuity_anchor: str
    action: str


@dataclass
class PromptBundle:
    prompt_text: str
    shot_plan: ShotPlan
    scene_conversation: str = ""


@dataclass
class SceneDirectorConfig:
    model_id: Optional[str] = None
    temperature: float = 0.7
    max_new_tokens: int = 512
    do_sample: bool = False
    shot_plan_defaults: str = "cinematic"  # cinematic | docu | action


class SceneDirector:
    """
    Converts a storyline into scene windows and refines prompts per window.
    """

    def __init__(self, config: SceneDirectorConfig, window_seconds: int = 10):
        self.config = config
        self.window_seconds = window_seconds
        self._tokenizer = None
        self._model = None
        self._torch = None

    def load(self) -> None:
        if not self.config.model_id:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required to run SceneDirector with --director_model_id."
            ) from exc
        self._torch = torch
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
            )
        except ValueError as exc:
            if "Qwen2Tokenizer" in str(exc):
                raise ValueError(
                    "Qwen2 tokenizer is unavailable in this environment. "
                    "Upgrade transformers/tokenizers (e.g. transformers>=4.41, tokenizers>=0.19) "
                    "or use a compatible director model."
                ) from exc
            raise
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            self._model.to("cuda")
        self._model.eval()

    def plan_windows(
        self,
        storyline: str,
        total_minutes: float,
        beats_override: Optional[List[Any]] = None,
    ) -> List[SceneWindow]:
        window_count = max(1, math.ceil((total_minutes * 60) / self.window_seconds))

        window_specs: List[Dict[str, Any]]
        if beats_override:
            window_specs = self._normalize_beats_override(beats_override)
            if not window_specs:
                raise ValueError("window_plan_json produced no usable beats")
            window_count = len(window_specs)
        else:
            beats = self._extract_story_beats(storyline)
            if not beats:
                beats = [storyline.strip() or "Continue the story naturally."]
            beat_plan = self._expand_beats_for_windows(beats, window_count)
            window_specs = [{"beat": beat} for beat in beat_plan]

        windows: List[SceneWindow] = []
        for i in range(window_count):
            spec = window_specs[i]
            beat = str(spec.get("beat", "")).strip()
            start_sec = i * self.window_seconds
            end_sec = start_sec + self.window_seconds
            prompt_seed = self._default_visual_seed(
                beat=beat,
                is_opening=(i == 0),
                environment_anchor=str(spec.get("environment_anchor", "")).strip(),
            )
            windows.append(
                SceneWindow(
                    index=i,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    beat=beat,
                    prompt_seed=prompt_seed,
                    scene_id=str(spec.get("scene_id", "")).strip(),
                    environment_anchor=str(spec.get("environment_anchor", "")).strip(),
                    character_lock=str(spec.get("character_lock", "")).strip(),
                    scene_change=spec.get("scene_change")
                    if isinstance(spec.get("scene_change"), bool)
                    else None,
                )
            )
        return windows

    def refine_prompt(
        self,
        storyline: str,
        window: SceneWindow,
        previous_prompt: str,
        previous_scene_conversation: str = "",
        memory_feedback: Optional[Dict[str, Any]] = None,
    ) -> PromptBundle:
        if self._model is None or self._tokenizer is None or self._torch is None:
            return self._heuristic_refine_prompt(
                storyline,
                window,
                previous_prompt,
                previous_scene_conversation,
                memory_feedback,
            )

        compact_prev = self._compact_previous_prompt(previous_prompt)
        compact_storyline = self._compact_storyline(storyline)
        memory_text = self._memory_feedback_text(memory_feedback)
        previous_conversation = self._compact_scene_conversation(previous_scene_conversation)
        context = {
            "storyline": compact_storyline,
            "window_index": window.index,
            "window_time": f"{window.start_sec}-{window.end_sec}s",
            "beat": window.beat,
            "scene_id": window.scene_id,
            "environment_anchor": window.environment_anchor,
            "character_lock": self._compact_storyline(window.character_lock),
            "scene_change": window.scene_change,
            "previous_prompt": compact_prev,
            "previous_scene_conversation": previous_conversation,
            "memory_feedback": memory_text,
        }
        prompt = (
            "You are a strict scene director for text-to-video generation.\n"
            "Task: produce ONE concise shot prompt for the CURRENT window only.\n"
            "Rules:\n"
            "1) Keep subject identity and scene continuity from previous prompt.\n"
            "2) Advance only the current beat.\n"
            "3) Avoid adding new random characters or objects.\n"
            "4) Keep camera and motion explicit and realistic.\n"
            "5) Use concrete visual language, not abstract prose.\n"
            "6) Show visible progress from the previous window; avoid restaging the same pose or tableau.\n"
            "7) Make the action and state change obvious on screen.\n"
            "8) Add a short scene_conversation cue that explains what the characters are saying or emotionally exchanging in this window. Use reported speech or a very short dialogue beat, and never mention subtitles.\n"
            "9) If previous_scene_conversation is provided and scene_change is false, make the new scene_conversation feel like the next conversational turn instead of restarting the exchange from zero.\n"
            "10) If scene_change is true, start a fresh exchange for the new location or time jump, but preserve the emotional tension or objective from the previous scene.\n"
            "11) For dialogue-heavy windows, prefer stable conversational coverage, readable faces, clear gestures, and the same visible background instead of flashy camera changes.\n"
            "12) Keep both speakers grounded in the same physical location unless the beat explicitly requires movement away from the background anchor.\n"
            "Return JSON only with keys:\n"
            "{\"shot_type\":\"...\",\"camera_angle\":\"...\",\"camera_motion\":\"...\","
            "\"subject_blocking\":\"...\",\"action\":\"...\",\"continuity_anchor\":\"...\",\"scene_conversation\":\"...\"}\n"
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
            shot_plan = self._to_shot_plan(parsed, window)
            scene_conversation = self._to_scene_conversation(
                parsed,
                storyline,
                window,
                previous_scene_conversation,
            )
            prompt_text = self._prompt_from_shot_plan(
                shot_plan,
                window,
                continuity_note=self._continuity_note(memory_feedback),
                previous_context=self._previous_context(previous_prompt),
            )
            cleaned = self._normalize_prompt(prompt_text)
            return PromptBundle(
                prompt_text=cleaned,
                shot_plan=shot_plan,
                scene_conversation=scene_conversation,
            )

        return self._heuristic_refine_prompt(
            storyline,
            window,
            previous_prompt,
            previous_scene_conversation,
            memory_feedback,
        )

    @staticmethod
    def _extract_story_beats(storyline: str) -> List[str]:
        text = storyline.strip()
        if not text:
            return []

        # Prefer sentence/semicolon boundaries first.
        beats = [b.strip() for b in re.split(r"[.\n;:]+", text) if b.strip()]

        # If user provided one long comma-delimited line, split that into beats.
        if len(beats) <= 1 and "," in text:
            beats = [b.strip() for b in re.split(r",\s*", text) if b.strip()]

        # If still a single block, break gentle temporal connectors into beats.
        if len(beats) <= 1:
            beats = [
                b.strip()
                for b in re.split(r"\b(?:then|after that|next|finally|eventually)\b", text, flags=re.IGNORECASE)
                if b.strip()
            ]

        return beats

    @staticmethod
    def _expand_beats_for_windows(beats: List[str], window_count: int) -> List[str]:
        if len(beats) >= window_count:
            return beats[:window_count]

        # Spread beats across windows so each beat gets contiguous windows with
        # explicit visual progression instead of repeated "start/end phase" text.
        plan: List[str] = []
        for i, beat in enumerate(beats):
            start = round(i * window_count / len(beats))
            end = round((i + 1) * window_count / len(beats))
            count = max(1, end - start)
            for j in range(count):
                phase = "start" if j == 0 else ("end" if j == count - 1 else "middle")
                if phase == "start":
                    plan.append(f"Start this beat clearly: {beat}. Show the action beginning on screen.")
                elif phase == "middle":
                    plan.append(f"Continue this beat with visible progress: {beat}. Show a clear mid-action change, not a reset.")
                else:
                    plan.append(f"Resolve this beat clearly: {beat}. Show the outcome or state change on screen.")
        return plan[:window_count]

    @staticmethod
    def _default_visual_seed(beat: str, is_opening: bool, environment_anchor: str = "") -> str:
        anchor_clause = f", environment anchor: {environment_anchor}" if environment_anchor else ""
        if is_opening:
            return (
                "Establishing cinematic shot, clear subjects, stable identity and environment, "
                f"story action: {beat}{anchor_clause}"
            )
        return (
            "Continue naturally from previous clip with motion continuity and clear story progression, "
            f"story action: {beat}{anchor_clause}"
        )

    def _heuristic_refine_prompt(
        self,
        storyline: str,
        window: SceneWindow,
        previous_prompt: str,
        previous_scene_conversation: str,
        memory_feedback: Optional[Dict[str, Any]],
    ) -> PromptBundle:
        continuity_note = self._continuity_note(memory_feedback)
        previous_context = self._previous_context(previous_prompt)
        shot_type, camera_angle, camera_motion = self._default_shot_plan(window.index)
        continuity_anchor = window.environment_anchor.strip() or "preserve location layout, lighting, and key objects"
        shot_plan = ShotPlan(
            shot_type=shot_type,
            camera_angle=camera_angle,
            camera_motion=camera_motion,
            subject_blocking="keep main subjects consistent and centered",
            continuity_anchor=continuity_anchor,
            action=window.beat,
        )
        prompt_text = SceneDirector._prompt_from_shot_plan(
            shot_plan,
            window,
            continuity_note=continuity_note,
            previous_context=previous_context,
        )
        scene_conversation = self._heuristic_scene_conversation(
            storyline,
            window,
            previous_scene_conversation,
        )
        return PromptBundle(
            prompt_text=prompt_text,
            shot_plan=shot_plan,
            scene_conversation=scene_conversation,
        )

    @staticmethod
    def _normalize_scene_conversation(text: str) -> str:
        convo = " ".join((text or "").strip().split())
        return convo[:240]

    @staticmethod
    def _compact_scene_conversation(text: str) -> str:
        convo = " ".join((text or "").strip().split())
        return convo[:180]

    @staticmethod
    def _extract_character_names(text: str) -> List[str]:
        deny = {
            "The",
            "This",
            "That",
            "At",
            "When",
            "Then",
            "After",
            "Before",
            "Later",
            "Meanwhile",
            "Finally",
            "Eventually",
            "Current",
            "Previous",
            "Start",
            "Continue",
            "Resolve",
            "Show",
            "Years",
            "Flashback",
            "Present",
            "Story",
            "Stories",
            "Epilogue",
            "Encouraged",
            "Objective",
            "Emotion",
        }
        names: List[str] = []
        for token in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text or ""):
            token = re.sub(r"^(?:At|In|On|Back|Inside|Outside|Later|Earlier|From)\s+", "", token).strip()
            if len(token) <= 2:
                continue
            if token in deny:
                continue
            if any(token == existing or token in existing or existing in token for existing in names):
                continue
            names.append(token)
            if len(names) >= 2:
                break
        return names

    def _heuristic_scene_conversation(
        self,
        storyline: str,
        window: SceneWindow,
        previous_scene_conversation: str,
    ) -> str:
        beat_text = self._normalize_scene_conversation(str(window.beat).rstrip("."))
        beat_names = self._extract_character_names(window.beat)
        story_names = []
        for name in self._extract_character_names(storyline):
            if any(name == beat_name or name in beat_name or beat_name in name for beat_name in beat_names):
                continue
            story_names.append(name)
        names = (beat_names + story_names)[:2]
        previous_turn = self._compact_scene_conversation(previous_scene_conversation)
        scene_change = bool(window.scene_change)
        if len(names) >= 2:
            if previous_turn:
                if scene_change:
                    return self._normalize_scene_conversation(
                        f"Carry over the emotional tension into a new exchange as {names[0]} and {names[1]} push this scene beat forward: {beat_text}."
                    )
                return self._normalize_scene_conversation(
                    f"Continuing the earlier exchange in the same setting, {names[0]} and {names[1]} speak and react in a way that pushes the scene toward this beat: {beat_text}."
                )
            return self._normalize_scene_conversation(
                f"{names[0]} and {names[1]} have a clear back-and-forth conversation that directly pushes this scene beat forward: {beat_text}."
            )
        if len(names) == 1:
            if previous_turn:
                if scene_change:
                    return self._normalize_scene_conversation(
                        f"Carry over the emotional tension into a new exchange as {names[0]} responds in a way that drives this scene beat forward: {beat_text}."
                    )
                return self._normalize_scene_conversation(
                    f"Continuing the earlier exchange in the same setting, {names[0]} responds while another speaker reacts, driving this scene beat forward: {beat_text}."
                )
            return self._normalize_scene_conversation(
                f"{names[0]} speaks while another person responds, creating a clear back-and-forth that drives this scene beat forward: {beat_text}."
            )
        if previous_turn:
            if scene_change:
                return self._normalize_scene_conversation(
                    f"Carry over the emotional tension into a new exchange while clearly advancing this scene beat: {beat_text}."
                )
            return self._normalize_scene_conversation(
                f"Continue the earlier exchange in the same setting while clearly advancing this scene beat: {beat_text}."
            )
        return self._normalize_scene_conversation(
            f"Two people in the same setting should have a clear conversation that advances this scene beat: {beat_text}."
        )

    def _to_scene_conversation(
        self,
        parsed: Dict[str, Any],
        storyline: str,
        window: SceneWindow,
        previous_scene_conversation: str,
    ) -> str:
        for key in ("scene_conversation", "dialogue_beat", "conversation"):
            value = parsed.get(key)
            if isinstance(value, str):
                cleaned = self._normalize_scene_conversation(value)
                if cleaned:
                    return cleaned
        return self._heuristic_scene_conversation(
            storyline,
            window,
            previous_scene_conversation,
        )

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        prompt = " ".join((text or "").strip().split())
        return prompt[:700]

    @staticmethod
    def _compact_previous_prompt(text: str) -> str:
        if not text:
            return ""
        cleaned = text.split(" Previous visual context:")[0].strip()
        return cleaned[:260]

    @staticmethod
    def _previous_context(previous_prompt: str) -> str:
        if not previous_prompt:
            return ""
        compact_prev = previous_prompt.split(" Previous visual context:")[0].strip()
        compact_prev = compact_prev[:240]
        return f" Previous visual context: {compact_prev}" if compact_prev else ""

    @staticmethod
    def _continuity_note(memory_feedback: Optional[Dict[str, Any]]) -> str:
        continuity = "maintain continuity with previous clip"
        if memory_feedback:
            note = memory_feedback.get("suggested_constraints")
            if isinstance(note, str) and note.strip():
                continuity = note.strip()
        return continuity

    @staticmethod
    def _normalize_beats_override(beats_override: List[Any]) -> List[Dict[str, Any]]:
        beats: List[Dict[str, Any]] = []
        for item in beats_override:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    beats.append({"beat": cleaned})
            elif isinstance(item, dict):
                beat = str(item.get("beat", "")).strip()
                if beat:
                    entry: Dict[str, Any] = {"beat": beat}
                    scene_id = str(item.get("scene_id", "")).strip()
                    if scene_id:
                        entry["scene_id"] = scene_id
                    environment_anchor = " ".join(str(item.get("environment_anchor", "")).split()).strip()
                    if environment_anchor:
                        entry["environment_anchor"] = environment_anchor
                    character_lock = " ".join(str(item.get("character_lock", "")).split()).strip()
                    if character_lock:
                        entry["character_lock"] = character_lock
                    scene_change = item.get("scene_change")
                    if isinstance(scene_change, bool):
                        entry["scene_change"] = scene_change
                    beats.append(entry)
        return beats

    def _default_shot_plan(self, index: int) -> tuple[str, str, str]:
        presets = {
            "cinematic": [
                ("wide establishing", "eye-level", "slow dolly-in"),
                ("medium two-shot", "eye-level", "gentle tracking"),
                ("close-up action", "slightly low", "subtle handheld"),
                ("medium profile", "eye-level", "locked-off"),
            ],
            "docu": [
                ("medium docu", "shoulder height", "handheld sway"),
                ("wide street", "eye-level", "walk-and-talk tracking"),
                ("tight interview", "eye-level", "locked tripod"),
                ("cutaway detail", "eye-level", "slow push"),
            ],
            "action": [
                ("wide action", "low angle", "fast dolly"),
                ("medium chase", "eye-level", "energetic tracking"),
                ("close impact", "slightly low", "handheld shake"),
                ("aerial reveal", "high angle", "crane down"),
            ],
        }
        chosen = presets.get(self.config.shot_plan_defaults, presets["cinematic"])
        return chosen[index % len(chosen)]

    def _to_shot_plan(self, parsed: Dict[str, Any], window: SceneWindow) -> ShotPlan:
        def _field(name: str, fallback: str) -> str:
            value = parsed.get(name)
            if isinstance(value, str):
                cleaned = " ".join(value.split()).strip()
                if cleaned:
                    return cleaned
            return fallback

        shot_type, camera_angle, camera_motion = self._default_shot_plan(window.index)
        return ShotPlan(
            shot_type=_field("shot_type", shot_type),
            camera_angle=_field("camera_angle", camera_angle),
            camera_motion=_field("camera_motion", camera_motion),
            subject_blocking=_field("subject_blocking", "keep main subjects centered and consistent"),
            continuity_anchor=_field(
                "continuity_anchor",
                window.environment_anchor.strip() or "preserve previous location layout and lighting",
            ),
            action=_field("action", window.beat),
        )

    @staticmethod
    def _prompt_from_shot_plan(
        shot_plan: ShotPlan,
        window: SceneWindow,
        continuity_note: str,
        previous_context: str,
    ) -> str:
        base = (
            f"Shot type: {shot_plan.shot_type}. "
            f"Camera angle: {shot_plan.camera_angle}. "
            f"Camera motion: {shot_plan.camera_motion}. "
            f"Subject blocking: {shot_plan.subject_blocking}. "
            f"Action: {shot_plan.action}. "
            f"Continuity anchor: {shot_plan.continuity_anchor}. "
            f"Time window {window.start_sec}-{window.end_sec}s, {continuity_note}.{previous_context}"
        )
        return SceneDirector._normalize_prompt(base)

    @staticmethod
    def _compact_storyline(storyline: str) -> str:
        text = " ".join((storyline or "").split())
        return text[:500]

    @staticmethod
    def _memory_feedback_text(memory_feedback: Optional[Dict[str, Any]]) -> str:
        if not memory_feedback:
            return ""
        note = memory_feedback.get("suggested_constraints")
        if isinstance(note, str):
            return note[:220]
        return ""

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None
