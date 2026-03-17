from __future__ import annotations

import re
from typing import Any, Dict, List


WINDOW_ID_RE = re.compile(r"^w_[0-9]{3,}$")

REQUIRED_SCENE_PLAN_FIELDS = {
    "story_id",
    "title",
    "window_seconds",
    "total_windows",
    "windows",
}

REQUIRED_WINDOW_FIELDS = {
    "window_id",
    "window_index",
    "beat_id",
    "scene_objective",
    "emotion",
    "character_names",
    "continuity_anchor",
}


def _add_error(errors: List[Dict[str, str]], path: str, message: str) -> None:
    errors.append({"path": path, "message": message})


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _norm_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact


def _split_sentences(text: str) -> List[str]:
    if not _is_non_empty_string(text):
        return []
    chunks = re.split(r"(?<=[.!?])\s+", _norm_text(text))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def validate_scene_plan(payload: Any) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if not isinstance(payload, dict):
        _add_error(errors, "$", "scene plan must be a JSON object")
        return errors

    missing = sorted(REQUIRED_SCENE_PLAN_FIELDS - set(payload.keys()))
    for field_name in missing:
        _add_error(errors, field_name, "missing required field")

    if "story_id" in payload and not _is_non_empty_string(payload.get("story_id")):
        _add_error(errors, "story_id", "must be a non-empty string")
    if "title" in payload and not _is_non_empty_string(payload.get("title")):
        _add_error(errors, "title", "must be a non-empty string")

    if "window_seconds" in payload:
        value = payload.get("window_seconds")
        if not isinstance(value, int) or value <= 0:
            _add_error(errors, "window_seconds", "must be a positive integer")

    if "total_windows" in payload:
        value = payload.get("total_windows")
        if not isinstance(value, int) or value <= 0:
            _add_error(errors, "total_windows", "must be a positive integer")

    windows = payload.get("windows")
    if "windows" in payload:
        if not isinstance(windows, list):
            _add_error(errors, "windows", "must be a list")
            windows = []
        elif not windows:
            _add_error(errors, "windows", "must contain at least one window")

    if isinstance(windows, list) and "total_windows" in payload and isinstance(payload.get("total_windows"), int):
        if len(windows) != payload["total_windows"]:
            _add_error(
                errors,
                "total_windows",
                f"declares {payload['total_windows']}, but windows has {len(windows)} items",
            )

    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for idx, window in enumerate(windows or []):
        base = f"windows[{idx}]"
        if not isinstance(window, dict):
            _add_error(errors, base, "must be an object")
            continue

        missing_window_fields = sorted(REQUIRED_WINDOW_FIELDS - set(window.keys()))
        for field_name in missing_window_fields:
            _add_error(errors, f"{base}.{field_name}", "missing required field")

        window_id = window.get("window_id")
        if "window_id" in window:
            if not _is_non_empty_string(window_id):
                _add_error(errors, f"{base}.window_id", "must be a non-empty string")
            elif not WINDOW_ID_RE.match(window_id):
                _add_error(errors, f"{base}.window_id", "must match pattern w_000")
            elif window_id in seen_ids:
                _add_error(errors, f"{base}.window_id", f"duplicate window_id '{window_id}'")
            else:
                seen_ids.add(window_id)

        window_index = window.get("window_index")
        if "window_index" in window:
            if not isinstance(window_index, int) or window_index < 0:
                _add_error(errors, f"{base}.window_index", "must be an integer >= 0")
            elif window_index in seen_indices:
                _add_error(errors, f"{base}.window_index", f"duplicate window_index '{window_index}'")
            else:
                seen_indices.add(window_index)

        for field_name in ("beat_id", "scene_objective", "emotion"):
            if field_name in window and not _is_non_empty_string(window.get(field_name)):
                _add_error(errors, f"{base}.{field_name}", "must be a non-empty string")

        character_names = window.get("character_names")
        if "character_names" in window:
            if not isinstance(character_names, list):
                _add_error(errors, f"{base}.character_names", "must be a list")
            elif not character_names:
                _add_error(errors, f"{base}.character_names", "must contain at least one character")
            else:
                for c_idx, char_name in enumerate(character_names):
                    if not _is_non_empty_string(char_name):
                        _add_error(
                            errors,
                            f"{base}.character_names[{c_idx}]",
                            "must be a non-empty string",
                        )

        continuity = window.get("continuity_anchor")
        if "continuity_anchor" in window:
            if not isinstance(continuity, dict):
                _add_error(errors, f"{base}.continuity_anchor", "must be an object")
            else:
                world_anchor = continuity.get("world_anchor")
                if not _is_non_empty_string(world_anchor):
                    _add_error(errors, f"{base}.continuity_anchor.world_anchor", "must be a non-empty string")
                previous_window_id = continuity.get("previous_window_id")
                if previous_window_id is not None and not _is_non_empty_string(previous_window_id):
                    _add_error(
                        errors,
                        f"{base}.continuity_anchor.previous_window_id",
                        "must be null or a non-empty string",
                    )

        if "expected_caption" in window and not _is_non_empty_string(window.get("expected_caption")):
            _add_error(errors, f"{base}.expected_caption", "if provided, must be a non-empty string")

    return errors


def _compact_caption(window: Dict[str, Any]) -> str:
    continuity = window.get("continuity_anchor", {})
    world_anchor = str(continuity.get("world_anchor", "scene")).replace("_", " ")
    characters = ", ".join(window["character_names"])
    emotion = _norm_text(window["emotion"].rstrip("."))
    objective = _norm_text(window["scene_objective"].rstrip("."))

    seed_caption = ""
    if _is_non_empty_string(window.get("expected_caption")):
        parts = _split_sentences(window["expected_caption"])
        if parts:
            seed_caption = parts[0].replace("_", " ").rstrip(".")

    if not seed_caption:
        seed_caption = f"{characters} drive the scene objective"

    lower_seed = seed_caption.lower()
    canonical_prefixes = (
        f"{characters.lower()} at {world_anchor.lower()}",
        f"{characters.lower()} in {world_anchor.lower()}",
    )
    if lower_seed.startswith(canonical_prefixes):
        primary_sentence = seed_caption
    else:
        primary_sentence = f"{characters} at {world_anchor}. {seed_caption}"

    beat_step = window.get("beat_step")
    beat_total = window.get("beat_total_steps")
    step_suffix = ""
    if isinstance(beat_step, int) and isinstance(beat_total, int) and beat_total > 0:
        step_suffix = f" Beat step: {beat_step}/{beat_total}."

    return _norm_text(f"{primary_sentence}. Objective: {objective}. Emotion: {emotion}.{step_suffix}")


def _detailed_caption(window: Dict[str, Any], compact_caption: str) -> str:
    continuity = window.get("continuity_anchor", {})
    world_anchor = str(continuity.get("world_anchor", "scene")).replace("_", " ")
    prev = continuity.get("previous_window_id")
    if prev is None:
        continuity_text = "Opens the scene arc."
    else:
        continuity_text = f"Must remain consistent with {prev}."

    return _norm_text(
        f"{compact_caption} Beat={window['beat_id']}. Anchor={world_anchor}. {continuity_text}"
    )


def build_expected_caption_pack(scene_plan: Dict[str, Any], *, style: str = "compact") -> Dict[str, Any]:
    errors = validate_scene_plan(scene_plan)
    if errors:
        rendered = "\n".join(f"- {err['path']}: {err['message']}" for err in errors)
        raise ValueError(f"Invalid scene plan:\n{rendered}")

    if style not in {"compact", "detailed"}:
        raise ValueError("style must be one of: compact, detailed")

    items: List[Dict[str, Any]] = []
    for window in scene_plan["windows"]:
        compact = _compact_caption(window)
        detailed = _detailed_caption(window, compact)

        expected = compact if style == "compact" else detailed
        continuity = window["continuity_anchor"]

        items.append(
            {
                "window_id": window["window_id"],
                "window_index": window["window_index"],
                "beat_id": window["beat_id"],
                "expected_caption": expected,
                "expected_caption_compact": compact,
                "expected_caption_detailed": detailed,
                "alignment_targets": {
                    "emotion": window["emotion"],
                    "character_names": window["character_names"],
                    "world_anchor": continuity["world_anchor"],
                    "previous_window_id": continuity.get("previous_window_id"),
                },
                "weights": {
                    "semantic_alignment": 0.60,
                    "entity_consistency": 0.25,
                    "continuity_consistency": 0.15,
                },
            }
        )

    return {
        "version": 1,
        "story_id": scene_plan["story_id"],
        "title": scene_plan["title"],
        "window_seconds": scene_plan["window_seconds"],
        "total_windows": scene_plan["total_windows"],
        "style": style,
        "source": {
            "module": "caption_gen",
            "scene_plan_version": scene_plan.get("version"),
        },
        "captions": items,
    }
