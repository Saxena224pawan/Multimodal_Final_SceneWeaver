import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable when running:
# `python scripts/run_story_pipeline.py ...`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from director_llm import SceneDirector, SceneDirectorConfig
from director_llm.scene_director import PromptBundle, ShotPlan
from Caption_Gen import build_dense_expected_caption_pack, build_expected_caption_pack
from memory_module import NarrativeMemory, VisionEmbedder, VisionEmbedderConfig
from memory_module.captioner import Captioner, CaptionerConfig
from memory_module.window_critic import evaluate_candidate
from pipeline_runtime import get_selected_model, load_model_links


def load_video_backbone() -> tuple[Any, Any]:
    try:
        from video_backbone import WanBackbone, WanBackboneConfig
    except Exception:  # pragma: no cover - fallback for partial worktrees
        from video_backbone.wan_backbone import WanBackbone, WanBackboneConfig
    return WanBackbone, WanBackboneConfig


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_jsonl(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_init_embedder(
    backend: str,
    model_id: Optional[str],
    adapter_ckpt: Optional[str],
    device: str,
) -> Optional[VisionEmbedder]:
    if backend == "none":
        return None
    # Backward-compatible with older VisionEmbedderConfig versions that
    # do not expose adapter_ckpt yet.
    try:
        cfg = VisionEmbedderConfig(
            backend=backend,
            model_id=model_id,
            adapter_ckpt=adapter_ckpt,
            device=device,
        )
    except TypeError:
        cfg = VisionEmbedderConfig(
            backend=backend,
            model_id=model_id,
            device=device,
        )
    embedder = VisionEmbedder(cfg)
    embedder.load()
    return embedder


def build_generation_prompt(
    refined_prompt: str,
    beat: str,
    style_prefix: str,
    character_lock: str,
    previous_environment_anchor: str,
    current_environment_anchor: str,
    scene_change_requested: bool,
    story_state_hint: str,
    scene_conversation: str = "",
    previous_scene_conversation: str = "",
    conversation_progress_instruction: str = "",
    story_progress_instruction: str = "",
    dialogue_scene: bool = False,
    repair_hint: str = "",
    shot_plan: Optional[ShotPlan] = None,
    shot_plan_enforce: bool = True,
) -> str:
    parts = []
    if shot_plan_enforce and shot_plan is not None:
        parts.append(
            " ".join(
                [
                    f"Shot type: {shot_plan.shot_type}.",
                    f"Camera angle: {shot_plan.camera_angle}.",
                    f"Camera motion: {shot_plan.camera_motion}.",
                    f"Subject blocking: {shot_plan.subject_blocking}.",
                    f"Continuity anchor: {shot_plan.continuity_anchor}.",
                ]
            )
        )
    if style_prefix.strip():
        parts.append(style_prefix.strip())
    if character_lock.strip():
        parts.append(f"Character continuity: {character_lock.strip()}")
    if previous_environment_anchor.strip() and scene_change_requested and current_environment_anchor.strip():
        parts.append(f"Previous environment anchor: {previous_environment_anchor.strip()}")
        parts.append(f"Current scene environment anchor: {current_environment_anchor.strip()}")
    elif current_environment_anchor.strip():
        parts.append(f"Environment continuity anchor: {current_environment_anchor.strip()}")
    elif previous_environment_anchor.strip():
        parts.append(f"Previous environment anchor: {previous_environment_anchor.strip()}")
    parts.append(f"Current beat: {beat.strip()}")
    if story_state_hint.strip():
        parts.append(f"Story state: {story_state_hint.strip()}")
    if dialogue_scene and previous_scene_conversation.strip():
        parts.append(f"Previous conversation context: {previous_scene_conversation.strip()}")
    if scene_conversation.strip():
        cue_label = "Scene conversation cue" if dialogue_scene else "Scene emotional cue"
        parts.append(f"{cue_label}: {scene_conversation.strip()}")
        if dialogue_scene:
            parts.append(
                "Show natural speaking, listening, reacting, or arguing body language that matches this conversation cue. No on-screen subtitles or text."
            )
            parts.append(
                "Keep the speakers in one shared physical setting with stable background landmarks, consistent seating or standing positions, and readable face-to-face eyelines."
            )
            parts.append(
                "Favor stable conversational coverage such as a medium two-shot or restrained over-the-shoulder continuation unless the beat clearly requires a different framing."
            )
        else:
            parts.append(
                "Treat this cue as a solo performance or internal reaction. Do not invent an extra speaker, subtitles, or an off-screen conversation."
            )
    if dialogue_scene and conversation_progress_instruction.strip():
        parts.append(f"Conversation progression instruction: {conversation_progress_instruction.strip()}")
    if story_progress_instruction.strip():
        parts.append(f"Story progression instruction: {story_progress_instruction.strip()}")
    parts.append(f"Shot prompt: {refined_prompt.strip()}")
    if scene_change_requested and current_environment_anchor.strip():
        parts.append(
            "Transition to the new planned location while preserving the exact background layout, lighting logic, and landmark props described by the current scene environment anchor."
        )
    elif scene_change_requested:
        parts.append("Beat suggests a setting change; transition from the previous clip naturally, not abruptly.")
    elif current_environment_anchor.strip():
        parts.append(
            "Preserve location, background layout, lighting, weather, time-of-day, and camera viewpoint from the environment anchor."
        )
        parts.append(
            "Lock the same background landmarks, prop placement, horizon line, and lighting direction across consecutive windows. Do not replace or relocate the main background objects."
        )
    parts.append(
        "Strictly follow the current beat and keep the same characters, identities, and scene context. "
        "No unrelated objects, no random scene changes."
    )
    if story_progress_instruction.strip():
        parts.append(
            "Do not repeat the exact previous pose, framing, or action loop. "
            "Show a visible state change that advances the story in this window."
        )
    if repair_hint.strip():
        parts.append(f"Critic repair constraints: {repair_hint.strip()}")
    return " ".join(parts)


def _compact_previous_prompt(prompt: str) -> str:
    if not prompt:
        return ""
    head = prompt.split(" Previous visual context:")[0].strip()
    return head[:240]


def _load_window_plan(path: str) -> List[Any]:
    plan_path = Path(path)
    if not plan_path.is_file():
        raise FileNotFoundError(f"window_plan_json not found: {plan_path}")
    with plan_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("window_plan_json must be a JSON array of beats")
    return data


def _cosine_similarity(v1: Optional[Any], v2: Optional[Any]) -> Optional[float]:
    if v1 is None or v2 is None:
        return None
    import numpy as np

    arr1 = np.asarray(v1)
    arr2 = np.asarray(v2)
    denom = (np.linalg.norm(arr1) * np.linalg.norm(arr2)) + 1e-12
    return float(np.dot(arr1, arr2) / denom)


def _extract_environment_anchor(prompt: str) -> str:
    if not prompt:
        return ""
    parts = re.split(r"[.;]", prompt)
    env_keywords = (
        "location",
        "setting",
        "environment",
        "background",
        "scene",
        "room",
        "house",
        "street",
        "forest",
        "field",
        "park",
        "beach",
        "mountain",
        "indoor",
        "outdoor",
        "interior",
        "exterior",
        "sky",
        "rain",
        "snow",
        "fog",
        "sunset",
        "night",
        "daylight",
        "lighting",
        "camera",
    )
    selected: List[str] = []
    for part in parts:
        candidate = " ".join(part.strip().split())
        if not candidate:
            continue
        lower = candidate.lower()
        if lower.startswith("previous visual context") or lower.startswith("time window"):
            continue
        if any(token in lower for token in env_keywords):
            selected.append(candidate)
        if len(selected) >= 2:
            break
    if selected:
        return "; ".join(selected)[:260]
    fallback = " ".join(prompt.strip().split())
    return fallback[:220]


def _window_scene_id(window: Any) -> str:
    return " ".join(str(getattr(window, "scene_id", "") or "").split()).strip()


def _window_environment_anchor(window: Any) -> str:
    return " ".join(str(getattr(window, "environment_anchor", "") or "").split()).strip()


def _window_character_lock(window: Any) -> str:
    return " ".join(str(getattr(window, "character_lock", "") or "").split()).strip()


def _window_scene_change(window: Any) -> Optional[bool]:
    value = getattr(window, "scene_change", None)
    return value if isinstance(value, bool) else None


def _window_story_phase(window: Any) -> str:
    return " ".join(str(getattr(window, "story_phase", "") or "").split()).strip()


def _window_character_progression(window: Any) -> str:
    return " ".join(str(getattr(window, "character_progression", "") or "").split()).strip()


def _window_relationship_dynamic(window: Any) -> str:
    return " ".join(str(getattr(window, "relationship_dynamic", "") or "").split()).strip()


def _window_visible_change(window: Any) -> str:
    return " ".join(str(getattr(window, "visible_change", "") or "").split()).strip()


def _same_scene_as_previous(previous_window: Optional[Any], current_window: Any) -> bool:
    if previous_window is None:
        return False
    previous_scene_id = _window_scene_id(previous_window)
    current_scene_id = _window_scene_id(current_window)
    if previous_scene_id and current_scene_id:
        return previous_scene_id == current_scene_id
    previous_anchor = _window_environment_anchor(previous_window)
    current_anchor = _window_environment_anchor(current_window)
    if previous_anchor and current_anchor:
        return previous_anchor == current_anchor
    return not _beat_requests_scene_change(current_window.beat)


def _scene_change_requested(previous_window: Optional[Any], current_window: Any) -> bool:
    explicit_change = _window_scene_change(current_window)
    if explicit_change is not None:
        return explicit_change
    if previous_window is None:
        return False
    previous_scene_id = _window_scene_id(previous_window)
    current_scene_id = _window_scene_id(current_window)
    if previous_scene_id and current_scene_id and previous_scene_id != current_scene_id:
        return True
    previous_anchor = _window_environment_anchor(previous_window)
    current_anchor = _window_environment_anchor(current_window)
    if previous_anchor and current_anchor and previous_anchor != current_anchor:
        return True
    return _beat_requests_scene_change(current_window.beat)


def _merge_character_lock(global_lock: str, window_lock: str) -> str:
    compact_global = " ".join((global_lock or "").split()).strip()
    compact_window = " ".join((window_lock or "").split()).strip()
    if compact_global and compact_window:
        if compact_window in compact_global:
            return compact_global
        return f"{compact_global} {compact_window}"
    return compact_window or compact_global


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


def _normalize_beat_core(beat: str) -> str:
    text = (beat or "").strip()
    text = re.sub(r"^(Start this beat clearly:|Continue this beat with visible progress:|Resolve this beat clearly:)\s*", "", text)
    text = re.sub(r"\s*Show .*? on screen\.?$", "", text)
    text = re.sub(r"\s*Show a clear mid-action change, not a reset\.?$", "", text)
    return " ".join(text.split()).strip(" .")


_LOCATION_NAME_TERMS = {
    "Mountain",
    "Mountains",
    "Valley",
    "Valleys",
    "Forest",
    "Forests",
    "Ruins",
    "Peak",
    "Peaks",
    "Path",
    "Road",
    "River",
    "Lake",
    "Sea",
    "Ocean",
    "Castle",
    "Village",
    "City",
    "Temple",
    "Tower",
    "Desert",
    "Cave",
    "Caves",
    "Kingdom",
    "Courtyard",
    "Station",
    "Platform",
    "Train",
}


_DIALOGUE_PARTICIPANT_MARKERS = (
    " says ",
    " ask ",
    " asks ",
    " tell ",
    " tells ",
    " reply ",
    " replies ",
    " respond ",
    " responds ",
    " answer ",
    " answers ",
    " shout ",
    " shouts ",
    " yell ",
    " yells ",
    " argue ",
    " argues ",
    " argues with ",
    " speak ",
    " speaks ",
    " talk ",
    " talks ",
    " debate ",
    " debates ",
    " negotiate ",
    " negotiates ",
    " conversation ",
    " dialogue ",
    " exchange ",
)


_SELF_TALK_MARKERS = (
    " to herself ",
    " to himself ",
    " to themself ",
    " to myself ",
    " under her breath ",
    " under his breath ",
    " under their breath ",
    " quietly to herself ",
    " quietly to himself ",
    " mutters to herself ",
    " mutters to himself ",
    " whispers to herself ",
    " whispers to himself ",
    " whispers to themself ",
    " alone ",
    " internal monologue ",
    " inner monologue ",
)


def _candidate_is_location_name(token: str) -> bool:
    words = [word for word in str(token or "").split() if word]
    return any(word in _LOCATION_NAME_TERMS for word in words)


def _normalized_lower(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    return f" {normalized} "


def _is_self_talk_cue(text: str) -> bool:
    lowered = _normalized_lower(text)
    return any(marker in lowered for marker in _SELF_TALK_MARKERS)


def _should_use_dialogue_staging(
    scene_conversation: str,
    beat: str,
    character_names: Optional[List[str]] = None,
) -> bool:
    if not str(scene_conversation or "").strip():
        return False
    if _is_self_talk_cue(scene_conversation) or _is_self_talk_cue(beat):
        return False

    resolved_names = [
        str(name).strip()
        for name in (character_names or [])
        if str(name).strip() and str(name).strip().lower() != "protagonist"
    ]
    if len(resolved_names) >= 2:
        return True

    lowered = _normalized_lower(f"{scene_conversation} {beat}")
    participant_markers = (
        " another person ",
        " another speaker ",
        " both speakers ",
        " two people ",
        " back-and-forth ",
        " each other ",
        " one another ",
        " conversation ",
        " dialogue ",
        " exchange ",
        " responds while ",
        " replies while ",
    )
    return any(marker in lowered for marker in participant_markers)


def _story_progress_instruction(previous_beat: str, current_beat: str) -> str:
    if not previous_beat.strip():
        return "Establish the first beat clearly with a readable action and objective."
    prev_core = _normalize_beat_core(previous_beat)
    curr_core = _normalize_beat_core(current_beat)
    if prev_core and curr_core and prev_core != curr_core:
        return (
            f"Transition away from the previous beat '{prev_core}' and make the new beat '{curr_core}' "
            "obvious on screen. The action change must be immediately readable."
        )
    return (
        "Show clear progress within the same beat. Change body pose, object state, and staging from the previous window; "
        "do not simply restage the same tableau."
    )


def _conversation_progress_instruction(
    previous_scene_conversation: str,
    current_beat: str,
    next_beat: str,
    scene_change_requested: bool,
) -> str:
    current_core = _normalize_beat_core(current_beat)
    next_core = _normalize_beat_core(next_beat)
    if previous_scene_conversation.strip():
        if scene_change_requested:
            base = "Start a fresh exchange in the new scene, but preserve the emotional thread from the previous window."
        else:
            base = (
                "Continue the emotional exchange from the previous window instead of restarting a new conversation. "
                "Keep both speakers in the same background and preserve the shared setting while the dialogue advances."
            )
    else:
        base = "Establish the first meaningful exchange for this part of the story with both speakers clearly grounded in one shared setting."
    if current_core:
        base = f"{base} Keep the spoken intent centered on '{current_core}'."
    if next_core:
        base = f"{base} Let this exchange naturally set up the next beat '{next_core}' without jumping there fully."
    else:
        base = f"{base} Let the exchange land clearly by the end of this window."
    return base


def _reference_strength_for_window(
    base_strength: float,
    scene_change_requested: bool,
    reference_source: str,
    same_scene_as_previous: bool,
) -> float:
    strength = float(base_strength)
    if reference_source == "window_reference_image":
        return min(max(strength, 0.72), 0.86)
    if reference_source == "initial_condition_image":
        return min(max(strength, 0.66), 0.78)
    if reference_source == "previous_window_tail":
        tail_strength = min(strength, 0.72)
        if same_scene_as_previous:
            return min(max(tail_strength, 0.60), 0.72)
        if scene_change_requested:
            return min(tail_strength, 0.38)
        return min(max(tail_strength, 0.56), 0.68)
    if scene_change_requested:
        return min(strength, 0.45)
    return min(max(strength, 0.62), 0.76)


def _merge_negative_prompt_terms(base_negative_prompt: str, extra_terms: List[str]) -> str:
    terms = [term.strip() for term in re.split(r",\s*", base_negative_prompt or "") if term.strip()]
    normalized = {term.lower() for term in terms}
    for term in extra_terms:
        cleaned = str(term or "").strip()
        if cleaned and cleaned.lower() not in normalized:
            terms.append(cleaned)
            normalized.add(cleaned.lower())
    return ", ".join(terms)


def _attempt_progress(attempt_idx: int, max_attempts: int) -> float:
    if max_attempts <= 1:
        return 0.0
    return max(0.0, min(1.0, attempt_idx / max(1, max_attempts - 1)))


def _tightened_retry_settings(
    *,
    base_guidance_scale: float,
    base_negative_prompt: str,
    base_reference_strength: float,
    base_noise_blend_strength: float,
    attempt_idx: int,
    max_attempts: int,
    progressive_tightening: bool,
    tightening_strength: float,
) -> Dict[str, Any]:
    progress = _attempt_progress(attempt_idx, max_attempts) if progressive_tightening else 0.0
    strength = max(0.0, float(tightening_strength))
    guidance_scale = min(14.0, max(0.0, float(base_guidance_scale) + (progress * strength * 1.25)))
    reference_strength = min(0.95, max(0.0, float(base_reference_strength) + (progress * strength * 0.12)))
    noise_blend_strength = max(0.0, float(base_noise_blend_strength))
    if noise_blend_strength > 0.0:
        noise_blend_strength = min(0.65, noise_blend_strength + (progress * strength * 0.06))

    extra_terms: List[str] = []
    repair_constraints: List[str] = []
    if progress > 0.0:
        extra_terms.extend([
            "identity drift",
            "background drift",
            "beat confusion",
            "unclear action",
            "subject inconsistency",
        ])
        repair_constraints.append(
            "Stronger retry pass: preserve exact character identity, environment layout, and readable beat progression."
        )
    if progress >= 0.45:
        extra_terms.extend([
            "pose reset",
            "emotion reset",
            "scene discontinuity",
            "extra characters",
            "duplicate subjects",
        ])
        repair_constraints.append(
            "No pose reset, no emotion reset, no background drift, no extra subjects, and no weakened action clarity."
        )
    if progress >= 0.8:
        extra_terms.extend([
            "staging drift",
            "prop inconsistency",
            "weak eyelines",
        ])
        repair_constraints.append(
            "Make the beat unmistakable on screen with cleaner staging, stronger eyelines, and tighter prop continuity."
        )

    return {
        "attempt_progress": round(progress, 4),
        "guidance_scale": round(guidance_scale, 4),
        "reference_strength": round(reference_strength, 4),
        "noise_blend_strength": round(noise_blend_strength, 4),
        "negative_prompt": _merge_negative_prompt_terms(base_negative_prompt, extra_terms),
        "repair_constraint": " ".join(repair_constraints).strip(),
    }


def _retry_convergence_status(
    score_history: List[float],
    *,
    threshold: float,
    patience: int,
    tolerance: float,
) -> tuple[bool, str]:
    if not score_history:
        return False, ""
    latest = float(score_history[-1])
    if latest >= float(threshold):
        return True, "threshold_met"
    if int(patience) <= 0 or len(score_history) < int(patience) + 1:
        return False, ""
    recent = [float(score) for score in score_history[-(int(patience) + 1) :]]
    if max(recent) - min(recent) <= max(0.0, float(tolerance)):
        return True, "score_plateau"
    return False, ""


def _combine_continuity_score(
    transition_similarity: Optional[float],
    environment_similarity: Optional[float],
    style_similarity: Optional[float],
    transition_weight: float,
    environment_weight: float,
    style_weight: float,
) -> Optional[float]:
    weighted_components = []
    if transition_similarity is not None and transition_weight > 0.0:
        weighted_components.append((transition_similarity, transition_weight))
    if environment_similarity is not None and environment_weight > 0.0:
        weighted_components.append((environment_similarity, environment_weight))
    if style_similarity is not None and style_weight > 0.0:
        weighted_components.append((style_similarity, style_weight))
    if not weighted_components:
        return None
    total_weight = sum(weight for _, weight in weighted_components)
    if total_weight <= 0.0:
        return None
    weighted_sum = sum(score * weight for score, weight in weighted_components)
    return float(weighted_sum / total_weight)


def _estimate_frame_visual_quality(frame: Any) -> float:
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3:
        gray = arr[..., :3].astype(np.float32).mean(axis=2)
    else:
        return 0.0

    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return 0.0
    min_val = float(finite.min())
    max_val = float(finite.max())
    if min_val >= 0.0 and max_val <= 1.0:
        gray = gray * 255.0
    elif min_val >= -1.0 and max_val <= 1.0:
        gray = (gray + 1.0) * 127.5

    grad_x = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    grad_y = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    sharpness = min(max(float(grad_x + grad_y) / 26.0, 0.0), 1.0)
    contrast = min(max(float(np.std(gray)) / 52.0, 0.0), 1.0)
    brightness = float(np.mean(gray) / 255.0)
    exposure = max(0.0, 1.0 - (abs(brightness - 0.48) / 0.42))
    return float((0.62 * sharpness) + (0.23 * contrast) + (0.15 * exposure))


def _estimate_clip_visual_quality(frames: List[Any], sample_count: int = 4) -> Optional[float]:
    if frames is None:
        return None
    try:
        total = len(frames)
    except TypeError:
        return None
    if total <= 0:
        return None
    if total <= sample_count:
        sampled = list(frames)
    else:
        step = max(1, total // sample_count)
        sampled = list(frames)[::step][:sample_count]
    if not sampled:
        return None
    scores = [_estimate_frame_visual_quality(frame) for frame in sampled]
    return float(sum(scores) / len(scores))


def _prepare_frame_style_array(frame: Any) -> Optional[Any]:
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim not in (2, 3):
        return None
    arr = arr.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    min_val = float(finite.min())
    max_val = float(finite.max())
    if min_val >= 0.0 and max_val <= 1.0:
        arr = arr * 255.0
    elif min_val >= -1.0 and max_val <= 1.0:
        arr = (arr + 1.0) * 127.5
    return np.clip(arr, 0.0, 255.0)


def _estimate_frame_style_signature(frame: Any) -> Optional[Dict[str, float]]:
    import numpy as np

    arr = _prepare_frame_style_array(frame)
    if arr is None:
        return None
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=2)
    else:
        if arr.shape[2] == 1:
            rgb = np.repeat(arr, 3, axis=2)
        else:
            rgb = arr[..., :3]

    gray = rgb.mean(axis=2)
    mean_rgb = rgb.reshape(-1, 3).mean(axis=0)
    brightness = float(gray.mean() / 255.0)
    contrast = float(min(max(np.std(gray) / 64.0, 0.0), 1.0))
    color_spread = float((max(mean_rgb) - min(mean_rgb)) / 255.0)
    channel_delta = np.abs(rgb - gray[..., None]).mean(axis=2)
    saturation = float(min(max(channel_delta.mean() / 85.0, 0.0), 1.0))
    grad_x = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    grad_y = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    texture = float(min(max((grad_x + grad_y) / 26.0, 0.0), 1.0))
    return {
        "mean_r": float(mean_rgb[0] / 255.0),
        "mean_g": float(mean_rgb[1] / 255.0),
        "mean_b": float(mean_rgb[2] / 255.0),
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "color_spread": color_spread,
        "texture": texture,
    }


def _estimate_clip_style_signature(frames: List[Any], sample_count: int = 4) -> Optional[Dict[str, float]]:
    if frames is None:
        return None
    try:
        total = len(frames)
    except TypeError:
        return None
    if total <= 0:
        return None
    if total <= sample_count:
        sampled = list(frames)
    else:
        step = max(1, total // sample_count)
        sampled = list(frames)[::step][:sample_count]
    if not sampled:
        return None

    signatures = [sig for sig in (_estimate_frame_style_signature(frame) for frame in sampled) if sig is not None]
    if not signatures:
        return None

    keys = signatures[0].keys()
    return {key: float(sum(sig[key] for sig in signatures) / len(signatures)) for key in keys}


def _style_signature_similarity(
    reference_signature: Optional[Dict[str, float]],
    candidate_signature: Optional[Dict[str, float]],
) -> Optional[float]:
    if not reference_signature or not candidate_signature:
        return None
    keys = [
        key
        for key in reference_signature.keys()
        if key in candidate_signature and reference_signature.get(key) is not None and candidate_signature.get(key) is not None
    ]
    if not keys:
        return None
    diffs = [abs(float(reference_signature[key]) - float(candidate_signature[key])) for key in keys]
    avg_diff = sum(diffs) / len(diffs)
    return float(min(max(1.0 - avg_diff, 0.0), 1.0))


def _visual_quality_feedback(visual_quality_score: Optional[float]) -> str:
    if visual_quality_score is None or visual_quality_score >= 0.58:
        return ""
    if visual_quality_score < 0.36:
        return (
            "Reduce accumulated denoising artifacts, haze, and blur. Keep faces, eyes, and mouths clean and sharp. "
            "Do not let the previous window reference overpower the new generation."
        )
    return "Preserve crisp facial detail and cleaner edges. Keep the continuation anchor gentle instead of over-copying noisy texture."


def _style_similarity_feedback(style_similarity: Optional[float]) -> str:
    if style_similarity is None or style_similarity >= 0.78:
        return ""
    if style_similarity >= 0.64:
        return "Keep the same overall visual style, palette, lighting mood, and texture level as the previous window."
    return "Visual style drifted badly; restore the same palette, lighting mood, contrast, and rendering texture from the earlier window."


def _build_story_state_hint(windows: List[Any], pos: int) -> str:
    previous_beat = windows[pos - 1].beat if pos > 0 else ""
    current_window = windows[pos]
    current_beat = current_window.beat
    next_beat = windows[pos + 1].beat if pos + 1 < len(windows) else ""
    future_beat = windows[pos + 2].beat if pos + 2 < len(windows) else ""

    hints: List[str] = []
    if previous_beat:
        hints.append(f"Completed previous beat: {previous_beat}.")
    hints.append(f"Required now: {current_beat}.")
    story_phase = _window_story_phase(current_window)
    if story_phase:
        hints.append(f"Story phase now: {story_phase}.")
    character_progression = _window_character_progression(current_window)
    if character_progression:
        hints.append(f"Character progression now: {character_progression}.")
    relationship_dynamic = _window_relationship_dynamic(current_window)
    if relationship_dynamic:
        hints.append(f"Relationship dynamic now: {relationship_dynamic}.")
    visible_change = _window_visible_change(current_window)
    if visible_change:
        hints.append(f"Visible change required: {visible_change}.")
    if previous_beat:
        if _normalize_beat_core(previous_beat) != _normalize_beat_core(current_beat):
            hints.append("Visibly change the action from the previous window; do not hold on the earlier state.")
        else:
            hints.append("Show clear within-beat progress rather than repeating the same pose or action loop.")
    if next_beat:
        hints.append(f"Next beat after this window: {next_beat}.")
    if future_beat:
        hints.append(f"Do not jump ahead to later beat yet: {future_beat}.")
    return " ".join(hints)


def _slugify(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return cleaned[:48] if cleaned else fallback


def _infer_emotion(beat_text: str) -> str:
    text = beat_text.lower()
    rules = (
        ("anxious", ("urgent", "worry", "tense", "panic", "race")),
        ("relieved", ("relief", "calm", "resolve", "peace")),
        ("joyful", ("happy", "smile", "celebrate", "laugh")),
        ("sad", ("cry", "loss", "grief", "sad", "hurt")),
        ("angry", ("angry", "rage", "furious", "shout")),
    )
    for emotion, hints in rules:
        if any(token in text for token in hints):
            return emotion
    return "focused"


def _extract_character_names(storyline: str, beat_text: str) -> List[str]:
    source = f"{storyline} {beat_text}"
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", source)
    deny = {
        "The",
        "This",
        "That",
        "When",
        "Then",
        "With",
        "From",
        "Into",
        "Over",
        "After",
        "Before",
        "Start",
        "Continue",
        "Resolve",
        "Show",
        "Current",
        "Previous",
        "Story",
        "Emotion",
        "Objective",
    }
    names: List[str] = []
    for token in candidates:
        token = re.sub(r"^(?:At|In|On|Back|Inside|Outside|Later|Earlier|From)\s+", "", token).strip()
        if len(token) <= 2 or token in deny or _candidate_is_location_name(token):
            continue
        if any(token == existing or token in existing or existing in token for existing in names):
            continue
        names.append(token)
        if len(names) >= 2:
            break
    if names:
        return names
    return ["Protagonist"]


def _build_scene_plan_like_payload(storyline: str, windows: List[Any], window_seconds: int) -> Dict[str, Any]:
    story_title = storyline.split(".")[0].strip()[:120] or "story"
    window_rows: List[Dict[str, Any]] = []
    previous_window_id: Optional[str] = None
    last_anchor = "main_location"

    for pos, window in enumerate(windows):
        beat_text = str(window.beat).strip()
        window_id = f"w_{window.index:03d}"
        explicit_scene_id = _window_scene_id(window)
        explicit_environment_anchor = _window_environment_anchor(window)
        anchor_text = explicit_environment_anchor or _extract_environment_anchor(beat_text)
        anchor_key = _slugify(explicit_scene_id or anchor_text, fallback=last_anchor)
        if anchor_key:
            last_anchor = anchor_key

        story_phase = _window_story_phase(window)
        character_progression = _window_character_progression(window)
        relationship_dynamic = _window_relationship_dynamic(window)
        visible_change = _window_visible_change(window)
        row = {
            "window_id": window_id,
            "window_index": int(window.index),
            "beat_id": _slugify(beat_text, fallback=f"beat_{window.index:03d}"),
            "scene_id": explicit_scene_id or anchor_key,
            "scene_objective": beat_text.rstrip("."),
            "emotion": _infer_emotion(beat_text),
            "story_phase": story_phase,
            "character_progression": character_progression,
            "relationship_dynamic": relationship_dynamic,
            "visible_change": visible_change,
            "character_names": _extract_character_names(storyline, beat_text),
            "continuity_anchor": {
                "world_anchor": anchor_key or "main_location",
                "planned_environment_anchor": anchor_text,
                "previous_window_id": previous_window_id,
            },
            "expected_caption": (
                f"{beat_text}. "
                f"Story phase: {story_phase or 'current progression'}. "
                f"Character progression: {character_progression or 'show the character changing under pressure'}. "
                f"Relationship dynamic: {relationship_dynamic or 'carry the emotional tension through behavior'}. "
                f"Visible change: {visible_change or 'show a clear shift in pose, spacing, or prop handling'}. "
                f"Progress within beat: {(pos % 3 + 1) / 3:.2f}."
            ),
            "beat_step": (pos % 3) + 1,
            "beat_total_steps": 3,
        }
        window_rows.append(row)
        previous_window_id = window_id

    return {
        "version": 1,
        "story_id": _slugify(story_title, fallback="story"),
        "title": story_title,
        "window_seconds": int(window_seconds),
        "runtime_seconds": int(len(window_rows) * window_seconds),
        "total_windows": len(window_rows),
        "windows": window_rows,
    }


def _dense_segments_by_window(dense_pack: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    by_window: Dict[str, List[Dict[str, Any]]] = {}
    for segment in dense_pack.get("segments", []):
        by_window.setdefault(segment["window_id"], []).append(segment)
    for window_id, segments in by_window.items():
        by_window[window_id] = sorted(segments, key=lambda item: int(item["start_frame"]))
    return by_window


def _caption_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _caption_similarity(expected_caption: str, actual_caption: str) -> float:
    expected_tokens = _caption_tokens(expected_caption)
    actual_tokens = _caption_tokens(actual_caption)
    if not expected_tokens and not actual_tokens:
        return 1.0
    if not expected_tokens or not actual_tokens:
        return 0.0
    overlap = expected_tokens.intersection(actual_tokens)
    union = expected_tokens.union(actual_tokens)
    return float(len(overlap) / max(1, len(union)))


def _expected_caption_for_frame(
    segments: List[Dict[str, Any]],
    frame_index: int,
    total_frames: int,
    fallback_caption: str,
) -> str:
    if not segments:
        return fallback_caption
    if total_frames <= 1:
        return str(segments[0].get("expected_caption", fallback_caption))
    seg_pos = int(round((frame_index / max(1, total_frames - 1)) * (len(segments) - 1)))
    seg_pos = max(0, min(seg_pos, len(segments) - 1))
    return str(segments[seg_pos].get("expected_caption", fallback_caption))


_SEMANTIC_STOPWORDS = {
    "the",
    "and",
    "with",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "near",
    "this",
    "that",
    "there",
    "their",
    "about",
    "while",
    "through",
    "across",
    "frame",
    "scene",
    "shows",
    "show",
    "showing",
    "video",
    "clip",
    "current",
    "next",
    "same",
    "keep",
    "maintain",
}


def _semantic_tokens(text: str) -> set[str]:
    return {
        token
        for token in _caption_tokens(text)
        if token not in _SEMANTIC_STOPWORDS and not token.isdigit() and len(token) > 2
    }


def _build_caption_alignment_report(
    *,
    captions: List[str],
    segments: List[Dict[str, Any]],
    fallback_caption: str,
    semantic_feedback_threshold: float = 0.60,
    caption_duplicates: bool = False,
) -> Dict[str, Any]:
    frame_comparison: List[Dict[str, Any]] = []
    expected_per_frame: List[str] = []
    expected_tokens: set[str] = set()
    actual_tokens: set[str] = set()
    weak_frames = 0

    for frame_idx, actual_caption in enumerate(captions):
        expected_caption = _expected_caption_for_frame(
            segments=segments,
            frame_index=frame_idx,
            total_frames=len(captions),
            fallback_caption=fallback_caption,
        )
        similarity = _caption_similarity(expected_caption, actual_caption)
        frame_comparison.append(
            {
                "frame_index": frame_idx,
                "expected_caption": expected_caption,
                "actual_caption": actual_caption,
                "similarity": round(similarity, 4),
                "loss": round(1.0 - similarity, 4),
            }
        )
        expected_per_frame.append(expected_caption)
        expected_tokens.update(_semantic_tokens(expected_caption))
        actual_tokens.update(_semantic_tokens(actual_caption))
        if similarity < semantic_feedback_threshold:
            weak_frames += 1

    avg_similarity = None
    avg_loss = None
    if frame_comparison:
        avg_similarity = sum(item["similarity"] for item in frame_comparison) / len(frame_comparison)
        avg_loss = 1.0 - avg_similarity

    missing_tokens = sorted(expected_tokens - actual_tokens)
    feedback_notes: List[str] = []
    if avg_similarity is not None and avg_similarity < semantic_feedback_threshold:
        feedback_notes.append(
            "Show the planned story beat more explicitly so the generated action matches the expected caption."
        )
    if weak_frames > max(1, len(captions) // 2):
        feedback_notes.append(
            "Keep the semantic action consistent across the whole window instead of drifting away from the planned moment."
        )
    if missing_tokens:
        feedback_notes.append(
            f"Emphasize these missing story elements: {', '.join(missing_tokens[:8])}."
        )
    if caption_duplicates:
        feedback_notes.append("Avoid duplicate extra subjects; keep only the intended characters in frame.")

    return {
        "expected_per_frame": expected_per_frame,
        "frame_comparison": frame_comparison,
        "alignment_score": round(avg_similarity, 4) if avg_similarity is not None else None,
        "alignment_loss": round(avg_loss, 4) if avg_loss is not None else None,
        "feedback": " ".join(feedback_notes).strip(),
    }


def _resolve_selected_model_id(
    selected: Optional[Dict[str, Any]],
    module_payload: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    local_only = _resolve_selected_local_model_path(selected, module_payload)
    if local_only:
        return local_only

    if not selected:
        return None

    repo_id = selected.get("repo_id")
    if isinstance(repo_id, str) and repo_id.strip():
        return repo_id

    if isinstance(module_payload, dict):
        fallbacks = module_payload.get("fallbacks")
        if isinstance(fallbacks, list):
            for fallback in fallbacks:
                if not isinstance(fallback, dict):
                    continue
                fallback_repo = fallback.get("repo_id")
                if isinstance(fallback_repo, str) and fallback_repo.strip():
                    return fallback_repo

    return None


def _resolve_selected_local_model_path(
    selected: Optional[Dict[str, Any]],
    module_payload: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if selected:
        local_path = selected.get("local_path")
        exists = selected.get("exists")
        if isinstance(exists, bool) and exists and isinstance(local_path, str) and local_path.strip():
            return local_path

    if isinstance(module_payload, dict):
        fallbacks = module_payload.get("fallbacks")
        if isinstance(fallbacks, list):
            for fallback in fallbacks:
                if not isinstance(fallback, dict):
                    continue
                fallback_local = fallback.get("local_path")
                fallback_exists = fallback.get("exists")
                if isinstance(fallback_exists, bool) and fallback_exists and isinstance(fallback_local, str) and fallback_local.strip():
                    return fallback_local

    return None


def _first_existing_path(paths: List[str]) -> Optional[str]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            return path.as_posix()
    return None


def _first_existing_diffusers_model_path(paths: List[str]) -> Optional[str]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists() and (path / "model_index.json").exists():
            return path.as_posix()
    return None


def _is_remote_model_ref(value: Optional[str]) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    candidate = value.strip()
    if candidate.startswith(("/", "./", "../")):
        return False
    return not Path(candidate).exists()


def _load_reference_images(image_paths: List[str], width: int, height: int) -> List[Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to load reference images. Install with: pip install pillow"
        ) from exc

    frames: List[Any] = []
    for image_path in image_paths:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"reference image not found: {path}")
        image = Image.open(path).convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), resample=Image.BICUBIC)
        frames.append(image)
    return frames


def _load_initial_condition_frames(image_path: str, width: int, height: int) -> List[Any]:
    return _load_reference_images([image_path], width=width, height=height)


def _load_window_reference_images(path: str) -> Dict[str, List[str]]:
    ref_path = Path(path)
    if not ref_path.is_file():
        raise FileNotFoundError(f"window_reference_images_json not found: {ref_path}")
    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("window_reference_images_json must be a JSON object keyed by window id/index")

    mapping: Dict[str, List[str]] = {}
    for raw_key, raw_value in payload.items():
        key = str(raw_key).strip()
        if not key:
            continue
        paths: List[str] = []
        if isinstance(raw_value, str):
            if raw_value.strip():
                paths = [raw_value.strip()]
        elif isinstance(raw_value, list):
            paths = [str(item).strip() for item in raw_value if str(item).strip()]
        elif isinstance(raw_value, dict):
            images = raw_value.get("images")
            image = raw_value.get("image")
            if isinstance(images, list):
                paths = [str(item).strip() for item in images if str(item).strip()]
            elif isinstance(image, str) and image.strip():
                paths = [image.strip()]
        if paths:
            mapping[key] = paths
    return mapping


def _lookup_window_reference_paths(
    mapping: Dict[str, List[str]],
    window_id: str,
    window_index: int,
) -> List[str]:
    for key in (window_id, str(window_index), f"{window_index:03d}"):
        if key in mapping:
            return mapping[key]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Storyline -> Scene Director -> Wan clip windows with local/global memory feedback."
    )
    parser.add_argument("--storyline", type=str, required=True, help="Full storyline or plot text.")
    parser.add_argument("--output_dir", type=str, default="outputs/story_run", help="Run output directory.")
    parser.add_argument(
        "--model_links",
        type=str,
        default="outputs/pipeline/model_links.json",
        help="Linked model manifest path (from scripts/00_pipeline/01_link_models.py).",
    )
    parser.add_argument(
        "--total_minutes",
        type=float,
        default=0.5,
        help="Target video length in minutes when using fixed window planning.",
    )
    parser.add_argument("--window_seconds", type=int, default=10, help="Seconds per clip window.")
    parser.add_argument(
        "--window_count_mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "fixed"],
        help="How to decide the number of windows: dynamic uses story length, fixed uses total_minutes/window_seconds.",
    )
    parser.add_argument(
        "--target_words_per_window",
        type=int,
        default=28,
        help="Story-length pacing target used only in dynamic window planning.",
    )
    parser.add_argument(
        "--min_dynamic_windows",
        type=int,
        default=1,
        help="Minimum generated window count when using dynamic window planning.",
    )
    parser.add_argument(
        "--max_dynamic_windows",
        type=int,
        default=24,
        help="Maximum generated window count when using dynamic window planning.",
    )
    parser.add_argument(
        "--window_plan_json",
        type=str,
        default="",
        help="Optional JSON file with preauthored beat list (array of strings or objects with 'beat').",
    )

    parser.add_argument("--director_model_id", type=str, default="", help="Optional HF LLM id for director.")
    parser.add_argument("--director_temperature", type=float, default=0.7, help="Director LLM temperature.")
    parser.add_argument(
        "--director_max_new_tokens",
        type=int,
        default=160,
        help="Max new tokens per director refinement call (lower is faster for Qwen).",
    )
    parser.add_argument(
        "--director_do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable stochastic sampling in director LLM generation.",
    )
    parser.add_argument(
        "--shot_plan_defaults",
        type=str,
        default="cinematic",
        choices=["cinematic", "docu", "action"],
        help="Default shot-plan preset for the scene director when planning camera coverage.",
    )
    parser.add_argument(
        "--shot_plan_enforce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to explicitly inject the director shot plan into the generation prompt.",
    )

    parser.add_argument(
        "--video_model_id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Wan model id.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no_cpu_offload", action="store_true")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--seed_strategy",
        type=str,
        default="fixed",
        choices=["fixed", "window_offset"],
        help="Seed scheduling across windows. 'fixed' keeps base seed constant for continuity.",
    )
    parser.add_argument(
        "--disable_random_generation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable stochastic candidate sampling. Uses a fixed deterministic seed path and "
            "single-candidate generation."
        ),
    )
    parser.add_argument(
        "--reference_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Condition each window on tail frames from the previous generated window.",
    )
    parser.add_argument(
        "--reference_tail_frames",
        type=int,
        default=16,  # Increased from 1 to ~2 seconds at 8fps
        help="Number of tail frames from previous window passed to the generator as conditioning reference.",
    )
    parser.add_argument(
        "--reference_strength",
        type=float,
        default=0.7,
        help="Reference-conditioning strength passed when the diffusion pipeline supports it.",
    )
    parser.add_argument(
        "--noise_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extract noise patterns from reference frames instead of using pixel values (experimental).",
    )
    parser.add_argument(
        "--noise_blend_strength",
        type=float,
        default=0.2,
        help="Blend amount used when converting reference frames into structure/noise guidance images.",
    )
    parser.add_argument(
        "--initial_condition_image",
        type=str,
        default="",
        help="Optional still image used to bootstrap the first window when reference conditioning is enabled.",
    )
    parser.add_argument(
        "--window_reference_images_json",
        type=str,
        default="",
        help="Optional JSON map of window ids/indexes to one or more reference image paths for I2V anchoring.",
    )
    parser.add_argument(
        "--style_prefix",
        type=str,
        default="cinematic realistic, coherent motion, stable camera, high detail",
        help="Global style and quality prefix prepended to each generation prompt.",
    )
    parser.add_argument(
        "--character_lock",
        type=str,
        default=(
            "keep the same main characters, wardrobe, key props, and scene scale across windows; "
            "no duplicate subjects and no unrelated new objects"
        ),
        help="Continuity constraints to keep subject identity stable across windows.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "blurry, low quality, flicker, frame jitter, deformed anatomy, duplicate subjects, extra limbs, "
            "extra animals, wrong species, text, subtitles, watermark, logo, collage, split-screen, glitch"
        ),
        help="Negative prompt passed into the video generator.",
    )

    parser.add_argument("--embedding_backend", type=str, default="clip", choices=["none", "clip", "dinov2"])
    parser.add_argument("--embedding_model_id", type=str, default="", help="Optional embedder model id.")
    parser.add_argument(
        "--embedding_adapter_ckpt",
        type=str,
        default="",
        help="Optional continuity adapter checkpoint (.pt) for the visual embedder.",
    )
    parser.add_argument(
        "--last_frame_memory",
        action="store_true",
        help="Use previous clip last-frame embedding to rank candidate next clips by first-frame continuity.",
    )
    parser.add_argument(
        "--continuity_candidates",
        type=int,
        default=2,
        help="Candidate clips per window when continuity ranking is enabled (higher is slower).",
    )
    parser.add_argument(
        "--environment_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use previous clip embedding to preserve environment across windows.",
    )
    parser.add_argument(
        "--transition_weight",
        type=float,
        default=0.65,
        help="Weight for first-frame to previous-last-frame similarity in candidate ranking.",
    )
    parser.add_argument(
        "--transition_floor",
        type=float,
        default=0.45,
        help="Minimum preferred first-frame continuity; candidates below this are deprioritized.",
    )
    parser.add_argument(
        "--environment_weight",
        type=float,
        default=0.35,
        help="Weight for whole-clip environment similarity in candidate ranking.",
    )
    parser.add_argument(
        "--scene_change_env_decay",
        type=float,
        default=0.25,
        help="Multiplier for environment_weight when beat indicates a location/setting change.",
    )
    parser.add_argument(
        "--continuity_min_score",
        type=float,
        default=0.72,
        help="Minimum critic score required to accept a window candidate.",
    )
    parser.add_argument(
        "--continuity_regen_attempts",
        type=int,
        default=4,
        help="Maximum retry iterations per window while tightening constraints until convergence.",
    )
    parser.add_argument(
        "--continuity_convergence_patience",
        type=int,
        default=2,
        help="Stop early when retry scores plateau for this many consecutive iterations.",
    )
    parser.add_argument(
        "--continuity_convergence_tolerance",
        type=float,
        default=0.015,
        help="Tolerance used to detect retry-score convergence plateaus.",
    )
    parser.add_argument(
        "--progressive_tightening",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tighten guidance, continuity pressure, and repair constraints on each retry iteration.",
    )
    parser.add_argument(
        "--tightening_strength",
        type=float,
        default=0.8,
        help="How aggressively retries strengthen guidance and continuity constraints.",
    )
    parser.add_argument(
        "--critic_story_weight",
        type=float,
        default=0.15,
        help="Weight for story progression score in critic final score.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only plan/refine prompts. No video generation.")
    parser.add_argument(
        "--window_shard_count",
        type=int,
        default=1,
        help="Total number of independent window shards. >1 enables sharded generation.",
    )
    parser.add_argument(
        "--window_shard_index",
        type=int,
        default=0,
        help="Zero-based shard index in [0, window_shard_count).",
    )
    parser.add_argument(
        "--parallel_window_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, windows are generated independently (no cross-window memory chaining). "
            "Use for multi-GPU/multi-node speed runs."
        ),
    )
    parser.add_argument(
        "--captioner_model_id",
        type=str,
        default="",
        help="Optional captioner model id (e.g., Salesforce/blip2-flan-t5-xl). Leave blank to disable.",
    )
    parser.add_argument(
        "--captioner_device",
        type=str,
        default="cpu",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for captioner; default cpu to avoid GPU contention.",
    )
    parser.add_argument(
        "--captioner_stub_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If captioner model fails to load, fall back to stub captions instead of crashing.",
    )
    parser.add_argument(
        "--semantic_alignment_weight",
        type=float,
        default=0.25,
        help="Weight for image-to-text semantic alignment when a real captioner is enabled.",
    )
    parser.add_argument(
        "--semantic_alignment_threshold",
        type=float,
        default=0.60,
        help="Minimum semantic caption alignment before semantic repair feedback is added.",
    )
    parser.add_argument(
        "--semantic_caption_sample_count",
        type=int,
        default=4,
        help="Frames sampled per candidate for semantic caption feedback; 0 captions every frame.",
    )
    parser.add_argument(
        "--visual_quality_weight",
        type=float,
        default=0.12,
        help="Weight for a lightweight visual-quality heuristic when ranking candidates to reduce recursive drift.",
    )
    parser.add_argument(
        "--style_weight",
        type=float,
        default=0.18,
        help="Weight for lightweight palette/lighting/style consistency when ranking candidates across windows.",
    )
    args = parser.parse_args()

    if args.window_shard_count < 1:
        raise ValueError("--window_shard_count must be >= 1")
    if args.window_shard_index < 0 or args.window_shard_index >= args.window_shard_count:
        raise ValueError("--window_shard_index must satisfy 0 <= index < window_shard_count")
    if args.reference_tail_frames < 1:
        raise ValueError("--reference_tail_frames must be >= 1")
    if args.reference_conditioning and args.parallel_window_mode:
        raise ValueError(
            "--reference_conditioning requires sequential generation. "
            "Disable --parallel_window_mode to use previous-window frame conditioning."
        )
    if args.initial_condition_image and not args.reference_conditioning:
        raise ValueError("--initial_condition_image requires --reference_conditioning")
    if not 0.0 <= float(args.noise_blend_strength) <= 1.0:
        raise ValueError("--noise_blend_strength must be between 0.0 and 1.0")
    if not 0.0 <= float(args.semantic_alignment_weight) <= 1.0:
        raise ValueError("--semantic_alignment_weight must be between 0.0 and 1.0")
    if not 0.0 <= float(args.semantic_alignment_threshold) <= 1.0:
        raise ValueError("--semantic_alignment_threshold must be between 0.0 and 1.0")
    if int(args.semantic_caption_sample_count) < 0:
        raise ValueError("--semantic_caption_sample_count must be >= 0")
    if not 0.0 <= float(args.style_weight) <= 1.0:
        raise ValueError("--style_weight must be between 0.0 and 1.0")
    if int(args.continuity_convergence_patience) < 0:
        raise ValueError("--continuity_convergence_patience must be >= 0")
    if not 0.0 <= float(args.continuity_convergence_tolerance) <= 1.0:
        raise ValueError("--continuity_convergence_tolerance must be between 0.0 and 1.0")
    if float(args.tightening_strength) < 0.0:
        raise ValueError("--tightening_strength must be >= 0.0")

    model_links_payload = load_model_links(Path(args.model_links))
    modules_payload = model_links_payload.get("modules", {}) if isinstance(model_links_payload, dict) else {}
    llm_payload = modules_payload.get("llm_model") if isinstance(modules_payload, dict) else None
    video_payload = modules_payload.get("video_backbone") if isinstance(modules_payload, dict) else None
    caption_gen_payload = modules_payload.get("caption_gen") if isinstance(modules_payload, dict) else None
    feedback_caption_payload = modules_payload.get("feedback_caption") if isinstance(modules_payload, dict) else None
    embedding_payload = modules_payload.get("global_local_emb_feedback") if isinstance(modules_payload, dict) else None

    selected_llm_model = get_selected_model(model_links_payload, "llm_model")
    selected_video_model = get_selected_model(model_links_payload, "video_backbone")
    selected_caption_gen_model = get_selected_model(model_links_payload, "caption_gen")
    selected_feedback_caption_model = get_selected_model(model_links_payload, "feedback_caption")
    selected_embedding_model = get_selected_model(model_links_payload, "global_local_emb_feedback")

    resolved_llm_id = _resolve_selected_model_id(selected_llm_model, llm_payload)
    resolved_video_id = _resolve_selected_model_id(selected_video_model, video_payload)
    resolved_captioner_id = _resolve_selected_model_id(selected_feedback_caption_model, feedback_caption_payload) or _resolve_selected_model_id(
        selected_caption_gen_model, caption_gen_payload
    )
    resolved_embedding_id = _resolve_selected_model_id(selected_embedding_model, embedding_payload)
    resolved_local_llm_id = _resolve_selected_local_model_path(selected_llm_model, llm_payload)
    resolved_local_video_id = _resolve_selected_local_model_path(selected_video_model, video_payload)
    resolved_local_captioner_id = _resolve_selected_local_model_path(selected_feedback_caption_model, feedback_caption_payload) or _resolve_selected_local_model_path(
        selected_caption_gen_model, caption_gen_payload
    )
    resolved_local_embedding_id = _resolve_selected_local_model_path(selected_embedding_model, embedding_payload)
    local_llm_id = _first_existing_path([
        str(PROJECT_ROOT / "LLM_MODEL" / "Qwen2.5-3B-Instruct"),
        "/home/vault/v123be/v123be37/Multimodal_Final_SceneWeaver/LLM_MODEL/Qwen2.5-3B-Instruct",
        "/home/vault/v123be/v123be37/Multimodal_Final_SceneWeaver/models/Qwen2.5-1.5B-Instruct",
    ]) or resolved_local_llm_id
    local_video_i2v_id = _first_existing_diffusers_model_path([
        "/home/vault/v123be/v123be37/sceneweaver_models/Wan2.2-I2V-A14B-Diffusers",
        "/home/vault/v123be/v123be37/sceneweaver_models/Wan2.1-I2V-14B-720P-Diffusers",
        "/home/vault/v123be/v123be37/sceneweaver_models/Wan2.1-I2V-14B-480P-Diffusers",
    ])
    local_video_t2v_id = _first_existing_path([
        "/home/vault/v123be/v123be37/Wan2.1-T2V-1.3B-Diffusers",
        "/home/vault/v123be/v123be37/Multimodal_Final_SceneWeaver/models/VIDEO_GENERATIVE_BACKBONE/Wan2.1-T2V-1.3B-Diffusers",
        str(PROJECT_ROOT / "models" / "VIDEO_GENERATIVE_BACKBONE" / "Wan2.1-T2V-1.3B-Diffusers"),
        str(PROJECT_ROOT / "VIDEO_GENERATIVE_BACKBONE" / "Wan2.1-T2V-1.3B-Diffusers"),
    ]) or resolved_local_video_id
    local_embedding_dinov2_id = _first_existing_path([
        "/home/vault/v123be/v123be37/facebook/dinov2-base",
        "/home/vault/v123be/v123be37/facebook/dinov2-small",
        str(PROJECT_ROOT / "Globa_Local_Emb_Feedback" / "dinov2-base"),
        str(PROJECT_ROOT / "Globa_Local_Emb_Feedback" / "dinov2-small"),
    ]) or resolved_local_embedding_id
    captioner_disabled = str(args.captioner_model_id).strip().lower() in {"none", "off", "disabled"}
    if captioner_disabled:
        args.captioner_model_id = ""

    if not args.director_model_id:
        args.director_model_id = local_llm_id or ""
    if args.video_model_id == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers":
        args.video_model_id = local_video_i2v_id if args.reference_conditioning else local_video_t2v_id or args.video_model_id
    if not captioner_disabled and not args.captioner_model_id:
        args.captioner_model_id = resolved_local_captioner_id or ""
    if args.embedding_backend == "dinov2" and not args.embedding_model_id:
        args.embedding_model_id = local_embedding_dinov2_id or ""
    elif args.embedding_backend == "clip" and not args.embedding_model_id:
        print("[local-only] no local CLIP embedding model configured; disabling embedding backend.")
        args.embedding_backend = "none"

    if _is_remote_model_ref(args.director_model_id):
        print("[local-only] remote director model reference suppressed; using heuristic director instead.")
        args.director_model_id = ""
    if _is_remote_model_ref(args.captioner_model_id):
        print("[local-only] remote captioner reference suppressed; captioner disabled.")
        args.captioner_model_id = ""
    if _is_remote_model_ref(args.embedding_model_id):
        print("[local-only] remote embedding model reference suppressed; embedding backend disabled.")
        args.embedding_model_id = ""
        args.embedding_backend = "none"
    if _is_remote_model_ref(args.video_model_id):
        raise RuntimeError(
            "Local-only mode is active, but the video model still resolved to a Hugging Face repo id. "
            "Set --video_model_id to a local directory or update outputs/pipeline/model_links.json to the correct local path."
        )

    out_dir = Path(args.output_dir)
    clips_dir = out_dir / "clips"
    ensure_dir(out_dir)
    ensure_dir(clips_dir)
    initial_condition_frames: Optional[List[Any]] = None
    if args.initial_condition_image:
        initial_condition_frames = _load_initial_condition_frames(
            args.initial_condition_image,
            width=args.width,
            height=args.height,
        )
    window_reference_images: Dict[str, List[str]] = {}
    if args.window_reference_images_json:
        window_reference_images = _load_window_reference_images(args.window_reference_images_json)

    director = SceneDirector(
        SceneDirectorConfig(
            model_id=args.director_model_id or None,
            temperature=args.director_temperature,
            max_new_tokens=args.director_max_new_tokens,
            do_sample=bool(args.director_do_sample),
            shot_plan_defaults=args.shot_plan_defaults,
            window_count_mode=args.window_count_mode,
            target_words_per_window=args.target_words_per_window,
            min_dynamic_windows=args.min_dynamic_windows,
            max_dynamic_windows=args.max_dynamic_windows,
        ),
        window_seconds=args.window_seconds,
    )
    director.load()
    beats_override = _load_window_plan(args.window_plan_json) if args.window_plan_json else None
    windows = director.plan_windows(
        storyline=args.storyline,
        total_minutes=args.total_minutes,
        beats_override=beats_override,
    )
    planned_runtime_seconds = len(windows) * args.window_seconds
    window_count_source = "window_plan_json" if beats_override else args.window_count_mode
    print(
        f"[window-plan] source={window_count_source} windows={len(windows)} "
        f"planned_runtime_seconds={planned_runtime_seconds} window_seconds={args.window_seconds}"
    )

    backbone = None
    if not args.dry_run:
        WanBackbone, WanBackboneConfig = load_video_backbone()
        backbone = WanBackbone(
            WanBackboneConfig(
                model_id=args.video_model_id,
                torch_dtype=args.dtype,
                device=args.device,
                enable_cpu_offload=not args.no_cpu_offload,
            )
        )
        backbone.load()
        if args.reference_conditioning and not backbone.supports_reference_conditioning():
            raise RuntimeError(
                "reference conditioning is enabled, but selected video model/pipeline does not expose "
                "a supported conditioning input (expected one of: conditioning_frames, video, frames, image, init_image). "
                f"video_model_id={args.video_model_id}"
            )
        first_window_id = f"w_{windows[0].index:03d}" if windows else "w_000"
        first_window_reference_paths = _lookup_window_reference_paths(window_reference_images, first_window_id, 0)
        if args.reference_conditioning and backbone.requires_reference_conditioning() and initial_condition_frames is None and not first_window_reference_paths:
            raise RuntimeError(
                "Selected video model requires conditioning input for the first window. "
                "Set --initial_condition_image or --window_reference_images_json, or disable --reference_conditioning. "
                f"video_model_id={args.video_model_id}"
            )

    embedder = None
    memory = None
    captioner = None
    if not args.dry_run and args.embedding_backend != "none":
        embedder = maybe_init_embedder(
            backend=args.embedding_backend,
            model_id=args.embedding_model_id or None,
            adapter_ckpt=args.embedding_adapter_ckpt or None,
            device=args.device,
        )
        memory = NarrativeMemory()
    if not args.dry_run and args.captioner_model_id:
        captioner = Captioner(
            CaptionerConfig(
                model_id=args.captioner_model_id,
                device=args.captioner_device,
                stub_fallback=bool(args.captioner_stub_fallback),
            )
        )
        captioner.load()

    previous_prompt = ""
    previous_scene_conversation = ""
    memory_feedback = None
    previous_last_frame_embedding = None
    previous_clip_embedding = None
    previous_style_signature = None
    previous_environment_anchor = ""
    previous_reference_frames: Optional[List[Any]] = None
    next_negative_prompt = args.negative_prompt
    log_rows: List[Dict[str, Any]] = []

    selected_windows = [
        (window_pos, window)
        for window_pos, window in enumerate(windows)
        if (window.index % args.window_shard_count) == args.window_shard_index
    ]

    caption_dir = out_dir / "caption"
    ensure_dir(caption_dir)
    scene_plan_payload = _build_scene_plan_like_payload(
        storyline=args.storyline,
        windows=windows,
        window_seconds=args.window_seconds,
    )
    expected_caption_pack = build_expected_caption_pack(scene_plan_payload, style="compact")
    dense_caption_pack = build_dense_expected_caption_pack(
        scene_plan_payload,
        fps=max(1, int(args.fps)),
        segment_length_frames=max(1, args.num_frames // 4),
        overlap_frames=0,
    )
    runtime_models: Dict[str, Dict[str, Any]] = {}
    if selected_caption_gen_model is not None:
        runtime_models["caption_gen"] = selected_caption_gen_model
    if selected_feedback_caption_model is not None:
        runtime_models["feedback_caption"] = selected_feedback_caption_model
    if runtime_models:
        expected_caption_pack["runtime_models"] = runtime_models
        dense_caption_pack["runtime_models"] = runtime_models
    expected_caption_path = caption_dir / "expected_video_captions.json"
    dense_caption_path = caption_dir / "dense_expected_video_captions.json"
    expected_caption_path.write_text(
        json.dumps(expected_caption_pack, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dense_caption_path.write_text(
        json.dumps(dense_caption_pack, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    expected_by_window = {item["window_id"]: item for item in expected_caption_pack.get("captions", [])}
    dense_by_window = _dense_segments_by_window(dense_caption_pack)

    print(
        f"[shard] index={args.window_shard_index}/{args.window_shard_count} "
        f"selected_windows={len(selected_windows)} total_windows={len(windows)} "
        f"parallel_window_mode={args.parallel_window_mode}"
    )
    print(
        "[generation] "
        f"reference_conditioning={args.reference_conditioning} "
        f"reference_tail_frames={args.reference_tail_frames} "
        f"noise_conditioning={args.noise_conditioning} "
        f"noise_blend_strength={args.noise_blend_strength} "
        f"visual_quality_weight={args.visual_quality_weight} "
        f"style_weight={args.style_weight} "
        f"disable_random_generation={args.disable_random_generation}"
    )

    for window_pos, window in selected_windows:
        if args.parallel_window_mode:
            previous_prompt = ""
            previous_scene_conversation = ""
            memory_feedback = None
            previous_last_frame_embedding = None
            previous_clip_embedding = None
            previous_style_signature = None
            previous_environment_anchor = ""
            previous_reference_frames = None
            next_negative_prompt = args.negative_prompt
        previous_window = windows[window_pos - 1] if (window_pos > 0 and not args.parallel_window_mode) else None
        next_window = windows[window_pos + 1] if (window_pos + 1 < len(windows) and not args.parallel_window_mode) else None
        scene_change_requested = _scene_change_requested(previous_window, window)
        same_scene_as_previous = _same_scene_as_previous(previous_window, window)
        story_state_hint = _build_story_state_hint(windows, window_pos)
        previous_beat = previous_window.beat if previous_window is not None else ""
        next_beat = next_window.beat if next_window is not None else ""
        story_progress_instruction = _story_progress_instruction(previous_beat, window.beat)
        window_scene_id = _window_scene_id(window)
        planned_environment_anchor = _window_environment_anchor(window)
        current_environment_anchor = planned_environment_anchor or previous_environment_anchor
        effective_character_lock = _merge_character_lock(args.character_lock, _window_character_lock(window))
        character_names = _extract_character_names(args.storyline, window.beat)
        window_id = f"w_{window.index:03d}"
        explicit_window_reference_paths = _lookup_window_reference_paths(window_reference_images, window_id, window.index)
        reference_frames_for_window: Optional[List[Any]] = None
        reference_source = "none"
        if args.reference_conditioning and explicit_window_reference_paths:
            reference_frames_for_window = _load_reference_images(
                explicit_window_reference_paths,
                width=args.width,
                height=args.height,
            )
            reference_source = "window_reference_image"
        elif args.reference_conditioning and previous_reference_frames:
            reference_frames_for_window = previous_reference_frames
            reference_source = "previous_window_tail"
        elif args.reference_conditioning and initial_condition_frames is not None:
            reference_frames_for_window = initial_condition_frames
            reference_source = "initial_condition_image"
        bundle: PromptBundle = director.refine_prompt(
            storyline=args.storyline,
            window=window,
            previous_prompt=previous_prompt,
            previous_scene_conversation=previous_scene_conversation,
            memory_feedback=memory_feedback.to_dict() if memory_feedback else None,
        )
        refined_prompt = bundle.prompt_text
        current_shot_plan = bundle.shot_plan
        scene_conversation = bundle.scene_conversation
        dialogue_scene = _should_use_dialogue_staging(
            scene_conversation=scene_conversation,
            beat=window.beat,
            character_names=character_names,
        )
        conversation_progress_instruction = (
            _conversation_progress_instruction(
                previous_scene_conversation,
                window.beat,
                next_beat,
                scene_change_requested=scene_change_requested,
            )
            if dialogue_scene
            else ""
        )
        generation_prompt = build_generation_prompt(
            refined_prompt=refined_prompt,
            beat=window.beat,
            style_prefix=args.style_prefix,
            character_lock=effective_character_lock,
            previous_environment_anchor=previous_environment_anchor,
            current_environment_anchor=current_environment_anchor,
            scene_change_requested=scene_change_requested,
            story_state_hint=story_state_hint,
            scene_conversation=scene_conversation,
            previous_scene_conversation=previous_scene_conversation,
            conversation_progress_instruction=conversation_progress_instruction,
            story_progress_instruction=story_progress_instruction,
            dialogue_scene=dialogue_scene,
            repair_hint="",
            shot_plan=current_shot_plan,
            shot_plan_enforce=args.shot_plan_enforce,
        )

        clip_path = clips_dir / f"window_{window.index:03d}.mp4"
        expected_entry = expected_by_window.get(window_id, {})
        dense_segments = dense_by_window.get(window_id, [])
        fallback_expected = str(
            expected_entry.get("expected_caption_compact")
            or expected_entry.get("expected_caption")
            or window.beat
        )
        conditioning_mode = "none"
        if reference_frames_for_window:
            if reference_source == "previous_window_tail" and len(reference_frames_for_window) >= 2:
                conditioning_mode = "tail_anchor_frame"
            else:
                conditioning_mode = "reference"

        row: Dict[str, Any] = {
            "window_id": window_id,
            "window_index": window.index,
            "time_range": [window.start_sec, window.end_sec],
            "beat": window.beat,
            "story_phase": _window_story_phase(window),
            "character_progression": _window_character_progression(window),
            "relationship_dynamic": _window_relationship_dynamic(window),
            "visible_change": _window_visible_change(window),
            "scene_id": window_scene_id or None,
            "prompt_seed": window.prompt_seed,
            "refined_prompt": refined_prompt,
            "shot_plan": current_shot_plan.__dict__,
            "scene_conversation": scene_conversation,
            "previous_scene_conversation": previous_scene_conversation,
            "conversation_progress_instruction": conversation_progress_instruction,
            "generation_prompt": generation_prompt,
            "scene_change_requested": scene_change_requested,
            "same_scene_as_previous": same_scene_as_previous,
            "planned_environment_anchor": planned_environment_anchor,
            "current_environment_anchor": current_environment_anchor,
            "character_lock": effective_character_lock,
            "negative_prompt": next_negative_prompt,
            "clip_path": clip_path.as_posix(),
            "expected_caption_compact": expected_entry.get("expected_caption_compact", ""),
            "expected_caption": expected_entry.get("expected_caption", ""),
            "generated": False,
            "memory_feedback": None,
            "reference_conditioning": bool(args.reference_conditioning),
            "reference_frames_used": len(reference_frames_for_window or []),
            "reference_source": reference_source,
            "conditioning_mode": conditioning_mode,
            "noise_conditioning": bool(args.noise_conditioning),
            "noise_blend_strength": float(args.noise_blend_strength),
            "visual_quality_weight": float(args.visual_quality_weight),
            "style_weight": float(args.style_weight),
            "reference_anchor_index": None,
            "semantic_alignment_weight": float(args.semantic_alignment_weight),
            "semantic_alignment_threshold": float(args.semantic_alignment_threshold),
            "semantic_caption_sample_count": int(args.semantic_caption_sample_count),
            "semantic_alignment_active": False,
            "window_reference_images": explicit_window_reference_paths,
            "initial_condition_image": args.initial_condition_image or None,
            "disable_random_generation": bool(args.disable_random_generation),
        }

        if not args.dry_run:
            if args.disable_random_generation:
                base_seed = args.seed if args.seed is not None else 0
            elif args.seed is None:
                base_seed = None
            elif args.seed_strategy == "window_offset":
                base_seed = args.seed + window.index
            else:
                base_seed = args.seed
            transition_ref_available = (
                embedder is not None and args.last_frame_memory and previous_last_frame_embedding is not None
            )
            environment_ref_available = (
                embedder is not None and args.environment_memory and previous_clip_embedding is not None
            )
            continuity_active = (
                embedder is not None
                and args.continuity_candidates > 1
                and (transition_ref_available or environment_ref_available)
            )
            num_candidates = args.continuity_candidates if continuity_active else 1
            max_attempts = max(1, int(args.continuity_regen_attempts))
            if args.disable_random_generation:
                num_candidates = 1
                max_attempts = 1
            candidate_rows: List[Dict[str, Any]] = []
            best_overall: Optional[Dict[str, Any]] = None
            selected: Optional[Dict[str, Any]] = None

            transition_weight = max(0.0, float(args.transition_weight))
            transition_floor = max(-1.0, min(1.0, float(args.transition_floor)))
            environment_weight = max(0.0, float(args.environment_weight))
            if scene_change_requested:
                scene_change_decay = max(0.0, float(args.scene_change_env_decay))
                environment_weight *= scene_change_decay
            semantic_weight = max(0.0, min(1.0, float(args.semantic_alignment_weight)))
            style_weight = max(0.0, min(0.35, float(args.style_weight)))
            semantic_alignment_active = captioner is not None and not captioner.is_stub and semantic_weight > 0.0
            reference_style_signature = _estimate_clip_style_signature(reference_frames_for_window) if reference_frames_for_window else None
            style_reference_signature = previous_style_signature
            if reference_source != "previous_window_tail" and reference_style_signature is not None:
                style_reference_signature = reference_style_signature
            elif style_reference_signature is None:
                style_reference_signature = reference_style_signature
            row["semantic_alignment_active"] = bool(semantic_alignment_active)
            repair_hint = ""
            attempt_score_history: List[float] = []
            convergence_reason = "max_iterations"
            planned_shot_plan = current_shot_plan
            for attempt_idx in range(max_attempts):
                base_reference_strength = _reference_strength_for_window(
                    base_strength=float(args.reference_strength),
                    scene_change_requested=scene_change_requested,
                    reference_source=reference_source,
                    same_scene_as_previous=same_scene_as_previous,
                )
                attempt_tightening = _tightened_retry_settings(
                    base_guidance_scale=float(args.guidance_scale),
                    base_negative_prompt=next_negative_prompt,
                    base_reference_strength=base_reference_strength,
                    base_noise_blend_strength=float(args.noise_blend_strength),
                    attempt_idx=attempt_idx,
                    max_attempts=max_attempts,
                    progressive_tightening=bool(args.progressive_tightening),
                    tightening_strength=float(args.tightening_strength),
                )
                attempt_repair_hint = " ".join(
                    part
                    for part in [repair_hint, str(attempt_tightening.get("repair_constraint", "")).strip()]
                    if part
                ).strip()
                generation_prompt = build_generation_prompt(
                    refined_prompt=refined_prompt,
                    beat=window.beat,
                    style_prefix=args.style_prefix,
                    character_lock=effective_character_lock,
                    previous_environment_anchor=previous_environment_anchor,
                    current_environment_anchor=current_environment_anchor,
                    scene_change_requested=scene_change_requested,
                    story_state_hint=story_state_hint,
                    scene_conversation=scene_conversation,
                    previous_scene_conversation=previous_scene_conversation,
                    conversation_progress_instruction=conversation_progress_instruction,
                    story_progress_instruction=story_progress_instruction,
                    dialogue_scene=dialogue_scene,
                    repair_hint=attempt_repair_hint,
                    shot_plan=current_shot_plan,
                    shot_plan_enforce=args.shot_plan_enforce,
                )

                best_attempt: Optional[Dict[str, Any]] = None
                for candidate_idx in range(num_candidates):
                    candidate_seed = base_seed if args.disable_random_generation else None
                    if not args.disable_random_generation and base_seed is not None:
                        candidate_seed = base_seed + candidate_idx + (attempt_idx * max(32, num_candidates))
                    candidate_frames = backbone.generate_clip(
                        prompt=generation_prompt,
                        negative_prompt=str(attempt_tightening["negative_prompt"]),
                        num_frames=args.num_frames,
                        num_inference_steps=args.steps,
                        guidance_scale=float(attempt_tightening["guidance_scale"]),
                        height=args.height,
                        width=args.width,
                        seed=candidate_seed,
                        reference_frames=reference_frames_for_window,
                        reference_strength=float(attempt_tightening["reference_strength"]),
                        reference_source=reference_source,
                        disable_random_generation=bool(args.disable_random_generation),
                        use_noise_conditioning=bool(args.noise_conditioning),
                        noise_blend_strength=float(attempt_tightening["noise_blend_strength"]),
                    )
                    row["conditioning_mode"] = getattr(backbone, "last_conditioning_mode", row["conditioning_mode"])
                    row["reference_anchor_index"] = getattr(backbone, "last_reference_anchor_index", row["reference_anchor_index"])

                    transition_similarity = None
                    if embedder is not None and args.last_frame_memory and previous_last_frame_embedding is not None:
                        candidate_first_embedding = embedder.embed_first_frame(candidate_frames)
                        transition_similarity = _cosine_similarity(
                            candidate_first_embedding,
                            previous_last_frame_embedding,
                        )

                    environment_similarity = None
                    candidate_embedding = None
                    if embedder is not None and args.environment_memory and previous_clip_embedding is not None:
                        candidate_embedding = embedder.embed_frames(candidate_frames)
                        environment_similarity = _cosine_similarity(candidate_embedding, previous_clip_embedding)

                    candidate_style_signature = _estimate_clip_style_signature(candidate_frames)
                    style_similarity = _style_signature_similarity(style_reference_signature, candidate_style_signature)

                    continuity_score = _combine_continuity_score(
                        transition_similarity=transition_similarity,
                        environment_similarity=environment_similarity,
                        style_similarity=style_similarity,
                        transition_weight=transition_weight,
                        environment_weight=environment_weight,
                        style_weight=style_weight,
                    )
                    critic = evaluate_candidate(
                        current_beat=window.beat,
                        previous_beat=previous_beat,
                        transition_similarity=transition_similarity,
                        environment_similarity=environment_similarity,
                        continuity_score=continuity_score,
                        story_weight=float(args.critic_story_weight),
                        continuity_weight=max(0.0, 1.0 - float(args.critic_story_weight)),
                        attempt_index=attempt_idx,
                    )

                    visual_quality_score = _estimate_clip_visual_quality(candidate_frames)
                    visual_quality_feedback = _visual_quality_feedback(visual_quality_score)
                    style_feedback = _style_similarity_feedback(style_similarity)

                    semantic_alignment_score = None
                    semantic_feedback = ""
                    if semantic_alignment_active:
                        candidate_captions, _candidate_caption_summary, candidate_caption_dupes = captioner.caption_frames(
                            candidate_frames,
                            sample_count=int(args.semantic_caption_sample_count),
                        )
                        semantic_report = _build_caption_alignment_report(
                            captions=candidate_captions,
                            segments=dense_segments,
                            fallback_caption=fallback_expected,
                            semantic_feedback_threshold=float(args.semantic_alignment_threshold),
                            caption_duplicates=bool(candidate_caption_dupes),
                        )
                        semantic_alignment_score = semantic_report["alignment_score"]
                        semantic_feedback = semantic_report["feedback"]
                    else:
                        semantic_report = None

                    quality_weight = max(0.0, min(0.35, float(args.visual_quality_weight)))
                    semantic_weight_used = semantic_weight if semantic_alignment_score is not None else 0.0
                    critic_weight = max(0.0, 1.0 - semantic_weight_used - quality_weight)
                    total_selection_weight = critic_weight + semantic_weight_used + quality_weight
                    if total_selection_weight <= 0.0:
                        total_selection_weight = 1.0
                    selection_score = (
                        (critic.final_score * critic_weight)
                        + ((float(semantic_alignment_score) if semantic_alignment_score is not None else 0.0) * semantic_weight_used)
                        + ((float(visual_quality_score) if visual_quality_score is not None else 0.0) * quality_weight)
                    ) / total_selection_weight

                    candidate_entry = {
                        "attempt_index": attempt_idx,
                        "candidate_index": candidate_idx,
                        "seed": candidate_seed,
                        "attempt_progress": attempt_tightening["attempt_progress"],
                        "guidance_scale": attempt_tightening["guidance_scale"],
                        "reference_strength": attempt_tightening["reference_strength"],
                        "noise_blend_strength": attempt_tightening["noise_blend_strength"],
                        "negative_prompt": attempt_tightening["negative_prompt"],
                        "repair_constraint": attempt_tightening["repair_constraint"],
                        "transition_similarity": transition_similarity,
                        "environment_similarity": environment_similarity,
                        "style_similarity": style_similarity,
                        "continuity_score": continuity_score,
                        "critic_score": critic.final_score,
                        "critic_story_progress_score": critic.story_progress_score,
                        "critic_feedback": critic.feedback,
                        "visual_quality_score": visual_quality_score,
                        "style_feedback": style_feedback,
                        "visual_quality_feedback": visual_quality_feedback,
                        "semantic_alignment_score": semantic_alignment_score,
                        "semantic_feedback": semantic_feedback,
                        "selection_score": selection_score,
                        "shot_plan": current_shot_plan.__dict__,
                    }
                    candidate_rows.append(candidate_entry)

                    candidate_state = {
                        "frames": candidate_frames,
                        "seed": candidate_seed,
                        "attempt_progress": attempt_tightening["attempt_progress"],
                        "guidance_scale": attempt_tightening["guidance_scale"],
                        "reference_strength": attempt_tightening["reference_strength"],
                        "noise_blend_strength": attempt_tightening["noise_blend_strength"],
                        "negative_prompt": attempt_tightening["negative_prompt"],
                        "repair_constraint": attempt_tightening["repair_constraint"],
                        "transition_similarity": transition_similarity,
                        "environment_similarity": environment_similarity,
                        "style_similarity": style_similarity,
                        "continuity_score": continuity_score,
                        "clip_embedding": candidate_embedding,
                        "style_signature": candidate_style_signature,
                        "critic_score": critic.final_score,
                        "critic_feedback": critic.feedback,
                        "visual_quality_score": visual_quality_score,
                        "visual_quality_feedback": visual_quality_feedback,
                        "style_feedback": style_feedback,
                        "semantic_alignment_score": semantic_alignment_score,
                        "semantic_feedback": semantic_feedback,
                        "selection_score": selection_score,
                        "semantic_report": semantic_report,
                        "generation_prompt": generation_prompt,
                        "attempt_index": attempt_idx,
                        "candidate_index": candidate_idx,
                    }
                    if best_attempt is None or candidate_state["selection_score"] > best_attempt["selection_score"]:
                        best_attempt = candidate_state

                if best_attempt is None:
                    continue
                if best_overall is None or best_attempt["selection_score"] > best_overall["selection_score"]:
                    best_overall = best_attempt
                attempt_score_history.append(float(best_attempt["selection_score"]))
                converged, attempt_reason = _retry_convergence_status(
                    attempt_score_history,
                    threshold=float(args.continuity_min_score),
                    patience=int(args.continuity_convergence_patience),
                    tolerance=float(args.continuity_convergence_tolerance),
                )
                if converged:
                    if attempt_reason == "threshold_met":
                        selected = best_attempt
                    convergence_reason = attempt_reason or convergence_reason
                    break
                repair_notes = [
                    best_attempt["critic_feedback"],
                    best_attempt.get("semantic_feedback", ""),
                    best_attempt.get("visual_quality_feedback", ""),
                    best_attempt.get("style_feedback", ""),
                ]
                repair_hint = " ".join(note for note in repair_notes if note).strip()
                combined_constraints = repair_hint
                if memory_feedback and memory_feedback.suggested_constraints:
                    combined_constraints = f"{combined_constraints} {memory_feedback.suggested_constraints}".strip()
                bundle = director.refine_prompt(
                    storyline=args.storyline,
                    window=window,
                    previous_prompt=previous_prompt,
                    previous_scene_conversation=previous_scene_conversation,
                    memory_feedback={
                        "suggested_constraints": combined_constraints,
                    },
                )
                refined_prompt = bundle.prompt_text
                current_shot_plan = bundle.shot_plan
                scene_conversation = bundle.scene_conversation
                if planned_shot_plan and (
                    planned_shot_plan.shot_type != current_shot_plan.shot_type
                    or planned_shot_plan.camera_angle != current_shot_plan.camera_angle
                ):
                    repair_hint = f"{repair_hint} Keep shot type '{planned_shot_plan.shot_type}' and camera angle '{planned_shot_plan.camera_angle}' consistent.".strip()

            selected = selected or best_overall
            if selected is None:
                raise RuntimeError(f"No candidate generated for window {window.index}")

            frames = selected["frames"]
            window_seed = selected["seed"]
            backbone.save_video(frames=frames, output_path=clip_path.as_posix(), fps=args.fps)
            row["generated"] = True
            row["seed"] = window_seed
            row["continuity_candidates"] = num_candidates
            row["selected_transition_similarity"] = selected["transition_similarity"]
            row["selected_environment_similarity"] = selected["environment_similarity"]
            row["selected_style_similarity"] = selected.get("style_similarity")
            row["selected_continuity_score"] = selected["continuity_score"]
            row["selected_critic_score"] = selected["critic_score"]
            row["selected_critic_feedback"] = selected["critic_feedback"]
            row["selected_visual_quality_score"] = selected.get("visual_quality_score")
            row["selected_visual_quality_feedback"] = selected.get("visual_quality_feedback", "")
            row["selected_style_feedback"] = selected.get("style_feedback", "")
            row["selected_semantic_alignment_score"] = selected.get("semantic_alignment_score")
            row["selected_semantic_feedback"] = selected.get("semantic_feedback", "")
            row["selected_selection_score"] = selected.get("selection_score", selected["critic_score"])
            row["generation_prompt"] = selected["generation_prompt"]
            row["selected_attempt_index"] = selected["attempt_index"]
            row["selected_attempt_progress"] = selected.get("attempt_progress")
            row["selected_guidance_scale"] = selected.get("guidance_scale")
            row["selected_reference_strength"] = selected.get("reference_strength")
            row["selected_noise_blend_strength"] = selected.get("noise_blend_strength")
            row["selected_negative_prompt"] = selected.get("negative_prompt", next_negative_prompt)
            row["selected_repair_constraint"] = selected.get("repair_constraint", "")
            row["reference_strength_used"] = selected.get("reference_strength", _reference_strength_for_window(
                base_strength=float(args.reference_strength),
                scene_change_requested=scene_change_requested,
                reference_source=reference_source,
                same_scene_as_previous=same_scene_as_previous,
            ))
            row["attempt_scores_history"] = attempt_score_history
            row["converged"] = convergence_reason in {"threshold_met", "score_plateau"}
            row["convergence_reason"] = convergence_reason
            row["progressive_tightening"] = bool(args.progressive_tightening)
            row["tightening_strength"] = float(args.tightening_strength)
            row["continuity_min_score"] = args.continuity_min_score
            row["continuity_regen_attempts"] = max_attempts
            row["continuity_convergence_patience"] = int(args.continuity_convergence_patience)
            row["continuity_convergence_tolerance"] = float(args.continuity_convergence_tolerance)
            row["shot_plan"] = current_shot_plan.__dict__
            row["scene_conversation"] = scene_conversation
            row["previous_scene_conversation"] = previous_scene_conversation
            row["conversation_progress_instruction"] = conversation_progress_instruction
            if len(candidate_rows) > 1:
                row["candidate_scores"] = candidate_rows

            if captioner is not None:
                captions, caption_summary, caption_dupes = captioner.caption_frames(frames, sample_count=0)
                row["captions"] = captions
                row["caption_summary"] = caption_summary
                row["caption_duplicates"] = caption_dupes
                frame_comparison: List[Dict[str, Any]] = []
                expected_per_frame: List[str] = []
                for frame_idx, actual_caption in enumerate(captions):
                    expected_caption = _expected_caption_for_frame(
                        segments=dense_segments,
                        frame_index=frame_idx,
                        total_frames=len(captions),
                        fallback_caption=fallback_expected,
                    )
                    similarity = _caption_similarity(expected_caption, actual_caption)
                    frame_comparison.append(
                        {
                            "frame_index": frame_idx,
                            "expected_caption": expected_caption,
                            "actual_caption": actual_caption,
                            "similarity": round(similarity, 4),
                            "loss": round(1.0 - similarity, 4),
                        }
                    )
                    expected_per_frame.append(expected_caption)
                row["expected_captions_per_frame"] = expected_per_frame
                row["caption_frame_comparison"] = frame_comparison
                if frame_comparison:
                    avg_similarity = sum(item["similarity"] for item in frame_comparison) / len(frame_comparison)
                    row["caption_alignment_score"] = round(avg_similarity, 4)
                    row["caption_alignment_loss"] = round(1.0 - avg_similarity, 4)
                else:
                    row["caption_alignment_score"] = None
                    row["caption_alignment_loss"] = None
                row["caption_alignment_metric"] = "token_jaccard_per_frame"

            if embedder is not None and memory is not None:
                best_clip_embedding = selected["clip_embedding"]
                embedding = best_clip_embedding if best_clip_embedding is not None else embedder.embed_frames(frames)
                memory_feedback = memory.register_window(
                    window.index,
                    embedding,
                    transition_similarity=selected["transition_similarity"],
                )
                memory_feedback.suggested_constraints = (
                    f"{memory_feedback.suggested_constraints} {selected['critic_feedback']}".strip()
                )
                row["memory_feedback"] = memory_feedback.to_dict()
                if args.last_frame_memory:
                    previous_last_frame_embedding = embedder.embed_last_frame(frames)
                if args.environment_memory:
                    previous_clip_embedding = embedding
            previous_style_signature = selected.get("style_signature")
            if previous_style_signature is None:
                previous_style_signature = _estimate_clip_style_signature(frames)
            if args.reference_conditioning:
                tail_count = max(1, int(args.reference_tail_frames))
                previous_reference_frames = list(frames[-tail_count:])

        if not args.parallel_window_mode:
            previous_prompt = _compact_previous_prompt(refined_prompt)
            previous_scene_conversation = scene_conversation if dialogue_scene else ""
            next_environment_anchor = planned_environment_anchor
            if not next_environment_anchor and captioner is not None and not captioner.is_stub and row.get("caption_summary"):
                next_environment_anchor = row["caption_summary"]
            if not next_environment_anchor:
                next_environment_anchor = _extract_environment_anchor(refined_prompt)
            if next_environment_anchor:
                previous_environment_anchor = next_environment_anchor
            if row.get("caption_duplicates"):
                next_negative_prompt = (
                    f"{args.negative_prompt}, two crows, duplicate bird, multiple birds, another crow"
                )
            else:
                next_negative_prompt = args.negative_prompt
        log_rows.append(row)
        print(f"[scene {window.index:03d}] {window.start_sec}-{window.end_sec}s ready")

    if args.window_shard_count > 1:
        shard_tag = f"shard_{args.window_shard_index:03d}_of_{args.window_shard_count:03d}"
        log_path = out_dir / f"run_log.{shard_tag}.jsonl"
        summary_path = out_dir / f"run_summary.{shard_tag}.json"
        scene_conversation_path = out_dir / f"scene_conversation_plan.{shard_tag}.json"
    else:
        log_path = out_dir / "run_log.jsonl"
        summary_path = out_dir / "run_summary.json"
        scene_conversation_path = out_dir / "scene_conversation_plan.json"

    export_jsonl(log_rows, log_path)
    scene_conversation_rows = [
        {
            "window_id": row.get("window_id"),
            "window_index": row.get("window_index"),
            "time_range": row.get("time_range"),
            "beat": row.get("beat"),
            "scene_id": row.get("scene_id"),
            "planned_environment_anchor": row.get("planned_environment_anchor", ""),
            "current_environment_anchor": row.get("current_environment_anchor", ""),
            "previous_scene_conversation": row.get("previous_scene_conversation", ""),
            "scene_conversation": row.get("scene_conversation", ""),
            "conversation_progress_instruction": row.get("conversation_progress_instruction", ""),
            "shot_plan": row.get("shot_plan", {}),
        }
        for row in log_rows
    ]
    scene_conversation_path.write_text(
        json.dumps(scene_conversation_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    caption_losses = [
        float(row["caption_alignment_loss"])
        for row in log_rows
        if row.get("caption_alignment_loss") is not None
    ]
    caption_scores = [
        float(row["caption_alignment_score"])
        for row in log_rows
        if row.get("caption_alignment_score") is not None
    ]
    summary = {
        "storyline": args.storyline,
        "total_minutes": args.total_minutes,
        "window_seconds": args.window_seconds,
        "planned_runtime_seconds": len(windows) * args.window_seconds,
        "num_windows": len(windows),
        "window_count_mode": "window_plan_json" if beats_override else args.window_count_mode,
        "target_words_per_window": args.target_words_per_window,
        "min_dynamic_windows": args.min_dynamic_windows,
        "max_dynamic_windows": args.max_dynamic_windows,
        "dry_run": args.dry_run,
        "director_model_id": args.director_model_id or None,
        "director_max_new_tokens": args.director_max_new_tokens,
        "director_do_sample": bool(args.director_do_sample),
        "video_model_id": None if args.dry_run else args.video_model_id,
        "embedding_backend": args.embedding_backend,
        "embedding_adapter_ckpt": args.embedding_adapter_ckpt or None,
        "last_frame_memory": args.last_frame_memory,
        "continuity_candidates": args.continuity_candidates,
        "environment_memory": args.environment_memory,
        "transition_weight": args.transition_weight,
        "transition_floor": args.transition_floor,
        "environment_weight": args.environment_weight,
        "scene_change_env_decay": args.scene_change_env_decay,
        "reference_conditioning": args.reference_conditioning,
        "reference_tail_frames": args.reference_tail_frames,
        "reference_strength": args.reference_strength,
        "noise_conditioning": args.noise_conditioning,
        "noise_blend_strength": args.noise_blend_strength,
        "semantic_alignment_weight": args.semantic_alignment_weight,
        "semantic_alignment_threshold": args.semantic_alignment_threshold,
        "semantic_caption_sample_count": args.semantic_caption_sample_count,
        "style_weight": args.style_weight,
        "window_reference_images_json": args.window_reference_images_json or None,
        "disable_random_generation": args.disable_random_generation,
        "seed": args.seed,
        "seed_strategy": args.seed_strategy,
        "continuity_min_score": args.continuity_min_score,
        "continuity_regen_attempts": args.continuity_regen_attempts,
        "continuity_convergence_patience": args.continuity_convergence_patience,
        "continuity_convergence_tolerance": args.continuity_convergence_tolerance,
        "progressive_tightening": bool(args.progressive_tightening),
        "tightening_strength": float(args.tightening_strength),
        "critic_story_weight": args.critic_story_weight,
        "window_shard_count": args.window_shard_count,
        "window_shard_index": args.window_shard_index,
        "parallel_window_mode": args.parallel_window_mode,
        "selected_windows": len(selected_windows),
        "shot_plan_enforce": args.shot_plan_enforce,
        "shot_plan_defaults": args.shot_plan_defaults,
        "window_plan_json": args.window_plan_json or None,
        "output_dir": out_dir.as_posix(),
        "run_log": log_path.as_posix(),
        "captioner_model_id": args.captioner_model_id or None,
        "expected_caption_path": expected_caption_path.as_posix(),
        "dense_caption_path": dense_caption_path.as_posix(),
        "scene_conversation_plan_path": scene_conversation_path.as_posix(),
        "model_links_path": args.model_links,
        "selected_models": {
            "llm_model": selected_llm_model,
            "video_backbone": selected_video_model,
            "caption_gen": selected_caption_gen_model,
            "feedback_caption": selected_feedback_caption_model,
            "global_local_emb_feedback": selected_embedding_model,
        },
        "caption_alignment_metric": "token_jaccard_per_frame",
        "caption_windows_evaluated": len(caption_losses),
        "caption_alignment_loss_mean": (sum(caption_losses) / len(caption_losses)) if caption_losses else None,
        "caption_alignment_score_mean": (sum(caption_scores) / len(caption_scores)) if caption_scores else None,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] windows: {len(windows)}")
    print(f"[done] logs: {log_path.as_posix()}")


if __name__ == "__main__":
    main()
