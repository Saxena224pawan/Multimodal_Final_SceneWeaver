from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .expected_video_captioner import validate_scene_plan


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _phase_name(segment_idx: int, total_segments: int) -> str:
    if total_segments <= 1:
        return "end"
    if segment_idx == 0:
        return "start"
    if segment_idx == total_segments - 1:
        return "end"
    ratio = segment_idx / max(1, total_segments - 1)
    if ratio < 0.55:
        return "middle"
    return "late"


def _phase_caption_text(
    *,
    frame_range: str,
    phase: str,
    characters: str,
    world_anchor: str,
    objective: str,
    emotion: str,
) -> str:
    anchor = world_anchor.replace("_", " ")
    objective_text = f'the objective "{objective}"'
    if phase == "start":
        core = f"Action starts for {characters} toward {objective_text}"
    elif phase == "middle":
        core = f"Action continues for {characters} toward {objective_text}"
    elif phase == "late":
        core = f"Action nears completion for {characters} toward {objective_text}"
    else:
        core = f"Action completes for {characters} with {objective_text}"
    return f"Frames {frame_range}: {core} at {anchor}. Emotion: {emotion}."


def _build_window_segments(
    *,
    start_frame: int,
    end_frame: int,
    segment_length_frames: int,
    overlap_frames: int,
) -> List[Tuple[int, int]]:
    stride = segment_length_frames - overlap_frames
    starts = list(range(start_frame, end_frame + 1, stride))

    segments: List[Tuple[int, int]] = []
    for start in starts:
        end = min(start + segment_length_frames - 1, end_frame)
        segments.append((start, end))
        if end == end_frame:
            break

    if segments and segments[-1][1] < end_frame:
        final_start = max(start_frame, end_frame - segment_length_frames + 1)
        final_segment = (final_start, end_frame)
        if final_segment not in segments:
            segments.append(final_segment)

    if not segments:
        segments.append((start_frame, end_frame))
    return segments


def build_dense_expected_caption_pack(
    scene_plan: Dict[str, Any],
    *,
    fps: int = 24,
    segment_length_frames: int = 6,
    overlap_frames: int = 0,
) -> Dict[str, Any]:
    errors = validate_scene_plan(scene_plan)
    if errors:
        rendered = "\n".join(f"- {err['path']}: {err['message']}" for err in errors)
        raise ValueError(f"Invalid scene plan:\n{rendered}")

    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    if not isinstance(segment_length_frames, int) or segment_length_frames <= 0:
        raise ValueError("segment_length_frames must be a positive integer")
    if not isinstance(overlap_frames, int) or overlap_frames < 0:
        raise ValueError("overlap_frames must be >= 0")
    if overlap_frames >= segment_length_frames:
        raise ValueError("overlap_frames must be smaller than segment_length_frames")

    window_seconds = int(scene_plan["window_seconds"])
    frames_per_window = window_seconds * fps
    if frames_per_window <= 0:
        raise ValueError("window_seconds * fps must be > 0")

    segments: List[Dict[str, Any]] = []
    previous_segment_id: str | None = None

    for window in scene_plan["windows"]:
        window_index = int(window["window_index"])
        window_id = window["window_id"]
        window_start = window_index * frames_per_window + 1
        window_end = window_start + frames_per_window - 1

        window_segments = _build_window_segments(
            start_frame=window_start,
            end_frame=window_end,
            segment_length_frames=segment_length_frames,
            overlap_frames=overlap_frames,
        )

        characters = ", ".join(window["character_names"])
        objective = str(window["scene_objective"]).strip().rstrip(".")
        emotion = str(window["emotion"]).strip().rstrip(".")
        world_anchor = str(window["continuity_anchor"]["world_anchor"]).strip()

        for idx, (start_frame, end_frame) in enumerate(window_segments):
            segment_id = f"seg_{len(segments):04d}"
            frame_range = f"{start_frame}-{end_frame}"
            phase = _phase_name(idx, len(window_segments))
            expected_caption = _phase_caption_text(
                frame_range=frame_range,
                phase=phase,
                characters=characters,
                world_anchor=world_anchor,
                objective=objective,
                emotion=emotion,
            )

            segments.append(
                {
                    "segment_id": segment_id,
                    "window_id": window_id,
                    "window_index": window_index,
                    "beat_id": window["beat_id"],
                    "phase": phase,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frame_range": frame_range,
                    "start_time_sec": round((start_frame - 1) / fps, 3),
                    "end_time_sec": round(end_frame / fps, 3),
                    "expected_caption": expected_caption,
                    "alignment_targets": {
                        "scene_objective": objective,
                        "emotion": emotion,
                        "character_names": window["character_names"],
                        "world_anchor": world_anchor,
                        "previous_window_id": window["continuity_anchor"].get("previous_window_id"),
                        "previous_segment_id": previous_segment_id,
                    },
                }
            )
            previous_segment_id = segment_id

    return {
        "version": 1,
        "story_id": scene_plan["story_id"],
        "title": scene_plan["title"],
        "window_seconds": scene_plan["window_seconds"],
        "total_windows": scene_plan["total_windows"],
        "fps": fps,
        "segment_length_frames": segment_length_frames,
        "overlap_frames": overlap_frames,
        "total_segments": len(segments),
        "source": {
            "module": "caption_gen",
            "scene_plan_version": scene_plan.get("version"),
            "mode": "dense_temporal_captioning",
        },
        "segments": segments,
    }
