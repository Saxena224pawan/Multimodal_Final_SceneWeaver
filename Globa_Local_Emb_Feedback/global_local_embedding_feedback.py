from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


WINDOW_ID_RE = re.compile(r"^w_[0-9]{3,}$")
FRAME_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


REQUIRED_SCENE_PLAN_FIELDS = {"story_id", "total_windows", "windows"}
REQUIRED_SCENE_WINDOW_FIELDS = {"window_id", "window_index", "beat_id", "continuity_anchor"}

REQUIRED_VIDEO_MANIFEST_FIELDS = {"story_id", "total_windows", "jobs"}
REQUIRED_VIDEO_JOB_FIELDS = {
    "window_id",
    "window_index",
    "beat_id",
    "frames_dir",
    "output_video_path",
}


def _add_error(errors: List[Dict[str, str]], path: str, message: str) -> None:
    errors.append({"path": path, "message": message})


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def _collect_frame_paths(frames_dir: Path) -> List[Path]:
    if not frames_dir.exists() or not frames_dir.is_dir():
        return []
    return sorted(
        [
            path
            for path in frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES
        ]
    )


def _sample_frame_paths(frame_paths: Sequence[Path], max_frames: int) -> List[Path]:
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if len(frame_paths) <= max_frames:
        return list(frame_paths)
    if max_frames == 1:
        return [frame_paths[0]]
    last_index = len(frame_paths) - 1
    sampled: List[Path] = []
    used_indices: set[int] = set()
    for slot in range(max_frames):
        idx = round((slot * last_index) / (max_frames - 1))
        if idx not in used_indices:
            sampled.append(frame_paths[idx])
            used_indices.add(idx)
    if len(sampled) < max_frames:
        for idx, path in enumerate(frame_paths):
            if idx in used_indices:
                continue
            sampled.append(path)
            if len(sampled) >= max_frames:
                break
    return sampled


def validate_scene_plan_for_embedding_feedback(payload: Any) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if not isinstance(payload, dict):
        _add_error(errors, "$", "scene plan must be a JSON object")
        return errors

    fields = set(payload.keys())
    for missing in sorted(REQUIRED_SCENE_PLAN_FIELDS - fields):
        _add_error(errors, missing, "missing required field")

    if "story_id" in payload and not _is_non_empty_string(payload.get("story_id")):
        _add_error(errors, "story_id", "must be a non-empty string")
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

    if isinstance(windows, list) and isinstance(payload.get("total_windows"), int):
        if len(windows) != payload["total_windows"]:
            _add_error(
                errors,
                "total_windows",
                f"declares {payload['total_windows']}, but windows has {len(windows)} items",
            )

    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for idx, item in enumerate(windows or []):
        base = f"windows[{idx}]"
        if not isinstance(item, dict):
            _add_error(errors, base, "must be an object")
            continue

        for missing in sorted(REQUIRED_SCENE_WINDOW_FIELDS - set(item.keys())):
            _add_error(errors, f"{base}.{missing}", "missing required field")

        window_id = item.get("window_id")
        if "window_id" in item:
            if not _is_non_empty_string(window_id):
                _add_error(errors, f"{base}.window_id", "must be a non-empty string")
            elif not WINDOW_ID_RE.match(window_id):
                _add_error(errors, f"{base}.window_id", "must match pattern w_000")
            elif window_id in seen_ids:
                _add_error(errors, f"{base}.window_id", f"duplicate window_id '{window_id}'")
            else:
                seen_ids.add(window_id)

        window_index = item.get("window_index")
        if "window_index" in item:
            if not isinstance(window_index, int) or window_index < 0:
                _add_error(errors, f"{base}.window_index", "must be an integer >= 0")
            elif window_index in seen_indices:
                _add_error(errors, f"{base}.window_index", f"duplicate window_index '{window_index}'")
            else:
                seen_indices.add(window_index)

        for field_name in ("beat_id",):
            if field_name in item and not _is_non_empty_string(item.get(field_name)):
                _add_error(errors, f"{base}.{field_name}", "must be a non-empty string")

        continuity = item.get("continuity_anchor")
        if "continuity_anchor" in item:
            if not isinstance(continuity, dict):
                _add_error(errors, f"{base}.continuity_anchor", "must be an object")
            else:
                world_anchor = continuity.get("world_anchor")
                if not _is_non_empty_string(world_anchor):
                    _add_error(errors, f"{base}.continuity_anchor.world_anchor", "must be a non-empty string")

    return errors


def validate_video_manifest_for_embedding_feedback(payload: Any) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if not isinstance(payload, dict):
        _add_error(errors, "$", "video manifest must be a JSON object")
        return errors

    fields = set(payload.keys())
    for missing in sorted(REQUIRED_VIDEO_MANIFEST_FIELDS - fields):
        _add_error(errors, missing, "missing required field")

    if "story_id" in payload and not _is_non_empty_string(payload.get("story_id")):
        _add_error(errors, "story_id", "must be a non-empty string")
    if "total_windows" in payload:
        value = payload.get("total_windows")
        if not isinstance(value, int) or value <= 0:
            _add_error(errors, "total_windows", "must be a positive integer")

    jobs = payload.get("jobs")
    if "jobs" in payload:
        if not isinstance(jobs, list):
            _add_error(errors, "jobs", "must be a list")
            jobs = []
        elif not jobs:
            _add_error(errors, "jobs", "must contain at least one job")

    if isinstance(jobs, list) and isinstance(payload.get("total_windows"), int):
        if len(jobs) != payload["total_windows"]:
            _add_error(
                errors,
                "total_windows",
                f"declares {payload['total_windows']}, but jobs has {len(jobs)} items",
            )

    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for idx, item in enumerate(jobs or []):
        base = f"jobs[{idx}]"
        if not isinstance(item, dict):
            _add_error(errors, base, "must be an object")
            continue

        for missing in sorted(REQUIRED_VIDEO_JOB_FIELDS - set(item.keys())):
            _add_error(errors, f"{base}.{missing}", "missing required field")

        window_id = item.get("window_id")
        if "window_id" in item:
            if not _is_non_empty_string(window_id):
                _add_error(errors, f"{base}.window_id", "must be a non-empty string")
            elif not WINDOW_ID_RE.match(window_id):
                _add_error(errors, f"{base}.window_id", "must match pattern w_000")
            elif window_id in seen_ids:
                _add_error(errors, f"{base}.window_id", f"duplicate window_id '{window_id}'")
            else:
                seen_ids.add(window_id)

        window_index = item.get("window_index")
        if "window_index" in item:
            if not isinstance(window_index, int) or window_index < 0:
                _add_error(errors, f"{base}.window_index", "must be an integer >= 0")
            elif window_index in seen_indices:
                _add_error(errors, f"{base}.window_index", f"duplicate window_index '{window_index}'")
            else:
                seen_indices.add(window_index)

        if "frames_dir" in item and not _is_non_empty_string(item.get("frames_dir")):
            _add_error(errors, f"{base}.frames_dir", "must be a non-empty string")
        if "output_video_path" in item and not _is_non_empty_string(item.get("output_video_path")):
            _add_error(errors, f"{base}.output_video_path", "must be a non-empty string")

    return errors


class _DinoV2Embedder:
    def __init__(self, *, model_source: str, device: str) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "DINOv2 embedding requires torch, pillow, and transformers. "
                "Install with: pip install torch pillow transformers"
            ) from exc

        self.torch = torch
        self.Image = Image

        resolved_device = device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved_device
        self.model_source = model_source

        self.processor = AutoImageProcessor.from_pretrained(model_source)
        self.model = AutoModel.from_pretrained(model_source)
        self.model.eval()
        self.model.to(self.device)

    def _normalize_vector(self, vector: Any) -> Any:
        norm = self.torch.norm(vector, p=2)
        if float(norm.item()) <= 1e-12:
            return vector
        return vector / norm

    def _normalize_rows(self, matrix: Any) -> Any:
        norms = self.torch.norm(matrix, p=2, dim=1, keepdim=True)
        safe = self.torch.where(norms > 1e-12, norms, self.torch.ones_like(norms))
        return matrix / safe

    def embed_images(self, frame_paths: Sequence[Path], *, local_grid: int) -> Tuple[Any, Any]:
        if local_grid <= 0:
            raise ValueError("local_grid must be > 0")

        global_vectors: List[Any] = []
        local_vectors: List[Any] = []

        for frame_path in frame_paths:
            image = self.Image.open(frame_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self.torch.no_grad():
                outputs = self.model(**inputs)

            hidden = outputs.last_hidden_state[0]
            cls_token = hidden[0]
            patch_tokens = hidden[1:]

            if patch_tokens.shape[0] == 0:
                local_per_frame = cls_token.repeat(local_grid * local_grid, 1)
            else:
                patch_count = int(patch_tokens.shape[0])
                side = int(math.sqrt(patch_count))
                if side * side != patch_count:
                    mean_patch = self.torch.mean(patch_tokens, dim=0)
                    local_per_frame = mean_patch.repeat(local_grid * local_grid, 1)
                else:
                    patch_map = patch_tokens.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0)
                    pooled = self.torch.nn.functional.adaptive_avg_pool2d(patch_map, (local_grid, local_grid))
                    local_per_frame = (
                        pooled.squeeze(0).permute(1, 2, 0).reshape(local_grid * local_grid, -1)
                    )

            global_vectors.append(cls_token.detach().cpu())
            local_vectors.append(local_per_frame.detach().cpu())

        global_window = self.torch.mean(self.torch.stack(global_vectors, dim=0), dim=0)
        local_window = self.torch.mean(self.torch.stack(local_vectors, dim=0), dim=0)

        global_window = self._normalize_vector(global_window)
        local_window = self._normalize_rows(local_window)
        return global_window, local_window


def _cosine_similarity(torch_mod: Any, vec_a: Any, vec_b: Any) -> Optional[float]:
    denom = torch_mod.norm(vec_a, p=2) * torch_mod.norm(vec_b, p=2)
    denom_value = float(denom.item())
    if denom_value <= 1e-12:
        return None
    value = torch_mod.dot(vec_a, vec_b) / denom
    return float(value.item())


def _mean_row_cosine_similarity(torch_mod: Any, rows_a: Any, rows_b: Any) -> Optional[float]:
    if rows_a.shape != rows_b.shape:
        return None
    if int(rows_a.shape[0]) == 0:
        return None
    dots = torch_mod.sum(rows_a * rows_b, dim=1)
    norms = torch_mod.norm(rows_a, p=2, dim=1) * torch_mod.norm(rows_b, p=2, dim=1)
    mask = norms > 1e-12
    if int(mask.sum().item()) == 0:
        return None
    sims = dots[mask] / norms[mask]
    return float(torch_mod.mean(sims).item())


def _to_vector_list(vector: Any, *, digits: int = 6) -> List[float]:
    return [round(float(v), digits) for v in vector.tolist()]


def _to_matrix_list(matrix: Any, *, digits: int = 6) -> List[List[float]]:
    return [[round(float(v), digits) for v in row] for row in matrix.tolist()]


def build_embedding_feedback(
    scene_plan: Dict[str, Any],
    video_manifest: Dict[str, Any],
    *,
    model_source: str,
    sample_frames: int = 8,
    local_grid: int = 3,
    global_min_similarity: float = 0.78,
    local_min_similarity: float = 0.72,
    device: str = "auto",
    allow_missing_frames: bool = True,
    include_vectors: bool = False,
) -> Dict[str, Any]:
    scene_errors = validate_scene_plan_for_embedding_feedback(scene_plan)
    manifest_errors = validate_video_manifest_for_embedding_feedback(video_manifest)
    errors = scene_errors + manifest_errors
    if errors:
        rendered = "\n".join(f"- {item['path']}: {item['message']}" for item in errors)
        raise ValueError(f"Invalid inputs for embedding feedback:\n{rendered}")

    if sample_frames <= 0:
        raise ValueError("sample_frames must be > 0")
    if local_grid <= 0:
        raise ValueError("local_grid must be > 0")
    if not (0.0 <= global_min_similarity <= 1.0):
        raise ValueError("global_min_similarity must be in [0, 1]")
    if not (0.0 <= local_min_similarity <= 1.0):
        raise ValueError("local_min_similarity must be in [0, 1]")
    if not _is_non_empty_string(model_source):
        raise ValueError("model_source must be a non-empty string")

    scene_windows_by_id: Dict[str, Dict[str, Any]] = {
        item["window_id"]: item for item in scene_plan["windows"]
    }
    jobs = sorted(video_manifest["jobs"], key=lambda item: int(item["window_index"]))

    embedder: Optional[_DinoV2Embedder] = None
    previous_global = None
    previous_local = None
    previous_embedding_window_id: Optional[str] = None

    rows: List[Dict[str, Any]] = []
    no_frames_windows = 0
    error_windows = 0
    computed_windows = 0
    compared_windows = 0
    pass_windows = 0
    fail_windows = 0
    first_embedding_window_id: Optional[str] = None

    for job in jobs:
        window_id = job["window_id"]
        window_index = int(job["window_index"])
        beat_id = job["beat_id"]

        scene_window = scene_windows_by_id.get(window_id, {})
        continuity = scene_window.get("continuity_anchor", {})
        world_anchor = continuity.get("world_anchor") if isinstance(continuity, dict) else None
        previous_window_id = continuity.get("previous_window_id") if isinstance(continuity, dict) else None
        entity_anchors = (
            scene_window.get("feedback_targets", {})
            .get("embedding_alignment", {})
            .get("entity_anchors")
        )
        if not isinstance(entity_anchors, list):
            entity_anchors = []

        frame_paths = _collect_frame_paths(Path(job["frames_dir"]))
        sampled_frame_paths = _sample_frame_paths(frame_paths, sample_frames) if frame_paths else []

        if not sampled_frame_paths:
            no_frames_windows += 1
            status = "no_frames" if allow_missing_frames else "error"
            if status == "error":
                error_windows += 1
            rows.append(
                {
                    "window_id": window_id,
                    "window_index": window_index,
                    "beat_id": beat_id,
                    "world_anchor": world_anchor,
                    "expected_previous_window_id": previous_window_id,
                    "previous_embedding_window_id": previous_embedding_window_id,
                    "frame_count": 0,
                    "sampled_frame_count": 0,
                    "status": status,
                    "global_similarity_prev": None,
                    "local_similarity_prev": None,
                    "drift_score": None,
                    "passes_thresholds": None,
                    "message": "No frame images found in frames_dir.",
                    "entity_anchors": entity_anchors,
                }
            )
            continue

        try:
            if embedder is None:
                embedder = _DinoV2Embedder(model_source=model_source, device=device)
            global_vec, local_vec = embedder.embed_images(sampled_frame_paths, local_grid=local_grid)
        except Exception as exc:  # noqa: BLE001 - explicit error surface for pipeline reporting
            error_windows += 1
            rows.append(
                {
                    "window_id": window_id,
                    "window_index": window_index,
                    "beat_id": beat_id,
                    "world_anchor": world_anchor,
                    "expected_previous_window_id": previous_window_id,
                    "previous_embedding_window_id": previous_embedding_window_id,
                    "frame_count": len(frame_paths),
                    "sampled_frame_count": len(sampled_frame_paths),
                    "status": "error",
                    "global_similarity_prev": None,
                    "local_similarity_prev": None,
                    "drift_score": None,
                    "passes_thresholds": None,
                    "message": str(exc),
                    "entity_anchors": entity_anchors,
                }
            )
            continue

        computed_windows += 1
        if first_embedding_window_id is None:
            first_embedding_window_id = window_id

        global_similarity_prev: Optional[float] = None
        local_similarity_prev: Optional[float] = None
        drift_score: Optional[float] = None
        passes_thresholds: Optional[bool]
        message = "Embedding baseline window."

        if previous_global is None or previous_local is None:
            passes_thresholds = True
        else:
            compared_windows += 1
            global_similarity_prev = _cosine_similarity(embedder.torch, global_vec, previous_global)
            local_similarity_prev = _mean_row_cosine_similarity(embedder.torch, local_vec, previous_local)
            if global_similarity_prev is None or local_similarity_prev is None:
                passes_thresholds = False
                message = "Unable to compute one or more similarity values."
            else:
                passes_thresholds = (
                    global_similarity_prev >= global_min_similarity
                    and local_similarity_prev >= local_min_similarity
                )
                drift_score = 1.0 - ((global_similarity_prev + local_similarity_prev) / 2.0)
                if passes_thresholds:
                    message = "Similarity above thresholds."
                else:
                    message = "Similarity below one or more thresholds."

            if passes_thresholds:
                pass_windows += 1
            else:
                fail_windows += 1

        row = {
            "window_id": window_id,
            "window_index": window_index,
            "beat_id": beat_id,
            "world_anchor": world_anchor,
            "expected_previous_window_id": previous_window_id,
            "previous_embedding_window_id": previous_embedding_window_id,
            "frame_count": len(frame_paths),
            "sampled_frame_count": len(sampled_frame_paths),
            "status": "ok",
            "global_similarity_prev": _round_or_none(global_similarity_prev),
            "local_similarity_prev": _round_or_none(local_similarity_prev),
            "drift_score": _round_or_none(drift_score),
            "passes_thresholds": passes_thresholds,
            "message": message,
            "entity_anchors": entity_anchors,
        }
        if include_vectors:
            row["global_embedding"] = _to_vector_list(global_vec)
            row["local_embedding_grid"] = _to_matrix_list(local_vec)
        rows.append(row)

        previous_global = global_vec
        previous_local = local_vec
        previous_embedding_window_id = window_id

    return {
        "version": 1,
        "story_id": scene_plan["story_id"],
        "source": {
            "module": "global_local_emb_feedback",
            "model_source": model_source,
            "model_family": "dinov2",
        },
        "config": {
            "sample_frames": sample_frames,
            "local_grid": local_grid,
            "global_min_similarity": global_min_similarity,
            "local_min_similarity": local_min_similarity,
            "device": device,
            "allow_missing_frames": allow_missing_frames,
            "include_vectors": include_vectors,
        },
        "summary": {
            "total_windows": len(jobs),
            "computed_windows": computed_windows,
            "compared_windows": compared_windows,
            "pass_windows": pass_windows,
            "fail_windows": fail_windows,
            "no_frames_windows": no_frames_windows,
            "error_windows": error_windows,
            "first_embedding_window_id": first_embedding_window_id,
        },
        "rows": rows,
    }
