from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple


@dataclass
class CaptionerConfig:
    model_id: str
    device: str = "cpu"
    stub_fallback: bool = True
    max_new_tokens: int = 32


class Captioner:
    """
    Lightweight captioner wrapper with a reliable stub mode.

    The active pipeline only needs three things from this class:
    per-frame captions, a compact summary, and a duplicate-subject hint.
    """

    def __init__(self, config: CaptionerConfig):
        self.config = config
        self.model = None
        self.processor = None
        self._caption_fn = None
        self.is_stub = False

    def load(self) -> None:
        model_id = (self.config.model_id or "").strip()
        if not model_id or model_id == "__stub__":
            self._enable_stub()
            return

        try:
            from transformers import pipeline
        except ImportError:
            if self.config.stub_fallback:
                self._enable_stub()
                return
            raise ImportError("transformers is required to load the configured captioner model.")

        task = "image-to-text"
        device = self._resolve_pipeline_device(self.config.device)
        try:
            self._caption_fn = pipeline(task=task, model=model_id, device=device)
            self.is_stub = False
        except Exception:
            if self.config.stub_fallback:
                self._enable_stub()
                return
            raise

    def caption_frames(self, frames: Sequence[Any], sample_count: int = 4) -> Tuple[List[str], str, bool]:
        if self._caption_fn is None and not self.is_stub:
            raise RuntimeError("Captioner not loaded. Call load() first.")
        frame_list = list(frames)
        if not frame_list:
            return [], "", False

        sampled = self._sample_frames(frame_list, sample_count=sample_count)
        if self.is_stub:
            captions = [self._stub_caption(frame, idx) for idx, frame in enumerate(sampled)]
        else:
            captions = [self._model_caption(frame) for frame in sampled]

        if sample_count == 0:
            expanded = captions
        else:
            expanded = self._expand_captions(captions, total_frames=len(frame_list))
        summary = self._build_summary(expanded)
        dupes = self._has_duplicate_subjects(summary)
        return expanded, summary, dupes

    def _enable_stub(self) -> None:
        self._caption_fn = None
        self.is_stub = True

    @staticmethod
    def _resolve_pipeline_device(device: str) -> int:
        normalized = (device or "cpu").strip().lower()
        if normalized in {"cpu", "mps", "auto"}:
            return -1
        if normalized == "cuda":
            return 0
        return -1

    @staticmethod
    def _sample_frames(frames: List[Any], sample_count: int) -> List[Any]:
        if sample_count <= 0 or len(frames) <= sample_count:
            return frames
        stride = max(1, len(frames) // sample_count)
        sampled = frames[::stride][:sample_count]
        return sampled if sampled else [frames[0]]

    @staticmethod
    def _expand_captions(sampled_captions: List[str], total_frames: int) -> List[str]:
        if not sampled_captions:
            return []
        if len(sampled_captions) >= total_frames:
            return sampled_captions[:total_frames]
        expanded: List[str] = []
        for idx in range(total_frames):
            source_idx = int(round((idx / max(1, total_frames - 1)) * (len(sampled_captions) - 1)))
            expanded.append(sampled_captions[source_idx])
        return expanded

    def _model_caption(self, frame: Any) -> str:
        result = self._caption_fn(self._to_pil(frame), max_new_tokens=int(self.config.max_new_tokens))
        if isinstance(result, list) and result:
            payload = result[0]
            if isinstance(payload, dict):
                text = payload.get("generated_text") or payload.get("caption") or ""
                return " ".join(str(text).split()).strip() or "scene frame"
        return "scene frame"

    def _stub_caption(self, frame: Any, frame_index: int) -> str:
        pil = self._to_pil(frame)
        arr = self._pil_to_array(pil)
        mean_rgb = arr.mean(axis=(0, 1))
        brightness = float(mean_rgb.mean()) / 255.0
        channel = int(mean_rgb.argmax())
        palette = ("reddish", "greenish", "bluish")
        light = "bright" if brightness >= 0.55 else "dim"
        dominant = palette[channel]
        return f"{light} {dominant} scene frame {frame_index + 1}"

    @staticmethod
    def _build_summary(captions: Sequence[str]) -> str:
        unique: List[str] = []
        for caption in captions:
            compact = " ".join((caption or "").split()).strip()
            if compact and compact not in unique:
                unique.append(compact)
            if len(unique) >= 3:
                break
        return "; ".join(unique) if unique else "scene continuation"

    @staticmethod
    def _has_duplicate_subjects(summary: str) -> bool:
        lowered = (summary or "").lower()
        duplicate_markers = (
            "two ",
            "multiple ",
            "duplicate ",
            "extra person",
            "extra people",
            "extra animal",
            "extra animals",
            "another crow",
            "two crows",
        )
        return any(marker in lowered for marker in duplicate_markers)

    @staticmethod
    def _to_pil(frame: Any):
        from PIL import Image
        import numpy as np

        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        arr = np.asarray(frame)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype.kind in ("f", "c"):
            arr = arr.clip(0.0, 1.0) * 255.0
        arr = arr.clip(0, 255).astype("uint8")
        return Image.fromarray(arr).convert("RGB")

    @staticmethod
    def _pil_to_array(image: Any):
        import numpy as np

        return np.asarray(image.convert("RGB"))
