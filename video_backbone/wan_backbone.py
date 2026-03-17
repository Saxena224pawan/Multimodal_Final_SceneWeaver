from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _default_local_wan_model_id() -> str:
    candidates = [
        "/home/vault/v123be/v123be37/sceneweaver_models/Wan2.2-I2V-A14B-Diffusers",
        "/home/vault/v123be/v123be37/sceneweaver_models/Wan2.1-I2V-14B-720P-Diffusers",
        "/home/vault/v123be/v123be37/sceneweaver_models/Wan2.1-I2V-14B-480P-Diffusers",
    ]
    for raw_path in candidates:
        path = Path(raw_path)
        if path.exists() and (path / "model_index.json").exists():
            return raw_path
    return candidates[-1]


@dataclass
class WanBackboneConfig:
    model_id: str = field(default_factory=_default_local_wan_model_id)
    torch_dtype: str = "bfloat16"
    device: str = "auto"
    enable_cpu_offload: bool = True


class WanBackbone:
    def __init__(self, config: WanBackboneConfig):
        self.config = config
        self.pipeline: Optional[Any] = None
        self._pipeline_call_params: Optional[set[str]] = None
        self._pipeline_accepts_kwargs: bool = False
        self._pipeline_required_params: set[str] = set()
        self.last_conditioning_mode: str = "none"
        self.last_reference_anchor_index: Optional[int] = None

    def _introspect_pipeline_call(self) -> None:
        if self.pipeline is None:
            return

        signature = inspect.signature(self.pipeline.__call__)
        params = set()
        accepts_kwargs = False
        self._pipeline_required_params = set()

        for name, param in signature.parameters.items():
            params.add(name)
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_kwargs = True
            if param.default is inspect.Parameter.empty:
                self._pipeline_required_params.add(name)

        self._pipeline_call_params = params
        self._pipeline_accepts_kwargs = accepts_kwargs

    def _supports_call_param(self, name: str) -> bool:
        if self._pipeline_call_params is None:
            return self._pipeline_accepts_kwargs
        return name in self._pipeline_call_params or self._pipeline_accepts_kwargs

    def supports_reference_conditioning(self) -> bool:
        return any(
            self._supports_call_param(p)
            for p in ("image", "last_image", "conditioning_frames", "video", "frames", "init_image")
        )

    def requires_reference_conditioning(self) -> bool:
        return any(
            p in self._pipeline_required_params
            for p in ("image", "last_image", "conditioning_frames", "video", "frames", "init_image")
        )

<<<<<<< Updated upstream
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Missing/unsupported runtime dependencies for text-to-video. "
                "Install or upgrade: torch, diffusers, transformers, accelerate. "
                "Example: pip install -U 'diffusers>=0.30' transformers accelerate"
            ) from exc
        try:
            from diffusers import AutoPipelineForText2Video as PipelineClass
        except Exception:
            try:
                # Fallback for diffusers builds that do not expose AutoPipelineForText2Video.
                from diffusers import DiffusionPipeline as PipelineClass
            except Exception as exc:
                detail = str(exc)
                xformers_hint = ""
                if "xformers" in detail.lower() or "jitcallable._set_src" in detail.lower():
                    xformers_hint = (
                        " Detected an xformers runtime mismatch. "
                        "Uninstall xformers or install a build matching this torch/cuda runtime."
                    )
                raise ImportError(
                    "Could not import a usable diffusers pipeline class. "
                    "Expected AutoPipelineForText2Video or DiffusionPipeline."
                    f"{xformers_hint}"
                ) from exc
=======
    def _to_pil(self, frame: Any, width: int, height: int):
        from PIL import Image
>>>>>>> Stashed changes

        img = Image.fromarray(self._frame_to_uint8(frame))
        if img.size != (width, height):
            img = img.resize((width, height))
        return img.convert("RGB")

    @staticmethod
    def _frame_to_uint8(frame: Any) -> np.ndarray:
        arr = np.asarray(frame)

        if arr.ndim == 4:
            arr = arr[-1]
        if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported frame shape {arr.shape!r}")

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 2:
            arr = np.concatenate([arr, arr[..., :1]], axis=-1)
        elif arr.shape[-1] >= 4:
            arr = arr[..., :3]

        if np.issubdtype(arr.dtype, np.floating):
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                arr = np.zeros_like(arr, dtype=np.float32)
            else:
                min_val = float(finite.min())
                max_val = float(finite.max())
                if min_val >= 0.0 and max_val <= 1.0:
                    arr = arr * 255.0
                elif min_val >= -1.0 and max_val <= 1.0:
                    arr = (arr + 1.0) * 127.5
                elif max_val <= 255.0 and min_val >= 0.0:
                    pass
                else:
                    scale = max(max_val - min_val, 1e-8)
                    arr = (arr - min_val) * (255.0 / scale)
        else:
            arr = arr.astype(np.float32, copy=False)

        return np.clip(arr, 0, 255).astype(np.uint8)

    def _extract_noise_pattern(self, frame: Any, strength: float = 0.2):
        arr = np.asarray(frame).astype(np.float32) / 255.0
        grad_x = np.abs(np.diff(arr, axis=1, append=arr[:, -1:]))
        grad_y = np.abs(np.diff(arr, axis=0, append=arr[-1:]))
        edges = np.sqrt(grad_x**2 + grad_y**2)
        if edges.ndim == 3:
            edges = np.mean(edges, axis=2)
        edges /= (np.max(edges) + 1e-8)
        noise = np.tanh(edges * 5.0)
        noise_rgb = np.stack([noise] * 3, axis=-1)
        mix = max(0.0, min(float(strength), 1.0))
        blended = arr * (1.0 - mix) + noise_rgb * mix
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

        from PIL import Image

        return Image.fromarray(blended)

    def _prepare_reference_frames(
        self,
        reference_frames,
        width: int,
        height: int,
        use_noise_conditioning: bool,
        noise_blend_strength: float,
    ):
        processed = [self._to_pil(frame, width, height) for frame in reference_frames]

        # Do not convert frames into edge/noise RGB images for Wan continuation.
        # The installed Wan I2V pipeline already supports first/last-frame conditioning
        # through `image` and `last_image`, which is much closer to temporal carryover
        # than the old handcrafted noise-map shortcut.
        _ = use_noise_conditioning
        _ = noise_blend_strength
        return processed

    @staticmethod
    def _reference_anchor_score(frame: Any) -> float:
        arr = np.asarray(frame).astype(np.float32)
        if arr.ndim == 2:
            gray = arr
        else:
            gray = arr[..., :3].mean(axis=2)
        if gray.size == 0:
            return 0.0

        grad_x = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
        grad_y = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
        sharpness = float(grad_x + grad_y)
        contrast = float(np.std(gray))
        brightness = float(np.mean(gray) / 255.0)
        exposure = max(0.0, 1.0 - (abs(brightness - 0.48) / 0.48))
        return sharpness * 0.75 + contrast * 0.25 + exposure * 12.0

    def _select_reference_anchor_frame(self, processed_frames, reference_source: str):
        if not processed_frames:
            return None, None
        if reference_source != "previous_window_tail" or len(processed_frames) == 1:
            return processed_frames[-1], len(processed_frames) - 1

        best_idx = max(
            range(len(processed_frames)),
            key=lambda idx: self._reference_anchor_score(processed_frames[idx]),
        )
        return processed_frames[best_idx], best_idx

    def _apply_reference_strength(self, call_kwargs: dict, reference_strength: float) -> None:
        for param_name in ("reference_strength", "conditioning_strength", "strength", "image_strength"):
            if self._supports_call_param(param_name):
                call_kwargs[param_name] = float(reference_strength)
                return

    def _supports_safe_last_image_conditioning(self) -> bool:
        if not (self._supports_call_param("image") and self._supports_call_param("last_image")):
            return False
        if self.pipeline is None:
            return False

        pipeline_cls = type(self.pipeline)
        module_name = getattr(pipeline_cls, "__module__", "")
        class_name = getattr(pipeline_cls, "__name__", "")

        # The installed Diffusers Wan I2V path currently encodes [image, last_image]
        # as a batch of 2 image embeddings while prompt embeddings remain batch 1.
        # That crashes inside transformer_wan during cross-attention concat on the
        # second window, so we keep using the last tail frame only until we add a
        # custom latent/image-embed continuation path.
        if class_name == "WanImageToVideoPipeline" or "diffusers.pipelines.wan.pipeline_wan_i2v" in module_name:
            return False
        return True

    def _apply_reference_conditioning(
        self,
        call_kwargs,
        reference_frames,
        reference_strength,
        width,
        height,
        reference_source: str = "none",
        use_noise_conditioning: bool = False,
        noise_blend_strength: float = 0.2,
    ):
        if not reference_frames:
            return "none"

        processed = self._prepare_reference_frames(
            reference_frames=reference_frames,
            width=width,
            height=height,
            use_noise_conditioning=use_noise_conditioning,
            noise_blend_strength=noise_blend_strength,
        )
        anchor_frame, anchor_index = self._select_reference_anchor_frame(processed, reference_source)
        self.last_reference_anchor_index = anchor_index

        if self._supports_safe_last_image_conditioning() and len(processed) >= 2:
            call_kwargs["image"] = anchor_frame if anchor_frame is not None else processed[0]
            call_kwargs["last_image"] = processed[-1]
            self._apply_reference_strength(call_kwargs, reference_strength)
            return "tail_pair"
        if self._supports_call_param("image"):
            call_kwargs["image"] = anchor_frame if anchor_frame is not None else processed[-1]
            self._apply_reference_strength(call_kwargs, reference_strength)
            if reference_source == "previous_window_tail" and len(processed) >= 2:
                return "tail_anchor_frame"
            return "reference"
        if self._supports_call_param("conditioning_frames"):
            call_kwargs["conditioning_frames"] = processed
            self._apply_reference_strength(call_kwargs, reference_strength)
            return "reference_sequence"
        if self._supports_call_param("frames"):
            call_kwargs["frames"] = processed
            self._apply_reference_strength(call_kwargs, reference_strength)
            return "reference_sequence"
        if self._supports_call_param("video"):
            call_kwargs["video"] = processed
            self._apply_reference_strength(call_kwargs, reference_strength)
            return "reference_sequence"
        if self._supports_call_param("init_image"):
            call_kwargs["init_image"] = processed[-1]
            self._apply_reference_strength(call_kwargs, reference_strength)
            return "reference"
        return "reference"

    def _get_torch_dtype(self, torch):
        if self.config.torch_dtype == "float16":
            return torch.float16
        if self.config.torch_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _resolve_device(self, torch: Any) -> str:
        normalized = (self.config.device or "auto").strip().lower()
        if normalized == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if normalized == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if normalized == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return normalized

    def load(self):
        import torch
        from diffusers import WanImageToVideoPipeline

        torch_dtype = self._get_torch_dtype(torch)
        pipe = WanImageToVideoPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
        )

        target_device = self._resolve_device(torch)
        if self.config.enable_cpu_offload and target_device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(target_device)

        self.pipeline = pipe
        self._introspect_pipeline_call()

    def generate_clip(
        self,
        prompt,
        negative_prompt="",
        num_frames=49,
        num_inference_steps=30,
        guidance_scale=6.0,
        height=480,
        width=832,
        seed=None,
        reference_frames=None,
        reference_strength=0.65,
        reference_source="none",
        use_noise_conditioning=False,
        noise_blend_strength=0.2,
        **kwargs,
    ):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded")

        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )

        if kwargs:
            for k, v in kwargs.items():
                if self._supports_call_param(k):
                    call_kwargs[k] = v

        self.last_reference_anchor_index = None
        conditioning_mode = self._apply_reference_conditioning(
            call_kwargs,
            reference_frames,
            reference_strength,
            width,
            height,
            reference_source=str(reference_source or "none"),
            use_noise_conditioning=bool(use_noise_conditioning),
            noise_blend_strength=float(noise_blend_strength),
        )
        self.last_conditioning_mode = conditioning_mode

        result = self.pipeline(**call_kwargs)
        frames = getattr(result, "frames", None)
        if frames is None:
            raise RuntimeError("No frames returned")
        return frames[0]

    @staticmethod
    def save_video(frames, output_path, fps=8):
        import imageio.v3 as iio

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        arr = np.stack([WanBackbone._frame_to_uint8(f) for f in frames], axis=0)
        iio.imwrite(out.as_posix(), arr, fps=fps)
