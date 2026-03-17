#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one window video clip from a prompt using a diffusers text-to-video pipeline."
    )
    parser.add_argument("--prompt-file", required=True, help="Path to prompt text file.")
    parser.add_argument("--out", required=True, help="Output video path.")
    parser.add_argument("--frames-dir", default=None, help="Optional directory to save PNG frames.")
    parser.add_argument(
        "--model-id",
        required=True,
        help="Primary model path or repo id (usually local path from pipeline manifest).",
    )
    parser.add_argument(
        "--fallback-model-id",
        default=None,
        help="Fallback repo id if --model-id points to a missing local path.",
    )
    parser.add_argument("--window-id", default=None, help="Optional window id used in logs.")
    parser.add_argument("--num-frames", type=int, default=24, help="Number of frames to generate.")
    parser.add_argument("--fps-out", type=int, default=8, help="Output video fps.")
    parser.add_argument("--steps", type=int, default=20, help="Diffusion steps.")
    parser.add_argument("--guidance-scale", type=float, default=6.5, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype.",
    )
    parser.add_argument("--height", type=int, default=320, help="Output height.")
    parser.add_argument("--width", type=int, default=576, help="Output width.")
    parser.add_argument("--negative-prompt", default="", help="Optional negative prompt.")
    return parser.parse_args()


def _resolve_model_id(model_id: str, fallback_model_id: Optional[str]) -> str:
    model_path = Path(model_id)
    if model_path.exists():
        return str(model_path)
    if fallback_model_id:
        return fallback_model_id
    return model_id


def _pick_device(requested: str) -> str:
    import torch

    if requested == "cuda":
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pick_dtype(requested: str, device: str):
    import torch

    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    if device == "cuda":
        return torch.float16
    return torch.float32


def _load_pipeline(model_id: str, dtype):
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

    attempts = []
    attempts.append({"torch_dtype": dtype, "variant": "fp16"} if str(dtype).endswith("float16") else {"torch_dtype": dtype})
    attempts.append({"torch_dtype": dtype})
    attempts.append({})

    last_error = None
    for kwargs in attempts:
        try:
            pipe = DiffusionPipeline.from_pretrained(model_id, **kwargs)
            if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            return pipe
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"failed to load pipeline: {last_error}")


def _save_frame_image(frame, frame_path: Path) -> None:
    if hasattr(frame, "save"):
        frame.save(frame_path)
        return

    import numpy as np
    from PIL import Image

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            if float(array.max(initial=0.0)) <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        Image.fromarray(array, mode="L").save(frame_path)
        return
    if array.ndim == 3 and array.shape[2] in (1, 3, 4):
        if array.shape[2] == 1:
            array = array[:, :, 0]
        Image.fromarray(array).save(frame_path)
        return

    raise ValueError(f"unsupported frame shape for image export: {array.shape}")


def main() -> int:
    args = parse_args()
    prompt_path = Path(args.prompt_file)
    out_path = Path(args.out)
    frames_dir = Path(args.frames_dir) if args.frames_dir else None

    if not prompt_path.exists():
        raise FileNotFoundError(f"prompt file not found: {prompt_path}")

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"prompt file is empty: {prompt_path}")

    model_id = _resolve_model_id(args.model_id, args.fallback_model_id)
    device = _pick_device(args.device)
    dtype = _pick_dtype(args.dtype, device)

    print(f"[INFO] window_id={args.window_id or '<unknown>'}")
    print(f"[INFO] model_id={model_id}")
    print(f"[INFO] device={device} dtype={dtype}")
    print(f"[INFO] num_frames={args.num_frames} steps={args.steps} fps_out={args.fps_out}")

    import torch
    from diffusers.utils import export_to_video

    pipe = _load_pipeline(model_id, dtype)

    if device == "cuda":
        pipe.to("cuda")
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("auto")
    else:
        pipe.to("cpu")

    generator = torch.Generator(device=device).manual_seed(args.seed)

    run_kwargs = {
        "prompt": prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "generator": generator,
    }
    if args.negative_prompt.strip():
        run_kwargs["negative_prompt"] = args.negative_prompt.strip()

    result = pipe(**run_kwargs)
    frames = result.frames[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out_path), fps=args.fps_out)
    print(f"[OK] video written: {out_path}")

    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        failed_frames = 0
        for idx, frame in enumerate(frames):
            frame_path = frames_dir / f"frame_{idx:04d}.png"
            try:
                _save_frame_image(frame, frame_path)
            except Exception as exc:  # noqa: BLE001
                failed_frames += 1
                print(f"[WARN] failed to save frame {idx} to {frame_path}: {exc}")
        written = len(frames) - failed_frames
        print(f"[OK] frames written: {frames_dir} ({written}/{len(frames)} frames)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
