#!/bin/bash -l
#SBATCH --job-name=gen_win_lora
#SBATCH --output=slurm_logs/gen_win_lora_%j.out
#SBATCH --error=slurm_logs/gen_win_lora_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
CONDA_SH="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
ENV_PATH="${ENV_PATH:-sceneweaver311}"

MODEL_ID="${MODEL_ID:-THUDM/CogVideoX-2b}"
MANIFEST_PATH="${MANIFEST_PATH:-${PROJECT_ROOT}/outputs/video/debug/generation_manifest.json}"
LORA_DIR="${LORA_DIR:-${PROJECT_ROOT}/outputs/lora/ghibli_cogvideox_lora_smoke}"
LORA_WEIGHT="${LORA_WEIGHT:-pytorch_lora_weights.safetensors}"

NUM_FRAMES="${NUM_FRAMES:-0}"  # 0 => auto from manifest window_seconds * fps_out
AUTO_NUM_FRAMES_FROM_WINDOW="${AUTO_NUM_FRAMES_FROM_WINDOW:-1}"
STEPS="${STEPS:-36}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.5}"
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-576}"
FPS_OUT="${FPS_OUT:-8}"
LORA_SCALE="${LORA_SCALE:-0.30}"
SEED_BASE="${SEED_BASE:-1234}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-420}"
SAVE_FRAMES="${SAVE_FRAMES:-1}"

# Optional: render one window without LoRA before LoRA generation.
RUN_BASELINE_CHECK="${RUN_BASELINE_CHECK:-1}"
BASELINE_WINDOW_ID="${BASELINE_WINDOW_ID:-w_000}"

# Optional comma-separated subset, e.g. WINDOW_FILTER="w_000,w_001"
WINDOW_FILTER="${WINDOW_FILTER:-}"

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-dark, underexposed, blurry, low contrast, artifacts, ghosting, duplicate people, deformed anatomy, text, watermark, logo}"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/.hf"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${ENV_PATH}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MODEL_ID=${MODEL_ID}"
echo "MANIFEST_PATH=${MANIFEST_PATH}"
echo "LORA_DIR=${LORA_DIR}"
echo "LORA_WEIGHT=${LORA_WEIGHT}"
echo "LORA_SCALE=${LORA_SCALE}"
echo "STEPS=${STEPS}"
echo "GUIDANCE_SCALE=${GUIDANCE_SCALE}"
echo "NUM_FRAMES=${NUM_FRAMES}"
echo "AUTO_NUM_FRAMES_FROM_WINDOW=${AUTO_NUM_FRAMES_FROM_WINDOW}"
echo "FPS_OUT=${FPS_OUT}"
echo "WINDOW_FILTER=${WINDOW_FILTER}"

nvidia-smi || true

python - <<'PY'
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _save_frame(frame, path: Path) -> None:
    if hasattr(frame, "save"):
        frame.save(path)
        return
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(arr.max(initial=0.0)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
        return
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    Image.fromarray(arr).save(path)


def _simplify_prompt(
    raw: str,
    expected_caption: str,
    max_chars: int,
    window_id: str,
    window_index: int,
    total_windows: int,
) -> str:
    txt = " ".join(raw.split())

    # Trim noisy constraint tails that often hurt composition quality.
    for marker in ("Constraints:", "Conflict focus:"):
        if marker in txt:
            txt = txt.split(marker, 1)[0].strip()

    fields: Dict[str, str] = {}
    for key in ("Beat", "Objective", "Emotion", "Characters", "Location"):
        m = re.search(rf"{key}:\s*([^.]*)", txt)
        if m:
            fields[key] = m.group(1).strip()

    parts: List[str] = []
    if fields.get("Beat"):
        parts.append(fields["Beat"])
    if fields.get("Objective"):
        parts.append(f"Objective: {fields['Objective']}")
    if fields.get("Emotion"):
        parts.append(f"Emotion: {fields['Emotion']}")
    if fields.get("Characters"):
        parts.append(f"Characters: {fields['Characters']}")
    if fields.get("Location"):
        parts.append(f"Location: {fields['Location']}")

    if not parts:
        parts = [txt]

    progress = ""
    m_prog = re.search(r"Progress within beat:\s*([0-9.]+)", expected_caption)
    if m_prog:
        progress = f"Beat progress marker: {m_prog.group(1)}."

    style = (
        "Ghibli-inspired hand-painted cinematic frame. Stable subject focus. "
        "Clear foreground character. Coherent background. Natural daylight. High detail."
    )
    continuity = (
        f"Window {window_id} ({window_index + 1}/{total_windows}). "
        "Keep character identity and location continuity from previous window."
    )
    cap = f"Expected moment: {expected_caption.strip()}" if expected_caption.strip() else ""
    prompt = " ".join(part for part in [". ".join(parts) + ".", cap, progress, continuity, style] if part).strip()

    if len(prompt) > max_chars:
        prompt = prompt[:max_chars].rsplit(" ", 1)[0].strip()
    return prompt


def _run_one(
    pipe: CogVideoXPipeline,
    prompt: str,
    out_path: Path,
    frames_dir: Optional[Path],
    num_frames: int,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    fps_out: int,
    seed: int,
    negative_prompt: str,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kwargs = dict(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        negative_prompt=negative_prompt,
    )
    try:
        result = pipe(**kwargs)
    except TypeError:
        kwargs.pop("negative_prompt", None)
        result = pipe(**kwargs)

    frames = result.frames[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out_path), fps=fps_out)

    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):
            _save_frame(frame, frames_dir / f"frame_{idx:04d}.png")


def main() -> int:
    project_root = Path(_env("PROJECT_ROOT", str(Path.cwd()))).resolve()
    manifest_path = Path(_env("MANIFEST_PATH", str(project_root / "outputs/video/debug/generation_manifest.json")))
    model_id = _env("MODEL_ID", "THUDM/CogVideoX-2b")
    lora_dir = Path(_env("LORA_DIR", str(project_root / "outputs/lora/ghibli_cogvideox_lora_smoke")))
    lora_weight = _env("LORA_WEIGHT", "pytorch_lora_weights.safetensors")

    num_frames = _env_int("NUM_FRAMES", 0)
    auto_num_frames = _env_int("AUTO_NUM_FRAMES_FROM_WINDOW", 1) == 1
    steps = _env_int("STEPS", 36)
    guidance_scale = _env_float("GUIDANCE_SCALE", 5.5)
    height = _env_int("HEIGHT", 320)
    width = _env_int("WIDTH", 576)
    fps_out = _env_int("FPS_OUT", 8)
    lora_scale = _env_float("LORA_SCALE", 0.30)
    seed_base = _env_int("SEED_BASE", 1234)
    prompt_max_chars = _env_int("PROMPT_MAX_CHARS", 280)
    save_frames = _env_int("SAVE_FRAMES", 1) == 1
    run_baseline_check = _env_int("RUN_BASELINE_CHECK", 1) == 1
    baseline_window_id = _env("BASELINE_WINDOW_ID", "w_000")
    window_filter_raw = _env("WINDOW_FILTER", "").strip()
    negative_prompt = _env(
        "NEGATIVE_PROMPT",
        "dark, underexposed, blurry, low contrast, artifacts, ghosting, duplicate people, deformed anatomy",
    )

    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if not (lora_dir / lora_weight).exists():
        raise FileNotFoundError(f"missing LoRA weights: {lora_dir / lora_weight}")

    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
    assert torch.cuda.is_available(), "GPU not visible in this job"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = sorted(manifest["jobs"], key=lambda item: int(item["window_index"]))

    if num_frames <= 0 and auto_num_frames:
        window_seconds = float(manifest.get("window_seconds", 4))
        num_frames = max(8, int(round(window_seconds * fps_out)))
        print(
            f"[INFO] auto num_frames from manifest: window_seconds={window_seconds}, "
            f"fps_out={fps_out} => num_frames={num_frames}"
        )
    elif num_frames <= 0:
        num_frames = 32
        print(f"[INFO] fallback num_frames={num_frames}")

    if window_filter_raw:
        wanted = {token.strip() for token in window_filter_raw.split(",") if token.strip()}
        jobs = [job for job in jobs if job["window_id"] in wanted]

    if not jobs:
        raise RuntimeError("no jobs selected from manifest")

    pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")

    if run_baseline_check:
        baseline_job = next((job for job in jobs if job["window_id"] == baseline_window_id), None)
        if baseline_job is not None:
            raw_prompt = Path(baseline_job["scene_prompt_file"]).read_text(encoding="utf-8")
            expected_caption = str(baseline_job.get("expected_caption", "")).strip()
            if not expected_caption and baseline_job.get("expected_caption_file"):
                epath = Path(str(baseline_job["expected_caption_file"]))
                if epath.exists():
                    expected_caption = epath.read_text(encoding="utf-8").strip()
            prompt = _simplify_prompt(
                raw=raw_prompt,
                expected_caption=expected_caption,
                max_chars=prompt_max_chars,
                window_id=str(baseline_job.get("window_id", baseline_window_id)),
                window_index=int(baseline_job.get("window_index", 0)),
                total_windows=len(jobs),
            )
            baseline_out = Path(baseline_job["output_video_path"]).with_name("generated_base.mp4")
            baseline_frames = (
                Path(baseline_job["frames_dir"]).parent / "frames_base"
                if save_frames
                else None
            )
            print(f"[INFO] baseline check window={baseline_window_id} out={baseline_out}")
            _run_one(
                pipe=pipe,
                prompt=prompt,
                out_path=baseline_out,
                frames_dir=baseline_frames,
                num_frames=num_frames,
                steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                fps_out=fps_out,
                seed=seed_base,
                negative_prompt=negative_prompt,
            )

    pipe.load_lora_weights(str(lora_dir), weight_name=lora_weight)
    pipe.fuse_lora(lora_scale=lora_scale)

    rows = []
    for idx, job in enumerate(jobs):
        wid = job["window_id"]
        prompt_file = Path(job["scene_prompt_file"])
        out_path = Path(job["output_video_path"])
        frames_dir = Path(job["frames_dir"]) if save_frames else None
        seed = seed_base + idx
        try:
            raw_prompt = prompt_file.read_text(encoding="utf-8")
            expected_caption = str(job.get("expected_caption", "")).strip()
            if not expected_caption and job.get("expected_caption_file"):
                epath = Path(str(job["expected_caption_file"]))
                if epath.exists():
                    expected_caption = epath.read_text(encoding="utf-8").strip()
            prompt = _simplify_prompt(
                raw=raw_prompt,
                expected_caption=expected_caption,
                max_chars=prompt_max_chars,
                window_id=wid,
                window_index=int(job["window_index"]),
                total_windows=len(jobs),
            )
            _run_one(
                pipe=pipe,
                prompt=prompt,
                out_path=out_path,
                frames_dir=frames_dir,
                num_frames=num_frames,
                steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                fps_out=fps_out,
                seed=seed,
                negative_prompt=negative_prompt,
            )
            print(f"[OK] window={wid} out={out_path}")
            rows.append(
                {
                    "window_id": wid,
                    "window_index": job["window_index"],
                    "status": "generated",
                    "output_video_path": str(out_path),
                    "seed": seed,
                    "prompt_file": str(prompt_file),
                    "expected_caption": expected_caption,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAILED] window={wid}: {exc}")
            rows.append(
                {
                    "window_id": wid,
                    "window_index": job["window_index"],
                    "status": "failed",
                    "output_video_path": str(out_path),
                    "error": str(exc),
                }
            )

    generated = sum(1 for row in rows if row["status"] == "generated")
    failed = sum(1 for row in rows if row["status"] == "failed")
    status_payload = {
        "version": 1,
        "story_id": manifest.get("story_id", "unknown"),
        "total_windows": len(rows),
        "generated_windows": generated,
        "failed_windows": failed,
        "rows": rows,
        "config": {
            "model_id": model_id,
            "lora_dir": str(lora_dir),
            "lora_weight": lora_weight,
            "lora_scale": lora_scale,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "num_frames": num_frames,
            "duration_seconds": float(num_frames) / float(fps_out),
            "height": height,
            "width": width,
            "fps_out": fps_out,
            "prompt_max_chars": prompt_max_chars,
        },
    }

    status_path = project_root / "outputs/video/metrics/generation_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[SUMMARY] generated={generated} failed={failed} status={status_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
PY
