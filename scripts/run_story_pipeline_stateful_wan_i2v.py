import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from director_llm import SceneDirectorConfig
from director_llm.scene_director import SceneWindow
from director_llm.stateful_scene_director import StatefulPromptBundle, StatefulSceneDirector, WindowState
from video_backbone import WanBackbone, WanBackboneConfig


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_jsonl(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compact_previous_prompt(prompt: str) -> str:
    if not prompt:
        return ""
    head = prompt.split(" Previous visual context:")[0].strip()
    return head[:240]


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


def build_generation_prompt(
    *,
    window: SceneWindow,
    refined_prompt: str,
    window_state: WindowState,
    style_prefix: str,
    character_lock: str,
    scene_change_requested: bool,
    has_reference_frames: bool,
) -> str:
    character_lines = []
    for name, char in window_state.characters.items():
        character_lines.append(
            f"{name} [{char.face_id}] hair {char.hair}, outfit {char.outfit}, emotion {char.emotion}."
        )
    must_preserve = ", ".join(window_state.continuity.must_preserve)
    parts = []
    if style_prefix.strip():
        parts.append(style_prefix.strip())
    if character_lock.strip():
        parts.append(f"Character continuity: {character_lock.strip()}")
    if character_lines:
        parts.append(f"Characters: {' '.join(character_lines)}")
    parts.append(
        "Location: "
        f"{window_state.location.place}, {window_state.location.time}, weather {window_state.location.weather}, "
        f"lighting {window_state.location.lighting}."
    )
    parts.append(
        f"Camera: lens {window_state.camera.lens}, height {window_state.camera.height}, movement {window_state.camera.movement}."
    )
    parts.append(f"Current beat: {window.beat}")
    parts.append(f"Previous action: {window_state.continuity.previous_action}")
    if must_preserve:
        parts.append(f"Must preserve: {must_preserve}")
    if scene_change_requested:
        parts.append("Scene change is allowed, but carry key props, wardrobe, and subject identity through the transition.")
    else:
        parts.append("Preserve the established scene layout, props, and camera continuity.")
    if has_reference_frames:
        parts.append(
            "The attached reference frames come from the immediately previous generated window. "
            "Match character identity, wardrobe, key props, and anchor object placement while advancing the current beat."
        )
    parts.append(
        "Motion rule: one continuous primary action only, smooth physically plausible movement, stable body proportions, "
        "stable subject scale, no abrupt acceleration, no sudden camera whip, no jitter, no pose popping."
    )
    parts.append(
        "Camera rule: keep the same lens, height, and movement across the whole clip; if moving, move slowly and evenly."
    )
    parts.append(f"Shot prompt: {refined_prompt}")
    parts.append(
        "Render one coherent clip for this window only. No extra characters. No unrelated objects. No abrupt identity drift. "
        "Favor steady composition over dramatic motion."
    )
    return " ".join(parts)


def _to_uint8_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported frame shape for image conditioning: {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype.kind in ("f", "c"):
        f_min = float(np.min(arr))
        f_max = float(np.max(arr))
        if 0.0 <= f_min and f_max <= 1.0:
            arr = (arr * 255.0).round().astype(np.uint8)
        elif -1.1 <= f_min and f_max <= 1.1:
            arr = (((arr + 1.0) / 2.0) * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def _frame_to_pil(frame: Any):
    from PIL import Image

    return Image.fromarray(_to_uint8_frame(frame), mode="RGB")


def _build_last3_conditioning(frames: List[Any], tail_frame_count: int) -> Tuple[Any, Any, List[Any]]:
    tail = frames[-max(1, tail_frame_count) :]
    pil_tail = [_frame_to_pil(frame) for frame in tail]
    tail_arrays = [np.asarray(img, dtype=np.float32) for img in pil_tail]
    weights = np.linspace(1.0, 2.0, num=len(tail_arrays), dtype=np.float32)
    weights = weights / weights.sum()
    blended = np.zeros_like(tail_arrays[0], dtype=np.float32)
    for weight, arr in zip(weights, tail_arrays):
        blended += weight * arr
    blended = np.clip(blended.round(), 0, 255).astype(np.uint8)

    from PIL import Image

    condition_image = Image.fromarray(blended, mode="RGB")
    last_image = pil_tail[-1]
    return condition_image, last_image, pil_tail


class WanI2VBackbone:
    def __init__(self, model_id: str, torch_dtype: str = "bfloat16", device: str = "auto", enable_cpu_offload: bool = True):
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device = device
        self.enable_cpu_offload = enable_cpu_offload
        self.pipeline = None

    def _get_torch_dtype(self, torch_module: Any):
        name = self.torch_dtype.lower()
        if name == "float16":
            return torch_module.float16
        if name == "float32":
            return torch_module.float32
        if name == "bfloat16":
            return torch_module.bfloat16
        raise ValueError(f"Unsupported torch_dtype={self.torch_dtype}")

    def load(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for Wan I2V.") from exc
        try:
            from diffusers import AutoPipelineForImage2Video as PipelineClass
        except Exception:
            from diffusers import WanImageToVideoPipeline as PipelineClass

        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if self.device == "auto":
            if cuda_available:
                resolved_device = "cuda"
            elif mps_available:
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = self.device
        if resolved_device == "cuda" and not cuda_available:
            resolved_device = "mps" if mps_available else "cpu"

        is_apple_silicon = platform.system() == "Darwin" and platform.machine().startswith("arm")
        if is_apple_silicon and resolved_device != "cuda" and "14b" in self.model_id.lower():
            raise RuntimeError("Wan I2V 14B is not a practical local target on Apple Silicon/MPS.")

        torch_dtype = self._get_torch_dtype(torch)
        pipe = PipelineClass.from_pretrained(self.model_id, torch_dtype=torch_dtype)
        WanBackbone._fix_text_encoder_embedding_tie(pipe)
        if self.enable_cpu_offload and resolved_device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(resolved_device)
        self.pipeline = pipe

    def generate_clip(
        self,
        *,
        image: Any,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 81,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        height: int = 480,
        width: int = 832,
        seed: Optional[int] = None,
        last_image: Optional[Any] = None,
    ) -> List[Any]:
        if self.pipeline is None:
            raise RuntimeError("Wan I2V pipeline is not loaded. Call load() first.")

        generator = None
        if seed is not None:
            import torch

            generator = torch.Generator(device="cpu").manual_seed(seed)

        result = self.pipeline(
            image=image,
            last_image=last_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            output_type="np",
        )
        frames = getattr(result, "frames", None)
        if frames is None:
            raise RuntimeError("Wan I2V output does not contain `frames`.")
        clip_frames = WanBackbone._select_single_clip(frames)
        if len(clip_frames) == 0:
            raise RuntimeError("Selected Wan I2V clip is empty.")
        return clip_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Stateful Wan T2V->I2V pipeline using the last 3 frames as continuity anchors.")
    parser.add_argument("--storyline", type=str, required=True, help="Full storyline or plot text.")
    parser.add_argument("--output_dir", type=str, default="outputs/story_run_stateful_wan_i2v", help="Run output directory.")
    parser.add_argument("--total_minutes", type=float, default=0.5, help="Target video length in minutes.")
    parser.add_argument("--window_seconds", type=int, default=10, help="Seconds per clip window.")
    parser.add_argument("--director_model_id", type=str, default="", help="Optional HF LLM id for director.")
    parser.add_argument("--director_temperature", type=float, default=0.7, help="Director LLM temperature.")
    parser.add_argument("--director_max_new_tokens", type=int, default=256, help="Max new tokens per director refinement call.")
    parser.add_argument("--director_do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shot_plan_defaults", type=str, default="cinematic", choices=["cinematic", "docu", "action"])
    parser.add_argument("--style_prefix", type=str, default="cinematic realistic, coherent motion, stable camera, steady composition, smooth natural movement, high detail")
    parser.add_argument("--character_lock", type=str, default="keep the same named characters, face ids, wardrobe, and key props across windows")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, flicker, frame jitter, camera shake, erratic motion, abrupt acceleration, pose popping, temporal warping, deformed anatomy, duplicate subjects, extra limbs, extra animals, wrong species, text, subtitles, watermark, logo, collage, split-screen, glitch")
    parser.add_argument("--t2v_model_id", type=str, required=True, help="Wan T2V model id/path for the first window.")
    parser.add_argument("--i2v_model_id", type=str, required=True, help="Wan I2V model id/path for later windows.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no_cpu_offload", action="store_true")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed_strategy", type=str, default="fixed", choices=["fixed", "window_offset"])
    parser.add_argument("--tail_frame_count", type=int, default=3, help="How many trailing frames from the previous window to use as continuity anchors.")
    parser.add_argument("--save_conditioning_frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true", help="Only plan/refine prompts. No video generation.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    clips_dir = out_dir / "clips"
    state_dir = out_dir / "window_states"
    conditioning_dir = out_dir / "conditioning_frames"
    ensure_dir(out_dir)
    ensure_dir(clips_dir)
    ensure_dir(state_dir)
    ensure_dir(conditioning_dir)

    director = StatefulSceneDirector(
        SceneDirectorConfig(
            model_id=args.director_model_id or None,
            temperature=args.director_temperature,
            max_new_tokens=args.director_max_new_tokens,
            do_sample=bool(args.director_do_sample),
            shot_plan_defaults=args.shot_plan_defaults,
        ),
        window_seconds=args.window_seconds,
    )
    director.load()
    windows = director.plan_windows(storyline=args.storyline, total_minutes=args.total_minutes)

    t2v_backbone = None
    i2v_backbone = None
    if not args.dry_run:
        t2v_backbone = WanBackbone(
            WanBackboneConfig(
                model_id=args.t2v_model_id,
                torch_dtype=args.dtype,
                device=args.device,
                enable_cpu_offload=not args.no_cpu_offload,
            )
        )
        t2v_backbone.load()
        i2v_backbone = WanI2VBackbone(
            model_id=args.i2v_model_id,
            torch_dtype=args.dtype,
            device=args.device,
            enable_cpu_offload=not args.no_cpu_offload,
        )
        i2v_backbone.load()

    previous_prompt = ""
    previous_window_state: Optional[WindowState] = None
    previous_frames: Optional[List[Any]] = None
    log_rows: List[Dict[str, Any]] = []

    for window in windows:
        bundle: StatefulPromptBundle = director.refine_prompt(
            storyline=args.storyline,
            window=window,
            previous_prompt=previous_prompt,
            memory_feedback=None,
            previous_window_state=previous_window_state,
        )
        scene_change_requested = _beat_requests_scene_change(window.beat)
        has_reference_frames = previous_frames is not None
        generation_prompt = build_generation_prompt(
            window=window,
            refined_prompt=bundle.prompt_text,
            window_state=bundle.window_state,
            style_prefix=args.style_prefix,
            character_lock=args.character_lock,
            scene_change_requested=scene_change_requested,
            has_reference_frames=has_reference_frames,
        )

        state_path = state_dir / f"window_{window.index:03d}.json"
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(bundle.window_state.to_dict(), f, ensure_ascii=False, indent=2)

        clip_path = clips_dir / f"window_{window.index:03d}.mp4"
        row: Dict[str, Any] = {
            "window_index": window.index,
            "time_range": [window.start_sec, window.end_sec],
            "beat": window.beat,
            "prompt_seed": window.prompt_seed,
            "refined_prompt": bundle.prompt_text,
            "generation_prompt": generation_prompt,
            "shot_plan": bundle.shot_plan.__dict__,
            "window_state": bundle.window_state.to_dict(),
            "window_state_path": state_path.as_posix(),
            "clip_path": clip_path.as_posix(),
            "generated": False,
        }

        if not args.dry_run:
            if args.seed is None:
                window_seed = None
            elif args.seed_strategy == "window_offset":
                window_seed = args.seed + window.index
            else:
                window_seed = args.seed

            if previous_frames is None:
                frames = t2v_backbone.generate_clip(
                    prompt=generation_prompt,
                    negative_prompt=args.negative_prompt,
                    num_frames=args.num_frames,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    seed=window_seed,
                )
                row["generation_mode"] = "t2v_prompt_only"
                row["conditioning_frame_paths"] = []
            else:
                condition_image, last_image, tail_images = _build_last3_conditioning(previous_frames, args.tail_frame_count)
                conditioning_paths: List[str] = []
                if args.save_conditioning_frames:
                    for idx, img in enumerate(tail_images, start=1):
                        ref_path = conditioning_dir / f"window_{window.index:03d}_prev_tail_{idx:02d}.png"
                        img.save(ref_path)
                        conditioning_paths.append(ref_path.as_posix())
                    blend_path = conditioning_dir / f"window_{window.index:03d}_condition_blend.png"
                    last_path = conditioning_dir / f"window_{window.index:03d}_condition_last.png"
                    condition_image.save(blend_path)
                    last_image.save(last_path)
                    conditioning_paths.extend([blend_path.as_posix(), last_path.as_posix()])
                frames = i2v_backbone.generate_clip(
                    image=condition_image,
                    last_image=last_image,
                    prompt=generation_prompt,
                    negative_prompt=args.negative_prompt,
                    num_frames=args.num_frames,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    seed=window_seed,
                )
                row["generation_mode"] = f"i2v_prev_tail_{min(args.tail_frame_count, len(previous_frames))}"
                row["conditioning_frame_paths"] = conditioning_paths

            WanBackbone.save_video(frames=frames, output_path=clip_path.as_posix(), fps=args.fps)
            row["generated"] = True
            row["seed"] = window_seed
            previous_frames = frames

        log_rows.append(row)
        previous_prompt = _compact_previous_prompt(bundle.prompt_text)
        previous_window_state = bundle.window_state
        print(f"[scene {window.index:03d}] {window.start_sec}-{window.end_sec}s ready")

    log_path = out_dir / "run_log.jsonl"
    summary_path = out_dir / "run_summary.json"
    export_jsonl(log_rows, log_path)
    summary = {
        "storyline": args.storyline,
        "total_minutes": args.total_minutes,
        "window_seconds": args.window_seconds,
        "num_windows": len(windows),
        "dry_run": args.dry_run,
        "director_model_id": args.director_model_id or None,
        "t2v_model_id": None if args.dry_run else args.t2v_model_id,
        "i2v_model_id": None if args.dry_run else args.i2v_model_id,
        "tail_frame_count": args.tail_frame_count,
        "output_dir": out_dir.as_posix(),
        "window_state_dir": state_dir.as_posix(),
        "conditioning_dir": conditioning_dir.as_posix(),
        "run_log": log_path.as_posix(),
        "structured_state": True,
        "seed": args.seed,
        "seed_strategy": args.seed_strategy,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] windows: {len(windows)}")
    print(f"[done] logs: {log_path.as_posix()}")
    print(f"[done] conditioning_dir: {conditioning_dir.as_posix()}")


if __name__ == "__main__":
    main()
