import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from director_llm import SceneDirectorConfig
from director_llm.scene_director import SceneWindow
from director_llm.stateful_scene_director import StatefulPromptBundle, StatefulSceneDirector, WindowState


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Storyline -> structured continuity state -> Wan clip windows.")
    parser.add_argument("--storyline", type=str, required=True, help="Full storyline or plot text.")
    parser.add_argument("--output_dir", type=str, default="outputs/story_run_stateful", help="Run output directory.")
    parser.add_argument("--total_minutes", type=float, default=0.5, help="Target video length in minutes.")
    parser.add_argument("--window_seconds", type=int, default=10, help="Seconds per clip window.")
    parser.add_argument("--director_model_id", type=str, default="", help="Optional HF LLM id for director.")
    parser.add_argument("--director_temperature", type=float, default=0.7, help="Director LLM temperature.")
    parser.add_argument(
        "--director_max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens per director refinement call.",
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
        help="Default shot-plan preset used when state JSON omits useful camera cues.",
    )
    parser.add_argument(
        "--style_prefix",
        type=str,
        default="cinematic realistic, coherent motion, stable camera, steady composition, smooth natural movement, high detail",
        help="Global style and quality prefix prepended to each generation prompt.",
    )
    parser.add_argument(
        "--character_lock",
        type=str,
        default="keep the same named characters, face ids, wardrobe, and key props across windows",
        help="Global continuity constraints added to every prompt.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "blurry, low quality, flicker, frame jitter, camera shake, erratic motion, abrupt acceleration, pose popping, temporal warping, deformed anatomy, duplicate subjects, extra limbs, "
            "extra animals, wrong species, text, subtitles, watermark, logo, collage, split-screen, glitch"
        ),
        help="Negative prompt passed into the video generator.",
    )
    parser.add_argument("--video_model_id", type=str, default="Wan-AI/Wan2.0-T2V-14B", help="Wan model id.")
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
        help="Seed scheduling across windows.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only plan/refine prompts. No video generation.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    clips_dir = out_dir / "clips"
    state_dir = out_dir / "window_states"
    ensure_dir(out_dir)
    ensure_dir(clips_dir)
    ensure_dir(state_dir)

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

    windows = director.plan_windows(
        storyline=args.storyline,
        total_minutes=args.total_minutes,
    )

    backbone = None
    if not args.dry_run:
        from video_backbone import WanBackbone, WanBackboneConfig

        backbone = WanBackbone(
            WanBackboneConfig(
                model_id=args.video_model_id,
                torch_dtype=args.dtype,
                device=args.device,
                enable_cpu_offload=not args.no_cpu_offload,
            )
        )
        backbone.load()

    previous_prompt = ""
    previous_window_state: Optional[WindowState] = None
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
        generation_prompt = build_generation_prompt(
            window=window,
            refined_prompt=bundle.prompt_text,
            window_state=bundle.window_state,
            style_prefix=args.style_prefix,
            character_lock=args.character_lock,
            scene_change_requested=scene_change_requested,
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
                candidate_seed = None
            elif args.seed_strategy == "window_offset":
                candidate_seed = args.seed + window.index
            else:
                candidate_seed = args.seed
            frames = backbone.generate_clip(
                prompt=generation_prompt,
                negative_prompt=args.negative_prompt,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                seed=candidate_seed,
            )
            backbone.save_video(frames=frames, output_path=clip_path.as_posix(), fps=args.fps)
            row["generated"] = True
            row["seed"] = candidate_seed

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
        "video_model_id": None if args.dry_run else args.video_model_id,
        "output_dir": out_dir.as_posix(),
        "window_state_dir": state_dir.as_posix(),
        "run_log": log_path.as_posix(),
        "structured_state": True,
        "seed": args.seed,
        "seed_strategy": args.seed_strategy,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] windows: {len(windows)}")
    print(f"[done] logs: {log_path.as_posix()}")
    print(f"[done] state_json_dir: {state_dir.as_posix()}")


if __name__ == "__main__":
    main()
