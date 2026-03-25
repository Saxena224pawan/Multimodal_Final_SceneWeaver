#!/usr/bin/env python3
"""Run SceneWeaver with multi-agent refinement on top of the active pipeline components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from director_llm import SceneDirector, SceneDirectorConfig
from memory_module import VisionEmbedder, VisionEmbedderConfig
from memory_module.captioner import Captioner, CaptionerConfig
from multi_agent_refinement.integrator import MultiAgentPipelineIntegrator
from scripts.run_story_pipeline import (
    _build_story_state_hint,
    _compact_previous_prompt,
    _conversation_progress_instruction,
    _extract_character_names,
    _load_window_plan,
    _merge_character_lock,
    _reference_strength_for_window,
    _same_scene_as_previous,
    _scene_change_requested,
    _should_use_dialogue_staging,
    _story_progress_instruction,
    _window_character_lock,
    _window_environment_anchor,
    build_generation_prompt,
)


def load_video_backbone() -> tuple[Any, Any]:
    try:
        from video_backbone import WanBackbone, WanBackboneConfig
    except Exception:
        from video_backbone.wan_backbone import WanBackbone, WanBackboneConfig
    return WanBackbone, WanBackboneConfig


def maybe_init_embedder(
    backend: str,
    model_id: Optional[str],
    device: str,
) -> Optional[VisionEmbedder]:
    if backend == "none":
        return None
    cfg = VisionEmbedderConfig(
        backend=backend,
        model_id=model_id,
        device=device,
    )
    embedder = VisionEmbedder(cfg)
    embedder.load()
    return embedder


def _default_local_director_model_id() -> Optional[str]:
    candidates = [
        PROJECT_ROOT / "LLM_MODEL" / "Qwen2.5-3B-Instruct",
        Path("/home/vault/v123be/v123be36/Multimodal_Final_SceneWeaver/LLM_MODEL/Qwen2.5-3B-Instruct"),
        Path("/home/vault/v123be/v123be36/Multimodal_Final_SceneWeaver/models/Qwen2.5-1.5B-Instruct"),
    ]
    for path in candidates:
        if path.is_dir():
            return path.as_posix()
    return None


def _default_local_embedding_model_id(backend: str) -> Optional[str]:
    if backend != "dinov2":
        return None
    candidates = [
        Path("/home/vault/v123be/v123be36/facebook/dinov2-base"),
        Path("/home/vault/v123be/v123be36/facebook/dinov2-small"),
        PROJECT_ROOT / "Globa_Local_Emb_Feedback" / "dinov2-base",
        PROJECT_ROOT / "Globa_Local_Emb_Feedback" / "dinov2-small",
    ]
    for path in candidates:
        if path.exists():
            return path.as_posix()
    return None


def _default_initial_condition_image() -> Optional[str]:
    candidates = [
        PROJECT_ROOT / "default_start_image.png",
        PROJECT_ROOT / "thirsty_crow_start_image.png",
    ]
    for path in candidates:
        if path.is_file():
            return path.as_posix()
    return None


def load_storyline(path_or_text: str) -> str:
    candidate = Path(path_or_text)
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return path_or_text


def load_initial_condition_frames(path: str, width: int, height: int) -> List[Any]:
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height))
    return [np.asarray(image)]


class DirectorLLMAdapter:
    def __init__(self, director: SceneDirector):
        self.director = director

    def generate(self, prompt: str) -> str:
        return self.director.generate_text(prompt)


def _resolve_agent_controls(args: argparse.Namespace) -> Dict[str, bool]:
    only_flags = {
        "continuity": bool(args.only_continuity),
        "storybeats": bool(args.only_storybeats),
        "physics": bool(args.only_physics),
    }
    if sum(1 for enabled in only_flags.values() if enabled) > 1:
        raise ValueError("Use at most one of --only-continuity, --only-storybeats, or --only-physics.")

    enabled_agents = {
        "continuity": True,
        "storybeats": True,
        "physics": True,
    }
    if only_flags["continuity"]:
        enabled_agents = {"continuity": True, "storybeats": False, "physics": False}
    elif only_flags["storybeats"]:
        enabled_agents = {"continuity": False, "storybeats": True, "physics": False}
    elif only_flags["physics"]:
        enabled_agents = {"continuity": False, "storybeats": False, "physics": True}

    if args.disable_continuity:
        enabled_agents["continuity"] = False
    if args.disable_storybeats:
        enabled_agents["storybeats"] = False
    if args.disable_physics:
        enabled_agents["physics"] = False

    if not any(enabled_agents.values()):
        raise ValueError("At least one refinement agent must remain enabled.")

    return enabled_agents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SceneWeaver with multi-agent refinement on real pipeline components."
    )
    parser.add_argument("--storyline", required=True, help="Path to storyline file or inline storyline text.")
    parser.add_argument("--window-plan-json", default="", help="Optional JSON file with authored window beats.")
    parser.add_argument("--output-dir", default="outputs/multi_agent_run", help="Directory for clips and metadata.")

    parser.add_argument("--video-model-id", default="", help="Local Wan diffusers model directory.")
    parser.add_argument("--director-model-id", default="", help="Local director LLM directory.")
    parser.add_argument("--director-temperature", type=float, default=0.7)
    parser.add_argument("--director-max-new-tokens", type=int, default=512)
    parser.add_argument("--director-do-sample", action="store_true")
    parser.add_argument("--shot-plan-defaults", default="cinematic", choices=["cinematic", "docu", "action"])

    parser.add_argument("--embedder-backend", default="dinov2", choices=["clip", "dinov2", "none"])
    parser.add_argument("--embedding-model-id", default="", help="Local embedding model directory.")
    parser.add_argument("--captioner-model-id", default="__stub__", help="Captioner model id or __stub__.")
    parser.add_argument("--captioner-device", default="cpu", help="Captioner device (cpu/cuda).")

    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--quality-threshold", type=float, default=0.74)
    parser.add_argument("--convergence-patience", type=int, default=2)
    parser.add_argument("--convergence-tolerance", type=float, default=0.015)
    parser.add_argument("--progressive-tightening", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tightening-strength", type=float, default=0.8)
    parser.add_argument("--disable-continuity", action="store_true", help="Run ablation without ContinuityAuditor.")
    parser.add_argument("--disable-storybeats", action="store_true", help="Run ablation without StorybeatsChecker.")
    parser.add_argument("--disable-physics", action="store_true", help="Run ablation without PhysicsValidator.")
    parser.add_argument("--only-continuity", action="store_true", help="Run only ContinuityAuditor.")
    parser.add_argument("--only-storybeats", action="store_true", help="Run only StorybeatsChecker.")
    parser.add_argument("--only-physics", action="store_true", help="Run only PhysicsValidator.")
    parser.add_argument("--num-test-stories", type=int, default=None, help="Limit the number of windows for testing.")

    parser.add_argument(
        "--total-minutes",
        type=float,
        default=0.8,
        help="Target video length in minutes when using fixed window planning.",
    )
    parser.add_argument("--window-seconds", type=int, default=8)
    parser.add_argument(
        "--window-count-mode",
        default="dynamic",
        choices=["dynamic", "fixed"],
        help="How to decide the number of windows: dynamic uses story length, fixed uses total-minutes/window-seconds.",
    )
    parser.add_argument(
        "--target-words-per-window",
        type=int,
        default=28,
        help="Story-length pacing target used only in dynamic window planning.",
    )
    parser.add_argument("--min-dynamic-windows", type=int, default=1)
    parser.add_argument("--max-dynamic-windows", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=65)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=5.4)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--reference-conditioning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-tail-frames", type=int, default=4)
    parser.add_argument("--reference-strength", type=float, default=0.68)
    parser.add_argument("--noise-conditioning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise-blend-strength", type=float, default=0.2)
    parser.add_argument("--disable-random-generation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--initial-condition-image", default="", help="Starter image for image-conditioned generation.")

    parser.add_argument(
        "--style-prefix",
        default="cinematic realistic, coherent motion, expressive actions, stable continuity, high detail",
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, flicker, frame jitter, duplicate subjects, identity drift, watermark, text, logo, collage",
    )
    parser.add_argument(
        "--character-lock",
        default="",
        help="Global continuity lock appended to each window prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if int(args.max_iterations) < 1:
        raise ValueError("--max-iterations must be >= 1")
    if int(args.convergence_patience) < 0:
        raise ValueError("--convergence-patience must be >= 0")
    if not 0.0 <= float(args.convergence_tolerance) <= 1.0:
        raise ValueError("--convergence-tolerance must be between 0.0 and 1.0")
    if float(args.tightening_strength) < 0.0:
        raise ValueError("--tightening-strength must be >= 0.0")

    agent_controls = _resolve_agent_controls(args)
    active_agents = [name for name, enabled in agent_controls.items() if enabled]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    storyline_text = load_storyline(args.storyline)
    if not args.director_model_id:
        args.director_model_id = _default_local_director_model_id() or ""
    if args.embedder_backend == "dinov2" and not args.embedding_model_id:
        args.embedding_model_id = _default_local_embedding_model_id("dinov2") or ""
    if args.embedder_backend == "clip" and not args.embedding_model_id:
        print("[multi-agent] no local CLIP model configured; disabling embedder backend.")
        args.embedder_backend = "none"
    if not args.initial_condition_image:
        args.initial_condition_image = _default_initial_condition_image() or ""

    print("\n" + "=" * 70)
    print("SceneWeaver Pipeline with Multi-Agent Refinement")
    print("=" * 70 + "\n")
    print(f"Story length: {len(storyline_text)} characters")
    print(f"Output dir: {output_dir}")
    print(f"Quality threshold: {args.quality_threshold:.2f}")
    print(f"Max iterations/window: {args.max_iterations}")
    print(f"Active agents: {', '.join(active_agents)}")
    print(f"Convergence patience/tolerance: {args.convergence_patience}/{args.convergence_tolerance}")
    print(f"Progressive tightening: {args.progressive_tightening} (strength={args.tightening_strength})")
    print(f"Window planning mode: {args.window_count_mode}")

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
    llm_model = DirectorLLMAdapter(director)

    WanBackbone, WanBackboneConfig = load_video_backbone()
    backbone = WanBackbone(
        WanBackboneConfig(
            model_id=args.video_model_id or WanBackboneConfig().model_id,
            torch_dtype=args.dtype,
            device=args.device,
            enable_cpu_offload=True,
        )
    )
    backbone.load()

    if backbone.requires_reference_conditioning() and not args.reference_conditioning:
        raise RuntimeError(
            "Selected video model requires image conditioning. Enable --reference-conditioning and provide "
            "--initial-condition-image for the first window."
        )

    initial_condition_frames: Optional[List[Any]] = None
    if args.reference_conditioning and args.initial_condition_image:
        initial_condition_frames = load_initial_condition_frames(
            args.initial_condition_image,
            width=args.width,
            height=args.height,
        )
    elif args.reference_conditioning and backbone.requires_reference_conditioning():
        raise RuntimeError(
            "reference conditioning is enabled implicitly by the selected Wan I2V model, but no "
            "--initial-condition-image was provided and no default starter image was found."
        )

    embedder = None
    if args.embedder_backend != "none":
        embedder = maybe_init_embedder(
            backend=args.embedder_backend,
            model_id=args.embedding_model_id or None,
            device=args.device,
        )

    captioner = Captioner(
        CaptionerConfig(
            model_id=args.captioner_model_id or "__stub__",
            device=args.captioner_device,
            stub_fallback=True,
        )
    )
    captioner.load()

    integrator = MultiAgentPipelineIntegrator(
        video_model=backbone,
        captioner=captioner,
        embedding_model=embedder,
        llm_model=llm_model,
        output_dir=output_dir.as_posix(),
        max_iterations=args.max_iterations,
        quality_threshold=args.quality_threshold,
        convergence_patience=args.convergence_patience,
        convergence_tolerance=args.convergence_tolerance,
        progressive_tightening=bool(args.progressive_tightening),
        tightening_strength=args.tightening_strength,
        enable_continuity=agent_controls["continuity"],
        enable_storybeats=agent_controls["storybeats"],
        enable_physics=agent_controls["physics"],
    )

    beats_override = _load_window_plan(args.window_plan_json) if args.window_plan_json else None
    windows = director.plan_windows(
        storyline=storyline_text,
        total_minutes=args.total_minutes,
        beats_override=beats_override,
    )
    if args.num_test_stories is not None:
        windows = windows[: max(1, int(args.num_test_stories))]

    planned_runtime_seconds = len(windows) * args.window_seconds
    window_count_source = "window_plan_json" if beats_override else args.window_count_mode
    print(
        f"Planned {len(windows)} windows ({planned_runtime_seconds}s runtime) "
        f"via {window_count_source} planning\n"
    )
    print("=" * 70)
    print("GENERATING WINDOWS WITH MULTI-AGENT REFINEMENT")
    print("=" * 70 + "\n")

    previous_full_frames: Optional[List[Any]] = None
    previous_reference_frames: Optional[List[Any]] = None
    previous_prompt = ""
    previous_scene_conversation = ""
    previous_environment_anchor = ""
    scene_anchor_frames_by_key: Dict[str, List[Any]] = {}
    all_results: List[Dict[str, Any]] = []

    for idx, window in enumerate(windows):
        print(f"\n[Window {idx + 1}/{len(windows)}] {window.beat[:70]}...")
        previous_window = windows[idx - 1] if idx > 0 else None
        next_window = windows[idx + 1] if idx + 1 < len(windows) else None

        scene_change_requested = _scene_change_requested(previous_window, window)
        same_scene_as_previous = _same_scene_as_previous(previous_window, window)
        current_environment_anchor = _window_environment_anchor(window) or previous_environment_anchor
        effective_character_lock = _merge_character_lock(args.character_lock, _window_character_lock(window))
        scene_key = (getattr(window, "scene_id", "") or current_environment_anchor or "global_scene").strip() or "global_scene"
        story_state_hint = _build_story_state_hint(windows, idx)
        previous_beat = previous_window.beat if previous_window is not None else ""
        next_beat = next_window.beat if next_window is not None else ""
        character_names = _extract_character_names(storyline_text, window.beat)

        bundle = director.refine_prompt(
            storyline=storyline_text,
            window=window,
            previous_prompt=previous_prompt,
            previous_scene_conversation=previous_scene_conversation,
            memory_feedback=None,
        )
        refined_prompt = bundle.prompt_text
        dialogue_scene = _should_use_dialogue_staging(
            scene_conversation=bundle.scene_conversation,
            beat=window.beat,
            character_names=character_names,
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
            scene_conversation=bundle.scene_conversation,
            previous_scene_conversation=previous_scene_conversation,
            conversation_progress_instruction=(
                _conversation_progress_instruction(
                    previous_scene_conversation,
                    window.beat,
                    next_beat,
                    scene_change_requested=scene_change_requested,
                )
                if dialogue_scene
                else ""
            ),
            story_progress_instruction=_story_progress_instruction(previous_beat, window.beat),
            dialogue_scene=dialogue_scene,
            shot_plan=bundle.shot_plan,
            shot_plan_enforce=True,
        )

        reference_frames = None
        reference_source = "none"
        if args.reference_conditioning and previous_reference_frames:
            reference_frames = previous_reference_frames
            reference_source = "previous_window_tail"
        elif args.reference_conditioning and initial_condition_frames is not None:
            reference_frames = initial_condition_frames
            reference_source = "initial_condition_image"

        scene_anchor_frames = scene_anchor_frames_by_key.get(scene_key)
        if scene_anchor_frames is not None and not scene_change_requested:
            generation_prompt = (
                f"{generation_prompt} Keep the exact same background landmarks, prop placement, and lighting from the established scene anchor."
            )

        generation_kwargs = {
            "negative_prompt": args.negative_prompt,
            "num_frames": args.num_frames,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "seed": None if args.seed is None else int(args.seed) + (window.index * 1000),
            "reference_frames": reference_frames,
            "reference_strength": _reference_strength_for_window(
                base_strength=float(args.reference_strength),
                scene_change_requested=scene_change_requested,
                reference_source=reference_source,
                same_scene_as_previous=same_scene_as_previous,
            ),
            "reference_source": reference_source,
            "use_noise_conditioning": bool(args.noise_conditioning),
            "noise_blend_strength": float(args.noise_blend_strength),
            "disable_random_generation": bool(args.disable_random_generation),
        }

        frames, metadata = integrator.generate_window(
            base_prompt=generation_prompt,
            narrative_beat=window.beat,
            window_idx=window.index,
            previous_frames=previous_full_frames,
            character_names=character_names,
            scene_location=current_environment_anchor or None,
            scene_anchor_frames=scene_anchor_frames,
            generation_kwargs=generation_kwargs,
        )

        if frames is None:
            print(f"  ❌ Generation failed for window {window.index}")
            continue

        integrator.save_metadata(metadata)
        clip_path = clips_dir / f"window_{window.index:03d}.mp4"
        backbone.save_video(frames, clip_path, fps=args.fps)

        final_score = metadata.scores_history[-1] if metadata.scores_history else 0.0
        print(f"  ✓ Generated in {metadata.total_iterations} iteration(s)")
        print(f"  ✓ Final score: {final_score:.3f}")
        print(f"  ✓ Time: {metadata.generation_time:.1f}s")
        print(f"  ✓ Clip: {clip_path}")

        all_results.append(
            {
                "window_idx": window.index,
                "beat": window.beat,
                "iterations": metadata.total_iterations,
                "final_score": final_score,
                "final_prompt": metadata.final_prompt,
                "clip_path": clip_path.as_posix(),
                "reference_source": reference_source,
                "enabled_agents": agent_controls,
            }
        )

        previous_full_frames = list(frames)
        if scene_key not in scene_anchor_frames_by_key and frames is not None and len(frames) > 0:
            scene_anchor_frames_by_key[scene_key] = [frames[0]]
        if args.reference_conditioning and frames is not None and len(frames) > 0:
            tail_count = max(1, int(args.reference_tail_frames))
            previous_reference_frames = list(frames[-tail_count:])
        else:
            previous_reference_frames = None
        previous_prompt = _compact_previous_prompt(metadata.final_prompt or refined_prompt)
        previous_scene_conversation = bundle.scene_conversation if dialogue_scene else ""
        previous_environment_anchor = current_environment_anchor

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70 + "\n")

    integrator.save_summary()
    stats = integrator.get_convergence_stats()
    print("📊 Convergence Statistics:")
    print(f"  Total windows: {stats['total_windows']}")
    print(f"  Avg iterations/window: {stats['avg_iterations']:.2f}")
    print(f"  Max iterations: {stats['max_iterations']}")
    print(f"  Avg final score: {stats['avg_final_score']:.3f}")
    print(f"  Windows at threshold: {stats['windows_at_threshold']}/{stats['total_windows']}")

    results_file = output_dir / "results.json"
    with results_file.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "total_windows": len(all_results),
                "convergence_stats": stats,
                "enabled_agents": agent_controls,
                "windows": all_results,
            },
            handle,
            indent=2,
        )

    try:
        from evaluation import generate_convergence_report

        report = generate_convergence_report(output_dir)
        report_path = output_dir / "convergence_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"  ✓ Convergence report: {report_path}")
    except Exception as exc:
        print(f"  ⚠️  Could not generate convergence report automatically: {exc}")

    print(f"\n✅ All results saved to: {output_dir}")
    print(f"   - Metadata: {output_dir / 'metadata'}")
    print(f"   - Summary: {output_dir / 'summary.json'}")
    print(f"   - Results: {results_file}")


if __name__ == "__main__":
    main()
