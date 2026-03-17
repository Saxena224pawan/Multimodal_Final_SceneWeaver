#!/usr/bin/env python3
"""
Multi-Agent Story Pipeline - Wrapper for SceneWeaver with Iterative Refinement

Usage:
    python scripts/run_story_pipeline_with_agents.py \
        --storyline "A hero journeys..." \
        --output-dir outputs/my_run \
        --max-iterations 3 \
        --quality-threshold 0.70

This script wraps the existing pipeline with multi-agent refinement.
Each window is evaluated by agents and refined if necessary.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from director_llm import SceneDirector, SceneDirectorConfig
from memory_module import VisionEmbedder, VisionEmbedderConfig
from memory_module.captioner import Captioner, CaptionerConfig
from multi_agent_refinement.integrator import MultiAgentPipelineIntegrator
from pipeline_runtime import get_selected_model, load_model_links


def load_video_backbone() -> tuple[Any, Any]:
    """Load video model"""
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
    """Initialize embedder if specified"""
    if backend == "none":
        return None
    try:
        cfg = VisionEmbedderConfig(
            backend=backend,
            model_id=model_id,
            device=device,
        )
    except TypeError:
        cfg = VisionEmbedderConfig(
            backend=backend,
            model_id=model_id,
        )
    embedder = VisionEmbedder(cfg)
    embedder.load()
    return embedder


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run SceneWeaver pipeline with multi-agent refinement"
    )

    # Story input
    parser.add_argument(
        "--storyline",
        required=True,
        help="Path to storyline file or storyline text",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="outputs/multi_agent_run",
        help="Output directory for results",
    )

    # Video model
    parser.add_argument(
        "--video-model-id",
        help="Video model ID (e.g., WAN path)",
    )

    # Director/LLM
    parser.add_argument(
        "--director-model-id",
        help="Director LLM model ID",
    )

    # Embedder
    parser.add_argument(
        "--embedder-backend",
        default="clip",
        choices=["clip", "dinov2", "none"],
        help="Embedder backend for continuity checks",
    )

    # Multi-agent settings
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum refinement iterations per window",
    )

    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.70,
        help="Quality score threshold to accept window",
    )

    # Pipeline settings
    parser.add_argument(
        "--num-frames",
        type=int,
        default=49,
        help="Number of frames per window",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width",
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=6.0,
        help="Diffusion guidance scale",
    )

    parser.add_argument(
        "--num-test-stories",
        type=int,
        default=None,
        help="Number of test stories to run (for quick testing)",
    )

    return parser.parse_args()


def load_storyline(path: str) -> str:
    """Load storyline from file or return as-is if text"""
    p = Path(path)
    if p.is_file():
        return p.read_text()
    return path


def main() -> None:
    """Main pipeline"""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SceneWeaver Pipeline with Multi-Agent Refinement")
    print("=" * 70 + "\n")

    # Load models
    print("Loading models...")

    # Load video model
    WanBackbone, WanBackboneConfig = load_video_backbone()
    if args.video_model_id:
        video_config = WanBackboneConfig(model_id=args.video_model_id)
    else:
        video_config = WanBackboneConfig()
    video_model = WanBackbone(video_config)
    video_model.load()
    print("  ✓ Video model loaded")

    # Load director/LLM
    director_config = SceneDirectorConfig()
    if args.director_model_id:
        director_config.model_id = args.director_model_id
    director = SceneDirector(director_config)
    print("  ✓ Director/LLM loaded")

    # Load embedder
    embedder = None
    if args.embedder_backend != "none":
        embedder = maybe_init_embedder(
            backend=args.embedder_backend,
            model_id=None,
            device="auto",
        )
        print(f"  ✓ Embedder ({args.embedder_backend}) loaded")

    # Load captioner
    caption_config = CaptionerConfig()
    captioner = Captioner(caption_config)
    captioner.load()
    print("  ✓ Captioner loaded")

    # Get LLM (from director for now)
    llm_model = director.model  # Use director's LLM

    # Initialize multi-agent integrator
    print(f"\nInitializing multi-agent system...")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Quality threshold: {args.quality_threshold}")

    integrator = MultiAgentPipelineIntegrator(
        video_model=video_model,
        captioner=captioner,
        embedding_model=embedder,
        llm_model=llm_model,
        output_dir=output_dir,
        max_iterations=args.max_iterations,
        quality_threshold=args.quality_threshold,
    )
    print("  ✓ Multi-agent system ready\n")

    # Load storyline
    print("Loading story...")
    storyline_text = load_storyline(args.storyline)
    print(f"  Story length: {len(storyline_text)} characters\n")

    # Plan windows
    print("Planning windows...")
    try:
        windows = director.plan_windows(
            storyline=storyline_text,
            target_duration_sec=60,  # 1 min demo
        )
        print(f"  ✓ Planned {len(windows)} windows\n")
    except Exception as e:
        print(f"  ❌ Window planning failed: {e}")
        return

    # Limit to test stories if specified
    if args.num_test_stories:
        windows = windows[: args.num_test_stories]
        print(f"  Limited to {len(windows)} windows for testing\n")

    # Generate windows with multi-agent refinement
    print("=" * 70)
    print("GENERATING WINDOWS WITH MULTI-AGENT REFINEMENT")
    print("=" * 70 + "\n")

    previous_frames = None
    all_results = []

    for i, window in enumerate(windows):
        print(f"\n[Window {i+1}/{len(windows)}] {window.beat[:50]}...")

        try:
            frames, metadata = integrator.generate_window(
                base_prompt=window.prompt,
                narrative_beat=window.beat,
                window_idx=i,
                previous_frames=previous_frames,
                character_names=getattr(window, "characters", None),
                scene_location=getattr(window, "scene_location", None),
            )

            if frames is None:
                print(f"  ❌ Generation failed for window {i}")
                continue

            # Save metadata
            integrator.save_metadata(metadata)

            # Save clip (mock for now - in real pipeline, encode frames to mp4)
            clip_path = clips_dir / f"window_{i:03d}.mp4"
            print(f"  ✓ Generated in {metadata.total_iterations} iteration(s)")
            print(f"  ✓ Final score: {metadata.scores_history[-1]:.3f}")
            print(f"  ✓ Time: {metadata.generation_time:.1f}s")

            result = {
                "window_idx": i,
                "beat": window.beat,
                "iterations": metadata.total_iterations,
                "final_score": metadata.scores_history[-1],
                "clip_path": clip_path.as_posix(),
            }
            all_results.append(result)

            previous_frames = frames

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()

    # Save summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70 + "\n")

    integrator.save_summary()

    # Print convergence stats
    stats = integrator.get_convergence_stats()
    print("📊 Convergence Statistics:")
    print(f"  Total windows: {stats['total_windows']}")
    print(f"  Avg iterations/window: {stats['avg_iterations']:.2f}")
    print(f"  Max iterations: {stats['max_iterations']}")
    print(f"  Avg final score: {stats['avg_final_score']:.3f}")
    print(f"  Windows at threshold: {stats['windows_at_threshold']}/{stats['total_windows']}")

    # Save results summary
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "total_windows": len(all_results),
                "convergence_stats": stats,
                "windows": all_results,
            },
            f,
            indent=2,
        )

    print(f"\n✅ All results saved to: {output_dir}")
    print(f"   - Metadata: {output_dir}/metadata/")
    print(f"   - Summary: {output_dir}/summary.json")
    print(f"   - Results: {output_dir}/results.json")


if __name__ == "__main__":
    main()
