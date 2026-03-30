#!/usr/bin/env python3
"""
Quick test script to generate videos with Wan I2V model.
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from diffusers import WanImageToVideoPipeline


def main():
    parser = argparse.ArgumentParser(description="Test Wan I2V video generation")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a car driving on a highway",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="test_video_base.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model (if None, uses base model)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to generate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps",
    )

    args = parser.parse_args()

    print("=== Wan I2V Video Generation Test ===")
    print(f"Model: {'Fine-tuned' if args.model_path else 'Base'}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output_path}")

    # Load model
    print("Loading Wan I2V model...")
    try:
        if args.model_path:
            pipeline = WanImageToVideoPipeline.from_pretrained(args.model_path)
        else:
            pipeline = WanImageToVideoPipeline.from_pretrained(
                "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                torch_dtype=torch.float16,
            )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return

    # Create a simple test image
    print("Creating test image...")
    image = Image.new("RGB", (224, 224), color=(100, 150, 200))  # Blue gradient
    print(f"✓ Test image created: {image.size}")

    # Generate video
    print("Generating video...")
    try:
        with torch.no_grad():
            output = pipeline(
                image=image,
                prompt=args.prompt,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=6.0,
            )

        # Handle different output formats
        if hasattr(output, 'frames'):
            video_frames = output.frames
        else:
            video_frames = output

        print(f"✓ Video generated: {len(video_frames) if isinstance(video_frames, list) else 'N/A'} frames")

        # Save video
        try:
            import imageio
            if isinstance(video_frames, list):
                imageio.mimsave(args.output_path, video_frames, fps=8)
            else:
                video_frames.save(args.output_path, fps=8)
            print(f"✓ Video saved to {args.output_path}")
        except Exception as save_err:
            print(f"✗ Failed to save video: {save_err}")

    except Exception as gen_err:
        print(f"✗ Video generation failed: {gen_err}")
        return


if __name__ == "__main__":
    main()