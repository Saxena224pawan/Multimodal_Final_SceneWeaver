#!/usr/bin/env python3
"""
Test script for conditioning and video-save regressions.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageDraw

from video_backbone.wan_backbone import WanBackbone, WanBackboneConfig


def create_test_frames(width=832, height=480, num_frames=16):
    """Create synthetic test frames with some temporal patterns."""
    frames = []
    for i in range(num_frames):
        img = Image.new("RGB", (width, height), color=(50, 100, 150))
        draw = ImageDraw.Draw(img)

        x_pos = int((i / num_frames) * width)
        draw.rectangle([x_pos, height // 4, x_pos + 50, height // 4 + 50], fill=(200, 100, 100))
        draw.ellipse(
            [width // 2 + i * 5, height // 2, width // 2 + i * 5 + 30, height // 2 + 30],
            fill=(100, 200, 100),
        )

        frames.append(np.array(img))
    return frames


def test_noise_extraction():
    """Test the legacy noise-pattern extraction helper."""
    print("Testing noise pattern extraction...")

    config = WanBackboneConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    backbone = WanBackbone(config)
    test_frame = create_test_frames(num_frames=1)[0]

    try:
        noise_frame = backbone._extract_noise_pattern(test_frame, strength=0.3)
        print("✓ Noise pattern extraction successful")
        print(f"  Original frame shape: {test_frame.shape}")
        print(f"  Noise frame type: {type(noise_frame)}")

        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)

        Image.fromarray(test_frame).save(output_dir / "original_frame.png")
        noise_frame.save(output_dir / "noise_pattern.png")
        print(f"  Test images saved to {output_dir}")
    except Exception as exc:
        print(f"✗ Noise pattern extraction failed: {exc}")
        return False

    return True


def test_conditioning_modes():
    """Smoke test conditioning helpers without a loaded pipeline."""
    print("\nTesting conditioning modes...")

    config = WanBackboneConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    backbone = WanBackbone(config)
    test_frames = create_test_frames(num_frames=4)

    call_kwargs_regular = {}
    mode_regular = backbone._apply_reference_conditioning(
        call_kwargs_regular, test_frames, 0.7, 832, 480, use_noise_conditioning=False
    )
    print(f"✓ Regular conditioning mode: {mode_regular}")

    call_kwargs_noise = {}
    mode_noise = backbone._apply_reference_conditioning(
        call_kwargs_noise, test_frames, 0.7, 832, 480, use_noise_conditioning=True
    )
    print(f"✓ Noise conditioning mode: {mode_noise}")

    return True


def test_wan_tail_frame_fallback():
    """Wan I2V should fall back to a single tail anchor and avoid last_image."""
    print("\nTesting Wan tail-frame fallback...")

    config = WanBackboneConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    backbone = WanBackbone(config)

    class FakeWanPipeline:
        def __call__(self, image=None, last_image=None, reference_strength=None):
            return None

    FakeWanPipeline.__module__ = "diffusers.pipelines.wan.pipeline_wan_i2v"
    backbone.pipeline = FakeWanPipeline()
    backbone._introspect_pipeline_call()

    call_kwargs = {}
    test_frames = create_test_frames(num_frames=4)
    mode = backbone._apply_reference_conditioning(
        call_kwargs,
        test_frames,
        0.7,
        832,
        480,
        reference_source="previous_window_tail",
        use_noise_conditioning=False,
    )

    if mode != "tail_anchor_frame":
        print(f"✗ Unexpected conditioning mode: {mode}")
        return False
    if "image" not in call_kwargs:
        print("✗ Tail-frame image anchor was not set")
        return False
    if "last_image" in call_kwargs:
        print("✗ last_image should not be forwarded for Wan fallback")
        return False

    print(f"✓ Wan fallback mode: {mode}")
    return True


def test_tail_anchor_selection_prefers_cleaner_frame():
    """The tail anchor should prefer the cleaner/sharper tail frame over a degraded final frame."""
    print("\nTesting tail-anchor selection...")

    config = WanBackboneConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    backbone = WanBackbone(config)

    class FakeWanPipeline:
        def __call__(self, image=None, reference_strength=None):
            return None

    FakeWanPipeline.__module__ = "diffusers.pipelines.wan.pipeline_wan_i2v"
    backbone.pipeline = FakeWanPipeline()
    backbone._introspect_pipeline_call()

    clean = np.zeros((96, 96, 3), dtype=np.uint8)
    clean[:, ::8] = 255
    clean[::8, :] = 255
    degraded = np.full((96, 96, 3), 118, dtype=np.uint8)
    frames = [degraded.copy(), clean.copy(), degraded.copy(), degraded.copy()]

    call_kwargs = {}
    mode = backbone._apply_reference_conditioning(
        call_kwargs,
        frames,
        0.7,
        96,
        96,
        reference_source="previous_window_tail",
        use_noise_conditioning=False,
    )

    if mode != "tail_anchor_frame":
        print(f"✗ Unexpected tail anchor mode: {mode}")
        return False
    if backbone.last_reference_anchor_index != 1:
        print(f"✗ Expected anchor index 1, got {backbone.last_reference_anchor_index}")
        return False

    selected = np.asarray(call_kwargs["image"])
    if selected.std() <= degraded.std():
        print("✗ Selected anchor does not look cleaner than the degraded tail frame")
        return False

    print(f"✓ Tail anchor index: {backbone.last_reference_anchor_index}")
    return True


def test_float_frame_video_save():
    """Ensure float frames are scaled instead of collapsing to black."""
    print("\nTesting float-frame video save...")

    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "float_frame_roundtrip.mp4"

    frames = []
    for i in range(4):
        frame = np.zeros((48, 64, 3), dtype=np.float32)
        frame[..., 0] = np.linspace(0.0, 1.0, 64, dtype=np.float32)
        frame[..., 1] = i / 3.0
        frame[12:36, 20:44, 2] = 1.0
        frames.append(frame)

    WanBackbone.save_video(frames, output_path, fps=4)

    import imageio.v3 as iio

    video = iio.imread(output_path.as_posix())
    print(f"✓ Saved video shape: {video.shape}")
    print(f"  Pixel range: {int(video.min())}..{int(video.max())}")

    if int(video.max()) <= 8:
        print("✗ Saved video is still nearly black")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Test conditioning and save-video regressions")
    parser.add_argument("--test-noise", action="store_true", help="Test noise extraction")
    parser.add_argument("--test-conditioning", action="store_true", help="Test conditioning modes")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--test-save-video", action="store_true", help="Test float-frame video writing")
    parser.add_argument("--test-wan-tail-fallback", action="store_true", help="Test Wan tail-frame fallback")
    parser.add_argument("--test-tail-anchor-selection", action="store_true", help="Test tail anchor selection")

    args = parser.parse_args()

    if args.all or args.test_noise:
        if not test_noise_extraction():
            sys.exit(1)

    if args.all or args.test_conditioning:
        if not test_conditioning_modes():
            sys.exit(1)

    if args.all or args.test_save_video:
        if not test_float_frame_video_save():
            sys.exit(1)

    if args.all or args.test_wan_tail_fallback:
        if not test_wan_tail_frame_fallback():
            sys.exit(1)

    if args.all or args.test_tail_anchor_selection:
        if not test_tail_anchor_selection_prefers_cleaner_frame():
            sys.exit(1)

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
