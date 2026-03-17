#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one anchor frame per generated window clip for I2V re-anchoring.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing clips/window_*.mp4")
    parser.add_argument("--output-dir", required=True, help="Directory to store extracted PNG anchors")
    parser.add_argument("--map-output", required=True, help="JSON output mapping window ids/indexes to anchor image paths")
    parser.add_argument("--frame-position", choices=["first", "middle", "last"], default="first")
    return parser.parse_args()


def pick_frame_index(frame_count: int, mode: str) -> int:
    if frame_count <= 1:
        return 0
    if mode == "last":
        return max(0, frame_count - 1)
    if mode == "middle":
        return max(0, frame_count // 2)
    return 0


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    clips_dir = run_dir / "clips"
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"clips directory not found: {clips_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_output = Path(args.map_output)
    map_output.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio
    from PIL import Image

    mapping: dict[str, str] = {}
    for clip_path in sorted(clips_dir.glob("window_*.mp4")):
        m = re.search(r"window_(\d+)\.mp4$", clip_path.name)
        if not m:
            continue
        idx = int(m.group(1))
        reader = imageio.get_reader(clip_path.as_posix())
        try:
            frame_count = reader.count_frames()
        except Exception:
            frame_count = 1
        frame_idx = pick_frame_index(frame_count, args.frame_position)
        frame = reader.get_data(frame_idx)
        out_path = output_dir / f"window_{idx:03d}.png"
        Image.fromarray(frame).save(out_path)
        abs_path = out_path.resolve().as_posix()
        mapping[f"w_{idx:03d}"] = abs_path
        mapping[str(idx)] = abs_path
        mapping[f"{idx:03d}"] = abs_path
        reader.close()

    map_output.write_text(json.dumps(mapping, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[OK] extracted {len(mapping) // 3} anchor images to {output_dir}")
    print(f"[OK] wrote map: {map_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
