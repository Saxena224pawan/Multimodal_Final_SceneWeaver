#!/usr/bin/env python3
"""Generate appendix-ready story figures from per-window video clips.

The script builds three figures for a story across the five paper variants:
1. Boundary continuity strips: last five frames of window k vs first five frames of window k+1
2. Boundary center-crop strips: same as above, but zoomed into the center
3. Window storyboard: one representative middle frame per window

It uses the per-window clips so the boundary comparisons are exact.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - runtime fallback only
    cv2 = None


REPO_ROOT = Path(__file__).resolve().parents[1]
FFMPEG_BIN = Path("/apps/python/3.9-anaconda/envs/pytorch-1.10/bin/ffmpeg")
FFPROBE_BIN = Path("/apps/python/3.9-anaconda/envs/pytorch-1.10/bin/ffprobe")
FFMPEG_LIB = Path("/apps/python/3.9-anaconda/envs/pytorch-1.10/lib")
SHORT_STORY_LIBRARY_PATH = REPO_ROOT / "configs" / "stories" / "short_stories.sh"
SHORT_STORY_ENTRY_RE = re.compile(r'^\[(?P<key>[^\]]+)\]="(?P<text>.*)"$')
DEFAULT_STORIES = (
    "fox_and_grapes",
    "lion_and_mouse",
    "thirsty_crow",
    "tortoise_and_hare",
)


@dataclass(frozen=True)
class MethodSpec:
    label: str
    patterns: Tuple[str, ...]
    color: Tuple[int, int, int]


METHOD_SPECS: Sequence[MethodSpec] = (
    MethodSpec(
        "Simple T2V",
        ("outputs/story_runs_origin_pavan/simple_{story}_*concat*/clips",),
        (220, 111, 52),
    ),
    MethodSpec(
        "Core T2V",
        ("outputs/story_runs_origin_pavan/core_{story}_*concat*/clips",),
        (181, 137, 0),
    ),
    MethodSpec(
        "Agentic T2V",
        ("outputs/story_runs_origin_pavan/agentic_{story}_*concat*/clips",),
        (166, 77, 121),
    ),
    MethodSpec(
        "Core I2V",
        (
            "outputs/story_runs_origin_pavan_i2v_runs/core_{story}_*/clips",
            "outputs/story_runs_origin_pavan_i2v_concat/core_{story}_*/clips",
        ),
        (34, 139, 230),
    ),
    MethodSpec(
        "Agentic I2V",
        (
            "outputs/story_runs_origin_pavan_i2v_runs/agentic_{story}_*/clips",
            "outputs/story_runs_origin_pavan_i2v_concat/agentic_{story}_*/clips",
        ),
        (0, 110, 170),
    ),
)


def ffmpeg_env() -> Dict[str, str]:
    env = os.environ.copy()
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{FFMPEG_LIB}:{prev}" if prev else str(FFMPEG_LIB)
    )
    return env


def run_checked(cmd: Sequence[str]) -> str:
    proc = subprocess.run(
        list(cmd),
        env=ffmpeg_env(),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout


def find_clip_dir(story: str, spec: MethodSpec) -> Path:
    matches: List[Path] = []
    for pattern in spec.patterns:
        matches.extend(REPO_ROOT.glob(pattern.format(story=story)))
    if not matches:
        raise FileNotFoundError(f"No clip directory found for {spec.label} / {story}")

    def rank(path: Path) -> Tuple[int, float, str]:
        story_dir = path.parent
        has_rerun = int(
            any("rerun" in child.name for child in story_dir.iterdir() if child.is_dir())
        )
        has_fullstory = int(any(story_dir.glob("*_full_story.mp4")))
        return (has_rerun + has_fullstory, story_dir.stat().st_mtime, story_dir.name)

    return sorted(matches, key=rank)[-1]


def probe_video(video_path: Path) -> Tuple[int, int, int]:
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if width > 0 and height > 0 and frame_count > 0:
                return width, height, frame_count
        else:
            cap.release()
    output = run_checked(
        (
            str(FFPROBE_BIN),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "default=noprint_wrappers=1",
            str(video_path),
        )
    )
    info: Dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()
    return int(info["width"]), int(info["height"]), int(info["nb_frames"])


def extract_frame(video_path: Path, frame_idx: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(frame_rgb).save(output_path)
                return
        else:
            cap.release()
    run_checked(
        (
            str(FFMPEG_BIN),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{frame_idx})",
            "-frames:v",
            "1",
            str(output_path),
        )
    )


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fitted_text_font(
    text: str,
    max_width: int,
    max_height: int,
    preferred_size: int,
    min_size: int = 10,
) -> ImageFont.ImageFont:
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    for size in range(preferred_size, min_size - 1, -1):
        font = load_font(size)
        bbox = probe.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= max_width and height <= max_height:
            return font
    return load_font(min_size)


def fit_image(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    fitted = image.copy()
    resample = getattr(Image, "Resampling", Image).LANCZOS
    fitted.thumbnail(target_size, resample)
    canvas = Image.new("RGB", target_size, "white")
    x = (target_size[0] - fitted.width) // 2
    y = (target_size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def center_crop(image: Image.Image, frac_w: float = 0.58, frac_h: float = 0.72) -> Image.Image:
    width, height = image.size
    crop_w = int(width * frac_w)
    crop_h = int(height * frac_h)
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))


def annotate(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font, fill, anchor="la") -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def load_story_library(path: Path = SHORT_STORY_LIBRARY_PATH) -> Dict[str, str]:
    stories: Dict[str, str] = {}
    if not path.is_file():
        return stories
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = SHORT_STORY_ENTRY_RE.match(line)
        if match is None:
            continue
        stories[match.group("key")] = normalize_prompt(match.group("text"))
    return stories


def slugify(text: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (text or "").strip()).strip("._").lower()
    return cleaned or fallback


def count_windows(clip_dir: Path) -> int:
    return len(sorted(clip_dir.glob("window_*.mp4")))


def padded_frame_indices(frame_count: int, count: int, from_end: bool) -> List[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if from_end:
        indices = list(range(max(0, frame_count - count), frame_count))
        while len(indices) < count:
            indices.insert(0, indices[0])
        return indices
    indices = list(range(min(frame_count, count)))
    while len(indices) < count:
        indices.append(indices[-1])
    return indices


def recover_story_prompt(story: str, methods: Sequence[Tuple[MethodSpec, Path]]) -> str:
    for _, clip_dir in methods:
        summary_path = clip_dir.parent / "run_summary.json"
        if not summary_path.is_file():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        storyline = normalize_prompt(str(payload.get("storyline", "")))
        if storyline:
            return storyline
    return load_story_library().get(story, "")


def wrap_prompt_lines(prompt: str, width: int = 160) -> List[str]:
    if not prompt:
        return ["Prompt: unavailable"]
    return textwrap.wrap(f"Prompt: {prompt}", width=width, break_long_words=False)


def select_representative_boundaries(boundary_count: int, target_count: int = 4) -> List[int]:
    if boundary_count <= 0:
        return []
    if boundary_count <= target_count:
        return list(range(boundary_count))
    max_index = boundary_count - 1
    selected = {
        int(round((idx * max_index) / float(target_count - 1)))
        for idx in range(target_count)
    }
    return sorted(selected)


def make_boundary_pairs(
    story: str,
    story_prompt: str,
    methods: Sequence[Tuple[MethodSpec, Path]],
    output_path: Path,
    crop: bool,
    boundary_indices: Optional[Sequence[int]] = None,
    boundary_cols_override: Optional[int] = None,
) -> None:
    thumb_size = (52, 32)
    pair_cols = 5
    gap_within_strip = 4
    gap_between_strip_rows = 18
    panel_gap_x = 22
    panel_gap_y = 22
    panel_pad_x = 12
    panel_pad_y = 14
    method_label_w = 126
    method_gap_y = 10
    method_header_h = 18
    method_row_h = method_header_h + thumb_size[1] * 2 + gap_between_strip_rows + 16
    panel_header_h = 30
    left_pad = 24
    right_pad = 24
    bottom_pad = 24
    prompt_lines = wrap_prompt_lines(story_prompt, width=105)
    prompt_box_h = 26 + 22 * len(prompt_lines)
    boundary_count = count_windows(methods[0][1]) - 1
    selected_boundaries = list(range(boundary_count)) if boundary_indices is None else list(boundary_indices)
    if not selected_boundaries:
        raise ValueError("No boundary indices selected.")
    boundary_cols = boundary_cols_override or 2
    boundary_rows = (len(selected_boundaries) + boundary_cols - 1) // boundary_cols
    strip_w = thumb_size[0] * pair_cols + gap_within_strip * (pair_cols - 1)
    method_w = method_label_w + 12 + strip_w
    panel_w = panel_pad_x * 2 + method_w
    panel_h = panel_header_h + panel_pad_y * 2 + len(methods) * method_row_h + (len(methods) - 1) * method_gap_y
    top_h = 118 + prompt_box_h
    width = left_pad + boundary_cols * panel_w + (boundary_cols - 1) * panel_gap_x + right_pad
    height = top_h + boundary_rows * panel_h + (boundary_rows - 1) * panel_gap_y + bottom_pad

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    header_font = load_font(18)
    label_font = load_font(20)
    small_font = load_font(14)
    prompt_font = load_font(16)

    title = f"{story.replace('_', ' ').title()}: Boundary Continuity"
    if boundary_indices is None:
        subtitle = (
            "Last 5 frames of window k and first 5 frames of window k+1"
            if not crop
            else "Center crop of the same 5-frame boundary strips"
        )
    else:
        subset_text = ", ".join(f"W{idx}->{idx + 1}" for idx in selected_boundaries)
        subtitle = (
            f"Representative subset of boundaries ({subset_text})"
            if not crop
            else f"Center crop of representative boundaries ({subset_text})"
        )
    annotate(draw, (left_pad, 24), title, title_font, (0, 0, 0))
    annotate(draw, (left_pad, 60), subtitle, small_font, (70, 70, 70))

    prompt_box = (left_pad, 76, width - right_pad, 76 + prompt_box_h)
    draw.rounded_rectangle(prompt_box, radius=16, fill=(247, 247, 247), outline=(220, 220, 220), width=2)
    draw.multiline_text(
        (left_pad + 16, 92),
        "\n".join(prompt_lines),
        font=prompt_font,
        fill=(20, 20, 20),
        spacing=6,
    )

    with tempfile.TemporaryDirectory(prefix=f"{story}_boundary_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for panel_idx, boundary_idx in enumerate(selected_boundaries):
            panel_col = panel_idx % boundary_cols
            panel_row = panel_idx // boundary_cols
            panel_x = left_pad + panel_col * (panel_w + panel_gap_x)
            panel_y = top_h + panel_row * (panel_h + panel_gap_y)
            draw.rounded_rectangle(
                (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
                radius=18,
                fill=(252, 252, 252),
                outline=(220, 220, 220),
                width=2,
            )
            annotate(
                draw,
                (panel_x + panel_w // 2, panel_y + 22),
                f"W{boundary_idx} -> W{boundary_idx + 1}",
                header_font,
                (0, 0, 0),
                anchor="ma",
            )

            for method_idx, (spec, clip_dir) in enumerate(methods):
                prev_clip = clip_dir / f"window_{boundary_idx:03d}.mp4"
                next_clip = clip_dir / f"window_{boundary_idx + 1:03d}.mp4"
                _, _, prev_n = probe_video(prev_clip)
                _, _, next_n = probe_video(next_clip)
                top_indices = padded_frame_indices(prev_n, pair_cols, from_end=True)
                bottom_indices = padded_frame_indices(next_n, pair_cols, from_end=False)

                method_y = panel_y + panel_header_h + panel_pad_y + method_idx * (method_row_h + method_gap_y)
                draw.rounded_rectangle(
                    (
                        panel_x + panel_pad_x,
                        method_y,
                        panel_x + panel_w - panel_pad_x,
                        method_y + method_row_h,
                    ),
                    radius=12,
                    fill=(255, 255, 255),
                    outline=(232, 232, 232),
                    width=2,
                )
                label_x0 = panel_x + panel_pad_x + 8
                label_y0 = method_y + 10
                draw.rounded_rectangle(
                    (
                        label_x0,
                        label_y0,
                        label_x0 + method_label_w - 16,
                        label_y0 + method_row_h - 20,
                    ),
                    radius=12,
                    fill=(247, 247, 247),
                    outline=spec.color,
                    width=3,
                )
                badge_w = method_label_w - 16
                badge_h = method_row_h - 20
                badge_font = fitted_text_font(
                    spec.label,
                    max_width=badge_w - 18,
                    max_height=badge_h - 18,
                    preferred_size=15,
                    min_size=11,
                )
                annotate(
                    draw,
                    (label_x0 + badge_w // 2, label_y0 + badge_h // 2),
                    spec.label,
                    badge_font,
                    spec.color,
                    anchor="mm",
                )

                strip_x = panel_x + panel_pad_x + method_label_w + 10
                annotate(
                    draw,
                    (strip_x, method_y + 6),
                    f"W{boundary_idx} end",
                    small_font,
                    (90, 90, 90),
                )
                top_thumb_y = method_y + method_header_h
                for col_idx, frame_idx in enumerate(top_indices):
                    frame_path = tmp_root / f"{slugify(spec.label)}_{boundary_idx}_prev_{col_idx}.png"
                    extract_frame(prev_clip, frame_idx, frame_path)
                    frame_img = Image.open(frame_path).convert("RGB")
                    if crop:
                        frame_img = center_crop(frame_img)
                    frame_thumb = fit_image(frame_img, thumb_size)
                    thumb_x = strip_x + col_idx * (thumb_size[0] + gap_within_strip)
                    canvas.paste(frame_thumb, (thumb_x, top_thumb_y))

                annotate(
                    draw,
                    (strip_x, top_thumb_y + thumb_size[1] + 4),
                    f"W{boundary_idx + 1} start",
                    small_font,
                    (90, 90, 90),
                )
                bottom_thumb_y = top_thumb_y + thumb_size[1] + gap_between_strip_rows
                for col_idx, frame_idx in enumerate(bottom_indices):
                    frame_path = tmp_root / f"{slugify(spec.label)}_{boundary_idx}_next_{col_idx}.png"
                    extract_frame(next_clip, frame_idx, frame_path)
                    frame_img = Image.open(frame_path).convert("RGB")
                    if crop:
                        frame_img = center_crop(frame_img)
                    frame_thumb = fit_image(frame_img, thumb_size)
                    thumb_x = strip_x + col_idx * (thumb_size[0] + gap_within_strip)
                    canvas.paste(frame_thumb, (thumb_x, bottom_thumb_y))

    canvas.save(output_path)


def make_boundary_pairs_paper(
    story: str,
    story_prompt: str,
    methods: Sequence[Tuple[MethodSpec, Path]],
    output_path: Path,
    boundary_indices: Sequence[int],
) -> None:
    if not boundary_indices:
        raise ValueError("No boundary indices selected.")

    thumb_size = (46, 28)
    pair_cols = 5
    gap_within_strip = 3
    cell_pad_x = 8
    cell_pad_y = 8
    label_col_w = 118
    col_gap = 14
    row_gap = 10
    boundary_header_h = 26
    left_pad = 18
    right_pad = 18
    bottom_pad = 18
    prompt_lines = wrap_prompt_lines(story_prompt, width=115)
    prompt_box_h = 22 + 18 * len(prompt_lines)
    top_h = 84 + prompt_box_h

    strip_w = thumb_size[0] * pair_cols + gap_within_strip * (pair_cols - 1)
    cell_w = cell_pad_x * 2 + strip_w
    cell_h = cell_pad_y * 2 + 12 + thumb_size[1] + 12 + thumb_size[1]
    width = (
        left_pad
        + label_col_w
        + 12
        + len(boundary_indices) * cell_w
        + max(0, len(boundary_indices) - 1) * col_gap
        + right_pad
    )
    height = (
        top_h
        + boundary_header_h
        + len(methods) * cell_h
        + max(0, len(methods) - 1) * row_gap
        + bottom_pad
    )

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(20)
    small_font = load_font(12)
    header_font = load_font(15)
    prompt_font = load_font(13)

    annotate(
        draw,
        (left_pad, 18),
        f"{story.replace('_', ' ').title()}: Boundary Continuity",
        title_font,
        (0, 0, 0),
    )
    annotate(
        draw,
        (left_pad, 44),
        "Representative boundaries (W0->1, W2->3, W4->5, W6->7)",
        small_font,
        (90, 90, 90),
    )

    prompt_box = (left_pad, 56, width - right_pad, 56 + prompt_box_h)
    draw.rounded_rectangle(
        prompt_box,
        radius=12,
        fill=(247, 247, 247),
        outline=(220, 220, 220),
        width=2,
    )
    draw.multiline_text(
        (left_pad + 12, 68),
        "\n".join(prompt_lines),
        font=prompt_font,
        fill=(20, 20, 20),
        spacing=4,
    )

    grid_top = top_h + boundary_header_h
    header_y = top_h
    grid_left = left_pad + label_col_w + 12

    for col_idx, boundary_idx in enumerate(boundary_indices):
        x0 = grid_left + col_idx * (cell_w + col_gap)
        annotate(
            draw,
            (x0 + cell_w // 2, header_y + 8),
            f"W{boundary_idx} -> W{boundary_idx + 1}",
            header_font,
            (0, 0, 0),
            anchor="ma",
        )

    with tempfile.TemporaryDirectory(prefix=f"{story}_boundary_paper_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for method_idx, (spec, clip_dir) in enumerate(methods):
            y0 = grid_top + method_idx * (cell_h + row_gap)
            badge_h = cell_h - 4
            label_x0 = left_pad
            label_y0 = y0 + 2
            draw.rounded_rectangle(
                (label_x0, label_y0, label_x0 + label_col_w, label_y0 + badge_h),
                radius=12,
                fill=(248, 248, 248),
                outline=spec.color,
                width=3,
            )
            badge_font = fitted_text_font(
                spec.label,
                max_width=label_col_w - 16,
                max_height=badge_h - 14,
                preferred_size=16,
                min_size=11,
            )
            annotate(
                draw,
                (label_x0 + label_col_w // 2, label_y0 + badge_h // 2),
                spec.label,
                badge_font,
                spec.color,
                anchor="mm",
            )

            for col_idx, boundary_idx in enumerate(boundary_indices):
                prev_clip = clip_dir / f"window_{boundary_idx:03d}.mp4"
                next_clip = clip_dir / f"window_{boundary_idx + 1:03d}.mp4"
                _, _, prev_n = probe_video(prev_clip)
                _, _, next_n = probe_video(next_clip)
                top_indices = padded_frame_indices(prev_n, pair_cols, from_end=True)
                bottom_indices = padded_frame_indices(next_n, pair_cols, from_end=False)

                x0 = grid_left + col_idx * (cell_w + col_gap)
                draw.rounded_rectangle(
                    (x0, y0, x0 + cell_w, y0 + cell_h),
                    radius=10,
                    fill=(252, 252, 252),
                    outline=(224, 224, 224),
                    width=2,
                )
                annotate(draw, (x0 + cell_pad_x, y0 + 4), f"W{boundary_idx} end", small_font, (90, 90, 90))
                top_thumb_y = y0 + cell_pad_y + 8
                for frame_pos, frame_idx in enumerate(top_indices):
                    frame_path = tmp_root / f"{slugify(spec.label)}_{boundary_idx}_prev_{frame_pos}.png"
                    extract_frame(prev_clip, frame_idx, frame_path)
                    frame_img = fit_image(Image.open(frame_path).convert("RGB"), thumb_size)
                    thumb_x = x0 + cell_pad_x + frame_pos * (thumb_size[0] + gap_within_strip)
                    canvas.paste(frame_img, (thumb_x, top_thumb_y))

                annotate(
                    draw,
                    (x0 + cell_pad_x, top_thumb_y + thumb_size[1] + 2),
                    f"W{boundary_idx + 1} start",
                    small_font,
                    (90, 90, 90),
                )
                bottom_thumb_y = top_thumb_y + thumb_size[1] + 12
                for frame_pos, frame_idx in enumerate(bottom_indices):
                    frame_path = tmp_root / f"{slugify(spec.label)}_{boundary_idx}_next_{frame_pos}.png"
                    extract_frame(next_clip, frame_idx, frame_path)
                    frame_img = fit_image(Image.open(frame_path).convert("RGB"), thumb_size)
                    thumb_x = x0 + cell_pad_x + frame_pos * (thumb_size[0] + gap_within_strip)
                    canvas.paste(frame_img, (thumb_x, bottom_thumb_y))

    canvas.save(output_path)


def make_storyboard(
    story: str,
    methods: Sequence[Tuple[MethodSpec, Path]],
    output_path: Path,
) -> None:
    thumb_size = (180, 108)
    label_w = 175
    top_h = 102
    row_h = thumb_size[1] + 30
    gap = 16
    left_pad = 24
    right_pad = 18
    bottom_pad = 24
    window_count = count_windows(methods[0][1])
    width = left_pad + label_w + window_count * thumb_size[0] + max(0, window_count - 1) * gap + right_pad
    height = top_h + len(methods) * row_h + bottom_pad

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    header_font = load_font(18)
    label_font = load_font(20)
    small_font = load_font(14)

    annotate(
        draw,
        (left_pad, 24),
        f"{story.replace('_', ' ').title()}: Window Storyboard",
        title_font,
        (0, 0, 0),
    )
    annotate(
        draw,
        (left_pad, 60),
        "Representative middle frame from each generation window",
        small_font,
        (70, 70, 70),
    )

    for col_idx in range(window_count):
        x0 = left_pad + label_w + col_idx * (thumb_size[0] + gap)
        annotate(
            draw,
            (x0 + thumb_size[0] // 2, 84),
            f"W{col_idx}",
            header_font,
            (0, 0, 0),
            anchor="ma",
        )

    with tempfile.TemporaryDirectory(prefix=f"{story}_storyboard_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for row_idx, (spec, clip_dir) in enumerate(methods):
            y0 = top_h + row_idx * row_h
            draw.rounded_rectangle(
                (left_pad, y0 + 12, left_pad + label_w - 16, y0 + 12 + thumb_size[1]),
                radius=16,
                fill=(247, 247, 247),
                outline=spec.color,
                width=3,
            )
            storyboard_badge_w = label_w - 16
            storyboard_badge_h = thumb_size[1]
            storyboard_font = fitted_text_font(
                spec.label,
                max_width=storyboard_badge_w - 24,
                max_height=storyboard_badge_h - 20,
                preferred_size=20,
                min_size=12,
            )
            annotate(
                draw,
                (left_pad + storyboard_badge_w // 2, y0 + 12 + storyboard_badge_h // 2),
                spec.label,
                storyboard_font,
                spec.color,
                anchor="mm",
            )

            for window_idx in range(window_count):
                clip_path = clip_dir / f"window_{window_idx:03d}.mp4"
                _, _, frame_count = probe_video(clip_path)
                middle_idx = frame_count // 2
                frame_path = tmp_root / f"{spec.label}_{window_idx}_mid.png"
                extract_frame(clip_path, middle_idx, frame_path)
                thumb = fit_image(Image.open(frame_path).convert("RGB"), thumb_size)
                x0 = left_pad + label_w + window_idx * (thumb_size[0] + gap)
                canvas.paste(thumb, (x0, y0 + 12))

    canvas.save(output_path)


def compute_boundary_differences(
    methods: Sequence[Tuple[MethodSpec, Path]]
) -> Dict[str, List[float]]:
    scores: Dict[str, List[float]] = {}
    boundary_count = count_windows(methods[0][1]) - 1
    with tempfile.TemporaryDirectory(prefix="boundary_scores_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for spec, clip_dir in methods:
            method_scores: List[float] = []
            for boundary_idx in range(boundary_count):
                prev_clip = clip_dir / f"window_{boundary_idx:03d}.mp4"
                next_clip = clip_dir / f"window_{boundary_idx + 1:03d}.mp4"
                _, _, prev_n = probe_video(prev_clip)
                _, _, next_n = probe_video(next_clip)
                tail_indices = padded_frame_indices(prev_n, 5, from_end=True)
                start_indices = padded_frame_indices(next_n, 5, from_end=False)
                pair_diffs: List[float] = []
                for pair_idx, (prev_idx, next_idx) in enumerate(zip(tail_indices, start_indices)):
                    prev_path = tmp_root / f"{slugify(spec.label)}_{boundary_idx}_prev_{pair_idx}.png"
                    next_path = tmp_root / f"{slugify(spec.label)}_{boundary_idx}_next_{pair_idx}.png"
                    extract_frame(prev_clip, prev_idx, prev_path)
                    extract_frame(next_clip, next_idx, next_path)
                    prev_img = np.asarray(Image.open(prev_path).convert("RGB"), dtype=np.float32)
                    next_img = np.asarray(Image.open(next_path).convert("RGB"), dtype=np.float32)
                    pair_diffs.append(float(np.abs(prev_img - next_img).mean() / 255.0))
                method_scores.append(sum(pair_diffs) / len(pair_diffs))
            scores[spec.label] = method_scores
    return scores


def write_caption_notes(
    story: str,
    story_prompt: str,
    output_dir: Path,
    methods: Sequence[Tuple[MethodSpec, Path]],
    boundary_scores: Optional[Dict[str, List[float]]],
) -> None:
    notes_path = output_dir / f"{story}_appendix_notes.md"
    lines = [
        f"# {story.replace('_', ' ').title()} Appendix Figures",
        "",
        f"Story prompt: {story_prompt or 'unavailable'}",
        "",
        "Generated files:",
        f"- `{story}_boundary_pairs.png`: full-frame last-5 to first-5 boundary strips.",
        (
            f"- `{story}_boundary_pairs_paper.png`: compact NeurIPS-friendly boundary figure "
            "with four representative transitions and one shared method-label column."
        ),
        f"- `{story}_boundary_pairs_center_crop.png`: center-crop version of the same 5-frame strips.",
        f"- `{story}_window_storyboard.png`: middle-frame storyboard across the eight windows.",
        "",
        "Suggested appendix caption:",
        (
            "Figure X compares cross-window continuity for the same story across the five "
            "generation variants using a compact matrix layout. Rows correspond to methods, "
            "with method labels shown once in the shared left column, and columns show four "
            "representative transitions (W0->W1, W2->W3, W4->W5, and W6->W7). Within each cell, "
            "the top strip shows the last five frames of window k and the bottom strip shows "
            "the first five frames of window k+1, making visual carryover across windows "
            "directly inspectable in a NeurIPS-friendly figure width."
        ),
        "",
        "Resolved clip sources:",
    ]
    for spec, clip_dir in methods:
        lines.append(f"- `{spec.label}`: `{clip_dir}`")
    if boundary_scores:
        lines.extend(
            [
                "",
                "Mean absolute boundary-frame differences (lower means smaller visual jump; use only as a supplementary signal):",
            ]
        )
        for label, scores in boundary_scores.items():
            mean_score = sum(scores) / len(scores)
            score_str = ", ".join(f"{score:.3f}" for score in scores)
            lines.append(f"- `{label}`: mean={mean_score:.3f}; boundaries=[{score_str}]")
    notes_path.write_text("\n".join(lines) + "\n")


def resolve_methods(story: str) -> List[Tuple[MethodSpec, Path]]:
    methods: List[Tuple[MethodSpec, Path]] = []
    for spec in METHOD_SPECS:
        methods.append((spec, find_clip_dir(story, spec)))
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--story", help="Story slug, e.g. fox_and_grapes")
    parser.add_argument(
        "--all-stories",
        action="store_true",
        help="Generate appendix figures for the default four paper stories",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the figures should be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.all_stories:
        stories = list(DEFAULT_STORIES)
    elif args.story:
        stories = [args.story]
    else:
        raise SystemExit("Provide --story <slug> or --all-stories.")

    for story in stories:
        methods = resolve_methods(story)
        story_prompt = recover_story_prompt(story, methods)
        paper_boundaries = select_representative_boundaries(count_windows(methods[0][1]) - 1)
        make_boundary_pairs(
            story,
            story_prompt,
            methods,
            output_dir / f"{story}_boundary_pairs.png",
            crop=False,
        )
        make_boundary_pairs_paper(
            story,
            story_prompt,
            methods,
            output_dir / f"{story}_boundary_pairs_paper.png",
            boundary_indices=paper_boundaries,
        )
        make_boundary_pairs(
            story,
            story_prompt,
            methods,
            output_dir / f"{story}_boundary_pairs_center_crop.png",
            crop=True,
        )
        make_storyboard(
            story,
            methods,
            output_dir / f"{story}_window_storyboard.png",
        )
        write_caption_notes(story, story_prompt, output_dir, methods, None)


if __name__ == "__main__":
    main()
