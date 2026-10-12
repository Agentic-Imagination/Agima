#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified for Agima: package imports and portable resource configuration.

"""
Create captioned GIF visualizations from VLM-Gym inference outputs.

This script expects an episode folder produced by the Agima evaluation entrypoints
containing:
  - episode_stats.json
  - renders/step_*.png (or other image formats) when --render was used

You can point --input to:
  - a single episode_stats.json file
  - a single episode directory (containing episode_stats.json)
  - a root directory; the script will recursively find all episode_stats.json

Examples:
  # Single episode
  python -m evaluation.visualization \
    --input runs/evaluation/episode/episode_stats.json

  # All episodes under a root
  python -m evaluation.visualization \
    --input runs/evaluation --out-root runs/visualizations

Output:
  By default, writes episode.gif inside each episode directory.
  Use --out-root to mirror episode structure under another root.
"""

import argparse
import ast
import base64
import io
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# -------------------------- Image helpers (robust loader) --------------------------

SUCCESS_BG = (35, 140, 70)
FAIL_BG = (180, 45, 45)
STATUS_TEXT = (255, 255, 255)


def _load_font(size: int) -> Optional[ImageFont.ImageFont]:
    try:
        font_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Times New Roman.ttf",
            "C:/Windows/Fonts/times.ttf",
        ]
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
        return ImageFont.load_default()
    except Exception:
        return None


def _to_pil(img_like):
    """
    Convert various image representations to PIL.Image.
    Supports:
      - np.ndarray (H,W,[C]) uint8 or float [0,1]
      - PIL.Image
      - dict with [image, rgb, frame, render, obs, pixels]
      - base64 string (optionally data URL) or bytes
      - file path string
    """
    if img_like is None:
        return None

    # Already a PIL image
    if isinstance(img_like, Image.Image):
        return img_like

    # Dict with a nested image
    if isinstance(img_like, dict):
        for k in ("image", "rgb", "frame", "render", "obs", "pixels"):
            if k in img_like:
                im = _to_pil(img_like[k])
                if im is not None:
                    return im
        return None

    # Bytes-like → try PIL
    if isinstance(img_like, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(img_like)).convert("RGB")
        except Exception:
            return None

    # String cases
    if isinstance(img_like, str):
        s = img_like.strip()

        # data URL?
        if s.startswith("data:image"):
            try:
                _, b64 = s.split(",", 1)
                data = base64.b64decode(b64)
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                return None

        # Try plain base64 (PNG/JPEG/GIF magic check)
        try:
            data = base64.b64decode(s, validate=True)
            if (
                data[:8] == b"\x89PNG\r\n\x1a\n"
                or data[:2] == b"\xff\xd8"
                or data[:6] in (b"GIF87a", b"GIF89a")
            ):
                return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            pass

        # Maybe it's a file path
        if os.path.exists(s):
            try:
                return Image.open(s).convert("RGB")
            except Exception:
                return None

        return None

    # Numpy array
    try:
        import numpy as _np

        arr = _np.array(img_like)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
            arr = _np.transpose(arr, (1, 2, 0))  # CHW→HWC
        if arr.dtype != _np.uint8:
            arr = _np.clip(arr, 0, 1) if arr.max() <= 1.0 else _np.clip(arr, 0, 255)
            arr = (arr * 255).astype("uint8") if arr.max() <= 1.0 else arr.astype("uint8")
        if arr.ndim == 2:
            arr = _np.stack([arr, arr, arr], axis=-1)
        return Image.fromarray(arr)
    except Exception:
        return None


def _wrap_lines(lines: List[str], draw: ImageDraw.ImageDraw, font: Optional[ImageFont.ImageFont], max_width_px: int) -> List[str]:
    wrapped: List[str] = []
    for line in [l for l in lines if l]:
        words = str(line).split()
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            text_width = bbox[2] - bbox[0]
            if text_width <= max_width_px:
                cur = test
            else:
                if cur:
                    wrapped.append(cur)
                cur = w
        if cur:
            wrapped.append(cur)
    return wrapped


def _compose_captioned_frame(
    img: Image.Image,
    lines: List[str],
    max_width: Optional[int] = None,
    min_width: Optional[int] = None,
    pad: int = 12,
    line_gap: int = 8,
    bg: Tuple[int, int, int] = (255, 255, 255),
    status_banner: Optional[Tuple[str, Tuple[int, int, int]]] = None,
) -> Image.Image:
    img = img.convert("RGB")
    canvas_width = img.width
    if min_width is not None:
        canvas_width = max(canvas_width, int(min_width))
    if max_width is not None:
        canvas_width = min(canvas_width, int(max_width))

    display_img = img
    if img.width > canvas_width:
        h = int(img.height * (canvas_width / img.width))
        display_img = img.resize((canvas_width, h), Image.Resampling.LANCZOS)
    elif canvas_width >= 512 and img.width < 384:
        target_img_width = min(384, canvas_width)
        h = int(img.height * (target_img_width / img.width))
        display_img = img.resize((target_img_width, h), Image.Resampling.NEAREST)

    tmp = Image.new("RGB", (canvas_width, 10), bg)
    draw = ImageDraw.Draw(tmp)
    font = _load_font(11)

    text_width = max(1, canvas_width - 2 * pad)
    n_cols = 3 if canvas_width >= 840 else (2 if canvas_width >= 560 else 1)
    col_gap = 18 if n_cols > 1 else 0
    col_width = max(1, (text_width - col_gap * (n_cols - 1)) // n_cols)
    wrapped = _wrap_lines(lines, draw, font, col_width)
    if font:
        bbox = draw.textbbox((0, 0), "Ag", font=font)
        line_h = bbox[3] - bbox[1]
    else:
        line_h = 10

    rows_per_col = 0 if not wrapped else (len(wrapped) + n_cols - 1) // n_cols
    caption_h = 0 if not wrapped else (rows_per_col * line_h + max(0, rows_per_col - 1) * line_gap)
    status_h = 34 if status_banner is not None else 0
    H = status_h + display_img.height + (0 if caption_h == 0 else (pad + caption_h + pad))
    canvas = Image.new("RGB", (canvas_width, H), bg)
    draw_canvas = ImageDraw.Draw(canvas)

    if status_banner is not None:
        label, status_bg = status_banner
        draw_canvas.rectangle((0, 0, canvas_width, status_h), fill=status_bg)
        status_font = _load_font(16) or font
        status_bbox = draw_canvas.textbbox((0, 0), label, font=status_font)
        text_w = status_bbox[2] - status_bbox[0]
        text_h = status_bbox[3] - status_bbox[1]
        text_x = max(0, (canvas_width - text_w) // 2)
        text_y = max(0, (status_h - text_h) // 2 - status_bbox[1])
        draw_canvas.text((text_x, text_y), label, fill=STATUS_TEXT, font=status_font)

    img_x = max(0, (canvas_width - display_img.width) // 2)
    canvas.paste(display_img, (img_x, status_h))

    if wrapped:
        caption_y = status_h + display_img.height + pad
        for col in range(n_cols):
            x = pad + col * (col_width + col_gap)
            y = caption_y
            for line in wrapped[col * rows_per_col : (col + 1) * rows_per_col]:
                draw_canvas.text((x, y), line, fill=(0, 0, 0), font=font)
                y += line_h + line_gap
    return canvas


# -------------------------- Episode IO --------------------------

def find_episode_jsons(input_path: Path) -> Iterable[Path]:
    """Yield episode_stats.json paths for a file/dir tree."""
    if input_path.is_file():
        if input_path.name == "episode_stats.json":
            yield input_path
        else:
            # treat file as the JSON
            yield input_path
        return

    # directory
    stats = list(input_path.rglob("episode_stats.json"))
    for p in sorted(stats):
        yield p


def load_episode(json_path: Path) -> Tuple[dict, Path]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    episode_dir = json_path.parent
    return data, episode_dir


def list_render_images(episode_dir: Path) -> List[Path]:
    renders_dir = episode_dir / "renders"
    if not renders_dir.exists() or not renders_dir.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    imgs = [p for p in sorted(renders_dir.iterdir()) if p.suffix.lower() in exts]
    return imgs


def _truncate_text(s: str, max_chars: int) -> str:
    s = s.replace("\n", " ")
    if len(s) <= max_chars:
        return s
    truncated = s[: max_chars - 3]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."


def build_caption_lines(step: dict, step_index: int, max_text_chars: int) -> List[str]:
    lines: List[str] = []
    prompt = step.get("prompt")
    if prompt:
        lines.append(_truncate_text(str(prompt), max_text_chars))
    lines.append(f"step {step_index}")

    action = step.get("action")
    if action is not None:
        lines.append(_truncate_text(str(action), max_text_chars))

    reason = step.get("verbalization") or step.get("reason") or step.get("vlm_output") or step.get("env_feedback")
    if reason:
        lines.append(_truncate_text(str(reason), max_text_chars))

    reward = step.get("reward")
    if reward is not None:
        try:
            lines.append(f"reward={float(reward):.3f}")
        except Exception:
            lines.append(f"reward={reward}")
    return lines


def _is_stop_action(action) -> bool:
    if isinstance(action, tuple):
        parsed = action
    else:
        try:
            parsed = ast.literal_eval(str(action))
        except Exception:
            return False
    return isinstance(parsed, tuple) and len(parsed) >= 1 and parsed[0] == "stop"


def _numeric_reward(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _is_true(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _explicit_success_value(episode_data: dict) -> Optional[bool]:
    stats = episode_data.get("stats") or {}
    for key in ("success", "is_success", "solved"):
        if key in stats:
            return _is_true(stats.get(key))
    history = episode_data.get("history") or []
    if history:
        last_step = history[-1]
        for key in ("success", "is_success", "solved"):
            if key in last_step:
                return _is_true(last_step.get(key))
        info = last_step.get("info")
        if isinstance(info, dict):
            for key in ("success", "is_success", "solved"):
                if key in info:
                    return _is_true(info.get(key))
    return None


def episode_succeeded(episode_data: dict) -> bool:
    history = episode_data.get("history") or []
    if not history:
        return False
    stats = episode_data.get("stats") or {}
    if not _is_true(stats.get("terminated")):
        return False
    last_step = history[-1]
    if not _is_stop_action(last_step.get("action")):
        return False
    explicit_success = _explicit_success_value(episode_data)
    if explicit_success is not None:
        return explicit_success
    reward = _numeric_reward(last_step.get("reward"))
    if reward is None:
        reward = _numeric_reward(stats.get("reward"))
    return reward is not None and reward > 0


def _episode_status_banner(episode_data: dict) -> Tuple[str, Tuple[int, int, int]]:
    return ("SUCCESS", SUCCESS_BG) if episode_succeeded(episode_data) else ("FAIL", FAIL_BG)


def _history_step_for_frame(history: List[dict], frame_index: int) -> Tuple[dict, int]:
    if frame_index < len(history):
        return history[frame_index], frame_index
    if history:
        return history[-1], frame_index
    return {}, frame_index


def save_episode_gif(
    episode_data: dict,
    frame_paths: List[Path],
    out_gif: Path,
    ms_per_frame: int = 300,
    start_hold_ms: Optional[int] = 2500,
    end_hold_ms: Optional[int] = 2000,
    max_text_chars: int = 400,
    max_width: Optional[int] = 1024,
    min_width: Optional[int] = None,
    captions: bool = True,
    preserve_quality: bool = True,
) -> bool:
    history = episode_data.get("history") or []
    status_banner = _episode_status_banner(episode_data)

    frames: List[Image.Image] = []
    num_frames = len(frame_paths)
    if num_frames == 0:
        # Fallback: try to render from history (if images embedded)
        for i, step in enumerate(history):
            img = None
            for k in ("obs", "frame", "image", "render", "extras"):
                if k in step:
                    img = _to_pil(step[k])
                    if img is not None:
                        break
            if img is None:
                continue
            lines = build_caption_lines(step, i, max_text_chars) if captions else []
            frames.append(
                _compose_captioned_frame(
                    img,
                    lines,
                    max_width=None if preserve_quality else max_width,
                    min_width=min_width,
                    status_banner=status_banner,
                )
            )
    else:
        for i, fp in enumerate(frame_paths):
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            step, step_index = _history_step_for_frame(history, i)
            lines = build_caption_lines(step, step_index, max_text_chars) if captions else []
            frames.append(
                _compose_captioned_frame(
                    img,
                    lines,
                    max_width=None if preserve_quality else max_width,
                    min_width=min_width,
                    status_banner=status_banner,
                )
            )

    if not frames:
        return False

    out_gif.parent.mkdir(parents=True, exist_ok=True)

    # Per-frame durations
    per_frame_ms = [max(20, int(ms_per_frame)) for _ in frames]
    if start_hold_ms is not None and len(per_frame_ms) > 0:
        per_frame_ms[0] = max(20, int(start_hold_ms))
    if end_hold_ms is not None and len(per_frame_ms) > 0:
        per_frame_ms[-1] = max(20, int(end_hold_ms))

    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=per_frame_ms,
        disposal=2,
        optimize=True,
        quality=100,
    )
    return True


def save_generated_data_gif(
    episode_data: dict,
    frame_paths: List[Path],
    out_gif: Path,
    ms_per_frame: int = 300,
    start_hold_ms: Optional[int] = 2500,
    end_hold_ms: Optional[int] = 2000,
    max_text_chars: int = 400,
    max_width: Optional[int] = 1024,
    min_width: Optional[int] = None,
    captions: bool = True,
    preserve_quality: bool = True,
) -> bool:
    history = episode_data.get("history") or []
    status_banner = _episode_status_banner(episode_data)

    frames: List[Image.Image] = []
    generated_frames: List[Tuple[int, dict, Image.Image]] = []
    for i, step in enumerate(history):
        img = None
        for k in ("generated_image",):
            if k in step:
                img = _to_pil(step[k])
                if img is not None:
                    break
        if img is not None:
            generated_frames.append((i, step, img))

    if generated_frames:
        for i, step, img in generated_frames:
            lines = build_caption_lines(step, i, max_text_chars) if captions else []
            frames.append(
                _compose_captioned_frame(
                    img,
                    lines,
                    max_width=None if preserve_quality else max_width,
                    min_width=min_width,
                    status_banner=status_banner,
                )
            )
    elif frame_paths:
        for i, fp in enumerate(frame_paths):
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            step, step_index = _history_step_for_frame(history, i)
            lines = build_caption_lines(step, step_index, max_text_chars) if captions else []
            frames.append(
                _compose_captioned_frame(
                    img,
                    lines,
                    max_width=None if preserve_quality else max_width,
                    min_width=min_width,
                    status_banner=status_banner,
                )
            )

    if not frames:
        return False

    out_gif.parent.mkdir(parents=True, exist_ok=True)

    # Per-frame durations
    per_frame_ms = [max(20, int(ms_per_frame)) for _ in frames]
    if start_hold_ms is not None and len(per_frame_ms) > 0:
        per_frame_ms[0] = max(20, int(start_hold_ms))
    if end_hold_ms is not None and len(per_frame_ms) > 0:
        per_frame_ms[-1] = max(20, int(end_hold_ms))

    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=per_frame_ms,
        disposal=2,
        optimize=True,
        quality=100,
    )
    return True


# -------------------------- CLI --------------------------

def _derive_out_path(
    episode_dir: Path,
    out_root: Optional[Path],
    input_root: Optional[Path],
    env_id: Optional[str],
    reward_str: str,
) -> Path:
    """Compute output GIF path, appending reward to filename, optionally mirroring tree."""
    filename = f"episode_r{reward_str}.gif"
    if out_root is None:
        return episode_dir / filename
    if input_root is not None:
        try:
            rel = episode_dir.relative_to(input_root)
            # Save directly under the parent (e.g., env folder), name includes episode folder
            dest_dir = out_root / rel.parent
            dest_name = f"{rel.name}_r{reward_str}.gif"
            return dest_dir / dest_name
        except Exception:
            pass
    tag = episode_dir.name
    if env_id:
        tag = f"{env_id.replace('/', '__')}__{tag}"
    return out_root / f"{tag}_r{reward_str}.gif"


def main():
    parser = argparse.ArgumentParser("VLM-Gym GIF visualizer (from renders folder + episode_stats.json)")
    parser.add_argument("--input", required=True, help="Path to episode_stats.json, an episode dir, or a root dir with episodes.")
    parser.add_argument("--out-root", default=None, help="Optional root to mirror outputs instead of writing into episode dirs.")
    parser.add_argument("--ms-per-frame", type=int, default=800)
    parser.add_argument("--start-hold-ms", type=int, default=2500)
    parser.add_argument("--end-hold-ms", type=int, default=2000)
    parser.add_argument("--max-text-chars", type=int, default=400)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--no-captions", action="store_true")
    parser.add_argument("--preserve-quality", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N episodes processed.")
    args_local = parser.parse_args()

    input_path = Path(args_local.input).resolve()
    out_root = Path(args_local.out_root).resolve() if args_local.out_root else None

    count = 0
    for json_path in find_episode_jsons(input_path):
        try:
            data, episode_dir = load_episode(json_path)
            frame_paths = list_render_images(episode_dir)
            # Build reward string for filename
            try:
                reward_val = data.get("stats", {}).get("reward", None)
            except Exception:
                reward_val = None
            if isinstance(reward_val, (int, float)):
                reward_str = f"{float(reward_val):.3f}"
            else:
                reward_str = "NA"
            env_id = data.get("env_id")
            input_root = input_path if input_path.is_dir() else input_path.parent
            out_gif = _derive_out_path(
                episode_dir=episode_dir,
                out_root=out_root,
                input_root=input_root,
                env_id=env_id,
                reward_str=reward_str,
            )

            ok = save_episode_gif(
                episode_data=data,
                frame_paths=frame_paths,
                out_gif=out_gif,
                ms_per_frame=args_local.ms_per_frame,
                start_hold_ms=args_local.start_hold_ms,
                end_hold_ms=args_local.end_hold_ms,
                max_text_chars=args_local.max_text_chars,
                max_width=args_local.max_width,
                captions=(not args_local.no_captions),
                preserve_quality=args_local.preserve_quality,
            )
            if ok:
                print(f"[OK] {out_gif}")
            else:
                print(f"[WARN] No frames found for {episode_dir}. Skipped.")
        except Exception as e:
            print(f"[ERROR] {json_path}: {e}")
        count += 1
        if args_local.limit and count >= args_local.limit:
            break

    print("Done.")


if __name__ == "__main__":
    main()
