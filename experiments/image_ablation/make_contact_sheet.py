#!/usr/bin/env python3
"""Stitch first-episode context images in exact model input order for visual checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def make_sheet(run_dir: Path, limit: int, out: Path, columns: int = 12) -> None:
    episodes = sorted(run_dir.rglob("context_sequence.json"))[:limit]
    if not episodes:
        raise FileNotFoundError(f"No trace context_sequence.json files under {run_dir}")
    if columns < 1:
        raise ValueError("columns must be positive")
    tile = 180
    margin = 16
    label_h = 40
    sequences = [json.loads(path.read_text()) for path in episodes]
    block_heights = [label_h + ((len(seq) + columns - 1) // columns) *
                     (tile + label_h + margin) + margin for seq in sequences]
    sheet = Image.new("RGB", (margin + columns * (tile + margin),
                              margin + sum(block_heights)), "white")
    draw = ImageDraw.Draw(sheet)
    y = margin
    for path, sequence, height in zip(episodes, sequences, block_heights):
        draw.text((margin, y), path.parent.name, fill="black")
        for image_idx, item in enumerate(sequence):
            x = margin + (image_idx % columns) * (tile + margin)
            tile_y = y + label_h + (image_idx // columns) * (tile + label_h + margin)
            with Image.open(path.parent / "context_frames" / item["file"]) as image:
                sheet.paste(image.convert("RGB").resize((tile, tile)), (x, tile_y))
            draw.text((x, tile_y + tile + 2),
                      f"{item['index']} {item['source']}", fill="black")
        y += height
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"Wrote {out} ({len(episodes)} episodes, "
          f"{max(map(len, sequences))} max context frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--columns", type=int, default=12)
    args = parser.parse_args()
    make_sheet(args.run_dir, args.limit, args.out, args.columns)
