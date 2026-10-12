#!/usr/bin/env python3
"""Check that saved context images match the selected source and real renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.actions import _jpeg_roundtrip


def pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def expected_real(ep_dir: Path, action_count: int, quality: int = 95) -> np.ndarray:
    path = ep_dir / "renders" / f"step_{action_count:04d}.png"
    with Image.open(path) as image:
        return np.asarray(_jpeg_roundtrip(image, quality, "JPEG"))


def expected_stale(ep_dir: Path, action_count: int, result: dict) -> np.ndarray:
    quality = int(result.get("image_quality", 95))
    frame = Image.fromarray(expected_real(ep_dir, action_count, quality))
    target = tuple(result.get("imagined_image_size", frame.size))
    if frame.size != target:
        frame = frame.resize(target, Image.Resampling.BICUBIC)
        if result.get("imagined_codec", "jpeg") == "jpeg":
            frame = _jpeg_roundtrip(frame, quality, "JPEG")
    return np.asarray(frame)


def audit_episode(stats_path: Path) -> dict:
    ep_dir = stats_path.parent
    stats = json.loads(stats_path.read_text())
    result = stats["result"]
    quality = int(result.get("image_quality", 95))
    condition = result["condition"]
    source = {"Original": "original", "Blank": "blank", "Stale": "stale",
              "Oracle GT": "oracle"}[condition]
    sequence = json.loads((ep_dir / "context_sequence.json").read_text())
    chunks = json.loads((ep_dir / "chunks.json").read_text())
    encoder_audits = json.loads((ep_dir / "image_encoding_audit.json").read_text())
    assert len(chunks) == result["num_chunks"]
    assert all((audit["generation_skipped"] and audit["generated"] is None
                if result["image_generation_skipped"] else
                audit["identical"] and audit["generated"] == audit["context"])
               for audit in encoder_audits)
    assert sequence[0]["source"] == "real_initial"
    assert np.array_equal(pixels(ep_dir / "context_frames" / sequence[0]["file"]),
                          expected_real(ep_dir, 0, quality))
    position = 1
    executed = 0
    imagined = 0
    boundaries = 0
    for ci, chunk in enumerate(chunks):
        stale = expected_stale(ep_dir, executed, result)
        for j in range(chunk["used_count"]):
            item = sequence[position]
            assert item["source"] == source
            selected = pixels(ep_dir / f"chunk_{ci:03d}" / f"used_{j:03d}.png")
            context = pixels(ep_dir / "context_frames" / item["file"])
            assert np.array_equal(context, selected), (ep_dir.name, ci, j, "context mismatch")
            if condition == "Original":
                generated = pixels(ep_dir / f"chunk_{ci:03d}" / f"generated_{j:03d}.png")
                assert np.array_equal(context, generated)
            elif condition == "Blank":
                assert np.all(context == 128)
            elif condition == "Stale":
                assert np.array_equal(context, stale)
            else:
                oracle = pixels(ep_dir / f"chunk_{ci:03d}" / f"oracle_{j:03d}.png")
                assert np.array_equal(context, oracle)
            position += 1
            imagined += 1
        executed += len(chunk["executed_actions"])
        if ci < len(chunks) - 1:
            item = sequence[position]
            assert item["source"] == "real_boundary"
            assert np.array_equal(pixels(ep_dir / "context_frames" / item["file"]),
                                  expected_real(ep_dir, executed, quality))
            position += 1
            boundaries += 1
    assert position == len(sequence)
    assert executed == result["num_executed_actions"]
    assert imagined == result["n_imagined"]
    return {"episode_id": result["episode_id"], "condition": condition,
            "imagined_frames": imagined, "real_boundaries": boundaries,
            "encoder_audits": len(encoder_audits), "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.run_dir.rglob("episode_stats.json"))[:args.limit]
    if len(paths) != args.limit:
        raise RuntimeError(f"Need {args.limit} completed traces, found {len(paths)}")
    results = [audit_episode(path) for path in paths]
    destination = args.output or args.run_dir / "canary_audit.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Passed {len(results)} full context-image audits: "
          f"{sum(item['imagined_frames'] for item in results)} imagined frames, "
          f"{sum(item['real_boundaries'] for item in results)} real boundaries")


if __name__ == "__main__":
    main()
