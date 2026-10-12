#!/usr/bin/env python3
"""Compare the JPEG-normalized Original cell with published paired Table 1 traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.image_ablation.preflight import EXPECTED, ROOT, archived_stats
from experiments.image_ablation.summarize import mcnemar, wilson


def check(task: str) -> dict:
    published = archived_stats(task, "cot")
    run_dir = ROOT / "runs" / f"{task}_original"
    current = [json.loads(path.read_text())["result"]
               for path in run_dir.rglob("episode_stats.json")]
    expected = EXPECTED[task]["n"]
    if len(current) != expected:
        raise RuntimeError(f"{task} Original incomplete: {len(current)}/{expected}")
    current_by_id = {item["episode_id"]: item for item in current}
    archived_by_id = {
        f"{task}:{item['global_index']:07d}:{item['episode_seed']}":
            {"success": bool(item["stats"]["success"])} for item in published
    }
    assert len(current_by_id) == expected and set(current_by_id) == set(archived_by_id)
    successes = sum(item["success"] for item in current)
    published_successes = sum(item["success"] for item in archived_by_id.values())
    low, high = wilson(successes, expected)
    errors = [error["pixel_mse_0_1"] for item in current for error in item["pixel_mse"]]
    result = {
        "task": task, "n": expected, "new_successes": successes,
        "published_table1_successes": published_successes,
        "new_rate": successes / expected,
        "new_wilson95": [low, high],
        "published_rate": published_successes / expected,
        "paired_exact_test": mcnemar(current_by_id, archived_by_id),
        "original_oracle_frames": len(errors),
        "original_oracle_mean_pixel_mse_0_1": sum(errors) / len(errors) if errors else None,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=tuple(EXPECTED))
    args = parser.parse_args()
    result = check(args.task)
    output = ROOT / "output" / f"{args.task}_original_alignment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
