#!/usr/bin/env python3
"""Verify episode pairing between test pools and staged K=5 traces."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
EXPECTED = {
    "sokoban": {"n": 100, "budget": 40,
                "sha256": "e3b62458901c695811514089e82c908bcabbaf7421fdfce5a78c95371369d1bd"},
    "pacman": {"n": 100, "budget": 120,
               "sha256": "9a98fa58b2466beead45acc876a28ee2b0ae258a9be3f77bcf4f3c190e6c6a16"},
}


def rows_for(task: str) -> tuple[list[dict], str]:
    path = ASSETS / f"{task}_hard_test.jsonl.gz"
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return rows, sha256


def archived_stats(task: str, arm: str) -> list[dict]:
    path = ASSETS / f"{task}_k5_{arm}.tar.gz"
    with tarfile.open(path, "r:gz") as archive:
        stats = [json.load(archive.extractfile(member)) for member in archive
                 if member.isfile() and member.name.endswith("/episode_stats.json")]
    return sorted(stats, key=lambda item: item["global_index"])


def check_task(task: str) -> tuple[dict, list[dict]]:
    expected = EXPECTED[task]
    rows, sha256 = rows_for(task)
    assert sha256 == expected["sha256"], f"{task}: eval shard checksum changed"
    assert len(rows) == expected["n"], f"{task}: expected {expected['n']} test episodes"
    keys = [(row["seed"], row["episode_seed"]) for row in rows]
    assert len(set(keys)) == len(keys), f"{task}: duplicate episode key"
    budgets = Counter(int(row["env_kwargs"]["max_steps"]) for row in rows)
    assert budgets == {expected["budget"]: expected["n"]}, (task, budgets)
    if task == "sokoban":
        assert all(row["num_boxes"] == 4 for row in rows)
    else:
        assert all(row["init_state"]["grid_size"] == 12 and
                   row["init_state"]["food_count"] == 12 and
                   row["init_state"].get("braid") is True and
                   row["init_state"].get("thin_corridors") is True and
                   row["env_kwargs"]["ghost_spawn_dist"] == 3 for row in rows)

    runs = {arm: archived_stats(task, arm) for arm in ("cot", "noncot")}
    for arm, stats in runs.items():
        assert len(stats) == len(rows), (task, arm, len(stats))
        assert [(x["seed"], x["episode_seed"]) for x in stats] == keys, (task, arm, "order mismatch")
        assert all(x["chunk_k"] == 5 and x["instruction_max_steps"] == expected["budget"]
                   for x in stats), (task, arm, "K/budget mismatch")

    baseline = []
    for row, stat in zip(rows, runs["noncot"]):
        baseline.append({
            "episode_id": f"{task}:{stat['global_index']:07d}:{stat['episode_seed']}",
            "task": task, "condition": "No-Imagine", "success": bool(stat["stats"]["success"]),
            "global_index": stat["global_index"], "seed": stat["seed"],
            "episode_seed": stat["episode_seed"], "num_executed_actions": stat["n_executed"],
            "num_chunks": stat["n_chunks"], "source": "staged_k5_noncot",
        })
    summary = {
        "task": task, "n": len(rows), "action_budget": expected["budget"],
        "eval_sha256": sha256,
        "archive_original_successes": sum(bool(x["stats"]["success"]) for x in runs["cot"]),
        "archive_noncot_successes": sum(bool(x["stats"]["success"]) for x in runs["noncot"]),
        "paired_episode_order": True,
        "seed_first": keys[0], "seed_last": keys[-1],
    }
    return summary, baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prepared")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, baselines = zip(*(check_task(task) for task in EXPECTED))
    (args.output_dir / "preflight.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (args.output_dir / "baseline.jsonl").open("w") as stream:
        for row in (item for rows in baselines for item in rows):
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
