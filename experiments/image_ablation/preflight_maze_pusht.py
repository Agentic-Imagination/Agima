"""Check episode pairing and export baselines from staged K=5 runs."""

import gzip
import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ARCHIVES = ASSETS / "table1_maze_pusht" / "table1"
CONFIG = {
    "maze": {
        "eval": ASSETS / "maze_hard_fixedstart/testing/shard_00000.jsonl.gz",
        "cot": ARCHIVES / "maze2d/ordinary_hard/k5_cot.tar.gz",
        "noncot": ARCHIVES / "maze2d/ordinary_hard/k5_noncot.tar.gz",
        "budget": 40,
    },
    "pusht": {
        "eval": ASSETS / "pusht_in_dist/data/test/batch_000000.jsonl.gz",
        "cot": ARCHIVES / "pusht/in_dist/k5_cot.tar.gz",
        "noncot": ARCHIVES / "pusht/in_dist/k5_noncot.tar.gz",
        "budget": 30,
    },
}


def archived_stats(path):
    with tarfile.open(path) as archive:
        stats = [json.load(archive.extractfile(member)) for member in archive
                 if member.isfile() and member.name.endswith("/episode_stats.json")]
    return sorted(stats, key=lambda row: int(row["global_index"]))


def main():
    summaries, baselines = [], []
    for task, spec in CONFIG.items():
        with gzip.open(spec["eval"], "rt") as stream:
            episodes = [json.loads(line) for line in stream if line.strip()]
        assert len(episodes) == 100
        traces = {arm: archived_stats(spec[arm]) for arm in ("cot", "noncot")}
        for arm, records in traces.items():
            assert len(records) == len(episodes)
            assert all(row["chunk_k"] == 5 and row["instruction_max_steps"] == spec["budget"]
                       for row in records)
        for index, episode in enumerate(episodes):
            for arm in ("cot", "noncot"):
                result = traces[arm][index]
                assert int(result["global_index"]) == index
                if task == "maze":
                    assert int(result["seed"]) == int(episode["alignment_index"]["seed"])
                    assert result["alignment_index"]["init_state_hash"] == episode["alignment_index"]["init_state_hash"]
                    assert episode["metadata"]["max_steps"] == spec["budget"]
                else:
                    assert int(result["episode_seed"]) == int(episode["episode_seed"])
                    assert episode["env_kwargs"]["max_steps"] == spec["budget"]
            noncot = traces["noncot"][index]
            baselines.append({"episode_id": f"{task}:{index:07d}:{noncot['episode_seed']}",
                              "task": task, "condition": "No-Imagine",
                              "success": bool(noncot["stats"]["success"]),
                              "global_index": index, "seed": noncot["seed"],
                              "episode_seed": noncot["episode_seed"],
                              "source": "staged_k5_noncot"})
        summaries.append({"task": task, "n": 100, "action_budget": spec["budget"],
                          "archive_original_successes": sum(bool(x["stats"]["success"]) for x in traces["cot"]),
                          "archive_noncot_successes": sum(bool(x["stats"]["success"]) for x in traces["noncot"]),
                          "eval_sha256": hashlib.sha256(spec["eval"].read_bytes()).hexdigest(),
                          "paired_episode_order": True})
    out = ROOT / "prepared"
    out.mkdir(exist_ok=True)
    (out / "maze_pusht_preflight.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (out / "maze_pusht_baseline.jsonl").open("w") as stream:
        for row in baselines:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
