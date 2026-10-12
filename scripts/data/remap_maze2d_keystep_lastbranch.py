#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""Remap the existing maze2d CoT data to the new 'last decisive branch before goal' key step,
WITHOUT regenerating the dataset SET. Reuses the existing retained plain_v2 rows (their
__bagel_order_key / global_index / init_state / actions are fixed), recomputes the key step
with the edited select_key_step (lastbranch v3), and re-builds ONLY the CoT rows via the
canonical build_pair_payload (so the old key step automatically reverts to a trivial step and
the new last-branch step gets the 4 imagined branch images). No re-collection, no re-dedup,
no retain step -> __bagel_order_key alignment with plain is guaranteed identical.

Output: maze2d_{easy,hard}_native256_stopreq_stepmsg_fixedstart_cot_lastbranch_ordered_<DATE>
(training/ + testing/), aligned 1:1 by __bagel_order_key with the existing plain_v2 dataset.
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import multiprocessing as mp
import scripts.data.build_maze2d_native256_ordered_pair as B

HF = ROOT / "datasets"
PLAIN_V2 = {
    "easy": "maze2d_easy_native256_stopreq_stepmsg_fixedstart_ordered_20260606_stepmsg_fixedstart_v2",
    "hard": "maze2d_hard_native256_stopreq_stepmsg_fixedstart_ordered_20260606_stepmsg_fixedstart_v2",
}


def read_rows_in_order(root: str, split_dir: str):
    rows = []
    for sh in sorted(glob.glob(str(HF / root / split_dir / "*.jsonl.gz"))):
        with gzip.open(sh, "rt") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: r[B.ORDER_KEY])  # canonical order
    return rows


def entry_from_plain_row(row: dict) -> dict:
    md = row.get("metadata", {})
    init_state = row["init_state"]
    actions = [s["action"] for s in row["trajectory_steps"]]
    maze_map = np.array(init_state["maze_map"], dtype=int)
    start = tuple(int(x) for x in init_state["agent_location"])
    target = tuple(int(x) for x in init_state["target_location"])
    path = B.shortest_path(maze_map, start, target)
    init_hash = row["hash"]
    key_step = B.select_key_step(maze_map, path, actions, target, init_hash)
    if key_step is None:
        raise RuntimeError(f"no key step for order_key={row.get(B.ORDER_KEY)} (should be impossible for an accepted row)")
    return {
        "manifest_version": B.MANIFEST_VERSION,
        "policy_version": B.POLICY_VERSION,
        "difficulty": md.get("difficulty"),
        "split": row.get("split"),
        "seed": md.get("seed"),
        "env_id": row["env_id"],
        "maze_width": md.get("maze_width"),
        "maze_height": md.get("maze_height"),
        "target_in_center": md.get("target_in_center"),
        "render_size": md.get("render_size"),
        "hash": init_hash,
        "init_state": init_state,
        "path": [list(c) for c in path],
        "actions": actions,
        "key_step": key_step,
        B.ORDER_KEY: row[B.ORDER_KEY],
        "global_index": row["alignment_index"]["global_index"],
    }


def _cot_bytes(task):
    entry, quality = task
    _plain, cot = B.build_pair_payload((entry, quality))
    return entry[B.ORDER_KEY], cot


def write_shards(items, out_dir: Path, n_shards: int):
    """items: list of (order_key, cot_bytes) already in order. Contiguous split into n_shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(items)
    per = (total + n_shards - 1) // n_shards
    written = 0
    for si in range(n_shards):
        chunk = items[si * per:(si + 1) * per]
        if not chunk and si > 0:
            continue
        with gzip.open(out_dir / f"shard_{si:05d}.jsonl.gz", "wt") as f:
            for _ok, cb in chunk:
                line = cb.decode() if isinstance(cb, (bytes, bytearray)) else cb
                if not line.endswith("\n"):
                    line += "\n"
                f.write(line)
        written += len(chunk)
    return written


def process(diff: str, out_root: Path, workers: int, quality: int, n_train_shards: int, limit: int | None):
    for split_dir, n_sh in (("training", n_train_shards), ("testing", 1)):
        rows = read_rows_in_order(PLAIN_V2[diff], split_dir)
        if limit:
            rows = rows[:limit]
        entries = [entry_from_plain_row(r) for r in rows]
        print(f"[{diff}/{split_dir}] {len(entries)} rows -> building cot_v3 with {workers} workers", flush=True)
        tasks = [(e, quality) for e in entries]
        results = []
        with mp.Pool(processes=workers) as pool:
            for i, res in enumerate(pool.imap(_cot_bytes, tasks, chunksize=16)):
                results.append(res)
                if (i + 1) % 5000 == 0:
                    print(f"  [{diff}/{split_dir}] {i+1}/{len(tasks)}", flush=True)
        # results already in submit order (imap preserves order) == order_key order
        w = write_shards(results, out_root / split_dir, n_sh)
        print(f"[{diff}/{split_dir}] wrote {w} cot rows -> {out_root/split_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--difficulty", choices=("easy", "hard", "both"), default="both")
    ap.add_argument("--date-tag", default="lastbranch_v3")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--train-shards", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap rows per split")
    ap.add_argument("--input-base", required=True, help="Parent directory of the retained plain datasets")
    ap.add_argument("--out-base", required=True)
    args = ap.parse_args()
    global HF
    HF = Path(args.input_base).expanduser().resolve()
    diffs = ["easy", "hard"] if args.difficulty == "both" else [args.difficulty]
    summary = {}
    for diff in diffs:
        out_root = Path(args.out_base) / f"maze2d_{diff}_native256_stopreq_stepmsg_fixedstart_cot_lastbranch_ordered_{args.date_tag}"
        process(diff, out_root, args.workers, args.quality, args.train_shards, args.limit)
        summary[diff] = str(out_root)
    print("DONE", json.dumps(summary))


if __name__ == "__main__":
    main()
