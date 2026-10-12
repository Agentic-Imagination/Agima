"""Fetch Maze/PushT episode pools for frame ablations; task weights are not included."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download


ASSETS = Path(__file__).resolve().parent / "assets"

FILES = [
    ("novastar111/maze2d_hard_plain_ordered", "dataset", "testing/shard_00000.jsonl.gz", ASSETS / "maze_hard_plain"),
    ("novastar111/maze2d_hard_cot_ordered", "dataset", "testing/shard_00000.jsonl.gz", ASSETS / "maze_hard_cot"),
    # The artifact inventory links the two randstartgoal pools above. The Table 1
    # archive's 100 init_state_hash values instead match this fixedstart v2 pool.
    ("novastar114/maze2d_hard_native256_stepmsg_fixedstart_plain", "dataset", "testing/shard_00000.jsonl.gz", ASSETS / "maze_hard_fixedstart"),
    ("novastar111/pusht_eval_in_dist_balanced100_20260820", "dataset", "data/test/batch_000000.jsonl.gz", ASSETS / "pusht_in_dist"),
    ("novastar111/pusht_eval_in_dist_balanced100_20260820", "dataset", "manifest.json", ASSETS / "pusht_in_dist"),
]


def fetch(item):
    repo, repo_type, filename, outdir = item
    path = hf_hub_download(repo_id=repo, repo_type=repo_type, filename=filename, local_dir=outdir)
    return repo, filename, path


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fut in as_completed([pool.submit(fetch, item) for item in FILES]):
            print(*fut.result(), flush=True)
