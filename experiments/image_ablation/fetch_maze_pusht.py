"""Fetch public Maze2D ablation pools; original PushT ablation inputs are supplied separately."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download


ASSETS = Path(__file__).resolve().parent / "assets"

FILES = [
    ("novastar114/maze2d_hard_native256_stepmsg_fixedstart_plain", "dataset", "testing/shard_00000.jsonl.gz", ASSETS / "maze_hard_fixedstart"),
]


def fetch(item):
    repo, repo_type, filename, outdir = item
    path = hf_hub_download(repo_id=repo, repo_type=repo_type, filename=filename, local_dir=outdir)
    return repo, filename, path


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fut in as_completed([pool.submit(fetch, item) for item in FILES]):
            print(*fut.result(), flush=True)
