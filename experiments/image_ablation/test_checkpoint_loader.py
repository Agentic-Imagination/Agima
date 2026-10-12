#!/usr/bin/env python3
"""The published fp32 EMA index must stage with both safetensors shards."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import evaluation.base as base
import evaluation.runtime as G


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        src = root / "checkpoint"
        src.mkdir()
        (src / "ema-00001-of-00002.safetensors").write_bytes(b"shard1")
        (src / "ema-00002-of-00002.safetensors").write_bytes(b"shard2")
        index = {"weight_map": {"a": "ema-00001-of-00002.safetensors",
                                "b": "ema-00002-of-00002.safetensors"}}
        (src / "ema.safetensors.index.json").write_text(json.dumps(index))
        assert Path(base._resolve_checkpoint_path(str(src))).name == "ema.safetensors.index.json"
        args = SimpleNamespace(stage_checkpoint=True, bagel_path=str(src),
                               local_ckpt_root=str(root / "staged"), stage_timeout_sec=3)
        staged = Path(G.stage_checkpoint_if_requested(args, local_rank=0, rank=0))
        for name in ["ema.safetensors.index.json", *set(index["weight_map"].values())]:
            assert (staged / name).read_bytes() == (src / name).read_bytes()
        print("Sharded EMA index and both shards staged correctly")


if __name__ == "__main__":
    main()
