"""Exchange evaluation summaries for one torchrun worker-group attempt."""

import datetime as dt
import json
import os
from pathlib import Path
import time
import uuid


_LOCAL_SESSION = uuid.uuid4().hex


def summary_session_id():
    run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    if run_id and run_id != "none":
        return f"{run_id}:{os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')}"
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Distributed evaluation requires a unique torchrun rendezvous ID; "
                         "use --standalone or set --rdzv-id for this job.")
    return _LOCAL_SESSION


def write_rank_summary(outdir: Path, rank: int, summary: dict):
    directory = outdir / "_rank_summaries"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank_{rank:05d}.json"
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    record = dict(summary, rank=rank, session_id=summary_session_id())
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_rank_summary(outdir, rank):
    path = outdir / "_rank_summaries" / f"rank_{rank:05d}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(record, dict) or record.get("session_id") != summary_session_id()
            or record.get("rank") != rank):
        return None
    return record


def wait_for_rank_summaries(outdir: Path, world_size: int, timeout_sec: int):
    deadline = time.monotonic() + float(timeout_sec)
    while True:
        missing = [rank for rank in range(world_size) if _read_rank_summary(outdir, rank) is None]
        if not missing:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for current-run rank summaries {missing} "
                               f"in {outdir / '_rank_summaries'}")
        time.sleep(min(5.0, remaining))


def aggregate_summaries(outdir: Path, world_size: int) -> dict:
    merged = {
        "session_id": summary_session_id(), "world_size": world_size,
        "episodes": 0, "completed": 0, "skipped": 0, "successes": 0,
        "errors": 0, "rank_summaries": [],
    }
    for rank in range(world_size):
        item = _read_rank_summary(outdir, rank)
        if item is None:
            raise ValueError(f"Missing current-run summary for rank {rank}; cannot aggregate {outdir}")
        merged["rank_summaries"].append(item)
        for key in ("episodes", "completed", "skipped", "successes", "errors"):
            merged[key] += int(item.get(key, 0))
    merged["done"] = merged["completed"] + merged["skipped"]
    complete = (merged["episodes"] > 0 and merged["errors"] == 0
                and merged["done"] == merged["episodes"])
    merged["status"] = "complete" if complete else "incomplete"
    merged["success_rate"] = merged["successes"] / merged["done"] if complete else None
    merged["timestamp"] = dt.datetime.now().isoformat()
    write_summary(outdir, merged)
    return merged


def write_summary(outdir: Path, summary: dict):
    path = outdir / "summary.json"
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
