#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""FWD2 -- Exp3 "forward-thinking" sokoban CoT-inf dataset remap (stage 2).

Consumes the stage-1 qualifying-level dataset written by
scripts/data/sokoban_forward_level_gen.py (<levels-src>/levels/shard_*.jsonl.gz; one row
per FORWARD-PLANNING qualifier: {init_state, init_frame_b64,
far_plan:{actions,frames_b64}, near_plan:{actions,frames_b64,end_state}, near_box,
far_box, checks, ...}) and emits CoT-inf SINGLE-TURN rows (2 messages) whose private
<think> IMAGINES BOTH box orders before committing the one that works:

    user:      [instruction -- byte-identical to the vanilla CoT-inf rows, frame0 (loss=0)]
    assistant: <think(auto-prepended by the loader; NOT in the row)>
               Two boxes need placing; let me imagine each order before committing.
               Option: start with the box at (r1, c1).      <- the two plan rollouts come in
                 a_1 <img> a_2 <img> ...                        SHUFFLED order (~50/50, seeded
               <templated qualitative END line>                 per-row via hashlib on
               Option: start with the box at (r2, c2).          init_state_hash -- no
                 a_1 <img> ...                                  positional leak); the headers
               <templated qualitative END line>                 are format-identical and
               Comparing the two orders, starting with the      NEVER say near/far
               box at (rF, cF) is the one that solves the
               puzzle; committing that plan.
               Done imagining; committing these actions:       <- THINK_CLOSE (chunk family)
               </think>
               {far_plan.actions "\n"-joined -- ends with ('stop', 'stop')}

INSTRUCTION SOURCE (byte-identity with the vanilla arm). The vanilla CoT-inf remap
(remap_sokoban_cot_chunk_inf.py) builds the user text as
cot_inf_instruction(history[0]['prompt'], max_steps). The v8 source rows' step-0 prompt is
EXACTLY  env.unwrapped.get_prompt() + "\n" + "This is step 1. You are allowed to take
{max_steps-1} more steps."  (visgym_training/sokoban_dataset_pipeline.py::_step_prompt
step_idx==0 branch -> _default_prompt -> gymnasium/envs/sokoban/sokoban.py::get_prompt;
byte-verified against the ordered v8 source). Stage-1 level rows carry no prompt, so
step0_prompt() rebuilds it from the SAME env (env_id + env_kwargs travel on every level
row) and feeds the SAME imported cot_inf_instruction() -- the user turn is byte-identical
to the vanilla rows for the same board (self-test check 3 asserts this, frame0 included:
render_branch_frame is byte-compatible with the recorded v8 frames).

MIXING (confound control): the output is NOT 100% trap levels. --forward-frac (default
0.55) of --total-rows (default 100000) are forward rows; the rest are VERBATIM vanilla
CoT-inf rows (--vanilla-src; messages untouched, stamped has_forward=False), evenly
interleaved by a deterministic Bresenham slot rule. Forward levels are repeat-sampled
(round-robin: forward index i -> level i % pool, repeat i // pool) ONLY if the stage-1
pool is smaller than forward_frac*total; the repeat factor is recorded. At the defaults
the completed stage-1 pool (55,128 unique levels) covers 0.55*100000 = 55,000 forward
rows with ZERO repeat-sampling (repeat_factor 1.0; the last 128 levels go unused). Every row is
stamped {has_forward, forward_first_tried, plan_lens, schema_version='exp3_forward_v1'};
the base CoT-inf marker moves to cot_schema_version (branch-remap convention).

LOADER: VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb',
data/interleave_datasets/edit_dataset_envfb.py). Every assistant plan frame is loss=True
(the envfb loader DROPS loss=False assistant images, edit_dataset_envfb.py:69-74); frame0
in the user turn is the ONLY loss-0 image. CE on all assistant text (opener, headers,
actions, end lines, conclusion, committed plan); MSE on every imagined frame of BOTH
rollouts.

TOKEN BUDGET: worst forward row ~ (near <=~10 + far <=20 frames) * ~840 tok + text << the
50k packing budget; rows whose estimate would exceed it are dropped and counted.

PARALLELISM: the driver Pools over MANY CONTIGUOUS OUTPUT ROW RANGES
(workers*tasks_per_worker tasks, NEVER a Pool bounded by the shard count). The slot rule
is monotone, so each output range maps to contiguous forward-level and vanilla row ranges;
each worker streams exactly those input rows and writes an independent gzip part
(.inprogress -> os.replace), and completed parts are concatenated IN RANGE ORDER (gzip
members concatenate legally) into >= --n-shards training/batch_XXXXXX.jsonl.gz shards
(.tmp -> os.replace). metadata/forward_summary.json records the mix.

Self-test (writes nothing):
  python scripts/data/remap_sokoban_cot_chunk_inf_forward.py --self-test 26 \
      --levels-src /path/to/levels --vanilla-src /path/to/vanilla \
      --out-root /path/to/output --src-ordered /path/to/ordered

Original data untouched; writes a NEW local repo. NEW file -- modifies no existing
script / env / loader.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the EXACT shared text builders so the user turn / think tail are byte-identical to
# the vanilla CoT-inf rows (train==eval). Robust to both run-as-script and
# import-as-scripts.data.<mod> contexts, matching the other remap modules.
try:
    from remap_sokoban_cot_chunk_inf import (
        cot_inf_instruction,
        transform_row as build_cot_inf_row,
        DEFAULT_SRC as ORDERED_SRC,
    )
    from remap_sokoban_cot_chunk_k import THINK_CLOSE
except ImportError:  # pragma: no cover - import-path shim
    from scripts.data.remap_sokoban_cot_chunk_inf import (
        cot_inf_instruction,
        transform_row as build_cot_inf_row,
        DEFAULT_SRC as ORDERED_SRC,
    )
    from scripts.data.remap_sokoban_cot_chunk_k import THINK_CLOSE

# ---------------------------------------------------------------------------
# Constants + fixed templates (Exp3 forward format; shared with the self-test)
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "exp3_forward_v1"
COT_BASE_SCHEMA = "sokoban_cot_perstep_kinf_v1"   # the CoT-inf single-turn family marker
STOP_ACTION = str(("stop", "stop"))

DATE_TAG = os.environ.get("DATE_TAG", "20260710_fwd")
DEFAULT_LEVELS_SRC = None
DEFAULT_VANILLA_SRC = None
DEFAULT_OUT = None

# seg0 opener: the loader PREPENDS '<think>' to the first assistant text seg, so the row
# text itself must NOT contain the literal '<think>'.
FWD_OPENER = "\nTwo boxes need placing; let me imagine each order before committing.\n"

# Format-identical option headers (coordinates only -- NEVER the words near/far, and the
# order is shuffled per row, so nothing before the rollouts reveals which order wins).
def option_header(box) -> str:
    return f"Option: start with the box at ({int(box[0])}, {int(box[1])})."

# Templated qualitative END lines (CE text). The near-first line states the certified
# plan-level failure; the far-first line states success. Neither names near/far.
NEAR_END_LINE = ("\nWith that box placed first, the remaining box can no longer be pushed "
                 "to its target — this order fails.")
FAR_END_LINE = "\nBoth boxes reach targets — this order works."


def conclusion(win_box) -> str:
    """The ONLY place the winning box is named (by its coordinates)."""
    return ("\nComparing the two orders, starting with the box at "
            f"({int(win_box[0])}, {int(win_box[1])}) is the one that solves the puzzle; "
            "committing that plan.")


# ---------------------------------------------------------------------------
# Instruction reconstruction for generated levels (byte-identical to vanilla)
# ---------------------------------------------------------------------------
_PROMPT_CACHE: Dict[Tuple[str, str], str] = {}


def step0_prompt(env_id: str, env_kwargs: Optional[dict]) -> str:
    """Rebuild the source-pipeline step-0 prompt for a level that carries no prompt.

    == env.unwrapped.get_prompt() + "\\n" + "This is step 1. You are allowed to take
    {max_steps-1} more steps."  -- exactly what visgym_training/sokoban_dataset_pipeline.py
    (_default_prompt + _step_prompt, step_idx==0) stamped on the v8 source rows
    (byte-verified). cot_inf_instruction() then cuts at "This is step 1." and adds the
    inf cadence + frame0 label, so the resulting user text is byte-identical to the
    vanilla CoT-inf rows.
    """
    kw = dict(env_kwargs or {})
    key = (str(env_id), json.dumps(kw, sort_keys=True))
    if key not in _PROMPT_CACHE:
        import gymnasium as gym  # heavy; imported lazily in the workers only
        env = gym.make(env_id, **kw)
        try:
            rules = str(env.unwrapped.get_prompt())
        finally:
            env.close()
        ms = int(kw.get("max_steps", 20))
        _PROMPT_CACHE[key] = (rules + "\n"
                              + f"This is step 1. You are allowed to take {max(0, ms - 1)} more steps.")
    return _PROMPT_CACHE[key]


# ---------------------------------------------------------------------------
# The forward row builder
# ---------------------------------------------------------------------------
def forward_first_tried(level_row: dict, repeat_idx: int) -> str:
    """'near'|'far': which plan is imagined FIRST -- sha256-seeded on a stable content key
    (init_state_hash + repeat index + schema tag). ~50/50, independent of file position
    (no positional leak); repeat-sampled duplicates of a level flip independently."""
    key = f"{level_row['init_state_hash']}|rep{int(repeat_idx)}|{SCHEMA_VERSION}"
    return "near" if hashlib.sha256(key.encode("utf-8")).digest()[0] % 2 == 0 else "far"


def forward_metadata(first_tried: str, near_len: int, far_len: int) -> dict:
    return {
        "has_forward": True,
        "forward_first_tried": first_tried,
        "plan_lens": {"near": int(near_len), "far": int(far_len)},
        "schema_version": SCHEMA_VERSION,
    }


def transform_forward_row(level: dict, repeat_idx: int = 0) -> dict:
    """One stage-1 qualifier -> one Exp3 forward CoT-inf single-turn row."""
    far_actions = [str(a) for a in level["far_plan"]["actions"]]
    far_frames = list(level["far_plan"]["frames_b64"])
    near_actions = [str(a) for a in level["near_plan"]["actions"]]
    near_frames = list(level["near_plan"]["frames_b64"])
    if len(far_actions) != len(far_frames) or len(near_actions) != len(near_frames):
        raise ValueError("plan actions/frames length mismatch")
    if not far_actions or not near_actions:
        raise ValueError("empty plan")
    if far_actions[-1] != STOP_ACTION:
        raise ValueError("far plan does not end with the stop action")
    near_box = (int(level["near_box"][0]), int(level["near_box"][1]))
    far_box = (int(level["far_box"][0]), int(level["far_box"][1]))

    first = forward_first_tried(level, repeat_idx)
    plans = {
        "near": (near_box, near_actions, near_frames, NEAR_END_LINE),
        "far": (far_box, far_actions, far_frames, FAR_END_LINE),
    }
    order = (first, "far" if first == "near" else "near")

    env_kwargs = dict(level.get("env_kwargs") or {})
    max_steps = int(env_kwargs.get("max_steps", 20))
    instr = cot_inf_instruction(step0_prompt(level["env_id"], env_kwargs), max_steps)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": instr},
        # initial board: context -- the ONLY loss-0 image in the row
        {"type": "image", "image": level["init_frame_b64"], "loss": False},
    ]}]

    content: List[dict] = []
    pending = FWD_OPENER                    # loader PREPENDS '<think>' to this first seg
    for pi, name in enumerate(order):
        box, actions, frames, end_line = plans[name]
        pending += option_header(box)
        for a, f in zip(actions, frames):
            content.append({"type": "text", "text": pending + "\n" + a})
            # MSE target; the envfb loader DROPS assistant images with loss=False
            content.append({"type": "image", "image": f, "loss": True})
            pending = ""
        pending = end_line + ("\n" if pi == 0 else "")
    full_plan = "\n".join(far_actions)      # committed plan == far_plan.actions VERBATIM
    content.append({"type": "text", "text": pending + conclusion(far_box)
                    + "\n" + THINK_CLOSE + "\n</think>\n" + full_plan})
    messages.append({"role": "assistant", "content": content})  # the ONE model turn

    out = {
        "task": "sokoban_easy",
        "env_id": level.get("env_id"),
        "env_kwargs": env_kwargs,
        "seed": level.get("seed"),
        "init_state": level.get("init_state"),
        "init_state_schema": level.get("init_state_schema"),
        "init_state_hash": level.get("init_state_hash"),
        "level_source": level.get("source"),
        "natural_provenance": level.get("natural_provenance"),
        "messages": messages,
        "chunk_size": "inf",
        "n_steps": len(far_actions),
        "n_chunks": 1,
        "n_imagined_frames": len(near_frames) + len(far_frames),
        "n_feedback_turns": 0,
        "n_feedback_frames": 0,
        "max_steps": max_steps,
        "near_box": [near_box[0], near_box[1]],
        "far_box": [far_box[0], far_box[1]],
        "forward_repeat_index": int(repeat_idx),
        "cot_schema_version": COT_BASE_SCHEMA,
        "cot_row_type": "forward_two_order_cot_singleturn_openloop",
    }
    out.update(forward_metadata(first, len(near_actions), len(far_actions)))
    return out


def stamp_vanilla(row: dict) -> dict:
    """VERBATIM vanilla CoT-inf row + the Exp3 stamps (messages untouched). The base
    schema marker moves to cot_schema_version, branch-remap convention."""
    row["cot_schema_version"] = row.get("schema_version")
    row["has_forward"] = False
    row["forward_first_tried"] = None
    row["plan_lens"] = None
    row["schema_version"] = SCHEMA_VERSION
    return row


# ---------------------------------------------------------------------------
# Token-budget estimate (conservative envelope)
# ---------------------------------------------------------------------------
IMG_TOKENS_EST = 840     # per 256px frame (VAE+ViT+specials envelope; K=10 envfb rows
                         # measured ~18.6k max at ~20 frames -> ~840/frame + text)
TOKEN_BUDGET = 50000     # the packing budget


def estimate_tokens(messages) -> int:
    n_img = 0
    n_chars = 0
    for m in messages:
        for c in m.get("content", []):
            if c.get("type") == "image":
                n_img += 1
            else:
                n_chars += len(c.get("text", ""))
    # chars/3 over-counts English text tokens (~4 chars/token) => safe upper bound
    return n_img * IMG_TOKENS_EST + (n_chars + 2) // 3 + 128


# ---------------------------------------------------------------------------
# Mixing math: deterministic even interleave (monotone => contiguous range maps)
# ---------------------------------------------------------------------------
def n_forward_before(t: int, total: int, nf: int) -> int:
    """# forward slots among output positions [0, t). Bresenham spread: exactly nf forward
    slots overall, evenly spaced, and MONOTONE -- any contiguous output range [lo, hi) maps
    to the contiguous forward range [nfb(lo), nfb(hi)) and vanilla range [lo-nfb(lo),
    hi-nfb(hi))."""
    return t * nf // total


def is_forward_slot(t: int, total: int, nf: int) -> bool:
    return n_forward_before(t + 1, total, nf) > n_forward_before(t, total, nf)


# ---------------------------------------------------------------------------
# Shard IO helpers
# ---------------------------------------------------------------------------
def _count_rows(path: str) -> int:
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _cumsum(counts: List[int]) -> List[int]:
    out = [0]
    for c in counts:
        out.append(out[-1] + int(c))
    return out


def _read_lines(shards: List[str], cum: List[int], lo: int, hi: int) -> List[str]:
    """Raw non-blank JSONL lines for global row indices [lo, hi) across ordered shards."""
    out: List[str] = []
    if hi <= lo:
        return out
    for si, path in enumerate(shards):
        s0, s1 = cum[si], cum[si + 1]
        if s1 <= lo or s0 >= hi:
            continue
        a, b = max(lo, s0) - s0, min(hi, s1) - s0
        j = -1
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                j += 1
                if j >= b:
                    break
                if j >= a:
                    out.append(line)
    return out


# ---------------------------------------------------------------------------
# Pool worker: one contiguous OUTPUT row range -> one part file
# ---------------------------------------------------------------------------
_W: Dict[str, Any] = {}


def _pool_init(cfg: Dict[str, Any]) -> None:
    global _W
    _W = cfg


def _forward_levels(cfg: Dict[str, Any], f_lo: int, f_hi: int) -> List[Tuple[dict, int]]:
    """(level_row, repeat_idx) for forward indices [f_lo, f_hi), in index order.
    Forward index i -> level index i % pool_used, repeat_idx i // pool_used (round-robin
    repeat-sampling; pool_used == pool when no repeats are needed)."""
    if f_hi <= f_lo:
        return []
    pool = cfg["pool_used"]
    out: List[Tuple[dict, int]] = []
    for rep in range(f_lo // pool, (f_hi - 1) // pool + 1):
        a = max(f_lo, rep * pool) - rep * pool
        b = min(f_hi, (rep + 1) * pool) - rep * pool
        if a >= b:
            continue
        for line in _read_lines(cfg["levels_shards"], cfg["levels_cum"], a, b):
            out.append((json.loads(line), rep))
    return out


def worker_range(task: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _W
    idx, lo, hi = task["idx"], task["lo"], task["hi"]
    total, nf = cfg["total_rows"], cfg["n_forward"]
    t_wall = time.perf_counter()
    f_lo, f_hi = n_forward_before(lo, total, nf), n_forward_before(hi, total, nf)
    v_lo, v_hi = lo - f_lo, hi - f_hi
    fwd = _forward_levels(cfg, f_lo, f_hi)
    van = _read_lines(cfg["vanilla_shards"], cfg["vanilla_cum"], v_lo, v_hi)
    if len(fwd) != f_hi - f_lo:
        raise RuntimeError(f"task {idx}: materialized {len(fwd)} forward levels, "
                           f"expected {f_hi - f_lo} (levels pool changed underneath?)")
    if len(van) != v_hi - v_lo:
        raise RuntimeError(f"task {idx}: materialized {len(van)} vanilla rows, "
                           f"expected {v_hi - v_lo}")

    final_path = os.path.join(cfg["parts_dir"], f"part_{idx:06d}.jsonl.gz")
    tmp = final_path + ".inprogress"
    stats: Counter = Counter()
    first_tried: Counter = Counter()
    tok_max = 0
    err_samples: List[str] = []
    fi = vi = 0
    n_written = 0
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fout:
        for t in range(lo, hi):
            if is_forward_slot(t, total, nf):
                level, rep = fwd[fi]
                fi += 1
                try:
                    row = transform_forward_row(level, rep)
                except Exception as exc:           # never kill the task on one level
                    stats[f"fwd_error:{type(exc).__name__}"] += 1
                    if len(err_samples) < 3:
                        err_samples.append(f"fwd_idx={f_lo + fi - 1}: {exc!r}")
                    continue
                est = estimate_tokens(row["messages"])
                if est >= TOKEN_BUDGET:            # spec: over-budget rows are DROPPED
                    stats["dropped_over_token_budget"] += 1
                    continue
                tok_max = max(tok_max, est)
                first_tried[row["forward_first_tried"]] += 1
                stats["forward"] += 1
            else:
                row = stamp_vanilla(json.loads(van[vi]))
                vi += 1
                stats["vanilla"] += 1
            fout.write(json.dumps(row) + "\n")
            n_written += 1
    os.replace(tmp, final_path)                    # atomic publish
    return {"idx": idx, "part": final_path, "n_rows": n_written, "stats": dict(stats),
            "first_tried": dict(first_tried), "tok_max": tok_max,
            "err_samples": err_samples, "wall_s": time.perf_counter() - t_wall}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def concat_parts_to_shards(part_infos: List[Tuple[str, int]], training_dir: Path,
                           n_shards: int) -> List[Dict[str, Any]]:
    """Concatenate part files IN RANGE ORDER into >= n_shards gzip shard files (gzip
    members concatenate legally). batch_XXXXXX.jsonl.gz naming, .tmp + os.replace."""
    parts = [(p, k) for p, k in part_infos if k > 0]
    total = sum(k for _p, k in parts)
    training_dir.mkdir(parents=True, exist_ok=True)
    shard_files: List[Dict[str, Any]] = []
    if not parts:
        return shard_files
    n_eff = min(n_shards, len(parts))
    groups: List[List[Tuple[str, int]]] = [[] for _ in range(n_eff)]
    cum = 0
    for p, k in parts:
        gi = min(n_eff - 1, cum * n_eff // max(1, total))
        groups[gi].append((p, k))
        cum += k
    si = 0
    for group in groups:
        if not group:
            continue
        path = training_dir / f"batch_{si:06d}.jsonl.gz"
        tmp = str(path) + ".tmp"
        n_rows = 0
        with open(tmp, "wb") as fout:
            for p, k in group:
                with open(p, "rb") as fin:
                    shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
                n_rows += k
        os.replace(tmp, str(path))
        shard_files.append({"path": str(path), "rows": n_rows})
        si += 1
    return shard_files


def git_rev() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels-src", required=True,
                    help="stage-1 forward-level dataset root (sokoban_forward_level_gen.py output)")
    ap.add_argument("--levels-subdir", default="levels")
    ap.add_argument("--vanilla-src", required=True,
                    help="vanilla CoT-inf dataset to interleave verbatim (confound control)")
    ap.add_argument("--vanilla-subdir", default="training")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--total-rows", type=int, default=100_000)
    ap.add_argument("--forward-frac", type=float, default=0.55,
                    help="fraction of --total-rows that are forward (trap-level) rows "
                         "(default 0.55: the 55,128-level stage-1 pool covers 55k forward "
                         "rows with ZERO repeat-sampling)")
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    ap.add_argument("--tasks-per-worker", type=int, default=4,
                    help="pool tasks = workers * this (contiguous OUTPUT row ranges, NOT shards)")
    ap.add_argument("--src-ordered", default=ORDERED_SRC,
                    help="self-test only: ordered plain v8 source, for the natural-episode "
                         "user-turn byte-equality check")
    ap.add_argument("--self-test", type=int, default=0, metavar="N",
                    help="dry-run: build forward rows from the first N level rows + a few "
                         "vanilla rows, run all format/loader checks, write NOTHING")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return _selftest(args)

    out_root = Path(args.out_root).resolve()
    training_dir = out_root / "training"
    if training_dir.exists() and any(training_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{training_dir} already has shards (use --overwrite)")
        shutil.rmtree(training_dir)
    parts_dir = out_root / "parts_tmp"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "metadata").mkdir(parents=True, exist_ok=True)

    levels_shards = sorted(glob.glob(os.path.join(args.levels_src, args.levels_subdir, "*.jsonl.gz")))
    vanilla_shards = sorted(glob.glob(os.path.join(args.vanilla_src, args.vanilla_subdir, "*.jsonl.gz")))
    if not levels_shards:
        raise FileNotFoundError(f"no *.jsonl.gz under {args.levels_src}/{args.levels_subdir}")
    if not vanilla_shards:
        raise FileNotFoundError(f"no *.jsonl.gz under {args.vanilla_src}/{args.vanilla_subdir}")

    print(f"[FWD2] counting rows: {len(levels_shards)} level shard(s), "
          f"{len(vanilla_shards)} vanilla shard(s) ...", flush=True)
    with mp.Pool(min(args.workers, len(levels_shards) + len(vanilla_shards))) as p:
        levels_counts = p.map(_count_rows, levels_shards)
        vanilla_counts = p.map(_count_rows, vanilla_shards)
    pool_total = sum(levels_counts)
    vanilla_total = sum(vanilla_counts)

    total = int(args.total_rows)
    nf = int(round(args.forward_frac * total))
    nv = total - nf
    if pool_total <= 0:
        raise RuntimeError(f"stage-1 levels pool is empty under {args.levels_src}")
    if nv > vanilla_total:
        raise RuntimeError(f"need {nv} vanilla rows but {args.vanilla_src} has {vanilla_total}")
    pool_used = min(pool_total, nf) if nf > 0 else 0
    repeat_factor = (nf / pool_used) if pool_used else 0.0
    print(f"[FWD2] total={total} forward={nf} vanilla={nv} | levels pool={pool_total} "
          f"used={pool_used} repeat_factor={repeat_factor:.3f} | vanilla pool={vanilla_total}",
          flush=True)

    cfg = {
        "total_rows": total,
        "n_forward": nf,
        "pool_used": pool_used,
        "levels_shards": levels_shards,
        "levels_cum": _cumsum(levels_counts),
        "vanilla_shards": vanilla_shards,
        "vanilla_cum": _cumsum(vanilla_counts),
        "parts_dir": str(parts_dir),
    }
    n_tasks = max(1, min(args.workers * args.tasks_per_worker, total))
    per, rem = divmod(total, n_tasks)
    tasks = []
    lo = 0
    for i in range(n_tasks):
        hi = lo + per + (1 if i < rem else 0)
        tasks.append({"idx": i, "lo": lo, "hi": hi})
        lo = hi

    print(f"[FWD2] cores={os.cpu_count()} workers={args.workers} range-tasks={len(tasks)} "
          f"out={out_root}", flush=True)
    results: List[Dict[str, Any]] = []
    agg_stats: Counter = Counter()
    agg_first: Counter = Counter()
    tok_max = 0
    err_samples: List[str] = []
    t0 = time.perf_counter()
    with mp.Pool(min(args.workers, len(tasks)), initializer=_pool_init, initargs=(cfg,)) as pool:
        for res in pool.imap_unordered(worker_range, tasks, chunksize=1):
            results.append(res)
            agg_stats.update(res["stats"])
            agg_first.update(res["first_tried"])
            tok_max = max(tok_max, res["tok_max"])
            err_samples.extend(res["err_samples"][: max(0, 5 - len(err_samples))])
            done = len(results)
            if done % max(1, len(tasks) // 40) == 0 or done == len(tasks):
                print(f"[progress] tasks {done}/{len(tasks)} rows="
                      f"{sum(r['n_rows'] for r in results)} wall={time.perf_counter() - t0:.0f}s",
                      flush=True)

    results.sort(key=lambda r: r["idx"])
    part_infos = [(r["part"], r["n_rows"]) for r in results]
    shard_files = concat_parts_to_shards(part_infos, training_dir, args.n_shards)
    shutil.rmtree(parts_dir, ignore_errors=True)
    total_written = sum(s["rows"] for s in shard_files)

    summary = {
        "study": "exp3_forward_thinking_sokoban_cot_inf",
        "schema_version": SCHEMA_VERSION,
        "date_tag": DATE_TAG,
        "git_rev": git_rev(),
        "argv": sys.argv[1:],
        "levels_src": args.levels_src,
        "vanilla_src": args.vanilla_src,
        "total_rows": total_written,
        "target_total_rows": total,
        "n_forward_target": nf,
        "n_forward_written": int(agg_stats.get("forward", 0)),
        "n_vanilla_written": int(agg_stats.get("vanilla", 0)),
        "forward_frac_target": args.forward_frac,
        "levels_pool": pool_total,
        "levels_used_distinct": pool_used,
        "repeat_factor": repeat_factor,
        "mixing": "deterministic Bresenham interleave (is_forward_slot); forward index i -> "
                  "level i % pool_used, repeat_idx i // pool_used (repeat-sampling only when "
                  "pool < forward_frac*total)",
        "first_tried_counts": dict(agg_first),
        "dropped_over_token_budget": int(agg_stats.get("dropped_over_token_budget", 0)),
        "fwd_errors": {k: v for k, v in sorted(agg_stats.items()) if k.startswith("fwd_error:")},
        "error_samples": err_samples,
        "token_estimate_max_forward": tok_max,
        "token_budget": TOKEN_BUDGET,
        "format": "SINGLE-TURN: user(vanilla CoT-inf instruction, frame0 ctx); ONE assistant "
                  "turn = <think(prepended)> opener + TWO shuffled 'Option: start with the box "
                  "at (r, c).' rollouts (per-step action+imagined-frame) each closed by a "
                  "templated qualitative end line + conclusion naming the winning box + "
                  "THINK_CLOSE </think> {far_plan.actions verbatim}",
        "templates": {
            "opener": FWD_OPENER,
            "option_header": "Option: start with the box at (r, c).",
            "near_end_line": NEAR_END_LINE,
            "far_end_line": FAR_END_LINE,
            "conclusion": "\nComparing the two orders, starting with the box at (r, c) is the "
                          "one that solves the puzzle; committing that plan.",
            "think_close": THINK_CLOSE,
        },
        "loss": "CE on assistant text (opener, headers, actions, end lines, conclusion, "
                "committed plan); MSE on every imagined frame of BOTH rollouts (all loss=True; "
                "the envfb loader DROPS loss=False assistant images); frame0/instruction = "
                "context (the ONLY loss-0 image)",
        "loader": "VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb'; "
                  "same row shape as the CoT-inf / branch sets, feedback branch never fires)",
        "vanilla_rows": "copied VERBATIM (messages untouched) + stamps {has_forward=False, "
                        "forward_first_tried=None, plan_lens=None, schema_version, "
                        "cot_schema_version=<original schema_version>}",
        "alignment_manifest": "NOT copied: rows are a forward/vanilla mixture, not 1:1 with "
                              "any single source",
        "shards": shard_files,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_root / "metadata" / "forward_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("shards", "templates")},
                     indent=2), flush=True)
    print(f"DONE FWD2 total_rows={total_written} forward={agg_stats.get('forward', 0)} "
          f"vanilla={agg_stats.get('vanilla', 0)} dropped_over_budget="
          f"{agg_stats.get('dropped_over_token_budget', 0)} shards={len(shard_files)}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Self-test (writes nothing)
# ---------------------------------------------------------------------------
def _load_level_rows(levels_src: str, subdir: str, n_want: int) -> List[dict]:
    shards = sorted(glob.glob(os.path.join(levels_src, subdir, "*.jsonl.gz")))
    if not shards:
        raise FileNotFoundError(f"no *.jsonl.gz under {levels_src}/{subdir}")
    rows: List[dict] = []
    for s in shards:
        with gzip.open(s, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if len(rows) >= n_want:
                    return rows
    return rows


def _load_source_row(src_ordered: str, prov: dict) -> dict:
    path = os.path.join(src_ordered, "trajectories/sokoban_easy/train", prov["shard"])
    want = int(prov["row_index_in_shard"])
    j = -1
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j += 1
            if j == want:
                return json.loads(line)
    raise IndexError(f"row {want} not found in {path}")


def _selftest(args: argparse.Namespace) -> int:
    import re

    print(f"[FWD2 self-test] loading up to {args.self_test} stage-1 level rows from "
          f"{args.levels_src}/{args.levels_subdir} ...", flush=True)
    levels = _load_level_rows(args.levels_src, args.levels_subdir, args.self_test)
    print(f"[FWD2 self-test] loaded {len(levels)} level rows "
          f"(generated={sum(1 for l in levels if l.get('source') == 'generated')}, "
          f"natural={sum(1 for l in levels if l.get('source') == 'natural')})", flush=True)

    vanilla_shards = sorted(glob.glob(os.path.join(args.vanilla_src, args.vanilla_subdir,
                                                   "*.jsonl.gz")))
    van_lines: List[str] = []
    if vanilla_shards:
        with gzip.open(vanilla_shards[0], "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    van_lines.append(line)
                if len(van_lines) >= 6:
                    break
    print(f"[FWD2 self-test] loaded {len(van_lines)} vanilla rows from "
          f"{vanilla_shards[0] if vanilla_shards else '<none>'}", flush=True)

    # The REAL envfb loader with stubbed sinks: exercises the true parse_row control flow
    # without tokenizer / VAE / ViT transforms (same pattern as the SK4 branch self-test).
    try:
        from data.interleave_datasets.edit_dataset_envfb import (
            VLMGymDatasetSokobanInterleavedCoTEnvFB)
    except ImportError:  # pragma: no cover
        from edit_dataset_envfb import VLMGymDatasetSokobanInterleavedCoTEnvFB

    class _StubLoader(VLMGymDatasetSokobanInterleavedCoTEnvFB):
        def __init__(self):           # skip heavy tokenizer/transform init
            pass

        def _init_data(self):
            return {"texts": [], "img_loss": []}

        def _add_text(self, data, text, need_loss, **kw):
            data["texts"].append((text, int(bool(need_loss))))
            return data

        def _add_image(self, data, image, need_loss, need_vae=True, need_vit=True, **kw):
            data["img_loss"].append(int(bool(need_loss)))
            return data

        def _decode_b64_image(self, b64s):
            return b64s               # passthrough; the stub _add_image ignores the image

    stub = _StubLoader()
    HEADER_RE = re.compile(r"^Option: start with the box at \((\d+), (\d+)\)\.$", re.M)
    WIN_RE = re.compile(r"starting with the box at \((\d+), (\d+)\) is the one that solves "
                        r"the puzzle")

    fails: List[Tuple[int, str, str]] = []

    def fail(i: int, check: str, msg: str) -> None:
        fails.append((i, check, msg))

    dist: Counter = Counter()
    flips = 0
    tok_max = 0
    n_nat_checked = 0
    van_instr = None
    if van_lines:
        van_instr = json.loads(van_lines[0])["messages"][0]["content"][0]["text"]

    for i, lv in enumerate(levels):
        out = transform_forward_row(lv, 0)
        parsed = stub.parse_row(json.dumps(out))
        texts = parsed["texts"]
        img_loss = parsed["img_loss"]
        all_text = "".join(t for t, _ in texts)
        last_seg = texts[-1][0] if texts else ""
        near_n = len(lv["near_plan"]["actions"])
        far_n = len(lv["far_plan"]["actions"])
        near_box = tuple(int(v) for v in lv["near_box"])
        far_box = tuple(int(v) for v in lv["far_box"])
        raw_ass_text = "".join(c["text"] for c in out["messages"][1]["content"]
                               if c["type"] == "text")

        # ---- C1 loader-compat ----
        if img_loss.count(0) != 1:
            fail(i, "C1", f"loss-0 images = {img_loss.count(0)} (want exactly 1 = frame0)")
        if not img_loss or img_loss[0] != 0:
            fail(i, "C1", "first parsed image is not the loss-0 frame0")
        if img_loss.count(1) != near_n + far_n:
            fail(i, "C1", f"loss-1 images = {img_loss.count(1)} != near+far = {near_n + far_n} "
                          f"(an assistant image was dropped or missing)")
        if all_text.count("<think>") != 1:
            fail(i, "C1", f"<think> count = {all_text.count('<think>')} (want 1, loader-prepended)")
        if "<think>" in raw_ass_text:
            fail(i, "C1", "raw row text contains a literal <think> (loader would double it)")
        if all_text.count("</think>") != 1 or "</think>" not in last_seg:
            fail(i, "C1", f"</think> count = {all_text.count('</think>')} / in last seg = "
                          f"{'</think>' in last_seg} (want exactly 1, in the last seg)")
        if any("</think>" in t for t, _ in texts[:-1]):
            fail(i, "C1", "</think> appears before the last text seg")
        want_plan = "\n".join(str(a) for a in lv["far_plan"]["actions"])
        committed = last_seg.split("</think>\n", 1)[1] if "</think>\n" in last_seg else None
        if committed != want_plan:
            fail(i, "C1", "committed plan != far_plan.actions verbatim")
        if not want_plan.endswith(STOP_ACTION):
            fail(i, "C1", "far plan does not end with ('stop', 'stop')")

        # ---- C2 no-leak ----
        heads = HEADER_RE.findall(all_text)
        if len(heads) != 2:
            fail(i, "C2", f"option headers matching the template = {len(heads)} (want 2)")
        else:
            head_boxes = {(int(r), int(c)) for r, c in heads}
            if head_boxes != {near_box, far_box}:
                fail(i, "C2", f"header boxes {sorted(head_boxes)} != level boxes "
                              f"{sorted({near_box, far_box})}")
            m0 = HEADER_RE.search(all_text)
            first_coords = (int(m0.group(1)), int(m0.group(2)))
            want_first = near_box if out["forward_first_tried"] == "near" else far_box
            if first_coords != want_first:
                fail(i, "C2", f"first header {first_coords} != forward_first_tried box {want_first}")
        wins = WIN_RE.findall(all_text)
        if len(wins) != 1:
            fail(i, "C2", f"winning-box conclusion count = {len(wins)} (want 1)")
        elif (int(wins[0][0]), int(wins[0][1])) != far_box:
            fail(i, "C2", f"conclusion names box {wins[0]} != far box {far_box}")
        if re.search(r"\b(near|nearer|nearest|far|farther|farthest)\b", raw_ass_text):
            fail(i, "C2", "near/far vocabulary leaks into assistant text")
        if all_text.count(NEAR_END_LINE.strip()) != 1:
            fail(i, "C2", "near-plan end line count != 1")
        if all_text.count(FAR_END_LINE.strip()) != 1:
            fail(i, "C2", "far-plan end line count != 1")
        dist[out["forward_first_tried"]] += 1
        if transform_forward_row(lv, 1)["forward_first_tried"] != out["forward_first_tried"]:
            flips += 1

        # ---- C3 instruction byte-equality ----
        instr = out["messages"][0]["content"][0]["text"]
        if van_instr is not None and instr != van_instr:
            fail(i, "C3", "user instruction != the live vanilla dataset instruction")
        if lv.get("source") == "natural" and lv.get("natural_provenance"):
            try:
                src_row = _load_source_row(args.src_ordered, lv["natural_provenance"])
                vrow = build_cot_inf_row(src_row)
                if (json.dumps(out["messages"][0], sort_keys=True)
                        != json.dumps(vrow["messages"][0], sort_keys=True)):
                    fail(i, "C3", "NATURAL episode: user turn != vanilla CoT-inf remap user "
                                  "turn (text/frame0/loss byte-compare)")
                n_nat_checked += 1
            except Exception as exc:
                fail(i, "C3", f"natural source lookup failed: {exc!r}")

        # ---- C4 token budget ----
        est = estimate_tokens(out["messages"])
        tok_max = max(tok_max, est)
        if est >= TOKEN_BUDGET:
            fail(i, "C4", f"token estimate {est} >= budget {TOKEN_BUDGET}")

        # metadata stamps
        if (out.get("has_forward") is not True or out.get("schema_version") != SCHEMA_VERSION
                or out.get("plan_lens") != {"near": near_n, "far": far_n}
                or out.get("cot_schema_version") != COT_BASE_SCHEMA):
            fail(i, "C1", "forward row stamps off (has_forward/plan_lens/schema_version)")

    # ---- vanilla verbatim-copy + stamp checks ----
    van_tok_max = 0
    for j, line in enumerate(van_lines):
        orig = json.loads(line)
        stamped = stamp_vanilla(json.loads(line))
        if stamped["messages"] != orig["messages"]:
            fail(j, "MIX", "vanilla messages were modified (must be verbatim)")
        if any(orig[k] != stamped[k] for k in orig if k != "schema_version"):
            fail(j, "MIX", "a non-stamp vanilla key changed")
        if (stamped.get("schema_version") != SCHEMA_VERSION
                or stamped.get("cot_schema_version") != orig.get("schema_version")
                or stamped.get("has_forward") is not False
                or stamped.get("forward_first_tried") is not None
                or stamped.get("plan_lens") is not None):
            fail(j, "MIX", "vanilla stamps off")
        parsed = stub.parse_row(json.dumps(stamped))
        if parsed["img_loss"].count(0) != 1:
            fail(j, "MIX", f"stamped vanilla loss-0 images = {parsed['img_loss'].count(0)}")
        van_tok_max = max(van_tok_max, estimate_tokens(stamped["messages"]))

    # ---- mixing slot-math unit check (pure math; covers the repeat mapping too) ----
    slot_ok = True
    for (tt, nfx) in ((20, 14), (10, 7), (100, 70), (7, 5), (100000, 55000), (100000, 70000)):
        slots = [is_forward_slot(t, tt, nfx) for t in range(tt)]
        cnt = 0
        for t in range(tt):
            if n_forward_before(t, tt, nfx) != cnt:
                slot_ok = False
            cnt += int(slots[t])
        if sum(slots) != nfx or cnt != nfx:
            slot_ok = False
    pool = 7
    rep_map = [(k % pool, k // pool) for k in range(20)]
    if (sorted(set(x for x, _ in rep_map)) != list(range(pool))
            or rep_map[7] != (0, 1) or rep_map[19] != (5, 2)):
        slot_ok = False
    if not slot_ok:
        fail(-1, "MIX", "slot/repeat math inconsistent")

    # ---- report ----
    n = len(levels)
    checks = [
        ("C1 loader-compat (envfb parse_row: frame0 the only loss-0 image, all plan frames "
         "loss=1, <think> once, </think> last-seg-only, committed plan verbatim)",
         not any(c == "C1" for _i, c, _m in fails)),
        ("C2 no-leak (2 format-identical headers, first==first_tried, winner only in "
         "conclusion, no near/far words, both end lines once)",
         not any(c == "C2" for _i, c, _m in fails)),
        ("C3 instruction byte-equality (vs live vanilla dataset"
         + (f"; {n_nat_checked} NATURAL episode(s) vs build_cot_inf_row user turn" if n_nat_checked else "")
         + ")", not any(c == "C3" for _i, c, _m in fails)),
        (f"C4 token budget (max forward estimate {tok_max} < {TOKEN_BUDGET}; "
         f"max vanilla estimate {van_tok_max})",
         not any(c == "C4" for _i, c, _m in fails) and tok_max < TOKEN_BUDGET),
        ("MIX vanilla-verbatim + stamps + slot math",
         not any(c == "MIX" for _i, c, _m in fails)),
    ]
    print(f"[FWD2 self-test] rows checked: {n} forward + {len(van_lines)} vanilla | "
          f"first_tried: {dict(dist)} | rep0->rep1 order flips: {flips}/{n} | "
          f"natural user-turn byte-checks: {n_nat_checked}", flush=True)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    dist_ok = (n < 6) or (dist.get("near", 0) > 0 and dist.get("far", 0) > 0)
    if not dist_ok:
        fail(-1, "C2", f"first_tried never mixes: {dict(dist)}")
        print("  [FAIL] C2 first_tried distribution (never mixes near/far)", flush=True)
    if fails:
        print(f"[FWD2 self-test] {len(fails)} FAILURE(S):", flush=True)
        for f_i, f_c, f_m in fails[:30]:
            print(f"    row {f_i} [{f_c}]: {f_m}", flush=True)
    ok_all = (not fails) and n > 0 and all(ok for _name, ok in checks) and dist_ok
    print(f"[FWD2 self-test] RESULT: {'PASS' if ok_all else 'FAIL'}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
