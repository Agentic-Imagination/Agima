#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""CoT K=INFINITY (whole-episode, open-loop) world-model dataset for sokoban easy.

The K=inf endpoint of the CoT action-chunking family (K=1/3/5/10 built by
remap_sokoban_cot_chunk_k.py / remap_sokoban_cot_chunk_k_envfb.py): the WHOLE episode is
ONE assistant turn — every step's (action, imagined frame) pair lives inside a single
<think>...</think>, then the model commits the full action plan. NO env-feedback turns and
NO GT re-grounding frames anywhere (open-loop after frame0):

    user:      [instruction (rules + inf cadence + "Starting board:"), frame0(ctx, loss=0)]
    assistant: <think> a_1<img> a_2<img> ... a_N<img(terminal)> </think> {ALL N actions}

Unlike the single-shot open-loop baseline (remap_sokoban_easy_singleshot_openloop.py, which
drops imagination entirely), the PER-STEP imagined frames are KEPT: MSE on every <img>
inside think, CE on all assistant text; frame0/instruction = context. Row `messages` shape
is IDENTICAL to the envfb K-chunk datasets, so the SAME loader consumes it:
VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb',
data/interleave_datasets/edit_dataset_envfb.py) — with zero feedback turns its envfb-vs-base
difference is dormant (frame0 loss=False is context either way).

Token budget: fits the 50k packing budget — an inf row is the matching envfb K=10 row MINUS
its GT feedback frames and per-chunk think boilerplate (real K=10 rows measured ~18.6k max).

Original data untouched; writes a NEW local repo. Whole-node multiprocess.
"""
import argparse, json, gzip, glob, os, shutil
from multiprocessing import Pool
from pathlib import Path

# reuse the EXACT shared text builders so instruction prefix / think template / frame0 label are
# byte-identical to the K-chunk family (train==eval; only the cadence sentence differs, below).
# robust to both contexts: run as a script (scripts/data on sys.path) or imported as scripts.data.<mod>
try:
    from remap_sokoban_cot_chunk_k import think_open, THINK_CLOSE, FRAME0_LABEL
    from remap_sokoban_chunk_k import _max_steps, _rules_prefix
except ImportError:
    from scripts.data.remap_sokoban_cot_chunk_k import think_open, THINK_CLOSE, FRAME0_LABEL
    from scripts.data.remap_sokoban_chunk_k import _max_steps, _rules_prefix

# ---- SHARED inf-cadence sentence (import from here in the inf eval runner; keep in sync) ----
# Replaces cot_instruction()'s K-chunk cadence ("You act in chunks of {K} actions.") — there is no
# feedback to promise. Wording follows the singleshot open-loop DIRECTIVE
# (remap_sokoban_easy_singleshot_openloop.py: "no further images or feedback" / plan-the-whole-
# episode) but KEEPS per-step imagination. Like cot_instruction, no stop/max_steps restatement:
# the rules already state the goal + stop requirement and the env enforces the step limit.
INF_CADENCE = (
    "You will see ONLY this initial frame and receive no further frames or feedback: "
    "plan the whole episode in one shot, imagining the board after every single move, "
    "then commit the complete action sequence."
)


def cot_inf_instruction(rules: str, max_steps: int) -> str:
    """Same structure as remap_sokoban_cot_chunk_k.cot_instruction (rules prefix + ONE cadence
    line + frame0 label), with the K-chunk cadence swapped for INF_CADENCE. `max_steps` kept
    for signature compatibility with cot_instruction (unused there too)."""
    return _rules_prefix(rules).rstrip() + "\n\n" + INF_CADENCE + "\n\n" + FRAME0_LABEL


def transform_row(row):
    """CoT, K=inf — SINGLE assistant turn covering the whole episode.
    Identical to remap_sokoban_cot_chunk_k_envfb.transform_row with one chunk spanning all N
    steps and the feedback branch removed: per-step imagined frames (MSE) after EVERY action,
    one think_open(N) ... THINK_CLOSE </think> wrap, then the full N-action plan (CE)."""
    history = row.get("history") or []
    if not history:
        raise ValueError(f"row {row.get('__bagel_order_key')} has no history")
    rules = history[0].get("prompt", "")
    max_steps = _max_steps(row)
    frame0 = history[0].get("image_prev") or history[0].get("image")
    n = len(history)

    messages = [{"role": "user", "content": [
        {"type": "text", "text": cot_inf_instruction(rules, max_steps)},
        {"type": "image", "image": frame0, "loss": False},      # initial board: context, no loss
    ]}]

    n_frames = 0
    content = []
    for j, h in enumerate(history):
        a = str(h.get("vlm_output") or h.get("action"))
        txt = ("\n" + think_open(n) + "\n" + a) if j == 0 else ("\n" + a)
        content.append({"type": "text", "text": txt})
        f = h.get("image_next")
        if f:                                       # PER-STEP imagined frame (every action); MSE target
            content.append({"type": "image", "image": f, "loss": True})
            n_frames += 1
    full_plan = "\n".join(str(h.get("vlm_output") or h.get("action")) for h in history)
    content.append({"type": "text", "text": "\n" + THINK_CLOSE + "\n</think>\n" + full_plan})
    messages.append({"role": "assistant", "content": content})   # the ONE model turn; nothing follows

    out = {k: row.get(k) for k in (
        "__bagel_order_key", "task", "env_id", "env_kwargs", "seed", "episode_seed",
        "extra_state", "init_state", "init_state_hash", "stats", "version",
        "split", "upstream_env_id", "action_schema_version", "source_dataset", "alignment_index",
    ) if k in row}
    out["messages"] = messages
    out["chunk_size"] = "inf"                       # whole episode = one chunk
    out["n_steps"] = n
    out["n_chunks"] = 1
    out["n_imagined_frames"] = n_frames
    out["n_feedback_turns"] = 0                     # open-loop: no env feedback ever
    out["n_feedback_frames"] = 0                    # no GT re-grounding frames
    out["max_steps"] = max_steps
    out["schema_version"] = "sokoban_cot_perstep_kinf_v1"
    out["cot_row_type"] = "perstep_cot_singleturn_openloop"
    return out


def process_shard(args_tuple):
    in_path, out_path = args_tuple
    n = 0
    tmp = out_path + ".tmp"
    with gzip.open(in_path, "rt") as fin, gzip.open(tmp, "wt") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(json.dumps(transform_row(json.loads(line))) + "\n")
            n += 1
    os.replace(tmp, out_path)
    return (os.path.basename(in_path), n)


DEFAULT_SRC = os.environ.get("AGIMA_SOURCE_DATA", str(Path(__file__).resolve().parents[2] / "datasets" / "sokoban_source"))
DATE_TAG = os.environ.get("DATE_TAG", "20260707_perseg")
DEFAULT_OUT = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--src-subdir", default="trajectories/sokoban_easy/train")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    args = ap.parse_args()
    os.makedirs(f"{args.out_root}/training", exist_ok=True)
    os.makedirs(f"{args.out_root}/metadata", exist_ok=True)
    shards = sorted(glob.glob(f"{args.src_root}/{args.src_subdir}/*.jsonl.gz"))
    assert shards, f"no shards under {args.src_root}/{args.src_subdir}"
    tasks = [(s, f"{args.out_root}/training/{os.path.basename(s)}") for s in shards]
    print(f"[CoT chunk KINF] cores={os.cpu_count()} workers={args.workers} shards={len(tasks)} out={args.out_root}", flush=True)
    with Pool(min(args.workers, max(len(tasks), 1))) as p:
        res = p.map(process_shard, tasks)
    tot = 0
    for name, n in sorted(res):
        print(f"  {name}: {n} rows", flush=True); tot += n
    man = f"{args.src_root}/metadata/train_alignment_manifest.jsonl"
    if os.path.exists(man):
        shutil.copy2(man, f"{args.out_root}/metadata/train_alignment_manifest.jsonl")
    json.dump({"study": "action_chunking_cot_perstep_singleturn_KINF", "chunk_size": "inf",
               "date_tag": DATE_TAG, "source_root": args.src_root, "total_rows": tot,
               "format": "SINGLE-TURN: user(rules+inf-cadence, frame0 ctx); ONE assistant turn = "
                         "<think> a_1<img> ... a_N<img> </think> {all N actions}; NO feedback turns, NO GT re-grounding frames",
               "cadence": INF_CADENCE,
               "loss": "CE on assistant text (think + full plan); MSE on every per-step imagined frame; "
                       "frame0/instruction = context (loss=0); zero loss-0 frames after frame0",
               "loader": "VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb'; "
                         "same row shape as the envfb K-chunk sets, feedback branch simply never fires)"},
              open(f"{args.out_root}/metadata/cot_chunk_summary.json", "w"), indent=2)
    print(f"DONE CoT KINF total_rows={tot}", flush=True)


if __name__ == "__main__":
    main()
