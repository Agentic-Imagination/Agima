#!/usr/bin/env python3
"""Build a CoT-reformulated K-action-chunk world-model dataset for sokoban (action-chunking study, CoT variant).

Reformulates the self-rollout world model into a single-generation CoT:

  OLD (turn-based world model):
    instruction + frame0 -> action_chunk -> [next frame] -> action_chunk -> [next frame] -> ...
  NEW (this, CoT):
    instruction + frame0 -> <think> action_chunk <img> action_chunk <img> ... <img(terminal)> </think>
                            <summarized full action list>

The whole imagined rollout (action chunks interleaved with the K-step-ahead frames) lives INSIDE
<think>; after </think> the model emits the complete action plan (the "answer"), which is what
gets executed open-loop at eval. NO prose reasoning yet (the think content is just action chunks
+ imagined frames). Per-step counters are dropped; for K>1 the imagination cadence is stated once
in the opening instruction.

THE TRAINING PIPELINE IS UNCHANGED: this writes the `messages` JSONL that
VLMGymDatasetSokobanInterleavedCoT already consumes (auto-prepends <think>, emits VISION_START +
MSE image per imagined frame, CE on all assistant text). Loss = CE on action chunks + the final
plan, MSE on the imagined frames, frame0/instruction = context.

Original data untouched; writes a NEW local repo per K. Whole-node multiprocess.
"""
import argparse, json, gzip, glob, os, shutil
from multiprocessing import Pool

# reuse the chunk slicer + helpers from the (non-CoT) chunk remap so chunks/frames are identical.
# robust to both contexts: run as a script (scripts/data on sys.path) or imported as scripts.data.<mod>
try:
    from remap_sokoban_chunk_k import _max_steps, _rules_prefix
except ImportError:
    from scripts.data.remap_sokoban_chunk_k import _max_steps, _rules_prefix


# ---- SHARED instruction builder (imported by the CoT eval runner; keep in sync there) ----
# Per-chunk <think> open/close templates: EVERY chunk opens its own imagination and closes before committing.
def think_open(n_actions: int) -> str:
    return ("Let me imagine the next move step by step." if n_actions == 1
            else f"Let me imagine the next {n_actions} moves step by step.")
THINK_CLOSE = "Done imagining; committing these actions:"
FRAME0_LABEL = "Starting board:"   # labels the initial board image (frame0), which immediately follows


def cot_instruction(rules: str, max_steps: int, K: int) -> str:
    """User instruction = game rules + ONE minimal cadence line (action-chunk size) + the frame0 label;
    the initial board image follows this text. The rules already state the goal + stop requirement; the env
    enforces the step limit. Imagination is ALWAYS per-step; K only sets the action-chunk size. `max_steps`
    kept for signature compatibility. `rules` is the FULL step-0 source prompt; the per-step counter is stripped."""
    cadence = ("You act one move at a time." if K == 1 else f"You act in chunks of {K} actions.")
    return _rules_prefix(rules).rstrip() + "\n\n" + cadence + "\n\n" + FRAME0_LABEL


def transform_row(row, K):
    """CoT, interval K — MULTI-TURN. Imagination is PER-STEP (a frame after EVERY action) for ALL K.
    K = the action-chunk size between ENV feedbacks. Each chunk is one assistant turn:
        assistant: <think> a_s<img> a_{s+1}<img> ... a_{e-1}<img> </think> {chunk's K actions}
    and BETWEEN chunks the ENVIRONMENT replies with a user turn:
        user: "Action executed."            <-- loss=0 (feedback, NOT a model target)
    The loader prepends <think> per assistant turn; EVERY chunk opens its think with think_open(n) and
    closes with THINK_CLOSE before committing the actions. No final summary (turn-based).
    """
    history = row.get("history") or []
    if not history:
        raise ValueError(f"row {row.get('__bagel_order_key')} has no history")
    rules = history[0].get("prompt", "")
    max_steps = _max_steps(row)
    frame0 = history[0].get("image_prev") or history[0].get("image")
    n = len(history)

    messages = [{"role": "user", "content": [
        {"type": "text", "text": cot_instruction(rules, max_steps, K)},
        {"type": "image", "image": frame0, "loss": False},      # initial board: context, no loss
    ]}]

    n_chunks = 0
    n_frames = 0
    s = 0
    while s < n:
        e = min(s + K, n)
        seg = history[s:e]
        n_act = len(seg)
        content = []
        for j, h in enumerate(seg):
            a = str(h.get("vlm_output") or h.get("action"))
            if j == 0:
                txt = "\n" + think_open(n_act) + "\n" + a   # EVERY chunk opens its think (loader prepends <think>)
            else:
                txt = "\n" + a
            content.append({"type": "text", "text": txt})
            f = h.get("image_next")
            if f:                                       # PER-STEP imagined frame (every action); MSE target
                content.append({"type": "image", "image": f, "loss": True})
                n_frames += 1
        chunk_actions = "\n".join(str(h.get("vlm_output") or h.get("action")) for h in seg)
        content.append({"type": "text", "text": "\n" + THINK_CLOSE + "\n</think>\n" + chunk_actions})  # close think + commit
        messages.append({"role": "assistant", "content": content})
        n_chunks += 1
        if e < n:                                       # ENV feedback BETWEEN chunks (loss=0 user turn)
            messages.append({"role": "user", "content": [{"type": "text", "text": "Action executed."}]})
        s = e

    out = {k: row.get(k) for k in (
        "__bagel_order_key", "task", "env_id", "env_kwargs", "seed", "episode_seed",
        "extra_state", "init_state", "init_state_hash", "stats", "version",
        "split", "upstream_env_id", "action_schema_version", "source_dataset", "alignment_index",
    ) if k in row}
    out["messages"] = messages
    out["chunk_size"] = K
    out["n_steps"] = n
    out["n_chunks"] = n_chunks
    out["n_imagined_frames"] = n_frames                 # == per-step frames (MSE targets)
    out["n_feedback_turns"] = n_chunks - 1              # # "Action executed." user turns (loss=0)
    out["max_steps"] = max_steps
    out["schema_version"] = f"sokoban_cot_perstep_turnfb_k{K}_v3"   # v3: + "Starting board:" label before frame0
    out["cot_row_type"] = "perstep_cot_multiturn_envfeedback"
    return out


def process_shard(args_tuple):
    in_path, out_path, K = args_tuple
    n = 0
    tmp = out_path + ".tmp"
    with gzip.open(in_path, "rt") as fin, gzip.open(tmp, "wt") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(json.dumps(transform_row(json.loads(line), K)) + "\n")
            n += 1
    os.replace(tmp, out_path)
    return (os.path.basename(in_path), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--src-subdir", default="trajectories/sokoban_easy/train")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    args = ap.parse_args()
    K = args.chunk_size
    os.makedirs(f"{args.out_root}/training", exist_ok=True)
    os.makedirs(f"{args.out_root}/metadata", exist_ok=True)
    shards = sorted(glob.glob(f"{args.src_root}/{args.src_subdir}/*.jsonl.gz"))
    assert shards, f"no shards under {args.src_root}/{args.src_subdir}"
    tasks = [(s, f"{args.out_root}/training/{os.path.basename(s)}", K) for s in shards]
    print(f"[CoT chunk] chunk_size={K} cores={os.cpu_count()} workers={args.workers} shards={len(tasks)}", flush=True)
    with Pool(min(args.workers, max(len(tasks), 1))) as p:
        res = p.map(process_shard, tasks)
    tot = 0
    for name, n in sorted(res):
        print(f"  {name}: {n} rows", flush=True); tot += n
    man = f"{args.src_root}/metadata/train_alignment_manifest.jsonl"
    if os.path.exists(man):
        shutil.copy2(man, f"{args.out_root}/metadata/train_alignment_manifest.jsonl")
    json.dump({"study": "action_chunking_cot_perstep_multiturn", "chunk_size": K, "source_root": args.src_root,
               "total_rows": tot,
               "format": "MULTI-TURN messages: user(rules+cadence, frame0 ctx) then per chunk -> "
                         "assistant(<think> a<img> ...K steps... </think> {K actions}) , user('Action executed.') between chunks",
               "imagination": "PER-STEP (a frame after EVERY action) for ALL K; K = action-chunk size between env feedbacks",
               "loss": "CE on assistant text (think + actions); MSE on every per-step imagined frame; "
                       "'Action executed.' is a USER turn = loss=0 (env feedback, NOT a model target); frame0/instruction = context",
               "loader": "VLMGymDatasetSokobanInterleavedCoT (unchanged; <think> prepended per assistant turn)"},
              open(f"{args.out_root}/metadata/cot_chunk_summary.json", "w"), indent=2)
    print(f"DONE CoT chunk_size={K} total_rows={tot}", flush=True)


if __name__ == "__main__":
    main()
