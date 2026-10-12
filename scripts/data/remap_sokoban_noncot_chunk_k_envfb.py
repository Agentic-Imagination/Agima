#!/usr/bin/env python3
"""NON-CoT (pure action-policy) K-action-chunk sokoban dataset with GT ENV-FEEDBACK re-grounding.

ABLATION BASELINE for the CoT envfb world model (remap_sokoban_cot_chunk_k_envfb.py): the SAME
envfb multi-turn format and the SAME instruction, but the assistant emits ONLY the K committed
actions (CE) -- NO <think>, NO per-step imagined frames, NO MSE targets. The model still SEES the
board: frame0 (loss=0 input) and the REAL chunk-end frame re-injected every K steps as
"Env Feedback:" (loss=0 input, from the env) -- it just never GENERATES a frame. Isolates whether
generating frames (world-modeling) helps at all vs a pure VLM policy with GT re-grounding.

    user:      [instruction, frame0 (loss=0 ctx)]
    assistant: ('move',a1)\n...\n('move',aK)               <- K actions, CE only; NO images
    user:      "Env Feedback:" <REAL chunk-end frame>      <- loss=0 INPUT (GT re-grounding, from env)
    assistant: ('move', ...) ...
    ...                                                    (no feedback after the final chunk)

Loader: the SAME envfb loader (VLMGymDatasetSokobanInterleavedCoTEnvFB, loss type
'interleaved_cot_envfb') -- an assistant turn with no images falls into its text branch -> CE on the
action text; the user-turn images (frame0 + Env-Feedback) keep loss=0 (context input). Instruction =
cot_instruction (SHARED, incl. the two-box rule via _rules_prefix) -> train==eval consistent. The
cadence line ("You act in chunks of K actions.") describes the action interval only; it never mentions
imagination, so it is unchanged. Original data untouched; writes a NEW local repo per K.
"""
import argparse, json, gzip, glob, os, shutil
from multiprocessing import Pool

# reuse the EXACT shared instruction builder (rules + two-box rule + cadence + frame0 label) so the
# non-CoT prompt matches the CoT study and the eval runner (train==eval).
try:
    from remap_sokoban_cot_chunk_k import cot_instruction, FRAME0_LABEL
    from remap_sokoban_cot_chunk_inf import INF_CADENCE
    from remap_sokoban_chunk_k import _max_steps, _rules_prefix
except ImportError:
    from scripts.data.remap_sokoban_cot_chunk_k import cot_instruction, FRAME0_LABEL
    from scripts.data.remap_sokoban_cot_chunk_inf import INF_CADENCE
    from scripts.data.remap_sokoban_chunk_k import _max_steps, _rules_prefix


def _instruction(rules, max_steps, K):
    """K int -> shared cot_instruction cadence; K == 'inf' -> whole-episode open-loop cadence."""
    if K == "inf":
        return _rules_prefix(rules).rstrip() + "\n\n" + INF_CADENCE + "\n\n" + FRAME0_LABEL
    return cot_instruction(rules, max_steps, K)

# SHARED env-feedback label (same text the CoT envfb study + eval runner use) -> train==eval match
ENV_FEEDBACK_LABEL = os.environ.get("BAGEL_ENV_FEEDBACK_LABEL", "Action executed.")  # env-overridable; complete feedback = this text + the GT image


def transform_row(row, K):
    """NON-CoT, interval K (int or 'inf'), MULTI-TURN with GT env-feedback re-grounding; assistant = K actions only.
    K == 'inf' -> single chunk of ALL actions, no feedback turns (whole-episode open-loop policy)."""
    history = row.get("history") or []
    if not history:
        raise ValueError(f"row {row.get('__bagel_order_key')} has no history")
    rules = history[0].get("prompt", "")
    max_steps = _max_steps(row)
    frame0 = history[0].get("image_prev") or history[0].get("image")
    n = len(history)
    K_step = n if K == "inf" else K

    messages = [{"role": "user", "content": [
        {"type": "text", "text": _instruction(rules, max_steps, K)},
        {"type": "image", "image": frame0, "loss": False},      # initial board: context, no loss
    ]}]

    n_chunks = 0
    n_fb_frames = 0
    s = 0
    while s < n:
        e = min(s + K_step, n)
        seg = history[s:e]
        # PURE POLICY: assistant emits ONLY the K committed actions (CE). No <think>, no imagined frames.
        chunk_actions = "\n".join(str(h.get("vlm_output") or h.get("action")) for h in seg)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk_actions}]})
        n_chunks += 1
        if e < n:                                       # ENV FEEDBACK: REAL chunk-end frame (loss=0 INPUT)
            gt_frame = seg[-1].get("image_next")        # real state after the chunk's last action
            fb_content = [{"type": "text", "text": ENV_FEEDBACK_LABEL}]
            if gt_frame:
                fb_content.append({"type": "image", "image": gt_frame, "loss": False})
                n_fb_frames += 1
            messages.append({"role": "user", "content": fb_content})
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
    out["n_imagined_frames"] = 0                        # pure policy: assistant generates NO frames
    out["n_feedback_turns"] = n_chunks - 1
    out["n_feedback_frames"] = n_fb_frames              # GT env-feedback frames (loss=0 inputs) == n_chunks-1
    out["max_steps"] = max_steps
    out["schema_version"] = f"sokoban_noncot_actiononly_envfb_k{K}_v1"
    out["cot_row_type"] = "noncot_actiononly_multiturn_gtenvfeedback"
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
    ap.add_argument("--chunk-size", required=True, help="int (1/3/5/10) or 'inf' for whole-episode single chunk")
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    args = ap.parse_args()
    K = "inf" if args.chunk_size.lower() == "inf" else int(args.chunk_size)
    os.makedirs(f"{args.out_root}/training", exist_ok=True)
    os.makedirs(f"{args.out_root}/metadata", exist_ok=True)
    shards = sorted(glob.glob(f"{args.src_root}/{args.src_subdir}/*.jsonl.gz"))
    assert shards, f"no shards under {args.src_root}/{args.src_subdir}"
    tasks = [(s, f"{args.out_root}/training/{os.path.basename(s)}", K) for s in shards]
    print(f"[NON-CoT chunk ENVFB] chunk_size={K} cores={os.cpu_count()} workers={args.workers} shards={len(tasks)}", flush=True)
    with Pool(min(args.workers, max(len(tasks), 1))) as p:
        res = p.map(process_shard, tasks)
    tot = 0
    for name, n in sorted(res):
        print(f"  {name}: {n} rows", flush=True); tot += n
    man = f"{args.src_root}/metadata/train_alignment_manifest.jsonl"
    if os.path.exists(man):
        shutil.copy2(man, f"{args.out_root}/metadata/train_alignment_manifest.jsonl")
    json.dump({"study": "action_chunking_NONCOT_actiononly_GT_ENVFEEDBACK", "chunk_size": K,
               "source_root": args.src_root, "total_rows": tot,
               "format": "MULTI-TURN: user(rules+cadence,frame0 ctx); per chunk assistant({K actions}, NO think/no imagined frames); "
                         "BETWEEN chunks user('Env Feedback:' + REAL chunk-end frame, loss=0 INPUT) for GT re-grounding",
               "imagination": "NONE -- pure action policy; assistant generates no frames (no MSE targets)",
               "loss": "CE on assistant action text only; frame0 + Env-Feedback frames = loss=0 INPUT context",
               "loader": "VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb'; assistant has no images -> CE-only turn)"},
              open(f"{args.out_root}/metadata/chunk_summary.json", "w"), indent=2)
    print(f"DONE NON-CoT ENVFB chunk_size={K} total_rows={tot}", flush=True)


if __name__ == "__main__":
    main()
