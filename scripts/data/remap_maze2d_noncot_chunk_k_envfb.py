#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""Maze2D NON-CoT (pure action-policy) K-action-chunk dataset with GT ENV-FEEDBACK re-grounding.

Maze2D analog of remap_sokoban_noncot_chunk_k_envfb.py and the ablation baseline for the maze2d
CoT chunk world model (remap_maze2d_cot_chunk_k_envfb.py): the SAME envfb multi-turn format and
the SAME instruction (cot_instruction is imported -- rules + total-budget sentence + env-feedback
cadence; see the PROMPT CONTRACT documented there), but the assistant emits ONLY the K committed
actions (CE) -- NO <think>, NO per-step imagined frames, NO MSE targets. The model still SEES the
maze: frame0 (loss=0 input) and the REAL chunk-end frame re-injected every K steps
(ENV_FEEDBACK_LABEL + image, loss=0 input) -- it just never GENERATES a frame. Isolates whether
generating frames (world-modeling) helps vs a pure VLM policy with GT re-grounding.

    user:      [instruction, frame0 (loss=0 ctx)]
    assistant: ('move', d1)\n...\n('move', dK)             <- K actions, CE only; NO images
    user:      ENV_FEEDBACK_LABEL <REAL chunk-end frame>   <- loss=0 INPUT (GT re-grounding)
    ...                                                    (no feedback after the final chunk)

K in {1,3,5,10} or 'inf'. K='inf' = single-turn open-loop: ONE assistant turn = the whole action
plan (all n actions incl. the final ('stop','stop')), ZERO feedback turns.

Loader: the SAME envfb loader ('interleaved_cot_envfb') -- an assistant turn with no images falls
into its text branch -> CE on the action text; user-turn images keep their explicit loss=False
(context input). Actions are copied VERBATIM from the source rows: "('move', d)" with d in 0-3
and the final "('stop', 'stop')". Original data untouched; writes a NEW repo per (difficulty, K).
"""
import argparse, json, gzip, glob, os, shutil
from multiprocessing import Pool

# reuse the EXACT shared maze2d instruction builder + feedback label (single chokepoint; the eval
# adapter imports the same symbols) so the non-CoT ablation and the CoT study match (train==eval).
try:
    from remap_maze2d_cot_chunk_k_envfb import (
        cot_instruction, _max_steps, parse_chunk_size, ENV_FEEDBACK_LABEL,
        CARRY_KEYS, SRC_V2, HF, DEFAULT_DATE_TAG, JPEG_QUALITY)
except ImportError:
    from scripts.data.remap_maze2d_cot_chunk_k_envfb import (
        cot_instruction, _max_steps, parse_chunk_size, ENV_FEEDBACK_LABEL,
        CARRY_KEYS, SRC_V2, HF, DEFAULT_DATE_TAG, JPEG_QUALITY)


def transform_row(row, K):
    """NON-CoT, interval K (int or 'inf'), MULTI-TURN with GT env-feedback re-grounding;
    assistant = K actions only. K='inf' -> one turn with the full plan, no feedback."""
    history = row.get("history") or []
    if not history:
        raise ValueError(f"row {row.get('__bagel_order_key')} has no history")
    rules = history[0].get("prompt", "")
    max_steps = _max_steps(row)
    frame0 = history[0].get("image_prev") or history[0].get("image")
    n = len(history)
    k_int = n if K == "inf" else int(K)             # inf = single chunk covering all n steps

    messages = [{"role": "user", "content": [
        {"type": "text", "text": cot_instruction(rules, max_steps, K)},
        {"type": "image", "image": frame0, "loss": False},      # initial maze: context, no loss
    ]}]

    n_chunks = 0
    n_fb_frames = 0
    s = 0
    while s < n:
        e = min(s + k_int, n)
        seg = history[s:e]
        # PURE POLICY: assistant emits ONLY the K committed actions (CE). No <think>, no imagined frames.
        chunk_actions = "\n".join(str(h.get("vlm_output") or h.get("action")) for h in seg)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk_actions}]})
        n_chunks += 1
        if e < n:                                   # ENV FEEDBACK: REAL chunk-end frame (loss=0 INPUT)
            gt_frame = seg[-1].get("image_next")    # real state after the chunk's last action
            fb_content = [{"type": "text", "text": ENV_FEEDBACK_LABEL}]
            if gt_frame:
                fb_content.append({"type": "image", "image": gt_frame, "loss": False})
                n_fb_frames += 1
            messages.append({"role": "user", "content": fb_content})
        s = e

    out = {k: row.get(k) for k in CARRY_KEYS if k in row}
    out["messages"] = messages
    out["chunk_size"] = K                           # int, or the string 'inf'
    out["n_steps"] = n
    out["n_chunks"] = n_chunks
    out["n_imagined_frames"] = 0                    # pure policy: assistant generates NO frames
    out["n_feedback_turns"] = n_chunks - 1          # 0 for K='inf' (open-loop)
    out["n_feedback_frames"] = n_fb_frames          # GT env-feedback frames (loss=0 inputs) == n_chunks-1
    out["max_steps"] = max_steps
    out["schema_version"] = f"maze2d_noncot_actiononly_envfb_k{K}_v1"
    out["cot_row_type"] = ("noncot_actiononly_singleturn_openloop" if K == "inf"
                           else "noncot_actiononly_multiturn_gtenvfeedback")
    return out


def process_shard(args_tuple):
    in_path, out_path, K, limit = args_tuple
    n = 0
    tmp = out_path + ".tmp"
    with gzip.open(in_path, "rt") as fin, gzip.open(tmp, "wt") as fout:
        for line in fin:
            if not line.strip():
                continue
            if limit and n >= limit:
                break
            fout.write(json.dumps(transform_row(json.loads(line), K)) + "\n")
            n += 1
    os.replace(tmp, out_path)
    return (os.path.basename(in_path), n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--difficulty", choices=("easy", "hard"), default="easy",
                    help="easy is the training difficulty for the easy->hard study")
    ap.add_argument("--chunk-size", required=True, help="1/3/5/10 or 'inf'")
    ap.add_argument("--date-tag", default=DEFAULT_DATE_TAG)
    ap.add_argument("--src-root", required=True, help='Input dataset directory')
    ap.add_argument("--src-subdir", default="training")
    ap.add_argument("--out-root", required=True,
                    help='Output dataset directory')
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap rows PER SHARD")
    args = ap.parse_args()
    K = parse_chunk_size(args.chunk_size)
    src_root = args.src_root or SRC_V2[args.difficulty]
    out_root = args.out_root or f"{HF}/maze2d_{args.difficulty}_native256_noncot_chunk_k{K}_{args.date_tag}"
    os.makedirs(f"{out_root}/training", exist_ok=True)
    os.makedirs(f"{out_root}/metadata", exist_ok=True)
    shards = sorted(glob.glob(f"{src_root}/{args.src_subdir}/*.jsonl.gz"))
    assert shards, f"no shards under {src_root}/{args.src_subdir}"
    tasks = [(s, f"{out_root}/training/{os.path.basename(s)}", K, args.limit) for s in shards]
    print(f"[Maze2D NON-CoT chunk ENVFB] difficulty={args.difficulty} chunk_size={K} "
          f"cores={os.cpu_count()} workers={args.workers} shards={len(tasks)} out={out_root}", flush=True)
    with Pool(min(args.workers, max(len(tasks), 1))) as p:
        res = p.map(process_shard, tasks)
    tot = 0
    for name, n in sorted(res):
        print(f"  {name}: {n} rows", flush=True); tot += n
    for cand in (f"{src_root}/metadata/train_manifest.jsonl",
                 f"{src_root}/metadata/train_alignment_manifest.jsonl"):
        if os.path.exists(cand):
            shutil.copy2(cand, f"{out_root}/metadata/{os.path.basename(cand)}"); break
    json.dump({"study": "maze2d_action_chunking_NONCOT_actiononly_GT_ENVFEEDBACK", "chunk_size": K,
               "difficulty": args.difficulty, "date_tag": args.date_tag,
               "source_root": src_root, "total_rows": tot,
               "prompt_contract": "SAME instruction as the CoT arm (imported cot_instruction: rules + "
                                  "total-budget sentence + env-feedback cadence + 'Starting maze:')",
               "format": "MULTI-TURN: user(rules+budget+cadence, frame0 ctx); per chunk assistant({K actions}, "
                         "NO think/no imagined frames); BETWEEN chunks user(ENV_FEEDBACK_LABEL + REAL chunk-end "
                         "frame, loss=0 INPUT); K='inf' -> ONE assistant turn (full plan), ZERO feedback",
               "env_feedback_label": ENV_FEEDBACK_LABEL,
               "imagination": "NONE -- pure action policy; assistant generates no frames (no MSE targets)",
               "loss": "CE on assistant action text only; frame0 + Env-Feedback frames = loss=0 INPUT context",
               "loader": "VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb'; "
                         "assistant has no images -> CE-only turn)",
               "jpeg_quality": JPEG_QUALITY},
              open(f"{out_root}/metadata/chunk_summary.json", "w"), indent=2)
    print(f"DONE Maze2D NON-CoT ENVFB difficulty={args.difficulty} chunk_size={K} total_rows={tot}", flush=True)


if __name__ == "__main__":
    main()
