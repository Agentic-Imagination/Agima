#!/usr/bin/env python3
"""PushT CoT K-action-chunk world-model dataset with GT ENV-FEEDBACK re-grounding (Structure A).

PushT analog of remap_sokoban_cot_chunk_k_envfb.py. Per-step imagination is kept (one imagined
next-frame MSE target after EVERY committed action); between chunks the ENV provides the REAL
chunk-end frame as a loss-0 CONTEXT INPUT (the model RE-SEES the true frame every K steps):

    user:      [instruction, frame0(ctx)]
    assistant: <think> a_s<img> ... a_{e-1}<img> </think> {K actions}   (imgs = MSE targets)
    user:      "Env Feedback:" <REAL chunk-end frame>   <-- loss=0 INPUT (GT re-grounding)
    ...                                                  (no feedback after the final chunk)

The REAL chunk-end frame = seg[-1]["image_next"] (same image used as the chunk's last imagined-frame
MSE target). At eval this turn is replaced by env.render() after executing the chunk (q90) -> train==eval.
Uses the SAME envfb loader as sokoban (VLMGymDatasetSokobanInterleavedCoTEnvFB, loss type
'interleaved_cot_envfb'; honors user-turn image loss=0). Instruction = PushT cot_instruction (shared).
Source = PushT PLAIN per-step history (pusht_..._stopreq_aligned_cot100k_..._ordered/data/train).
Original data untouched.
"""
import argparse, json, gzip, glob, os, shutil
from multiprocessing import Pool

try:
    from remap_pusht_cot_chunk_k import cot_instruction, think_open, THINK_CLOSE, _max_steps
except ImportError:
    from scripts.data.remap_pusht_cot_chunk_k import cot_instruction, think_open, THINK_CLOSE, _max_steps

# SHARED env-feedback label (same text the sokoban envfb study + eval runner use) -> train==eval match
ENV_FEEDBACK_LABEL = os.environ.get("BAGEL_ENV_FEEDBACK_LABEL", "Action executed.")  # env-overridable; complete feedback = this text + the GT image


def transform_row(row, K):
    history = row.get("history") or []
    if not history:
        raise ValueError(f"row {row.get('__bagel_order_key')} has no history")
    rules = history[0].get("prompt", "")
    max_steps = _max_steps(row)
    frame0 = history[0].get("image_prev") or history[0].get("image")
    n = len(history)

    messages = [{"role": "user", "content": [
        {"type": "text", "text": cot_instruction(rules, max_steps, K)},
        {"type": "image", "image": frame0, "loss": False},      # initial frame: context, no loss
    ]}]

    n_chunks = 0
    n_frames = 0
    n_fb_frames = 0
    s = 0
    while s < n:
        e = min(s + K, n)
        seg = history[s:e]
        n_act = len(seg)
        content = []
        for j, h in enumerate(seg):
            a = str(h.get("vlm_output") or h.get("action"))
            txt = ("\n" + think_open(n_act) + "\n" + a) if j == 0 else ("\n" + a)
            content.append({"type": "text", "text": txt})
            f = h.get("image_next")
            if f:                                       # PER-STEP imagined frame (every action); MSE target
                content.append({"type": "image", "image": f, "loss": True})
                n_frames += 1
        chunk_actions = "\n".join(str(h.get("vlm_output") or h.get("action")) for h in seg)
        content.append({"type": "text", "text": "\n" + THINK_CLOSE + "\n</think>\n" + chunk_actions})
        messages.append({"role": "assistant", "content": content})
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
        "__bagel_order_key", "task", "env_id", "env_kwargs", "seed", "episode_seed", "episode",
        "extra_state", "init_state", "init_state_hash", "stats", "version",
        "split", "upstream_env_id", "action_schema_version", "source_dataset", "alignment_index",
    ) if k in row}
    out["messages"] = messages
    out["chunk_size"] = K
    out["n_steps"] = n
    out["n_chunks"] = n_chunks
    out["n_imagined_frames"] = n_frames
    out["n_feedback_turns"] = n_chunks - 1
    out["n_feedback_frames"] = n_fb_frames              # GT env-feedback frames (loss=0 inputs) == n_chunks-1
    out["max_steps"] = max_steps
    out["schema_version"] = f"pusht_cot_perstep_envfb_k{K}_v4"
    out["cot_row_type"] = "perstep_cot_multiturn_gtenvfeedback"
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
    ap.add_argument("--src-subdir", default="data/train")
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
    print(f"[PushT CoT chunk ENVFB] chunk_size={K} cores={os.cpu_count()} workers={args.workers} shards={len(tasks)}", flush=True)
    with Pool(min(args.workers, max(len(tasks), 1))) as p:
        res = p.map(process_shard, tasks)
    tot = 0
    for name, n in sorted(res):
        print(f"  {name}: {n} rows", flush=True); tot += n
    for cand in (f"{args.src_root}/metadata/train_alignment_manifest.jsonl", f"{args.src_root}/data/train_alignment_manifest.jsonl"):
        if os.path.exists(cand):
            shutil.copy2(cand, f"{args.out_root}/metadata/train_alignment_manifest.jsonl"); break
    json.dump({"study": "pusht_action_chunking_cot_perstep_GT_ENVFEEDBACK", "chunk_size": K,
               "source_root": args.src_root, "total_rows": tot,
               "format": "MULTI-TURN: user(rules+cadence,frame0 ctx); per chunk assistant(<think> a<img> ...K... </think> {K actions}); "
                         "BETWEEN chunks user('Env Feedback:' + REAL chunk-end frame, loss=0 INPUT) for GT re-grounding",
               "loss": "CE on assistant text; MSE on per-step imagined frames; Env-Feedback frame = loss=0 INPUT; frame0/instruction = context",
               "loader": "VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb')",
               "jpeg_quality": 90},
              open(f"{args.out_root}/metadata/cot_chunk_summary.json", "w"), indent=2)
    print(f"DONE PushT CoT ENVFB chunk_size={K} total_rows={tot}", flush=True)


if __name__ == "__main__":
    main()
