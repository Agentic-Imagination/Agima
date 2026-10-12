#!/usr/bin/env python3
"""Pacman-2D NON-CoT (pure action-policy) K-action-chunk dataset with GT ENV-FEEDBACK re-grounding.

Pacman analog of remap_pusht_noncot_chunk_k_envfb.py and the ablation baseline for the pacman CoT
world model: SAME envfb multi-turn format + SAME instruction, but the assistant emits ONLY the K
committed `('move', action_id)` actions (CE) -- NO <think>, NO per-step imagined frames, NO MSE.
The model still SEES frame0 (loss=0) and the REAL chunk-end frame re-injected every K steps
("Action executed." loss=0 input); it never GENERATES a frame. Same envfb loader. No prefix-mask.
"""
import argparse, json, gzip, glob, os
from multiprocessing import Pool

try:
    from remap_pacman_cot_chunk_k import cot_instruction, _max_steps
except ImportError:
    from scripts.data.remap_pacman_cot_chunk_k import cot_instruction, _max_steps

ENV_FEEDBACK_LABEL = os.environ.get("BAGEL_ENV_FEEDBACK_LABEL", "Action executed.")


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
        {"type": "image", "image": frame0, "loss": False},
    ]}]

    n_chunks = n_fb_frames = 0
    s = 0
    while s < n:
        e = min(s + K, n)
        seg = history[s:e]
        chunk_actions = "\n".join(str(h.get("vlm_output") or h.get("action")) for h in seg)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk_actions}]})
        n_chunks += 1
        if e < n:
            gt_frame = seg[-1].get("image_next")
            fb_content = [{"type": "text", "text": ENV_FEEDBACK_LABEL}]
            if gt_frame:
                fb_content.append({"type": "image", "image": gt_frame, "loss": False})
                n_fb_frames += 1
            messages.append({"role": "user", "content": fb_content})
        s = e

    out = {k: row.get(k) for k in (
        "__bagel_order_key", "task", "env_id", "env_kwargs", "seed", "episode_seed", "episode",
        "extra_state", "init_state", "init_state_hash", "stats", "version", "alignment_index",
    ) if k in row}
    out["messages"] = messages
    out["chunk_size"] = K
    out["n_steps"] = n
    out["n_chunks"] = n_chunks
    out["n_imagined_frames"] = 0
    out["n_feedback_turns"] = n_chunks - 1
    out["n_feedback_frames"] = n_fb_frames
    out["max_steps"] = max_steps
    out["schema_version"] = f"pacman_noncot_actiononly_envfb_k{K}_v1"
    out["cot_row_type"] = "actiononly_multiturn_gtenvfeedback"
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
    print(f"[Pacman nonCoT chunk ENVFB] chunk_size={K} workers={args.workers} shards={len(tasks)}", flush=True)
    with Pool(min(args.workers, max(len(tasks), 1))) as p:
        res = p.map(process_shard, tasks)
    tot = 0
    for name, n in sorted(res):
        print(f"  {name}: {n} rows", flush=True); tot += n
    json.dump({"study": "pacman_action_chunking_noncot_GT_ENVFEEDBACK", "chunk_size": K,
               "source_root": args.src_root, "total_rows": tot,
               "loss": "CE on committed action text only; frame0 + Env-Feedback frames = loss=0 inputs; no MSE",
               "loader": "VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb')",
               "jpeg_quality": 95},
              open(f"{args.out_root}/metadata/noncot_chunk_summary.json", "w"), indent=2)
    print(f"DONE Pacman nonCoT ENVFB chunk_size={K} total_rows={tot}", flush=True)


if __name__ == "__main__":
    main()
