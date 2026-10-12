#!/usr/bin/env python3
"""Pacman-2D CoT K=inf (whole-episode single-chunk) world-model dataset — NO env feedback.

CoT-inf arm of the pacman feedback-interval study (analog of remap_pusht_cot_chunk_inf.py). SAME row
shape as the K-chunk envfb datasets but ONE assistant turn covers the WHOLE episode and there are
ZERO between-chunk feedback turns: the model sees frame0 once, imagines the next frame after EVERY
committed action (per-step MSE targets), then commits the full action sequence (incl. final
('stop','stop')) in one shot. Same envfb loader; frame0 loss=False is honored as context.
"""
import argparse, json, gzip, glob, os
from multiprocessing import Pool

try:
    from remap_pacman_cot_chunk_k import cot_instruction_inf, think_open, THINK_CLOSE, _max_steps
except ImportError:
    from scripts.data.remap_pacman_cot_chunk_k import cot_instruction_inf, think_open, THINK_CLOSE, _max_steps


def transform_row(row):
    history = row.get("history") or []
    if not history:
        raise ValueError(f"row {row.get('__bagel_order_key')} has no history")
    rules = history[0].get("prompt", "")
    max_steps = _max_steps(row)
    frame0 = history[0].get("image_prev") or history[0].get("image")
    n = len(history)

    messages = [{"role": "user", "content": [
        {"type": "text", "text": cot_instruction_inf(rules, max_steps)},
        {"type": "image", "image": frame0, "loss": False},
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
    all_actions = "\n".join(str(h.get("vlm_output") or h.get("action")) for h in history)
    content.append({"type": "text", "text": "\n" + THINK_CLOSE + "\n</think>\n" + all_actions})
    messages.append({"role": "assistant", "content": content})
    # NO feedback turn.

    out = {k: row.get(k) for k in (
        "__bagel_order_key", "task", "env_id", "env_kwargs", "seed", "episode_seed", "episode",
        "extra_state", "init_state", "init_state_hash", "stats", "version", "alignment_index",
    ) if k in row}
    out["messages"] = messages
    out["chunk_size"] = "inf"
    out["n_steps"] = n
    out["n_chunks"] = 1
    out["n_imagined_frames"] = n_frames
    out["n_feedback_turns"] = 0
    out["n_feedback_frames"] = 0
    out["max_steps"] = max_steps
    out["schema_version"] = "pacman_cot_perstep_kinf_v1"
    out["cot_row_type"] = "perstep_cot_singleturn_wholeepisode_nofeedback"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--src-subdir", default="data/train")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    args = ap.parse_args()
    os.makedirs(f"{args.out_root}/training", exist_ok=True)
    os.makedirs(f"{args.out_root}/metadata", exist_ok=True)
    shards = sorted(glob.glob(f"{args.src_root}/{args.src_subdir}/*.jsonl.gz"))
    assert shards, f"no shards under {args.src_root}/{args.src_subdir}"
    tasks = [(s, f"{args.out_root}/training/{os.path.basename(s)}") for s in shards]
    print(f"[Pacman CoT K=inf] workers={args.workers} shards={len(tasks)}", flush=True)
    with Pool(min(args.workers, max(len(tasks), 1))) as p:
        res = p.map(process_shard, tasks)
    tot = 0
    for name, n in sorted(res):
        print(f"  {name}: {n} rows", flush=True); tot += n
    json.dump({"study": "pacman_cot_perstep_kinf_nofeedback", "source_root": args.src_root, "total_rows": tot,
               "loss": "CE on assistant text; MSE on per-step imagined frames; frame0 = loss=0 input; NO feedback turns",
               "loader": "VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb')",
               "jpeg_quality": 95},
              open(f"{args.out_root}/metadata/cot_kinf_summary.json", "w"), indent=2)
    print(f"DONE Pacman CoT K=inf total_rows={tot}", flush=True)


if __name__ == "__main__":
    main()
