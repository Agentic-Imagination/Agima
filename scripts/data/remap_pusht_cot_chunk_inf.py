#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""PushT CoT K=inf (whole-episode single-chunk) world-model dataset — NO env feedback.

CoT-inf arm of the PushT feedback-interval study. SAME row shape as the K-chunk envfb
datasets (remap_pusht_cot_chunk_k_envfb.py) but ONE assistant turn covers the WHOLE
episode and there are ZERO between-chunk 'Env Feedback' turns: the model sees frame0
once, imagines the next frame after EVERY committed action (per-step MSE targets), then
commits the full action sequence — including the final ('stop', 'stop') — in one shot:

    user:      [instruction (cadence reworded for inf), frame0(ctx, loss=0)]
    assistant: <think> a_1<img> a_2<img> ... a_n<img> </think> {ALL n actions}  (imgs = MSE)

Trained with the SAME envfb loader (VLMGymDatasetSokobanInterleavedCoTEnvFB, loss type
'interleaved_cot_envfb'): degenerate case with no user-turn feedback images; frame0's
explicit loss=False is honored as context. The K builders' ENV_FEEDBACK_LABEL
("Action executed.") never appears here — there are no feedback turns to label.

Actions are copied VERBATIM from the source rows (`vlm_output` or `action`): pusht norm4
text "('move', (0.xxxx, 0.yyyy))" (4-decimal normalized image coords) and the final
"('stop', 'stop')" — eval-side parsing depends on this exact serialization.

Token budget: an inf row is a strict content-subset of the K=10 envfb row of the SAME
episode (identical per-step imagined frames; MINUS the feedback frames/labels; same
actions) -> fits the 50000-token budget the K sweep trains with (K=10 max ~18.6k).
PushT hard cap: n_steps <= max_steps=30 -> <= 30*768 + 512 image tokens + text << 50k.

Source = PushT PLAIN per-step history (pusht_..._stopreq_aligned_cot100k_..._ordered/
data/train, q90 JPEG baked into the source b64 frames). Original data untouched.
Select the source and output directories explicitly with --src-root and --out-root.
"""
import argparse, json, gzip, glob, os, shutil
from multiprocessing import Pool
from pathlib import Path

try:
    from remap_pusht_cot_chunk_k import _rules_prefix, _max_steps, think_open, THINK_CLOSE, FRAME0_LABEL
except ImportError:
    from scripts.data.remap_pusht_cot_chunk_k import (
        _rules_prefix, _max_steps, think_open, THINK_CLOSE, FRAME0_LABEL)

HF = str(Path(os.environ.get("AGIMA_DATA_ROOT", Path(__file__).resolve().parents[2] / "datasets")).expanduser().resolve())
DEFAULT_SRC = f"{HF}/pusht_96_norm4_visual_nomarker_stopreq_aligned_cot100k_20260604_ordered"
DEFAULT_DATE_TAG = os.environ.get("DATE_TAG", "20260707_perseg")


def cadence_inf() -> str:
    """K=inf rewording of the K-chunk cadence line ('You act in chunks of {K} actions.').
    The eval adapter MUST import this (train==eval instruction match, like cot_instruction for K)."""
    return ("You act in ONE chunk covering the whole episode: this is the only frame you will see. "
            "Imagine every move step by step, then commit the entire action sequence at once, "
            "with no environment feedback in between.")


def cot_instruction_inf(rules: str, max_steps: int) -> str:
    """Same structure as remap_pusht_cot_chunk_k.cot_instruction (rules + ONE cadence line +
    frame0 label) with the cadence reworded for K=inf. `max_steps` kept for signature parity."""
    return _rules_prefix(rules).rstrip() + "\n\n" + cadence_inf() + "\n\n" + FRAME0_LABEL


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
        {"type": "image", "image": frame0, "loss": False},      # initial frame: context, no loss
    ]}]

    # ONE assistant turn = the whole episode as a single chunk (seg = history[0:n]).
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
    # NO feedback turn: the episode ends with the committed full plan.

    out = {k: row.get(k) for k in (
        "__bagel_order_key", "task", "env_id", "env_kwargs", "seed", "episode_seed", "episode",
        "extra_state", "init_state", "init_state_hash", "stats", "version",
        "split", "upstream_env_id", "action_schema_version", "source_dataset", "alignment_index",
    ) if k in row}
    out["messages"] = messages
    out["chunk_size"] = "inf"
    out["n_steps"] = n
    out["n_chunks"] = 1
    out["n_imagined_frames"] = n_frames
    out["n_feedback_turns"] = 0
    out["n_feedback_frames"] = 0
    out["max_steps"] = max_steps
    out["schema_version"] = "pusht_cot_perstep_kinf_v1"
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
    ap.add_argument("--date-tag", default=DEFAULT_DATE_TAG)
    ap.add_argument("--out-root", required=True,
                    help='Output dataset directory')
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    args = ap.parse_args()
    out_root = args.out_root or f"{HF}/pusht_96_norm4_cot_chunk_kinf_{args.date_tag}"
    os.makedirs(f"{out_root}/training", exist_ok=True)
    os.makedirs(f"{out_root}/metadata", exist_ok=True)
    shards = sorted(glob.glob(f"{args.src_root}/{args.src_subdir}/*.jsonl.gz"))
    assert shards, f"no shards under {args.src_root}/{args.src_subdir}"
    tasks = [(s, f"{out_root}/training/{os.path.basename(s)}") for s in shards]
    print(f"[PushT CoT chunk KINF] cores={os.cpu_count()} workers={args.workers} "
          f"shards={len(tasks)} out={out_root}", flush=True)
    with Pool(min(args.workers, max(len(tasks), 1))) as p:
        res = p.map(process_shard, tasks)
    tot = 0
    for name, n in sorted(res):
        print(f"  {name}: {n} rows", flush=True); tot += n
    for cand in (f"{args.src_root}/metadata/train_alignment_manifest.jsonl", f"{args.src_root}/data/train_alignment_manifest.jsonl"):
        if os.path.exists(cand):
            shutil.copy2(cand, f"{out_root}/metadata/train_alignment_manifest.jsonl"); break
    json.dump({"study": "pusht_action_chunking_cot_perstep_KINF_singlechunk_nofeedback", "chunk_size": "inf",
               "source_root": args.src_root, "total_rows": tot,
               "format": "SINGLE-TURN: user(rules+inf-cadence,frame0 ctx); ONE assistant(<think> a<img> ...n... </think> "
                         "{ALL n actions incl ('stop','stop')}); ZERO Env-Feedback turns",
               "loss": "CE on assistant text; MSE on per-step imagined frames; frame0/instruction = context",
               "loader": "VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb')",
               "jpeg_quality": 90},
              open(f"{out_root}/metadata/cot_chunk_summary.json", "w"), indent=2)
    print(f"DONE PushT CoT KINF total_rows={tot}", flush=True)


if __name__ == "__main__":
    main()
