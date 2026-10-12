#!/usr/bin/env python3
"""Build a K-action-CHUNK world-model dataset for sokoban (action-chunking study).

Interpolates between per-step (K=1, the existing plain CE+MSE world model) and single-shot
open-loop (K=inf, the baseline): the model observes a frame, emits K actions, then observes the
frame K steps later — re-observing (GT) or re-imagining (self-rollout) every K actions.

THE TRAINING PIPELINE IS UNCHANGED. Pure data remap into the SAME plain-history JSONL that
VLMGymDatasetCELOSS reads: each output "history" entry is one K-chunk —
  image_prev = obs at chunk start (step cK)            -> ctx; loss=True (MSE) for c>=1
  prompt     = chunk_prompt(...) (see below)
  vlm_output = the K actions of the chunk, "\n"-joined (CE target)
  image_next = obs K steps later (step (c+1)K)         (final chunk -> terminal frame; MSE)

PROMPT DESIGN (explicit feedback interval). The model is TOLD its interval K once, at step 0:
  step 0  : <game rules> + <interval sentence stating K> + <plain step counter>
  step >0 : <plain step counter only>   ("This is step N. You are allowed to take M more steps.")
The in-session feedback is the plain per-step counter (identical to the per-step baseline),
appearing every K env-steps. `chunk_step0_prompt`/`chunk_counter` are SHARED with the eval
runner (inference_VLM_Gym_chunk_eval_ddp.py) so train/eval prompts cannot drift.

Original data untouched; writes a NEW local repo per K. Whole-node multiprocess.
"""
import argparse, json, gzip, glob, os, shutil
from multiprocessing import Pool


# ---- SHARED prompt builder (imported by the chunk eval runner; keep in sync there) ----
def interval_sentence(K: int) -> str:
    return (f"You will be shown a new game frame after every {K} actions: at each frame, output up to "
            f"{K} `('move', action_id)` actions (one per line), then you will see the resulting frame and "
            f"continue. Finish with `('stop', 'stop')` once every crate is on a goal tile.")


def chunk_counter(step_idx: int, max_steps: int) -> str:
    return f"This is step {step_idx + 1}. You are allowed to take {max_steps - step_idx - 1} more steps."


def chunk_step0_prompt(rules: str, max_steps: int, K: int) -> str:
    """Step-0 prompt = game rules + interval sentence + plain counter."""
    return rules.rstrip() + "\n\n" + interval_sentence(K) + "\n" + chunk_counter(0, max_steps)


def chunk_prompt(rules: str, step_idx: int, max_steps: int, K: int) -> str:
    """Prompt the model sees at the observation at global step `step_idx`."""
    return chunk_step0_prompt(rules, max_steps, K) if step_idx == 0 else chunk_counter(step_idx, max_steps)


def _max_steps(row: dict) -> int:
    for src in (row.get("env_kwargs"), row.get("init_state")):
        if isinstance(src, dict) and src.get("max_steps") is not None:
            return int(src["max_steps"])
    return 20


# Explicit single-crate constraint. The source prompt states it implicitly ("...pushes it only
# if the cell behind the crate is empty"); the shared rule also explains that two crates
# cannot be pushed in a line at once. Injected in _rules_prefix() — the
# SHARED rules chokepoint imported by every remap (chunk / cot_chunk / cot_chunk_envfb) AND the eval
# runners -> the rule appears in every training instruction and the matching eval prompt (train==eval).
TWO_BOX_RULE = ("You can push at most one crate at a time: if the cell directly behind the crate you "
                "are pushing is blocked by another crate or a wall, nothing moves (neither the crates "
                "nor you) — you cannot push two crates in a row.")


def _rules_prefix(step0_prompt: str) -> str:
    cut = step0_prompt.find("This is step 1.")
    base = (step0_prompt[:cut] if cut != -1 else step0_prompt).rstrip()
    if os.environ.get("BAGEL_NO_TWO_BOX") == "1":
        return base                               # old-ckpt eval (pre-5f8a1b0): NO two-box rule, to match training
    if TWO_BOX_RULE in base:
        return base
    anchor = "the cell behind the crate is empty."
    if anchor in base:                       # place it right after the implicit push rule
        return base.replace(anchor, anchor + " " + TWO_BOX_RULE, 1)
    return base + "\n" + TWO_BOX_RULE        # fallback: append to the end of the rules


def chunk_history(history, K, rules, max_steps):
    T = len(history)
    out = []
    c = 0
    while c * K < T:
        s = c * K
        e = min((c + 1) * K, T)
        seg = history[s:e]
        out.append({
            "step": s,
            "chunk_index": c,
            "prompt": chunk_prompt(rules, s, max_steps, K),   # step-0: rules+interval+counter; else: plain counter
            "vlm_output": "\n".join(str(h.get("vlm_output") or h.get("action")) for h in seg),
            "action": "\n".join(str(h.get("action")) for h in seg),
            "image_prev": history[s].get("image_prev") or history[s].get("image"),
            "image": history[s].get("image"),
            "image_next": history[e - 1].get("image_next"),
            "reward": float(sum(float(h.get("reward") or 0.0) for h in seg)),
            "info": history[e - 1].get("info"),
            "n_actions_in_chunk": len(seg),
        })
        c += 1
    return out


def transform_row(row, K):
    history = row.get("history") or []
    if not history:
        raise ValueError(f"row {row.get('__bagel_order_key')} has no history")
    rules = _rules_prefix(history[0].get("prompt", ""))
    max_steps = _max_steps(row)
    out = {k: row.get(k) for k in (
        "__bagel_order_key", "task", "env_id", "env_kwargs", "seed", "episode_seed",
        "extra_state", "init_state", "init_state_hash", "stats", "version", "episode",
        "split", "upstream_env_id", "action_schema_version", "source_dataset", "alignment_index",
    ) if k in row}
    out["history"] = chunk_history(history, K, rules, max_steps)
    out["chunk_size"] = K
    out["n_chunks"] = len(out["history"])
    out["n_raw_steps"] = len(history)
    out["max_steps"] = max_steps
    out["schema_version"] = f"sokoban_action_chunk_k{K}_intvprompt_v1"
    out["cot_row_type"] = "action_chunk_world_model"
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
    print(f"chunk_size={K} cores={os.cpu_count()} workers={args.workers} shards={len(tasks)}", flush=True)
    with Pool(min(args.workers, max(len(tasks), 1))) as p:
        res = p.map(process_shard, tasks)
    tot = 0
    for name, n in sorted(res):
        print(f"  {name}: {n} rows", flush=True); tot += n
    man = f"{args.src_root}/metadata/train_alignment_manifest.jsonl"
    if os.path.exists(man):
        shutil.copy2(man, f"{args.out_root}/metadata/train_alignment_manifest.jsonl")
    json.dump({"study": "action_chunking", "chunk_size": K, "source_root": args.src_root,
               "total_rows": tot, "prompt_design": "explicit feedback interval stated once at step 0; plain counter in-session",
               "interval_sentence_example": interval_sentence(K),
               "loader": "VLMGymDatasetCELOSS (unchanged)"},
              open(f"{args.out_root}/metadata/chunk_summary.json", "w"), indent=2)
    print(f"DONE chunk_size={K} total_rows={tot}", flush=True)


if __name__ == "__main__":
    main()
