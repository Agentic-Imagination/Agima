#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""Maze2D CoT K-action-chunk world-model dataset with GT ENV-FEEDBACK re-grounding (Exp1).

Maze2D analog of remap_sokoban_cot_chunk_k_envfb.py / remap_pusht_cot_chunk_k_envfb.py, built
from the PLAIN per-step source (maze2d_{difficulty}_native256_stopreq_stepmsg_fixedstart_ordered_
20260606_stepmsg_fixedstart_v2: history entries {prompt, image_prev, image_next, vlm_output,
action, terminated}; frames are q95 JPEG baked into the source b64 and copied VERBATIM here).

    user:      [instruction, frame0(ctx, loss=0)]
    assistant: <think> a_s<img> ... a_{e-1}<img> </think> {K actions}   (imgs = per-step MSE targets)
    user:      ENV_FEEDBACK_LABEL <REAL chunk-end frame>   <-- loss=0 INPUT (GT re-grounding)
    ...                                                     (no feedback after the final chunk)

K in {1,3,5,10} or 'inf'. K='inf' = the CoT-inf endpoint: ONE assistant turn covers the WHOLE
episode (single chunk) and there are ZERO feedback turns (open-loop after frame0) -- same row
shape, the feedback branch simply never fires (mirrors remap_sokoban_cot_chunk_inf.py).

================================ PROMPT CONTRACT (design decision) ================================
The source's per-step counter messages ("This is step N. You are allowed to take {40-N} more
steps.", prompt_contract step0_full_stepgt0_step_count_only_v2) are DROPPED: the chunk family is
not per-step, so a per-step countdown would be unfaithful (it also leaks the executed-step count
into what should be a K-interval feedback signal). Instead the instruction turn states everything
ONCE up front:

    instruction = <maze2d rules>            (step-0 source prompt verbatim; it carries no counter)
                + <budget sentence>         "You are allowed to take at most {max_steps} steps in total."
                + <cadence sentence>        env-feedback interval; mirrors sokoban cot_instruction's
                                            cadence slot (remap_sokoban_cot_chunk_k.py) and, per the
                                            envfb design, states the feedback interval explicitly:
                                            K=1  "You act one move at a time, receiving environment
                                                  feedback after every action."
                                            K>1  "You act in chunks of {K} actions, receiving
                                                  environment feedback after every {K} actions."
                                            inf  one-chunk / only-frame-you-will-see / no feedback
                + FRAME0_LABEL              then frame0 (loss=0 ctx) follows

The budget sentence is NEW vs sokoban/pusht cot_instruction (which omit it): maze2d conveyed its
budget ONLY via the dropped per-step counters, so the total budget moves into the instruction.
No cadence sentence (incl. inf) ever mentions imagination: the instruction is SHARED with the
non-CoT ablation (remap_maze2d_noncot_chunk_k_envfb.py imports cot_instruction), exactly like the
sokoban pair -- see remap_sokoban_noncot_chunk_k_envfb.py's "never mentions imagination" note.
The maze2d chunk eval adapter MUST import cot_instruction / ENV_FEEDBACK_LABEL / think templates
from THIS module (train==eval).
===================================================================================================

Loader: the SAME task-agnostic envfb loader as sokoban/pusht (loss type 'interleaved_cot_envfb',
data/interleave_datasets/edit_dataset_envfb.py): assistant per-step imagined frames -> MSE targets;
user-turn images (frame0 + Env-Feedback) honor their explicit loss=False -> context.

Token budget: fits the 50000-token packing budget the K sweep trains with. Worst case is K=1 at
max_steps=40: 40 imagined + 39 feedback + frame0 = 80 images x ~512 tokens (native256: VAE
(256/16)^2=256 + ViT (224/14)^2=256) ~= 41k + short action text << 50k; easy episodes are far
shorter (median ~9 steps -> K=10 rows are mostly single-chunk).

Original data untouched; writes a NEW local repo per (difficulty, K). Whole-node multiprocess.
"""
import argparse, json, gzip, glob, os, shutil
from multiprocessing import Pool
from pathlib import Path

# SHARED env-feedback label: import the sokoban chunk family's single chokepoint (the envfb eval
# runner uses the same symbol) -> the feedback text is byte-identical across tasks and train==eval.
try:
    from remap_sokoban_cot_chunk_k_envfb import ENV_FEEDBACK_LABEL
except ImportError:
    from scripts.data.remap_sokoban_cot_chunk_k_envfb import ENV_FEEDBACK_LABEL

HF = str(Path(os.environ.get("AGIMA_DATA_ROOT", Path(__file__).resolve().parents[2] / "datasets")).expanduser().resolve())
SRC_V2 = {  # plain per-step source (stepmsg_fixedstart v2); the counters it carries are dropped here
    "easy": f"{HF}/maze2d_easy_native256_stopreq_stepmsg_fixedstart_ordered_20260606_stepmsg_fixedstart_v2",
    "hard": f"{HF}/maze2d_hard_native256_stopreq_stepmsg_fixedstart_ordered_20260606_stepmsg_fixedstart_v2",
}
DEFAULT_DATE_TAG = os.environ.get("DATE_TAG", "20260707_perseg")
JPEG_QUALITY = 95   # Maze2D source frames use q95 JPEG and are copied verbatim.


# ---- SHARED maze2d text builders (the maze2d chunk eval adapter imports these; keep in sync) ----
def _rules_prefix(step0_prompt: str) -> str:
    """Maze2D rules = the step-0 source prompt. Contract step0_full_stepgt0_step_count_only_v2 puts
    NO counter at step 0, so this is normally the identity; the defensive cut (mirroring the pusht
    _rules_prefix) guards against a counter ever being appended upstream."""
    cut = step0_prompt.find("This is step 1.")
    return (step0_prompt[:cut] if cut != -1 else step0_prompt).rstrip()


def _max_steps(row: dict) -> int:
    """Maze2D carries max_steps in row.metadata (=40 for both difficulties), not env_kwargs."""
    md = row.get("metadata")
    if isinstance(md, dict) and md.get("max_steps") is not None:
        return int(md["max_steps"])
    return 40   # maze2d default (metadata.max_steps=40, easy and hard)


# Per-chunk <think> open/close templates: byte-identical wording to the sokoban/pusht chunk family.
def think_open(n_actions: int) -> str:
    return ("Let me imagine the next move step by step." if n_actions == 1
            else f"Let me imagine the next {n_actions} moves step by step.")


THINK_CLOSE = "Done imagining; committing these actions:"
FRAME0_LABEL = "Starting maze:"   # labels the initial maze image (frame0), which immediately follows


def budget_sentence(max_steps: int) -> str:
    """TOTAL step budget, stated once (replaces the dropped per-step '{40-N} more steps' counters)."""
    return f"You are allowed to take at most {max_steps} steps in total."


# K='inf' cadence: one chunk, only frame the model will ever see, no feedback. NO imagination wording
# (shared with the non-CoT arm); otherwise mirrors remap_pusht_cot_chunk_inf.cadence_inf's structure.
CADENCE_INF = ("You act in ONE chunk covering the whole episode: this is the only frame you will see. "
               "Plan the entire route from this initial maze and commit the complete action sequence "
               "at once, with no environment feedback in between.")


def cadence(K) -> str:
    """Env-feedback interval sentence (the cot_instruction cadence slot)."""
    if K == "inf":
        return CADENCE_INF
    K = int(K)
    if K == 1:
        return "You act one move at a time, receiving environment feedback after every action."
    return (f"You act in chunks of {K} actions, receiving environment feedback "
            f"after every {K} actions.")


def cot_instruction(rules: str, max_steps: int, K) -> str:
    """User instruction = maze2d rules + total budget + ONE cadence line + the frame0 label; the
    initial maze image follows this text. Same slot structure as the sokoban/pusht cot_instruction
    (rules / cadence / FRAME0_LABEL) with the maze2d budget sentence added (see PROMPT CONTRACT).
    K is an int or 'inf'. SHARED by the CoT and non-CoT remaps and the eval adapter."""
    return (_rules_prefix(rules).rstrip() + "\n\n"
            + budget_sentence(max_steps) + " " + cadence(K) + "\n\n"
            + FRAME0_LABEL)


# maze2d plain-v2 row keys carried through (maze2d analog of the sokoban/pusht carry-key lists;
# trajectory_steps is the tiny {step_idx, action} list -- kept for eval/provenance alignment)
CARRY_KEYS = ("__bagel_order_key", "task", "env_id", "hash", "split", "alignment_index",
              "metadata", "init_state", "trajectory_steps", "schema")


def transform_row(row, K):
    """CoT, interval K (int or 'inf'), MULTI-TURN with GT env-feedback re-grounding.
    Imagination is PER-STEP (a frame after EVERY action, incl. the final ('stop','stop')) for all K.
    K='inf' -> one chunk spanning the whole episode; the feedback branch never fires."""
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
    n_frames = 0
    n_fb_frames = 0
    s = 0
    while s < n:
        e = min(s + k_int, n)
        seg = history[s:e]
        n_act = len(seg)
        content = []
        for j, h in enumerate(seg):
            a = str(h.get("vlm_output") or h.get("action"))
            txt = ("\n" + think_open(n_act) + "\n" + a) if j == 0 else ("\n" + a)
            content.append({"type": "text", "text": txt})
            f = h.get("image_next")
            if f:                                   # PER-STEP imagined frame (every action); MSE target
                content.append({"type": "image", "image": f, "loss": True})
                n_frames += 1
        chunk_actions = "\n".join(str(h.get("vlm_output") or h.get("action")) for h in seg)
        content.append({"type": "text", "text": "\n" + THINK_CLOSE + "\n</think>\n" + chunk_actions})
        messages.append({"role": "assistant", "content": content})
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
    out["n_imagined_frames"] = n_frames
    out["n_feedback_turns"] = n_chunks - 1          # 0 for K='inf' (open-loop)
    out["n_feedback_frames"] = n_fb_frames          # GT env-feedback frames (loss=0 inputs) == n_chunks-1
    out["max_steps"] = max_steps
    out["schema_version"] = f"maze2d_cot_perstep_envfb_k{K}_v1"
    out["cot_row_type"] = ("perstep_cot_singleturn_openloop" if K == "inf"
                           else "perstep_cot_multiturn_gtenvfeedback")
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


def parse_chunk_size(raw: str):
    raw = str(raw).strip().lower()
    if raw == "inf":
        return "inf"
    K = int(raw)
    assert K >= 1, f"chunk-size must be >=1 or 'inf', got {raw}"
    return K


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
    out_root = args.out_root or f"{HF}/maze2d_{args.difficulty}_native256_cot_chunk_k{K}_{args.date_tag}"
    os.makedirs(f"{out_root}/training", exist_ok=True)
    os.makedirs(f"{out_root}/metadata", exist_ok=True)
    shards = sorted(glob.glob(f"{src_root}/{args.src_subdir}/*.jsonl.gz"))
    assert shards, f"no shards under {src_root}/{args.src_subdir}"
    tasks = [(s, f"{out_root}/training/{os.path.basename(s)}", K, args.limit) for s in shards]
    print(f"[Maze2D CoT chunk ENVFB] difficulty={args.difficulty} chunk_size={K} "
          f"cores={os.cpu_count()} workers={args.workers} shards={len(tasks)} out={out_root}", flush=True)
    with Pool(min(args.workers, max(len(tasks), 1))) as p:
        res = p.map(process_shard, tasks)
    tot = 0
    for name, n in sorted(res):
        print(f"  {name}: {n} rows", flush=True); tot += n
    # maze2d source manifest name is train_manifest.jsonl (not the sokoban/pusht alignment name)
    for cand in (f"{src_root}/metadata/train_manifest.jsonl",
                 f"{src_root}/metadata/train_alignment_manifest.jsonl"):
        if os.path.exists(cand):
            shutil.copy2(cand, f"{out_root}/metadata/{os.path.basename(cand)}"); break
    json.dump({"study": "maze2d_action_chunking_cot_perstep_GT_ENVFEEDBACK", "chunk_size": K,
               "difficulty": args.difficulty, "date_tag": args.date_tag,
               "source_root": src_root, "total_rows": tot,
               "prompt_contract": "per-step counters DROPPED; instruction = rules + total-budget sentence + "
                                  "env-feedback cadence + 'Starting maze:'; K='inf' = single turn, no feedback",
               "format": "MULTI-TURN: user(rules+budget+cadence, frame0 ctx); per chunk assistant(<think> a<img> "
                         "...K... </think> {K actions}); BETWEEN chunks user(ENV_FEEDBACK_LABEL + REAL chunk-end "
                         "frame, loss=0 INPUT) for GT re-grounding; K='inf' -> ONE assistant turn, ZERO feedback",
               "env_feedback_label": ENV_FEEDBACK_LABEL,
               "loss": "CE on assistant text; MSE on per-step imagined frames; Env-Feedback frame + label = USER "
                       "turn loss=0; frame0/instruction = context",
               "loader": "VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb')",
               "jpeg_quality": JPEG_QUALITY},
              open(f"{out_root}/metadata/cot_chunk_summary.json", "w"), indent=2)
    print(f"DONE Maze2D CoT ENVFB difficulty={args.difficulty} chunk_size={K} total_rows={tot}", flush=True)


if __name__ == "__main__":
    main()
