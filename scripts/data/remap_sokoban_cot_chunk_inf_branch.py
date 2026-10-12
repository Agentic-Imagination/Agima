#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""SK4 -- Sokoban irreversibility BRANCH-splice CoT-inf world-model dataset (Exp2 "branching").

Extends the CoT-inf single-turn open-loop sokoban dataset
(remap_sokoban_cot_chunk_inf.py) by splicing ONE counterfactual "dead-end" detour into
the model's private <think> at the earliest irreversible decision point:

    ...expert imagination up to step j-1...  a_{j-1}<img_{j-1}>
    <BRANCH_OPEN>  "What if instead I play <alt_push>? <templated deadlock rationale>"
    <alt_deadlock_frame (loss=True)>
    <BACKTRACK> <BRANCH_CLOSE>
    a_j<img_j> ...rest of episode...  </think>
    {full committed plan, byte-for-byte UNCHANGED}

Only the private imagination gains a rejected detour; the post-</think> committed plan is
IDENTICAL to the unbranched CoT-inf row. The base row is built by REUSING
remap_sokoban_cot_chunk_inf.transform_row VERBATIM; the critical step + proven-deadlock alt
frame come from sokoban_irrev_detect.detect_critical_step; the splice + the three neutral
natural-language markers + the row metadata come from branch_format. ONE branch per
trajectory max (the earliest proven-deadlocking box-push).

Episodes with no proven-deadlocking counterfactual (detect returns None, ~1/3 of easy eps)
emit the PLAIN CoT-inf row + no_branch_metadata -- in-distribution CONTROL rows. The row
`messages` shape is IDENTICAL to the CoT-inf / envfb K-chunk sets, so the SAME loader
consumes both branched and control rows: VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type
'interleaved_cot_envfb'). Every spliced alt frame is loss=True (an MSE target: the envfb
loader DROPS assistant images with loss=False), and frame0 stays the ONLY loss-0 image.

Original data untouched; writes a NEW local repo. Whole-node multiprocess. NEW file --
modifies no existing script / env / loader.
"""
import argparse, json, gzip, glob, os, shutil
from multiprocessing import Pool

# Robust to both run-as-script (scripts/data on sys.path[0]) and import-as-module
# (scripts.data.<mod>) contexts, matching the other remap/data modules.
try:
    from remap_sokoban_cot_chunk_inf import transform_row as build_cot_inf_row, DEFAULT_SRC
    from sokoban_irrev_detect import detect_critical_step
    from branch_format import (
        splice_branch, step_action_index, branch_metadata, no_branch_metadata,
        count_branch, BRANCH_OPEN, BACKTRACK, BRANCH_CLOSE, SCHEMA_VERSION,
    )
except ImportError:  # pragma: no cover - import-path shim
    from scripts.data.remap_sokoban_cot_chunk_inf import transform_row as build_cot_inf_row, DEFAULT_SRC
    from scripts.data.sokoban_irrev_detect import detect_critical_step
    from scripts.data.branch_format import (
        splice_branch, step_action_index, branch_metadata, no_branch_metadata,
        count_branch, BRANCH_OPEN, BACKTRACK, BRANCH_CLOSE, SCHEMA_VERSION,
    )


# ---- Templated sokoban deadlock rationale (CE-trained plain text; leads with the alt push so
#      the imagined frame is conditioned on that counterfactual action, mirroring the main-line
#      "a_j<img_j>"). One fixed template; the certifier already PROVED the board unsolvable. ----
_ALT_RATIONALE = (
    "pushing that box lands it in a deadlocked square it can never be moved out of, so the "
    "puzzle can no longer be solved"
)


def _alt_step_text(alt_action: str) -> str:
    """The single branch text run spliced between BRANCH_OPEN and the alt deadlock frame."""
    return f"What if instead I play {alt_action}? Here, {_ALT_RATIONALE}."


def build_row(row):
    """Build one SK4 output row from a raw source episode.

    Returns (out_dict, status) where status is one of:
      'branch'     -- a counterfactual detour was spliced (has_branch=True)
      'control'    -- no proven-deadlocking push found; plain CoT-inf row (has_branch=False)
      'detect_err' -- detect_critical_step raised; emitted as a control row (robust batch)
      'splice_err' -- the splice raised (broken stride); emitted as a control row

    The out row's `messages` are always a valid CoT-inf single-turn row; branching only adds
    a rejected detour strictly BETWEEN the <think> opener and the verbatim committed plan.
    """
    out = build_cot_inf_row(row)                     # VERBATIM base CoT-inf row (unchanged reuse)
    out["cot_schema_version"] = out.get("schema_version")  # preserve the cot-format marker

    try:
        det = detect_critical_step(row)
    except Exception as exc:                         # never let one bad episode kill a shard
        out.update(no_branch_metadata())
        out["branch_error"] = f"detect:{type(exc).__name__}:{exc}"
        return out, "detect_err"

    if det is None:
        out.update(no_branch_metadata())            # in-distribution CONTROL row
        return out, "control"

    j = det["step"]
    assistant_content = out["messages"][1]["content"]
    try:
        at = step_action_index(assistant_content, j)  # index of step-j action on a clean row
        alt_text_frames = [(_alt_step_text(det["alt_action"]), det["alt_deadlock_frame_b64"])]
        out["messages"][1]["content"] = splice_branch(assistant_content, at, alt_text_frames)
    except Exception as exc:                         # e.g. broken 2*j stride -> fall back to control
        out.update(no_branch_metadata())
        out["branch_error"] = f"splice:{type(exc).__name__}:{exc}"
        return out, "splice_err"

    n_branch_steps, n_branch_frames = count_branch(alt_text_frames)   # (1, 1)
    out.update(branch_metadata(True, j, n_branch_steps, n_branch_frames))
    # diagnostics (not part of the loader contract; safe to ignore downstream)
    out["branch_alt_action"] = det["alt_action"]
    out["branch_expert_action"] = det["expert_action"]
    out["branch_certify_states_expanded"] = det.get("certify_states_expanded")
    return out, "branch"


def process_shard(args_tuple):
    in_path, out_path = args_tuple
    n = n_branch = n_control = n_detect_err = n_splice_err = 0
    tmp = out_path + ".tmp"
    with gzip.open(in_path, "rt") as fin, gzip.open(tmp, "wt") as fout:
        for line in fin:
            if not line.strip():
                continue
            out, status = build_row(json.loads(line))
            fout.write(json.dumps(out) + "\n")
            n += 1
            if status == "branch":
                n_branch += 1
            elif status == "control":
                n_control += 1
            elif status == "detect_err":
                n_detect_err += 1
            elif status == "splice_err":
                n_splice_err += 1
    os.replace(tmp, out_path)                        # atomic publish (CoT-inf convention)
    return (os.path.basename(in_path), n, n_branch, n_control, n_detect_err, n_splice_err)


# defaults follow remap_sokoban_cot_chunk_inf (same SRC), out naming per SK4 spec:
# sokoban_easy_v8_cot_chunk_kinf_branch_<DATE_TAG>, DATE_TAG env default 20260709_perseg.
DATE_TAG = os.environ.get("DATE_TAG", "20260709_perseg")
DEFAULT_OUT = None


def _load_source_rows(n_want, src_root=None, subdir="trajectories/sokoban_easy/train"):
    """First `n_want` source episodes in the exact order the remap iterates them."""
    src_root = src_root or DEFAULT_SRC
    shards = sorted(glob.glob(f"{src_root}/{subdir}/*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {src_root}/{subdir}")
    out = []
    for shard in shards:
        with gzip.open(shard, "rt") as fin:
            for line in fin:
                if not line.strip():
                    continue
                out.append(json.loads(line))
                if len(out) >= n_want:
                    return out
    return out


def _selftest(n_want=20, src_root=None):
    """Dry-run: build SK4 rows for `n_want` source episodes, report has_branch ratio, and
    assert (a) the REAL envfb parse_row (with stubbed sinks) yields deadlock-frame loss=True,
    exactly frame0 loss=0, <think>/</think> once each, committed plan verbatim vs the plain
    row; and (b) exactly one branch per branched row."""
    print(f"[SK4 self-test] loading first {n_want} source episodes from {src_root or DEFAULT_SRC} ...", flush=True)
    rows = _load_source_rows(n_want, src_root=src_root)
    print(f"[SK4 self-test] loaded {len(rows)} source rows", flush=True)

    # Lazily import the REAL envfb loader; stub only its sinks so we exercise the true
    # parse_row control flow without needing a tokenizer / VAE / ViT transforms.
    try:
        from data.interleave_datasets.edit_dataset_envfb import VLMGymDatasetSokobanInterleavedCoTEnvFB
    except ImportError:
        from edit_dataset_envfb import VLMGymDatasetSokobanInterleavedCoTEnvFB  # pragma: no cover

    class _StubLoader(VLMGymDatasetSokobanInterleavedCoTEnvFB):
        def __init__(self):          # skip heavy tokenizer/transform init
            pass
        def _init_data(self):
            return {"texts": [], "img_loss": []}
        def _add_text(self, data, text, need_loss, **kw):
            data["texts"].append((text, int(bool(need_loss))))
            return data
        def _add_image(self, data, image, need_loss, need_vae=True, need_vit=True, **kw):
            data["img_loss"].append(int(bool(need_loss)))
            return data
        def _decode_b64_image(self, b64s):
            return b64s              # passthrough; the stub _add_image ignores the image
    stub = _StubLoader()

    def run_loader(out_row):
        parsed = stub.parse_row(json.dumps(out_row))
        all_text = "".join(t for t, _ in parsed["texts"])
        last_seg = parsed["texts"][-1][0] if parsed["texts"] else ""
        return all_text, parsed["img_loss"], last_seg

    n_branch = n_control = n_detect_err = n_splice_err = 0
    failures = []
    example = None

    for i, row in enumerate(rows):
        plain = build_cot_inf_row(row)               # independent PLAIN baseline (no aliasing)
        out, status = build_row(row)
        if status == "branch":
            n_branch += 1
        elif status == "control":
            n_control += 1
        elif status == "detect_err":
            n_detect_err += 1
            failures.append((i, status, out.get("branch_error")))
        elif status == "splice_err":
            n_splice_err += 1
            failures.append((i, status, out.get("branch_error")))

        b_text, b_img, b_last = run_loader(out)
        p_text, p_img, p_last = run_loader(plain)

        # --- checks for EVERY row (branched + control) ---
        if b_img.count(0) != 1:
            failures.append((i, status, f"loss0 images = {b_img.count(0)} (want exactly 1 = frame0)"))
        if b_text.count("<think>") != 1 or b_text.count("</think>") != 1:
            failures.append((i, status, f"<think>={b_text.count('<think>')} </think>={b_text.count('</think>')} (want 1 each)"))
        if b_last != p_last:
            failures.append((i, status, "committed-plan loader seg differs from plain row"))
        p_plan = plain["messages"][1]["content"][-1]["text"]
        b_plan = out["messages"][1]["content"][-1]["text"]
        if b_plan != p_plan:
            failures.append((i, status, "committed-plan messages text differs from plain row"))
        if out["messages"][1]["content"][0]["text"] != plain["messages"][1]["content"][0]["text"]:
            failures.append((i, status, "think-opener seg differs from plain row"))

        if status == "branch":
            # (a) the spliced alt deadlock frame became a loss=True MSE target: exactly ONE
            #     extra loss=1 image vs the plain row, and no assistant image is loss=0.
            if b_img.count(1) != p_img.count(1) + 1:
                failures.append((i, status, f"branch loss1 imgs {b_img.count(1)} != plain {p_img.count(1)}+1 (deadlock frame not a loss target?)"))
            if b_img.count(1) != len(b_img) - 1:
                failures.append((i, status, "some assistant image is loss=0 (would be DROPPED by envfb loader)"))
            # (b) exactly ONE branch: each of the three markers appears exactly once.
            for mark in (BRANCH_OPEN, BACKTRACK, BRANCH_CLOSE):
                if b_text.count(mark) != 1:
                    failures.append((i, status, f"marker {mark!r} count = {b_text.count(mark)} (want exactly 1)"))
            if (not out.get("has_branch")) or out.get("n_branch_steps") != 1 or out.get("n_branch_frames") != 1:
                failures.append((i, status, f"branch metadata off: has_branch={out.get('has_branch')} "
                                            f"n_branch_steps={out.get('n_branch_steps')} n_branch_frames={out.get('n_branch_frames')}"))
            if example is None:
                example = (i, out.get("branch_step"), out.get("branch_alt_action"),
                           out.get("branch_expert_action"), len(out["messages"][1]["content"]))
        else:
            if BRANCH_OPEN in b_text:
                failures.append((i, status, "control row contains a BRANCH_OPEN marker"))
            if out.get("has_branch"):
                failures.append((i, status, "control row has_branch=True"))

    n = len(rows)
    ratio = n_branch / n if n else 0.0
    print(f"[SK4 self-test] has_branch: {n_branch}/{n} = {ratio:.0%}  "
          f"(control={n_control}, detect_err={n_detect_err}, splice_err={n_splice_err})", flush=True)
    if example is not None:
        i, bstep, aa, ea, nseg = example
        print(f"[SK4 self-test] example branched ep {i}: branch_step={bstep} "
              f"alt_action={aa!r} expert_action={ea!r} assistant_content_segs={nseg}", flush=True)
    if not (0.40 <= ratio <= 0.95):
        print(f"[SK4 self-test] NOTE: has_branch ratio {ratio:.0%} is outside the ~67% band [40%,95%]", flush=True)

    ok = (not failures) and (n_detect_err == 0) and (n_splice_err == 0) and (n_branch >= 1)
    if failures:
        print(f"[SK4 self-test] {len(failures)} FAILURE(S):", flush=True)
        for f in failures[:25]:
            print(f"    ep {f[0]} [{f[1]}]: {f[2]}", flush=True)
    print(f"[SK4 self-test] RESULT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--src-subdir", default="trajectories/sokoban_easy/train")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    ap.add_argument("--selftest", type=int, default=0,
                    help="dry-run on N source episodes and exit (no dataset written)")
    args = ap.parse_args()

    if args.selftest:
        import sys
        sys.exit(_selftest(args.selftest, src_root=args.src_root))

    os.makedirs(f"{args.out_root}/training", exist_ok=True)
    os.makedirs(f"{args.out_root}/metadata", exist_ok=True)
    shards = sorted(glob.glob(f"{args.src_root}/{args.src_subdir}/*.jsonl.gz"))
    assert shards, f"no shards under {args.src_root}/{args.src_subdir}"
    tasks = [(s, f"{args.out_root}/training/{os.path.basename(s)}") for s in shards]
    print(f"[SK4 branch] cores={os.cpu_count()} workers={args.workers} shards={len(tasks)} out={args.out_root}", flush=True)
    with Pool(min(args.workers, max(len(tasks), 1))) as p:
        res = p.map(process_shard, tasks)

    tot = tot_branch = tot_control = tot_detect_err = tot_splice_err = 0
    for name, n, nb, nc, nde, nse in sorted(res):
        print(f"  {name}: {n} rows (branch={nb} control={nc} detect_err={nde} splice_err={nse})", flush=True)
        tot += n; tot_branch += nb; tot_control += nc; tot_detect_err += nde; tot_splice_err += nse
    ratio = tot_branch / tot if tot else 0.0

    man = f"{args.src_root}/metadata/train_alignment_manifest.jsonl"
    if os.path.exists(man):
        shutil.copy2(man, f"{args.out_root}/metadata/train_alignment_manifest.jsonl")
    json.dump({
        "study": "exp2_branching_sokoban_irreversibility_cot_inf_splice", "chunk_size": "inf",
        "date_tag": DATE_TAG, "source_root": args.src_root,
        "total_rows": tot, "n_branched": tot_branch, "n_control": tot_control,
        "n_detect_errors": tot_detect_err, "n_splice_errors": tot_splice_err,
        "branch_ratio": ratio, "schema_version": SCHEMA_VERSION,
        "base_format": "remap_sokoban_cot_chunk_inf (CoT-inf single-turn open-loop; transform_row reused VERBATIM)",
        "branch": "ONE counterfactual detour spliced into <think> before the earliest proven-deadlocking box-push "
                  "(sokoban_irrev_detect.detect_critical_step); BRANCH_OPEN <alt push + templated deadlock rationale> "
                  "<alt deadlock frame loss=True> BACKTRACK BRANCH_CLOSE; committed plan after </think> is UNCHANGED. "
                  "None-detect episodes emit the plain CoT-inf row as in-distribution controls.",
        "alt_rationale_template": _ALT_RATIONALE,
        "markers": {"open": BRANCH_OPEN, "backtrack": BACKTRACK, "close": BRANCH_CLOSE},
        "loss": "CE on assistant text (think + branch markers + rationale + full plan); MSE on every per-step "
                "imagined frame AND the spliced alt deadlock frame (all loss=True); frame0/instruction = context "
                "(the ONLY loss-0 image); zero loss-0 frames after frame0",
        "loader": "VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb'; same row shape as "
                  "the CoT-inf / envfb K-chunk sets, feedback branch never fires; alt frame is a normal MSE target)",
    }, open(f"{args.out_root}/metadata/branch_summary.json", "w"), indent=2)
    print(f"DONE SK4 branch total_rows={tot} branched={tot_branch} ({ratio:.0%}) "
          f"control={tot_control} detect_err={tot_detect_err} splice_err={tot_splice_err}", flush=True)


if __name__ == "__main__":
    main()
