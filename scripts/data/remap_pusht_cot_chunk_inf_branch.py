#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""PT4 -- PushT CoT-inf single-turn dataset WITH a spliced Gaussian-coverage failed branch.

Exp2 "branching" arm for PushT. This is the CoT-inf whole-episode single-turn row
(remap_pusht_cot_chunk_inf.transform_row, REUSED VERBATIM) with, on the ~fraction of
episodes that host a clean counterfactual, ONE rejected detour spliced inside the model's
<think>: right before it commits the expert action a_{k*} at the key step, it first imagines
a WORSE push (a Gaussian-coverage failed alternative), renders the resulting frames as it is
rolled forward by a recovered PPO policy, recognizes the alternative underperforms, and
backtracks to the committed line. The post-</think> committed plan is IDENTICAL to the
unbranched row -- only the private imagination gains the rejected detour.

    ...expert imagination up to k*-1... a_{k*-1}<img_{k*-1}>
    <BRANCH_OPEN>
      failed_alt <alt_img>  ppo_1 <img> ... ppo_H <img>   (all MSE targets, loss=True)
      "This push moves the T the wrong way and lowers coverage ..."   (templated rationale)
    <BACKTRACK> <BRANCH_CLOSE>
    a_{k*}<img_{k*}> ...rest of episode... </think>
    {full committed plan, UNCHANGED}

Reuse map (NO logic re-derived here):
  * BASE ROW      : remap_pusht_cot_chunk_inf.transform_row (byte-for-byte the CoT-inf row).
  * BRANCH CORE   : pusht_gaussian_branch.build_branch (key step, Gaussian alts, GT coverage,
                    failed alt = argmin, PPO recovery continuation, expert continuation line).
  * SPLICE FORMAT : branch_format.splice_branch / step_action_index / count_branch /
                    branch_metadata / no_branch_metadata (the shared Exp2 vocabulary + the
                    loss=True alt-frame contract the envfb loader requires).

SPLICE GATE (control otherwise): splice ONLY IF
  (1) build_branch accepted (expert is argmax-coverage at k*), AND
  (2) key_step >= 1  -- step 0's action is fused with the <think> opener seg, so a branch
      cannot land before it (splice_branch requires at_index >= 1), AND
  (3) the failed alt GENUINELY UNDERPERFORMS across the WHOLE continuation: every
      alt_coverages[i] stays clearly below the corresponding expert_continuation_coverages[i]
      by at least `underperf_margin` (drops the ~2/34 cases where the recovered PPO catches
      the alt back up to the expert line), AND
  (4) the base row is a clean one-frame-per-step CoT-inf list at k* (integrity check: the
      content slot at 2*k* is exactly the expert key-action text run "\\n<key_action>").
Any miss -> plain CoT-inf row + no_branch_metadata (a control row, identical to PT's CoT-inf).

The base row's schema_version ("pusht_cot_perstep_kinf_v1") is preserved; the Exp2 branch
schema is recorded separately under "branch_schema_version" so both provenances survive.

Loader: VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb') -- SAME
loader as the CoT-inf control dataset. Alt frames are assistant images with loss=True (MSE
targets); frame0 stays a loss-0 user-turn context input; the committed plan is verbatim.

Output: the dataset directory passed with --out-root
(DATE_TAG env defaults to 20260709_perseg). training/ shards + metadata/ summary, mirroring
remap_pusht_cot_chunk_inf. CPU-only + PPO-per-episode -> whole-node build (slow); DO NOT run
the launcher casually.

Self-test (dry-run; detour mode needs the PPO checkpoint on CPU):
    PYTHONPATH=. python scripts/data/remap_pusht_cot_chunk_inf_branch.py \
        --self-test 10 --src-root /path/to/source --out-root /path/to/output \
        --mode detour --ppo-ckpt /path/to/policy.pt
"""
import argparse, json, gzip, glob, os, shutil, ast
from multiprocessing import Pool

# ---- base CoT-inf row builder (REUSED VERBATIM) ----------------------------------------
try:
    from remap_pusht_cot_chunk_inf import transform_row, DEFAULT_SRC, HF
except ImportError:
    from scripts.data.remap_pusht_cot_chunk_inf import transform_row, DEFAULT_SRC, HF

# ---- Exp2 branch splice format ----------------------------------------------------------
try:
    from branch_format import (
        splice_branch, step_action_index, count_branch,
        branch_metadata, no_branch_metadata,
        BRANCH_OPEN, BACKTRACK, BRANCH_CLOSE,
    )
except ImportError:
    from scripts.data.branch_format import (
        splice_branch, step_action_index, count_branch,
        branch_metadata, no_branch_metadata,
        BRANCH_OPEN, BACKTRACK, BRANCH_CLOSE,
    )

# ---- PushT Gaussian-coverage branch core ------------------------------------------------
try:
    import pusht_gaussian_branch as PGB
    from pusht_gaussian_branch import build_branch, build_candidate_comparison, DEFAULT_PPO_CKPT
except ImportError:
    import scripts.data.pusht_gaussian_branch as PGB
    from scripts.data.pusht_gaussian_branch import (
        build_branch, build_candidate_comparison, DEFAULT_PPO_CKPT,
    )

DEFAULT_DATE_TAG = os.environ.get("DATE_TAG", "20260709_perseg")
DEFAULT_MARGIN = float(os.environ.get("UNDERPERF_MARGIN", "0.02"))
DEFAULT_MODE = os.environ.get("BRANCH_MODE", "comparison")

# ---- COMPARISON mode (the plan's spec; DEFAULT) -----------------------------------------
# No-leak candidate comparison: the think weighs n_show candidates -- the (n_show-1)
# hardest Gaussian alternatives PLUS the expert action -- in a SHUFFLED order with
# IDENTICAL "Let me try ..." formatting (no positional or textual tell of which is
# correct). Every candidate gets its one-step imagined outcome frame (loss=True MSE).
# Only the CONCLUSION names the winner ("... improves the coverage the most"), which the
# acceptance rule (expert must be argmax-coverage, else resample) guarantees is truthful.
# NO PPO policy, NO multi-step continuation -- pure remap + one env step per candidate.
COMPARE_OPEN = "Before committing my next move, let me imagine a few candidate moves:"


def compare_try_text(action_str):
    """The IDENTICAL per-candidate try line (formatting must not leak which is correct)."""
    return f"Let me try {action_str}:"


def compare_close_text(winner_action_str):
    """Conclusion: names the winning move -- the only place the answer appears."""
    return (f"Comparing what I imagined, {winner_action_str} improves the coverage the "
            "most, so I commit it.")


# ------------------------------------------------------------------------------------- #
# Underperformance guard (pure function -> unit-testable without the PPO ckpt)
# ------------------------------------------------------------------------------------- #
def alt_underperforms(alt_coverages, expert_continuation_coverages, margin):
    """True IFF the failed-alt coverage line stays CLEARLY below the expert continuation
    line at every comparable step: alt_coverages[i] < expert_continuation_coverages[i] -
    margin for all i in the overlap. This is what rejects the ~2/34 episodes where the
    recovered PPO drags the alt back up to (or above) the expert line -- there the branch
    would teach the wrong lesson ("the alternative is fine"), so we drop it to a control row.
    Requires at least one comparable step; empty inputs -> False (cannot certify)."""
    if not alt_coverages or not expert_continuation_coverages:
        return False
    m = min(len(alt_coverages), len(expert_continuation_coverages))
    if m == 0:
        return False
    return all(
        float(alt_coverages[i]) < float(expert_continuation_coverages[i]) - float(margin)
        for i in range(m)
    )


def _pusht_rationale(alt_cov0, expert_cov):
    """Short templated PushT rationale (CE-trained as ordinary text; goes just before the
    BACKTRACK marker). Concrete but templated: names the failure mode + the coverage gap."""
    # Qualitative only — do NOT bake numeric coverage into CE-trained text (the model
    # cannot compute coverage at inference; teaching it to emit numbers => hallucinated precision).
    del alt_cov0, expert_cov  # kept in signature for caller compat; intentionally unused
    return (
        "This push moves the T the wrong way: it leaves much of the target "
        "uncovered and clearly does worse than the committed move, so the T "
        "drifts off the goal."
    )


def _stamp_branch_meta(out, meta):
    """Apply Exp2 branch metadata WITHOUT clobbering the base row's schema_version: the
    Exp2 schema is recorded under branch_schema_version, the rest of the keys are merged."""
    meta = dict(meta)
    out["branch_schema_version"] = meta.pop("schema_version")
    out.update(meta)


# ------------------------------------------------------------------------------------- #
# Row transform: base CoT-inf row + optional spliced branch
# ------------------------------------------------------------------------------------- #
def branch_transform_row(row, cfg):
    """Return (out_row, branched_bool). out_row is the CoT-inf row (transform_row verbatim);
    on a hosting episode it additionally carries the spliced branch + branch metadata,
    otherwise no_branch_metadata (a control row). NEVER raises on a branch-build failure:
    a broken branch degrades to a control row so the shard always emits every source row.
    Dispatches on cfg['mode']: 'comparison' (the plan's spec; default) or 'detour' (the
    PPO-continuation ablation)."""
    if cfg.get("mode", "comparison") == "comparison":
        return _transform_comparison(row, cfg)
    return _transform_detour(row, cfg)


def _transform_comparison(row, cfg):
    """COMPARISON mode: shuffled no-leak candidate set (expert included), identical
    'Let me try ...' lines + one-step imagined frames, conclusion picks the winner."""
    out = transform_row(row)  # base CoT-inf row, unmodified

    br = None
    try:
        br = build_candidate_comparison(
            row,
            n_alts=cfg["n_alts"],
            n_show=cfg["n_show"],
            sigma_px=cfg["sigma_px"],
            max_resample=cfg["max_resample"],
        )
    except Exception as e:  # noqa: BLE001 -- degrade to control, never drop the row
        out["branch_build_error"] = f"{type(e).__name__}: {e}"
        br = None

    branched = False
    if br is not None and br.get("accepted"):
        key_step = int(br["key_step"])
        content = out["messages"][1]["content"]
        idx = 2 * key_step
        clean_slot = (
            0 <= idx < len(content)
            and content[idx].get("type") == "text"
            and content[idx].get("text") == "\n" + str(br["key_action"])
        )
        if key_step >= 1 and clean_slot:
            cand_pairs = list(zip(
                [compare_try_text(a) for a in br["cand_actions"]],
                [str(f) for f in br["cand_frames_b64"]],
            ))
            n_bsteps, n_bframes = count_branch(cand_pairs)

            at_index = step_action_index(content, key_step)
            out["messages"][1]["content"] = splice_branch(
                content, at_index, cand_pairs,
                open_text=COMPARE_OPEN,
                close_text=compare_close_text(br["key_action"]),
            )

            _stamp_branch_meta(
                out,
                branch_metadata(
                    True,
                    branch_step=key_step,
                    n_branch_steps=n_bsteps,
                    n_branch_frames=n_bframes,
                ),
            )
            # QA / no-leak provenance (not required by the loader).
            out["branch_mode"] = "comparison"
            out["branch_key_action"] = str(br["key_action"])
            out["branch_correct_index"] = int(
                br["cand_actions"].index(str(br["key_action"]))
            )
            out["branch_cand_coverages"] = [float(c) for c in br["cand_coverages"]]
            out["branch_expert_coverage"] = float(br["expert_coverage"])
            out["branch_n_resamples"] = int(br.get("n_resamples", 0))
            branched = True

    if not branched:
        _stamp_branch_meta(out, no_branch_metadata())
        out["branch_mode"] = cfg.get("mode", "comparison")
    return out, branched


def _transform_detour(row, cfg):
    """DETOUR mode (ablation): ONE failed alt + H-step recovered-PPO continuation,
    then backtrack. Original PT4 behavior, unchanged."""
    out = transform_row(row)  # base CoT-inf row, unmodified

    br = None
    try:
        br = build_branch(
            row,
            ppo_ckpt=cfg["ppo_ckpt"],
            device="cpu",
            n_alts=cfg["n_alts"],
            sigma_px=cfg["sigma_px"],
            max_resample=cfg["max_resample"],
            recovery_steps=cfg["recovery_steps"],
        )
    except Exception as e:  # noqa: BLE001 -- degrade to control, never drop the row
        out["branch_build_error"] = f"{type(e).__name__}: {e}"
        br = None

    branched = False
    if br is not None and br.get("accepted"):
        key_step = int(br["key_step"])
        content = out["messages"][1]["content"]
        idx = 2 * key_step
        clean_slot = (
            0 <= idx < len(content)
            and content[idx].get("type") == "text"
            and content[idx].get("text") == "\n" + str(br["key_action"])
        )
        if (
            key_step >= 1
            and clean_slot
            and alt_underperforms(
                br["alt_coverages"], br["expert_continuation_coverages"], cfg["margin"]
            )
        ):
            # alt (action, frame) pairs -> loss=True MSE frames via splice_branch.
            alt_action_tuples = list(
                zip(
                    [str(a) for a in br["alt_actions"]],
                    [str(f) for f in br["alt_frames_b64"]],
                )
            )
            n_bsteps, n_bframes = count_branch(alt_action_tuples)
            rationale = _pusht_rationale(br["alt_coverages"][0], br["expert_coverage"])
            alt_text_frames = alt_action_tuples + [(rationale, None)]  # rationale = text-only

            at_index = step_action_index(content, key_step)  # == 2*key_step, validated
            out["messages"][1]["content"] = splice_branch(content, at_index, alt_text_frames)

            _stamp_branch_meta(
                out,
                branch_metadata(
                    True,
                    branch_step=key_step,
                    n_branch_steps=n_bsteps,
                    n_branch_frames=n_bframes,
                ),
            )
            # QA / storyboard provenance (not required by the loader).
            out["branch_mode"] = "detour"
            out["branch_key_action"] = str(br["key_action"])
            out["branch_failed_alt_action"] = str(br["failed_alt_action"])
            out["branch_expert_coverage"] = float(br["expert_coverage"])
            out["branch_alt_coverages"] = [float(c) for c in br["alt_coverages"]]
            out["branch_expert_continuation_coverages"] = [
                float(c) for c in br["expert_continuation_coverages"]
            ]
            out["branch_n_resamples"] = int(br.get("n_resamples", 0))
            out["branch_underperf_margin"] = float(cfg["margin"])
            branched = True

    if not branched:
        _stamp_branch_meta(out, no_branch_metadata())
    return out, branched


def _cfg_from_args(args):
    return {
        "mode": args.mode,
        "n_show": args.n_show,
        "ppo_ckpt": args.ppo_ckpt,
        "margin": args.underperf_margin,
        "n_alts": args.n_alts,
        "sigma_px": args.sigma_px,
        "max_resample": args.max_resample,
        "recovery_steps": args.recovery_steps,
    }


# ------------------------------------------------------------------------------------- #
# Shard worker + main
# ------------------------------------------------------------------------------------- #
def process_shard(task):
    in_path, out_path, cfg = task
    n = 0
    n_branch = 0
    tmp = out_path + ".tmp"
    with gzip.open(in_path, "rt") as fin, gzip.open(tmp, "wt") as fout:
        for line in fin:
            if not line.strip():
                continue
            out, branched = branch_transform_row(json.loads(line), cfg)
            fout.write(json.dumps(out) + "\n")
            n += 1
            n_branch += int(branched)
    os.replace(tmp, out_path)
    return (os.path.basename(in_path), n, n_branch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--src-subdir", default="data/train")
    ap.add_argument("--date-tag", default=DEFAULT_DATE_TAG)
    ap.add_argument("--out-root", required=True,
                    help='Output dataset directory')
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    ap.add_argument("--mode", choices=("comparison", "detour"), default=DEFAULT_MODE,
                    help="comparison = the plan's no-leak shuffled candidate set (DEFAULT, "
                         "no PPO); detour = failed-alt + PPO continuation (ablation)")
    ap.add_argument("--n-show", type=int, default=4,
                    help="comparison mode: total candidates shown INCL. the expert (a/b/c/d=4)")
    ap.add_argument("--ppo-ckpt", default=DEFAULT_PPO_CKPT)
    ap.add_argument("--underperf-margin", type=float, default=DEFAULT_MARGIN,
                    help="alt must stay this far below the expert continuation line at every step")
    ap.add_argument("--n-alts", type=int, default=PGB.N_ALTS)
    ap.add_argument("--sigma-px", type=float, default=PGB.SIGMA_PX)
    ap.add_argument("--max-resample", type=int, default=PGB.MAX_RESAMPLE)
    ap.add_argument("--recovery-steps", type=int, default=PGB.RECOVERY_STEPS)
    ap.add_argument("--self-test", type=int, default=0, metavar="N",
                    help="dry-run the branch splice on N source episodes and exit (no writes)")
    args = ap.parse_args()
    if args.mode == "detour" and not args.ppo_ckpt:
        ap.error("--ppo-ckpt or AGIMA_PUSHT_PPO_CHECKPOINT is required for detour mode")

    if args.self_test:
        raise SystemExit(_self_test(args, args.self_test))

    cfg = _cfg_from_args(args)
    suffix = "branchcmp" if args.mode == "comparison" else "branch"
    out_root = args.out_root or f"{HF}/pusht_96_norm4_cot_chunk_kinf_{suffix}_{args.date_tag}"
    os.makedirs(f"{out_root}/training", exist_ok=True)
    os.makedirs(f"{out_root}/metadata", exist_ok=True)
    shards = sorted(glob.glob(f"{args.src_root}/{args.src_subdir}/*.jsonl.gz"))
    assert shards, f"no shards under {args.src_root}/{args.src_subdir}"
    tasks = [(s, f"{out_root}/training/{os.path.basename(s)}", cfg) for s in shards]
    print(f"[PushT CoT-inf BRANCH mode={args.mode}] cores={os.cpu_count()} "
          f"workers={args.workers} shards={len(tasks)} out={out_root}", flush=True)
    with Pool(min(args.workers, max(len(tasks), 1))) as p:
        res = p.map(process_shard, tasks)
    tot = 0
    tot_branch = 0
    for name, n, nb in sorted(res):
        print(f"  {name}: {n} rows ({nb} branched)", flush=True)
        tot += n
        tot_branch += nb
    for cand in (f"{args.src_root}/metadata/train_alignment_manifest.jsonl",
                 f"{args.src_root}/data/train_alignment_manifest.jsonl"):
        if os.path.exists(cand):
            shutil.copy2(cand, f"{out_root}/metadata/train_alignment_manifest.jsonl")
            break
    summary = {"study": f"pusht_exp2_branching_cot_inf_gaussian_coverage_{args.mode}",
               "mode": args.mode,
               "chunk_size": "inf", "source_root": args.src_root,
               "total_rows": tot, "total_branched": tot_branch,
               "branch_ratio": (tot_branch / tot if tot else 0.0),
               "n_alts": cfg["n_alts"], "sigma_px": cfg["sigma_px"],
               "max_resample": cfg["max_resample"],
               "loader": "VLMGymDatasetSokobanInterleavedCoTEnvFB (loss type 'interleaved_cot_envfb')",
               "jpeg_quality": 90}
    if args.mode == "comparison":
        summary.update({
            "n_show": cfg["n_show"],
            "format": "CoT-inf single-turn + NO-LEAK shuffled candidate comparison inside "
                      "<think> before the key step: n_show candidates (expert + hardest "
                      "Gaussian alts) tried with IDENTICAL 'Let me try ...' lines, one-step "
                      "imagined frame each (loss=True); conclusion names the winner (expert, "
                      "guaranteed argmax-coverage by the resample rule); committed plan verbatim",
            "markers": {"open": COMPARE_OPEN,
                        "try": compare_try_text("<action>"),
                        "close": compare_close_text("<winner>")},
            "no_leak": "candidate order shuffled per-episode (seeded); expert formatted "
                       "identically to alternatives; winner named only in the conclusion",
        })
    else:
        summary.update({
            "underperf_margin": cfg["margin"], "recovery_steps": cfg["recovery_steps"],
            "ppo_ckpt": cfg["ppo_ckpt"],
            "format": "CoT-inf single-turn + spliced Gaussian-coverage failed branch "
                      "(failed alt + PPO continuation) inside <think> before the key step; "
                      "committed plan verbatim; alt frames loss=True",
        })
    json.dump(summary, open(f"{out_root}/metadata/cot_chunk_branch_summary.json", "w"), indent=2)
    print(f"DONE PushT CoT-inf BRANCH total_rows={tot} branched={tot_branch} "
          f"({tot_branch / tot * 100 if tot else 0:.1f}%)", flush=True)


# ------------------------------------------------------------------------------------- #
# Self-test
# ------------------------------------------------------------------------------------- #
def _load_source_episodes(src_root, subdir, n):
    shards = sorted(glob.glob(f"{src_root}/{subdir}/*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {src_root}/{subdir}")
    rows = []
    for shard in shards:
        with gzip.open(shard, "rt") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    if len(rows) >= n:
                        return rows
    return rows


def _assert_loader_compat(base_out, branched_out, br_meta):
    """Assert the spliced row satisfies the interleaved_cot_envfb loader contract vs the
    unbranched base row. Returns a dict of individual check results."""
    checks = {}
    base_content = base_out["messages"][1]["content"]
    content = branched_out["messages"][1]["content"]

    # (a) frame0 is a loss-0 USER-turn context input (unchanged).
    user_imgs = [c for c in branched_out["messages"][0]["content"] if c.get("type") == "image"]
    checks["frame0_user_context_loss0"] = (
        len(user_imgs) == 1 and user_imgs[0].get("loss") is False
    )

    # (b) EVERY assistant image is loss=True (the loader DROPS loss=False assistant images).
    a_imgs = [c for c in content if c.get("type") == "image"]
    checks["all_assistant_images_loss_true"] = bool(a_imgs) and all(
        c.get("loss") is True for c in a_imgs
    )

    # (c) <think> opener seg (index 0) preserved byte-for-byte.
    checks["think_opener_verbatim"] = content[0] == base_content[0]

    # (d) committed plan (last seg) preserved byte-for-byte vs the unbranched base row.
    checks["committed_plan_verbatim"] = content[-1] == base_content[-1]

    # (e) </think> appears ONLY in the last text seg (loader strips it from last seg only).
    text_with_close = [
        i for i, c in enumerate(content)
        if c.get("type") == "text" and "</think>" in c.get("text", "")
    ]
    checks["think_close_only_last"] = text_with_close == [len(content) - 1]

    # (f) branch is spliced BETWEEN opener (0) and committed plan (last): both markers present.
    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
    joined = "\n".join(texts)
    checks["branch_markers_present"] = (
        BRANCH_OPEN in joined and BACKTRACK in joined and BRANCH_CLOSE in joined
    )

    # (g) branch adds exactly n_branch_frames extra loss=True images vs the base row.
    base_imgs = [c for c in base_content if c.get("type") == "image"]
    checks["branch_frame_count_matches_meta"] = (
        len(a_imgs) - len(base_imgs) == br_meta["n_branch_frames"]
    )

    # (h) branch lands right before the key step's committed action (stride integrity):
    #     the committed key-action text run still exists downstream of the branch markers.
    open_idx = next(i for i, c in enumerate(content)
                    if c.get("type") == "text" and c.get("text", "").endswith(BRANCH_OPEN))
    close_idx = next(i for i, c in enumerate(content)
                     if c.get("type") == "text" and BRANCH_CLOSE in c.get("text", ""))
    checks["branch_before_key_step"] = (
        open_idx == 2 * br_meta["branch_step"]
        and close_idx > open_idx
        and content[close_idx + 1].get("type") == "text"
        and content[close_idx + 1].get("text") == "\n" + branched_out["branch_key_action"]
    )
    return checks


def _self_test_comparison(args, n):
    """Dry-run the COMPARISON (no-leak) splice on N source episodes. No PPO needed."""
    import re

    cfg = _cfg_from_args(args)
    print(f"[PT4-cmp self-test] src={args.src_root} n_show={cfg['n_show']} "
          f"n_alts={cfg['n_alts']} sigma_px={cfg['sigma_px']}")
    episodes = _load_source_episodes(args.src_root, args.src_subdir, n)
    print(f"[PT4-cmp self-test] loaded {len(episodes)} source episodes")

    checks = {}
    n_branch = n_control = 0
    correct_positions = []
    try_re = re.compile(r"^Let me try \('move', \([0-9.]+, [0-9.]+\)\):$")

    for ep in episodes:
        base_out = transform_row(ep)
        out, branched = branch_transform_row(ep, cfg)
        if not branched:
            n_control += 1
            checks.setdefault("control_rows_verbatim", True)
            if out["messages"] != base_out["messages"]:
                checks["control_rows_verbatim"] = False
            continue
        n_branch += 1
        content = out["messages"][1]["content"]
        base_content = base_out["messages"][1]["content"]

        # loader-compat basics
        a_imgs = [c for c in content if c.get("type") == "image"]
        base_imgs = [c for c in base_content if c.get("type") == "image"]
        checks.setdefault("all_assistant_images_loss_true", True)
        if not all(c.get("loss") is True for c in a_imgs):
            checks["all_assistant_images_loss_true"] = False
        checks.setdefault("committed_plan_verbatim", True)
        if content[-1] != base_content[-1]:
            checks["committed_plan_verbatim"] = False
        checks.setdefault("think_opener_verbatim", True)
        if content[0] != base_content[0]:
            checks["think_opener_verbatim"] = False

        # comparison structure: n_show identical try-lines + n_show frames added
        texts = [c.get("text", "").strip() for c in content if c.get("type") == "text"]
        try_lines = [t for t in texts if t.startswith("Let me try ")]
        checks.setdefault("n_show_try_lines", True)
        if len(try_lines) != out["n_branch_steps"]:
            checks["n_show_try_lines"] = False
        checks.setdefault("try_lines_format_identical_no_leak", True)
        if not all(try_re.match(t) for t in try_lines):
            checks["try_lines_format_identical_no_leak"] = False
        checks.setdefault("frames_added_matches", True)
        if len(a_imgs) - len(base_imgs) != out["n_branch_frames"]:
            checks["frames_added_matches"] = False

        # winner named ONLY in the conclusion; is among the tried candidates; lands before a_k*
        key_action = out["branch_key_action"]
        checks.setdefault("winner_among_tried", True)
        if f"Let me try {key_action}:" not in [t for t in try_lines]:
            checks["winner_among_tried"] = False
        close_idx = next(i for i, c in enumerate(content) if c.get("type") == "text"
                         and "so I commit it." in c.get("text", ""))
        checks.setdefault("branch_lands_before_key_action", True)
        if not (content[close_idx + 1].get("type") == "text"
                and content[close_idx + 1].get("text") == "\n" + key_action):
            checks["branch_lands_before_key_action"] = False
        correct_positions.append(int(out["branch_correct_index"]))

    print(f"[PT4-cmp self-test] branched={n_branch} control={n_control}")
    if correct_positions:
        from collections import Counter
        print(f"[PT4-cmp self-test] correct-index positions (no-leak shuffle): "
              f"{dict(Counter(correct_positions))}")
        checks["correct_position_varies"] = (
            len(set(correct_positions)) > 1 if len(correct_positions) >= 4 else True
        )
    ok = all(checks.values())
    for k, v in sorted(checks.items()):
        print(f"  {'PASS' if v else 'FAIL'} {k}")
    print(f"[PT4-cmp self-test] RESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


def _self_test(args, n):
    if args.mode == "comparison":
        return _self_test_comparison(args, n)
    from pusht_gaussian_branch import (
        format_move_action, decode_move_payload_xy, ACTION_CODEC,
    )

    cfg = _cfg_from_args(args)
    print(f"[PT4 self-test] src={args.src_root}")
    print(f"[PT4 self-test] ppo_ckpt={cfg['ppo_ckpt']}")
    print(f"[PT4 self-test] margin={cfg['margin']} n_alts={cfg['n_alts']} "
          f"sigma_px={cfg['sigma_px']} recovery_steps={cfg['recovery_steps']}")

    checks = {}

    # ---- (0) guard unit test: no ckpt needed; proves the underperformance guard FIRES ----
    below = alt_underperforms([0.10, 0.20, 0.30], [0.50, 0.60, 0.70], cfg["margin"])
    recover = alt_underperforms([0.10, 0.65, 0.30], [0.50, 0.60, 0.70], cfg["margin"])  # catches up @ i=1
    tie = alt_underperforms([0.10, 0.59, 0.30], [0.50, 0.60, 0.70], cfg["margin"])  # within margin @ i=1
    empty = alt_underperforms([], [0.5], cfg["margin"])
    print(f"[PT4 self-test] guard: clearly-below->{below}  recover->{recover}  "
          f"within-margin->{tie}  empty->{empty}")
    checks["guard_accepts_clearly_below"] = (below is True)
    checks["guard_drops_ppo_recovery"] = (recover is False)      # <-- guard actually fires
    checks["guard_drops_within_margin"] = (tie is False)
    checks["guard_drops_empty"] = (empty is False)

    # ---- (1) dry-run branch_transform_row on N source episodes ----
    episodes = _load_source_episodes(args.src_root, args.src_subdir, n)
    print(f"[PT4 self-test] loaded {len(episodes)} source episodes")

    n_branch = 0
    n_control = 0
    loader_checks = None
    serialize_ok = True
    remap_match_ok = True
    guard_fired_on_data = False
    example = None

    for ei, ep in enumerate(episodes):
        okk = str(ep.get("__bagel_order_key"))
        base_out = transform_row(ep)  # independent base row for verbatim comparisons
        out, branched = branch_transform_row(ep, cfg)

        # Was a branch built + accepted but then dropped by our gate (guard / key_step / clean)?
        try:
            br = build_branch(ep, ppo_ckpt=cfg["ppo_ckpt"], device="cpu",
                              n_alts=cfg["n_alts"], sigma_px=cfg["sigma_px"],
                              max_resample=cfg["max_resample"], recovery_steps=cfg["recovery_steps"])
        except Exception:
            br = None
        if (br is not None and br.get("accepted") and not branched
                and int(br["key_step"]) >= 1
                and not alt_underperforms(br["alt_coverages"],
                                          br["expert_continuation_coverages"], cfg["margin"])):
            guard_fired_on_data = True  # a real episode where PPO recovered -> dropped to control

        if not branched:
            n_control += 1
            assert out.get("has_branch") is False, f"control row {okk} has_branch != False"
            assert out.get("branch_step") == -1, f"control row {okk} branch_step != -1"
            # control row body must equal the plain CoT-inf row (only metadata differs)
            assert out["messages"] == base_out["messages"], f"control {okk} messages differ from base"
            print(f"[PT4 self-test] ep {ei} ({okk}): CONTROL")
            continue

        n_branch += 1
        step = out["branch_step"]
        print(f"[PT4 self-test] ep {ei} ({okk}): BRANCH @ key_step={step}  "
              f"n_branch_frames={out['n_branch_frames']}  "
              f"failed_alt={out['branch_failed_alt_action']}  "
              f"alt_cov={['%.3f' % c for c in out['branch_alt_coverages']]}  "
              f"exp_cont={['%.3f' % c for c in out['branch_expert_continuation_coverages']]}")

        # loader-compat checks on the FIRST branched row.
        if loader_checks is None:
            loader_checks = _assert_loader_compat(base_out, out, out)
            example = okk

        # alt action text byte-matches the norm4 remap serialization.
        content = out["messages"][1]["content"]
        # locate branch region (between BRANCH_OPEN and BACKTRACK/BRANCH_CLOSE)
        oi = next(i for i, c in enumerate(content)
                  if c.get("type") == "text" and c.get("text", "").endswith(BRANCH_OPEN))
        ci = next(i for i, c in enumerate(content)
                  if c.get("type") == "text" and BRANCH_CLOSE in c.get("text", ""))
        for c in content[oi + 1:ci]:
            if c.get("type") != "text":
                continue
            t = c.get("text", "").lstrip("\n")
            if not t.startswith("('move'"):
                continue  # skip the templated rationale line
            try:
                payload = ast.literal_eval(t)[1]
                rt = format_move_action(
                    decode_move_payload_xy(payload, action_codec=ACTION_CODEC),
                    action_codec=ACTION_CODEC,
                )
            except Exception:
                serialize_ok = False
                continue
            if rt != t:
                serialize_ok = False
                print(f"    SERIALIZE MISMATCH: {t!r} != {rt!r}")

        # the spliced alt action strings equal what build_branch produced (== remap serialization).
        spliced_moves = [
            c.get("text", "").lstrip("\n")
            for c in content[oi + 1:ci]
            if c.get("type") == "text" and c.get("text", "").lstrip("\n").startswith("('move'")
        ]
        # branch_failed_alt_action must be the first spliced move (build_branch's failed alt).
        if spliced_moves and spliced_moves[0] != out["branch_failed_alt_action"]:
            remap_match_ok = False
            print(f"    FIRST SPLICED MOVE {spliced_moves[0]!r} != failed_alt "
                  f"{out['branch_failed_alt_action']!r}")

    checks["at_least_one_branch"] = n_branch >= 1
    checks["alt_actions_norm4_serialization"] = serialize_ok
    checks["first_spliced_move_is_failed_alt"] = remap_match_ok
    if loader_checks is not None:
        checks.update({f"loader::{k}": v for k, v in loader_checks.items()})
    else:
        print("[PT4 self-test] WARNING: no branch produced in the sample; "
              "increase --self-test N (loader-compat checks skipped)")

    print(f"\n[PT4 self-test] sample: {n_branch} branch / {n_control} control "
          f"/ {len(episodes)} total  (branch_ratio={n_branch / max(len(episodes),1):.3f})")
    if guard_fired_on_data:
        print("[PT4 self-test] NOTE: underperformance guard fired on a REAL episode "
              "(accepted branch where PPO recovered -> dropped to control)")
    print("[PT4 self-test] CHECKS:")
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    ok_all = all(checks.values())
    print(f"[PT4 self-test] RESULT: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    main()
