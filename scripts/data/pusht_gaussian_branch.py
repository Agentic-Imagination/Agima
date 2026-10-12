#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""PT2+PT3 -- PushT Gaussian-coverage branch core (EXP2 branching substrate, pusht arm).

Companion to the Sokoban branching substrate (branch_format / sokoban_bfs_certify /
sokoban_branch_render). Where Sokoban proves a *deadlock* with BFS, PushT is continuous and
never truly "deadlocks": the counterfactual here is a **worse push** -- an alternative move
target that yields lower goal coverage than the expert, whose consequences are then made
concrete by rolling a *recovered* PPO policy forward so the model sees the block drifting
and coverage staying below the expert line.

WHAT THIS BUILDS
----------------
`build_branch(episode) -> dict | None` for one PushT expert (success) source row:
  1. KEY STEP  k*  -- the max-positive-coverage-jump move step, chosen by the SAME logic as
     export_pusht_allstep_thinking_cot.find_key_jump_annotation (argmax coverage_delta,
     ties -> nearest-to-stop, then earliest). The source only stores the step-0 init_state,
     so we replay the expert init_state -> k* through the real PushT physics env to recover
     the physics `state` at k* (the pre-key-action snapshot).
  2. GAUSSIAN ALTS -- N=8 alternative move targets ~ Gaussian(expert_target, sigma~=48px)
     in the 512-px world frame (clipped to [0,512]), serialized to norm4 EXACTLY like the
     source / remap_pusht_cot_chunk_inf rows (format_move_action(..., 'norm4')).
  3. COVERAGE SCORE -- each candidate's one-step resulting coverage via
     pusht_coverage_score.score_action (the checkpoint-free GT simulator, PT1) from the k*
     state.  ACCEPT the branch only if the expert action is the argmax-coverage candidate
     among {expert}+8 alts (i.e. the key step really is an expert-optimal decision); else
     RESAMPLE up to R=4 times; else DROP (return None).
  4. FAILED ALT -- the argmin-coverage alt of the accepted draw (the clearly-worse push).
  5. CONTINUE H steps -- apply the failed alt from the k* state (the divergence), then roll
     the RECOVERED PPO policy (delta64 / norm4 / control_hz=1, obs_dim=10) for H=recovery
     steps from the diverged physics state, rendering every frame (native 96x96, q90 -- the
     byte-identical source render path, verified) and recording coverage. The failed-branch
     coverage stays below the expert line.

RENDER / BYTE-IDENTITY (verified, do not "fix"):
  The CoT-inf source frames (pusht_96_norm4_..._cot100k) are 96x96 q90 JPEGs produced by
  `PIL.Image.fromarray(env.render()).convert('RGB').save(JPEG, quality=90)` where the env is
  built from the row's env_kwargs (visualization_width=96). Reproducing that path yields
  BYTE-IDENTICAL frames to the source (frame0 + every image_next matched, verified), so a
  spliced branch frame carries no JPEG/resolution tell -- exactly the Sokoban no-tell
  contract. pusht is trained/evaled NATIVE 96 (gen_eval profile 'pusht_norm4_native' applies
  no resize), so branch frames are stored at 96; `upscale_frame_b64` provides the downstream
  96->256 BICUBIC step (same as gen_eval's RenderResizeEnv) if a 256 VAE variant is wanted.

SERIALIZATION: alt actions are `('move', (x, y))` with x,y normalized [0,1] to 4 decimals --
byte-identical to what remap_pusht_cot_chunk_inf emits (it copies source `vlm_output`/`action`
verbatim). Round-trip format_move_action(decode_move_payload_xy(payload,'norm4'),'norm4') ==
source string, verified over an episode (0 mismatches).

LOADER CONTRACT (caller's responsibility): a spliced branch frame lives in the assistant
turn; interleaved_cot_envfb DROPS assistant images with loss=False, so the caller MUST emit
each branch frame with loss=True (MSE target). This module only returns base64 strings.

Self-test (loads the recovered PPO ckpt, runs build_branch on the first 2 source episodes):
    PYTHONPATH=. python scripts/data/pusht_gaussian_branch.py --src-root /path/to/source --ppo-ckpt /path/to/policy.pt
"""
from __future__ import annotations

import ast
import base64
import copy
import io
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Headless physics/render, matching pusht_coverage_score / the dataset generators.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# PT1 coverage scorer (the GT simulator) -- also gives us make_env with the exact ctor-param
# filtering used to instantiate a PushTEnv from a row's env_kwargs.
try:
    from pusht_coverage_score import make_env, score_action, score_rollout
except ImportError:  # imported as scripts.data.pusht_gaussian_branch
    from scripts.data.pusht_coverage_score import make_env, score_action, score_rollout

# Codec + recovered PPO policy. NOTE: gymnasium.envs.pusht.pusht re-exports
# decode_move_payload_xy but NOT format_move_action, so both are imported from ppo_policy
# (their canonical home) -- this also avoids importing export_pusht_allstep_thinking_cot,
# which fails at import time under PYTHONPATH=uwm-exp (it imports format_move_action from the
# pusht module) and pulls in the stale _make_planning_env planning helpers.
from gymnasium.envs.pusht.ppo_policy import (  # noqa: E402
    PushTPPOPolicy,
    build_pusht_ppo_observation,
    decode_move_payload_xy,
    format_move_action,
)

# ------------------------------------------------------------------------------------- #
# Config
# ------------------------------------------------------------------------------------- #
DEFAULT_PPO_CKPT = os.environ.get("AGIMA_PUSHT_PPO_CHECKPOINT")

try:
    from remap_pusht_cot_chunk_inf import DEFAULT_SRC  # noqa: F401
except ImportError:
    from scripts.data.remap_pusht_cot_chunk_inf import DEFAULT_SRC  # noqa: F401

ACTION_CODEC = "norm4"
WORLD_SIZE_PX = 512.0
N_ALTS = 8
SIGMA_PX = 48.0
MAX_RESAMPLE = 4          # R: additional draws of N alts after the first (<= R+1 draws total)
RECOVERY_STEPS = 4        # H: PPO-recovery steps rolled after the failed alt divergence
JPEG_QUALITY = 90         # source frames are q90
NATIVE_SIZE = 96          # source render size (visualization_width)
UPSCALE_SIZE = 256        # downstream VAE / RenderResizeEnv BICUBIC target

_PPO_CACHE: Dict[Tuple[str, str], PushTPPOPolicy] = {}


# ------------------------------------------------------------------------------------- #
# Helpers
# ------------------------------------------------------------------------------------- #
def load_ppo(ckpt: str = DEFAULT_PPO_CKPT, *, device: str = "cpu") -> PushTPPOPolicy:
    """Load (and cache) the recovered PushT PPO policy. Confirms delta64 / norm4 codec."""
    if not ckpt:
        raise ValueError("Supply a PPO checkpoint explicitly or set AGIMA_PUSHT_PPO_CHECKPOINT")
    key = (str(ckpt), str(device))
    pol = _PPO_CACHE.get(key)
    if pol is None:
        pol = PushTPPOPolicy.load(checkpoint_path=ckpt, device=device)
        _PPO_CACHE[key] = pol
    return pol


def _action_str(step: dict) -> str:
    """The verbatim norm4 action string, using the SAME accessor as remap_pusht_cot_chunk_inf."""
    return str(step.get("vlm_output") or step.get("action"))


def _render_b64(env: Any, *, quality: int = JPEG_QUALITY) -> str:
    """Render the env to a base64 JPEG via the EXACT source path (byte-identical, verified)."""
    img = Image.fromarray(np.asarray(env.render(), dtype=np.uint8))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def upscale_frame_b64(b64: str, size: int = UPSCALE_SIZE, *, quality: int = JPEG_QUALITY) -> str:
    """Downstream 96->256 BICUBIC upscale (matches gen_eval RenderResizeEnv). The stored
    branch frames are native 96; call this only for a 256 VAE variant."""
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    img = img.resize((int(size), int(size)), resample=Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _ppo_action_str(env: Any, policy: PushTPPOPolicy) -> str:
    """One deterministic PPO action from the env's current physics state -> norm4 string."""
    st = env.get_init_state()
    obs = build_pusht_ppo_observation(
        st["agent_position"],
        [st["block_position"][0], st["block_position"][1], st["block_angle"]],
        st["goal_pose"],
    )
    target_xy = policy.action(obs, first_step=False)  # pixel xy on the norm4 grid
    return format_move_action(target_xy, action_codec=ACTION_CODEC)


def _is_success(episode: dict) -> bool:
    """Mirror export_pusht_allstep_thinking_cot.is_success_record (the find_key_jump gate)."""
    stats = dict(episode.get("stats") or {})
    if bool(stats.get("success")):
        return True
    thr = float((episode.get("env_kwargs") or {}).get("success_threshold", 0.9))
    return (
        bool(stats.get("terminated"))
        and not bool(stats.get("truncated"))
        and float(stats.get("coverage", 0.0)) >= thr
    )


def _select_key_step(
    env_kwargs: dict, init_state: Any, actions: Sequence[str]
) -> Optional[Dict[str, Any]]:
    """Replay the expert init_state -> end through the real physics env and pick the key step
    = argmax positive coverage_delta over MOVE steps. This is the core selection logic of
    export_pusht_allstep_thinking_cot.find_key_jump_annotation (lines 442-476), reused here;
    the per-step pre-action physics `state` is captured so a branch can be scored from k*.
    Returns the winning candidate dict, or None if there is no move step."""
    env = make_env(env_kwargs)
    try:
        env.reset(init_state=copy.deepcopy(init_state))
        candidates: List[Dict[str, Any]] = []
        for step_idx, action in enumerate(actions):
            try:
                parsed = ast.literal_eval(str(action))
            except Exception:
                break
            if not (isinstance(parsed, tuple) and len(parsed) == 2):
                break
            state = copy.deepcopy(env.get_init_state())
            cov_before = float(env._get_coverage())
            _obs, _r, terminated, truncated, _info = env.step(str(action))
            cov_after = float(env._get_coverage())
            if str(parsed[0]) == "move":
                candidates.append(
                    {
                        "step_idx": int(step_idx),
                        "state": state,
                        "chosen_action": str(action),
                        "coverage_before": cov_before,
                        "coverage_after": cov_after,
                        "coverage_delta": cov_after - cov_before,
                        "distance_to_stop": len(actions) - 1 - step_idx,
                    }
                )
            if terminated or truncated:
                break
        if not candidates:
            return None
        candidates.sort(
            key=lambda it: (
                float(it["coverage_delta"]),
                -int(it["distance_to_stop"]),
                -int(it["step_idx"]),
            ),
            reverse=True,
        )
        return candidates[0]
    finally:
        env.close()


def _sample_alts(
    expert_xy: np.ndarray, rng: np.random.Generator, *, n: int, sigma_px: float
) -> List[Tuple[str, np.ndarray]]:
    """Sample n move targets ~ Gaussian(expert_xy, sigma_px) in the 512px world frame, clip
    to [0,512], serialize to norm4. (Perturbing the absolute target ≡ perturbing the delta,
    since the agent position at k* is fixed -- delta = target - agent.)"""
    out: List[Tuple[str, np.ndarray]] = []
    for _ in range(int(n)):
        xy = np.clip(
            np.asarray(expert_xy, dtype=np.float64) + rng.normal(0.0, float(sigma_px), size=2),
            0.0,
            WORLD_SIZE_PX,
        ).astype(np.float32)
        out.append((format_move_action(xy, action_codec=ACTION_CODEC), xy))
    return out


# ------------------------------------------------------------------------------------- #
# Public API
# ------------------------------------------------------------------------------------- #
def build_branch(
    episode: dict,
    *,
    ppo_ckpt: str = DEFAULT_PPO_CKPT,
    device: str = "cpu",
    n_alts: int = N_ALTS,
    sigma_px: float = SIGMA_PX,
    max_resample: int = MAX_RESAMPLE,
    recovery_steps: int = RECOVERY_STEPS,
    require_success: bool = True,
    seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Build a Gaussian-coverage failed branch for one PushT expert (success) source row.

    Returns None when the row cannot host a valid branch (not a success, no move step, or the
    expert is not argmax-coverage after max_resample resamples). Otherwise returns:
      {
        key_step:            int    -- k* (index into the episode history / action list)
        accepted:            bool   -- True (an accepted branch; None is returned on reject)
        expert_is_argmax:    bool   -- expert action was argmax-coverage among {expert}+alts
        alt_actions:         [str]  -- failed-branch action sequence: failed alt + H PPO acts
        alt_frames_b64:      [str]  -- native 96x96 q90 JPEG b64, one per alt_actions entry
        alt_coverages:       [float]-- GT coverage after each alt_actions entry (expert line ↓)
        expert_coverage:     float  -- one-step coverage of the expert action from k* (the line)
        # extras (not in the minimal contract, useful for storyboard/QA):
        key_action:          str    -- the expert move action at k*
        failed_alt_action:   str
        n_alts, sigma_px, n_resamples, recovery_steps
        alt_candidate_coverages:      [float]  -- one-step coverage of the 8 accepted alts
        expert_continuation_coverages:[float]  -- expert GT coverage at k*..k*+H (the line)
      }
    """
    if require_success and not _is_success(episode):
        return None
    history = episode.get("history") or []
    if not history or not episode.get("init_state"):
        return None
    env_kwargs = episode.get("env_kwargs") or {}
    init_state = episode["init_state"]
    actions = [_action_str(h) for h in history]

    key = _select_key_step(env_kwargs, init_state, actions)
    if key is None:
        return None
    key_step = int(key["step_idx"])
    key_state = key["state"]
    key_action = str(key["chosen_action"])

    # RNG seeded per-episode for reproducibility. Use a STABLE hash (hashlib) -- Python's
    # builtin hash() on strings is salted per-process (PYTHONHASHSEED) and would make the
    # accept/drop decision and the sampled alts non-reproducible across runs.
    if seed is None:
        ok = episode.get("__bagel_order_key")
        if ok is not None:
            import hashlib

            seed = int(hashlib.md5(str(ok).encode("utf-8")).hexdigest(), 16) % (2 ** 31)
        else:
            seed = key_step + 12345
    rng = np.random.default_rng(int(seed))

    # Expert one-step coverage from k* (the "expert line") via the SAME GT scorer as the alts.
    expert_coverage = float(score_action(env_kwargs, key_state, key_action))
    expert_xy = decode_move_payload_xy(ast.literal_eval(key_action)[1], action_codec=ACTION_CODEC)

    # ACCEPT only if expert is argmax-coverage; else resample up to max_resample times.
    accepted_alts: Optional[List[Tuple[str, np.ndarray, float]]] = None
    n_resamples = 0
    for attempt in range(int(max_resample) + 1):
        sampled = _sample_alts(expert_xy, rng, n=n_alts, sigma_px=sigma_px)
        scored = [
            (astr, xy, float(score_action(env_kwargs, key_state, astr))) for astr, xy in sampled
        ]
        alt_covs = [c for _, _, c in scored]
        if expert_coverage >= max(alt_covs):
            accepted_alts = scored
            n_resamples = attempt
            break
    if accepted_alts is None:
        return None  # expert never argmax -> not a clean key step; drop.

    alt_candidate_coverages = [c for _, _, c in accepted_alts]
    # FAILED alt = argmin one-step coverage among the accepted draw's alts.
    fi = int(np.argmin(alt_candidate_coverages))
    failed_alt_action = str(accepted_alts[fi][0])

    # CONTINUE: apply the failed alt (divergence), then roll recovered PPO for H steps.
    policy = load_ppo(ppo_ckpt, device=device)
    alt_actions: List[str] = []
    alt_frames_b64: List[str] = []
    alt_coverages: List[float] = []
    env = make_env(env_kwargs)
    try:
        env.reset(init_state=copy.deepcopy(key_state))
        _o, _r, terminated, truncated, _i = env.step(failed_alt_action)
        alt_actions.append(failed_alt_action)
        alt_frames_b64.append(_render_b64(env))
        alt_coverages.append(float(env._get_coverage()))
        for _ in range(int(recovery_steps)):
            if terminated or truncated:
                break
            ppo_str = _ppo_action_str(env, policy)
            _o, _r, terminated, truncated, _i = env.step(ppo_str)
            alt_actions.append(ppo_str)
            alt_frames_b64.append(_render_b64(env))
            alt_coverages.append(float(env._get_coverage()))
    finally:
        env.close()

    # Expert continuation coverage k*..k*+H (the full "expert line" over the branch horizon).
    horizon = 1 + int(recovery_steps)
    expert_seg = actions[key_step: key_step + horizon]
    expert_continuation_coverages = [
        float(c) for c in score_rollout(env_kwargs, key_state, expert_seg, stop_on_terminal=True)
    ]

    return {
        "key_step": key_step,
        "accepted": True,
        "expert_is_argmax": True,
        "alt_actions": alt_actions,
        "alt_frames_b64": alt_frames_b64,
        "alt_coverages": alt_coverages,
        "expert_coverage": expert_coverage,
        # extras
        "key_action": key_action,
        "failed_alt_action": failed_alt_action,
        "n_alts": int(n_alts),
        "sigma_px": float(sigma_px),
        "n_resamples": int(n_resamples),
        "recovery_steps": int(recovery_steps),
        "alt_candidate_coverages": alt_candidate_coverages,
        "expert_continuation_coverages": expert_continuation_coverages,
    }


def build_candidate_comparison(
    episode: dict,
    *,
    n_alts: int = N_ALTS,
    n_show: int = 3,
    sigma_px: float = SIGMA_PX,
    max_resample: int = MAX_RESAMPLE,
    require_success: bool = True,
    seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """The user-spec COMPARISON branch (plan wording: "choose from a few Gaussian-perturbed
    ones and the correct one"). NO PPO policy, NO multi-step continuation — pure remap plus
    ONE env step per candidate:

      1. key step k* = argmax coverage jump (same `_select_key_step` as build_branch);
      2. sample n_alts candidate moves ~ Gaussian(expert_xy, sigma_px);
      3. ACCEPT iff the recorded expert action is argmax one-step coverage among
         {expert} + candidates (else resample <= max_resample, else None);
      4. return a NO-LEAK candidate set of n_show total: the (n_show-1) hardest
         strictly-worse alternatives PLUS the expert action, SHUFFLED (seeded
         per-episode -- the correct one sits at a uniformly random position, formatted
         identically), each with its ONE-step rendered outcome (native 96x96 q90 b64).

    Returns None when the row cannot host a branch (not a success / no move step / expert
    never argmax / no strictly-worse candidate to show). Otherwise:
      { key_step, accepted=True, expert_is_argmax=True, key_action, expert_coverage,
        cand_actions:[str]*n_show, cand_frames_b64:[str]*n_show, cand_coverages:[float],
        n_alts, n_show, sigma_px, n_resamples, alt_candidate_coverages }
    """
    if require_success and not _is_success(episode):
        return None
    history = episode.get("history") or []
    if not history or not episode.get("init_state"):
        return None
    env_kwargs = episode.get("env_kwargs") or {}
    init_state = episode["init_state"]
    actions = [_action_str(h) for h in history]

    key = _select_key_step(env_kwargs, init_state, actions)
    if key is None:
        return None
    key_step = int(key["step_idx"])
    key_state = key["state"]
    key_action = str(key["chosen_action"])

    if seed is None:
        ok = episode.get("__bagel_order_key")
        if ok is not None:
            import hashlib

            seed = int(hashlib.md5(str(ok).encode("utf-8")).hexdigest(), 16) % (2 ** 31)
        else:
            seed = key_step + 12345
    rng = np.random.default_rng(int(seed))

    expert_coverage = float(score_action(env_kwargs, key_state, key_action))
    expert_xy = decode_move_payload_xy(ast.literal_eval(key_action)[1], action_codec=ACTION_CODEC)

    accepted_alts: Optional[List[Tuple[str, np.ndarray, float]]] = None
    n_resamples = 0
    for attempt in range(int(max_resample) + 1):
        sampled = _sample_alts(expert_xy, rng, n=n_alts, sigma_px=sigma_px)
        scored = [
            (astr, xy, float(score_action(env_kwargs, key_state, astr))) for astr, xy in sampled
        ]
        alt_covs = [c for _, _, c in scored]
        if expert_coverage >= max(alt_covs):
            accepted_alts = scored
            n_resamples = attempt
            break
    if accepted_alts is None:
        return None  # expert never argmax -> not a clean key step; drop.

    # NO-LEAK candidate set: the (n_show-1) hardest STRICTLY-worse alternatives PLUS the
    # expert action itself, SHUFFLED (seeded per-episode) so the correct one sits at a
    # uniformly random position and is formatted identically to the others. Ties are
    # excluded from the shown alts (a tie would contradict "improves the coverage the most").
    strictly_worse = [t for t in accepted_alts if t[2] < expert_coverage - 1e-9]
    if not strictly_worse:
        return None
    n_alt_show = max(1, int(n_show) - 1)
    shown_alts = sorted(strictly_worse, key=lambda t: t[2], reverse=True)[:n_alt_show]
    candidates = list(shown_alts) + [(key_action, expert_xy, expert_coverage)]
    order = rng.permutation(len(candidates))
    candidates = [candidates[int(i)] for i in order]

    cand_actions: List[str] = []
    cand_frames_b64: List[str] = []
    cand_coverages: List[float] = []
    env = make_env(env_kwargs)
    try:
        for astr, _xy, cov in candidates:
            env.reset(init_state=copy.deepcopy(key_state))
            env.step(str(astr))
            cand_actions.append(str(astr))
            cand_frames_b64.append(_render_b64(env))
            cand_coverages.append(float(cov))
    finally:
        env.close()

    return {
        "key_step": key_step,
        "accepted": True,
        "expert_is_argmax": True,
        "key_action": key_action,
        "expert_coverage": expert_coverage,
        "cand_actions": cand_actions,
        "cand_frames_b64": cand_frames_b64,
        "cand_coverages": cand_coverages,
        "n_alts": int(n_alts),
        "n_show": int(len(cand_actions)),
        "sigma_px": float(sigma_px),
        "n_resamples": int(n_resamples),
        "alt_candidate_coverages": [c for _, _, c in accepted_alts],
    }


# ------------------------------------------------------------------------------------- #
# Self-test
# ------------------------------------------------------------------------------------- #
def _load_source_episodes(src_root: str, n: int) -> List[dict]:
    import glob
    import gzip
    import json

    shards = sorted(glob.glob(f"{src_root}/data/train/*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {src_root}/data/train")
    rows: List[dict] = []
    for shard in shards:
        with gzip.open(shard, "rt") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    if len(rows) >= n:
                        return rows
    return rows


def _decode_size(b64: str) -> Tuple[int, int]:
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    img.load()
    return img.size


def _self_test(src_root: str, ppo_ckpt: str, n_episodes: int = 2) -> int:
    import glob
    import gzip
    import json

    print(f"[PT2+PT3 self-test] ppo_ckpt={ppo_ckpt}")
    print(f"[PT2+PT3 self-test] src={src_root}")

    # 1. PPO ckpt loads + delta64/norm4 confirmed.
    pol = load_ppo(ppo_ckpt, device="cpu")
    ppo_ok = (
        pol.action_control == "delta"
        and abs(pol.delta_action_scale - 64.0) < 1e-9
        and pol.action_codec == "norm4"
        and pol.observation_dim == 10
    )
    print(
        f"[PT2+PT3 self-test] PPO loaded: control={pol.action_control} delta_scale={pol.delta_action_scale} "
        f"codec={pol.action_codec} control_hz={pol.control_hz} obs_dim={pol.observation_dim} -> "
        f"{'OK' if ppo_ok else 'BAD'}"
    )

    episodes = _load_source_episodes(src_root, n_episodes)
    print(f"[PT2+PT3 self-test] loaded {len(episodes)} source episodes")

    n_accept = 0
    checks: Dict[str, bool] = {"ppo_delta64_norm4_obs10": ppo_ok}
    frame_size_ok = True
    upscale_ok = True
    serialize_ok = True
    below_line_any = False

    for ei, ep in enumerate(episodes):
        actions = [_action_str(h) for h in (ep.get("history") or [])]
        br = build_branch(ep, ppo_ckpt=ppo_ckpt, device="cpu")
        okk = str(ep.get("__bagel_order_key"))
        if br is None:
            print(f"[PT2+PT3 self-test] episode {ei} (order_key={okk}): DROPPED (expert not argmax / no move step)")
            continue
        n_accept += 1
        k = br["key_step"]
        failed_cov0 = br["alt_coverages"][0]
        exp_cov = br["expert_coverage"]
        below = failed_cov0 < exp_cov
        below_line_any = below_line_any or below
        n_cont = len(br["alt_frames_b64"])

        print(f"\n[PT2+PT3 self-test] episode {ei} (order_key={okk}): ACCEPTED")
        print(f"    key step k*                 = {k}  (action {br['key_action']})")
        print(f"    expert is argmax-coverage   = {br['expert_is_argmax']}  "
              f"(resamples used = {br['n_resamples']}/{MAX_RESAMPLE})")
        print(f"    expert one-step coverage    = {exp_cov:.4f}   (the expert line)")
        print(f"    8 alt one-step coverages    = {['%.4f' % c for c in br['alt_candidate_coverages']]}")
        print(f"    FAILED alt {br['failed_alt_action']}")
        print(f"       failed-alt coverage      = {failed_cov0:.4f}  ->  below expert: {below}")
        print(f"    continuation frames         = {n_cont}  (1 failed alt + {br['recovery_steps']} PPO recovery)")
        print(f"    branch coverages (alt line) = {['%.4f' % c for c in br['alt_coverages']]}")
        print(f"    expert continuation (line)  = {['%.4f' % c for c in br['expert_continuation_coverages']]}")

        # frame sizes: native 96, and 96->256 BICUBIC upscale.
        for fb in br["alt_frames_b64"]:
            w, h = _decode_size(fb)
            if (w, h) != (NATIVE_SIZE, NATIVE_SIZE):
                frame_size_ok = False
            up = upscale_frame_b64(fb, UPSCALE_SIZE)
            if _decode_size(up) != (UPSCALE_SIZE, UPSCALE_SIZE):
                upscale_ok = False
        print(f"    frames rendered             = 96x96 native  ->  256x256 (BICUBIC upscale) OK")

        # serialization: every alt action matches the norm4 remap serialization exactly.
        for a in br["alt_actions"]:
            payload = ast.literal_eval(a)[1]
            rt = format_move_action(
                decode_move_payload_xy(payload, action_codec=ACTION_CODEC), action_codec=ACTION_CODEC
            )
            if rt != a:
                serialize_ok = False
                print(f"    SERIALIZE MISMATCH: {a} != {rt}")

    # Independent proof that our format matches what remap emits verbatim: round-trip every
    # source action of episode 0 (remap copies these strings verbatim into the CoT rows).
    src_rt_mismatch = 0
    src_moves = 0
    for h in (episodes[0].get("history") or []):
        a = _action_str(h)
        p = ast.literal_eval(a)
        if p[0] != "move":
            continue
        src_moves += 1
        rt = format_move_action(decode_move_payload_xy(p[1], action_codec=ACTION_CODEC), action_codec=ACTION_CODEC)
        if rt != a:
            src_rt_mismatch += 1
    print(f"\n[PT2+PT3 self-test] remap-serialization round-trip on source ep0: "
          f"{src_moves - src_rt_mismatch}/{src_moves} move actions byte-identical")

    checks["at_least_one_accepted"] = n_accept >= 1
    checks["continuation_frames_96_native"] = frame_size_ok
    checks["upscale_96_to_256_ok"] = upscale_ok
    checks["alt_actions_match_remap_norm4"] = serialize_ok
    checks["source_roundtrip_byte_identical"] = (src_rt_mismatch == 0 and src_moves > 0)
    checks["failed_alt_below_expert_line"] = below_line_any if n_accept else True

    print(f"\n[PT2+PT3 self-test] accept rate: {n_accept}/{len(episodes)}")
    print("[PT2+PT3 self-test] CHECKS:")
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    ok_all = all(checks.values())
    print(f"[PT2+PT3 self-test] RESULT: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify Gaussian PushT branches with a PPO checkpoint.")
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--ppo-ckpt", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()
    raise SystemExit(_self_test(args.src_root, args.ppo_ckpt, args.episodes))
