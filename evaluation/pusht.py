#!/usr/bin/env python3
# Modified for Agima: package imports and portable resource configuration.
# =============================================================================
# PushT CoT K-action-chunk CLOSED-LOOP eval runner (envfb GT re-grounding) — MULTI-TURN.
#
# PushT analog of evaluation/sokoban.py (the sokoban envfb chunk
# runner; same architecture, same DDP scaffolding, same gen_cot_chunk_span generation path).
# Matches the pusht_96_norm4_cot_chunk_k{K}_20260622_perseg training rows
# (scripts/data/remap_pusht_cot_chunk_k_envfb.py): per chunk the model emits one assistant turn
#   <think> a<vis>imagine a<vis>imagine ... (K per-step frames) </think>  {the chunk's K actions}
# then the ENVIRONMENT replies with "Action executed." plus the REAL chunk-end frame (q90),
# repeat until ('stop','stop') or max_steps (PushT: 30). Per episode loop:
#   gen one chunk (gen_cot_chunk_span, in-span; text -> imagined frame -> ... until <|im_end|>)
#   -> parse the chunk's K actions (after </think>; tolerant to nonCoT chunks with no </think>)
#   -> execute them in the env -> append ENV_FEEDBACK_LABEL + GT chunk-end frame -> regenerate.
#
# PushT deltas vs the sokoban runner (each grounded in the base pusht eval,
# evaluation.runtime.run_pusht_episode):
#   * profile pusht_norm4_native, env built by G.make_profiled_env (enable_push_macros=False
#     + validate_pusht_contract: norm4 codec, marker none, stop-required prompt);
#   * reset from init_state (--pusht-reset-source init_state, required non-None);
#   * observation frames are 96x96 env renders JPEG-roundtripped at q90 (training data is q90;
#     launcher passes --image-quality 90) then upscaled by the VAE 256/256 / ViT 224/224 transforms;
#   * actions are CONTINUOUS norm4 strings "('move', (0.1234, 0.5678))" / "('stop', 'stop')" —
#     parse_action_chunk's balanced-paren + ast scan passes the exact substring to env.step
#     (the env ast.literal_eval's it), so the training serialization round-trips unchanged;
#   * success = stop-required coverage: terminated AND NOT truncated AND final action
#     ('stop','stop') AND info.is_success (env: coverage >= success_threshold=0.9,
#     gymnasium/envs/pusht/pusht.py _get_info) — identical predicate to run_pusht_episode.
# In-dist eval only (no easy/hard split for PushT — by design).
#
# Generation: CFG (image-gen) defaults to 1.0/1.0 = OFF (cond_dropout=0 => uncond branch
# untrained); override via BAGEL_COT_CFG_TEXT/IMG. BAGEL_CHUNK_K must match the trained K.
# BAGEL_NO_REGROUND=1 -> text-only feedback between chunks (true self-rollout ablation).
# Reuses gen_eval_ddp (G) scaffolding by monkeypatching G.run_episode: DDP sharding by
# idx % world_size with empty-rank early-return (no model load, no NCCL hang).
# =============================================================================
from __future__ import annotations
import json, os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import evaluation.runtime as G
import evaluation.base as base
from evaluation.environments import _reset_preserving_init_state
from data.data_utils import pil_img2rgb
# reuse single-shot helpers: task-agnostic chunk parsing + JPEG round-trip (q from args)
from evaluation.actions import parse_action_chunk, _jpeg_roundtrip
# SHARED PushT instruction builder (same module the remap imports) -> train==eval byte-identical
from scripts.data.remap_pusht_cot_chunk_k import cot_instruction
# PushT GT env-feedback label (train==eval; may differ from sokoban's, so import the PUSHT remap)
from scripts.data.remap_pusht_cot_chunk_k_envfb import ENV_FEEDBACK_LABEL
try:  # K=inf instruction (train==eval for the pusht kinf rows). Guarded: finite-K runs must
    # never depend on this module existing.
    from scripts.data.remap_pusht_cot_chunk_inf import cot_instruction_inf
except ImportError:  # pragma: no cover
    cot_instruction_inf = None

# BAGEL_CHUNK_K accepts 1/3/5/10 or 'inf'. K=inf = single-gen OPEN LOOP: one chunk covering
# the whole episode, ZERO feedback turns (matches the pusht kinf training rows, mirroring the
# sokoban/maze2d chunk runners' inf handling).
_RAW_K = os.environ.get("BAGEL_CHUNK_K", "0").strip().lower()
_IS_INF = _RAW_K in ("inf", "infinity", "kinf")
_CHUNK_K = 0 if _IS_INF else int(_RAW_K)
if _IS_INF and cot_instruction_inf is None:
    raise RuntimeError("BAGEL_CHUNK_K=inf needs scripts/data/remap_pusht_cot_chunk_inf.py "
                       "(cot_instruction_inf) importable — the train==eval K=inf instruction builder")
_DBG = os.environ.get("BAGEL_DEBUG_PROMPTS", "0") == "1"
# BAGEL_NO_REGROUND=1 -> TRUE self-rollout: between chunks feed TEXT-ONLY feedback (no real frame),
# so the model never re-sees the ground-truth observation and must rely on its own imagined frames.
# Default (0) keeps the GT re-grounding behavior (ENV_FEEDBACK_LABEL + the real chunk-end frame).
_NO_REGROUND = os.environ.get("BAGEL_NO_REGROUND", "0") == "1"
# BAGEL_DECODER: 'span' (default; persistent-cache gen_cot_chunk_span — single-span-shaped) or
# 'perseg' (rebuild-per-chunk _gen_one_chunk — matches PER-SEGMENT-trained ckpts). The sokoban
# K-anomaly investigation proved the shape matters: perseg ckpts under span decode collapse at
# K=1/3 (66%/19% easy) but score 90%/73% under the per-segment decoder. 'perseg' + default
# GT re-grounding = the previously-missing per-segment+envfb decode path.
_DECODER = os.environ.get("BAGEL_DECODER", "span").strip().lower()
if os.environ.get("RANK", "0") == "0":
    print(f"[DECODER-CHECK] BAGEL_DECODER={_DECODER}", flush=True)
# Text-only between-chunk feedback when re-grounding is disabled. Reuses the trained env-feedback
# label so the text channel still matches training; only the image is withheld.
NO_REGROUND_LABEL = os.environ.get("BAGEL_NO_REGROUND_LABEL", ENV_FEEDBACK_LABEL)
if os.environ.get("RANK", "0") == "0":
    print(f"[REGROUND-CHECK] BAGEL_NO_REGROUND={int(_NO_REGROUND)}  "
          f"({'TEXT-ONLY feedback, true self-rollout' if _NO_REGROUND else 'GT-frame re-grounding'})",
          flush=True)

# tokens / shapes (from the tested allstep CoT path). PushT VAE transform is 256/256, so
# imagined frames are generated at 256x256 (training MSE targets = 96x96 q90 frames upscaled
# to 256 by the same transform).
IM_END_ID = 151645
VISION_START = "<|vision_start|>"
GEN_IMG_SHAPE = (256, 256)
# Diagnostic: replace the model's self-imagined frame with the ACTUAL env-rendered frame
# after executing the imagined action. Tests whether self-rollout drift is the CoT
# bottleneck. K=1 only (imagined_action == committed_action byte-for-byte, so a single
# env.step per <|vision_start|> is unambiguous). See Task 2 in the plan file.
_INJECT_GT_FRAME = os.environ.get("BAGEL_PUSHT_INJECT_GT_FRAME", "0") == "1"
MAX_COT_ITERS = int(os.environ.get("BAGEL_COT_MAX_ITERS", "48"))   # max (text,image) pairs; generous
# K=inf generates the WHOLE episode's think + plan in one chunk -> needs a larger text cap
# (finite K keeps the historical 600 unless overridden).
MAX_TEXT_TOKENS = int(os.environ.get("BAGEL_COT_MAX_TEXT_TOKENS", "1600" if _IS_INF else "600"))
# image-gen CFG OFF by default (1.0/1.0) to match the training configuration: training uses cond_dropout=0, so the
# unconditional branch CFG would extrapolate from is untrained/OOD. Override via BAGEL_COT_CFG_TEXT/IMG.
CFG_TEXT = float(os.environ.get("BAGEL_COT_CFG_TEXT", "1.0"))
CFG_IMG = float(os.environ.get("BAGEL_COT_CFG_IMG", "1.0"))
if os.environ.get("RANK", "0") == "0":
    print(f"[CFG-CHECK] CFG_TEXT={CFG_TEXT} CFG_IMG={CFG_IMG}  (1.0/1.0 == CFG OFF)", flush=True)
NUM_TIMESTEPS = int(os.environ.get("BAGEL_COT_NUM_TIMESTEPS", "50"))

# --- Train-echo b64 override (diagnostic) --------------------------------------------------
# BAGEL_PUSHT_TRAIN_B64_SOURCE=<training shard(s), colon-separated jsonl.gz paths>
# When set, replaces the env's frame0 AND the GT re-grounding feedback frame (chunk end)
# with the byte-identical training row's b64 images (looked up by episode_seed).
# Isolates whether pixel content is the whole issue: if actions match training under this
# override, the model has memorized specific training pixels and rasterization/render drift
# is causing the collapse. If actions still drift, the model has a broader generalization gap.
_TRAIN_B64_SRC = os.environ.get("BAGEL_PUSHT_TRAIN_B64_SOURCE", "").strip()
_TRAIN_B64_INDEX: dict = {}
if _TRAIN_B64_SRC:
    import base64, gzip, io
    for _p in _TRAIN_B64_SRC.split(":"):
        _p = _p.strip()
        if not _p:
            continue
        with gzip.open(_p, "rt") as _fh:
            for _ln in _fh:
                try:
                    _row = json.loads(_ln)
                    _es = int(_row.get("episode_seed") or _row.get("seed") or -1)
                    if _es >= 0:
                        _TRAIN_B64_INDEX[_es] = _row
                except Exception:
                    continue
    if os.environ.get("RANK", "0") == "0":
        print(f"[TRAIN-B64] loaded {len(_TRAIN_B64_INDEX)} training rows from {_TRAIN_B64_SRC}", flush=True)


def _lookup_train_frame(episode_seed: int, chunk_index):
    """Return the training-row user-turn image for this episode_seed, or None.
    chunk_index=None -> initial frame0 (msg0[1]); int i -> GT feedback frame after chunk i (msg[2*(i+1)][1])."""
    if not _TRAIN_B64_INDEX:
        return None
    row = _TRAIN_B64_INDEX.get(int(episode_seed))
    if row is None:
        return None
    msgs = row.get("messages", [])
    msg_idx = 0 if chunk_index is None else 2 * (int(chunk_index) + 1)
    if msg_idx >= len(msgs):
        return None
    parts = msgs[msg_idx].get("content", [])
    for part in parts:
        if part.get("type") == "image":
            b64 = part.get("image") or part.get("image_b64")
            if not b64:
                continue
            import base64, io
            try:
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            except Exception:
                return None
    return None


@torch.inference_mode()
def _gen_one_chunk(infr, conversation, max_iters, env=None, image_quality=90, image_format="jpeg"):
    """Rebuild the context from `conversation` (list of str + PIL, as the TESTED AllstepCoTVLM does) and
    generate ONE assistant chunk: gen_text -> on <|vision_start|> gen_image (feed back vae+vit) -> repeat,
    until a gen_text that does NOT end in <|vision_start|> (the assistant turn ended at <|im_end|>).

    When _INJECT_GT_FRAME=True AND `env` is passed: after each <|vision_start|> we parse the imagined
    action from the emitted text so far, execute env.step(action), and feed env.render() as the
    "imagined" frame instead of calling gen_image. K=1 only. Returned executed_actions is the list of
    action strings already env.step'd inside this chunk — the outer loop must skip re-executing them.

    Returns (chunk_text, chunk_imgs, executed_actions).
    """
    parts: list = []
    imgs: list = []
    executed_actions: list = []
    inject_gt = _INJECT_GT_FRAME and env is not None and _CHUNK_K == 1
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):   # MUST cover the context rebuild (VAE/ViT encode) too, not just gen
        gen = infr.init_gen_context()
        cfg_text = deepcopy(gen)
        cfg_img = deepcopy(gen)
        for item in conversation:
            if isinstance(item, str):
                cfg_text = deepcopy(gen)
                gen = infr.update_context_text(item, gen)
                cfg_img = infr.update_context_text(item, cfg_img)
            else:
                im = pil_img2rgb(item)
                gen = infr.update_context_image(im, gen, vae=True, vit=True)
                # BUGFIX(image-CFG): cfg_img is the IMAGE-NEGATIVE branch -> text-only, NEVER fed images.
                # (was: cfg_img = update_context_image(im, cfg_img) which made cfg_img == gen -> image-CFG a no-op)
                cfg_text = deepcopy(gen)
        for _ in range(max_iters):
            text_out = infr.gen_text(gen, max_length=MAX_TEXT_TOKENS, do_sample=False,
                                     temperature=1.0, end_token_id=[IM_END_ID])
            parts.append(text_out)
            cfg_text = deepcopy(gen)
            gen = infr.update_context_text(text_out, gen)
            cfg_img = infr.update_context_text(text_out, cfg_img)
            if text_out.rstrip().endswith(VISION_START):
                if inject_gt:
                    # Parse the imagined action just emitted. For pushT cot k=1 the model produces
                    # `<think>...\n('move', (x,y))<|vision_start|>` — take the LAST action tuple in parts.
                    parsed_acts = parse_action_chunk("".join(parts))
                    # Skip any actions we've already executed in this chunk (defensive; for K=1 there
                    # should be exactly 1 new one per <|vision_start|>).
                    new_acts = parsed_acts[len(executed_actions):]
                    if not new_acts:
                        # No parseable new action — safest fallback is to abort GT-injection for this
                        # step and use standard gen_image (avoids feeding stale frames back).
                        img = infr.gen_image(image_shape=GEN_IMG_SHAPE, gen_context=gen,
                                             cfg_text_precontext=cfg_text, cfg_img_precontext=cfg_img,
                                             cfg_text_scale=CFG_TEXT, cfg_img_scale=CFG_IMG,
                                             cfg_interval=(0.4, 1.0), num_timesteps=NUM_TIMESTEPS)
                    else:
                        act = new_acts[0]
                        # Advance real env by this action. If the action is 'stop', env doesn't move —
                        # just render current state. If it's a valid 'move' with continuous coords, step.
                        try:
                            step_result = env.step(act)
                        except Exception as e:
                            # Malformed action; fall back to gen_image
                            img = infr.gen_image(image_shape=GEN_IMG_SHAPE, gen_context=gen,
                                                 cfg_text_precontext=cfg_text, cfg_img_precontext=cfg_img,
                                                 cfg_text_scale=CFG_TEXT, cfg_img_scale=CFG_IMG,
                                                 cfg_interval=(0.4, 1.0), num_timesteps=NUM_TIMESTEPS)
                        else:
                            # Track BOTH the executed action AND its env.step return tuple. The outer
                            # loop needs (reward, terminated, truncated, info) to correctly compute
                            # final_info / final_action for the success predicate at line 512-515.
                            executed_actions.append((act, step_result))
                            # Render env and JPEG-roundtrip to 256x256 (match training pipeline).
                            raw = base._render_like_to_array(env.render())
                            gt_rt = _jpeg_roundtrip(np.asarray(raw), image_quality, image_format)
                            img = Image.fromarray(np.asarray(gt_rt)).convert("RGB")
                            if img.size != GEN_IMG_SHAPE:
                                img = img.resize(GEN_IMG_SHAPE, Image.Resampling.BICUBIC)
                else:
                    img = infr.gen_image(image_shape=GEN_IMG_SHAPE, gen_context=gen,
                                         cfg_text_precontext=cfg_text, cfg_img_precontext=cfg_img,
                                         cfg_text_scale=CFG_TEXT, cfg_img_scale=CFG_IMG,
                                         cfg_interval=(0.4, 1.0), num_timesteps=NUM_TIMESTEPS)
                    if img.size != GEN_IMG_SHAPE:
                        img = img.resize(GEN_IMG_SHAPE, Image.Resampling.LANCZOS)
                imgs.append(img)   # keep original 256 for viz
                # Training data: raw 96x96 env frames bicubic-upsampled to 256 (see remap_pusht_cot_chunk_k_envfb.py).
                # Native gen_image() output is a real 256x256 with high-frequency content the VAE never saw in
                # training -> latent OOD -> accumulated drift makes the model emit <|im_end|> right after
                # <|vision_start|> from chunk 3+ onward (34-65% CoT-format collapse on K=3-10).
                # Fix: pass the imagined frame through the SAME 96->256 bicubic bottleneck training used
                # BEFORE VAE encode, so the fed-back frame has the same frequency envelope as training.
                if os.environ.get("BAGEL_PUSHT_BICUBIC_ROUNDTRIP", "1") == "1":
                    img_feed = img.resize((96, 96), Image.Resampling.BICUBIC).resize(GEN_IMG_SHAPE, Image.Resampling.BICUBIC)
                else:
                    img_feed = img
                ri = pil_img2rgb(img_feed)
                gen = infr.update_context_image(ri, gen, vae=True, vit=True)   # feed self-imagined frame back vae+vit
                # BUGFIX(image-CFG): cfg_img stays text-only (do NOT feed the generated frame in)
                cfg_text = deepcopy(gen)
            else:
                break   # gen_text ended on <|im_end|> -> this assistant chunk is done
    return "".join(parts), imgs, executed_actions


def _chunk_to_history_items(chunk_text, chunk_imgs):
    """Turn a generated chunk back into conversation items (text frags + imagined PIL frames interleaved),
    so the NEXT chunk's rebuilt context contains this chunk exactly as training stored it. The chunk's
    generated text has a <|vision_start|> marker where each frame was produced."""
    frags = chunk_text.split(VISION_START)
    items: list = []
    for i, frag in enumerate(frags):
        txt = frag + (VISION_START if i < len(frags) - 1 else "")   # re-attach the marker split removed
        if txt:
            items.append(txt)
        if i < len(chunk_imgs):
            items.append(chunk_imgs[i])
    return items


def _save_gif(frames, path: Path, ms: int = 600) -> None:
    if len(frames) < 2:
        return
    try:
        fr = [f.convert("RGB") for f in frames]
        fr[0].save(path.as_posix(), save_all=True, append_images=fr[1:], duration=ms, loop=0)
    except Exception:
        pass


def run_pusht_cot_episode(env, vlm, extra_state_key, ep: dict, args, rank: int) -> dict:
    env_id = ep["env_id"]
    if not G.is_pusht_env_id(env_id) and args.profile != "pusht_norm4_native":
        raise RuntimeError(f"PushT CoT chunk runner is pusht-only; got {env_id!r}")
    if _CHUNK_K <= 0 and not _IS_INF:
        raise RuntimeError("set BAGEL_CHUNK_K (chunk size, or 'inf') so the instruction cadence matches the trained model")
    seed = int(ep.get("seed") or ep.get("episode_seed") or 0)
    episode_seed = int(ep.get("episode_seed") or seed)
    max_steps = base._episode_max_steps(ep, args.max_steps)   # env_kwargs.max_steps=30 on the test split

    out_episode_dir = (Path(args.outdir) / base._env_slug(env_id) / args.profile / "cot_selfrollout"
                       / f"idx{int(ep['global_index']):07d}_seed{seed}_episode{episode_seed}")
    json_file = out_episode_dir / "episode_stats.json"
    if not args.overwrite and json_file.exists():
        try:
            st = (json.loads(json_file.read_text()).get("stats") or {})
            if st.get("success") is not None:
                return {"skipped": True, "success": bool(st.get("success"))}
        except Exception:
            pass

    # PushT resets from the recorded physics state, not the seed (run_pusht_episode contract).
    init_state = ep.get("init_state")
    if args.pusht_reset_source == "init_state" and init_state is None:
        raise RuntimeError(f"{env_id} episode {ep.get('global_index')} has no init_state for PushT eval.")
    base.set_global_seed(seed)
    reset_kwargs = {"seed": episode_seed}
    if args.pusht_reset_source == "init_state":
        reset_kwargs["init_state"] = init_state
    _reset_preserving_init_state(env, **reset_kwargs)
    infr = vlm.inferencer

    arr0 = base._render_like_to_array(env.render())
    frame0 = _jpeg_roundtrip(arr0, args.image_quality, args.image_format)   # q90 frame-0, as training
    # Diagnostic: override frame0 with the byte-identical training-row b64 image if configured.
    _train_f0 = _lookup_train_frame(episode_seed, None)
    if _train_f0 is not None:
        frame0 = np.asarray(_train_f0)
        if rank == 0 and int(ep.get("global_index", 0)) < 2:
            print(f"[TRAIN-B64] ep{ep.get('global_index')} using training b64 for frame0", flush=True)
    # Training rules = history[0]["prompt"] = the step-0 full prompt; the env reproduces it via
    # get_prompt(step_idx=0, ...) (prompt_mode first_full_then_step_only). cot_instruction's
    # _rules_prefix cuts at "This is step 1." so the step counter never enters the instruction.
    # SHARED with the remaps -> train==eval: finite K uses the chunk cadence; inf uses the
    # pusht CoT-inf instruction (remap_pusht_cot_chunk_inf.cot_instruction_inf).
    inner = getattr(env, "unwrapped", env)
    if _IS_INF:
        instruction = cot_instruction_inf(inner.get_prompt(step_idx=0, max_steps=max_steps), max_steps)
    else:
        instruction = cot_instruction(inner.get_prompt(step_idx=0, max_steps=max_steps), max_steps, _CHUNK_K)
    if _DBG and rank == 0 and int(ep.get("global_index", 0)) < 2:
        print(f"[COT_DBG] ep{ep.get('global_index')} K={_CHUNK_K} max_steps={max_steps}\n"
              f"  INSTRUCTION(pre-tokenizer)={instruction!r}", flush=True)

    # --- MULTI-TURN envfb rollout: per chunk -> generate <think> per-step imagine </think> + K actions,
    #     execute the K actions in the env, then append ENV_FEEDBACK_LABEL plus the real chunk-end
    #     frame and continue, until ('stop','stop') or max_steps.
    frames = [Image.fromarray(np.asarray(arr0)).convert("RGB")]
    hist: list = []
    imagined: list = []
    chunk_texts: list = []
    terminated = truncated = done = False
    final_info: dict = {}
    final_action = None
    cum_reward = 0.0
    gstep = 0
    MAX_CHUNKS = max_steps + 2

    # Native single-span CoT: ONE persistent gen_context carried across the whole episode.
    # Prefix the user turn (instruction + frame0) ONCE; each chunk is generated natively in-span by
    # gen_cot_chunk_span (token-by-token; switches to gen_image at <|vision_start|>, continues until
    # <|im_end|>; mutates `gen` IN PLACE so the cache carries). The GT env-feedback turn
    # (ENV_FEEDBACK_LABEL + the REAL chunk-end frame, q90, loss-0 INPUT == the dataset's image_next
    # -> train==eval) is appended into the SAME cache.
    if _DECODER == "perseg":
        # PER-SEGMENT decode (shape-correct for per-segment-trained ckpts): rebuild the context
        # from a [str, PIL, ...] conversation each chunk via _gen_one_chunk, and append the
        # envfb GT re-grounding turn (label + REAL chunk-end frame) into the conversation —
        # the per-segment + GT-re-grounding path that previously did not exist.
        conversation: list = [instruction, Image.fromarray(np.asarray(frame0)).convert("RGB")]
        for ci in range(MAX_CHUNKS):
            if done or gstep >= max_steps:
                break
            chunk_text, chunk_imgs, executed_in_chunk = _gen_one_chunk(
                infr, conversation, MAX_COT_ITERS,
                env=(env if _INJECT_GT_FRAME else None),
                image_quality=args.image_quality, image_format=args.image_format,
            )
            chunk_texts.append(chunk_text)
            imagined.extend(chunk_imgs)
            seg = chunk_text.split("</think>")[-1] if "</think>" in chunk_text else chunk_text
            actions = parse_action_chunk(seg)
            if _DBG and rank == 0 and int(ep.get("global_index", 0)) < 2:
                print(f"[COT_DBG] ep{ep.get('global_index')} chunk{ci} K={_CHUNK_K} n_imgs={len(chunk_imgs)} "
                      f"n_actions={len(actions)} n_executed_in_gen={len(executed_in_chunk)} "
                      f"[perseg{' +GTinject' if _INJECT_GT_FRAME else ''}]\n"
                      f"  CHUNK_TEXT[:320]={chunk_text[:320]!r}", flush=True)
            if not actions:
                break
            # When GT-frame injection ran, `executed_in_chunk` contains (action, step_result) tuples
            # for actions env.step'd inside _gen_one_chunk. Committed actions after </think> should be
            # SAME (K=1 dup shortcut) — skip the first len(executed_in_chunk) to avoid double-step.
            n_pre_exec = len(executed_in_chunk)
            actions_to_step = actions[n_pre_exec:] if _INJECT_GT_FRAME else actions
            # Log/record the pre-executed actions in hist AND propagate their env.step return
            # values (reward/terminated/info) to the outer state — otherwise `final_info` stays
            # empty at the epilogue and coverage/is_success default to 0/False.
            for a, step_result in executed_in_chunk:
                if gstep >= max_steps:
                    truncated = True; done = True; break
                _o, reward, terminated, truncated, info = step_result
                cum_reward += float(reward or 0.0)
                final_info = info or {}
                final_action = a
                gstep += 1
                frames.append(Image.fromarray(np.asarray(base._render_like_to_array(env.render()))).convert("RGB"))
                hist.append({"step": gstep - 1, "chunk": ci, "action": a, "reward": float(reward or 0.0),
                             "terminated": bool(terminated), "truncated": bool(truncated),
                             "info": {**{"gt_injected": True},
                                      **{k: base._json_safe(final_info.get(k)) for k in
                                         ("is_success", "coverage", "env_feedback", "n_contacts")}}})
                if G._is_stop_action(a) or terminated or truncated:
                    done = True; break
            if done:
                # if we already finished during the pre-executed batch (e.g. stop), skip outer loop
                pass
            else:
                for a in actions_to_step:
                    if gstep >= max_steps:
                        truncated = True
                        done = True
                        break
                    _o, reward, terminated, truncated, info = env.step(a)
                    cum_reward += float(reward or 0.0)
                    final_info = info or {}
                    final_action = a
                    gstep += 1
                    frames.append(Image.fromarray(np.asarray(base._render_like_to_array(env.render()))).convert("RGB"))
                    hist.append({"step": gstep - 1, "chunk": ci, "action": a, "reward": float(reward or 0.0),
                                 "terminated": bool(terminated), "truncated": bool(truncated),
                                 "info": {k: base._json_safe(final_info.get(k)) for k in
                                          ("is_success", "coverage", "env_feedback", "n_contacts")}})
                    if G._is_stop_action(a) or terminated or truncated:
                        done = True
                        break
            if done:
                break
            if _IS_INF:
                break   # K=inf: single open-loop chunk, no feedback turn
            conversation.extend(_chunk_to_history_items(chunk_text, chunk_imgs))
            if _NO_REGROUND:
                conversation.append(NO_REGROUND_LABEL)
            else:
                gt_obs = _jpeg_roundtrip(np.asarray(frames[-1]), args.image_quality, args.image_format)
                # Diagnostic: override GT feedback frame with the training-row b64 user-turn image.
                _train_gt = _lookup_train_frame(episode_seed, ci)
                if _train_gt is not None:
                    gt_obs = np.asarray(_train_gt)
                conversation.append(ENV_FEEDBACK_LABEL)
                conversation.append(Image.fromarray(np.asarray(gt_obs)).convert("RGB"))
        gen = None  # unified epilogue below expects the span path's names; nothing else to do
    else:
      gen = infr.init_gen_context()
      with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen = infr.update_context_text(instruction, gen)
        gen = infr.update_context_image(pil_img2rgb(frame0), gen, vae=True, vit=True)
        for ci in range(MAX_CHUNKS):
            if done or gstep >= max_steps:
                break
            chunk_text, chunk_imgs = infr.gen_cot_chunk_span(
                gen, image_shape=GEN_IMG_SHAPE, max_tokens=MAX_TEXT_TOKENS,
                cfg_text_scale=CFG_TEXT, cfg_img_scale=CFG_IMG, num_timesteps=NUM_TIMESTEPS)
            chunk_texts.append(chunk_text)
            imagined.extend(chunk_imgs)
            # committed actions come after </think>; nonCoT ckpts emit no </think> (and zero
            # imagined frames) -> parse the whole chunk text. parse_action_chunk stops at the
            # first ('stop','stop'); inner (x, y) coordinate tuples are rejected by the
            # str-first-element action-tuple check, so continuous PushT actions parse cleanly.
            seg = chunk_text.split("</think>")[-1] if "</think>" in chunk_text else chunk_text
            actions = parse_action_chunk(seg)
            if _DBG and rank == 0 and int(ep.get("global_index", 0)) < 2:
                print(f"[COT_DBG] ep{ep.get('global_index')} chunk{ci} K={_CHUNK_K} n_imgs={len(chunk_imgs)} "
                      f"n_actions={len(actions)}\n  CHUNK_TEXT[:320]={chunk_text[:320]!r}", flush=True)
            if not actions:
                break                                    # nothing parseable -> stop
            for a in actions:
                if gstep >= max_steps:
                    truncated = True
                    done = True
                    break
                _o, reward, terminated, truncated, info = env.step(a)
                cum_reward += float(reward or 0.0)
                final_info = info or {}
                final_action = a
                gstep += 1
                frames.append(Image.fromarray(np.asarray(base._render_like_to_array(env.render()))).convert("RGB"))
                hist.append({"step": gstep - 1, "chunk": ci, "action": a, "reward": float(reward or 0.0),
                             "terminated": bool(terminated), "truncated": bool(truncated),
                             "info": {k: base._json_safe(final_info.get(k)) for k in
                                      ("is_success", "coverage", "env_feedback", "n_contacts")}})
                if G._is_stop_action(a) or terminated or truncated:
                    done = True
                    break
            if done:
                break
            if _IS_INF:
                # K=inf single-gen OPEN LOOP: one chunk covers the whole episode; NO feedback
                # turn is ever appended (matches the pusht kinf training rows).
                break
            if _NO_REGROUND:
                # TRUE self-rollout: feed TEXT-ONLY feedback between chunks (NO image). The model never
                # re-sees the real observation; it must carry forward its own imagined frames.
                gen = infr.update_context_text(NO_REGROUND_LABEL, gen)
            else:
                # GT RE-GROUNDING feedback into the SAME cache: ENV_FEEDBACK_LABEL + the REAL
                # chunk-end frame (q90 round-trip == the training image_next JPEG surface).
                gt_obs = _jpeg_roundtrip(np.asarray(frames[-1]), args.image_quality, args.image_format)
                _train_gt = _lookup_train_frame(episode_seed, ci)
                if _train_gt is not None:
                    gt_obs = np.asarray(_train_gt)
                gen = infr.update_context_text(ENV_FEEDBACK_LABEL, gen)
                gen = infr.update_context_image(pil_img2rgb(gt_obs), gen, vae=True, vit=True)

    # stop-required coverage success, byte-identical predicate to G.run_pusht_episode:
    # env sets info.is_success = coverage >= success_threshold (0.9 in the test env_kwargs).
    success = (bool(terminated) and not bool(truncated)
               and G._is_stop_action(final_action) and bool(final_info.get("is_success", False)))

    out_episode_dir.mkdir(parents=True, exist_ok=True)
    if args.render or args.make_gif:
        rdir = out_episode_dir / "renders"
        for fi, fr in enumerate(frames):
            base.save_step_render(np.asarray(fr), rdir, fi)
        if args.make_gif:
            _save_gif(frames, out_episode_dir / "rollout.gif", ms=int(getattr(args, "gif_ms_per_frame", 600)))
    if getattr(args, "save_generated_images", False) and imagined:
        gdir = out_episode_dir / "gen_obs"
        gdir.mkdir(parents=True, exist_ok=True)
        for ii, im in enumerate(imagined):   # the per-step frames the model imagined inside the <think> blocks
            im.save((gdir / f"imagined{ii:02d}.png").as_posix())
    if getattr(args, "save_generated_images", False):
        (out_episode_dir / "cot_full_text.txt").write_text("\n\n----- CHUNK -----\n\n".join(chunk_texts))

    # provenance metadata distinguishes reground modes (fixes the sokoban envfb runner's
    # ambiguity where NO_REGROUND runs were labeled feedback_mode=gt_frame).
    feedback_mode = ("none_singlegen" if _IS_INF
                     else ("text_only_noreground" if _NO_REGROUND else "gt_frame"))
    result = {
        "env_id": env_id, "seed": seed, "episode_seed": episode_seed,
        "global_index": int(ep["global_index"]), "rank": rank, "profile": args.profile,
        "study": "pusht_action_chunking_cot", "runner": "pusht_cot_chunk_envfb", "decoder": _DECODER,
        "chunk_eval_mode": f"cot_multiturn_envfb_{feedback_mode}",
        "feedback_mode": feedback_mode,
        "feedback_label": (None if _IS_INF else (NO_REGROUND_LABEL if _NO_REGROUND else ENV_FEEDBACK_LABEL)),
        "no_reground": bool(_NO_REGROUND and not _IS_INF),
        "cot_frame_mode": os.environ.get("BAGEL_COT_FRAME_MODE", "exact"),
        "cfg_text_scale": CFG_TEXT, "cfg_img_scale": CFG_IMG,
        "chunk_k": ("inf" if _IS_INF else _CHUNK_K), "is_inf": bool(_IS_INF),
        "reset_source": args.pusht_reset_source,
        "version": "v1_pusht_cot_chunk_envfb_eval_ddp_multiturn", "instruction_max_steps": max_steps,
        "n_chunks": len(chunk_texts), "n_imagined": len(imagined), "n_executed": len(hist),
        "history": hist,
        "stats": {"step": len(hist), "reward": cum_reward,
                  "terminated": bool(terminated), "truncated": bool(truncated),
                  "success": bool(success), "final_action": final_action,
                  "coverage": float(final_info.get("coverage", 0.0)) if final_info else 0.0,
                  "n_chunks": len(chunk_texts), "n_imagined": len(imagined)},
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(base._json_safe(result), f, indent=2)
    return {"success": bool(success)}


if __name__ == "__main__":
    # Swap ONLY the per-episode logic; everything else (args, model load, pusht env build +
    # contract validation, idx%world_size sharding, empty-rank summaries) is the proven
    # gen_eval_ddp scaffolding.
    G.run_episode = run_pusht_cot_episode
    G.main()
