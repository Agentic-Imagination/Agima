#!/usr/bin/env python3
# Modified for Agima: package imports and portable resource configuration.
# =============================================================================
# CoT-reformulated K-action-chunk sokoban eval runner (action-chunking study, CoT variant) — MULTI-TURN.
#
# Matches the multi-turn per-step training data: per chunk the model emits one assistant turn
#   <think> a<vis>imagine a<vis>imagine ... (K per-step frames) </think>  {the chunk's K actions}
# then the ENVIRONMENT replies with a user turn "Action executed." plus the real chunk-end frame, repeat,
# until ('stop','stop') or max_steps. Per episode loop:
#   gen one chunk (gen_text -> <|vision_start|> -> gen_image, feed back vae+vit; until <|im_end|>)
#   -> parse the chunk's K actions (after </think>) -> execute them in env (tracks true state for scoring)
#   -> append the chunk + "Action executed." + GT chunk-end frame to the conversation -> regenerate.
# The model SELF-IMAGINES every per-step frame inside each chunk, then the env re-grounds the next
# chunk with the real post-chunk frame, matching the envfb training rows. Success = stop_then_solved.
#
# Generation reuses the TESTED AllstepCoTVLM rebuild-per-turn pattern; CFG (image-gen) defaults to
# 1.0/1.0 = OFF, aligned with training (cond_dropout=0 => uncond branch untrained). Override via
# BAGEL_COT_CFG_TEXT/IMG. Reuses gen_eval_ddp (G) scaffolding
# by monkeypatching G.run_episode. BAGEL_CHUNK_K must match the trained K (instruction cadence).
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
# reuse single-shot helpers (chunk parsing + q95 round-trip) + the SHARED instruction builder
from evaluation.actions import parse_action_chunk, _jpeg_roundtrip
from scripts.data.remap_sokoban_cot_chunk_k import cot_instruction
from scripts.data.remap_sokoban_cot_chunk_k_envfb import ENV_FEEDBACK_LABEL   # GT env-feedback label (train==eval)
try:  # K=inf instruction (train==eval for kinf/branch/forward). Guarded: finite-K runs must
    # never depend on this module existing.
    from scripts.data.remap_sokoban_cot_chunk_inf import cot_inf_instruction
except ImportError:  # pragma: no cover
    cot_inf_instruction = None

# BAGEL_CHUNK_K accepts 1/3/5/10 or 'inf'. K=inf = single-gen OPEN LOOP: one chunk covering
# the whole episode, ZERO feedback turns (matches the kinf / branch / forward training rows,
# mirroring the maze2d chunk runner's inf handling).
_RAW_K = os.environ.get("BAGEL_CHUNK_K", "0").strip().lower()
_IS_INF = _RAW_K in ("inf", "infinity", "kinf")
_CHUNK_K = 0 if _IS_INF else int(_RAW_K)
_DBG = os.environ.get("BAGEL_DEBUG_PROMPTS", "0") == "1"
# BAGEL_NO_REGROUND=1 -> TRUE self-rollout: between chunks feed TEXT-ONLY feedback (no real frame),
# so the model never re-sees the ground-truth observation and must rely on its own imagined frames.
# Default (0) keeps the GT re-grounding behavior (ENV_FEEDBACK_LABEL + the real chunk-end frame).
_NO_REGROUND = os.environ.get("BAGEL_NO_REGROUND", "0") == "1"
# BAGEL_DECODER: 'span' (default; persistent-cache gen_cot_chunk_span — single-span-shaped) or
# 'perseg' (rebuild-per-chunk _gen_one_chunk — matches PER-SEGMENT-trained ckpts). The K-anomaly
# investigation proved the shape matters: perseg ckpts under span decode collapse at K=1/3
# (66%/19% easy) but score 90%/73% under the per-segment decoder. 'perseg' + default
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

# tokens / shapes (from the tested allstep CoT path)
IM_END_ID = 151645
VISION_START = "<|vision_start|>"
GEN_IMG_SHAPE = (256, 256)
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


@torch.inference_mode()
def _gen_one_chunk(infr, conversation, max_iters):
    """Rebuild the context from `conversation` (list of str + PIL, as the TESTED AllstepCoTVLM does) and
    generate ONE assistant chunk: gen_text -> on <|vision_start|> gen_image (feed back vae+vit) -> repeat,
    until a gen_text that does NOT end in <|vision_start|> (the assistant turn ended at <|im_end|>).
    Returns (chunk_text, chunk_imgs). The model self-imagines every per-step frame (self-rollout)."""
    parts: list = []
    imgs: list = []
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
                img = infr.gen_image(image_shape=GEN_IMG_SHAPE, gen_context=gen,
                                     cfg_text_precontext=cfg_text, cfg_img_precontext=cfg_img,
                                     cfg_text_scale=CFG_TEXT, cfg_img_scale=CFG_IMG,
                                     cfg_interval=(0.4, 1.0), num_timesteps=NUM_TIMESTEPS)
                if img.size != GEN_IMG_SHAPE:
                    img = img.resize(GEN_IMG_SHAPE, Image.Resampling.LANCZOS)
                imgs.append(img)
                ri = pil_img2rgb(img)
                gen = infr.update_context_image(ri, gen, vae=True, vit=True)   # feed self-imagined frame back vae+vit
                # BUGFIX(image-CFG): cfg_img stays text-only (do NOT feed the generated frame in)
                cfg_text = deepcopy(gen)
            else:
                break   # gen_text ended on <|im_end|> -> this assistant chunk is done
    return "".join(parts), imgs


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


def run_cot_episode(env, vlm, extra_state_key, ep: dict, args, rank: int) -> dict:
    env_id = ep["env_id"]
    if not G.is_sokoban_env_id(env_id) and args.profile != "sokoban_v8_res256":
        raise RuntimeError(f"CoT chunk runner is sokoban-only; got {env_id!r}")
    if _CHUNK_K <= 0 and not _IS_INF:
        raise RuntimeError("set BAGEL_CHUNK_K (chunk size, or 'inf') so the instruction cadence matches the trained model")
    seed = int(ep.get("seed") or ep.get("episode_seed") or 0)
    episode_seed = int(ep.get("episode_seed") or seed)
    max_steps = base._episode_max_steps(ep, args.max_steps)

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

    base.set_global_seed(seed)
    _reset_preserving_init_state(env, seed=episode_seed, init_state=ep.get("init_state"))
    infr = vlm.inferencer

    arr0 = base._render_like_to_array(env.render())
    frame0 = _jpeg_roundtrip(arr0, args.image_quality, args.image_format)   # q95 frame-0, as training
    # SHARED with the remaps -> train==eval: finite K uses the chunk cadence; inf uses the
    # CoT-inf instruction (identical across vanilla-kinf / branch / forward rows, byte-verified).
    if _IS_INF:
        instruction = cot_inf_instruction(env.get_prompt(), max_steps)
    else:
        instruction = cot_instruction(env.get_prompt(), max_steps, _CHUNK_K)
    # Diagnostic override: if BAGEL_OVERRIDE_INSTRUCTION_FILE is set to a valid file path, replace
    # the standard cot_instruction/cot_inf_instruction wrapping with the exact file contents. Used
    # to run eval with an alternate prompt (e.g. checking whether a train-eval prompt mismatch
    # explains a bad eval score for a ckpt trained with an inherited-source prompt).
    # Default behavior (env var unset / empty) is UNCHANGED — `instruction` stays as produced above.
    # Fails loudly on a set-but-missing path so we don't silently regress the standard prompt.
    _override = os.environ.get("BAGEL_OVERRIDE_INSTRUCTION_FILE", "").strip()
    if _override:
        assert os.path.isfile(_override), (
            f"BAGEL_OVERRIDE_INSTRUCTION_FILE={_override!r} is set but not a readable file. "
            f"Unset the env var to use the standard prompt, or point it at a valid file."
        )
        with open(_override, "r") as _f:
            instruction = _f.read()
        if rank == 0 and int(ep.get("global_index", 0)) < 2:
            print(f"[COT_DBG] OVERRIDE instruction from {_override} ({len(instruction)} chars)", flush=True)
    if _DBG and rank == 0 and int(ep.get("global_index", 0)) < 2:
        print(f"[COT_DBG] ep{ep.get('global_index')} K={_CHUNK_K} max_steps={max_steps}\n"
              f"  INSTRUCTION(pre-tokenizer)={instruction!r}", flush=True)

    # --- MULTI-TURN envfb rollout: per chunk -> generate <think> per-step imagine </think> + K actions,
    #     execute the K actions in the env, then append "Action executed." plus the real chunk-end
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

    # APPROACH A (native single-span CoT): ONE persistent gen_context carried across the whole episode.
    # Prefix the user turn (instruction + frame0) ONCE; each chunk is generated natively in-span by
    # gen_cot_chunk_span (token-by-token; switches to gen_image at <|vision_start|>, continues until
    # <|im_end|>; mutates `gen` IN PLACE so the cache carries). The GT env-feedback turn ("Env Feedback:" +
    # the REAL chunk-end frame, q95, loss-0 INPUT == the dataset's image_next -> train==eval) is appended
    # into the SAME cache. No per-chunk rebuild (the old _gen_one_chunk per-segment loop barreled past the
    # intermediate <|vision_start|> on v4 single-span ckpts and produced ~0 imagined frames).
    if _DECODER == "perseg":
        # PER-SEGMENT decode (shape-correct for per-segment-trained ckpts): rebuild the context
        # from a [str, PIL, ...] conversation each chunk via _gen_one_chunk, and append the
        # envfb GT re-grounding turn (label + REAL chunk-end frame) into the conversation —
        # the per-segment + GT-re-grounding path that previously did not exist.
        conversation: list = [instruction, Image.fromarray(np.asarray(frame0)).convert("RGB")]
        for ci in range(MAX_CHUNKS):
            if done or gstep >= max_steps:
                break
            chunk_text, chunk_imgs = _gen_one_chunk(infr, conversation, MAX_COT_ITERS)
            chunk_texts.append(chunk_text)
            imagined.extend(chunk_imgs)
            seg = chunk_text.split("</think>")[-1] if "</think>" in chunk_text else chunk_text
            actions = parse_action_chunk(seg)
            if _DBG and rank == 0 and int(ep.get("global_index", 0)) < 2:
                print(f"[COT_DBG] ep{ep.get('global_index')} chunk{ci} K={_CHUNK_K} n_imgs={len(chunk_imgs)} "
                      f"n_actions={len(actions)} [perseg]\n  CHUNK_TEXT[:320]={chunk_text[:320]!r}", flush=True)
            if not actions:
                break
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
                             "info": {k: final_info.get(k) for k in
                                      ("is_success", "is_solved", "boxes_on_target", "num_boxes", "steps", "action_mode")}})
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
            seg = chunk_text.split("</think>")[-1] if "</think>" in chunk_text else chunk_text
            actions = parse_action_chunk(seg)            # the chunk's committed K actions (after </think>)
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
                             "info": {k: final_info.get(k) for k in
                                      ("is_success", "is_solved", "boxes_on_target", "num_boxes", "steps", "action_mode")}})
                if G._is_stop_action(a) or terminated or truncated:
                    done = True
                    break
            if done:
                break
            if _IS_INF:
                # K=inf single-gen OPEN LOOP: one chunk covers the whole episode; NO feedback
                # turn is ever appended (matches the kinf/branch/forward training rows).
                break
            if _NO_REGROUND:
                # TRUE self-rollout: feed TEXT-ONLY feedback between chunks (NO image). The model never
                # re-sees the real observation; it must carry forward its own imagined frames.
                gen = infr.update_context_text(NO_REGROUND_LABEL, gen)
            else:
                # GT RE-GROUNDING feedback into the SAME cache: "Action executed." + the REAL chunk-end frame.
                gt_obs = _jpeg_roundtrip(np.asarray(frames[-1]), args.image_quality, args.image_format)
                gen = infr.update_context_text(ENV_FEEDBACK_LABEL, gen)
                gen = infr.update_context_image(pil_img2rgb(gt_obs), gen, vae=True, vit=True)

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

    feedback_mode = ("none_singlegen" if _IS_INF
                     else ("text_only_noreground" if _NO_REGROUND else "gt_frame"))
    result = {
        "env_id": env_id, "seed": seed, "episode_seed": episode_seed,
        "global_index": int(ep["global_index"]), "rank": rank, "profile": args.profile,
        "study": "action_chunking_cot", "runner": "cot_chunk_envfb", "decoder": _DECODER,
        "chunk_eval_mode": f"cot_multiturn_envfb_{feedback_mode}",
        "feedback_mode": feedback_mode,
        "feedback_label": (None if _IS_INF else (NO_REGROUND_LABEL if _NO_REGROUND else ENV_FEEDBACK_LABEL)),
        "no_reground": bool(_NO_REGROUND and not _IS_INF),
        "cot_frame_mode": os.environ.get("BAGEL_COT_FRAME_MODE", "exact"),
        "cfg_text_scale": CFG_TEXT, "cfg_img_scale": CFG_IMG,
        "chunk_k": ("inf" if _IS_INF else _CHUNK_K), "is_inf": bool(_IS_INF),
        "reset_source": "init_state",
        "version": "v3_cot_chunk_envfb_eval_ddp_multiturn", "instruction_max_steps": max_steps,
        "n_chunks": len(chunk_texts), "n_imagined": len(imagined), "n_executed": len(hist),
        "history": hist,
        "stats": {"step": len(hist), "reward": cum_reward,
                  "terminated": bool(terminated), "truncated": bool(truncated),
                  "success": bool(success), "final_action": final_action,
                  "n_chunks": len(chunk_texts), "n_imagined": len(imagined)},
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(base._json_safe(result), f, indent=2)
    return {"success": bool(success)}


if __name__ == "__main__":
    G.run_episode = run_cot_episode
    G.main()
