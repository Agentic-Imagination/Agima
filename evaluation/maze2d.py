#!/usr/bin/env python3
# Modified for Agima: package imports and portable resource configuration.
# =============================================================================
# Maze2D CoT K-action-chunk CLOSED-LOOP eval runner (envfb GT re-grounding) — MULTI-TURN.
#
# Maze2D analog of evaluation/pusht.py / the sokoban envfb
# chunk runner (evaluation/sokoban.py). Same DDP scaffolding, same
# gen_cot_chunk_span native single-span generation, same GT re-grounding feedback protocol.
# Scores the 20260707_perseg maze2d chunk ckpts (cot + noncot, K=1/3/5/10/inf), easy->hard.
#
# Matches the maze2d_{easy}_native256_{cot,noncot}_chunk_k{K}_20260707_perseg training rows
# (scripts/data/remap_maze2d_cot_chunk_k_envfb.py; the non-CoT arm imports the SAME
# cot_instruction). Per chunk the model emits one assistant turn
#   <think> Let me imagine ... a<vis>imagine ... (K per-step frames) </think>  {the chunk's K actions}
# then the ENVIRONMENT replies with "Action executed." (ENV_FEEDBACK_LABEL) plus the REAL
# chunk-end frame (q95, loss-0 INPUT), repeat until ('stop','stop') or max_steps (40). Per episode:
#   gen one chunk (gen_cot_chunk_span, in-span; text -> imagined frame -> ... until <|im_end|>)
#   -> parse the chunk's K actions (after </think>; tolerant to nonCoT chunks with no </think>)
#   -> execute them in the env -> append ENV_FEEDBACK_LABEL + GT chunk-end frame -> regenerate.
#
# Maze2D deltas vs the pusht runner (each grounded in the base maze2d eval,
# evaluation.runtime.run_maze2d_episode and the original all-step CoT protocol):
#   * profile maze2d_native256, env built by G.make_profiled_env (maze2d env_kwargs from
#     metadata + validate_maze2d_contract: stop-required prompt);
#   * reset from init_state (maze2d is init_state-only; required non-None), like the plain runner;
#   * observation frames are native-256 env renders JPEG-roundtripped at q95 (maze2d contract;
#     launcher passes --image-quality 95) then the VAE 256/256 / ViT 224/224 transforms;
#   * actions are DISCRETE strings "('move', 0)".."('move', 3)" / "('stop', 'stop')" — the remap
#     serializes str(vlm_output); parse_action_chunk's balanced-paren + ast scan recovers the exact
#     substring and env.step ast.literal_eval's it, so the training serialization round-trips
#     unchanged. ('move', int)'s bare int second element is fine; _quote_first_identifier only
#     touches a bare leading identifier and 'move'/'stop' are already quoted in the emitted text;
#   * SUCCESS = maze2d stop-required contract: terminated AND NOT truncated AND final action
#     ('stop','stop') AND info.distance == 0.0 AND cumulative reward > 0 — byte-identical predicate
#     to G.run_maze2d_episode.
#   * TRAIN==EVAL: the instruction (rules + total-budget sentence + ONE env-feedback cadence line +
#     "Starting maze:") and ENV_FEEDBACK_LABEL are IMPORTED from the maze2d remap (not re-hardcoded);
#     the chunk instruction has NO per-step "This is step N ... {40-N} more steps" counters (the
#     chunk remap DROPPED them). rules come from the test row's history[0].prompt (== the source the
#     remap builds from) with an env.get_prompt() fallback. Verified byte-identical to every K dataset.
#
# Easy->hard: the easy ckpt is evaluated on the easy OR hard test split (EVAL_DIFFICULTY in the
# launcher); the runner is difficulty-agnostic (max_steps=40 for both, per metadata).
#
# K='inf' (BAGEL_CHUNK_K=inf): single-gen, OPEN-LOOP — one assistant chunk covers the whole
# episode and there are ZERO feedback turns (the model was trained with no feedback for inf),
# so we generate exactly one chunk, execute open-loop, and never re-ground.
#
# Generation: CFG (image-gen) defaults to 1.0/1.0 = OFF (cond_dropout=0 => uncond branch
# untrained); override via BAGEL_COT_CFG_TEXT/IMG. BAGEL_CHUNK_K must match the trained K.
# BAGEL_NO_REGROUND=1 -> text-only feedback between chunks (true self-rollout ablation; ignored
# for K=inf which is already feedback-free). nonCoT ckpts emit zero imagined frames automatically.
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
# SHARED maze2d instruction builder + GT env-feedback label + max-steps resolver — the SAME module
# the CoT and non-CoT remaps import -> train==eval byte-identical (verified across K=1/3/5/10/inf).
from scripts.data.remap_maze2d_cot_chunk_k_envfb import (
    cot_instruction,
    ENV_FEEDBACK_LABEL,
    _max_steps as _remap_max_steps,
)


def _parse_chunk_k():
    """BAGEL_CHUNK_K is '1'/'3'/'5'/'10' or 'inf' (matches the trained ckpt's cadence). Returns
    (is_inf, k_int_or_-1, k_for_instruction) where k_for_instruction is int or the string 'inf'."""
    raw = os.environ.get("BAGEL_CHUNK_K", "").strip().lower()
    if raw == "inf":
        return True, -1, "inf"
    return False, int(raw or "0"), int(raw or "0")


_IS_INF, _CHUNK_K, _K_FOR_INSTR = _parse_chunk_k()
_DBG = os.environ.get("BAGEL_DEBUG_PROMPTS", "0") == "1"
# BAGEL_NO_REGROUND=1 -> TRUE self-rollout: between chunks feed TEXT-ONLY feedback (no real frame),
# so the model never re-sees the ground-truth observation and must rely on its own imagined frames.
# Default (0) keeps the GT re-grounding behavior (ENV_FEEDBACK_LABEL + the real chunk-end frame).
# (No effect for K=inf, which is trained feedback-free and runs open-loop regardless.)
_NO_REGROUND = os.environ.get("BAGEL_NO_REGROUND", "0") == "1"
# BAGEL_DECODER: 'span' (default; persistent-cache gen_cot_chunk_span — single-span-shaped) or
# 'perseg' (rebuild-per-chunk _gen_one_chunk — matches PER-SEGMENT-trained ckpts). The sokoban
# K-anomaly investigation proved the shape matters: perseg ckpts under span decode collapse at
# K=1/3 (66%/19% easy) but score 90%/73% under the per-segment decoder. 'perseg' + default
# GT re-grounding = the previously-missing per-segment+envfb decode path. Composes with K=inf
# (single rebuilt chunk, no feedback turn).
_DECODER = os.environ.get("BAGEL_DECODER", "span").strip().lower()
if os.environ.get("RANK", "0") == "0":
    print(f"[DECODER-CHECK] BAGEL_DECODER={_DECODER}", flush=True)
# Text-only between-chunk feedback when re-grounding is disabled. Reuses the trained env-feedback
# label so the text channel still matches training; only the image is withheld.
NO_REGROUND_LABEL = os.environ.get("BAGEL_NO_REGROUND_LABEL", ENV_FEEDBACK_LABEL)
if os.environ.get("RANK", "0") == "0":
    _mode = ("SINGLE-GEN open-loop (K=inf, no feedback)" if _IS_INF
             else ("TEXT-ONLY feedback, true self-rollout" if _NO_REGROUND else "GT-frame re-grounding"))
    print(f"[REGROUND-CHECK] K={_K_FOR_INSTR} BAGEL_NO_REGROUND={int(_NO_REGROUND)}  ({_mode})", flush=True)

# tokens / shapes (from the tested allstep CoT path). Maze2D VAE transform is 256/256, so imagined
# frames are generated at 256x256 (training MSE targets = native256 q95 frames via the same transform).
IM_END_ID = 151645
VISION_START = "<|vision_start|>"
GEN_IMG_SHAPE = (256, 256)
MAX_COT_ITERS = int(os.environ.get("BAGEL_COT_MAX_ITERS", "48"))   # max (text,image) pairs; generous
MAX_TEXT_TOKENS = int(os.environ.get("BAGEL_COT_MAX_TEXT_TOKENS", "600"))
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


def _maze2d_rules(ep: dict, env) -> str:
    """Step-0 rules for the instruction. Prefer the test row's history[0].prompt (what the remap
    builds `rules` from -> byte-identical); fall back to the env's own prompt. cot_instruction's
    _rules_prefix cuts at 'This is step 1.' (absent at step 0) so no step counter enters."""
    prompts = ep.get("history_prompts") or []
    if prompts and isinstance(prompts[0], str) and prompts[0].strip():
        return prompts[0]
    inner = getattr(env, "unwrapped", env)
    return inner.get_prompt()


def run_maze2d_cot_episode(env, vlm, extra_state_key, ep: dict, args, rank: int) -> dict:
    env_id = ep["env_id"]
    if not G.is_maze2d_env_id(env_id) and args.profile != "maze2d_native256":
        raise RuntimeError(f"Maze2D CoT chunk runner is maze2d-only; got {env_id!r}")
    if not _IS_INF and _CHUNK_K <= 0:
        raise RuntimeError("set BAGEL_CHUNK_K (1/3/5/10 or 'inf') so the instruction cadence matches the trained model")
    # seed / global_index resolved the SAME way as G.run_maze2d_episode (maze2d rows carry no
    # top-level seed; it lives in alignment_index/metadata).
    seed = int(
        ep.get("seed")
        or ep.get("episode_seed")
        or (ep.get("alignment_index") or {}).get("seed")
        or (ep.get("metadata") or {}).get("seed")
        or 0
    )
    episode_seed = int(ep.get("episode_seed") or seed)
    global_index = int((ep.get("alignment_index") or {}).get("global_index", ep.get("global_index", 0)))
    max_steps = _remap_max_steps(ep)   # metadata.max_steps == 40 (easy and hard); == the instruction budget

    out_episode_dir = (Path(args.outdir) / base._env_slug(env_id) / args.profile / "cot_selfrollout"
                       / f"idx{global_index:07d}_seed{seed}_episode{episode_seed}")
    json_file = out_episode_dir / "episode_stats.json"
    if not args.overwrite and json_file.exists():
        try:
            st = (json.loads(json_file.read_text()).get("stats") or {})
            if st.get("success") is not None:
                return {"skipped": True, "success": bool(st.get("success"))}
        except Exception:
            pass

    # Maze2D resets from the recorded physics state (init_state), not the seed (plain runner contract).
    init_state = ep.get("init_state")
    if init_state is None:
        raise RuntimeError(f"{env_id} episode {global_index} has no init_state for Maze2D eval.")
    base.set_global_seed(seed)
    _reset_preserving_init_state(env, seed=episode_seed, init_state=init_state)
    infr = vlm.inferencer

    arr0 = base._render_like_to_array(env.render())
    frame0 = _jpeg_roundtrip(arr0, args.image_quality, args.image_format)   # q95 frame-0, as training
    instruction = cot_instruction(_maze2d_rules(ep, env), max_steps, _K_FOR_INSTR)   # SHARED w/ remap -> train==eval
    if _DBG and rank == 0 and global_index < 2:
        print(f"[COT_DBG] ep{global_index} K={_K_FOR_INSTR} max_steps={max_steps}\n"
              f"  INSTRUCTION(pre-tokenizer)={instruction!r}", flush=True)

    # --- MULTI-TURN envfb rollout: per chunk -> generate <think> per-step imagine </think> + K actions,
    #     execute the K actions in the env, then append ENV_FEEDBACK_LABEL plus the real chunk-end
    #     frame and continue, until ('stop','stop') or max_steps. K=inf -> ONE chunk, no feedback.
    frames = [Image.fromarray(np.asarray(arr0)).convert("RGB")]
    hist: list = []
    imagined: list = []
    chunk_texts: list = []
    terminated = truncated = done = False
    final_info: dict = {}
    final_action = None
    cum_reward = 0.0
    gstep = 0
    MAX_CHUNKS = 1 if _IS_INF else (max_steps + 2)

    # Native single-span CoT: ONE persistent gen_context carried across the whole episode.
    # Prefix the user turn (instruction + frame0) ONCE; each chunk is generated natively in-span by
    # gen_cot_chunk_span (token-by-token; switches to gen_image at <|vision_start|>, continues until
    # <|im_end|>; mutates `gen` IN PLACE so the cache carries). The GT env-feedback turn
    # (ENV_FEEDBACK_LABEL + the REAL chunk-end frame, q95, loss-0 INPUT == the dataset's image_next
    # -> train==eval) is appended into the SAME cache.
    if _DECODER == "perseg":
        # PER-SEGMENT decode (shape-correct for per-segment-trained ckpts): rebuild the context
        # from a [str, PIL, ...] conversation each chunk via _gen_one_chunk, and append the
        # envfb GT re-grounding turn (label + REAL chunk-end frame) into the conversation —
        # the per-segment + GT-re-grounding path that previously did not exist.
        # K=inf composes: MAX_CHUNKS==1 + the break below -> ONE rebuilt chunk, no feedback turn.
        conversation: list = [instruction, Image.fromarray(np.asarray(frame0)).convert("RGB")]
        for ci in range(MAX_CHUNKS):
            if done or gstep >= max_steps:
                break
            chunk_text, chunk_imgs = _gen_one_chunk(infr, conversation, MAX_COT_ITERS)
            chunk_texts.append(chunk_text)
            imagined.extend(chunk_imgs)
            seg = chunk_text.split("</think>")[-1] if "</think>" in chunk_text else chunk_text
            actions = parse_action_chunk(seg)
            if _DBG and rank == 0 and global_index < 2:
                print(f"[COT_DBG] ep{global_index} chunk{ci} K={_K_FOR_INSTR} n_imgs={len(chunk_imgs)} "
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
                             "info": {k: base._json_safe(final_info.get(k)) for k in
                                      ("distance", "is_success", "success")}})
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
            # committed actions come after </think>; nonCoT ckpts emit no </think> (and zero
            # imagined frames) -> parse the whole chunk text. parse_action_chunk stops at the first
            # ('stop','stop'); ('move', int) discrete actions parse cleanly (str-first-element).
            seg = chunk_text.split("</think>")[-1] if "</think>" in chunk_text else chunk_text
            actions = parse_action_chunk(seg)
            if _DBG and rank == 0 and global_index < 2:
                print(f"[COT_DBG] ep{global_index} chunk{ci} K={_K_FOR_INSTR} n_imgs={len(chunk_imgs)} "
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
                                      ("distance", "is_success", "success")}})
                if G._is_stop_action(a) or terminated or truncated:
                    done = True
                    break
            if done:
                break
            if _IS_INF:
                break                                    # K=inf: single-gen open-loop, no feedback turn
            if _NO_REGROUND:
                # TRUE self-rollout: feed TEXT-ONLY feedback between chunks (NO image). The model never
                # re-sees the real observation; it must carry forward its own imagined frames.
                gen = infr.update_context_text(NO_REGROUND_LABEL, gen)
            else:
                # GT RE-GROUNDING feedback into the SAME cache: ENV_FEEDBACK_LABEL + the REAL
                # chunk-end frame (q95 round-trip == the training image_next JPEG surface).
                gt_obs = _jpeg_roundtrip(np.asarray(frames[-1]), args.image_quality, args.image_format)
                gen = infr.update_context_text(ENV_FEEDBACK_LABEL, gen)
                gen = infr.update_context_image(pil_img2rgb(gt_obs), gen, vae=True, vit=True)

    # maze2d stop-required success, byte-identical predicate to G.run_maze2d_episode:
    # terminated AND NOT truncated AND final action ('stop','stop') AND distance==0 AND reward>0.
    success = (bool(terminated) and not bool(truncated)
               and G._is_stop_action(final_action)
               and float(final_info.get("distance", 1.0)) == 0.0
               and cum_reward > 0.0)

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
        "global_index": global_index, "rank": rank, "profile": args.profile,
        "hash": ep.get("hash"), "alignment_index": ep.get("alignment_index"),
        "study": "maze2d_action_chunking_cot", "runner": "maze2d_cot_chunk_envfb", "decoder": _DECODER,
        "chunk_eval_mode": f"cot_multiturn_envfb_{feedback_mode}",
        "feedback_mode": feedback_mode,
        "feedback_label": (None if _IS_INF else (NO_REGROUND_LABEL if _NO_REGROUND else ENV_FEEDBACK_LABEL)),
        "no_reground": bool(_NO_REGROUND and not _IS_INF),
        "cot_frame_mode": os.environ.get("BAGEL_COT_FRAME_MODE", "exact"),
        "cfg_text_scale": CFG_TEXT, "cfg_img_scale": CFG_IMG,
        "chunk_k": _K_FOR_INSTR, "is_inf": bool(_IS_INF),
        "reset_source": "init_state",
        "version": "v1_maze2d_cot_chunk_envfb_eval_ddp_multiturn", "instruction_max_steps": max_steps,
        "n_chunks": len(chunk_texts), "n_imagined": len(imagined), "n_executed": len(hist),
        "history": hist,
        "stats": {"step": len(hist), "reward": cum_reward,
                  "terminated": bool(terminated), "truncated": bool(truncated),
                  "success": bool(success), "final_action": final_action,
                  "distance": float(final_info.get("distance", 1.0)) if final_info else None,
                  "n_chunks": len(chunk_texts), "n_imagined": len(imagined)},
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(base._json_safe(result), f, indent=2)
    return {"success": bool(success)}


if __name__ == "__main__":
    # Swap ONLY the per-episode logic; everything else (args, model load, maze2d env build +
    # contract validation, idx%world_size sharding, empty-rank summaries) is the proven
    # gen_eval_ddp scaffolding.
    G.run_episode = run_maze2d_cot_episode
    G.main()
