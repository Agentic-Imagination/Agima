#!/usr/bin/env python3
"""K=5 Table 1 evaluation with only imagined-frame pixels changed.

All conditions retain the published per-segment text decoder and 50-step diffusion.
Generated images are decoded and JPEG-round-tripped in every condition; replacement
conditions discard those pixels and feed the selected frame through the same VAE+ViT path.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import evaluation.runtime as G
import evaluation.base as base
from data.data_utils import pil_img2rgb
from evaluation.environments import _reset_preserving_init_state
from evaluation.actions import parse_action_chunk, _jpeg_roundtrip
from scripts.data.remap_pacman_cot_chunk_k import cot_instruction as pacman_instruction
from scripts.data.remap_pacman_cot_chunk_k_envfb import ENV_FEEDBACK_LABEL as PACMAN_FEEDBACK
from scripts.data.remap_sokoban_cot_chunk_k import cot_instruction as sokoban_instruction
from scripts.data.remap_sokoban_cot_chunk_k_envfb import ENV_FEEDBACK_LABEL as SOKOBAN_FEEDBACK
from experiments.image_ablation.oracle import OracleReplica, invalid_action, physical_state, render_rgb


CONDITION = os.environ.get("AGIMA_CONDITION", "").lower()
if CONDITION not in {"original", "blank", "stale", "oracle"}:
    raise RuntimeError("Set AGIMA_CONDITION=original|blank|stale|oracle")
if os.environ.get("BAGEL_DECODER", "perseg") != "perseg":
    raise RuntimeError("Table 1 K=5 checkpoint requires BAGEL_DECODER=perseg")
if os.environ.get("BAGEL_CHUNK_K", "5") != "5":
    raise RuntimeError("This ablation is fixed to the K=5 checkpoint and protocol")

K = 5
IM_END_ID = 151645
VISION_START = "<|vision_start|>"
IMAGE_SIZE = (256, 256)
MAX_COT_ITERS = 48
MAX_TEXT_TOKENS = 600
IMAGE_TIMESTEPS = 50
CFG_TEXT = 1.0
CFG_IMG = 1.0


def _history_items(chunk_text: str, images: list[Image.Image]) -> list:
    fragments = chunk_text.split(VISION_START)
    items: list = []
    for i, fragment in enumerate(fragments):
        text = fragment + (VISION_START if i < len(fragments) - 1 else "")
        if text:
            items.append(text)
        if i < len(images):
            items.append(images[i])
    return items


def _tensor_audit(infr, generated: Image.Image, used: Image.Image) -> dict:
    """Record the exact pre-encoder tensor path used by update_context_image."""
    def describe(image: Image.Image) -> dict:
        rgb = pil_img2rgb(image)
        arr = np.asarray(rgb)
        vae = infr.vae_transform(rgb)
        vit = infr.vit_transform(rgb)
        return {
            "pixel_shape": list(arr.shape), "pixel_dtype": str(arr.dtype),
            "vae_transform_shape": list(vae.shape), "vae_transform_dtype": str(vae.dtype),
            "vit_transform_shape": list(vit.shape), "vit_transform_dtype": str(vit.dtype),
            "encode_modules": ["Bagel.prepare_vae_images -> forward_cache_update_vae",
                               "Bagel.prepare_vit_images -> forward_cache_update_vit"],
        }
    generated_path = describe(generated)
    used_path = describe(used)
    assert generated_path == used_path, "Generated and replacement encoding paths differ"
    return {"generated": generated_path, "context": used_path, "identical": True}


def _mse(a: Image.Image, b: Image.Image) -> dict:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return {"pixel_mse_0_255": float(np.mean((aa - bb) ** 2)),
            "pixel_mse_0_1": float(np.mean(((aa - bb) / 255.0) ** 2))}


@torch.inference_mode()
def _generate_chunk(infr, conversation: list, real_env, stale_obs: Image.Image,
                    image_quality: int, image_format: str) -> dict:
    parts: list[str] = []
    generated_images: list[Image.Image] = []
    used_images: list[Image.Image] = []
    oracle_images: list[Image.Image] = []
    imagined_actions: list[str] = []
    mse: list[dict] = []
    audit = None
    imagined_parse_failure = False
    token_limit_hit = False
    loop_limit_hit = False
    gray = _jpeg_roundtrip(Image.new("RGB", IMAGE_SIZE, (128, 128, 128)),
                           image_quality, image_format)
    oracle = OracleReplica(real_env) if CONDITION in {"original", "oracle"} else None
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen = infr.init_gen_context()
            cfg_text = deepcopy(gen)
            cfg_img = deepcopy(gen)
            for item in conversation:
                if isinstance(item, str):
                    cfg_text = deepcopy(gen)
                    gen = infr.update_context_text(item, gen)
                    cfg_img = infr.update_context_text(item, cfg_img)
                else:
                    image = pil_img2rgb(item)
                    gen = infr.update_context_image(image, gen, vae=True, vit=True)
                    cfg_text = deepcopy(gen)

            for iteration in range(MAX_COT_ITERS):
                text_out = infr.gen_text(gen, max_length=MAX_TEXT_TOKENS, do_sample=False,
                                         temperature=1.0, end_token_id=[IM_END_ID])
                parts.append(text_out)
                token_limit_hit |= len(infr.tokenizer.encode(
                    text_out, add_special_tokens=False)) >= MAX_TEXT_TOKENS
                cfg_text = deepcopy(gen)
                gen = infr.update_context_text(text_out, gen)
                cfg_img = infr.update_context_text(text_out, cfg_img)
                if not text_out.rstrip().endswith(VISION_START):
                    break

                # The trained format has one proposed action directly before each image.
                # The unchanged Table 1 parser is also used for these proposals.
                proposed = parse_action_chunk(text_out.removesuffix(VISION_START))
                if len(proposed) != 1:
                    imagined_parse_failure = True
                imagined_actions.extend(proposed)

                generated = infr.gen_image(
                    image_shape=IMAGE_SIZE, gen_context=gen,
                    cfg_text_precontext=cfg_text, cfg_img_precontext=cfg_img,
                    cfg_text_scale=CFG_TEXT, cfg_img_scale=CFG_IMG,
                    cfg_interval=(0.4, 1.0), num_timesteps=IMAGE_TIMESTEPS)
                if generated.size != IMAGE_SIZE:
                    generated = generated.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                generated = _jpeg_roundtrip(generated, image_quality, image_format)
                generated_images.append(generated)

                if oracle is not None:
                    oracle_frame = oracle.last_frame
                    for action in proposed:
                        oracle_frame = oracle.step(action)
                    oracle_image = _jpeg_roundtrip(oracle_frame, image_quality, image_format)
                    oracle_images.append(oracle_image)

                if CONDITION == "original":
                    used = generated
                    mse.append(_mse(generated, oracle_image))
                elif CONDITION == "blank":
                    used = gray
                elif CONDITION == "stale":
                    used = stale_obs
                else:
                    used = oracle_image
                used_images.append(used.copy())
                if audit is None:
                    audit = _tensor_audit(infr, generated, used)
                gen = infr.update_context_image(pil_img2rgb(used), gen, vae=True, vit=True)
                cfg_text = deepcopy(gen)
            else:
                loop_limit_hit = True
    finally:
        if oracle is not None:
            oracle.close()
    return {
        "text": "".join(parts), "generated": generated_images, "used": used_images,
        "oracle": oracle_images, "imagined_actions": imagined_actions, "mse": mse,
        "audit": audit, "imagined_parse_failure": imagined_parse_failure,
        "token_limit_hit": token_limit_hit or loop_limit_hit,
    }


def _save_trace(outdir: Path, conversation_images: list[tuple[str, Image.Image]],
                chunk_texts: list[str], chunk_records: list[dict],
                frames: list[Image.Image], audits: list[dict]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cot_full_text.txt").write_text("\n\n----- CHUNK -----\n\n".join(chunk_texts))
    (outdir / "chunks.json").write_text(json.dumps(chunk_records, indent=2))
    (outdir / "image_encoding_audit.json").write_text(json.dumps(audits, indent=2))
    context_dir = outdir / "context_frames"
    context_dir.mkdir(exist_ok=True)
    sequence = []
    for i, (source, image) in enumerate(conversation_images):
        name = f"{i:04d}_{source}.png"
        image.save(context_dir / name)
        sequence.append({"index": i, "source": source, "file": name})
    (outdir / "context_sequence.json").write_text(json.dumps(sequence, indent=2))
    render_dir = outdir / "renders"
    render_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(render_dir / f"step_{i:04d}.png")


def run_episode(env, vlm, extra_state_key, ep: dict, args, rank: int) -> dict:
    task = "sokoban" if ep["env_id"].startswith("sokoban/") else "pacman"
    required_profile = "sokoban_v8_res256" if task == "sokoban" else "pacman_res256"
    if args.profile != required_profile:
        raise RuntimeError(f"Expected {required_profile}, got {args.profile}")
    if args.image_quality != 95 or args.image_format.upper() != "JPEG":
        raise RuntimeError("Table 1 uses 256x256 JPEG quality 95")
    seed = int(ep.get("seed") or ep.get("episode_seed") or 0)
    episode_seed = int(ep.get("episode_seed") or seed)
    max_steps = base._episode_max_steps(ep, args.max_steps)
    index = int(ep["global_index"])
    outdir = (Path(args.outdir) / base._env_slug(ep["env_id"]) / args.profile /
              "cot_selfrollout" / f"idx{index:07d}_seed{seed}_episode{episode_seed}")
    stats_path = outdir / "episode_stats.json"
    if stats_path.exists() and not args.overwrite:
        try:
            prior = json.loads(stats_path.read_text())
            if prior.get("stats", {}).get("success") is not None:
                return {"skipped": True, "success": bool(prior["stats"]["success"])}
        except (OSError, ValueError):
            pass

    base.set_global_seed(seed)
    _reset_preserving_init_state(env, seed=episode_seed, init_state=ep.get("init_state"))
    raw0 = render_rgb(env)
    frame0 = _jpeg_roundtrip(raw0, args.image_quality, args.image_format)
    instruction = (sokoban_instruction if task == "sokoban" else pacman_instruction)(
        env.get_prompt(), max_steps, K)
    feedback = SOKOBAN_FEEDBACK if task == "sokoban" else PACMAN_FEEDBACK
    conversation: list = [instruction, frame0]
    conversation_images = [("real_initial", frame0)]
    frames = [Image.fromarray(raw0)]
    chunk_texts: list[str] = []
    chunk_records: list[dict] = []
    audits: list[dict] = []
    history: list[dict] = []
    all_imagined_actions: list[str] = []
    all_mse: list[dict] = []
    num_invalid = 0
    commit_parse_failure = False
    imagined_parse_failure = False
    token_limit_hit = False
    final_info = {}
    final_action = None
    terminated = env_truncated = done = False
    total_reward = 0.0

    for chunk_idx in range(max_steps + 2):
        if done or len(history) >= max_steps:
            break
        stale_obs = _jpeg_roundtrip(render_rgb(env), args.image_quality, args.image_format)
        generated = _generate_chunk(vlm.inferencer, conversation, env, stale_obs,
                                    args.image_quality, args.image_format)
        text = generated["text"]
        chunk_texts.append(text)
        all_imagined_actions.extend(generated["imagined_actions"])
        all_mse.extend(generated["mse"])
        imagined_parse_failure |= generated["imagined_parse_failure"]
        token_limit_hit |= generated["token_limit_hit"]
        if generated["audit"] is not None:
            audits.append(generated["audit"])
        for image in generated["used"]:
            conversation_images.append((CONDITION, image))

        committed_text = text.split("</think>")[-1] if "</think>" in text else text
        actions = parse_action_chunk(committed_text)
        if not actions:
            commit_parse_failure = True
        record = {
            "chunk": chunk_idx, "text": text, "imagined_actions": generated["imagined_actions"],
            "committed_actions": actions, "generated_count": len(generated["generated"]),
            "used_count": len(generated["used"]), "executed_actions": [],
            "pixel_mse": generated["mse"],
        }
        chunk_records.append(record)
        if index < 10:
            image_dir = outdir / f"chunk_{chunk_idx:03d}"
            image_dir.mkdir(parents=True, exist_ok=True)
            for label in ("generated", "used", "oracle"):
                for j, image in enumerate(generated[label]):
                    image.save(image_dir / f"{label}_{j:03d}.png")

        for action in actions:
            if len(history) >= max_steps:
                env_truncated = True
                done = True
                break
            before = physical_state(env, task)
            _, reward, terminated, env_truncated, info = env.step(action)
            total_reward += float(reward or 0.0)
            final_info = info or {}
            final_action = action
            after = physical_state(env, task)
            invalid = invalid_action(task, before, after, final_info, G._is_stop_action(action))
            num_invalid += int(invalid)
            frame = Image.fromarray(render_rgb(env))
            frames.append(frame)
            step = len(history)
            history.append({"step": step, "chunk": chunk_idx, "action": action,
                            "reward": float(reward or 0.0), "terminated": bool(terminated),
                            "truncated": bool(env_truncated), "invalid": invalid,
                            "info": base._json_safe(final_info)})
            record["executed_actions"].append(action)
            if G._is_stop_action(action) or terminated or env_truncated:
                done = True
                break
        if done or not actions:
            break
        if len(history) < max_steps:
            conversation.extend(_history_items(text, generated["used"]))
            gt_obs = _jpeg_roundtrip(render_rgb(env), args.image_quality, args.image_format)
            conversation.extend([feedback, gt_obs])
            conversation_images.append(("real_boundary", gt_obs))

    success = (bool(terminated) and not bool(env_truncated) and
               G._is_stop_action(final_action) and bool(final_info.get("is_success", False)))
    if success:
        reason = "solved"
    elif task == "pacman" and final_info.get("alive") is False:
        reason = "caught_by_ghost"
    elif token_limit_hit:
        reason = "truncated"
    elif commit_parse_failure:
        reason = "parse_failure"
    elif env_truncated or len(history) >= max_steps:
        reason = "budget_exhausted"
    else:
        reason = "other"
    executed = [item["action"] for item in history]
    result = {
        "episode_id": f"{task}:{index:07d}:{episode_seed}", "task": task,
        "condition": CONDITION.capitalize() if CONDITION != "oracle" else "Oracle GT",
        "success": bool(success), "termination_reason": reason,
        "num_executed_actions": len(executed), "num_chunks": len(chunk_texts),
        "parse_failure": bool(commit_parse_failure or imagined_parse_failure),
        "num_invalid_actions": num_invalid, "stop_emitted": G._is_stop_action(final_action),
        "truncated": bool(token_limit_hit),
        "committed_matches_imagined": executed == all_imagined_actions,
        "global_index": index, "seed": seed, "episode_seed": episode_seed,
        "n_imagined": sum(record["used_count"] for record in chunk_records),
        "image_generation_skipped": False, "image_timesteps": IMAGE_TIMESTEPS,
        "image_quality": args.image_quality, "image_format": args.image_format,
        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "pixel_mse": all_mse,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    if index < 10:
        _save_trace(outdir, conversation_images, chunk_texts, chunk_records, frames, audits)
    stats = {
        "env_id": ep["env_id"], "seed": seed, "episode_seed": episode_seed,
        "global_index": index, "rank": rank, "profile": args.profile,
        "runner": "agima_image_ablation", "decoder": "perseg", "chunk_k": K,
        "instruction_max_steps": max_steps, "n_chunks": len(chunk_texts),
        "n_imagined": result["n_imagined"], "n_executed": len(history),
        "history": history, "result": result,
        "stats": {"success": bool(success), "terminated": bool(terminated),
                  "truncated": bool(env_truncated), "final_action": final_action,
                  "reward": total_reward},
    }
    stats_path.write_text(json.dumps(base._json_safe(stats), indent=2))
    return {"success": bool(success)}


if __name__ == "__main__":
    G.run_episode = run_episode
    G.main()
