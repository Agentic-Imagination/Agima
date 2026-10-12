#!/usr/bin/env python3
"""Intervene on every K=5 imagined frame in the original Maze/PushT span evaluator.

The production episode runners still own prompts, action parsing, environment stepping,
feedback, and success criteria. This wrapper changes only the pixels passed to the
span decoder's frame callback and records the resulting paired evaluation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import evaluation.runtime as G
import evaluation.base as base
from data.data_utils import pil_img2rgb
from evaluation.environments import _reset_preserving_init_state
from evaluation.actions import parse_action_chunk, _jpeg_roundtrip
from experiments.image_ablation.oracle import OracleReplica, render_rgb


TASK = os.environ.get("AGIMA_TASK", "").lower()
CONDITION = os.environ.get("AGIMA_CONDITION", "").lower()
if TASK not in {"maze", "pusht"} or CONDITION not in {"original", "blank", "stale", "oracle"}:
    raise RuntimeError("Set AGIMA_TASK=maze|pusht and AGIMA_CONDITION=original|blank|stale|oracle")
EXPECTED_DECODER = "span" if TASK == "maze" else "perseg"
if os.environ.get("BAGEL_DECODER") != EXPECTED_DECODER or os.environ.get("BAGEL_CHUNK_K") != "5":
    raise RuntimeError(f"Table 1 {TASK} K=5 requires the {EXPECTED_DECODER} decoder")

if TASK == "maze":
    import evaluation.maze2d as production
else:
    import evaluation.pusht as production

PUSHT_FRAME_INPUT = os.environ.get("AGIMA_PUSHT_FRAME_INPUT", "96")
if TASK == "pusht" and PUSHT_FRAME_INPUT not in {"96", "256"}:
    raise RuntimeError("AGIMA_PUSHT_FRAME_INPUT must be 96 or 256")
NATIVE_SIZE = (256, 256) if TASK == "maze" or PUSHT_FRAME_INPUT == "256" else (96, 96)
IMAGINED_CODEC = os.environ.get("AGIMA_IMAGINED_CODEC", "jpeg")
if IMAGINED_CODEC not in {"jpeg", "raw"}:
    raise RuntimeError("AGIMA_IMAGINED_CODEC must be jpeg or raw")
QUALITY = 95 if TASK == "maze" else 90


def _codec(image: Image.Image | np.ndarray, args) -> Image.Image:
    rgb = pil_img2rgb(image if isinstance(image, Image.Image)
                      else Image.fromarray(np.asarray(image, dtype=np.uint8)))
    if rgb.size != NATIVE_SIZE:
        rgb = rgb.resize(NATIVE_SIZE, Image.Resampling.BICUBIC if TASK == "pusht" else Image.Resampling.LANCZOS)
    return (_jpeg_roundtrip(rgb, args.image_quality, args.image_format).convert("RGB")
            if IMAGINED_CODEC == "jpeg" else rgb)


def _real_to_imagined(real_observation: Image.Image, args) -> Image.Image:
    if real_observation.size == NATIVE_SIZE and IMAGINED_CODEC == "jpeg":
        return real_observation.copy()
    return _codec(real_observation, args)


def _audit(infr, generated: Image.Image | None, used: Image.Image) -> dict:
    def describe(img):
        rgb = pil_img2rgb(img)
        arr = np.asarray(rgb)
        encoder_rgb = (rgb.resize((96, 96), Image.Resampling.BICUBIC)
                       .resize((256, 256), Image.Resampling.BICUBIC)
                       if TASK == "pusht" and EXPECTED_DECODER == "perseg" and
                       os.environ.get("BAGEL_PUSHT_BICUBIC_ROUNDTRIP", "1") == "1"
                       else rgb)
        vae = infr.vae_transform(encoder_rgb)
        vit = infr.vit_transform(encoder_rgb)
        return {"pixel_shape": list(arr.shape), "pixel_dtype": str(arr.dtype),
                "immediate_feed_shape": list(np.asarray(encoder_rgb).shape),
                "immediate_feed_dtype": str(np.asarray(encoder_rgb).dtype),
                "vae_transform_shape": list(vae.shape), "vae_transform_dtype": str(vae.dtype),
                "vit_transform_shape": list(vit.shape), "vit_transform_dtype": str(vit.dtype),
                "immediate_preprocess": ("bicubic 256->96->256" if encoder_rgb is not rgb else "identity"),
                "encode_modules": ["prepare_vae_images -> forward_cache_update_vae",
                                   "prepare_vit_images -> forward_cache_update_vit"]}
    a, b = (describe(generated) if generated is not None else None), describe(used)
    if a is not None and a != b:
        raise AssertionError(f"Image encoding paths differ: {a} versus {b}")
    return {"generated": a, "context": b, "identical": a == b if a is not None else None,
            "generation_skipped": generated is None}


def _mse(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    return float(np.mean((x - y) ** 2))


class _Replica:
    def __init__(self, real_env, ep, args, past_actions=()):
        if TASK == "maze":
            self.delegate = OracleReplica(real_env)
            self.env = self.delegate.env
        else:
            # Pymunk space deepcopy and mid-trajectory init_state restore do not
            # preserve contact caches. Reconstruct the identical hidden solver state
            # by replaying only the model's already committed actions from the
            # episode's exact recorded start state on a fresh deterministic env.
            self.delegate = None
            self.env, _ = G.make_profiled_env(ep, args)
            seed = int(ep.get("episode_seed") or ep.get("seed") or 0)
            _reset_preserving_init_state(self.env, seed=seed, init_state=ep["init_state"])
            for action in past_actions:
                self.env.step(action)
        self.terminal = False
        self.last_frame = render_rgb(self.env)
        if not np.array_equal(self.last_frame, render_rgb(real_env)):
            raise AssertionError("Oracle replica does not match real chunk-start frame")

    def step(self, action):
        if not self.terminal:
            if self.delegate is not None:
                self.last_frame = self.delegate.step(action)
                self.terminal = self.delegate.terminal
            else:
                _, _, term, trunc, _ = self.env.step(action)
                self.last_frame = render_rgb(self.env)
                self.terminal = bool(term or trunc)
        return self.last_frame.copy()

    def close(self):
        self.env.close()


def _state(env):
    state = env.unwrapped.get_init_state()
    keys = ("agent_location",) if TASK == "maze" else (
        "agent_position", "agent_velocity", "block_position", "block_velocity", "block_angle", "block_angvel")
    return tuple(json.dumps(state.get(k), sort_keys=True) for k in keys)


def _outdir(ep, args):
    seed = int(ep.get("seed") or ep.get("episode_seed") or
               (ep.get("alignment_index") or {}).get("seed") or
               (ep.get("metadata") or {}).get("seed") or 0)
    episode_seed = int(ep.get("episode_seed") or seed)
    index = int(ep["global_index"])
    return (Path(args.outdir) / base._env_slug(ep["env_id"]) / args.profile /
            "cot_selfrollout" / f"idx{index:07d}_seed{seed}_episode{episode_seed}")


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        return None


def _completed_result(outdir, ep, args):
    """Resume only a finished, matching intervention; keep conflicting results intact."""
    data = _read_json(outdir / "episode_stats.json")
    if not isinstance(data, dict) or not isinstance(data.get("result"), dict):
        return None
    result = data["result"]
    boolean_fields = ("success", "parse_failure", "stop_emitted", "truncated",
                      "committed_matches_imagined", "image_generation_skipped")
    count_fields = ("num_executed_actions", "num_chunks", "num_invalid_actions", "n_imagined")
    required = {"episode_id", "task", "condition", "termination_reason", "global_index",
                "seed", "episode_seed", "image_timesteps", "image_quality", "image_format",
                "imagined_image_size", "imagined_codec", "pixel_mse", *boolean_fields, *count_fields}
    if (not required.issubset(result)
            or any(type(result[key]) is not bool for key in boolean_fields)
            or any(type(result[key]) is not int or result[key] < 0 for key in count_fields)
            or not isinstance(result["pixel_mse"], list)):
        return None

    seed = int(ep.get("seed") or ep.get("episode_seed") or
               (ep.get("alignment_index") or {}).get("seed") or
               (ep.get("metadata") or {}).get("seed") or 0)
    episode_seed = int(ep.get("episode_seed") or seed)
    expected = {
        "episode_id": f"{TASK}:{int(ep['global_index']):07d}:{episode_seed}",
        "task": TASK, "condition": "Oracle GT" if CONDITION == "oracle" else CONDITION.capitalize(),
        "global_index": int(ep["global_index"]), "seed": seed, "episode_seed": episode_seed,
        "image_generation_skipped": CONDITION != "original",
        "image_timesteps": 50 if CONDITION == "original" else 0,
        "image_quality": QUALITY, "image_format": "JPEG",
        "imagined_image_size": list(NATIVE_SIZE), "imagined_codec": IMAGINED_CODEC,
    }
    mismatches = [key for key, value in expected.items() if result.get(key) != value]
    mismatches.extend(key for key, value in (("decoder", EXPECTED_DECODER), ("profile", args.profile))
                      if data.get(key) != value)
    if mismatches:
        raise ValueError(f"Completed ablation at {outdir} has different settings ({', '.join(mismatches)}); "
                         "use a new output directory. Existing results were preserved.")
    if (data.get("runner") != "agima_image_ablation_maze_pusht"
            or not isinstance(data.get("stats"), dict)
            or data["stats"].get("success") is not result["success"]):
        return None
    # These artifacts are written for the first ten episodes in every condition.
    if int(ep["global_index"]) < 10:
        trace = _read_json(outdir / "context_sequence.json")
        if (not isinstance(trace, list)
                or not isinstance(_read_json(outdir / "chunks.json"), list)
                or not isinstance(_read_json(outdir / "image_encoding_audit.json"), list)
                or any(not isinstance(item, dict) or not isinstance(item.get("file"), str)
                       or not (outdir / "context_frames" / item["file"]).is_file() for item in trace)):
            return None
    return {"skipped": True, "success": result["success"]}


def run_episode(env, vlm, extra_state_key, ep, args, rank):
    if args.image_quality != QUALITY or args.image_format.upper() != "JPEG":
        raise RuntimeError(f"{TASK} Table 1 requires JPEG quality {QUALITY}")
    runner = (production.run_maze2d_cot_episode if TASK == "maze" else production.run_pusht_cot_episode)
    outdir = _outdir(ep, args)
    completed = _completed_result(outdir, ep, args)
    if completed is not None and not args.overwrite:
        return completed

    infr = vlm.inferencer
    native_generate = infr.gen_cot_chunk_span
    native_chunk = production._gen_one_chunk if TASK == "pusht" else None
    native_step = env.step
    chunks = []
    context_frames = []
    invalid_actions = []
    past_actions = []
    audits = []
    all_mse = []

    def counted_step(action):
        before = _state(env) if not G._is_stop_action(action) else None
        outcome = native_step(action)
        past_actions.append(action)
        after = _state(env) if before is not None else None
        invalid_actions.append(bool(before is not None and before == after))
        return outcome

    def hooked_generate(*gen_args, **gen_kwargs):
        ci = len(chunks)
        real_pixels = render_rgb(env)
        real_observation = _jpeg_roundtrip(real_pixels, args.image_quality, args.image_format).convert("RGB")
        current = _real_to_imagined(real_observation, args)
        context_frames.append(("real_initial" if ci == 0 else "real_boundary", real_observation))
        replica = _Replica(env, ep, args, past_actions) if CONDITION in {"original", "oracle"} else None
        record = {"chunk": ci, "raw_generated": [], "generated": [], "used": [], "oracle": [],
                  "imagined_actions": [], "text": "", "parse_failure": False,
                  "pixel_mse": []}
        chunks.append(record)
        gray = _codec(Image.new("RGB", NATIVE_SIZE, (128, 128, 128)), args)

        def on_frame(generated_image, prefix, frame_index):
            generated = _codec(generated_image, args) if generated_image is not None else None
            if generated_image is not None:
                record["raw_generated"].append(pil_img2rgb(generated_image).copy())
            actions = parse_action_chunk(prefix)
            fresh = actions[len(record["imagined_actions"]):]
            if len(fresh) != 1:
                record["parse_failure"] = True
            record["imagined_actions"].extend(fresh)
            oracle = None
            if replica is not None:
                frame = replica.last_frame
                for action in fresh:
                    frame = replica.step(action)
                oracle = _real_to_imagined(
                    _jpeg_roundtrip(frame, args.image_quality, args.image_format), args)
                if CONDITION == "original":
                    loss = _mse(generated, oracle)
                    record["pixel_mse"].append(loss)
                    all_mse.append(loss)
            used = (generated if CONDITION == "original" else
                    gray if CONDITION == "blank" else
                    current if CONDITION == "stale" else oracle)
            if used is None:
                raise AssertionError("Oracle frame is missing")
            if not audits:
                audits.append(_audit(infr, generated, used))
            if generated is not None:
                record["generated"].append(generated.copy())
            record["used"].append(used.copy())
            if oracle is not None:
                record["oracle"].append(oracle.copy())
            context_frames.append((CONDITION, used.copy()))
            return used

        try:
            if CONDITION == "original":
                gen_kwargs["frame_callback"] = on_frame
            else:
                gen_kwargs["frame_provider"] = lambda prefix, frame_index: on_frame(
                    None, prefix, frame_index)
            text, images = native_generate(*gen_args, **gen_kwargs)
            record["text"] = text
            record["committed_actions"] = parse_action_chunk(
                text.split("</think>")[-1] if "</think>" in text else text)
            record["token_count"] = len(infr.tokenizer.encode(text, add_special_tokens=False))
            return text, images
        finally:
            if replica is not None:
                replica.close()

    def hooked_chunk(chunk_infr, conversation, max_iters, **chunk_kwargs):
        """Intercept only generated image pixels in PushT's published per-segment decoder.

        The production function still owns text decoding, VAE/ViT feedback, its
        bicubic 256→96→256 bottleneck, and conversation reconstruction. Returning
        the selected image from gen_image also replaces it in later chunk history.
        """
        if chunk_infr is not infr:
            raise AssertionError("Unexpected PushT inferencer instance")
        ci = len(chunks)
        real_pixels = render_rgb(env)
        real_observation = _jpeg_roundtrip(real_pixels, args.image_quality, args.image_format).convert("RGB")
        current = _real_to_imagined(real_observation, args)
        context_frames.append(("real_initial" if ci == 0 else "real_boundary", real_observation))
        replica = _Replica(env, ep, args, past_actions) if CONDITION in {"original", "oracle"} else None
        record = {"chunk": ci, "raw_generated": [], "generated": [], "used": [], "oracle": [],
                  "imagined_actions": [], "text": "", "parse_failure": False,
                  "pixel_mse": []}
        chunks.append(record)
        gray = _codec(Image.new("RGB", NATIVE_SIZE, (128, 128, 128)), args)
        text_parts = []
        native_text = infr.gen_text
        native_image = infr.gen_image

        def capture_text(*text_args, **text_kwargs):
            segment = native_text(*text_args, **text_kwargs)
            text_parts.append(segment)
            return segment

        def capture_image(*image_args, **image_kwargs):
            raw = native_image(*image_args, **image_kwargs) if CONDITION == "original" else None
            if raw is not None and raw.size != NATIVE_SIZE:
                raw = raw.resize(NATIVE_SIZE, Image.Resampling.LANCZOS)
            generated = _codec(raw, args) if raw is not None else None
            actions = parse_action_chunk("".join(text_parts))
            fresh = actions[len(record["imagined_actions"]):]
            if len(fresh) != 1:
                record["parse_failure"] = True
            record["imagined_actions"].extend(fresh)
            oracle = None
            if replica is not None:
                frame = replica.last_frame
                for action in fresh:
                    frame = replica.step(action)
                oracle = _real_to_imagined(
                    _jpeg_roundtrip(frame, args.image_quality, args.image_format), args)
                if CONDITION == "original":
                    loss = _mse(generated, oracle)
                    record["pixel_mse"].append(loss)
                    all_mse.append(loss)
            used = (generated if CONDITION == "original" else
                    gray if CONDITION == "blank" else
                    current if CONDITION == "stale" else oracle)
            if used is None:
                raise AssertionError("Oracle frame is missing")
            if not audits:
                audits.append(_audit(infr, generated, used))
            if raw is not None:
                record["raw_generated"].append(pil_img2rgb(raw).copy())
                record["generated"].append(generated.copy())
            record["used"].append(used.copy())
            if oracle is not None:
                record["oracle"].append(oracle.copy())
            context_frames.append((CONDITION, used.copy()))
            return used

        infr.gen_text = capture_text
        infr.gen_image = capture_image
        try:
            text, images, executed = native_chunk(chunk_infr, conversation, max_iters, **chunk_kwargs)
            record["text"] = text
            record["committed_actions"] = parse_action_chunk(
                text.split("</think>")[-1] if "</think>" in text else text)
            record["token_count"] = len(infr.tokenizer.encode(text, add_special_tokens=False))
            return text, images, executed
        finally:
            infr.gen_text = native_text
            infr.gen_image = native_image
            if replica is not None:
                replica.close()

    if TASK == "maze":
        infr.gen_cot_chunk_span = hooked_generate
    else:
        production._gen_one_chunk = hooked_chunk
    env.step = counted_step
    saved_render = args.render
    saved_images = args.save_generated_images
    saved_overwrite = args.overwrite
    if int(ep["global_index"]) >= 10:
        args.render = False
        args.save_generated_images = False
    try:
        # A production-only record is incomplete here; bypass its ordinary skip check.
        args.overwrite = True
        runner(env, vlm, extra_state_key, ep, args, rank)
    finally:
        if TASK == "maze":
            infr.gen_cot_chunk_span = native_generate
        else:
            production._gen_one_chunk = native_chunk
        env.step = native_step
        args.render = saved_render
        args.save_generated_images = saved_images
        args.overwrite = saved_overwrite

    stats_path = outdir / "episode_stats.json"
    data = json.loads(stats_path.read_text())
    hist = data.get("history") or []
    score = data["stats"]
    success = bool(score["success"])
    token_limit = any(r["token_count"] >= 600 for r in chunks)
    parse_failure = any(r["parse_failure"] or not r["committed_actions"] for r in chunks)
    stop = G._is_stop_action(score.get("final_action"))
    if success:
        reason = "solved"
    elif token_limit:
        reason = "truncated"
    elif parse_failure:
        reason = "parse_failure"
    elif score.get("truncated") or len(hist) >= int(data["instruction_max_steps"]):
        reason = "budget_exhausted"
    else:
        reason = "other"
    imagined_actions = [action for r in chunks for action in r["imagined_actions"]]
    executed_actions = [item["action"] for item in hist]
    for record in chunks:
        record["used_count"] = len(record["used"])
        record["executed_actions"] = [item["action"] for item in hist
                                      if int(item["chunk"]) == record["chunk"]]
    result = {"episode_id": f"{TASK}:{int(ep['global_index']):07d}:{int(data['episode_seed'])}",
              "task": TASK, "condition": "Oracle GT" if CONDITION == "oracle" else CONDITION.capitalize(),
              "success": success, "termination_reason": reason,
              "num_executed_actions": len(hist), "num_chunks": len(chunks),
              "parse_failure": parse_failure, "num_invalid_actions": sum(invalid_actions),
              "stop_emitted": bool(stop), "truncated": token_limit,
              "committed_matches_imagined": executed_actions == imagined_actions,
              "global_index": int(ep["global_index"]), "seed": int(data["seed"]),
              "episode_seed": int(data["episode_seed"]), "n_imagined": sum(len(r["used"]) for r in chunks),
              "image_generation_skipped": CONDITION != "original", "image_timesteps": (50 if CONDITION == "original" else 0),
              "image_quality": QUALITY, "image_format": "JPEG",
              "imagined_image_size": list(NATIVE_SIZE),
              "imagined_codec": IMAGINED_CODEC,
              "pixel_mse": all_mse}
    data["result"] = result
    data["runner"] = "agima_image_ablation_maze_pusht"
    if int(ep["global_index"]) < 10:
        (outdir / "image_encoding_audit.json").write_text(json.dumps(audits, indent=2))
        trace = []
        folder = outdir / "context_frames"
        folder.mkdir(exist_ok=True)
        for i, (source, image) in enumerate(context_frames):
            name = f"{i:04d}_{source}.png"
            image.save(folder / name)
            trace.append({"index": i, "source": source, "file": name})
        (outdir / "context_sequence.json").write_text(json.dumps(trace, indent=2))
        for record in chunks:
            ci = record["chunk"]
            folder = outdir / f"chunk_{ci:03d}"
            folder.mkdir(exist_ok=True)
            for label in ("raw_generated", "generated", "used", "oracle"):
                for j, image in enumerate(record[label]):
                    image.save(folder / f"{label}_{j:03d}.png")
        meta = [{k: v for k, v in r.items() if k not in ("raw_generated", "generated", "used", "oracle")}
                for r in chunks]
        (outdir / "chunks.json").write_text(json.dumps(meta, indent=2))
    # Publish the completed record only after its traces, with an atomic replacement.
    temporary_stats = stats_path.with_suffix(".json.tmp")
    temporary_stats.write_text(json.dumps(base._json_safe(data), indent=2))
    temporary_stats.replace(stats_path)
    return {"success": success}


if __name__ == "__main__":
    G.run_episode = run_episode
    G.main()
