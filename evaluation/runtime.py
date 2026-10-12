#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified for Agima: package imports and portable resource configuration.
"""
DDP-style Sokoban gen-eval runner.

This file intentionally lives next to evaluation/base.py without
modifying it.  It imports the existing gen-eval env/model/GIF helpers, then
adds the Sokoban-specific distributed path here:

  * torchrun ranks shard episodes across nodes;
  * rank 0 builds compact shard caches;
  * local rank 0 can stage one checkpoint copy per node;
  * Sokoban eval is forced to move_only + stop_then_solved;
  * v7/v8 eval input profiles are explicit.

The "252" Sokoban detail belongs to the VIT transform, not the env render:
v8 frames are resized to 256x256, then the training VIT transform
ImageTransform(max=336, min=224, stride=14) rounds 256 to 252.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import hashlib
import json
import os
import random
import shutil
import socket
import sys
import time
import traceback
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("VISGYM_ROOT", str(Path(__file__).resolve().parents[1]))
_VISGYM_ROOT = Path(os.environ["VISGYM_ROOT"]).expanduser().resolve()
if str(_VISGYM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VISGYM_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image
from tqdm import tqdm

import evaluation.base as base
from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from evaluation.environments import _reset_preserving_init_state
from evaluation.rank_summaries import (aggregate_summaries, summary_session_id,
                                       wait_for_rank_summaries, write_rank_summary, write_summary)
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer


PROFILE_CHOICES = (
    "sokoban_v7_native",
    "sokoban_v8_res256",
    "pacman_res256",
    "pusht_norm4_native",
    "maze2d_native256",
)
DEFAULT_BAGEL_BASE = os.environ.get("BAGEL_BASE_PATH")


class RenderResizeEnv:
    """Resize rendered frames before Interactor encodes them for the VLM."""

    def __init__(self, env, size: tuple[int, int], resample=Image.Resampling.NEAREST):
        self._env = env
        self.size = tuple(int(v) for v in size)
        self.resample = resample

    def render(self, *args, **kwargs):
        frame = self._env.render(*args, **kwargs)
        pil = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame).astype(np.uint8))
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        if pil.size != self.size:
            pil = pil.resize(self.size, self.resample)
        return np.asarray(pil)

    def reset(self, **kwargs):
        return _reset_preserving_init_state(self._env, **kwargs)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(f"{type(self).__name__!r} has no attribute 'env'")
        return getattr(self._env, name)

    def __str__(self):
        return f"RenderResizeEnv({self._env}, size={self.size})"

    def __repr__(self):
        return self.__str__()


class TextOnlyUnifiedVLM(base.UnifiedVLM):
    """Plain-action BAGEL wrapper that skips test-time image generation."""

    def query(
        self,
        image_base64: str,
        prompt: str,
        final_goal_prompt: Optional[str] = None,
        final_goal_img_b64: Optional[str] = None,
    ) -> str:
        print(f"[vlm-text-only] step={self.idx} starting query", flush=True)
        image = Image.open(BytesIO(base64.b64decode(image_base64)))
        self.conversation_history.append(image)
        if final_goal_img_b64:
            final_goal_image = Image.open(BytesIO(base64.b64decode(final_goal_img_b64)))
            self.conversation_history.append(final_goal_image)
        if final_goal_prompt:
            self.conversation_history.append(final_goal_prompt)
        self.conversation_history.append(prompt)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = self.inferencer(interleaved_input=self.conversation_history, understanding_output=True, vae=True)
            output_text = output["text"][0].split("\n")[0]
            if ":" in output_text:
                output_text = output_text.split(":", 1)[1]
            output_text = output_text.strip("`").strip()
        self.conversation_history.append(output_text)
        self.idx += 1
        print(f"[vlm-text-only] step={self.idx - 1} action_text={output_text!r}", flush=True)
        return output_text


def _rank_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def init_distributed(*, require_cuda: bool) -> tuple[int, int, int]:
    rank = _rank_env_int("RANK", 0)
    world_size = _rank_env_int("WORLD_SIZE", 1)
    local_rank = _rank_env_int("LOCAL_RANK", 0)

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if require_cuda:
        if cuda_count <= local_rank:
            raise RuntimeError(
                f"CUDA device for local_rank={local_rank} is not available "
                f"(visible cuda device count={cuda_count})."
            )
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def barrier():
    if dist.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[_rank_env_int("LOCAL_RANK", 0)])
        else:
            dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wait_for_path(path: Path, *, timeout_sec: int, label: str):
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {label}: {path}")


def wait_for_staged_checkpoint(marker: Path, weight: Path, expected_size: int, timeout_sec: int):
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        if marker.exists() and weight.exists():
            try:
                if weight.stat().st_size == expected_size:
                    return
            except FileNotFoundError:
                pass
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for staged checkpoint: {weight}")


def _checkpoint_files(weight: Path) -> list[Path]:
    """Include every shard named by a Hugging Face safetensors index."""
    if not weight.name.endswith(".safetensors.index.json"):
        return [weight]
    index = json.loads(weight.read_text())
    shards = sorted(set(index["weight_map"].values()))
    files = [weight] + [weight.parent / name for name in shards]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint shard is missing: {path}")
    return files


def _resolve_weight_path(model_path: str) -> Path:
    p = Path(model_path).expanduser()
    if p.is_file():
        return p
    return Path(base._resolve_checkpoint_path(str(p)))


def _safe_name(path: str) -> str:
    p = Path(path).expanduser()
    digest = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{p.parent.name}_{p.name}_{digest}".replace("/", "_")


def stage_checkpoint_if_requested(args, local_rank: int, rank: int) -> str:
    if not args.stage_checkpoint:
        return args.bagel_path

    source_weight = _resolve_weight_path(args.bagel_path)
    source_files = _checkpoint_files(source_weight)
    root = Path(args.local_ckpt_root).expanduser()
    dest_dir = root / _safe_name(args.bagel_path)
    dest_files = [dest_dir / path.name for path in source_files]
    marker = dest_dir / ".complete.json"

    if local_rank == 0:
        dest_dir.mkdir(parents=True, exist_ok=True)
        missing = [
            (src, dst) for src, dst in zip(source_files, dest_files)
            if not dst.exists() or dst.stat().st_size != src.stat().st_size
        ]
        if missing:
            if marker.exists():
                marker.unlink()
            for src, dst in missing:
                tmp_weight = dest_dir / f".{src.name}.tmp"
                if tmp_weight.exists():
                    tmp_weight.unlink()
                print(f"[rank {rank}] staging checkpoint {src} -> {dst}", flush=True)
                shutil.copy2(src, tmp_weight)
                tmp_weight.replace(dst)
        tmp_marker = dest_dir / ".complete.json.tmp"
        tmp_marker.write_text(
            json.dumps(
                {
                    "source": str(source_weight),
                    "staged": [str(path) for path in dest_files],
                    "hostname": socket.gethostname(),
                    "timestamp": dt.datetime.now().isoformat(),
                },
                indent=2,
            )
        )
        tmp_marker.replace(marker)

    deadline = time.time() + float(args.stage_timeout_sec)
    while time.time() < deadline:
        if marker.exists() and all(
            dst.exists() and dst.stat().st_size == src.stat().st_size
            for src, dst in zip(source_files, dest_files)
        ):
            break
        time.sleep(1.0)
    else:
        raise TimeoutError(f"Timed out waiting for staged checkpoint shards: {dest_dir}")
    return str(dest_dir)


def load_bagel_ddp(
    *,
    model_path: str,
    local_rank: int,
    max_mem_per_gpu: str,
    bagel_base_path: str,
    vae_path: Optional[str],
    vit_config_path: Optional[str],
    vit_max_num_patch_per_side: int,
    latent_patch_size: int,
    max_latent_size: int,
    vae_transform_max_size: int,
    vae_transform_min_size: int,
    vae_transform_stride: int,
    vit_transform_max_size: int,
    vit_transform_min_size: int,
    vit_transform_stride: int,
):
    model_path = str(Path(model_path).expanduser())
    configs_path = str(Path(bagel_base_path).expanduser().resolve())
    vae_file = Path(vae_path).expanduser() if vae_path else Path(configs_path, "ae.safetensors")
    vit_config_file = Path(vit_config_path).expanduser() if vit_config_path else Path(configs_path, "vit_config.json")

    print(f"[load_bagel_ddp] model_path={model_path}", flush=True)
    print(f"[load_bagel_ddp] configs_path={configs_path}", flush=True)
    print(f"[load_bagel_ddp] rank local cuda device={local_rank}", flush=True)

    llm_config = Qwen2Config.from_json_file(Path(configs_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = getattr(llm_config, "bos_token_id", getattr(llm_config, "eos_token_id", 0))

    vit_config = SiglipVisionConfig.from_json_file(vit_config_file)
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=vae_file)
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=vit_max_num_patch_per_side,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=latent_patch_size,
        max_latent_size=max_latent_size,
    )

    tokenizer = Qwen2Tokenizer.from_pretrained(configs_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    ckpt_path = _resolve_weight_path(model_path)
    print(f"[load_bagel_ddp] checkpoint_path={ckpt_path}", flush=True)
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        bagel = Bagel(language_model, vit_model, config)
        bagel.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    vae_transform = ImageTransform(
        vae_transform_max_size,
        vae_transform_min_size,
        vae_transform_stride,
    )
    vit_transform = ImageTransform(
        vit_transform_max_size,
        vit_transform_min_size,
        vit_transform_stride,
    )
    print(
        "[load_bagel_ddp] transforms "
        f"vae=({vae_transform_max_size},{vae_transform_min_size},{vae_transform_stride}) "
        f"vit=({vit_transform_max_size},{vit_transform_min_size},{vit_transform_stride})",
        flush=True,
    )

    device_id = int(local_rank)
    device_map = infer_auto_device_map(
        bagel,
        max_memory={device_id: max_mem_per_gpu},
        no_split_module_classes=[
            "Bagel",
            "Qwen2MoTDecoderLayer",
            "SiglipEncoderLayer",
            "SiglipVisionTransformer",
            "MLPconnector",
            "TimestepEmbedder",
            "PositionEmbedding",
        ],
    )
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    for name in same_device_modules:
        device_map[name] = device_id
    print(f"[load_bagel_ddp] device_map={device_map}", flush=True)

    try:
        bagel = load_checkpoint_and_dispatch(
            bagel,
            checkpoint=str(ckpt_path),
            device_map=device_map,
            offload_buffers=True,
            offload_folder=f"offload_rank{local_rank}",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    except Exception as exc:
        raise RuntimeError(f"Failed to dispatch checkpoint {ckpt_path}") from exc

    inferencer = InterleaveInferencer(
        model=bagel,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    meta = {
        "tokenizer": tokenizer,
        "new_token_ids": new_token_ids,
        "vae_transform": vae_transform,
        "vit_transform": vit_transform,
        "device_map": device_map,
        "configs_path": configs_path,
        "vae_path": str(vae_file),
        "vit_config_path": str(vit_config_file),
    }
    return inferencer, meta


def is_sokoban_env_id(env_id: str) -> bool:
    return str(env_id).startswith("sokoban/")


def is_pusht_env_id(env_id: str) -> bool:
    return str(env_id).startswith("pusht/")


def is_maze2d_env_id(env_id: str) -> bool:
    return str(env_id).startswith("maze_2d/")


def prepare_sokoban_episode(ep: dict, args) -> dict:
    if not is_sokoban_env_id(ep.get("env_id", "")):
        return ep
    out = deepcopy(ep)
    env_kwargs = dict(out.get("env_kwargs") or {})
    env_kwargs["action_mode_override"] = args.sokoban_action_mode_override
    env_kwargs["success_mode"] = args.sokoban_success_mode
    out["env_kwargs"] = env_kwargs

    init_state = out.get("init_state")
    if isinstance(init_state, dict):
        init_state = dict(init_state)
        init_state["action_mode"] = args.sokoban_action_mode_override
        init_state["success_mode"] = args.sokoban_success_mode
        out["init_state"] = init_state
    return out


def apply_profile_wrapper(env, profile: str):
    if profile == "sokoban_v7_native":
        return env
    if profile == "sokoban_v8_res256":
        return RenderResizeEnv(env, size=(256, 256), resample=Image.Resampling.NEAREST)
    if profile == "pacman_res256":
        # Pacman res256 SFT data was rendered native (grid*tile=264) then BICUBIC
        # to 256x256 inside the env. The episode env_kwargs carry render_size=256
        # so the env already renders 256; this wrapper is a BICUBIC no-op safety
        # net (and matches the data-gen resampling if render_size is ever dropped).
        return RenderResizeEnv(env, size=(256, 256), resample=Image.Resampling.BICUBIC)
    if profile == "pusht_norm4_native":
        return env
    if profile == "maze2d_native256":
        return env
    raise ValueError(f"Unsupported profile: {profile}")


def validate_sokoban_contract(env, env_id: str):
    if not is_sokoban_env_id(env_id):
        return
    prompt = env.get_prompt()
    prompt_l = prompt.lower()
    if "stop is optional" in prompt_l:
        raise RuntimeError(f"{env_id} prompt still says stop is optional.")
    if "('stop', 'stop')" not in prompt:
        raise RuntimeError(f"{env_id} prompt does not require ('stop', 'stop').")
    if "push" not in prompt_l:
        raise RuntimeError(f"{env_id} prompt does not describe push semantics.")
    action_mode = getattr(getattr(env, "unwrapped", env), "action_mode", None)
    if action_mode != "move_only":
        raise RuntimeError(f"{env_id} action_mode={action_mode!r}; expected move_only.")


def validate_pusht_contract(env, env_id: str):
    if not is_pusht_env_id(env_id):
        return
    inner = getattr(env, "unwrapped", env)
    prompt = inner.get_prompt()
    if "Stop is required after the target is aligned." not in prompt:
        raise RuntimeError(f"{env_id} prompt is not the stop-required PushT prompt.")
    if "Environment feedback:" in prompt:
        raise RuntimeError(f"{env_id} prompt unexpectedly includes env feedback text.")
    if getattr(inner, "action_codec", None) != "norm4":
        raise RuntimeError(f"{env_id} action_codec={getattr(inner, 'action_codec', None)!r}; expected norm4.")
    if getattr(inner, "action_marker_mode", None) != "none":
        raise RuntimeError(
            f"{env_id} action_marker_mode={getattr(inner, 'action_marker_mode', None)!r}; expected none."
        )
    if getattr(inner, "enable_push_macros", None) is not False:
        raise RuntimeError(f"{env_id} enable_push_macros is not disabled for norm4 PushT eval.")


def _maze2d_env_kwargs_from_metadata(ep: dict) -> dict:
    env_kwargs = dict(ep.get("env_kwargs") or {})
    metadata = ep.get("metadata") or {}
    for key in ("maze_width", "maze_height", "target_in_center", "render_size"):
        if key in metadata and key not in env_kwargs:
            env_kwargs[key] = metadata[key]
    return env_kwargs


def validate_maze2d_contract(env, env_id: str, ep: dict):
    if not is_maze2d_env_id(env_id):
        return
    inner = getattr(env, "unwrapped", env)
    prompt = inner.get_prompt()
    if "('stop', 'stop')" not in prompt:
        raise RuntimeError(f"{env_id} prompt does not expose stop action.")
    if "Reaching the target without stopping is not a success." not in prompt:
        raise RuntimeError(f"{env_id} prompt is not stop-required.")
    metadata = ep.get("metadata") or {}
    for attr, key in (
        ("maze_width", "maze_width"),
        ("maze_height", "maze_height"),
        ("target_in_center", "target_in_center"),
        ("render_size", "render_size"),
    ):
        if key in metadata and getattr(inner, attr, None) != metadata[key]:
            raise RuntimeError(
                f"{env_id} {attr}={getattr(inner, attr, None)!r}; expected {metadata[key]!r}."
            )


def make_profiled_env(ep: dict, args):
    env_id = ep["env_id"]
    env_spec = base.get_spec_from_episode(ep)
    if is_pusht_env_id(env_id):
        env_kwargs = dict(env_spec.get("env_kwargs") or {})
        env_kwargs["enable_push_macros"] = False
        env_spec["env_kwargs"] = env_kwargs
    if is_maze2d_env_id(env_id):
        env_spec["env_kwargs"] = _maze2d_env_kwargs_from_metadata(ep)
    seed = int(ep.get("seed") or ep.get("episode_seed") or 0)
    raw_env, _seed_key, extra_state_key = base.make_env_from_registry(env_id, env_spec, seed)
    env = base.apply_gen_eval_wrappers(raw_env, args)
    env = apply_profile_wrapper(env, args.profile)
    validate_sokoban_contract(env, env_id)
    validate_pusht_contract(env, env_id)
    validate_maze2d_contract(env, env_id, ep)
    return env, extra_state_key


def _open_jsonl(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _compact_episode(ep: dict, source: Path, source_line: int, global_index: int) -> dict:
    keep = {
        "env_id": ep.get("env_id"),
        "seed": ep.get("seed"),
        "episode_seed": ep.get("episode_seed"),
        "seed_key": ep.get("seed_key"),
        "env_kwargs": ep.get("env_kwargs"),
        "init_state": ep.get("init_state"),
        "extra_state": ep.get("extra_state") if isinstance(ep.get("extra_state"), str) else None,
        "source_file": str(source),
        "source_line": source_line,
        "global_index": global_index,
    }
    env_id = ep.get("env_id", "")
    if is_maze2d_env_id(env_id):
        keep["metadata"] = ep.get("metadata")
        keep["alignment_index"] = ep.get("alignment_index")
        keep["hash"] = ep.get("hash")
        keep["trajectory_steps"] = ep.get("trajectory_steps")
        history = ep.get("history") or []
        keep["history_prompts"] = [
            item.get("prompt")
            for item in history
            if isinstance(item, dict) and isinstance(item.get("prompt"), str)
        ]
    return {k: v for k, v in keep.items() if v is not None}


def scan_compact_episodes(episodes_root: str, limit: Optional[int], env_filter: Optional[set[str]]) -> list[dict]:
    episodes: list[dict] = []
    for jsonl_path in base.iter_jsonl_files(episodes_root):
        with _open_jsonl(jsonl_path) as f:
            for source_line, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"Invalid episode JSON at {jsonl_path}:{source_line}: {exc}") from exc
                if not isinstance(ep, dict):
                    raise ValueError(f"Expected an episode object at {jsonl_path}:{source_line}")
                env_id = ep.get("env_id")
                if not isinstance(env_id, str) or not env_id:
                    raise ValueError(f"Missing episode env_id at {jsonl_path}:{source_line}")
                if env_filter and env_id not in env_filter:
                    continue
                episodes.append(_compact_episode(ep, jsonl_path, source_line, len(episodes)))
                if limit is not None and len(episodes) >= limit:
                    return episodes
    if not episodes:
        raise ValueError(f"No matching evaluation episodes found at {episodes_root}")
    return episodes


def _cache_manifest(args, world_size: int) -> dict:
    return {
        # A job rebuilds its small episode cache so changed inputs cannot reuse
        # an earlier job's rows. The ID also lets late workers join safely.
        "session_id": summary_session_id(),
        "episodes_root": str(Path(args.episodes_root).expanduser()),
        "limit": args.limit,
        "env_filter": sorted(args.env_filter or []),
        "world_size": world_size,
        "profile": args.profile,
        "sokoban_action_mode_override": args.sokoban_action_mode_override,
        "sokoban_success_mode": args.sokoban_success_mode,
        "pusht_reset_source": getattr(args, "pusht_reset_source", "init_state"),
        "skip_image_generation": bool(getattr(args, "skip_image_generation", False)),
    }


def wait_for_shard_manifest(
    manifest_path: Path,
    request: dict,
    *,
    timeout_sec: int,
):
    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                time.sleep(1.0)
                continue
            if manifest.get("request") == request:
                return manifest
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for matching shard manifest: {manifest_path}")


def prepare_shard_cache(args, rank: int, world_size: int) -> list[dict]:
    cache_dir = Path(args.shard_cache_dir or Path(args.outdir) / "_episode_shards").expanduser()
    manifest_path = cache_dir / "manifest.json"
    env_filter = set(args.env_filter) if args.env_filter else None
    manifest = _cache_manifest(args, world_size)

    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        use_existing = False
        if not args.rebuild_shard_cache and manifest_path.exists():
            try:
                old = json.loads(manifest_path.read_text())
                use_existing = old.get("request") == manifest
            except Exception:
                use_existing = False

        if not use_existing:
            if manifest_path.exists():
                manifest_path.unlink()
            for shard in cache_dir.glob("shard_*.jsonl"):
                shard.unlink()
            episodes = scan_compact_episodes(args.episodes_root, args.limit, env_filter)
            shard_counts = [0 for _ in range(world_size)]
            handles = [
                open(cache_dir / f"shard_{idx:05d}.jsonl", "w", encoding="utf-8")
                for idx in range(world_size)
            ]
            try:
                for idx, ep in enumerate(episodes):
                    shard = idx % world_size
                    handles[shard].write(json.dumps(ep) + "\n")
                    shard_counts[shard] += 1
            finally:
                for handle in handles:
                    handle.close()
            temporary_manifest = manifest_path.with_suffix(".json.tmp")
            temporary_manifest.write_text(
                json.dumps(
                    {
                        "request": manifest,
                        "episode_count": len(episodes),
                        "shard_counts": shard_counts,
                        "timestamp": dt.datetime.now().isoformat(),
                    },
                    indent=2,
                )
            )
            temporary_manifest.replace(manifest_path)
            print(f"[rank 0] built shard cache {cache_dir} episodes={len(episodes)}", flush=True)
        else:
            print(f"[rank 0] reusing shard cache {cache_dir}", flush=True)

    wait_for_shard_manifest(
        manifest_path,
        manifest,
        timeout_sec=args.shard_cache_timeout_sec,
    )
    shard_path = cache_dir / f"shard_{rank:05d}.jsonl"
    wait_for_path(shard_path, timeout_sec=args.shard_cache_timeout_sec, label=f"episode shard rank {rank}")
    if not shard_path.exists():
        return []
    with open(shard_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _light_history(history: list[dict], generated_count: int) -> list[dict]:
    out = []
    for idx, step in enumerate(history):
        info = step.get("info")
        item = {
            "step": step.get("step"),
            "prompt": step.get("prompt"),
            "vlm_output": step.get("vlm_output"),
            "action": step.get("action"),
            "reward": step.get("reward"),
            "info": info,
        }
        if isinstance(info, dict) and "env_feedback" in info:
            item["env_feedback"] = info.get("env_feedback")
        if "input_source" in step:
            item["input_source"] = step.get("input_source")
        if "cot_self_rollout" in step:
            item["cot_self_rollout"] = step.get("cot_self_rollout")
        for key in (
            "actual_input_source",
            "env_current_image_path",
            "env_next_image_path",
            "model_input_image_path",
            "generated_image_path",
            "cot_generated_image_paths",
        ):
            if key in step:
                item[key] = step.get(key)
        if idx < generated_count:
            item["generated_image_available"] = True
        out.append(item)
    return out


def _save_render_artifacts(env, interactor, result: dict, artifact_history: list[dict], out_episode_dir: Path, args):
    if (args.render or args.make_gif) and interactor.hist:
        for i, step in enumerate(interactor.hist):
            img = None
            for key in ("image", "frame", "obs", "render", "rgb"):
                if key in step:
                    img = step[key]
                    break
            if img is None:
                try:
                    img = env.render(mode="rgb_array")
                except Exception:
                    img = None
            arr = base._render_like_to_array(img) if img is not None else None
            if arr is not None:
                base.save_step_render(arr, out_episode_dir / "renders", i)

        try:
            img = env.render(mode="rgb_array")
        except TypeError:
            try:
                img = env.render()
            except Exception:
                img = None
        except Exception:
            img = None
        arr = base._render_like_to_array(img) if img is not None else None
        if arr is not None:
            base.save_step_render(arr, out_episode_dir / "renders", len(interactor.hist))

    if args.save_generated_images:
        for i, step in enumerate(artifact_history):
            encoded = step.get("generated_image")
            if not encoded:
                continue
            try:
                img = Image.open(BytesIO(base64.b64decode(encoded)))
            except Exception:
                continue
            base.save_step_pil(img, out_episode_dir / "generated_renders", i)

    if not args.make_gif:
        return

    try:
        frame_paths = base.list_render_images(out_episode_dir)
        artifact_result = dict(result)
        artifact_result["history"] = artifact_history
        has_generated_frames = any(step.get("generated_image") for step in artifact_history)
        reward_val = result.get("stats", {}).get("reward", None)
        reward_str = f"{float(reward_val):.3f}" if isinstance(reward_val, (int, float)) else "NA"
        tag = f"idx{result.get('global_index')}_seed{result.get('seed')}_episode{result.get('episode_seed')}"
        env_slug_dir = out_episode_dir.parent
        env_gif_dir = env_slug_dir / args.env_gif_subdir
        env_gif_dir.mkdir(parents=True, exist_ok=True)
        base.save_episode_gif(
            episode_data=artifact_result,
            frame_paths=frame_paths,
            out_gif=env_gif_dir / f"{tag}_env_r{reward_str}.gif",
            ms_per_frame=args.gif_ms_per_frame,
            start_hold_ms=args.gif_start_hold_ms,
            end_hold_ms=args.gif_end_hold_ms,
            max_text_chars=args.gif_max_text_chars,
            max_width=args.gif_max_width,
            min_width=512,
            captions=True,
            preserve_quality=args.gif_preserve_quality,
        )
        if has_generated_frames:
            generated_gif_dir = env_slug_dir / args.generated_gif_subdir
            generated_gif_dir.mkdir(parents=True, exist_ok=True)
            base.save_generated_data_gif(
                episode_data=artifact_result,
                frame_paths=[],
                out_gif=generated_gif_dir / f"{tag}_generated_r{reward_str}.gif",
                ms_per_frame=args.gif_ms_per_frame,
                start_hold_ms=args.gif_start_hold_ms,
                end_hold_ms=args.gif_end_hold_ms,
                max_text_chars=args.gif_max_text_chars,
                max_width=args.gif_max_width,
                captions=True,
                preserve_quality=args.gif_preserve_quality,
            )
    except Exception:
        traceback.print_exc()


def _image_to_b64_for_eval(
    img: np.ndarray | Image.Image,
    *,
    image_format: str,
    image_quality: int,
    max_long_side: Optional[int],
) -> str:
    pil = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img).astype(np.uint8))
    if max_long_side is not None:
        w, h = pil.size
        max_dim = max(w, h)
        if max_dim > int(max_long_side):
            scale = float(max_long_side) / float(max_dim)
            pil = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    if pil.mode in ("I", "F", "I;16", "I;16B", "P", "LA", "RGBA"):
        pil = pil.convert("RGB")
    buf = BytesIO()
    if str(image_format).upper() == "PNG":
        pil.save(buf, format="PNG")
    else:
        pil.save(buf, format="JPEG", quality=int(image_quality))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_artifact_to_pil(img: Any) -> Optional[Image.Image]:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = base._render_like_to_array(img)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _write_pusht_image_artifact(out_episode_dir: Path, rel_path: str, img: Any) -> Optional[str]:
    pil = _image_artifact_to_pil(img)
    if pil is None:
        return None
    path = out_episode_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path)
    return rel_path


def record_pusht_step_artifacts(
    *,
    out_episode_dir: Path,
    step_idx: int,
    step_record: dict[str, Any],
    env_current_frame: Any,
    env_next_frame: Any,
    vlm: Any,
    args: Any,
) -> None:
    """Persist the exact image surfaces needed to audit PushT self-rollout."""
    should_save_env = bool(getattr(args, "render", False))
    should_save_generated = bool(getattr(args, "save_generated_images", False))
    if should_save_env:
        current_rel = _write_pusht_image_artifact(
            out_episode_dir,
            f"env_current/step_{step_idx:04d}.png",
            env_current_frame,
        )
        next_rel = _write_pusht_image_artifact(
            out_episode_dir,
            f"env_next/step_{step_idx:04d}.png",
            env_next_frame,
        )
        if current_rel:
            step_record["env_current_image_path"] = current_rel
        if next_rel:
            step_record["env_next_image_path"] = next_rel

    input_source = getattr(vlm, "last_input_source", None)
    if input_source is not None:
        step_record["actual_input_source"] = input_source

    if should_save_generated:
        input_rel = _write_pusht_image_artifact(
            out_episode_dir,
            f"model_inputs/step_{step_idx:04d}.png",
            getattr(vlm, "last_input_image", None),
        )
        if input_rel:
            step_record["model_input_image_path"] = input_rel

        last_generated_image = getattr(vlm, "last_generated_image", None)
        if last_generated_image is None and str(input_source or "").startswith("generated"):
            last_generated_image = getattr(vlm, "last_input_image", None)
        generated_rel = _write_pusht_image_artifact(
            out_episode_dir,
            f"generated_renders/step_{step_idx:04d}.png",
            last_generated_image,
        )
        if generated_rel:
            step_record["generated_image_path"] = generated_rel
            step_record["generated_image_available"] = True

        cot_generated_paths = []
        for k, gen_img in enumerate(getattr(vlm, "last_gen_imgs", []) or []):
            rel = _write_pusht_image_artifact(
                out_episode_dir,
                f"cot_generated_renders/step_{step_idx:04d}_gen_{k:02d}.png",
                gen_img,
            )
            if rel:
                cot_generated_paths.append(rel)
        if cot_generated_paths:
            step_record["cot_generated_image_paths"] = cot_generated_paths


def make_pusht_artifact_gifs(out_episode_dir: Path, result: dict[str, Any], args: Any) -> None:
    if not getattr(args, "make_gif", False):
        return
    reward_val = result.get("stats", {}).get("reward", None)
    reward_str = f"{float(reward_val):.3f}" if isinstance(reward_val, (int, float)) else "NA"
    tag = f"idx{result.get('global_index')}_seed{result.get('seed')}_episode{result.get('episode_seed')}"
    env_slug_dir = out_episode_dir.parent

    env_frames = sorted((out_episode_dir / "env_current").glob("step_*.png"))
    env_next_frames = sorted((out_episode_dir / "env_next").glob("step_*.png"))
    if env_next_frames:
        env_frames = env_frames + [env_next_frames[-1]]
    if env_frames:
        env_gif_dir = env_slug_dir / args.env_gif_subdir
        env_gif_dir.mkdir(parents=True, exist_ok=True)
        base.save_episode_gif(
            episode_data=result,
            frame_paths=env_frames,
            out_gif=env_gif_dir / f"{tag}_env_r{reward_str}.gif",
            ms_per_frame=args.gif_ms_per_frame,
            start_hold_ms=args.gif_start_hold_ms,
            end_hold_ms=args.gif_end_hold_ms,
            max_text_chars=args.gif_max_text_chars,
            max_width=args.gif_max_width,
            min_width=args.gif_max_width,
            preserve_quality=args.gif_preserve_quality,
        )

    generated_frames = sorted((out_episode_dir / "generated_renders").glob("step_*.png"))
    if generated_frames:
        generated_gif_dir = env_slug_dir / args.generated_gif_subdir
        generated_gif_dir.mkdir(parents=True, exist_ok=True)
        base.save_generated_data_gif(
            episode_data=result,
            frame_paths=generated_frames,
            out_gif=generated_gif_dir / f"{tag}_generated_r{reward_str}.gif",
            ms_per_frame=args.gif_ms_per_frame,
            start_hold_ms=args.gif_start_hold_ms,
            end_hold_ms=args.gif_end_hold_ms,
            max_text_chars=args.gif_max_text_chars,
            max_width=args.gif_max_width,
            min_width=args.gif_max_width,
            preserve_quality=args.gif_preserve_quality,
        )

    cot_generated_frames = sorted((out_episode_dir / "cot_generated_renders").glob("step_*.png"))
    if cot_generated_frames:
        cot_gif_dir = env_slug_dir / "cot_generated_gif"
        cot_gif_dir.mkdir(parents=True, exist_ok=True)
        base.save_generated_data_gif(
            episode_data=result,
            frame_paths=cot_generated_frames,
            out_gif=cot_gif_dir / f"{tag}_cot_generated_r{reward_str}.gif",
            ms_per_frame=args.gif_ms_per_frame,
            start_hold_ms=args.gif_start_hold_ms,
            end_hold_ms=args.gif_end_hold_ms,
            max_text_chars=args.gif_max_text_chars,
            max_width=args.gif_max_width,
            min_width=args.gif_max_width,
            preserve_quality=args.gif_preserve_quality,
        )


def run_pusht_episode(env, vlm, ep: dict, args, rank: int) -> dict:
    if args.self_rollout and args.skip_image_generation:
        raise RuntimeError("PushT self-rollout requires image generation; remove --skip-image-generation.")
    env_id = ep["env_id"]
    seed = int(ep.get("seed") or ep.get("episode_seed") or 0)
    episode_seed = int(ep.get("episode_seed") or seed)
    max_steps = 1 if args.sanity_check else base._episode_max_steps(ep, args.max_steps)
    eval_mode = "selfrollout" if args.self_rollout else "gt_input"
    out_episode_dir = (
        Path(args.outdir)
        / base._env_slug(env_id)
        / args.profile
        / eval_mode
        / f"idx{int(ep['global_index']):07d}_seed{seed}_episode{episode_seed}"
    )

    json_file = out_episode_dir / "episode_stats.json"
    if not args.overwrite and json_file.exists():
        try:
            old = json.loads(json_file.read_text())
            if base.looks_complete(old):
                return {"skipped": True, "success": bool(old.get("stats", {}).get("success", False))}
        except Exception:
            pass

    init_state = ep.get("init_state")
    if args.pusht_reset_source == "init_state" and init_state is None:
        raise RuntimeError(f"{env_id} episode {ep.get('global_index')} has no init_state for PushT eval.")

    if hasattr(vlm, "reset_conversation"):
        vlm.reset_conversation()
    reset_kwargs = {"seed": episode_seed}
    if args.pusht_reset_source == "init_state":
        reset_kwargs["init_state"] = init_state
    _reset_preserving_init_state(env, **reset_kwargs)
    out_episode_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    final_action = None
    terminated = False
    truncated = False
    cumulative_reward = 0.0

    for step_idx in range(max_steps):
        frame = env.render()
        img_b64 = _image_to_b64_for_eval(
            frame,
            image_format=args.image_format,
            image_quality=args.image_quality,
            max_long_side=args.max_long_side,
        )
        inner = getattr(env, "unwrapped", env)
        prompt = inner.get_prompt(step_idx=step_idx, max_steps=max_steps)
        vlm_out = vlm.query(img_b64, prompt)
        action = base.Interactor.parse_output(None, vlm_out)
        _obs, reward, terminated, truncated, info = env.step(action)
        next_frame = env.render()
        cumulative_reward += float(reward or 0.0)
        final_info = dict(info or {})
        final_action = action
        step_record = {
            "step": step_idx,
            "prompt": prompt,
            "vlm_output": vlm_out,
            "action": action,
            "reward": float(reward or 0.0),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": base._json_safe(info),
            "input_source": "generated" if args.self_rollout and step_idx > 0 else "env",
        }
        record_pusht_step_artifacts(
            out_episode_dir=out_episode_dir,
            step_idx=step_idx,
            step_record=step_record,
            env_current_frame=frame,
            env_next_frame=next_frame,
            vlm=vlm,
            args=args,
        )
        history.append(step_record)
        if terminated or truncated:
            break

    success = (
        bool(terminated)
        and not bool(truncated)
        and _is_stop_action(final_action)
        and bool(final_info.get("is_success", False))
    )
    result = {
        "env_id": env_id,
        "seed": seed,
        "episode_seed": episode_seed,
        "global_index": int(ep["global_index"]),
        "source_file": ep.get("source_file"),
        "source_line": ep.get("source_line"),
        "rank": rank,
        "timestamp": dt.datetime.now().isoformat(),
        "profile": args.profile,
        "reset_source": args.pusht_reset_source,
        "run_args": {
            "gt_input": not bool(args.self_rollout),
            "self_rollout": bool(args.self_rollout),
            "prompt_source": "env.get_prompt",
            "include_env_feedback": False,
            "elicit_verbalization": False,
            "skip_image_generation": bool(args.skip_image_generation),
            "max_steps": max_steps,
        },
        "history": _light_history(history, len(getattr(vlm, "generated_images", []) or [])) if args.light_output else history,
        "stats": {
            "step": len(history),
            "reward": cumulative_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(success),
            "coverage": float(final_info.get("coverage", 0.0)) if final_info else 0.0,
            "final_action": final_action,
        },
        "version": "v2_pusht_plain_ddp",
    }
    json_file.write_text(json.dumps(base._json_safe(result), indent=2))
    make_pusht_artifact_gifs(out_episode_dir, result, args)
    return {
        "skipped": False,
        "success": bool(success),
        "reward": cumulative_reward,
        "steps": len(history),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "out_dir": str(out_episode_dir),
    }


def _maze2d_step_prompt(step_idx: int, max_steps: int = 40) -> str:
    remaining = max(0, int(max_steps) - int(step_idx))
    return f"This is step {int(step_idx)}. You are allowed to take {remaining} more steps."


def _maze2d_prompt_from_jsonl(ep: dict, env, step_idx: int, max_steps: int = 40) -> str:
    prompts = ep.get("history_prompts") or []
    if step_idx < len(prompts) and isinstance(prompts[step_idx], str) and prompts[step_idx].strip():
        return prompts[step_idx]
    inner = getattr(env, "unwrapped", env)
    if step_idx <= 0:
        return inner.get_prompt()
    return _maze2d_step_prompt(step_idx, max_steps=max_steps)


def run_maze2d_episode(env, vlm, ep: dict, args, rank: int) -> dict:
    if args.self_rollout and args.skip_image_generation:
        raise RuntimeError("Maze2D self-rollout requires image generation; remove --skip-image-generation.")
    env_id = ep["env_id"]
    seed = int(
        ep.get("seed")
        or ep.get("episode_seed")
        or (ep.get("alignment_index") or {}).get("seed")
        or (ep.get("metadata") or {}).get("seed")
        or 0
    )
    episode_seed = int(ep.get("episode_seed") or seed)
    global_index = int((ep.get("alignment_index") or {}).get("global_index", ep.get("global_index", 0)))
    max_steps = 1 if args.sanity_check else base._episode_max_steps(ep, args.max_steps)
    eval_mode = "selfrollout" if args.self_rollout else "gt_input"
    out_episode_dir = (
        Path(args.outdir)
        / base._env_slug(env_id)
        / args.profile
        / eval_mode
        / f"idx{global_index:07d}_seed{seed}_episode{episode_seed}"
    )

    json_file = out_episode_dir / "episode_stats.json"
    if not args.overwrite and json_file.exists():
        try:
            old = json.loads(json_file.read_text())
            if base.looks_complete(old):
                return {"skipped": True, "success": bool(old.get("stats", {}).get("success", False))}
        except Exception:
            pass

    init_state = ep.get("init_state")
    if init_state is None:
        raise RuntimeError(f"{env_id} episode {global_index} has no init_state.")

    if hasattr(vlm, "reset_conversation"):
        vlm.reset_conversation()
    _reset_preserving_init_state(env, init_state=init_state, seed=episode_seed)
    out_episode_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    final_action = None
    terminated = False
    truncated = False
    cumulative_reward = 0.0

    for step_idx in range(max_steps):
        frame = env.render()
        img_b64 = _image_to_b64_for_eval(
            frame,
            image_format=args.image_format,
            image_quality=args.image_quality,
            max_long_side=args.max_long_side,
        )
        prompt = _maze2d_prompt_from_jsonl(ep, env, step_idx, max_steps=max_steps)
        vlm_out = vlm.query(img_b64, prompt)
        action = base.Interactor.parse_output(None, vlm_out)
        _obs, reward, terminated, truncated, info = env.step(action)
        next_frame = env.render()
        cumulative_reward += float(reward or 0.0)
        final_info = dict(info or {})
        final_action = action
        step_record = {
            "step": step_idx,
            "prompt": prompt,
            "vlm_output": vlm_out,
            "action": action,
            "reward": float(reward or 0.0),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": base._json_safe(info),
            "input_source": "generated" if args.self_rollout and step_idx > 0 else "env",
        }
        record_pusht_step_artifacts(
            out_episode_dir=out_episode_dir,
            step_idx=step_idx,
            step_record=step_record,
            env_current_frame=frame,
            env_next_frame=next_frame,
            vlm=vlm,
            args=args,
        )
        history.append(step_record)
        if terminated or truncated:
            break

    success = (
        bool(terminated)
        and not bool(truncated)
        and _is_stop_action(final_action)
        and float(final_info.get("distance", 1.0)) == 0.0
        and cumulative_reward > 0.0
    )
    result = {
        "env_id": env_id,
        "seed": seed,
        "episode_seed": episode_seed,
        "global_index": global_index,
        "source_file": ep.get("source_file"),
        "source_line": ep.get("source_line"),
        "hash": ep.get("hash"),
        "alignment_index": ep.get("alignment_index"),
        "rank": rank,
        "timestamp": dt.datetime.now().isoformat(),
        "profile": args.profile,
        "reset_source": "init_state",
        "run_args": {
            "gt_input": not bool(args.self_rollout),
            "self_rollout": bool(args.self_rollout),
            "prompt_source": "JSONL history[*].prompt",
            "include_env_feedback": False,
            "elicit_verbalization": False,
            "skip_image_generation": bool(args.skip_image_generation),
            "max_steps": max_steps,
        },
        "history": _light_history(history, len(getattr(vlm, "generated_images", []) or [])) if args.light_output else history,
        "stats": {
            "step": len(history),
            "reward": cumulative_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(success),
            "distance": float(final_info.get("distance", 1.0)) if final_info else None,
            "final_action": final_action,
        },
        "version": "v2_maze2d_plain_ddp",
    }
    json_file.write_text(json.dumps(base._json_safe(result), indent=2))
    make_pusht_artifact_gifs(out_episode_dir, result, args)
    return {
        "skipped": False,
        "success": bool(success),
        "reward": cumulative_reward,
        "steps": len(history),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "out_dir": str(out_episode_dir),
    }


def run_episode(env, vlm, extra_state_key: Optional[str], ep: dict, args, rank: int) -> dict:
    if args.profile == "pusht_norm4_native" or is_pusht_env_id(ep.get("env_id", "")):
        return run_pusht_episode(env, vlm, ep, args, rank)
    if args.profile == "maze2d_native256" or is_maze2d_env_id(ep.get("env_id", "")):
        return run_maze2d_episode(env, vlm, ep, args, rank)

    env_id = ep["env_id"]
    seed = int(ep.get("seed") or ep.get("episode_seed") or 0)
    episode_seed = int(ep.get("episode_seed") or seed)
    max_steps = 1 if args.sanity_check else base._episode_max_steps(ep, args.max_steps)
    out_episode_dir = (
        Path(args.outdir)
        / base._env_slug(env_id)
        / args.profile
        / base._gen_tag(args)
        / f"idx{int(ep['global_index']):07d}_seed{seed}_episode{episode_seed}"
    )

    json_file = out_episode_dir / "episode_stats.json"
    if not args.overwrite and json_file.exists():
        try:
            old = json.loads(json_file.read_text())
            if base.looks_complete(old):
                return {"skipped": True, "success": bool(float(old.get("stats", {}).get("reward", 0.0)) > 0.0)}
        except Exception:
            pass

    interactor = base.Interactor(
        vlm,
        env,
        max_steps=max_steps,
        master_seed=seed,
        max_long_side=args.max_long_side,
        image_quality=args.image_quality,
        image_format=args.image_format,
    )
    stats = interactor.run_episode(
        render=args.render,
        verbose=args.verbose,
        save_verbose=args.save_verbose,
        seed=episode_seed,
        extra_state=extra_state_key,
        init_state=ep.get("init_state"),
        elicit_verbalization=args.elicit_verbalization,
        include_env_feedback=args.include_env_feedback,
        num_interaction_history=args.num_interaction_history,
    )

    out_episode_dir.mkdir(parents=True, exist_ok=True)
    artifact_history = interactor.hist
    for i, step in enumerate(artifact_history):
        info = step.get("info")
        if isinstance(info, dict) and "env_feedback" in info and "env_feedback" not in step:
            step["env_feedback"] = info.get("env_feedback")
        if "response" in step and "action_text" not in step:
            step["action_text"] = step.get("response")
        if i < len(vlm.generated_images):
            step["generated_image"] = vlm.generated_images[i]

    # F8 fix: this generic (sokoban-plain) path previously wrote no stats.success /
    # stats.final_action, so a per-episode re-tally reading stats.success mislabeled every
    # sokoban-plain episode as a failure. Derive both from the env outcome (last history
    # step's info), matching render_eval_storyboards_0606.episode_outcome()'s fallback;
    # reward>0 is the stop_then_solved fallback. Other runners already write these.
    _last_step = artifact_history[-1] if artifact_history else {}
    _last_info = (_last_step.get("info") or {}) if isinstance(_last_step, dict) else {}
    _episode_success = _last_info.get("is_success")
    if _episode_success is None:
        _episode_success = _last_info.get("success")
    if _episode_success is None:
        _episode_success = float(stats.get("reward") or 0.0) > 0.0
    _final_action = (
        _last_step.get("action") or _last_step.get("action_text") or _last_step.get("response")
    ) if isinstance(_last_step, dict) else None
    result = {
        "env_id": env_id,
        "seed": seed,
        "episode_seed": episode_seed,
        "global_index": int(ep["global_index"]),
        "source_file": ep.get("source_file"),
        "source_line": ep.get("source_line"),
        "rank": rank,
        "timestamp": dt.datetime.now().isoformat(),
        "profile": args.profile,
        "init_args": interactor.init_args,
        "run_args": interactor._run_args,
        "history": _light_history(artifact_history, len(vlm.generated_images)) if args.light_output else artifact_history,
        "stats": {
            "step": stats.get("step"),
            "reward": stats.get("reward"),
            "terminated": stats.get("terminated"),
            "truncated": stats.get("truncated"),
            "success": bool(_episode_success),
            "final_action": _final_action,
        },
        "version": "v1_gen_eval_ddp",
        "extra_state": stats.get("extra_state", None),
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(base._json_safe(result), f, indent=2)

    if args.save_verbose and artifact_history and not args.light_output:
        with open(out_episode_dir / "history.jsonl", "w", encoding="utf-8") as f:
            for step in artifact_history:
                f.write(json.dumps(base._json_safe(step)) + "\n")

    _save_render_artifacts(env, interactor, result, artifact_history, out_episode_dir, args)

    reward = float(stats.get("reward") or 0.0)
    return {
        "skipped": False,
        "success": reward > 0.0,
        "reward": reward,
        "steps": int(stats.get("step") or 0),
        "terminated": bool(stats.get("terminated")),
        "truncated": bool(stats.get("truncated")),
        "out_dir": str(out_episode_dir),
    }


def prompt_sanity_episode(env, ep: dict) -> dict:
    seed = int(ep.get("seed") or ep.get("episode_seed") or 0)
    episode_seed = int(ep.get("episode_seed") or seed)
    _reset_preserving_init_state(env, init_state=ep.get("init_state"), seed=episode_seed)
    inner = getattr(env, "unwrapped", env)
    prompt = inner.get_prompt()
    return {
        "success": "('stop', 'stop')" in prompt and "stop is optional" not in prompt.lower(),
        "prompt_chars": len(prompt),
    }


def _is_stop_action(final_action) -> bool:
    """F3 fix: canonical stop check. The env ast.literal_eval's the action, so any spelling
    that parses to ('stop','stop') (no-space, double-quotes, ...) actually terminates+solves;
    gate on the parsed value, not the exact string, to avoid false-negatives."""
    import ast
    try:
        return ast.literal_eval(str(final_action)) == ("stop", "stop")
    except Exception:
        return False


def run_shard(args, rank: int, world_size: int, local_rank: int) -> dict:
    shard = prepare_shard_cache(args, rank, world_size)
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    if not shard:
        summary = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": socket.gethostname(),
            "episodes": 0,
            "completed": 0,
            "skipped": 0,
            "successes": 0,
            "errors": 0,
            "error_examples": [],
            "started_at": dt.datetime.now().isoformat(),
            "finished_at": dt.datetime.now().isoformat(),
        }
        write_rank_summary(outdir, rank, summary)
        return summary

    vlm = None
    if not args.prompt_sanity_only:
        staged_path = stage_checkpoint_if_requested(args, local_rank, rank)
        inferencer, _meta = load_bagel_ddp(
            model_path=staged_path,
            local_rank=local_rank,
            max_mem_per_gpu=args.max_mem_per_gpu,
            bagel_base_path=args.bagel_base_path,
            vae_path=args.vae_path,
            vit_config_path=args.vit_config_path,
            vit_max_num_patch_per_side=args.vit_max_num_patch_per_side,
            latent_patch_size=args.latent_patch_size,
            max_latent_size=args.max_latent_size,
            vae_transform_max_size=args.vae_transform_max_size,
            vae_transform_min_size=args.vae_transform_min_size,
            vae_transform_stride=args.vae_transform_stride,
            vit_transform_max_size=args.vit_transform_max_size,
            vit_transform_min_size=args.vit_transform_min_size,
            vit_transform_stride=args.vit_transform_stride,
        )
        vlm_cls = TextOnlyUnifiedVLM if args.skip_image_generation else base.UnifiedVLM
        vlm = vlm_cls(
            inferencer=inferencer,
            max_new_tokens=args.max_new_tokens,
            self_option=args.self_option,
            use_self_rollout=args.self_rollout,
        )

    summary = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "episodes": len(shard),
        "completed": 0,
        "skipped": 0,
        "successes": 0,
        "errors": 0,
        "error_examples": [],
        "started_at": dt.datetime.now().isoformat(),
    }

    env = None
    extra_state_key = None
    active_env_key = None

    for ep_raw in tqdm(shard, desc=f"rank {rank}", disable=not args.progress):
        ep = prepare_sokoban_episode(ep_raw, args)
        env_id = ep.get("env_id")
        seed = int(ep.get("seed") or ep.get("episode_seed") or 0)
        env_spec = base.get_spec_from_episode(ep)
        env_key = base._env_spec_key(env_id, env_spec, seed) + (args.profile,)
        try:
            if env is None or env_key != active_env_key:
                if env is not None:
                    env.close()
                base.set_global_seed(seed)
                env, extra_state_key = make_profiled_env(ep, args)
                active_env_key = env_key

            if args.prompt_sanity_only:
                result = prompt_sanity_episode(env, ep)
            else:
                result = run_episode(env, vlm, extra_state_key, ep, args, rank)

            if result.get("skipped"):
                summary["skipped"] += 1
            else:
                summary["completed"] += 1
            if result.get("success"):
                summary["successes"] += 1
        except Exception as exc:
            summary["errors"] += 1
            err = {
                "global_index": ep_raw.get("global_index"),
                "env_id": ep_raw.get("env_id"),
                "error": repr(exc),
            }
            summary["error_examples"].append(err)
            err_dir = outdir / "_errors" / f"rank{rank:05d}_idx{int(ep_raw.get('global_index', -1)):07d}"
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.txt").write_text("Exception: " + repr(exc) + "\n\n" + traceback.format_exc())
            print(f"[rank {rank}] episode failed: {err}", flush=True)

    if env is not None:
        env.close()
    summary["finished_at"] = dt.datetime.now().isoformat()
    write_rank_summary(outdir, rank, summary)
    return summary


def add_gen_eval_args(parser: argparse.ArgumentParser):
    parser.add_argument("--action-preset", type=str, default=None, choices=list(base.ACTION_PRESETS.keys()))
    parser.add_argument("--action-name-map", type=str, default=None)
    parser.add_argument("--sim2real", action="store_true")
    parser.add_argument("--sim2real-model", type=str, default="nanobana", choices=["nanobana", "identity"])
    parser.add_argument("--sim2real-preset", type=str, default=None, choices=list(base.SIM2REAL_PRESETS.keys()))
    parser.add_argument("--sim2real-prompt", type=str, default=None)


def parse_args():
    p = argparse.ArgumentParser("DDP Sokoban BAGEL gen-eval runner")
    p.add_argument("--episodes-root", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--bagel-path", required=True)
    p.add_argument("--profile", required=True, choices=PROFILE_CHOICES)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--env-filter", nargs="+", default=None)
    p.add_argument("--shard-cache-dir", default=None)
    p.add_argument("--rebuild-shard-cache", action="store_true")
    p.add_argument("--shard-cache-timeout-sec", type=int, default=1800)
    p.add_argument("--stage-timeout-sec", type=int, default=7200)
    p.add_argument("--summary-wait-timeout-sec", type=int, default=21600)

    p.add_argument("--render", action="store_true")
    p.add_argument("--make-gif", action="store_true")
    p.add_argument("--save-generated-images", action="store_true")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-verbose", action="store_true")
    p.add_argument("--sanity-check", action="store_true")
    p.add_argument("--prompt-sanity-only", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-image-generation", action="store_true", default=False)
    p.add_argument("--light-output", dest="light_output", action="store_true", default=True)
    p.add_argument("--full-output", dest="light_output", action="store_false")
    p.add_argument("--elicit-verbalization", action="store_true", default=False)
    p.add_argument("--include-env-feedback", action="store_true", default=False)
    p.add_argument("--num-interaction-history", type=int, default=-1)
    p.add_argument("--max-long-side", type=int, default=None)
    p.add_argument(
        "--image-format",
        choices=["JPEG", "PNG", "JPEG-NONTURBO"],
        default="JPEG",
    )
    # Re-encode quality for observation frames. Default 90 preserves prior pusht/maze2d
    # behavior; v8 sokoban training data is q95, so the sokoban launcher passes 95.
    p.add_argument("--image-quality", type=int, default=90)

    p.add_argument("--make-gif-env-subdir", dest="env_gif_subdir", default="env_gif")
    p.add_argument("--make-gif-generated-subdir", dest="generated_gif_subdir", default="self_gen_gif")
    p.add_argument("--gif-ms-per-frame", type=int, default=800)
    p.add_argument("--gif-start-hold-ms", type=int, default=2500)
    p.add_argument("--gif-end-hold-ms", type=int, default=2000)
    p.add_argument("--gif-max-text-chars", type=int, default=400)
    p.add_argument("--gif-max-width", type=int, default=1024)
    p.add_argument("--gif-preserve-quality", action="store_true")

    p.add_argument("--stage-checkpoint", action="store_true")
    p.add_argument(
        "--local-ckpt-root",
        default=os.environ.get("SLURM_TMPDIR", f"/tmp/{os.environ.get('USER', 'user')}/sokoban_eval_ckpts"),
    )

    p.add_argument("--bagel-base-path", default=DEFAULT_BAGEL_BASE, required=not bool(DEFAULT_BAGEL_BASE))
    p.add_argument("--vae-path", default=None)
    p.add_argument("--vit-config-path", default=None)
    p.add_argument("--max-mem-per-gpu", default="75GiB")
    p.add_argument("--vit-max-num-patch-per-side", type=int, default=70)
    p.add_argument("--latent-patch-size", type=int, default=2)
    p.add_argument("--max-latent-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--self-option", type=str, default="direct", choices=["direct", "generation_first", "generation_last"])
    p.add_argument("--self-rollout", action="store_true", default=False)

    p.add_argument("--vae-transform-max-size", type=int, default=512)
    p.add_argument("--vae-transform-min-size", type=int, default=256)
    p.add_argument("--vae-transform-stride", type=int, default=16)
    p.add_argument("--vit-transform-max-size", type=int, default=336)
    p.add_argument("--vit-transform-min-size", type=int, default=224)
    p.add_argument("--vit-transform-stride", type=int, default=14)

    p.add_argument("--sokoban-action-mode-override", default="move_only")
    p.add_argument("--sokoban-success-mode", default="stop_then_solved")
    p.add_argument("--pusht-reset-source", choices=("init_state", "seed"), default="init_state")
    add_gen_eval_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    summary_session_id()  # Reject ambiguous distributed jobs before loading model assets.
    rank, world_size, local_rank = init_distributed(require_cuda=not args.prompt_sanity_only)
    started = time.time()

    if rank == 0:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        (Path(args.outdir) / "summary.json").unlink(missing_ok=True)
        print(
            f"[main] outdir={args.outdir} world_size={world_size} profile={args.profile} "
            f"episodes_root={args.episodes_root}",
            flush=True,
        )

    try:
        summary = run_shard(args, rank, world_size, local_rank)
        print(f"[rank {rank}] summary={summary}", flush=True)
        # Keep every node's torchrun agent alive until the slowest episode has
        # finished. Otherwise an early node can hit torchrun's five-minute
        # exit-barrier timeout while another node is still doing inference.
        if rank == 0 or os.environ.get("BAGEL_WAIT_ALL_RANK_SUMMARIES") == "1":
            wait_for_rank_summaries(
                Path(args.outdir),
                world_size,
                timeout_sec=args.summary_wait_timeout_sec,
            )
        if rank == 0:
            merged = aggregate_summaries(Path(args.outdir), world_size)
            merged["elapsed_sec"] = round(time.time() - started, 3)
            write_summary(Path(args.outdir), merged)
            print(f"[main] merged summary={merged}", flush=True)
            if merged["status"] != "complete":
                raise RuntimeError(
                    f"Evaluation incomplete: {merged['done']}/{merged['episodes']} episodes "
                    f"completed, {merged['errors']} errors. Inspect summary.json and _errors/ "
                    "before resuming the same run."
                )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
