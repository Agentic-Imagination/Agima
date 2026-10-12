#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified for Agima: package imports and portable resource configuration.
"""Shared model loading, episode handling and environment construction.

The task entrypoints in this package use these helpers through evaluation.runtime.
Use scripts/evaluate.py for the documented portable evaluation protocol.
"""

# -----------------------------------------------------------------------------
# System env (must be first)
# -----------------------------------------------------------------------------
import os as _os
_os.environ.setdefault("MUJOCO_GL", "egl")
_os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys as _sys
from pathlib import Path as _Path

_os.environ.setdefault("VISGYM_ROOT", str(_Path(__file__).resolve().parents[1]))
_VISGYM_ROOT = _Path(_os.environ["VISGYM_ROOT"]).expanduser().resolve()
if str(_VISGYM_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_VISGYM_ROOT))

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
import gzip
import json
import traceback
import datetime as dt
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from copy import deepcopy

import numpy as np
import torch
import multiprocessing as mp
from tqdm import tqdm
import gymnasium as gym

from gymnasium.vlm.vlm import BaseVLM
from gymnasium.vlm.interactor import Interactor

from evaluation.registry import ENV_REGISTRY
import base64
from io import BytesIO
from PIL import Image

from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

from modeling.bagel import (
    BagelConfig, Bagel,
    Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache  # noqa: F401
from modeling.autoencoder import load_ae
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens

from inferencer import InterleaveInferencer
from safetensors.torch import save_file

from training.fsdp_utils import FSDPCheckpoint
from training.train_utils import *

# Generalization wrappers
from evaluation.environments import (
    ActionNameRemapEnv, Sim2RealEnv, ACTION_PRESETS, SIM2REAL_PRESETS,
    make_nanobana_sim2real_fn, make_identity_style_fn,
)


from evaluation.visualization import (
    save_episode_gif, list_render_images, save_generated_data_gif,
)


# -----------------------------------------------------------------------------
# Helpers (same as ce_only)
# -----------------------------------------------------------------------------
def _json_safe(obj):
    if isinstance(obj, Image.Image):
        return {
            "_type": "PIL.Image",
            "mode": obj.mode,
            "size": list(obj.size),
        }
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _env_slug(env_id: str) -> str:
    return env_id.replace("/", "__").replace(":", "__")


def iter_jsonl_files(root_or_file: str):
    p = Path(root_or_file)
    if p.is_file():
        yield p
        return
    for f in sorted(list(p.rglob("*.jsonl")) + list(p.rglob("*.jsonl.gz"))):
        yield f


def load_episodes(jsonl_path: Path):
    if str(jsonl_path).endswith(".gz"):
        import gzip
        opener = lambda: gzip.open(jsonl_path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(jsonl_path, "r", encoding="utf-8")
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


_INIT_STATE_ENV_KWARG_KEYS = (
    "difficulty",
    "grid_size",
    "food_count",
    "wall_density",
    "max_steps",
    "horizon",
    "tile_size",
)


def _episode_env_kwargs(ep: dict) -> Dict[str, Any]:
    env_kwargs = dict(ep.get("env_kwargs") or {})
    init_state = ep.get("init_state")
    env_id = ep.get("env_id", "")
    # Sokoban uses class-based difficulty (SokobanEasyEnv/SokobanHardEnv) and does not
    # accept difficulty/max_steps/etc. as constructor kwargs — skip the init_state pull
    # for sokoban entirely.
    if env_id.startswith("sokoban/"):
        return env_kwargs
    if isinstance(init_state, dict):
        for key in _INIT_STATE_ENV_KWARG_KEYS:
            if key in init_state and key not in env_kwargs:
                env_kwargs[key] = init_state[key]
    return env_kwargs


def _episode_max_steps(ep: dict, default: int) -> int:
    for source in (ep.get("env_kwargs"), ep.get("init_state")):
        if isinstance(source, dict) and source.get("max_steps") is not None:
            return int(source["max_steps"])
    return int(default)


def _env_spec_key(env_id: str, env_spec: dict, seed: int) -> tuple:
    seed_key = env_spec.get("seed_key")
    seed_value = int(seed) if seed_key is not None else None
    return (
        env_id,
        json.dumps(env_spec.get("env_kwargs", {}), sort_keys=True, default=str),
        seed_key,
        seed_value,
    )


def _resolve_checkpoint_path(model_path: str) -> str:
    model_dir = Path(model_path)
    candidates = [
        model_dir / "ema.safetensors",
        model_dir / "ema.safetensors.index.json",
        model_dir / "model.safetensors",
        model_dir / "model.pt",
        model_dir / "pytorch_model.bin",
        model_dir / "model.safetensors.index.json",
        model_dir / "pytorch_model.bin.index.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    extras = []
    if model_dir.exists():
        extras = sorted(
            p.name for p in model_dir.iterdir()
            if p.is_file() and (
                p.suffix in {".pt", ".bin", ".json", ".safetensors"}
                or p.name.endswith(".index.json")
            )
        )
    raise FileNotFoundError(
        f"No checkpoint weights found under {model_dir}. "
        f"Looked for {[p.name for p in candidates]}. "
        f"Directory files: {extras or '<<missing or empty directory>>'}"
    )


# -----------------------------------------------------------------------------
# Bagel loader (same as ce_only)
# -----------------------------------------------------------------------------
def _normalize_keys(sd):
    out = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("_fsdp_wrapped_module."):
            k2 = k2[len("_fsdp_wrapped_module."):]
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        out[k2] = v
    return out


def load_bagel(
    model_path: str,
    max_mem_per_gpu: str = "30GiB",
    vit_max_num_patch_per_side: int = 70,
    latent_patch_size: int = 2,
    max_latent_size: int = 64,
):
    model_path = str(model_path)
    configs_path = _os.environ.get("BAGEL_BASE_PATH")
    if not configs_path:
        raise ValueError("Set BAGEL_BASE_PATH to the local BAGEL config, tokenizer and VAE directory.")
    print(f"[load_bagel] model_path={model_path}", flush=True)
    print(f"[load_bagel] configs_path={configs_path}", flush=True)

    print("[load_bagel] loading llm config", flush=True)
    llm_config = Qwen2Config.from_json_file(Path(configs_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = getattr(
            llm_config,
            "bos_token_id",
            getattr(llm_config, "eos_token_id", 0),
        )

    print("[load_bagel] loading vit config", flush=True)
    vit_config = SiglipVisionConfig.from_json_file(Path(configs_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    print("[load_bagel] loading ae", flush=True)
    vae_model, vae_config = load_ae(local_path=Path(configs_path, "ae.safetensors"))

    print("[load_bagel] building bagel config", flush=True)
    config = BagelConfig(
        visual_gen=True, visual_und=True,
        llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=vit_max_num_patch_per_side,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=latent_patch_size,
        max_latent_size=max_latent_size,
    )

    print("[load_bagel] loading tokenizer", flush=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(configs_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    ckpt_path = _resolve_checkpoint_path(model_path)
    print(f"[load_bagel] checkpoint_path={ckpt_path}", flush=True)
    print("[load_bagel] instantiating meta model", flush=True)
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        bagel = Bagel(language_model, vit_model, config)
        bagel.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    print("[load_bagel] creating transforms", flush=True)
    # vae_transform = ImageTransform(512, 256, 16)
    # vit_transform = ImageTransform(336, 224, 14)
    vae_transform = ImageTransform(256, 256, 16)
    vit_transform = ImageTransform(224, 224, 14)
    print("[load_bagel] inferring device map", flush=True)
    device_map = infer_auto_device_map(
        bagel,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=[
            "Bagel", "Qwen2MoTDecoderLayer",
            "SiglipEncoderLayer", "SiglipVisionTransformer",
            "MLPconnector", "TimestepEmbedder", "PositionEmbedding",
        ],
    )
    print(device_map, flush=True)

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder", "latent_pos_embed", "vae2llm", "llm2vae", "connector", "vit_pos_embed",
    ]
    if torch.cuda.device_count() == 1:
        first_device = "cuda:0"
        for k in same_device_modules:
            device_map[k] = first_device
    else:
        first_device = device_map.get(same_device_modules[0], next(iter(device_map.values())))
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    print("[load_bagel] dispatching checkpoint", flush=True)
    try:
        bagel = load_checkpoint_and_dispatch(
            bagel,
            checkpoint=ckpt_path,
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to dispatch checkpoint {ckpt_path} for model_path={model_path}"
        ) from exc

    print("[load_bagel] creating inferencer", flush=True)
    inferencer = InterleaveInferencer(
        model=bagel,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    meta = dict(
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        device_map=device_map,
    )
    print("[load_bagel] ready", flush=True)
    return inferencer, meta


# -----------------------------------------------------------------------------
# VLM wrapper (same as ce_only)
# -----------------------------------------------------------------------------
IMG_GENERATION_PROMPT = "First, generate an image of what you think the next state would be to achieve the goal. Do not copy the previous image."
TEXT_GENERATION_PROMPT = "Then, generate the action to reach the next state that you imagined. Action Only:"
IMG_GENERATION_PROMPT_LAST = "Generate an image of what you think the next state would be after taking the action."


class UnifiedVLM(BaseVLM):
    def __init__(self, inferencer: InterleaveInferencer, max_new_tokens: int = 512, self_option: str = "direct",
                 use_self_rollout: bool = False):
        super().__init__()
        self.inferencer = inferencer
        self.max_new_tokens = max_new_tokens
        self.conversation_history = []
        self.conversation_history_for_image_gen = []
        self.idx = 0
        self.option = self_option
        self.use_self_rollout = use_self_rollout
        self._last_generated_img = None
        self.env_image_size: Optional[Tuple[int, int]] = None
        self.generated_images = []
        self.last_input_source = None
        self.last_input_image = None
        self.last_generated_image = None

    def reset_conversation(self):
        self.conversation_history = []
        self.conversation_history_for_image_gen = []
        self._last_generated_img = None
        self.env_image_size = None
        self.generated_images = []
        self.last_input_source = None
        self.last_input_image = None
        self.last_generated_image = None
        self.idx = 0

    def query(
        self,
        image_base64: str,
        prompt: str,
        final_goal_prompt: Optional[str] = None,
        final_goal_img_b64: Optional[str] = None,
    ) -> str:
        print(f"[vlm] step={self.idx} starting query", flush=True)
        if self.use_self_rollout and self.idx > 0:
            # Self-rollout: use model's previously generated image instead of real frame
            if self._last_generated_img is None:
                raise RuntimeError("Self-rollout requested a generated observation, but no previous generated image exists.")
            image = self._last_generated_img
            self.last_input_source = "generated"
            self.conversation_history.append(image)
        else:
            # Normal: use the real/sim2real image from env
            image = Image.open(BytesIO(base64.b64decode(image_base64)))
            if self.env_image_size is None:
                self.env_image_size = image.size
            self.last_input_source = "env"
            self.conversation_history.append(image)
        self.last_input_image = image

        if final_goal_img_b64:
            final_goal_image = Image.open(BytesIO(base64.b64decode(final_goal_img_b64)))
            self.conversation_history.append(final_goal_image)
        if final_goal_prompt:
            self.conversation_history.append(final_goal_prompt)
        self.conversation_history.append(prompt)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            print(f"[vlm] step={self.idx} generating text", flush=True)
            output_text = self.inferencer(interleaved_input=self.conversation_history, understanding_output=True, vae=True)
            output_text = output_text["text"][0]
            output_text = output_text.split('\n')[0]
            if ':' in output_text:
                output_text = output_text.split(':')[1]
            output_text = output_text.strip('`')
            print(f"[vlm] step={self.idx} action_text={output_text!r}", flush=True)

            conversation_history_for_image_gen = deepcopy(self.conversation_history)
            conversation_history_for_image_gen.append(output_text)
            print(f"[vlm] step={self.idx} generating image", flush=True)
            output_img = self.inferencer(interleaved_input=conversation_history_for_image_gen, understanding_output=False)
            output_img = output_img["image"][0]
            if self.use_self_rollout and self.env_image_size is not None and output_img.size != self.env_image_size:
                output_img = output_img.resize(self.env_image_size, Image.Resampling.LANCZOS)
            self._last_generated_img = output_img
            self.last_generated_image = output_img
            buf = BytesIO()
            output_img.convert("RGB").save(buf, format="JPEG")
            self.generated_images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
            self.idx += 1

        self.conversation_history.append(output_text)
        print(f"[vlm] step={self.idx - 1} query complete", flush=True)
        return output_text


# -----------------------------------------------------------------------------
# Env factory
# -----------------------------------------------------------------------------

def get_spec_from_episode(ep: dict) -> Dict[str, Any]:
    extra_state = ep.get("extra_state", None)
    if extra_state is not None and not isinstance(extra_state, str):
        extra_state = None
    return {
        "env_kwargs": _episode_env_kwargs(ep),
        "seed_key": ep.get("seed_key", None),
        "extra_state": extra_state,
    }


_FILTER_KWARGS_WARNED: set = set()


def _filter_env_kwargs(env_id: str, env_kwargs: dict) -> dict:
    try:
        import inspect, importlib
        spec = gym.spec(env_id)
        ep = spec.entry_point
        if isinstance(ep, str):
            mod_name, _, cls_name = ep.partition(":")
            cls = getattr(importlib.import_module(mod_name), cls_name)
        else:
            cls = ep
        init_fn = None
        for base in cls.__mro__:
            if "__init__" in base.__dict__:
                init_fn = base.__init__
                break
        accepted = set()
        if init_fn is not None:
            for p in inspect.signature(init_fn).parameters.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    return env_kwargs
                if p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    accepted.add(p.name)
        accepted.discard("self")
        if not accepted:
            return env_kwargs
        dropped = {k: v for k, v in env_kwargs.items() if k not in accepted}
        if dropped and env_id not in _FILTER_KWARGS_WARNED:
            print(f"[make_env] {env_id}: dropping unsupported env_kwargs {dropped}", flush=True)
            _FILTER_KWARGS_WARNED.add(env_id)
        return {k: v for k, v in env_kwargs.items() if k in accepted}
    except Exception as exc:
        print(f"[make_env] kwarg filter failed for {env_id}: {exc}", flush=True)
        return env_kwargs


def make_env_from_registry(env_id: str, env_spec: dict, episode_seed: int):
    spec = env_spec
    if spec is None:
        env = gym.make(env_id)
        return env, None, None

    env_kwargs = dict(spec.get("env_kwargs", {}))
    seed_key = spec.get("seed_key", None)
    if seed_key is not None:
        env_kwargs[seed_key] = episode_seed

    extra_state_key = spec.get("extra_state")
    if extra_state_key is None:
        extra_state_key = ENV_REGISTRY.get(env_id, {}).get("extra_state")

    env_kwargs = _filter_env_kwargs(env_id, env_kwargs)
    env = gym.make(env_id, **env_kwargs)
    return env, seed_key, extra_state_key


def apply_gen_eval_wrappers(env, args):
    """
    Wrap `env` with generalization-eval wrappers according to CLI args.

    Order: ActionNameRemap is applied last so that sim2real does not
    accidentally modify prompt text that was already substituted.

    Action-name remap is only meaningful for ref_dot and colorization.
    Jigsaw is intentionally excluded from these presets.
    """
    # 1. Sim2Real
    if args.sim2real:
        if args.sim2real_model == "nanobana":
            # Priority: explicit --sim2real-prompt > --sim2real-preset > default mujoco prompt
            if args.sim2real_prompt:
                prompt = args.sim2real_prompt
            elif args.sim2real_preset:
                prompt = SIM2REAL_PRESETS[args.sim2real_preset]
            else:
                prompt = SIM2REAL_PRESETS["mujoco"]
            style_fn = make_nanobana_sim2real_fn(prompt=prompt)
            print(f"[gen_eval] Sim2Real prompt preset: '{args.sim2real_preset or 'mujoco'}'")
        else:  # identity
            style_fn = make_identity_style_fn()
        env = Sim2RealEnv(env, style_fn=style_fn)
        print(f"[gen_eval] Sim2Real wrapper applied (model={args.sim2real_model})")

    # 2. Action-name remap — resolve preset or raw JSON map
    name_map = None
    if args.action_preset:
        name_map = ACTION_PRESETS[args.action_preset]
        print(f"[gen_eval] Using preset '{args.action_preset}': {name_map}")
    elif args.action_name_map:
        name_map = json.loads(args.action_name_map)

    if name_map:
        env = ActionNameRemapEnv(env, name_map=name_map)
        print(f"[gen_eval] ActionNameRemap wrapper applied: {name_map}")

    return env


# -----------------------------------------------------------------------------
# Episode runner (same as ce_only)
# -----------------------------------------------------------------------------

def save_step_render(render_img, out_dir: Path, step_idx: int):
    try:
        import matplotlib.pyplot as plt
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"step_{step_idx:04d}.png"
        plt.imsave(path.as_posix(), render_img)
        plt.close()
    except Exception:
        pass


def save_step_pil(render_img: Image.Image, out_dir: Path, step_idx: int):
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"step_{step_idx:04d}.png"
        render_img.convert("RGB").save(path.as_posix())
    except Exception:
        pass


def _render_like_to_array(render_img):
    if isinstance(render_img, Image.Image):
        return np.array(render_img.convert("RGB"))
    if isinstance(render_img, str):
        data = render_img.strip()
        if data.startswith("data:image"):
            try:
                _, data = data.split(",", 1)
            except ValueError:
                return None
        try:
            return np.array(Image.open(BytesIO(base64.b64decode(data))).convert("RGB"))
        except Exception:
            return None
    try:
        arr = np.array(render_img)
    except Exception:
        return None
    if arr.ndim == 0:
        return None
    return arr


def set_global_seed(seed: int):
    random.seed(seed)
    # NumPy's legacy RandomState accepts only 32-bit seeds. Some published
    # PushT evaluation episode IDs exceed that range; preserve the full seed
    # for Python/PyTorch while mapping only NumPy's seed into its valid range.
    np.random.seed(int(seed) % (2**32))
    _os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_inference_on_episode(
    env,
    vlm: UnifiedVLM,
    extra_state_key: Optional[str],
    env_id: str,
    seed: int,
    episode_seed: int,
    render: bool,
    max_steps: int,
    verbose: bool,
    save_verbose: bool,
    out_episode_dir: Path,
    max_long_side: Optional[int],
    elicit_verbalization: bool,
    include_env_feedback: bool,
    num_interaction_history: int,
    init_state: Optional[Dict[str, Any]] = None,
    make_gif: bool = False,
    gif_ms_per_frame: int = 300,
    gif_start_hold_ms: int = 2500,
    gif_end_hold_ms: int = 2000,
    gif_max_text_chars: int = 400,
    gif_max_width: int = 1024,
    gif_preserve_quality: bool = True,
    env_gif_subdir: str = "env_gif",
    generated_gif_subdir: str = "self_gen_gif",
    image_format: str = "JPEG",
):
    interactor = Interactor(vlm, env, max_steps=max_steps, master_seed=seed, max_long_side=max_long_side, image_quality=90, image_format=image_format)

    stats = interactor.run_episode(
        render=render,
        verbose=verbose,
        save_verbose=save_verbose,
        seed=episode_seed,
        extra_state=extra_state_key,
        init_state=init_state,
        elicit_verbalization=elicit_verbalization,
        include_env_feedback=include_env_feedback,
        num_interaction_history=num_interaction_history,
    )

    out_episode_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "env_id": env_id,
        "seed": seed,
        "episode_seed": episode_seed,
        "timestamp": dt.datetime.now().isoformat(),
        "init_args": interactor.init_args,
        "run_args": interactor._run_args,
        "history": interactor.hist,
        "stats": {
            "step": stats.get("step"),
            "reward": stats.get("reward"),
            "terminated": stats.get("terminated"),
            "truncated": stats.get("truncated"),
        },
        "version": "v1_gen_eval",
        "extra_state": stats.get("extra_state", None),
    }

    for i, step in enumerate(result["history"]):
        info = step.get("info")
        if isinstance(info, dict) and "env_feedback" in info and "env_feedback" not in step:
            step["env_feedback"] = info.get("env_feedback")
        if isinstance(step, dict) and "response" in step and "action_text" not in step:
            step["action_text"] = step.get("response")
        if i < len(vlm.generated_images):
            step["generated_image"] = vlm.generated_images[i]

    with open((out_episode_dir / "episode_stats.json").as_posix(), "w") as f:
        json.dump(_json_safe(result), f, indent=2)

    if save_verbose and result["history"]:
        with open((out_episode_dir / "history.jsonl").as_posix(), "w") as f:
            for step in result["history"]:
                f.write(json.dumps(step) + "\n")

    if (render or make_gif) and interactor.hist:
        for i, step in enumerate(interactor.hist):
            img = None
            for k in ("image", "frame", "obs", "render", "rgb"):
                if k in step:
                    img = step[k]
                    break
            if img is None:
                try:
                    img = env.render(mode="rgb_array")
                except Exception:
                    img = None
            if img is not None:
                arr = _render_like_to_array(img)
                if arr is not None:
                    save_step_render(arr, out_episode_dir / "renders", i)
        img = None
        try:
            img = env.render(mode="rgb_array")
        except TypeError:
            try:
                img = env.render()
            except Exception:
                img = None
        except Exception:
            img = None
        if img is not None:
            arr = _render_like_to_array(img)
            if arr is not None:
                save_step_render(arr, out_episode_dir / "renders", len(interactor.hist))

    if result["history"]:
        for i, step in enumerate(result["history"]):
            encoded = step.get("generated_image")
            if not encoded:
                continue
            try:
                img = Image.open(BytesIO(base64.b64decode(encoded)))
            except Exception:
                continue
            save_step_pil(img, out_episode_dir / "generated_renders", i)

    if make_gif:
        try:
            frame_paths = list_render_images(out_episode_dir)
            has_generated_frames = any(step.get("generated_image") for step in result["history"])
            reward_val = result.get("stats", {}).get("reward", None)
            reward_str = f"{float(reward_val):.3f}" if isinstance(reward_val, (int, float)) else "NA"
            tag = f"seed{seed}_episode{episode_seed}"
            env_slug_dir = out_episode_dir.parent
            env_gif_dir = env_slug_dir / env_gif_subdir
            env_gif_dir.mkdir(parents=True, exist_ok=True)
            out_gif = env_gif_dir / f"{tag}_env_r{reward_str}.gif"
            save_episode_gif(
                episode_data=result,
                frame_paths=frame_paths,
                out_gif=out_gif,
                ms_per_frame=gif_ms_per_frame,
                start_hold_ms=gif_start_hold_ms,
                end_hold_ms=gif_end_hold_ms,
                max_text_chars=gif_max_text_chars,
                max_width=gif_max_width,
                min_width=512,
                captions=True,
                preserve_quality=gif_preserve_quality,
            )
            if has_generated_frames:
                generated_gif_dir = env_slug_dir / generated_gif_subdir
                generated_gif_dir.mkdir(parents=True, exist_ok=True)
                gen_out_gif = generated_gif_dir / f"{tag}_generated_r{reward_str}.gif"
                save_generated_data_gif(
                    episode_data=result,
                    frame_paths=[],
                    out_gif=gen_out_gif,
                    ms_per_frame=gif_ms_per_frame,
                    start_hold_ms=gif_start_hold_ms,
                    end_hold_ms=gif_end_hold_ms,
                    max_text_chars=gif_max_text_chars,
                    max_width=gif_max_width,
                    captions=True,
                    preserve_quality=gif_preserve_quality,
                )
        except Exception:
            pass

    return result


# -----------------------------------------------------------------------------
# File runner
# -----------------------------------------------------------------------------

def looks_complete(episode_json: Dict[str, Any]) -> bool:
    hist = episode_json.get("history") or []
    return len(hist) > 0


def run_inference_on_file(jsonl_path: Path, args, out_root: Path, vlm: UnifiedVLM):
    ok = 0
    env = None
    extra_state_key = None
    active_env_key = None

    for i, ep in enumerate(tqdm(load_episodes(jsonl_path))):
        if args.test_mode and i > 10:
            break
        if args.hard_mode and i >= 35:
            print(f"Hard mode: {i} episodes processed!")
            break

        env_id = ep.get("env_id")
        seed = ep.get("seed")
        episode_seed = ep.get("episode_seed")
        if env_id is None or seed is None:
            seed = seed or ep.get("episode_seed")
        if env_id is None or seed is None:
            continue

        if args.env_filter and (env_id not in args.env_filter):
            continue

        env_spec = get_spec_from_episode(ep)
        env_key = _env_spec_key(env_id, env_spec, int(seed))
        if env is None or env_key != active_env_key:
            set_global_seed(int(seed))
            base_env, _seed_key, extra_state_key = make_env_from_registry(env_id, env_spec, int(seed))
            env = apply_gen_eval_wrappers(base_env, args)
            active_env_key = env_key

        env_slug = _env_slug(env_id)
        # Tag the output dir with the active wrappers so results don't collide
        # with standard runs
        gen_tag = _gen_tag(args)
        out_dir = out_root / env_slug / gen_tag / f"seed{seed}_episode{episode_seed}"

        init_state = ep.get("init_state")
        max_steps = 1 if args.sanity_check else _episode_max_steps(ep, args.max_steps)
        json_file = out_dir / "episode_stats.json"
        if json_file.exists():
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                if looks_complete(data):
                    print(f"Episode {out_dir} complete, skipping.")
                    env.reset(init_state=init_state, seed=int(episode_seed))
                    continue
            except Exception:
                pass

        try:
            _ = run_inference_on_episode(
                env=env,
                vlm=vlm,
                extra_state_key=extra_state_key,
                env_id=env_id,
                seed=int(seed),
                episode_seed=int(episode_seed),
                init_state=init_state,
                render=args.render,
                max_steps=max_steps,
                verbose=args.verbose,
                save_verbose=args.save_verbose,
                out_episode_dir=out_dir,
                max_long_side=args.max_long_side,
                elicit_verbalization=args.elicit_verbalization,
                include_env_feedback=args.include_env_feedback,
                num_interaction_history=args.num_interaction_history,
                make_gif=args.make_gif,
                gif_ms_per_frame=args.gif_ms_per_frame,
                gif_start_hold_ms=args.gif_start_hold_ms,
                gif_end_hold_ms=args.gif_end_hold_ms,
                gif_max_text_chars=args.gif_max_text_chars,
                gif_max_width=args.gif_max_width,
                gif_preserve_quality=args.gif_preserve_quality,
                image_format=args.image_format,
            )
            ok += 1
            print(f"Saved episode {ok} to {out_dir}")
        except Exception as e:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open((out_dir / "error.txt").as_posix(), "w") as f:
                f.write("Exception: " + repr(e) + "\n\n")
                f.write(traceback.format_exc())
            print(f"Episode failed → {out_dir}/error.txt")


def _gen_tag(args) -> str:
    """Build a short directory tag describing active wrappers."""
    parts = []
    if args.action_name_map:
        name_map = json.loads(args.action_name_map)
        alias = "_".join(f"{k}4{v}" for k, v in name_map.items())
        parts.append(f"rename_{alias}")
    if args.sim2real:
        parts.append(f"sim2real_{args.sim2real_model}")
    return "_".join(parts) if parts else "baseline"


# -----------------------------------------------------------------------------
# Worker init & main
# -----------------------------------------------------------------------------

def _init_worker(ns, out_root_str, bagel_path: str):
    global ARGS, OUT_ROOT, VLM
    ARGS = ns
    OUT_ROOT = Path(out_root_str)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"[worker] pid={_os.getpid()} visible_devices={_os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"cuda_device_count={torch.cuda.device_count()}",
        flush=True,
    )

    inferencer, _meta = load_bagel(
        model_path=bagel_path,
        max_mem_per_gpu=ARGS.max_mem_per_gpu,
        vit_max_num_patch_per_side=ARGS.vit_max_num_patch_per_side,
        latent_patch_size=ARGS.latent_patch_size,
        max_latent_size=ARGS.max_latent_size,
    )
    VLM = UnifiedVLM(inferencer=inferencer, max_new_tokens=ARGS.max_new_tokens, self_option=ARGS.self_option,
                     use_self_rollout=ARGS.self_rollout)


def _run_file(jsonl_path: Path):
    print(f"[run] starting file {jsonl_path}", flush=True)
    run_inference_on_file(jsonl_path, ARGS, OUT_ROOT, VLM)
    print(f"[run] finished file {jsonl_path}", flush=True)


def main():
    p = argparse.ArgumentParser("Bagel + VLM-Gym Generalization Eval Runner")
    p.add_argument("--episodes-root", required=True)
    p.add_argument("--outdir", default="runs/gen_eval")
    p.add_argument("--bagel-path", required=True)

    # Core episode controls (same as ce_only)
    p.add_argument("--render", action="store_true")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--test-mode", action="store_true")
    p.add_argument("--hard-mode", action="store_true")
    p.add_argument("--elicit-verbalization", action="store_true", default=False)
    p.add_argument("--include-env-feedback", action="store_true", default=False)
    p.add_argument("--num-interaction-history", type=int, default=-1)
    p.add_argument("--save-verbose", action="store_true")
    p.add_argument("--sanity-check", action="store_true")
    p.add_argument("--env-filter", nargs="+", default=None)
    p.add_argument("--max-long-side", type=int, default=336)

    # GIF
    p.add_argument("--make-gif", action="store_true")
    p.add_argument("--gif-ms-per-frame", type=int, default=800)
    p.add_argument("--gif-start-hold-ms", type=int, default=2500)
    p.add_argument("--gif-end-hold-ms", type=int, default=2000)
    p.add_argument("--gif-max-text-chars", type=int, default=400)
    p.add_argument("--gif-max-width", type=int, default=1024)
    p.add_argument("--gif-preserve-quality", action="store_true")
    p.add_argument("--image-format", choices=["JPEG", "PNG", "JPEG-NONTURBO"], default="JPEG",
                   help="Image format for env→b64 path. JPEG=libjpeg-turbo (default). "
                        "PNG=lossless. JPEG-NONTURBO=route encode via nonturbo IJG libjpeg "
                        "subprocess (byte-matches training data distribution).")

    # Bagel loader
    p.add_argument("--max-mem-per-gpu", dest="max_mem_per_gpu", default="30GiB")
    p.add_argument("--vit-max-num-patch-per-side", type=int, default=70)
    p.add_argument("--latent-patch-size", type=int, default=2)
    p.add_argument("--max-latent-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--self-option", type=str, default="direct",
                   choices=["direct", "generation_first", "generation_last"])
    p.add_argument("--self-rollout", action="store_true", default=False,
                   help="Use self-rollout: first frame is real/sim2real, subsequent frames use model's generated image.")

    # ------------------------------------------------------------------ #
    # Generalization eval axes                                            #
    # ------------------------------------------------------------------ #
    preset_choices = list(ACTION_PRESETS.keys())
    p.add_argument(
        "--action-preset",
        type=str,
        default=None,
        choices=preset_choices,
        help=(
            "Pre-defined action-name remap preset. Intended for ref_dot and colorization only.\n"
            f"Choices: {preset_choices}\n"
            "  ref_dot_click            – mark→click\n"
            "  colorization_rename      – rotate→hue_shift, saturate→adjust_saturation\n"
            "  ref_dot_and_colorization – both combined\n"
            "Use together with --env-filter to restrict to the relevant env(s)."
        ),
    )
    p.add_argument(
        "--action-name-map",
        type=str,
        default=None,
        help=(
            'Custom JSON dict mapping new action name → original name.\n'
            'e.g. \'{"click":"mark"}\'\n'
            'Overridden by --action-preset if both are given.'
        ),
    )
    p.add_argument(
        "--sim2real",
        action="store_true",
        help="Enable sim2real rendering: pass each rendered frame through a style-transfer model.",
    )
    p.add_argument(
        "--sim2real-model",
        type=str,
        default="nanobana",
        choices=["nanobana", "identity"],
        help=(
            "Style-transfer backend.\n"
            "  nanobana – Gemini 3.1 Flash via OpenRouter (requires OPENROUTER_API_KEY)\n"
            "  identity – no transformation (sanity check)"
        ),
    )
    p.add_argument(
        "--sim2real-preset",
        type=str,
        default=None,
        choices=list(SIM2REAL_PRESETS.keys()),
        help=(
            "Named sim2real prompt preset.\n"
            "  mujoco – robot arm / lab scene (default)\n"
            "  cube   – 3D cube-block objects on white background\n"
            "Overridden by --sim2real-prompt if both are given."
        ),
    )
    p.add_argument(
        "--sim2real-prompt",
        type=str,
        default=None,
        help="Override the sim2real prompt sent to NanoBanana (takes priority over --sim2real-preset).",
    )

    args = p.parse_args()

    out_root_path = Path(args.outdir)
    out_root_path.mkdir(parents=True, exist_ok=True)

    jsonl_paths = list(iter_jsonl_files(args.episodes_root))
    if not jsonl_paths:
        raise FileNotFoundError(f"No JSONL files found under {args.episodes_root}")

    if args.sanity_check:
        args.max_long_side = None

    print(
        f"[main] jsonl_files={len(jsonl_paths)} outdir={out_root_path} "
        f"visible_devices={_os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    if len(jsonl_paths) == 1:
        _init_worker(args, str(out_root_path), args.bagel_path)
        _run_file(jsonl_paths[0])
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=1,
            initializer=_init_worker,
            initargs=(args, str(out_root_path), args.bagel_path),
        ) as pool:
            _ = list(pool.imap_unordered(_run_file, jsonl_paths))

    print(f"\nAll done. Outputs under {out_root_path}")


if __name__ == "__main__":
    main()
