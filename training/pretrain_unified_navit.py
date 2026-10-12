# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# Modified for Agima: portable configuration defaults and distributed input validation.

import functools
import os
import sys
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from data.transforms import ImageTransform
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, PackedDatasetEvalOnce, PackedDatasetEvalPerStep, collate_wrapper
from data.data_utils import add_special_tokens
from data.dataset_info import DATASET_INFO
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from training.train_utils import create_logger, get_latest_ckpt
from training.distributed_checks import check_batch_participation
from training.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)

from modeling.bagel.qwen2_navit import NaiveCache
import gc

@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="configs/sokoban/train.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    train_data_dir: str = field(
        default=None,
        metadata={"help": "Override data_dir for training datasets."}
    )
    train_jsonl_path: str = field(
        default=None,
        metadata={"help": "Override jsonl_path for training datasets."}
    )
    eval_data_dir: str = field(
        default=None,
        metadata={"help": "Override data_dir for evaluation datasets."}
    )
    eval_jsonl_path: str = field(
        default=None,
        metadata={"help": "Override jsonl_path for evaluation datasets."}
    )
    inference_hash_file: str = field(
        default=None,
        metadata={"help": "Path to the inference hash file for filtering."}
    )
    task_name: str = field(
        default=None,
        metadata={"help": "Name of the task for instruction paraphrasing."}
    )
    instructions_dir: str = field(
        default=None,
        metadata={"help": "Path to the instructions directory."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=1,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )
    wandb_dir: str = field(
        default=None,
        metadata={"help": "Directory where wandb metadata and files are stored."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=500,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    extra_save_steps: str = field(
        default="",
        metadata={"help": "Comma-separated extra step numbers to ALSO save a checkpoint at (e.g. '500')."}
    )
    probe_mode: str = field(
        default="ce",
        metadata={"help": "Hard-OOD probe: 'ce' = run_eval_mse_ce (velocity MSE + CE); 'gen50' = run_gen50_probe (eval-matched 50-step generation MSE)."}
    )
    gen50_num_timesteps: int = field(default=50, metadata={"help": "Denoise steps for the gen50 probe."})
    gen50_n_per_rank: int = field(default=16, metadata={"help": "Probe trajectories per rank (collective-aligned)."})
    gen50_timestep_shift: float = field(default=3.0, metadata={"help": "Denoise schedule shift for the gen50 probe; 3.0 matches the deployed CoT eval (gen_image default)."})
    total_steps: int = field(
        default=1000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.993,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: int = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )
    
    # --- evaluation ---
    eval_every: int = field(default=500, metadata={"help": "Run small eval every N steps."})
    num_eval_batches: int = field(default=20, metadata={"help": "Batches to use in eval pass."})
    use_ema_for_eval: bool = field(default=True, metadata={"help": "Evaluate EMA weights instead of raw model."})
    eval_log_dir: str = field(
        default=None,
        metadata={"help": "Directory to dump eval-used-row manifests (rank0). If None, uses checkpoint_dir/eval_used_rows."}
    )
    eval_run_tag: str = field(
        default="",
        metadata={"help": "Optional run tag prefix for eval manifest filenames (e.g., exp name). Step will be appended."}
    )
    viz_every: int = field(default=500, metadata={"help": "Save qualitative viz every N steps (rank 0 only)."})
    viz_n: int = field(default=8, metadata={"help": "How many recon pairs to dump each viz run."})
    viz_outdir: str = field(default="results/viz", metadata={"help": "Where to save qualitative images."})

    eval_dataset_config_file: str = field(
        default=None,
        metadata={"help": "YAML for eval subset; falls back to training dataset_config_file if None."}
    )
    viz_dataset_config_file: str = field(
        default=None,
        metadata={"help": "YAML for viz subset; falls back to eval or train YAML if None."}
    )
    eval_print_n: int = field(default=3, metadata={"help": "Print first N eval sample fingerprints (rank0) for verifying eval-set stability."})
    
    # fsdp save method
    save_ema_only: bool = field(
        default=True,
        metadata={"help": "Save EMA weights only."}
    )
    save_optimizer: bool = field(
        default=False,
        metadata={"help": "Save optimizer weights."}
    )

def make_eval_loader(dataset_config_eval, tokenizer, new_token_ids, data_args, training_args, model_args,
                     eval_log_dir=None, eval_run_tag=None):

    eval_dataset = PackedDatasetEvalPerStep(
        dataset_config_eval,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=0,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=None,
        inference_hash_file=data_args.inference_hash_file,
        task_name=data_args.task_name,
        instructions_dir=data_args.instructions_dir,
        eval_log_dir=eval_log_dir,
        eval_run_tag=eval_run_tag,
    )
    eval_dataset.set_epoch(data_args.data_seed)
    return DataLoader(eval_dataset, batch_size=1, num_workers=0, pin_memory=True, collate_fn=collate_wrapper())

@torch.no_grad()
def run_eval_mse_ce(fsdp_model, ema_model, vae_model, device, training_args, eval_loader, max_batches):
    model_ref = ema_model if training_args.use_ema_for_eval else fsdp_model

    # keep original mode to restore later
    was_training = model_ref.training
    model_ref.train()  # <-- force training path; accepts packed_sequence
    try:
        mse_sum = torch.tensor(0.0, device=device)
        ce_sum  = torch.tensor(0.0, device=device)
        n_mse_tokens = torch.tensor(0, device=device, dtype=torch.long)
        n_ce_tokens  = torch.tensor(0, device=device, dtype=torch.long)
        fp_list = []

        
        it = iter(eval_loader)
        while True:
            try:
                data = next(it)
            except StopIteration:
                break
            data = data.cuda(device).to_dict()
            # eval debug: print first N eval sample fingerprints
            if dist.get_rank() == 0 and len(fp_list) < getattr(training_args, "eval_print_n", 0):
                bidx = data.get("batch_data_indexes", None)
                # bidx often contains row_idx/worker_id/dataset_name from JsonLIterableDataset
                fp_list.append(str(bidx))
                
            # Remove metadata that Bagel.forward does not accept.
            for k in ["batch_data_indexes", "ce_loss_weights"]:
                data.pop(k, None)

            # Optional MSE-only evaluation can skip batches without ViT inputs.
            # if training_args.visual_und and ("vit_token_seqlens" not in data or data["vit_token_seqlens"] is None):
            #     continue
            w = float(data.get("eval_weight", 1.0))

            missing_vit = training_args.visual_und and (("vit_token_seqlens" not in data) or (data["vit_token_seqlens"] is None))
            if missing_vit:
                w = 0.0

            if training_args.visual_gen and "padded_images" in data:
                with torch.no_grad():
                    data["padded_latent"] = vae_model.encode(data.pop("padded_images"))

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                loss_dict = model_ref(**data)

            # Aggregate losses with the training loop's normalization.
            ce_local = loss_dict.get("ce", None)
            mse_local = loss_dict.get("mse", None)

            # --- CE: always accumulate a scalar ---
            if ce_local is None:
                ce_local_sum = torch.tensor(0.0, device=device)
                ce_tokens_local = torch.tensor(0, device=device, dtype=torch.long)
            else:
                # ce_local is typically per-token loss vector; sum it
                ce_local_sum = ce_local.sum() if ce_local.dim() > 0 else ce_local
                ce_tokens_local = torch.tensor(len(data.get("ce_loss_indexes", [])), device=device, dtype=torch.long)

            # MSE: sum per-token losses.
            if mse_local is None:
                mse_local_sum = torch.tensor(0.0, device=device)
                mse_tokens_local = torch.tensor(0, device=device, dtype=torch.long)
            else:
                # mse_local is per-token (and maybe per-dim) -> scalar sum over tokens
                mse_local_sum = mse_local.mean(dim=-1).sum() if mse_local.dim() > 0 else mse_local
                mse_tokens_local = torch.tensor(len(data.get("mse_loss_indexes", [])), device=device, dtype=torch.long)

            ce_sum  += ce_local_sum
            mse_sum += mse_local_sum
            n_ce_tokens  += ce_tokens_local
            n_mse_tokens += mse_tokens_local


        # reduce
        for t in [ce_sum, mse_sum, n_ce_tokens, n_mse_tokens]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        ce_avg  = (ce_sum / n_ce_tokens.clamp(min=1)).item()
        mse_avg = (mse_sum / n_mse_tokens.clamp(min=1)).item()
        if dist.get_rank() == 0 and getattr(training_args, "eval_print_n", 0) > 0:
            print(f"[eval debug] first {len(fp_list)} batch fingerprints:")
            for i, fp in enumerate(fp_list):
                print(f"  fp[{i}]: {fp}")
            print(f"ce_avg: {ce_avg}, mse_avg: {mse_avg}")
        return {"eval/ce": ce_avg, "eval/mse": mse_avg}
    finally:
        # restore original train/eval mode
        if was_training:
            model_ref.train()
        else:
            model_ref.eval()

import os
import torch
import torch.distributed as dist
import wandb
import numpy as np
from PIL import Image

# Bagel uses NaiveCache for KV during forward_inference


def _unwrap_fsdp(m):
    return m.module if hasattr(m, "module") else m

def _to_dict_batch(batch, device):
    # Move SimpleCustomBatch to the device before converting it to a dictionary.
    if hasattr(batch, "cuda"):
        batch = batch.cuda(device)
    if hasattr(batch, "to_dict"):
        batch = batch.to_dict()
    return batch

def _tensor_to_pil(x_chw):
    # x: [3,H,W] float, usually in [-1,1]
    x = x_chw.detach().float().cpu()
    x = (x.clamp(-1, 1) + 1) * 0.5
    x = (x * 255.0).round().byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)

@torch.no_grad()
def dump_visual_viz(
    fsdp_model,
    ema_model,
    vae_model,
    device,
    training_args,
    viz_loader,
    *,
    global_step: int,
    tokenizer=None,
    new_token_ids=None,
    vit_transform=None,
    num_samples: int = 4,
    num_img_timesteps: int = 24,
    outdir: str = None,
):
    """
    Returns:
      - rank0: wandb.Table
      - other ranks: None
    IMPORTANT:
      - All ranks must execute the generation steps (FSDP collectives).
      - Only rank0 creates/logs the wandb.Table.
    """

    # ---- pick which weights to visualize ----
    model_ref = ema_model if getattr(training_args, "use_ema_for_eval", False) else fsdp_model
    model_ref.eval()
    model = _unwrap_fsdp(model_ref)

    # ---- resolve tokenizer / new_token_ids / vit_transform ----
    if tokenizer is None:
        tokenizer = globals().get("tokenizer", None)
    if new_token_ids is None:
        new_token_ids = globals().get("new_token_ids", None)
    if vit_transform is None:
        vit_transform = globals().get("vit_transform", None)

    if tokenizer is None or new_token_ids is None:
        raise RuntimeError("dump_visual_viz needs tokenizer and new_token_ids (pass them in or set as globals).")

    if vit_transform is None:
        # Build the default transform from the model's ViT configuration.
        from data.transforms import ImageTransform
        patch = getattr(getattr(model, "vit_model", None), "config", None)
        patch_size = getattr(patch, "patch_size", 14)
        vit_max = getattr(getattr(model, "config", None), "vit_max_num_patch_per_side", 70)
        vit_transform = ImageTransform(vit_max * patch_size, 224, patch_size)

    # ---- rank logic ----
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    do_log = (rank == 0)

    if outdir is None:
        outdir = os.path.join(getattr(training_args, "output_dir", "."), "viz")
    if do_log:
        os.makedirs(outdir, exist_ok=True)
        table = wandb.Table(columns=[
            "step",
            "gt_image",
            "caption_prompt",
            "gen_caption",
            "t2i_prompt",
            "gen_image",
        ])
    else:
        table = None

    it = iter(viz_loader)
    added = 0

    # oversample a bit in case some batches don't have images
    for _ in range(num_samples * 3):
        if added >= num_samples:
            break
        try:
            batch = next(it)
        except StopIteration:
            break

        batch = _to_dict_batch(batch, device)

        # Skip batches without image inputs.
        if "padded_images" not in batch or batch["padded_images"] is None:
            continue

        gt_img_t = batch["padded_images"][0]  # [3,H,W]
        gt_pil = _tensor_to_pil(gt_img_t)

        # ---- (A) image -> text (caption) ----
        caption_prompt = "Describe this image in one sentence."
        try:
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                gen_caption = model.chat(
                    tokenizer,
                    new_token_ids,
                    vit_transform,
                    [gt_pil],
                    caption_prompt,
                    max_length=128,
                )
        except Exception as e:
            gen_caption = f"[caption failed: {type(e).__name__}: {e}]"

        # ---- (B) text -> image ----
        H, W = gt_img_t.shape[-2], gt_img_t.shape[-1]
        ld = int(getattr(model, "latent_downsample", 16))
        H2 = max(ld, (H // ld) * ld)
        W2 = max(ld, (W // ld) * ld)

        t2i_prompt = f"Generate an image that matches: {gen_caption}"

        gen_pil = None
        try:
            curr_kvlens = [0]
            curr_rope = [0]

            gen_in_text, newlens, newrope = model.prepare_prompts(
                curr_kvlens=curr_kvlens,
                curr_rope=curr_rope,
                prompts=[t2i_prompt],
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )
            gen_in_text = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in gen_in_text.items()}

            from modeling.bagel import NaiveCache
            past = NaiveCache(model.config.llm_config.num_hidden_layers)

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past = model.forward_cache_update_text(past, **gen_in_text)

                gen_in_img = model.prepare_vae_latent(
                    curr_kvlens=newlens,
                    curr_rope=newrope,
                    image_sizes=[(H2, W2)],
                    new_token_ids=new_token_ids,
                )
                gen_in_img = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in gen_in_img.items()}

                unpacked_latent_tokens = model.generate_image(
                    past_key_values=past,
                    num_timesteps=num_img_timesteps,
                    timestep_shift=getattr(model, "timestep_shift", 1.0),
                    cfg_text_scale=1.0,
                    cfg_img_scale=1.0,
                    **gen_in_img,
                )

            # tokens -> latent map -> VAE decode
            p = int(getattr(model, "latent_patch_size", 2))
            C = int(getattr(model, "latent_channel", 16))
            h = H2 // ld
            w = W2 // ld

            tok = unpacked_latent_tokens[0][: (h * w)].to(device=device, dtype=torch.float32)  # [h*w, p*p*C]
            tok = tok.view(h, w, p, p, C)               # [h,w,p,p,C]
            latent = torch.einsum("hwpqc->chpwq", tok)  # [C,h,p,w,p]
            latent = latent.reshape(C, h * p, w * p).unsqueeze(0)  # [1,C,*,*]

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                out_img = vae_model.decode(latent.to(dtype=torch.bfloat16))
            gen_pil = _tensor_to_pil(out_img[0])

        except Exception as e:
            gen_pil = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
            t2i_prompt = f"{t2i_prompt}\n[t2i failed: {type(e).__name__}: {e}]"

        # ---- only rank0 records/logs ----
        if do_log:
            gt_path = os.path.join(outdir, f"step{global_step}_sample{added}_gt.png")
            gen_path = os.path.join(outdir, f"step{global_step}_sample{added}_gen.png")
            gt_pil.save(gt_path)
            gen_pil.save(gen_path)

            table.add_data(
                global_step,
                wandb.Image(gt_pil),
                caption_prompt,
                gen_caption,
                t2i_prompt,
                wandb.Image(gen_pil),
            )
        added += 1

    return table if do_log else None

# mutable counter for filenames
dump_visual_viz.global_step = 0


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Override dataset info if provided
    if data_args.train_data_dir or data_args.train_jsonl_path:
        with open(data_args.dataset_config_file, "r") as stream:
            dataset_meta = yaml.safe_load(stream)
        for grouped_dataset_name, dataset_args_dict in dataset_meta.items():
            for item in dataset_args_dict.get('dataset_names', []):
                if data_args.train_data_dir:
                    DATASET_INFO[grouped_dataset_name][item]['data_dir'] = data_args.train_data_dir
                if data_args.train_jsonl_path:
                    DATASET_INFO[grouped_dataset_name][item]['jsonl_path'] = data_args.train_jsonl_path
    
    if data_args.eval_data_dir or data_args.eval_jsonl_path:
        if training_args.eval_dataset_config_file:
             with open(training_args.eval_dataset_config_file, "r") as stream:
                dataset_meta = yaml.safe_load(stream)
             for grouped_dataset_name, dataset_args_dict in dataset_meta.items():
                for item in dataset_args_dict.get('dataset_names', []):
                    if data_args.eval_data_dir:
                        DATASET_INFO[grouped_dataset_name][item]['data_dir'] = data_args.eval_data_dir
                    if data_args.eval_jsonl_path:
                        DATASET_INFO[grouped_dataset_name][item]['jsonl_path'] = data_args.eval_jsonl_path

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
            dir=training_args.wandb_dir
        )
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args,    allow_val_change=True)
        wandb.config.update(data_args,     allow_val_change=True)
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:  
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            if training_args.finetune_from_hf else model_args.vae_path
        )
        vae_model.to(dtype=torch.bfloat16)

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)


    
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # Setup packed dataloader
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
    dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
    dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
    dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
        inference_hash_file=data_args.inference_hash_file,
        task_name=data_args.task_name,
        instructions_dir=data_args.instructions_dir,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )
    # ------------ build eval/viz loaders (optional configs) -------------
    eval_cfg = training_args.eval_dataset_config_file
    viz_cfg  = training_args.viz_dataset_config_file


    with open(eval_cfg, "r") as stream:
        dataset_meta_eval = yaml.safe_load(stream)
    dataset_config_eval = DataConfig(grouped_datasets=dataset_meta_eval)
    if training_args.visual_und:
        dataset_config_eval.vit_patch_size = model_args.vit_patch_size
        dataset_config_eval.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config_eval.vae_image_downsample = vae_image_downsample
        dataset_config_eval.max_latent_size = model_args.max_latent_size
    dataset_config_eval.text_cond_dropout_prob = 0.0
    dataset_config_eval.vae_cond_dropout_prob = 0.0
    dataset_config_eval.vit_cond_dropout_prob = 0.0

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    ema_model.eval()

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    for curr_step, data in enumerate(train_loader, start=train_step):
        if curr_step > training_args.total_steps:
            break
        # optimizer.zero_grad(set_to_none=True)
        
        # periodic tiny eval (all ranks for correct all-reduce)
        if training_args.eval_every > 0 and curr_step % training_args.eval_every == 0:
            base_dir = training_args.eval_log_dir
            if base_dir is None:
                base_dir = os.path.join(training_args.checkpoint_dir, "eval_used_rows")
            base_tag = training_args.eval_run_tag or training_args.wandb_name or "run"
            step_tag = f"{base_tag}_step{curr_step}"
            print(f"base_dir is {base_dir}, step_tag is {step_tag}")
            eval_loader = make_eval_loader(
                dataset_config_eval, tokenizer, new_token_ids, data_args, training_args, model_args,
                eval_log_dir=base_dir,
                eval_run_tag=step_tag,
            )
            if getattr(training_args, "probe_mode", "ce") == "gen50":
                from training.gen50_probe import run_gen50_probe
                metrics = run_gen50_probe(
                    fsdp_model, ema_model, vae_model if training_args.visual_gen else None,
                    device, training_args, eval_loader,
                    num_timesteps=training_args.gen50_num_timesteps,
                    n_per_rank=training_args.gen50_n_per_rank,
                    timestep_shift=training_args.gen50_timestep_shift,
                )
            else:
                metrics = run_eval_mse_ce(
                    fsdp_model, ema_model, vae_model if training_args.visual_gen else None,
                    device, training_args, eval_loader, training_args.num_eval_batches
                )
            if dist.get_rank() == 0:
                wandb.log(metrics, step=curr_step)
                print(f"[HARD-PROBE] step={curr_step} {metrics}", flush=True)
            del eval_loader, metrics
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)

                
        data = data.cuda(device).to_dict()
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)
        check_batch_participation(data, visual_gen=training_args.visual_gen,
                                  visual_und=training_args.visual_und, device=device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen and "padded_images" in data:
                with torch.no_grad():
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
            loss_dict = fsdp_model(**data)

        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            # assert not training_args.visual_und
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device, dtype=torch.long)

        # len()>0 guard: skip the MSE term for CE-only batches (zero gen targets), else
        # total_mse_tokens==0 -> division by zero -> nan. Falls through to the else (mse=0,
        # not added to loss). Non-empty batches are unaffected (condition unchanged for them).
        if training_args.visual_gen and "mse_loss_indexes" in data and len(data['mse_loss_indexes']) > 0 and loss_dict.get("mse") is not None:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device, dtype=torch.long)

        optimizer.zero_grad()
        # NaN-skip guard: if loss is NaN/inf, skip backward+step to avoid corrupting params.
        # Only active when BAGEL_NAN_SKIP=1. Counts skips across all ranks and prints them.
        loss_is_bad = bool(torch.isnan(loss).any() or torch.isinf(loss).any())
        skip_batch = os.environ.get("BAGEL_NAN_SKIP", "0") == "1" and loss_is_bad
        if skip_batch:
            # do NOT backward; do NOT step optimizer; do NOT step scheduler; do NOT ema update
            if not hasattr(sys.modules[__name__], "_nan_skip_count"):
                sys.modules[__name__]._nan_skip_count = 0
            sys.modules[__name__]._nan_skip_count += 1
            if dist.get_rank() == 0:
                print(f"[NAN_SKIP] step={curr_step} loss={loss.item():.4f} nan_skip_total={sys.modules[__name__]._nan_skip_count}", flush=True)
        else:
            loss.backward()
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)


        # Log loss values:
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = training_args.log_every / (end_time - start_time)
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
            logger.info(message)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()


        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        _extra_save = {int(s) for s in str(training_args.extra_save_steps).split(",") if s.strip().isdigit()}
        if curr_step > 0 and (curr_step % training_args.save_every == 0 or curr_step in _extra_save):
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            dist.gather_object(data_status, gather_list, dst=0)

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=curr_step, 
                model=fsdp_model, 
                ema_model=ema_model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list,
                save_ema_only=training_args.save_ema_only,
                save_optimizer=training_args.save_optimizer,
            )

    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
