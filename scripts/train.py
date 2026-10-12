#!/usr/bin/env python3
"""Launch the preserved BAGEL trainer with an explicit local chunk-data recipe."""

import argparse
import uuid

if __package__:
    from ._common import (ROOT, absolute, file_fingerprints, launch, positive_float, positive_int,
                         require_file, runtime_environment, torchrun, validate_base,
                         validate_episodes, validate_output_location, validate_rows)
else:
    from _common import (ROOT, absolute, file_fingerprints, launch, positive_float, positive_int,
                         require_file, runtime_environment, torchrun, validate_base,
                         validate_episodes, validate_output_location, validate_rows)


CONFIGS = {
    "sokoban": "sokoban/train.yaml",
    "pusht": "pusht/train.yaml",
    "maze2d": "maze2d/train.yaml",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=CONFIGS, required=True)
    parser.add_argument("--variant", choices=("cot", "noncot"), required=True,
                        help="Documents the data variant; the supplied rows determine the loss.")
    parser.add_argument("--chunk-size", choices=("1", "3", "5", "10", "inf"), required=True,
                        help="Must match the supplied dataset; this does not remap its rows.")
    parser.add_argument("--model-path", type=absolute, required=True,
                        help="Local vanilla BAGEL directory with configs, tokenizer, VAE and EMA.")
    parser.add_argument("--data-dir", type=absolute, required=True,
                        help="Directory directly containing prepared JSONL or JSONL.gz shards (not recursive).")
    parser.add_argument("--output-dir", type=absolute, required=True,
                        help="New or empty run directory; checkpoints are saved under checkpoints/.")
    parser.add_argument("--dataset-config", type=absolute,
                        help="Optional replacement YAML, e.g. the Sokoban 50k forward-data config.")
    parser.add_argument("--gpus", type=positive_int, default=8)
    parser.add_argument("--token-budget", type=positive_int,
                        help="Default 50000; Maze2D CoT K=1 uses 24000.")
    parser.add_argument("--total-steps", type=positive_int, default=5000,
                        help="Last zero-based step index, inclusive: N executes steps 0 through N (N+1 updates).")
    parser.add_argument("--save-every", type=positive_int,
                        help="Default 2500 for PushT, 1000 otherwise; also save step 500 and the final step.")
    parser.add_argument("--warmup-steps", type=positive_int, default=100)
    parser.add_argument("--lr", type=positive_float, default=2e-5)
    parser.add_argument("--wandb-online", action="store_true", help="Default is local offline logging.")
    parser.add_argument("--run-name", default=None,
                        help="W&B display name; each cold start gets a new run ID and never resumes logging.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without execution; the W&B ID placeholder is replaced on a real launch.")
    args = parser.parse_args()
    config = args.dataset_config or ROOT / "configs" / CONFIGS[args.task]
    budget = args.token_budget or (24000 if (args.task, args.variant, args.chunk_size) ==
                                  ("maze2d", "cot", "1") else 50000)
    save_every = args.save_every or (2500 if args.task == "pusht" else 1000)
    name = args.run_name or f"agima_{args.task}_{args.variant}_k{args.chunk_size}"
    assets = {}
    try:
        require_file(config)
        if not args.dry_run:
            base_files = validate_base(args.model_path, training=True)
            data_files = validate_episodes(args.data_dir, directory_only=True)
            validate_rows(data_files, minimum=args.gpus)
            validate_output_location(args.output_dir, (args.model_path, args.data_dir, config))
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise ValueError("--output-dir must be empty for this cold-start recipe; choose a new directory")
            assets = {"base": file_fingerprints(base_files),
                      "dataset_config": file_fingerprints([config], hash_all=True),
                      "training_data": file_fingerprints(data_files)}
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    command = torchrun(args.gpus, "training/pretrain_unified_navit.py")
    run_id = "dry-run-preview" if args.dry_run else uuid.uuid4().hex
    # The trainer always opens an eval YAML. With probes disabled, reuse this config
    # and data; never override the shared registry with a different eval-data path.
    options = {
        "dataset_config_file": config, "eval_dataset_config_file": config,
        "viz_dataset_config_file": config,
        "train_data_dir": args.data_dir, "train_jsonl_path": args.data_dir,
        "model_path": args.model_path, "resume_from": args.model_path,
        "layer_module": "Qwen2MoTDecoderLayer", "max_latent_size": 64,
        "finetune_from_hf": True, "auto_resume": False,
        "resume_model_only": True, "finetune_from_ema": True,
        "log_every": 1, "lr": args.lr, "warmup_steps": args.warmup_steps,
        "lr_scheduler": "cosine", "num_workers": 1,
        "expected_num_tokens": budget, "max_num_tokens": budget,
        "max_num_tokens_per_sample": budget, "max_buffer_size": 20,
        "num_shard": args.gpus, "visual_und": True, "visual_gen": True,
        "eval_every": 0, "viz_every": 0, "save_every": save_every,
        "extra_save_steps": ",".join(map(str, sorted({500, args.total_steps}))),
        "total_steps": args.total_steps,
        "text_cond_dropout_prob": 0.0, "vae_cond_dropout_prob": 0.0,
        "vit_cond_dropout_prob": 0.0, "ema": 0.993,
        "checkpoint_dir": args.output_dir / "checkpoints",
        "results_dir": args.output_dir / "logs", "wandb_dir": args.output_dir,
        "wandb_project": "agima", "wandb_name": name,
        "wandb_runid": run_id, "wandb_resume": "never",
        "wandb_offline": not args.wandb_online,
    }
    for key, value in options.items():
        command.extend(("--" + key, str(value)))
    try:
        return launch(command, runtime_environment(), args.output_dir, args.dry_run,
                      {"task": args.task, "variant": args.variant, "chunk_size": args.chunk_size,
                       "recipe": "portable chunk SFT; probes disabled; cold start"},
                      gpus=args.gpus, assets=assets)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
