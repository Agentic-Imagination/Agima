#!/usr/bin/env python3
"""Launch the preserved task-specific chunk evaluator with explicit assets and decoder."""

import argparse

if __package__:
    from ._common import (absolute, file_fingerprints, launch, positive_int, runtime_environment,
                         torchrun, validate_base, validate_checkpoint, validate_episodes,
                         validate_output_location, validate_rows)
else:
    from _common import (absolute, file_fingerprints, launch, positive_int, runtime_environment,
                         torchrun, validate_base, validate_checkpoint, validate_episodes,
                         validate_output_location, validate_rows)


TASKS = {
    "sokoban": ("evaluation.sokoban", "sokoban_v8_res256", 512, 336, 95, 40),
    "pusht": ("evaluation.pusht", "pusht_norm4_native", 256, 224, 90, 30),
    "maze2d": ("evaluation.maze2d", "maze2d_native256", 256, 224, 95, 40),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--variant", choices=("cot", "noncot"), required=True)
    parser.add_argument("--chunk-size", choices=("1", "3", "5", "10", "inf"), required=True)
    parser.add_argument("--decoder", choices=("perseg", "span"), required=True,
                        help="Choose the checkpoint's training format, including for nonCoT envfb rows.")
    parser.add_argument("--model-path", type=absolute, required=True,
                        help="Local BAGEL base configs, tokenizer and ae.safetensors directory.")
    parser.add_argument("--checkpoint", type=absolute, required=True,
                        help="Task directory containing ema.safetensors or its sharded index.")
    parser.add_argument("--episodes-root", type=absolute, required=True,
                        help="Held-out JSONL(.gz) file or directory; include full init_state.")
    parser.add_argument("--output-dir", type=absolute, required=True,
                        help="New directory, or the same settings/assets as an existing run-contract.json.")
    parser.add_argument("--gpus", type=positive_int, default=1)
    parser.add_argument("--limit", type=positive_int, default=100)
    parser.add_argument("--max-steps", type=positive_int,
                        help="Sokoban/PushT fallback only; Maze2D uses metadata.max_steps or 40.")
    parser.add_argument("--max-mem-per-gpu", default="75GiB")
    parser.add_argument("--max-text-tokens", type=positive_int,
                        help="Default 600; Sokoban/PushT K=inf uses 1600.")
    parser.add_argument("--render", action="store_true", help="Save full traces, generated frames and GIFs.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun completed episodes with the same run settings and assets.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.task == "maze2d" and args.max_steps is not None:
        parser.error("--max-steps is unsupported for Maze2D: its runner reads metadata.max_steps from each episode, falling back to 40")
    assets = {}
    try:
        if not args.dry_run:
            base_files = validate_base(args.model_path)
            checkpoint_files = validate_checkpoint(args.checkpoint)
            episode_files = validate_episodes(args.episodes_root)
            validate_rows(episode_files, task=args.task, limit=args.limit)
            validate_output_location(args.output_dir, (args.model_path, args.checkpoint, args.episodes_root))
            assets = {"base": file_fingerprints(base_files),
                      "checkpoint": file_fingerprints(checkpoint_files),
                      "episodes": file_fingerprints(episode_files, hash_all=True)}
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    runner, profile, vae_max, vit_max, quality, steps = TASKS[args.task]
    text_tokens = args.max_text_tokens or (1600 if args.chunk_size == "inf" and args.task != "maze2d" else 600)
    settings = {
        "BAGEL_CHUNK_K": args.chunk_size, "BAGEL_BASE_PATH": str(args.model_path),
        "BAGEL_NO_REGROUND": "0", "BAGEL_COT_FRAME_MODE": "exact",
        "BAGEL_COT_CFG_TEXT": "1.0", "BAGEL_COT_CFG_IMG": "1.0",
        "BAGEL_COT_NUM_TIMESTEPS": "50", "BAGEL_COT_MAX_TEXT_TOKENS": str(text_tokens),
        "BAGEL_ENV_FEEDBACK_LABEL": "Action executed.",
        "BAGEL_PUSHT_INJECT_GT_FRAME": "0", "BAGEL_PUSHT_BICUBIC_ROUNDTRIP": "1",
        "BAGEL_WAIT_ALL_RANK_SUMMARIES": "1",
    }
    settings["BAGEL_DECODER"] = args.decoder
    command = torchrun(args.gpus, runner, module=True)
    options = {
        "episodes-root": args.episodes_root, "outdir": args.output_dir,
        "bagel-path": args.checkpoint, "bagel-base-path": args.model_path,
        "profile": profile, "limit": args.limit,
        "max-new-tokens": text_tokens, "max-mem-per-gpu": args.max_mem_per_gpu,
        "vae-transform-max-size": vae_max, "vae-transform-min-size": 256,
        "vit-transform-max-size": vit_max, "vit-transform-min-size": 224,
        "image-quality": quality,
    }
    if args.task != "maze2d":
        options["max-steps"] = args.max_steps or steps
    if args.task == "sokoban":
        options.update({"sokoban-success-mode": "stop_then_solved", "sokoban-action-mode-override": "move_only"})
    elif args.task == "pusht":
        options["pusht-reset-source"] = "init_state"
    for key, value in options.items():
        command.extend(("--" + key, str(value)))
    if args.render:
        command.extend(("--render", "--make-gif", "--full-output", "--save-generated-images"))
    if args.overwrite:
        command.append("--overwrite")
    try:
        return launch(command, runtime_environment(settings), args.output_dir, args.dry_run,
                      {"task": args.task, "variant": args.variant,
                       "chunk_size": args.chunk_size, "decoder": args.decoder},
                      gpus=args.gpus, assets=assets, evaluation=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
