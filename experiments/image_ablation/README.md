# Imagined-frame ablations

Compare Original, Blank, Stale and Oracle GT imagined frames with K=5 policies. Real observations at chunk boundaries stay intact. No-Imagine uses a separate action-only checkpoint.

Run from the repository root after installing Agima and downloading compatible checkpoints. These research runners inherit `BAGEL_*` shell settings; start in a fresh shell.

## Episode pools

Save these files under `experiments/image_ablation/assets/`, using the paths below. Keep the compressed files unchanged: the Sokoban/Pacman preflight checks their hashes.

| Task | Download | Local path under `assets/` |
| --- | --- | --- |
| Sokoban | [Test episodes](https://huggingface.co/datasets/novastar111/sokoban_hard_init_state_pool/resolve/88b5402b1f88a3ec8bca74e02c108616ad794fc8/test.jsonl.gz) | `sokoban_hard_test.jsonl.gz` |
| Pacman hard | [Test episodes](https://huggingface.co/datasets/novastar111/pacman_eval_hard_braidthin_random100_20260817/resolve/38acbdf956e3ddf271c0e34a91caa6a1b260452b/data/test/batch_000000.jsonl.gz) | `pacman_hard_test.jsonl.gz` |
| Maze2D hard | [Fixed-start episodes](https://huggingface.co/datasets/novastar114/maze2d_hard_native256_stepmsg_fixedstart_plain/resolve/104c1f0aa4b6d6461a1b22e7448a221d9df4b4ba/testing/shard_00000.jsonl.gz) | `maze_hard_fixedstart/testing/shard_00000.jsonl.gz` |

Only the PushT far evaluation set is included in this release. The paper's PushT in-distribution ablation inputs are excluded; supply the original inputs separately to run its paired analysis. Far results are a different evaluation and cannot replace that paper cell.

## Run an intervention

This example runs five Sokoban episodes. Set `AGIMA_CONDITION` to `original`, `blank`, `stale` or `oracle`, and use a separate output directory for each condition.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VISGYM_ROOT="$PWD" SDL_VIDEODRIVER=dummy
export AGIMA_CONDITION=original BAGEL_CHUNK_K=5 BAGEL_DECODER=perseg
export BAGEL_COT_CFG_TEXT=1.0 BAGEL_COT_CFG_IMG=1.0 BAGEL_COT_NUM_TIMESTEPS=50
export BAGEL_NO_REGROUND=0 BAGEL_WAIT_ALL_RANK_SUMMARIES=1
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  -m experiments.image_ablation.run_ablation \
  --episodes-root experiments/image_ablation/assets/sokoban_hard_test.jsonl.gz \
  --bagel-path models/sokoban-cot-k5 --bagel-base-path models/BAGEL-7B-MoT \
  --profile sokoban_v8_res256 --max-steps 40 --limit 5 \
  --outdir experiments/image_ablation/runs/sokoban_original \
  --max-new-tokens 600 --image-format JPEG --image-quality 95 \
  --vae-transform-max-size 512 --vae-transform-min-size 256 \
  --vit-transform-max-size 336 --vit-transform-min-size 224 \
  --sokoban-success-mode stop_then_solved --sokoban-action-mode-override move_only \
  --max-mem-per-gpu 75GiB --full-output --save-generated-images
```

For Pacman, switch the checkpoint and episode file, use `--profile pacman_res256 --max-steps 120`, and name the output `runs/pacman_<condition>`. Keep the remaining settings. After inspecting the traces, use `--limit 100` for each condition.

Maze2D and PushT use `experiments.image_ablation.run_maze_pusht`. Replace the task assets and the settings below; keep K=5, guidance 1/1, 50 image steps and 600 text tokens.

| Setting | Maze2D | PushT |
| --- | --- | --- |
| `AGIMA_TASK` / `BAGEL_DECODER` | `maze` / `span` | `pusht` / `perseg` |
| `AGIMA_IMAGINED_CODEC` | `raw` | `raw` |
| `AGIMA_PUSHT_FRAME_INPUT` | Not used | `256` |
| `BAGEL_COT_FRAME_MODE` / `BAGEL_ENV_FEEDBACK_LABEL` | `exact` / `Action executed.` | `exact` / `Action executed.` |
| `BAGEL_PUSHT_BICUBIC_ROUNDTRIP` / `BAGEL_PUSHT_INJECT_GT_FRAME` | Not used | `1` / `0` |
| `--profile` / `--image-quality` | `maze2d_native256` / `95` | `pusht_norm4_native` / `90` |
| VAE max/min; ViT max/min | 256/256; 224/224 | 256/256; 224/224 |
| Action budget | 40, from episode metadata | 30, from episode fields |
| Additional flags | `--render` | `--render --pusht-reset-source init_state` |

Use `runs/maze_<condition>` and `runs/pusht_<condition>`. Set `raw` and PushT input `256` explicitly; module defaults are `jpeg` and `96`. Maze2D's ablation requires weights trained for `span`; the standard training recipes use `perseg`.

Sokoban/Pacman generate diffusion images before substituting JPEG q95 pixels. Maze/PushT skip diffusion for replacement conditions and use raw imagined frames; PushT keeps the bicubic round trip for real feedback. The modules do not support `--dry-run`.

## Paired analysis

The reports require all four conditions on the same 100 episodes per task, plus matching CoT and No-Imagine trajectory archives. These archives are user-supplied; checkpoints alone are insufficient. Stage them under `assets/`:

- Sokoban/Pacman: `<task>_k5_cot.tar.gz` and `<task>_k5_noncot.tar.gz`.
- Maze2D: `table1_maze_pusht/table1/maze2d/ordinary_hard/k5_{cot,noncot}.tar.gz`.
- PushT: `table1_maze_pusht/table1/pusht/in_dist/k5_{cot,noncot}.tar.gz`.

Keep episode order, initial states and action budgets fixed across conditions. Maze2D must use the fixed-start pool above. Once the runs and archives are complete:

```bash
python -m experiments.image_ablation.preflight
python -m experiments.image_ablation.preflight_maze_pusht
python -m experiments.image_ablation.baseline_metrics
python -m experiments.image_ablation.baseline_metrics_maze_pusht
mkdir -p experiments/image_ablation/output
for task in sokoban pacman maze pusht; do
  for condition in original blank stale oracle; do
    python -m experiments.image_ablation.audit_traces \
      "experiments/image_ablation/runs/${task}_${condition}" --limit 10 \
      --output "experiments/image_ablation/output/${task}_${condition}_first10_audit.json" || exit 1
  done
done
python -m experiments.image_ablation.summarize
python -m experiments.image_ablation.summarize_maze_pusht
python -m experiments.image_ablation.make_combined_paper_table
```

Results are written to `experiments/image_ablation/output/`. No-Imagine metrics come from replayed archives; paired comparisons use the newly measured Original runs. Each preflight and summary processes its full task pair (Sokoban/Pacman or Maze/PushT).
