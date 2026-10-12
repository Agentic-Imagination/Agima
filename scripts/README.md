# Scripts

Run commands from the repository root with the [installed environment](../docs/installation.md).
Model execution requires the CUDA dependencies. Source checks and launcher dry runs
need only Python's standard library.

## Training and evaluation

| Entry point | Purpose | Guide |
| --- | --- | --- |
| [`train.py`](train.py) | Single-node fine-tuning on action-chunk datasets | [Training](../docs/training.md) |
| [`evaluate.py`](evaluate.py) | Sokoban, PushT and Maze2D evaluation | [Evaluation](../docs/evaluation.md) |
| [`tools/check_repo.py`](../tools/check_repo.py) | Repository, configuration and source checks | [Validation](../docs/validation.md) |
| [`summarize_evaluation.py`](summarize_evaluation.py) | Recount unique episode outcomes from a completed or resumed run | [Output checks](../docs/evaluation.md#inspect-outputs-and-resume) |

```bash
python scripts/train.py --help
python scripts/evaluate.py --help
python tools/check_repo.py
```

## Prepare data

For the training quickstart, [download the prepared Sokoban K=5 shards](../docs/data-and-models.md#download-prepared-training-data)
and pass their `training/` directory to the launcher. The tools below are for preparing
your own trajectory data.

The [`data/`](data/) modules construct the training formats and supply prompt functions
used by evaluation. Choose the builder for your task, trajectory schema and chunk
format. The envfb builders include real observations at chunk boundaries.

| Family | Modules |
| --- | --- |
| Chunk formats | `remap_{sokoban,pusht,maze2d,pacman}_*chunk*` |
| Shared branch representation | [`branch_format.py`](data/branch_format.py) |
| Sokoban branch/forward datasets | `sokoban_*` helpers and `remap_sokoban_cot_chunk_inf_*` |
| PushT branch datasets | [`pusht_coverage_score.py`](data/pusht_coverage_score.py), [`pusht_gaussian_branch.py`](data/pusht_gaussian_branch.py), [`remap_pusht_cot_chunk_inf_branch.py`](data/remap_pusht_cot_chunk_inf_branch.py) |
| Pacman forward datasets | [`remap_pacman_cot_chunk_inf_forward_shared.py`](data/remap_pacman_cot_chunk_inf_forward_shared.py) |
| Maze generation and remapping | [`build_maze2d_native256_ordered_pair.py`](data/build_maze2d_native256_ordered_pair.py), key-step remappers |
| Dataset audits | [`audit_init_state_parity.py`](data/audit_init_state_parity.py), [`audit_pacman_cot_chunk_inf_forward_shared.py`](data/audit_pacman_cot_chunk_inf_forward_shared.py) |

For example, this converts a recorded Sokoban trajectory dataset into K=5 CoT envfb rows:

```bash
python scripts/data/remap_sokoban_cot_chunk_k_envfb.py \
  --src-root /path/to/recorded-sokoban \
  --src-subdir trajectories/sokoban_easy/train \
  --out-root datasets/sokoban-cot-k5 --chunk-size 5 --workers 8
```

The source must contain the builder's recorded trajectory schema (`history`, actions,
previous/next images and episode metadata); the init-state-only evaluation pool is not
sufficient. Output shards go under `training/`. Use a new output directory and inspect
the module's overwrite behavior before regenerating data.

Keep frame ordering, image loss flags and shared prompt wording unchanged. Specify
source and output paths explicitly. PushT detour generation
also needs `--ppo-ckpt` or `AGIMA_PUSHT_PPO_CHECKPOINT`; natural-source Sokoban generation
needs `--natural-src`. Some helper modules are importable libraries or self-tests rather
than standalone dataset builders. Their `--help` may still import model/environment
dependencies.

See [dataset formats](../docs/data-and-models.md#dataset-formats) and the
[imagined-frame ablation tools](../experiments/image_ablation/README.md) for related
workflows.
