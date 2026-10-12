# Task configurations

Training and evaluation recipes are grouped by task. The YAML files define dataset
registry keys, sample counts, loader groups, transforms and loss settings; they do not
download or rewrite data.

| Task | Training | Evaluation |
| --- | --- | --- |
| Sokoban | [`sokoban/train.yaml`](sokoban/train.yaml): 100,000 rows; [`sokoban/train_50k.yaml`](sokoban/train_50k.yaml): 50,000-row forward-thinking recipe | [`sokoban/eval.yaml`](sokoban/eval.yaml) |
| PushT | [`pusht/train.yaml`](pusht/train.yaml) | [`pusht/eval.yaml`](pusht/eval.yaml) |
| Maze2D | [`maze2d/train.yaml`](maze2d/train.yaml) | The standalone evaluator uses episode files directly |
| Pacman easy | [`pacman/train.yaml`](pacman/train.yaml) | [`pacman/eval.yaml`](pacman/eval.yaml) |

The [training launcher](../docs/training.md), `scripts/train.py`, selects
`configs/<task>/train.yaml` for Sokoban, PushT or Maze2D. Use `--dataset-config` to select
an alternative, for example `configs/sokoban/train_50k.yaml`. The launcher overrides
registry locations using the supplied `--data-dir`; `AGIMA_DATA_ROOT` sets registry
defaults for direct use of `training/pretrain_unified_navit.py`.

For Pacman, pass its YAML explicitly to the underlying trainer and supply the complete
experiment recipe. It is outside the three-task training launcher. The standalone
[evaluation launcher](../docs/evaluation.md) takes `--episodes-root` rather than a YAML;
the `eval.yaml` recipes configure dataset-loader evaluation/probes in the trainer.

Use keys defined in the [dataset registry](../data/dataset_info.py). Keep the supplied
JSONL schema, chunk size, frame order, image encoding and loss masks consistent with
the selected recipe.

When enabling trainer evaluation probes, use distinct train/eval registry keys so an
evaluation path override cannot replace the training directory. The standard training
launcher disables these probes; see its [advanced-use notes](../docs/training.md#custom-datasets-and-advanced-use).
