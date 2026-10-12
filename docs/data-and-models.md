# Data and models

Agima requires three separate inputs for evaluation: BAGEL base assets, a task checkpoint,
and a held-out episode pool. The [artifact catalog](artifacts/README.md) lists released
task checkpoints and datasets, with pinned revisions and file manifests. You can also
train from upstream BAGEL using the [training guide](training.md).

## Prepare the Sokoban K=5 example

Download these pinned base assets, task weights, and episodes for the Sokoban K=5 example:

| Asset | Source | Pinned revision | Required files |
| --- | --- | --- | --- |
| Base assets | [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) | `5019f57d168e5816e8f3f701b17cc816bb7cf24b` | Configs, tokenizer files, `ae.safetensors` |
| Sokoban CoT K=5 | [Task checkpoint](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_k5_world_model) | `887196bd83c0f3ccd1ad99eb6d468148011de327` | All EMA shards and `ema.safetensors.index.json` |
| Sokoban hard test pool | [Episode dataset](https://huggingface.co/datasets/novastar111/sokoban_hard_init_state_pool) | `88b5402b1f88a3ec8bca74e02c108616ad794fc8` | Root-level `test.jsonl.gz` |

The task checkpoint and episode pool are labeled CC BY-NC 4.0. Read each artifact's license before use;
the repository's source license does not replace external artifact terms.

After [installation](installation.md), run this from the repository root to download
the base assets, task checkpoint, and evaluation episodes:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ByteDance-Seed/BAGEL-7B-MoT",
    revision="5019f57d168e5816e8f3f701b17cc816bb7cf24b",
    local_dir="models/BAGEL-7B-MoT",
    allow_patterns=[
        "llm_config.json", "vit_config.json", "ae.safetensors",
        "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
    ],
)
snapshot_download(
    repo_id="novastar111/sokoban_easy_cot_chunk_k5_world_model",
    revision="887196bd83c0f3ccd1ad99eb6d468148011de327",
    local_dir="models/sokoban-cot-k5",
    allow_patterns=["README.md", "ema*.safetensors", "ema.safetensors.index.json"],
)
snapshot_download(
    repo_id="novastar111/sokoban_hard_init_state_pool",
    repo_type="dataset",
    revision="88b5402b1f88a3ec8bca74e02c108616ad794fc8",
    local_dir="datasets/sokoban-eval",
    allow_patterns=["README.md", "test.jsonl.gz"],
)
```

Use `--episodes-root datasets/sokoban-eval/test.jsonl.gz` in the
[evaluation command](evaluation.md#run-sokoban-k5). The pool also has a `train.jsonl.gz`;
pointing the evaluator at a directory containing both splits would load both recursively.

The task download populates `models/sokoban-cot-k5/` for the evaluation example.
Alternatively, pass the directory containing `ema.safetensors` from your own training
run to `--checkpoint`. Base BAGEL weights alone are not a trained task policy.

For training, also download the base model's single `ema.safetensors` file at the same
pinned revision:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="ByteDance-Seed/BAGEL-7B-MoT",
    revision="5019f57d168e5816e8f3f701b17cc816bb7cf24b",
    filename="ema.safetensors",
    local_dir="models/BAGEL-7B-MoT",
)
```

Evaluation accepts `ema.safetensors` or an `ema.safetensors.index.json` with every referenced
shard. The training loader requires a single base EMA file. EMA-only task exports cannot
resume optimizer and scheduler state.

## Download prepared training data

The public [Sokoban CoT K=5 training dataset](https://huggingface.co/datasets/novastar111/sokoban_easy_cot_chunk_k5_train)
contains prepared envfb rows with inline JPEG images. Download its eight compressed
shards into the directory used by the training example:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="novastar111/sokoban_easy_cot_chunk_k5_train",
    repo_type="dataset",
    revision="4d93607db2a883e41314543fb4f750fda09d78e2",
    local_dir="datasets/sokoban-cot-k5",
    allow_patterns=["README.md", "training/*.jsonl.gz"],
)
```

Pass `--data-dir datasets/sokoban-cot-k5/training` to `scripts/train.py`, together with
the vanilla BAGEL EMA file downloaded above. These rows are already prepared for
`--task sokoban --variant cot --chunk-size 5`; no remapping is needed. The complete
dataset is a large download. Model and dataset cards contain their respective terms.

For another task, variant or K, select matching prepared rows from the
[dataset catalog](artifacts/datasets.md), or construct your own using the
[data preparation tools](../scripts/README.md#prepare-data).

## Other evaluation assets

Choose your checkpoint, pool, decoder, K and action budget together. The table below
lists episode pools used by the [imagined-frame ablations](../experiments/image_ablation/README.md).
Select matching weights from the [checkpoint catalog](artifacts/checkpoints.md).
The decoder column records each ablation's
protocol, not the default for all checkpoints of that task.

| Task | Evaluation pool / test file | Ablation decoder |
| --- | --- | --- |
| Sokoban | [sokoban_hard_init_state_pool](https://huggingface.co/datasets/novastar111/sokoban_hard_init_state_pool), `test.jsonl.gz` | `perseg` |
| PushT | [pusht_eval_in_dist_balanced100_20260820](https://huggingface.co/datasets/novastar111/pusht_eval_in_dist_balanced100_20260820), `data/test/batch_000000.jsonl.gz` | `perseg` |
| Maze2D | [maze2d_hard_native256_stepmsg_fixedstart_plain](https://huggingface.co/datasets/novastar114/maze2d_hard_native256_stepmsg_fixedstart_plain), `testing/shard_00000.jsonl.gz` | `span` |
| Pacman | [pacman_eval_hard_braidthin_random100_20260817](https://huggingface.co/datasets/novastar111/pacman_eval_hard_braidthin_random100_20260817), `data/test/batch_000000.jsonl.gz` | `perseg`; research runner only |

Maze2D's fixed-start pool matches the initial-state manifest used by the ablation.
Use it for [archive pairing](../experiments/image_ablation/README.md#episode-pairing).
CoT checkpoints trained with this repository's default recipes use `perseg`, including
Maze2D; they are not interchangeable with the `span` weights required by the Maze2D
ablation runner. Use the decoder and preprocessing settings specified for your weights.

## Dataset formats

Evaluation reads a JSON object per line from `.jsonl` or `.jsonl.gz`. Each episode needs
`env_id`, the task's `env_kwargs`, seeds, and the complete `init_state` needed to restore
the simulator. Maze2D additionally uses its `metadata`, `alignment_index`, and recorded
prompt/trajectory fields. Preserve source episode ordering and task metadata. A training
row with only `messages` is not an evaluation episode.

Training uses prepared interleaved rows, typically under `training/`. A row's `messages`
contains `user` and `assistant` turns with ordered text/image content. Images are inline
base64 strings in `image` or `image_b64`; `loss` controls their role. Initial observations
and real feedback images are user context with `loss: false`. Imagined assistant images
use `loss: true`. The envfb loader omits assistant images with `loss: false`. Keep these
flags, prompt text, image encoding and chunk boundaries intact.

Download prepared rows as above, or use the matching [data remappers](../scripts/README.md#prepare-data)
for your own trajectories. The training launcher does not convert raw episodes or change
K. Match the checkpoint's preprocessing and model settings to the chosen recipe.

## Local storage and run records

`models/`, `datasets/`, `runs/`, and `results/` are ignored by Git. Keep downloaded assets
under `models/` and `datasets/`, and use a separate output directory for each experiment.
For a reported result, record the code commit, artifact revisions, dataset hashes,
resolved environment, full command, K, decoder, observation transforms and action budget.
