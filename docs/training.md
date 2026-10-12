# Training

`scripts/train.py` fine-tunes BAGEL on prepared action-chunk datasets using the
main distributed trainer. It supports Sokoban, PushT and Maze2D on one CUDA node.
Install the [training environment](installation.md) before running it.

## Prepare inputs

| Argument | Required input |
| --- | --- |
| `--model-path` | Vanilla BAGEL directory with `llm_config.json`, `vit_config.json`, `ae.safetensors`, tokenizer files and a single `ema.safetensors` |
| `--data-dir` | Directory directly containing prepared interleaved JSONL or `.jsonl.gz` training shards; discovery is not recursive |
| `--output-dir` | New or empty directory for checkpoints, logs and launch metadata |
| `--variant`, `--chunk-size` | Labels matching the supplied data; these flags do not remap rows |

See [prepared training data](data-and-models.md#download-prepared-training-data) for a
revision-pinned Sokoban K=5 download and the required row format. The evaluation-only
base download must be supplemented with its full `ema.safetensors` before training.
This recipe initializes from the single vanilla BAGEL EMA file. The trainer cannot
use a sharded EMA index for this initialization.

Point `--data-dir` at the actual `training/` shard directory, not its parent. Startup
checks that the first rows contain nonempty user and assistant `messages` and that there
are at least as many rows as training GPUs. This catches common input mistakes but does
not validate every row, image, loss mask or custom YAML sampling limit. Keep input assets
outside the output directory and use a dataset/configuration sized for the worker count.

Blank JSONL lines are ignored. In the default JSONL loader, `num_used_data` is a
positive row cap across the selected shards. It must leave usable rows for every
rank and data-loader worker. An empty worker partition, or a complete pass with no
valid parsed rows, raises an error instead of waiting indefinitely.

Training uses FSDP. The default recipe runs eight large-memory CUDA GPUs; changing
`--gpus` changes sharding and effective packing and should be recorded as a different
configuration. CPU and Apple GPU training are unsupported.

Use one consistent target format across the run: CoT rows contain imagined-image
targets; nonCoT rows contain action targets only. Before each distributed forward,
the trainer checks that all ranks participate in the same text-loss, image-loss
and vision-encoder branches. Inconsistent batches fail with a diagnostic instead
of entering mismatched collective operations. Mixing target formats requires a
separate training recipe.

## Preview and launch

Run from the repository root after preparing a matching Sokoban CoT K=5 dataset:

```bash
python scripts/train.py \
  --task sokoban --variant cot --chunk-size 5 \
  --model-path models/BAGEL-7B-MoT \
  --data-dir datasets/sokoban-cot-k5/training \
  --output-dir runs/sokoban-cot-k5-train \
  --gpus 8 --dry-run
```

Remove `--dry-run` after reviewing the command. Dry runs and `--help` need only Python's
standard library; they write no files and do not validate model assets, datasets or
GPUs. Relative path arguments are resolved from the calling directory; the training
process itself runs from the repository root.

For PushT or Maze2D, change `--task` and supply the corresponding training shards.
For an action-only baseline, use `--variant noncot` with nonCoT envfb rows. The shared
loader retains real observation context and applies text loss to assistant actions;
it has no imagined-image targets when those rows contain no assistant images.

W&B logging is offline by default. Use `--wandb-online` with your configured account
when desired. Each cold start has a unique W&B run ID; `--run-name` sets the display
name and does not resume another run. Each launch that starts workers writes
`launch-<timestamp>.json` with code identity, command, selected environment, asset
fingerprints and exit status. For large training shards and weights, this record uses
file sizes and modification times; retain download revisions and checksums separately.
Checkpoints are saved under
`checkpoints/`, trainer logs under `logs/`, and W&B files under the run directory.
This launcher always starts from base EMA weights and does not resume optimizer state.

## Default recipe

| Setting | Value |
| --- | --- |
| Initialization | Vanilla BAGEL EMA, model-only cold start |
| Learning rate / schedule | `2e-5`, 100 warmup steps, cosine, final step index 5,000 |
| EMA decay | `0.993` |
| Conditioning dropout | Text, VAE and ViT: `0.0` |
| Packing token budget | 50,000; Maze2D CoT K=1: 24,000 |
| Checkpoints | Every 1,000 steps, or 2,500 for PushT; also index 500 and the final index |
| Evaluation / visualization probes | Disabled |
| Loader | `interleaved_cot_envfb` for CoT and nonCoT |

This loader encodes each assistant text segment with its own start/end tokens.
Use `--decoder perseg` when evaluating CoT checkpoints trained with these recipes,
including Maze2D. The separate Maze2D image-ablation protocol uses `span` and requires
weights trained for that format. See the [decoder guide](evaluation.md#select-a-checkpoint-contract).

The trainer uses an inclusive, zero-based step index: `--total-steps N` executes
updates numbered `0` through `N` (N+1 updates). The learning-rate schedule and
checkpoint names use this same step index. For example, a short run with
`--total-steps 10` executes 11 updates and saves
`checkpoints/0000010/ema.safetensors`. The launcher always requests a checkpoint at
the final index, even when it falls between periodic saves. Pass that checkpoint's
directory to `scripts/evaluate.py --checkpoint`, together with the original base assets.

Override `--token-budget`, `--lr`, `--warmup-steps`, `--total-steps` or `--save-every`
when defining a new run. To reproduce a specific checkpoint, also use its data revision,
preprocessing and model settings. The defaults here describe the chunk-SFT recipe.

| Task | Config in `configs/` | VAE max/min | ViT max/min | Data JPEG |
| --- | --- | --- | --- | --- |
| Sokoban | `sokoban/train.yaml` | 512/256 | 336/224 | q95 |
| PushT | `pusht/train.yaml` | 256/256 | 224/224 | q90 |
| Maze2D | `maze2d/train.yaml` | 256/256 | 224/224 | q95 |

Transform sizes are bounds; patch-stride rounding is part of the transform. Keep
original image encoding, frame order and loss masks instead of resizing or recompressing
prepared data to the nominal bounds.

The token budget includes text, image tokens and sequence-boundary overhead.
Samples exactly at the per-sample limit are accepted. An oversized sample raises
an error with its data index; increase `--token-budget` if the GPU capacity permits,
or prepare an explicitly shorter dataset. The loader does not silently discard it.

## Custom datasets and advanced use

The default YAML recipes request 100,000 rows. For the 50,000-row Sokoban
forward-thinking dataset, select its configuration explicitly:

```bash
python scripts/train.py \
  --task sokoban --variant cot --chunk-size inf \
  --dataset-config configs/sokoban/train_50k.yaml \
  --model-path models/BAGEL-7B-MoT \
  --data-dir datasets/sokoban-forward/training \
  --output-dir runs/sokoban-forward-train --gpus 8 --dry-run
```

Custom YAML must use existing [dataset registry](../data/dataset_info.py) keys and
match the supplied schema and row count. The launcher overrides data paths explicitly;
`AGIMA_DATA_ROOT` configures registry defaults when invoking the trainer directly.
Prepared data can be generated with the [data remappers](../scripts/README.md#prepare-data).

The trainer always opens evaluation configuration files. With probes disabled, the
launcher supplies the training configuration to these slots and leaves evaluation data
paths unset. If you enable probes in `training/pretrain_unified_navit.py`, use separate
train/eval registry keys: an evaluation override on a shared key can replace the
training directory.

The launcher clears inherited `BAGEL_*` experiment switches, preserves settings such as
`CUDA_VISIBLE_DEVICES`, and sets repository-local `PYTHONPATH` and `VISGYM_ROOT`.
Configure Pacman training, full-state resume and multi-node jobs directly with the
underlying trainer. When using its JSONL resume state,
keep the data, row ordering, shuffle seed, world size and worker count unchanged.
Saved cursors resume after the last yielded row in each worker's partition.
See [architecture](architecture.md) for the implementation map
and [evaluation](evaluation.md) for checkpoint-specific decoding.
