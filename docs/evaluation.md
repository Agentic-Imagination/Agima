# Evaluation

The single-node launcher evaluates Sokoban, PushT and Maze2D with real observations at
chunk boundaries. Install the [CUDA environment](installation.md) and prepare the
[base assets, task weights and held-out episodes](data-and-models.md) first.
Use a released checkpoint from the [artifact catalog](artifacts/README.md) or a
compatible checkpoint you trained. Dry-run examples work without weights.

## Run Sokoban K=5

From the repository root, using the pinned base assets, task checkpoint, and episodes
from the download guide:

```bash
python scripts/evaluate.py \
  --task sokoban --variant cot --chunk-size 5 --decoder perseg \
  --model-path models/BAGEL-7B-MoT \
  --checkpoint models/sokoban-cot-k5 \
  --episodes-root datasets/sokoban-eval/test.jsonl.gz \
  --output-dir runs/sokoban-k5-eval \
  --gpus 1 --limit 5 --render --dry-run
```

Review the printed command, then remove `--dry-run` to execute it. `--help` and dry runs
need only Python's standard library and write no files; they do not validate downloaded
assets, installed model dependencies or GPU capacity. A real launch checks nonempty
assets, checkpoint shard references, the selected episode rows and CUDA availability
before starting workers. Check the checkpoint's training format against the decoder
and preprocessing settings below.

`--gpus` controls the number of local workers; each worker loads a complete inference
model and processes a subset of episodes. Its default memory cap is 75 GiB per GPU.
`--max-mem-per-gpu` changes that cap, not the model's memory requirement.

## Select a checkpoint contract

The decoders implement different text/image token boundaries:

| Decoder | Generation format |
| --- | --- |
| `perseg` | Generates each text segment as a separate assistant span, interleaves imagined images, and rebuilds conversation context for each chunk. |
| `span` | Generates a chunk within one continuous assistant span, interleaving imagined images while retaining the generation cache across chunks. |

The required `--decoder` must match the checkpoint's training format. CoT checkpoints
trained with this repository's default [envfb recipes](training.md#default-recipe) use
`perseg` for all three tasks, including Maze2D. A matching task name alone is insufficient.

For a Maze2D K=5 checkpoint trained with these default recipes:

```bash
python scripts/evaluate.py \
  --task maze2d --variant cot --chunk-size 5 --decoder perseg \
  --model-path models/BAGEL-7B-MoT \
  --checkpoint models/maze2d-cot-k5 \
  --episodes-root datasets/maze2d-eval/testing/shard_00000.jsonl.gz \
  --output-dir runs/maze2d-k5-eval --gpus 1 --limit 5 --render --dry-run
```

The separate [Maze2D ablation workflow](../experiments/image_ablation/README.md) uses
`span` and requires a checkpoint trained for that token-boundary format. It cannot be
reproduced by changing the decoder on a checkpoint trained for `perseg`.

For PushT, use `--task pusht --decoder perseg` with its matching checkpoint and
`data/test/batch_000000.jsonl.gz` pool. The launcher applies each task's observation
transforms automatically.

For action-only envfb checkpoints, specify `--variant noncot`. This flag records the
variant in the launch manifest without changing the weights or generation path.
The task runner handles text-only output without imagined frames.
K=inf runs a complete action plan without another feedback turn. Use this only with
weights trained on the corresponding envfb format. Plain-chunk and
single-shot protocols use different prompts; check the model card before selecting
a checkpoint for these envfb launchers.

## Input and decoding contracts

Pass a test file or a directory containing only the intended test shards. Directories
are scanned recursively in sorted path order; `--limit` selects the first N valid episode
records before distributing them to workers. Do not mix training and test splits.
Keep the full `init_state` and task metadata to reproduce simulator resets.

| Task | Profile | VAE max/min | ViT max/min | Observation JPEG | Action budget |
| --- | --- | --- | --- | --- | --- |
| Sokoban | `sokoban_v8_res256` | 512/256 | 336/224 | q95 | Episode fields, then CLI fallback 40 |
| PushT | `pusht_norm4_native` | 256/256 | 224/224 | q90 | Episode fields, then CLI fallback 30 |
| Maze2D | `maze2d_native256` | 256/256 | 224/224 | q95 | `metadata.max_steps`, otherwise 40 |

Sokoban and PushT read `env_kwargs.max_steps`, then `init_state.max_steps`, then the
`--max-steps` fallback. Changing the CLI value cannot override those episode fields.
Maze2D uses a separate metadata-based budget; the launcher rejects `--max-steps` for this
task. Use episodes whose metadata and simulator state agree.

- K must match training: finite K re-observes the environment between chunks; `inf`
  plans the whole episode from the initial frame.
- Text and image guidance are both `1.0`, with 50 diffusion steps. The text budget is
  600 tokens, or 1,600 for Sokoban/PushT K=inf; `--max-text-tokens` changes this setting.
- Feedback uses the shared `Action executed.` label and the same prompt builders as
  the data remappers. Sokoban uses move-only actions.
- PushT uses native 96-pixel observations, 256-pixel imagined frames and the existing
  bicubic feedback round trip. Changing image size or JPEG settings changes the protocol.

The launcher clears inherited `BAGEL_*` switches before setting these values, preserves
`CUDA_VISIBLE_DEVICES`, and adds this checkout to the Python path. Direct research
entrypoints do not reset those switches; record their complete environment.

## Success criteria

All three task runners require an explicit final `('stop', 'stop')`, termination,
and no truncation. They then apply the task-specific predicate below:

| Task | Additional condition | Implementation |
| --- | --- | --- |
| Sokoban | The environment reports `is_success` for the solved board. | [`sokoban.py`](../evaluation/sokoban.py) |
| PushT | The environment reports `is_success`: target coverage meets the episode's `success_threshold` (0.9 in the linked evaluation pool). | [`pusht.py`](../evaluation/pusht.py) |
| Maze2D | Final distance to the goal is exactly zero and cumulative reward is positive. | [`maze2d.py`](../evaluation/maze2d.py) |

Reaching the target without submitting `stop`, or exhausting the action budget,
does not satisfy these predicates. Use `stats.success` in `episode_stats.json` for
the reported outcome; reward or visual inspection alone is insufficient.

## Inspect outputs and resume

Each launch that starts workers writes `launch-<timestamp>.json` with its command,
selected environment, Python/PyTorch versions, Git identity, runtime source fingerprint,
asset identities, timestamps and exit status. Small metadata files and evaluation shards
are SHA-256 hashed; large weight files are identified by path, size and modification time.
Keep download revisions and weight checksums separately.
Episode directories contain `episode_stats.json`; `--render` also requests full traces,
generated images and GIFs. Execution failures are recorded under `_errors/` and in rank
summaries. Check episode counts with the summary command below before reporting a rate.

`run-contract.json` protects an output directory from reuse with different settings.
To resume an interrupted run, repeat the same command with the same assets, runtime,
source and selected environment. Completed episodes are skipped. `--overwrite` reruns
the same experiment; it does not bypass the contract. Changing any other setting,
including `--limit`, `--gpus` or `--render`, requires a new output directory. For example,
after a five-episode smoke run, use `--limit 100 --output-dir runs/sokoban-k5-eval-100`.
Nonempty output directories without a contract cannot be resumed by this wrapper.

An exclusive `.launch.lock` prevents two launchers writing to the same directory. It is
removed on normal completion. Interrupted launches retain the lock because distributed
workers may still be running. Confirm that the old launcher and all its workers have
stopped before removing a stale lock. The wrapper clears stale rank and aggregate summary
files before resuming so aggregation waits for this invocation's workers. Inputs and
outputs must not overlap.

Each worker group rebuilds its episode shards from the selected inputs. Malformed
JSONL rows and empty selections fail explicitly. If any episode raises an exception,
the job writes its diagnostics, marks `summary.json` as `incomplete` with a null
`success_rate`, and exits unsuccessfully. Resume the same run after fixing the
underlying problem; failed episodes are never silently removed from a benchmark rate.

Recount unique episode records, especially after a resumed run:

```bash
python scripts/summarize_evaluation.py runs/sokoban-k5-eval --expected-episodes 5
```

The tally deduplicates episode indices, reports matching duplicates, and rejects corrupt
records, conflicting outcomes/identities, invalid success flags, and an unexpected
episode count. Such runs show `INVALID` instead of a rate and exit with status 1. Use
`--json` for machine-readable results. Pass individual run directories or a quoted glob
such as `'runs/sokoban-*'`; a parent containing nested run contracts is rejected to
prevent combining different experiments. The tool also checks episode errors and the
planned count in `summary.json` when available. Always supply `--expected-episodes`
to verify the intended denominator, including runs with no aggregate summary. Also
inspect `_errors/` and rank summaries. Use a small run to check loading and trace
generation, then evaluate the full intended pool for reported metrics.

## Research workflows

The [imagined-frame ablation tools](../experiments/image_ablation/README.md) include
Pacman K=5 and controlled image interventions. Pacman is outside this three-task launcher.
Multi-node jobs and other decoding protocols require their own experiment setup.

When comparing results, keep the episode pool, initial states, preprocessing, decoder,
action budget and checkpoint revision fixed. The [experiment guide](../experiments/image_ablation/README.md#episode-pairing)
describes the additional pairing requirements for archived trajectories.
