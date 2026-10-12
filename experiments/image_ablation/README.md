# Imagined-frame ablations

These runners test K=5 policies with Original, Blank, Stale and Oracle GT imagined-frame
feedback. The analysis tools compare them with a No-Imagine action-only checkpoint
using matching saved trajectories. Select an image intervention with `AGIMA_CONDITION`;
No-Imagine is evaluated separately.

Use the [CUDA installation](../../docs/installation.md), [model assets](../../docs/data-and-models.md)
and the exact episode pools for the selected experiment. The asset locations, decoder
settings and episode pairing requirements are listed below. Select compatible weights
from the [checkpoint catalog](../../docs/artifacts/checkpoints.md). The runners produce
new trajectories. Archive analysis needs saved CoT and nonCoT runs, which are not
distributed in the public catalog, but does not load model weights.

## Runners and image contracts

| Task | Module | Required decoder | Observation / budget |
| --- | --- | --- | --- |
| Sokoban | `experiments.image_ablation.run_ablation` | `perseg` | 256px, JPEG q95, 40 actions |
| Pacman hard | `experiments.image_ablation.run_ablation` | `perseg` | 256px, JPEG q95, 120 actions |
| Maze2D hard | `experiments.image_ablation.run_maze_pusht` with `AGIMA_TASK=maze` | `span` | 256px, JPEG q95, 40 actions |
| PushT | `experiments.image_ablation.run_maze_pusht` with `AGIMA_TASK=pusht` | `perseg` | Native 96px, JPEG q90, 30 actions |

Set `AGIMA_CONDITION=original|blank|stale|oracle`, `BAGEL_CHUNK_K=5` and the decoder before
importing the runner. Sokoban/Pacman select the task from episode `env_id`; Maze/PushT
also require `AGIMA_TASK`.

Sokoban/Pacman generate diffusion images in every condition, then substitute pixels
before the shared VAE and ViT context encoding. Their generated and replacement images
use JPEG q95. Maze/PushT replacement conditions skip diffusion; Original uses generated
frames. Their recorded experiment used `AGIMA_IMAGINED_CODEC=raw` and
`AGIMA_PUSHT_FRAME_INPUT=256`. Set these explicitly: the Python module defaults are
`jpeg` and `96`, respectively. PushT additionally uses the production bicubic feedback
round trip. Real observations at chunk boundaries are never replaced.

## Prepare assets for validation

Download the relevant [episode pools](../../docs/data-and-models.md#other-evaluation-assets)
and supply checkpoints with the decoder and preprocessing settings above.
Preflight and analysis tools expect the following locations relative to this directory:

| Local path under `assets/` | Source dataset / file |
| --- | --- |
| `sokoban_hard_test.jsonl.gz` | Sokoban hard pool: `test.jsonl.gz` |
| `pacman_hard_test.jsonl.gz` | Pacman hard pool: `data/test/batch_000000.jsonl.gz` |
| `maze_hard_fixedstart/testing/shard_00000.jsonl.gz` | Maze fixed-start pool: `testing/shard_00000.jsonl.gz` |
| `pusht_in_dist/data/test/batch_000000.jsonl.gz` | PushT in-distribution pool: `data/test/batch_000000.jsonl.gz` |

The first two files must retain the exact compressed bytes checked by `preflight.py`:
Sokoban pool revision `88b5402b1f88a3ec8bca74e02c108616ad794fc8`, Pacman pool revision
`38acbdf956e3ddf271c0e34a91caa6a1b260452b`. Copy/rename downloaded files without
recompressing them.

For archive analysis, package CoT and nonCoT runs with matching episode identities
and budgets under the names below. If you have no saved runs, skip the archive
preflights and run the model interventions directly.

| Task | Local archive paths under `assets/` |
| --- | --- |
| Sokoban | `sokoban_k5_{cot,noncot}.tar.gz` |
| Pacman | `pacman_k5_{cot,noncot}.tar.gz` |
| Maze2D | `table1_maze_pusht/table1/maze2d/ordinary_hard/k5_{cot,noncot}.tar.gz` |
| PushT | `table1_maze_pusht/table1/pusht/in_dist/k5_{cot,noncot}.tar.gz` |

`fetch_maze_pusht.py` downloads its listed Maze/PushT episode pools at the Hub's
current revision. It does not download model weights, base assets or trajectory archives. For a pinned reproduction, use explicit revisions with
`huggingface_hub` as shown in the data guide. `assets/`, `prepared/`, `runs/`, `output/`
and the repository's `models/` directory are ignored by Git.

From the repository root, verify the staged episode and archive pairing:

```bash
python -m experiments.image_ablation.preflight
python -m experiments.image_ablation.preflight_maze_pusht
```

The first command checks both Sokoban and Pacman; the second checks both Maze and PushT.
Each needs all of its listed inputs and writes validation/baseline records under
`prepared/`. These tools check recorded counts, episode identity and action budgets;
they do not run a model.

## Run a Sokoban condition

The example uses the model paths from the data guide and the staged evaluation file
above. Run from the repository root in a fresh shell with no unrelated `BAGEL_*`
experiment overrides. Unlike the standard launcher, these research modules inherit the
shell's experiment switches.

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

For Pacman, use its checkpoint and staged pool, `--profile pacman_res256`,
`--max-steps 120`, and a separate `runs/pacman_<condition>` directory. Other settings
above stay the same. For another intervention, change `AGIMA_CONDITION` and the output
directory together. After inspecting the initial traces, resume the same condition
with `--limit 100`; do not mix settings in one output directory.
Use `torchrun --standalone` as above for single-node runs. Multi-node runs must use
a unique rendezvous ID for each job. Rank summaries are matched to the current job
and restart attempt, so an earlier partial run cannot satisfy the completion check.

For Maze/PushT, invoke `-m experiments.image_ablation.run_maze_pusht` with the task's
assets and the additional settings below. Its CLI is the shared evaluation runtime's
CLI used above; these modules do not support the standard launcher's `--dry-run` option.

| Setting | Maze2D | PushT |
| --- | --- | --- |
| `AGIMA_TASK` | `maze` | `pusht` |
| `BAGEL_DECODER` | `span` | `perseg` |
| `AGIMA_IMAGINED_CODEC` | `raw` | `raw` |
| `AGIMA_PUSHT_FRAME_INPUT` | Not used | `256` |
| `BAGEL_COT_FRAME_MODE` | `exact` | `exact` |
| `BAGEL_ENV_FEEDBACK_LABEL` | `Action executed.` | `Action executed.` |
| `BAGEL_PUSHT_BICUBIC_ROUNDTRIP` / `BAGEL_PUSHT_INJECT_GT_FRAME` | Not used | `1` / `0` |
| `--profile` | `maze2d_native256` | `pusht_norm4_native` |
| `--image-quality` | `95` | `90` |
| VAE max/min; ViT max/min | 256/256; 224/224 | 256/256; 224/224 |
| `--max-steps` | 40; episode metadata governs | 30; episode fields govern |
| Additional CLI flags | `--render` | `--render --pusht-reset-source init_state` |

Keep K=5, guidance 1/1, 50 image steps, 600 text tokens and real feedback enabled.
Configure distributed resources for your own system and record the resulting command.

## Inspect and summarize

| Tool | Purpose |
| --- | --- |
| `test_oracle.py`, `test_maze_pusht_oracle.py` | Compare simulator replica renders using staged episodes |
| `test_intervention.py` | Check intervention encoding with a fake inferencer |
| `test_checkpoint_loader.py` | Check sharded-weight loading behavior |
| `baseline_metrics.py`, `baseline_metrics_maze_pusht.py` | Replay matching archived action-only baselines |
| `audit_traces.py`, `make_contact_sheet.py` | Inspect image/context traces |
| `summarize.py`, `summarize_maze_pusht.py` | Aggregate completed task/condition cells |
| `make_combined_paper_table.py` | Combine the generated four-task reports |

These tools may need full model/environment imports and staged assets, even when they
perform CPU checks. Summary scripts expect their configured complete cells, baseline
records and trace audits; an arbitrary partial output directory is insufficient.

After completing all four conditions on 100 episodes per task, run the following
from the repository root in the installed environment. Keep the directory names
`runs/<task>_<condition>` used above: the summary tools discover their inputs there.

First replay the archived No-Imagine actions to compute invalid-action and termination
diagnostics. The preflight files alone do not contain these metrics:

```bash
python -m experiments.image_ablation.baseline_metrics
python -m experiments.image_ablation.baseline_metrics_maze_pusht
```

Next audit all ten saved traces in each condition and write the filenames expected
by the summarizer. `audit_traces` otherwise defaults to five traces and a different
output filename.

```bash
mkdir -p experiments/image_ablation/output
for task in sokoban pacman maze pusht; do
  for condition in original blank stale oracle; do
    python -m experiments.image_ablation.audit_traces \
      "experiments/image_ablation/runs/${task}_${condition}" \
      --limit 10 \
      --output "experiments/image_ablation/output/${task}_${condition}_first10_audit.json" || exit 1
  done
done
python -m experiments.image_ablation.summarize
python -m experiments.image_ablation.summarize_maze_pusht
python -m experiments.image_ablation.make_combined_paper_table
```

For Sokoban/Pacman only, run `baseline_metrics`, restrict the audit loop to
`sokoban pacman`, and run `summarize`. For Maze/PushT only, run
`baseline_metrics_maze_pusht`, audit `maze pusht`, and run `summarize_maze_pusht`.
The combined table needs both task-pair summaries. All these analysis commands read
existing trajectories; they do not load model weights or generate new policy actions.

The simulator replica checks can run either task independently, or the whole suite:

```bash
python -m unittest experiments.image_ablation.test_maze_pusht_oracle.OraclePixelTest.test_maze -v
python -m unittest experiments.image_ablation.test_maze_pusht_oracle.OraclePixelTest.test_pusht -v
python -m unittest experiments.image_ablation.test_maze_pusht_oracle -v
```

They require the staged episode pools; the full suite also reads the CoT trajectory
archives to check recorded action sequences across chunk boundaries.

`summary.csv` and `maze_pusht_summary.csv` contain counts from the supplied episode
records. No-Imagine rows come from replayed baseline archives; all other conditions
come from the current run directories. Paired tests use the measured Original run.
`paper_table.csv` and `table.tex` present the same counts as percentages over 100
episodes per task. The CSV `source` column distinguishes the replayed baseline from
the current runs.

## Episode pairing

The Maze2D pool linked by the general artifact inventory has different initial states
from the fixed-start pool matched by the ablation. Use the latter for the archive
pairing checks. Set the Pacman hard action budget to 120.

Compare each condition against trajectories from the same episode pool, checkpoint
family and action budget. Preflight verifies episode identities before baseline replay
and aggregation.
