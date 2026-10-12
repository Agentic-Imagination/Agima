# Architecture

Agima connects interleaved text/image modeling with action-chunk datasets and task
simulators. Training learns from prepared trajectories; evaluation alternates model
generation, simulator actions and real observation feedback.

## Implementation map

| Component | Location | Responsibility |
| --- | --- | --- |
| Model | [`modeling/`](../modeling/) | BAGEL, language/vision backbones and the image autoencoder |
| Inference | [`inferencer.py`](../inferencer.py) | Text/image generation and context-cache updates |
| Training | [`training/`](../training/) | Distributed optimization, EMA weights and checkpoints |
| Data loading | [`data/`](../data/) | Dataset registry, interleaved loaders, transforms and sequence packing |
| Task recipes | [`configs/`](../configs/README.md) | Dataset groups, sampling and transform settings |
| Task environments | [`gymnasium/`](../gymnasium/) | Sokoban, PushT, Maze2D and Pacman simulation and rendering |
| Evaluation | [`evaluation/`](../evaluation/) | Episode loading, task runners, distributed execution and result records |
| Data preparation | [`scripts/data/`](../scripts/data/) | Trajectory remapping, prompt builders and dataset generators |
| Public commands | [`scripts/`](../scripts/README.md) | Training, evaluation and result-summary entrypoints |
| Image interventions | [`experiments/image_ablation/`](../experiments/image_ablation/README.md) | Original, Blank, Stale and Oracle GT experiments |
| Source checks | [`tools/check_repo.py`](../tools/check_repo.py) | Repository, configuration, import and documentation checks |

## Training flow

1. A data remapper converts recorded trajectories into interleaved `messages` rows.
   Real observations provide user context; CoT rows additionally include assistant
   image targets. The [data guide](data-and-models.md#dataset-formats) describes the format.
2. [`scripts/train.py`](../scripts/train.py) selects a task recipe and passes explicit
   model/data paths to [`training/pretrain_unified_navit.py`](../training/pretrain_unified_navit.py).
   Dataset names resolve through [`data/dataset_info.py`](../data/dataset_info.py).
3. The [envfb loader](../data/interleave_datasets/edit_dataset_envfb.py) interprets
   text/image turns and loss flags. [`PackedDataset`](../data/dataset_base.py) packs
   samples into the configured token budget.
4. The trainer encodes image inputs, computes losses for the batch's text/image targets,
   updates weights and EMA state, and saves checkpoints through
   [`training/fsdp_utils.py`](../training/fsdp_utils.py).

The launcher covers single-node chunk SFT for Sokoban, PushT and Maze2D. See
[training](training.md) for target formats, token budgets, decoder compatibility and
checkpoint loading. The lower-level trainer also contains evaluation/probe paths that
require their own configuration.

## Evaluation flow

1. [`scripts/evaluate.py`](../scripts/evaluate.py) validates the selected assets and
   output settings, configures the task protocol and starts the distributed workers.
2. [`evaluation/runtime.py`](../evaluation/runtime.py) selects and shards episode
   records, loads the model and dispatches to a task runner. Environment adapters
   restore the recorded initial state.
3. The task runner builds the prompt and observation context, then calls the shared
   inferencer. It decodes and executes an action chunk and supplies a real observation
   for the next chunk. K=inf uses a single open-loop plan.
4. Task runners write episode records and optional traces. Rank summaries describe
   each worker's results; [`scripts/summarize_evaluation.py`](../scripts/summarize_evaluation.py)
   checks unique episode outcomes against the requested episode count.

The [evaluation guide](evaluation.md) specifies decoder boundaries, observation
preprocessing, action budgets and success criteria. Image-ablation runners use the
same model and task simulators with changes to imagined-frame feedback;
their [experiment guide](../experiments/image_ablation/README.md) defines those protocols.

## Shared contracts

Training and evaluation use prompt builders from `scripts/data/`. Keep their wording,
action representation, chunk boundaries and image preprocessing aligned with the
checkpoint. YAML files select dataset and transform settings; they do not convert raw
trajectories or change a prepared row's chunk size.

The repository's `gymnasium/` includes custom environments and must be imported from
this checkout. Model assets, episode pools and generated runs live outside the source
packages; the launchers accept their locations explicitly. See
[installation](installation.md) for the Python path and dependency setup.

## Models, data and credits

- [Models and data](data-and-models.md): base-asset and dataset downloads, local checkpoint requirements and input formats.
- [Artifact catalog](artifacts/README.md): released checkpoints and datasets with pinned file manifests.
- [Third-party notices](../THIRD_PARTY_NOTICES.md): component sources, copyright notices and licenses.
