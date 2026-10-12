# Validation

Agima has checks for repository consistency, launcher behavior, result integrity and
CPU simulator replay. This guide covers development checks and model smoke tests.

## Run the CPU checks

From the repository root with Python 3.10 or later:

```bash
python3 -B -S tools/check_repo.py
python3 -B -S -m unittest discover -s tests -v
git diff --check
```

No model weights, episode datasets, PyTorch or CUDA installation is required. The checks
cover Python syntax, local module references, bundled render assets, license text hashes,
task configuration references, private-path hygiene, Markdown links and anchors, launcher
commands, and consistency between data and evaluation prompts. README help and dry-run examples
run with Python site packages disabled. Simulator and PyTorch distributed tests are
skipped in this mode.

Launcher tests exercise asset validation, command construction and PyTorch/CUDA preflight
failure behavior using controlled substitutes. Regression tests also cover run-contract
mismatches, stale rank summaries, launch locks, incomplete evaluation tallies and
conflicting duplicate results.

CI runs these checks on Python 3.10 and 3.12, plus critical Ruff checks for undefined names
and syntax errors. Install `requirements/dev.txt` and run
`python -m ruff check --select E9,F821,F822,F823 .` to reproduce the lint step.

## Exercise the simulators

The separate CI simulator job uses Linux and Python 3.10 without model weights or CUDA:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/smoke.txt
AGIMA_REQUIRE_ENV_SMOKE=1 SDL_VIDEODRIVER=dummy python -m unittest discover -s tests -p 'test_environments.py' -v
```

The four checks instantiate Sokoban, PushT, Maze2D and Pacman, render an RGB frame,
execute an action, restore the saved initial state and compare the replayed frame and
transition. `AGIMA_REQUIRE_ENV_SMOKE=1` turns missing dependencies into failures rather
than skips. These tests exercise the simulators independently of the policy model.

The Linux CI job requires all four simulator tests. See the
[workflow runs](https://github.com/Agentic-Imagination/Agima/actions/workflows/repository-checks.yml)
for results associated with a specific commit.

## Check JPEG worker transport

With the CPU dependencies above, run:

```bash
AGIMA_REQUIRE_ENV_SMOKE=1 python -m unittest discover -s tests -p 'test_jpeg_encoder.py' -v
```

These checks send image frames through real subprocess pipes, compare complete JPEG
responses, exercise recovery after an interrupted frame, and verify rejection of a
turbo codec when the non-turbo backend is requested. Transport checks use the installed
Pillow codec with a substituted capability response; they do not establish pixel
equivalence with a separately installed IJG encoder. The tests need NumPy and Pillow,
with no model weights, datasets or CUDA.

## Check distributed batch validation

With the CPU PyTorch installation above, run:

```bash
AGIMA_REQUIRE_DISTRIBUTED_SMOKE=1 python -m unittest discover -s tests -p 'test_distributed_training.py' -v
```

This test starts two Gloo workers and checks consistent CoT/nonCoT participation,
mixed text/image targets, missing vision inputs and empty-target errors. Both workers
must reach the same outcome and complete a subsequent collective. CI requires this
test to run; missing PyTorch or Gloo is an error in strict mode. It exercises batch
validation and synchronization without loading BAGEL or executing GPU FSDP training.

## Validate on a GPU host

On the intended training or evaluation host:

1. Follow [installation](installation.md), check PyTorch/CUDA and FlashAttention imports,
   and record resolved package versions and GPU hardware.
2. Prepare the upstream base assets and dataset revisions from [data and models](data-and-models.md),
   and select a matching released or locally trained checkpoint for evaluation.
   Check artifact hashes, episode identity, decoder, preprocessing and action budget.
3. Run a short [training job](training.md) or a five-episode [evaluation](evaluation.md).
   Inspect loading, outputs, image traces and execution errors before extending the run.
4. For reported metrics, evaluate the complete intended pool and keep episode-level
   results, launch manifests and expected episode counts together.

The launchers check PyTorch/CUDA availability before creating run output. Check available
VRAM and the installed CUDA extension build on the target host as well.

The standard launchers cover Sokoban, PushT and Maze2D chunk workflows. Pacman
interventions and other research settings use the separate
[image-ablation tools](../experiments/image_ablation/README.md), which have additional
asset and protocol requirements.
