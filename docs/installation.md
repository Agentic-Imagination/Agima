# Installation

Run all commands below from the Agima repository root. The launchers use the active
Python interpreter, so activate the intended environment before starting a job.

## Requirements

| Use | Environment |
| --- | --- |
| Source checks, tests, launcher help and dry runs | Python 3.10+; standard library only |
| Training and checkpoint evaluation | Linux, NVIDIA CUDA GPUs, compatible CUDA toolkit/compiler, FlashAttention |
| Dependency baseline | Python 3.10 (3.11 is within the scientific-package constraints), PyTorch 2.5.1, torchvision 0.20.1, Transformers 4.49.0, FlashAttention 2.5.8 |

Python 3.12 can run the source-only checks but cannot install the complete pinned
model environment: [SciPy 1.10.1 requires Python <3.12](https://pypi.org/pypi/scipy/1.10.1/json).
Keep the documented pins when reproducing an experiment instead of upgrading them to
force a newer interpreter.

The training recipes use eight large-memory GPUs. Evaluation loads a full
model per worker; its default `75GiB` memory cap is a placement limit, not a measured
minimum VRAM requirement. CPU and Apple GPU model execution are unsupported by these
launchers. The model uses BF16 and FlashAttention 2, so select hardware supporting
that combination.

## Install for model execution

The example selects the CUDA 12.1 PyTorch wheel. Choose the appropriate build from the
[PyTorch 2.5.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v251)
for your driver/toolkit setup.

```bash
conda create -n agima python=3.10
conda activate agima
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install packaging ninja wheel psutil
python -m pip install --no-build-isolation -r requirements/flash-attn.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pip check
```

FlashAttention is imported by the model even when FLEX packing is enabled. Install it
after PyTorch because its extension builds against the installed PyTorch and CUDA
toolchain; see the [FlashAttention 2.5.8 build instructions](https://github.com/Dao-AILab/flash-attention/tree/v2.5.8#installation-and-features).
The PyTorch wheel alone does not provide a full CUDA development toolkit for compiling
extensions. Install both `requirements.txt` and the FlashAttention requirements above.

The [dependency groups](../requirements/README.md) specify the model and task dependencies.
Resolve any remaining unpinned packages on the target
GPU host, then record the working environment after a smoke run.

## Check the environment

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
python -c 'import flash_attn; print(flash_attn.__version__)'
python -c 'import gymnasium; print(gymnasium.__file__)'
```

CUDA must be available, and the visible GPU count must be at least the launcher's
`--gpus` value. Use `CUDA_VISIBLE_DEVICES` to select the devices for a run. Launchers
perform this PyTorch/CUDA check before creating run output; this does not validate
available VRAM, GPU architecture, every dependency, or extension ABI compatibility.
`gymnasium.__file__` must point into this checkout's `gymnasium/`; the PyPI package does
not contain these custom tasks. Keep the repository first on `PYTHONPATH`.

The launchers default to `SDL_VIDEODRIVER=dummy` for headless rendering.
Preserve the bundled render assets and observation encoding when reproducing an
experiment.

## Configure assets

Use the [models and data guide](data-and-models.md) to download upstream base assets,
released task weights, and held-out episodes. Pass their paths explicitly to the launchers. An optional
shell convenience file is available:

```bash
cp .env.example .env
# Edit .env for your local asset locations.
source .env
```

The launchers do not load `.env` automatically. Pass these variables as CLI arguments
as shown in the [training](training.md) and [evaluation](evaluation.md) guides. For direct
loader and dataset-tool use, the available environment variables are documented in
`.env.example`.

## Source-only checks

No dependency installation, model download, or CUDA host is needed:

```bash
python3 tools/check_repo.py
python3 -m unittest discover -s tests -v
python3 scripts/train.py --help
python3 scripts/evaluate.py --help
```

These checks validate source code and launcher behavior without running the model.
To check for syntax errors and undefined names with Ruff:

```bash
python3 -m pip install -r requirements/dev.txt
python3 -m ruff check --select E9,F821,F822,F823 .
```

For CPU simulator checks, use the dependencies in `requirements/smoke.txt`. Install
the model requirements separately for training and checkpoint evaluation.
See [validation](validation.md) for commands, platform coverage and limitations.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `torch.cuda.is_available()` is false | Active environment, CUDA-enabled PyTorch wheel, driver, job allocation, and `CUDA_VISIBLE_DEVICES`. |
| FlashAttention installation or import fails | Install PyTorch first; check `nvcc --version`, the compiler/toolkit, and the extension's build instructions. |
| Custom Gymnasium task or space is missing | Run from this checkout and put it first on `PYTHONPATH`; inspect `gymnasium.__file__`. |
| A required base-model file is missing | Evaluation still needs the tokenizer, configs, and VAE in addition to the task EMA weights. |
| Checkpoint validation reports a missing shard | Download the index and every shard from the same checkpoint revision. |
| Training rejects an existing output directory | Training starts from base weights. Choose a new directory; this launcher does not resume optimizer state. |
| Evaluation skips completed episodes | Use a fresh directory for changed settings, or `--overwrite` to explicitly rerun the same experiment. |

For a reproducible issue report, include the command, full traceback, versions, and
asset revisions listed in [CONTRIBUTING.md](../CONTRIBUTING.md).
