# Dependency groups

The model environment uses PyTorch 2.5.1, torchvision 0.20.1, Transformers 4.49.0,
NumPy 1.24.4 and SciPy 1.10.1. Some utility packages have no exact version pin, so
save the resolved package versions after installation when reproducing an experiment.

Use Linux with CUDA and Python 3.10 as the starting environment. The pinned scientific
packages support Python 3.10 and 3.11; Python 3.12 is for source-only checks, because
[SciPy 1.10.1 excludes Python 3.12](https://pypi.org/pypi/scipy/1.10.1/json).

From the repository root, after selecting a PyTorch/CUDA installation matching the
host as described in [installation](../docs/installation.md):

```bash
python -m pip install -r requirements.txt
python -m pip install packaging ninja wheel psutil
python -m pip install --no-build-isolation -r requirements/flash-attn.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pip check
```

FlashAttention is imported directly by the model, including when FLEX packing is
used. It must be installed after PyTorch and needs a compatible CUDA development
toolkit and compiler. Its [2.5.8 build script](https://github.com/Dao-AILab/flash-attention/blob/v2.5.8/setup.py)
also uses `psutil` to select a build parallelism limit.

| File | Purpose |
| --- | --- |
| [`core.txt`](core.txt) | Model, training and data dependencies, including `orjson` for Sokoban replay. |
| [`envs.txt`](envs.txt) | Dependencies for the bundled environments and shared evaluator. Included by the root manifest. |
| [`flash-attn.txt`](flash-attn.txt) | Required CUDA extension, built separately after PyTorch. |
| [`dev.txt`](dev.txt) | Pinned Ruff for critical-error lint; unit tests and AST checks use the standard library. |
| [`smoke.txt`](smoke.txt) | Python 3.10 dependencies for CPU simulator tests. Linux CI installs PyTorch 2.5.1 separately from its official CPU wheel index. |
| [`video.txt`](video.txt) | Optional MoviePy for Gymnasium's `RecordVideo` and `save_video`; not required for the task evaluators' PIL/GIF output. |

The repository provides a modified `gymnasium/` whose version string is 1.1.1.
The PyPI package does not contain these custom environments. Run scripts from the
repository root with the `PYTHONPATH` above and confirm that
`python -c 'import gymnasium; print(gymnasium.__file__)'` resolves to this copy.

The shared evaluator imports `gymnasium.vlm.vlm` and `gymnasium.vlm.interactor` at
module load time. Consequently `openai`, `ipywidgets` and `IPython` are required even
for local, non-interactive BAGEL evaluation and are included in `envs.txt`. Installing
the OpenAI SDK does not require API credentials for local BAGEL inference.

PushT calls `Space.on_collision()`, requiring `pymunk>=7.0`.
See the [Pymunk 7.0 change log](https://www.pymunk.org/en/latest/changelog.html#pymunk-7-0-0-2025-05-28).
Record the resolved Pymunk version when comparing trajectories or results.

## Optional utilities

For the vendored video-recording utilities, install the optional group alongside
the core constraints:

```bash
python -m pip install -r requirements.txt -r requirements/video.txt
python -m pip check
```

The MoviePy range retains the 1.x API and NumPy compatibility: [1.0.3 requires
NumPy >=1.17.3](https://github.com/Zulko/moviepy/blob/v1.0.3/setup.py), whereas
[2.2.1 requires NumPy >=1.25](https://pypi.org/pypi/moviepy/2.2.1/json), conflicting
with the NumPy 1.24.4 pin. Video encoding also needs FFmpeg.

JAX wrappers and the alternative Qwen VLM backend are not part of the documented
BAGEL workflow or default installation. The alternative backend expects
`Qwen3VLForConditionalGeneration`, which is absent from the pinned Transformers
4.49.0 baseline; installing `qwen-vl-utils` alone does not make that backend usable.
Do not upgrade the model stack solely to enable an unused optional backend.
The alternative non-turbo JPEG encoder needs a separately configured Python
environment via `NONTURBO_PYTHON`; it is not selected automatically.

Capture resolved package versions on the target CUDA machine after a successful
smoke run. Model weights, episode datasets and external API credentials are not
installed by pip.
