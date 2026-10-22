# Usage

Run commands from the repository root. The training and evaluation launchers support Sokoban, PushT, and Maze2D chunk policies. For Pacman evaluation, use the [image-ablation runner](../experiments/image_ablation/README.md).

## Installation

Use Linux, Python 3.10, and NVIDIA CUDA GPUs. This example uses CUDA 12.1; FlashAttention also needs a compatible CUDA toolkit and compiler.

```bash
git clone https://github.com/Agentic-Imagination/Agima.git
cd Agima
conda create -n agima python=3.10
conda activate agima
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install packaging ninja wheel psutil
python -m pip install --no-build-isolation -r requirements/flash-attn.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## Download

Choose a Sokoban, PushT, or Maze2D model and its training data from the [chunk policies table](checkpoints.md#chunk-policies). Commands below are templates: replace `MODEL_REPO` and `TRAIN_REPO` with the Hugging Face repository IDs you choose.

Download the shared BAGEL assets:

```bash
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT \
  --revision 5019f57d168e5816e8f3f701b17cc816bb7cf24b \
  --local-dir models/BAGEL-7B-MoT \
  --include llm_config.json vit_config.json ae.safetensors \
    tokenizer_config.json tokenizer.json vocab.json merges.txt
```

For evaluation, also download the task weights. Use a separate local model/data directory for each selection.

```bash
huggingface-cli download MODEL_REPO \
  --local-dir models/policy --include "ema*.safetensors" ema.safetensors.index.json
```

For evaluation, download one of these episode files and save it as `datasets/test.jsonl.gz`:

| Task | Evaluation episodes |
| --- | --- |
| Sokoban | [Hard](https://huggingface.co/datasets/novastar111/sokoban_hard_init_state_pool/resolve/88b5402b1f88a3ec8bca74e02c108616ad794fc8/test.jsonl.gz) |
| PushT | [Far](https://huggingface.co/datasets/novastar111/pusht_eval_far_balanced100_20260810/resolve/9bed70c33c415f940d6bb8de8268572f9b1f37d0/data/test/batch_000000.jsonl.gz) |
| Maze2D | [Hard, fixed start](https://huggingface.co/datasets/novastar114/maze2d_hard_native256_stepmsg_fixedstart_plain/resolve/104c1f0aa4b6d6461a1b22e7448a221d9df4b4ba/testing/shard_00000.jsonl.gz) |

## Evaluation

Replace `TASK` with `sokoban`, `pusht`, or `maze2d`; set `VARIANT` to `cot` or `noncot` and `K` to the chosen checkpoint's chunk size (`1`, `3`, `5`, `10`, or `inf`). Use the checkpoint's `DECODER` (`perseg` or `span`); the released Maze2D K=5 ablation uses `span`, while this repository's default CoT training recipes use `perseg`.

```bash
python scripts/evaluate.py \
  --task TASK --variant VARIANT --chunk-size K --decoder DECODER \
  --model-path models/BAGEL-7B-MoT --checkpoint models/policy \
  --episodes-root datasets/test.jsonl.gz \
  --output-dir runs/evaluation --gpus 1 --limit 5 --render

python scripts/summarize_evaluation.py runs/evaluation --expected-episodes 5
```

For PushT, use the paper's final-coverage threshold (strictly > 0.80):

```bash
python scripts/summarize_evaluation.py runs/evaluation --expected-episodes 5 --pusht-success-threshold 0.8
```

For a paper comparison, use `--limit 100` and `--expected-episodes 100`. Results and GIFs are saved in the output directory. Each worker loads a full model. Use a fresh output directory when changing the checkpoint or settings, and pass the test file explicitly.

## Training

Download the vanilla BAGEL EMA weights and the training shards paired with your chosen task, variant, and K:

```bash
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT ema.safetensors \
  --revision 5019f57d168e5816e8f3f701b17cc816bb7cf24b \
  --local-dir models/BAGEL-7B-MoT

huggingface-cli download TRAIN_REPO --repo-type dataset \
  --local-dir datasets/train --include "training/*.jsonl.gz"

python scripts/train.py \
  --task TASK --variant VARIANT --chunk-size K \
  --model-path models/BAGEL-7B-MoT --data-dir datasets/train/training \
  --output-dir runs/training --gpus 8
```

Training uses eight large-memory GPUs, starts from the base EMA weights, and logs to W&B offline. The variant and K flags must match the prepared rows; they do not transform the data. Use a new output directory; weights are saved under `checkpoints/`. Recipes are in `configs/`, and data builders are in `scripts/data/`.

Add `--dry-run` to either launcher to preview the command. For all options:

```bash
python scripts/train.py --help
python scripts/evaluate.py --help
```
