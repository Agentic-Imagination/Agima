# Models and datasets

Checkpoints and datasets for Training an Imaginative Agent (Agima). Choose a checkpoint and matching episode pool, then follow the [download guide](../data-and-models.md) and [evaluation guide](../evaluation.md).

| Catalog | Contents | Download manifest |
| --- | --- | --- |
| [Checkpoints](checkpoints.md) | 150 EMA exports: chunk policies, forward/branch planning, and Sokoban RL checkpoints | [models.json](models.json) |
| [Datasets](datasets.md) | 70 repositories: prepared training rows, source trajectories, RL inputs, and evaluation pools | [datasets.json](datasets.json) |

## Start with a K=5 policy

| Task | Imagination checkpoint | Action-only checkpoint | Training data |
| --- | --- | --- | --- |
| Sokoban | [CoT K=5](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_k5_world_model/tree/887196bd83c0f3ccd1ad99eb6d468148011de327) | [nonCoT K=5](https://huggingface.co/novastar111/sokoban_easy_noncot_chunk_k5_world_model/tree/b14f57ff51af94b5215b8c3e227687c3f5f40a25) | [CoT K=5 rows](https://huggingface.co/datasets/novastar111/sokoban_easy_cot_chunk_k5_train/tree/4d93607db2a883e41314543fb4f750fda09d78e2) |
| PushT | [CoT K=5](https://huggingface.co/novastar111/pusht_cot_chunk_k5_world_model/tree/391c75622b2b8e2354123aa147bd80e7bd99b386) | [nonCoT K=5](https://huggingface.co/novastar111/pusht_noncot_chunk_k5_world_model/tree/b6b82303542475a91d5188dda5c21f4526b8e72d) | [CoT K=5 rows](https://huggingface.co/datasets/novastar111/pusht_cot_chunk_k5_train/tree/9695863342ec0907934c58c70819b5629eb91eb7) |
| Maze2D | [CoT K=5](https://huggingface.co/novastar111/maze2d_easy_cot_chunk_k5_world_model/tree/db30142e05bd2c206b2151fe096af54010127583) | [nonCoT K=5](https://huggingface.co/novastar111/maze2d_easy_noncot_chunk_k5_world_model/tree/b92b041eeda871a53bc6f3cd215c1bb253456ba7) | [CoT K=5 rows](https://huggingface.co/datasets/novastar111/maze2d_easy_cot_chunk_k5_train/tree/3aab18c46706694ef99f81ead1b2c8a7174da219) |
| Pacman | [CoT K=5](https://huggingface.co/novastar111/pacman_braidthin_cot_chunk_k5_world_model/tree/a639e1ba19eccd6708c160110e77b6445888463a) | [nonCoT K=5](https://huggingface.co/novastar111/pacman_braidthin_noncot_chunk_k5_world_model/tree/1de20f9f7757c8fad2dbacf1c34ebb7c426220c6) | [CoT K=5 rows](https://huggingface.co/datasets/novastar111/pacman_braidthin_cot_chunk_k5_train/tree/a81fd6bd73216dc2a753e837b0626e64aef55847) |

For the checkpoints above, use `perseg` for Sokoban and PushT CoT evaluation and `span` for the Maze2D K=5 ablation. Maze2D checkpoints trained with the default envfb recipes in this repository use `perseg`. Pacman evaluation uses the [ablation runner](../../experiments/image_ablation/README.md).

PushT finite-K CoT exports use the diverse-start, keep-last configuration. Its open-loop CoT export uses the diverse-start configuration with `vae_last_only_attn=True`. Match the data, preprocessing, and decoding settings to the checkpoint's model card.

## Download exact files

Each manifest entry includes `repo_id`, `repo_type`, a fixed `revision`, and the selected `files`. For example, run this from the repository root after installation:

```python
import json
from pathlib import Path
from huggingface_hub import snapshot_download

manifest = json.loads(Path("docs/artifacts/models.json").read_text())
artifact = next(
    item for item in manifest["artifacts"]
    if item["repo_id"] == "novastar111/sokoban_easy_cot_chunk_k5_world_model"
)
snapshot_download(
    repo_id=artifact["repo_id"],
    repo_type=artifact["repo_type"],
    revision=artifact["revision"],
    allow_patterns=[file["path"] for file in artifact["files"]],
    local_dir="models/sokoban-cot-k5",
)
```

Use `datasets.json` in the same way for datasets. Keep each task and protocol in its own local directory. The manifests include file sizes, LFS SHA-256 hashes, and Git blob SHA-1 identifiers for regular files. Use them to estimate download size and verify files. Runner compatibility depends on the checkpoint's training format and the settings described above.

## Scope and terms

Each entry can be downloaded independently. The catalog also includes Pacman forward-thinking models and training data, strict route-order test pools, and Sokoban RL checkpoints. Result tables, evaluation outputs, and training logs are excluded.

Model weights and datasets retain their own terms. Read their model/dataset cards and the upstream BAGEL terms; the code repository's license does not grant additional rights to external assets.
