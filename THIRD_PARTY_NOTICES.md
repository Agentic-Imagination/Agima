# Third-party notices

Agima is derived from `unified_world_model` at source commit
`be2d55cb50880375d99ec58463a657de62a1f5cc`.
Original copyright headers and the upstream [Apache-2.0 license](LICENSE) are
retained. Included components retain the licenses and notices listed below.
Model weights, datasets and installed dependencies have separate terms.

## Included components

| Included component | Source / copyright notices | License or terms |
| --- | --- | --- |
| BAGEL model, training and data code, except separately licensed files below | [ByteDance BAGEL](https://github.com/ByteDance-Seed/Bagel); ByteDance 2025 and per-file notices | [Apache-2.0](LICENSE) |
| `modeling/bagel/modeling_utils.py` | [Facebook Research DiT](https://github.com/facebookresearch/DiT); Facebook 2022, ByteDance modifications 2025 | [CC BY-NC 4.0](LICENSES/DiT-CC-BY-NC-4.0.txt) |
| `modeling/autoencoder.py` | [Black Forest Labs FLUX](https://github.com/black-forest-labs/flux); Black Forest Labs 2024, ByteDance modifications 2025 | [Apache-2.0](LICENSES/FLUX-Apache-2.0.txt) |
| Qwen/SigLIP model code | [Hugging Face Transformers](https://github.com/huggingface/transformers); Qwen Team and Hugging Face notices | [Apache-2.0](LICENSES/Transformers-Apache-2.0.txt) |
| `data/video_utils.py` | [OpenGVLab InternVL](https://github.com/OpenGVLab/InternVL); OpenGVLab 2023, ByteDance modifications 2025 | [MIT](LICENSES/InternVL-MIT.txt) |
| Vendored Gymnasium base library | [Gymnasium 1.1.1](https://github.com/Farama-Foundation/Gymnasium/tree/v1.1.1); OpenAI 2016 and Farama Foundation 2022 | [MIT](LICENSES/Gymnasium-MIT.txt) |
| `gymnasium/wrappers/jax_to_torch.py` conversion functions | [Brax](https://github.com/google/brax/blob/9d6b7ced2a13da0d074b5e9fbd3aad8311e26997/brax/io/torch.py) | [Apache-2.0](LICENSES/Brax-Apache-2.0.txt) |
| PushT Pymunk helpers | [gym-pusht](https://github.com/huggingface/gym-pusht), including Diffusion Policy's Columbia AIR Lab notice; [Pymunk](https://github.com/viblo/pymunk) | [gym-pusht notices](LICENSES/Gym-PushT-Apache-2.0.txt), [Pymunk MIT](LICENSES/Pymunk-MIT.txt) |
| Sokoban room generation and seven surface sprites | [gym-sokoban](https://github.com/mpSchrader/gym-sokoban/tree/8e06e44e8bf3bb8bc73eeb1e7f0354508ce3fc89); Max-Philipp Schrader 2018 | [MIT](LICENSES/Gym-Sokoban-MIT.txt) |
| Maze2D environment ancestry credited in `gymnasium/envs/maze_2d/` | [maze-world](https://github.com/koulanurag/maze-world/tree/a080e6d394f75ed6542615dfbe66ca916ffa9a0e); Anurag Koul 2024 | [Apache-2.0](LICENSES/Maze-World-Apache-2.0.txt) |
| `WilsonMazeGenerator` in `gymnasium/envs/maze_2d/maze_2d.py` | [WilsonMazeGenerator](https://github.com/CaptainFl1nt/WilsonMazeGenerator/tree/c7d9aeabe382a868616c20d4dfad791bdf414152); CaptainFl1nt 2021 | [MIT](LICENSES/WilsonMazeGenerator-MIT.txt) |
| Pacman rendering GIFs under `gymnasium/envs/pacman_2d/assets/greyblue9_pacman_python/` | [pacman-python](https://github.com/greyblue9/pacman-python/tree/2dc773e6c0a28cc03b55f7156c3b5f298f751bde); David Reilly, contributions by Andy Sommerville (2007) | Upstream [README terms](https://github.com/greyblue9/pacman-python/blob/2dc773e6c0a28cc03b55f7156c3b5f298f751bde/README.md) and [asset attribution](gymnasium/envs/pacman_2d/assets/greyblue9_pacman_python/ASSET_NOTES.md) |

License copies and their source URLs are listed in
[`LICENSES/sources.json`](LICENSES/sources.json). Refer to the retained file-level
notices and the linked upstream terms for each component.
