# Paper checkpoints and data

53 checkpoints and 11 evaluation datasets used in the paper. CoT includes imagined frames; nonCoT predicts actions only. K is the action-chunk length; K=∞ is open loop.

## Chunk policies

| Task | Variant | K=1 | K=3 | K=5 | K=10 | K=∞ |
| --- | --- | --- | --- | --- | --- | --- |
| Sokoban | CoT | — | [Model](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_k3_world_model) · [Data](https://huggingface.co/datasets/novastar111/sokoban_easy_cot_chunk_k3_train) | [Model](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_k5_world_model) · [Data](https://huggingface.co/datasets/novastar111/sokoban_easy_cot_chunk_k5_train) | [Model](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_k10_world_model) · [Data](https://huggingface.co/datasets/novastar111/sokoban_easy_cot_chunk_k10_train) | [Model](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_kinf_world_model) · [Data](https://huggingface.co/datasets/novastar111/sokoban_easy_cot_chunk_kinf_train) |
| Sokoban | nonCoT | [Model](https://huggingface.co/novastar111/sokoban_easy_noncot_chunk_k1_world_model) · [Data](https://huggingface.co/datasets/novastar111/sokoban_easy_noncot_chunk_k1_train) | [Model](https://huggingface.co/novastar111/sokoban_easy_noncot_chunk_k3_world_model) · [Data](https://huggingface.co/datasets/novastar111/sokoban_easy_noncot_chunk_k3_train) | [Model](https://huggingface.co/novastar111/sokoban_easy_noncot_chunk_k5_world_model) · [Data](https://huggingface.co/datasets/novastar111/sokoban_easy_noncot_chunk_k5_train) | [Model](https://huggingface.co/novastar111/sokoban_easy_noncot_chunk_k10_world_model) · [Data](https://huggingface.co/datasets/novastar111/sokoban_easy_noncot_chunk_k10_train) | [Model](https://huggingface.co/novastar111/sokoban_easy_noncot_chunk_kinf_world_model) · [Data](https://huggingface.co/datasets/novastar111/sokoban_easy_noncot_chunk_kinf_train) |
| PushT | CoT | — | [Model](https://huggingface.co/novastar111/pusht_cot_chunk_k3_world_model) · [Data](https://huggingface.co/datasets/novastar111/pusht_cot_chunk_k3_train) | [Model](https://huggingface.co/novastar111/pusht_cot_chunk_k5_world_model) · [Data](https://huggingface.co/datasets/novastar111/pusht_cot_chunk_k5_train) | [Model](https://huggingface.co/novastar111/pusht_cot_chunk_k10_world_model) · [Data](https://huggingface.co/datasets/novastar111/pusht_cot_chunk_k10_train) | [Model](https://huggingface.co/novastar111/pusht_cot_chunk_kinf_world_model) · [Data](https://huggingface.co/datasets/novastar111/pusht_cot_chunk_kinf_train) |
| PushT | nonCoT | [Model](https://huggingface.co/novastar111/pusht_noncot_chunk_k1_world_model) · [Data](https://huggingface.co/datasets/novastar111/pusht_noncot_chunk_k1_train) | [Model](https://huggingface.co/novastar111/pusht_noncot_chunk_k3_world_model) · [Data](https://huggingface.co/datasets/novastar111/pusht_noncot_chunk_k3_train) | [Model](https://huggingface.co/novastar111/pusht_noncot_chunk_k5_world_model) · [Data](https://huggingface.co/datasets/novastar111/pusht_noncot_chunk_k5_train) | [Model](https://huggingface.co/novastar111/pusht_noncot_chunk_k10_world_model) · [Data](https://huggingface.co/datasets/novastar111/pusht_noncot_chunk_k10_train) | [Model](https://huggingface.co/novastar111/pusht_singleshot) · [Data](https://huggingface.co/datasets/novastar111/pusht_singleshot_train) |
| Maze2D | CoT | — | [Model](https://huggingface.co/novastar111/maze2d_easy_cot_chunk_k3_world_model) · [Data](https://huggingface.co/datasets/novastar111/maze2d_easy_cot_chunk_k3_train) | [Model](https://huggingface.co/novastar111/maze2d_easy_cot_chunk_k5_world_model) · [Data](https://huggingface.co/datasets/novastar111/maze2d_easy_cot_chunk_k5_train) | [Model](https://huggingface.co/novastar111/maze2d_easy_cot_chunk_k10_world_model) · [Data](https://huggingface.co/datasets/novastar111/maze2d_easy_cot_chunk_k10_train) | [Model](https://huggingface.co/novastar111/maze2d_easy_cot_chunk_kinf_world_model) · [Data](https://huggingface.co/datasets/novastar111/maze2d_easy_cot_chunk_kinf_train) |
| Maze2D | nonCoT | [Model](https://huggingface.co/novastar111/maze2d_easy_noncot_chunk_k1_world_model) · [Data](https://huggingface.co/datasets/novastar111/maze2d_easy_noncot_chunk_k1_train) | [Model](https://huggingface.co/novastar111/maze2d_easy_noncot_chunk_k3_world_model) · [Data](https://huggingface.co/datasets/novastar111/maze2d_easy_noncot_chunk_k3_train) | [Model](https://huggingface.co/novastar111/maze2d_easy_noncot_chunk_k5_world_model) · [Data](https://huggingface.co/datasets/novastar111/maze2d_easy_noncot_chunk_k5_train) | [Model](https://huggingface.co/novastar111/maze2d_easy_noncot_chunk_k10_world_model) · [Data](https://huggingface.co/datasets/novastar111/maze2d_easy_noncot_chunk_k10_train) | [Model](https://huggingface.co/novastar111/maze2d_easy_noncot_chunk_kinf_world_model) · [Data](https://huggingface.co/datasets/novastar111/maze2d_easy_noncot_chunk_kinf_train) |
| Pacman | CoT | — | [Model](https://huggingface.co/novastar111/pacman_braidthin_cot_chunk_k3_world_model) · [Data](https://huggingface.co/datasets/novastar111/pacman_braidthin_cot_chunk_k3_train) | [Model](https://huggingface.co/novastar111/pacman_braidthin_cot_chunk_k5_world_model) · [Data](https://huggingface.co/datasets/novastar111/pacman_braidthin_cot_chunk_k5_train) | [Model](https://huggingface.co/novastar111/pacman_braidthin_cot_chunk_k10_world_model) · [Data](https://huggingface.co/datasets/novastar111/pacman_braidthin_cot_chunk_k10_train) | [Model](https://huggingface.co/novastar111/pacman_braidthin_cot_chunk_kinf_world_model) · [Data](https://huggingface.co/datasets/novastar111/pacman_braidthin_cot_chunk_kinf_train) |
| Pacman | nonCoT | [Model](https://huggingface.co/novastar111/pacman_braidthin_noncot_chunk_k1_world_model) · [Data](https://huggingface.co/datasets/novastar111/pacman_braidthin_noncot_chunk_k1_train) | [Model](https://huggingface.co/novastar111/pacman_braidthin_noncot_chunk_k3_world_model) · [Data](https://huggingface.co/datasets/novastar111/pacman_braidthin_noncot_chunk_k3_train) | [Model](https://huggingface.co/novastar111/pacman_braidthin_noncot_chunk_k5_world_model) · [Data](https://huggingface.co/datasets/novastar111/pacman_braidthin_noncot_chunk_k5_train) | [Model](https://huggingface.co/novastar111/pacman_braidthin_noncot_chunk_k10_world_model) · [Data](https://huggingface.co/datasets/novastar111/pacman_braidthin_noncot_chunk_k10_train) | [Model](https://huggingface.co/novastar111/pacman_braidthin_noncot_chunk_kinf_world_model) · [Data](https://huggingface.co/datasets/novastar111/pacman_braidthin_noncot_chunk_kinf_train) |

## Planning

| Task | Paper model | Checkpoint |
| --- | --- | --- |
| Sokoban | Branching | [Model](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_kinf_branch_multi) |
| Sokoban | Forward | [Model](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_kinf_compare2) |
| Pacman | Forward (v4) | [Model](https://huggingface.co/JiaxinGe/pacman_braidthin_forward_v4_shorttrap_only_kinf_world_model) |
| Sokoban | Branching, before RL | [Model](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_sft3k) |
| Sokoban | Forward, before RL | [Model](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_sft3k) |

## Sokoban reinforcement learning

| Reported RL step | Agima | Branching | Forward |
| --- | --- | --- | --- |
| 50 | [Model](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step50) | [Model](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step50) | [Model](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step50) |
| 100 | [Model](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step100) | [Model](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step100) | [Model](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step100) |
| 150 | [Model](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step150) | [Model](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step150) | [Model](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step150) |
| 200 | [Model](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step200) | [Model](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step200) | [Model](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step200) |

## Evaluation datasets

| Task | Paper benchmark | Dataset |
| --- | --- | --- |
| Sokoban | Ordinary hard, 4 boxes | [Data](https://huggingface.co/datasets/novastar111/sokoban_hard_init_state_pool) |
| Sokoban | Deadlock, 2 boxes | [Data](https://huggingface.co/datasets/novastar111/sokoban_eval_2box_tempting_deadlock_100) |
| Sokoban | Deadlock, 3 boxes | [Data](https://huggingface.co/datasets/novastar111/sokoban_eval_3box_certified_deadlock_100) |
| Sokoban | Deadlock, 4 boxes | [Data](https://huggingface.co/datasets/novastar111/sokoban_eval_4box_deadlock_100) |
| Sokoban | Structural Fork, 2 boxes | [Data](https://huggingface.co/datasets/novastar111/sokoban_eval_2box_structural_fork_100) |
| Sokoban | Structural Fork, 3 boxes | [Data](https://huggingface.co/datasets/novastar111/sokoban_eval_3box_structural_fork_100) |
| Sokoban | Structural Fork, 4 boxes | [Data](https://huggingface.co/datasets/novastar111/sokoban_eval_4box_structural_fork_100) |
| Maze2D | Hard, fixed start | [Data](https://huggingface.co/datasets/novastar114/maze2d_hard_native256_stepmsg_fixedstart_plain) |
| Pacman | Hard | [Data](https://huggingface.co/datasets/novastar111/pacman_eval_hard_braidthin_random100_20260817) |
| Pacman | Route-order traps | [Data](https://huggingface.co/datasets/JiaxinGe/pacman_route_order_test_sets) |
| PushT | FAR | [Data](https://huggingface.co/datasets/novastar111/pusht_eval_far_balanced100_20260810) |

PushT success uses final coverage > 0.80.

Pinned revisions, file lists, and hashes: [models.json](artifacts/models.json) · [datasets.json](artifacts/datasets.json).
