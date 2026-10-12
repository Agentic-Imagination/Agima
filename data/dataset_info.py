# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# VLM Gym Dataset Configuration
# =============================================================================
#
# This file dynamically generates dataset registry entries for all VLM Gym tasks.
#
# Entries are generated from VLM_GYM_TASKS and VLM_GYM_LOSS_TYPES below.
# Each task/loss pair has a training key and an evaluation key:
#   - vlm_gym_{task}_{loss}
#   - vlm_gym_{task}_{loss}_evalonce
#
# Example usage:
#   vlm_gym_sokoban_easy_celoss          # Sokoban with CE+MSE loss
#   vlm_gym_maze2d_easy_celoss_no_mse     # Maze2D with CE loss only
#   vlm_gym_pusht_v0_celoss_evalonce     # PushT evaluation
#
# To add a new task:
#   Add the task directory name to VLM_GYM_TASKS list below.
#
# To change the base data path:
#   Set AGIMA_DATA_ROOT or use the trainer's explicit data-path overrides.
#
# =============================================================================

import os
from pathlib import Path

from .interleave_datasets import (
    UnifiedEditIterableDataset,
    VLMGymDatasetCELOSS,
    VLMGymDatasetCELOSSNoMSE,
    VLMGymDatasetCELOSSTextOnly,
    VLMGymDatasetCELOSSTextOnly_EvalOnce,
    VLMGymDatasetMSELOSSOnly,
    VLMGymDatasetCELOSS_EvalOnce,
    VLMGymDatasetCELOSSNoMSE_EvalOnce,
    VLMGymDatasetMSELOSSOnly_EvalOnce,
    VLMGymDatasetSokobanInterleavedCoT,
    VLMGymDatasetSokobanInterleavedCoT_EvalOnce,
    VLMGymDatasetSokobanInterleavedCoTEnvFB,
    VLMGymDatasetSokobanInterleavedCoTEnvFB_EvalOnce,
)
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .interleaved_iterable import InterleavedJSONLIterableDataset

# =============================================================================
# VLM Gym Configuration
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[1]
VLM_GYM_BASE_PATH = str(Path(os.environ.get(
    "AGIMA_DATA_ROOT", _REPO_ROOT / "datasets"
)).expanduser().resolve())
_EXAMPLE_DATA_ROOT = Path(os.environ.get(
    "AGIMA_EXAMPLE_DATA_ROOT", _REPO_ROOT / "data" / "bagel_example"
)).expanduser().resolve()

# All available tasks in VLM Gym
VLM_GYM_TASKS = [
    "colorization_v5",
    "counting-guess_only_v5",
    "counting-mark_all_v5",
    "jigsaw-reorder_v5",
    "jigsaw-swap_v5",
    "match_equation-bfs_v7",
    "match_equation-dfs_v7",
    "match_equation-sos_v7",
    "match_move_fix3_unit_v7",
    "mental_rotation_2d_v5",
    "mental_rotation_3d_pad3_by_axis_v7",
    "mujoco_pick_and_place_v5",
    "mujoco_reach_v5",
    "patch_reassembly_alt_v5",
    "reference_dot_v5",
    "sliding_block_v5",
    "toy_maze_2d_v5",
    "toy_maze_3d_v5",
    "zoom_in-reorder_v5",
    "zoom_in-swap_v5",
    "video_unshuffle-reorder_v7",
    "video_unshuffle-swap_v7",
    "mental_rotation_3d_objaverse_pad3_by_axis_v7",
    "sokoban_short_v0",
    "sokoban_easy",
    "sokoban_hard",
    "pusht_v0",
    "maze2d_easy",
    "maze2d_hard",
    "maze2d_easy_imagined_cot_v0",
    "maze2d_hard_imagined_cot_v0",
    "pacman_2d_ultrahard_v0",
    "pacman_2d_easy_v0",
    "pacman_2d_hard_v0",
    "sokoban_easy_imagined_cot_v0",
    "sokoban_hard_imagined_cot_v0",
    "pacman_2d_ultrahard_imagined_cot_v0",
    "pacman_2d_easy_imagined_cot_v0",
    "pusht_imagined_cot_v0",
]

# Loss type configurations: (suffix, dataset_class, eval_dataset_class)
VLM_GYM_LOSS_TYPES = {
    "celoss": (VLMGymDatasetCELOSS, VLMGymDatasetCELOSS_EvalOnce),
    "celoss_no_mse": (VLMGymDatasetCELOSSNoMSE, VLMGymDatasetCELOSSNoMSE_EvalOnce),
    "celoss_text_only": (VLMGymDatasetCELOSSTextOnly, VLMGymDatasetCELOSSTextOnly_EvalOnce),
    "mse_loss_only": (VLMGymDatasetMSELOSSOnly, VLMGymDatasetMSELOSSOnly_EvalOnce),
    "interleaved_cot": (VLMGymDatasetSokobanInterleavedCoT, VLMGymDatasetSokobanInterleavedCoT_EvalOnce),
    "interleaved_cot_envfb": (VLMGymDatasetSokobanInterleavedCoTEnvFB, VLMGymDatasetSokobanInterleavedCoTEnvFB_EvalOnce),
}


def _task_to_key(task_name: str) -> str:
    """Convert task directory name to a clean key (e.g., 'jigsaw-swap_v5' -> 'jigsaw_swap')."""
    # Remove _v5/_v7 suffix and replace hyphens with underscores
    key = task_name.replace("_v7", "").replace("_v5", "").replace("_v0", "").replace("-", "_")
    return key


def _generate_vlm_gym_registry() -> dict:
    """Generate DATASET_REGISTRY entries for all VLM Gym task/loss combinations."""
    registry = {}
    for task in VLM_GYM_TASKS:
        task_key = _task_to_key(task)
        for loss_name, (train_cls, eval_cls) in VLM_GYM_LOSS_TYPES.items():
            # Training entry: vlm_gym_{task}_{loss}
            registry[f"vlm_gym_{task_key}_{loss_name}"] = train_cls
            # Eval entry: vlm_gym_{task}_{loss}_evalonce
            registry[f"vlm_gym_{task_key}_{loss_name}_evalonce"] = eval_cls
    return registry


def _generate_vlm_gym_info() -> dict:
    """Generate DATASET_INFO entries for all VLM Gym task/loss combinations."""
    info = {}
    for task in VLM_GYM_TASKS:
        task_key = _task_to_key(task)
        task_path = f"{VLM_GYM_BASE_PATH}/{task}"

        for loss_name in VLM_GYM_LOSS_TYPES.keys():
            # Training entry
            train_key = f"vlm_gym_{task_key}_{loss_name}"
            info[train_key] = {
                f"vlm_gym_{task_key}_train": {
                    "data_dir": f"{task_path}/train/",
                    "jsonl_path": f"{task_path}/train/",
                    "num_total_samples": 10000,
                    "task_name": task_key,
                },
                f"vlm_gym_{task_key}_val": {
                    "data_dir": f"{task_path}/train/",
                    "jsonl_path": f"{task_path}/train/",
                    "num_total_samples": 1000,
                    "task_name": task_key,
                },
            }

            # Eval-once entry (val only)
            eval_key = f"vlm_gym_{task_key}_{loss_name}_evalonce"
            info[eval_key] = {
                f"vlm_gym_{task_key}_val": {
                    "data_dir": f"{task_path}/train/",
                    "jsonl_path": f"{task_path}/train/",
                    "num_total_samples": 1000,
                    "task_name": task_key,
                },
            }
    return info


# =============================================================================
# Dataset Registry
# =============================================================================

DATASET_REGISTRY = {
    # Non-VLM Gym datasets
    "t2i_pretrain": T2IIterableDataset,
    "vlm_sft": SftJSONLIterableDataset,
    "unified_edit": UnifiedEditIterableDataset,
    "interleaved_sft": InterleavedJSONLIterableDataset,
    # VLM Gym datasets (dynamically generated)
    **_generate_vlm_gym_registry(),
}

# =============================================================================
# Dataset Info
# =============================================================================

DATASET_INFO = {
    # Non-VLM Gym dataset info
    "t2i_pretrain": {
        "t2i": {
            "data_dir": str(_EXAMPLE_DATA_ROOT / "t2i"),
            "num_files": 10,
            "num_total_samples": 1000,
        },
    },
    "unified_edit": {
        "seedxedit_multi": {
            "data_dir": str(_EXAMPLE_DATA_ROOT / "editing" / "seedxedit_multi"),
            "num_files": 10,
            "num_total_samples": 1000,
            "parquet_info_path": str(_EXAMPLE_DATA_ROOT / "editing" / "parquet_info" / "seedxedit_multi.json"),
        },
    },
    "vlm_sft": {
        "llava_ov": {
            "data_dir": str(_EXAMPLE_DATA_ROOT / "vlm" / "images"),
            "jsonl_path": str(_EXAMPLE_DATA_ROOT / "vlm" / "llava_ov_si.jsonl"),
            "num_total_samples": 1000,
        },
    },
    "interleaved_sft": {
        "auto_trigger": {
            "data_dir": str(_EXAMPLE_DATA_ROOT / "test"),
            "jsonl_path": str(_EXAMPLE_DATA_ROOT / "test" / "train_auto_trigger.jsonl"),
            "num_total_samples": 1000,
        },
    },
    # VLM Gym dataset info (dynamically generated)
    **_generate_vlm_gym_info(),
}
