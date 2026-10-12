# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# Modified for Agima: reject empty distributed data partitions.

import random
import torch


class DistributedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_name, local_rank=0, world_size=1, num_workers=8):
        self.dataset_name = dataset_name
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.rng = random.Random()
        self.data_paths = None

    def get_data_paths(self, *args, **kwargs):
        raise NotImplementedError

    def set_epoch(self, seed=42):
        if self.data_paths is None:
            return
        if not self.data_paths:
            raise ValueError(f"Dataset {self.dataset_name!r} has no data entries")
        if isinstance(self.data_paths[0], tuple):
            data_paths = sorted(self.data_paths, key=lambda x: (x[0], x[1]))
        elif isinstance(self.data_paths[0], str):
            data_paths = sorted(self.data_paths)
        else:
            raise ValueError(f"Unknown data_paths type: {type(self.data_paths[0])}")

        self.rng.seed(seed)
        self.rng.shuffle(data_paths)

        num_files_per_rank = len(data_paths) // self.world_size
        local_start = self.local_rank * num_files_per_rank
        local_end = (self.local_rank + 1) * num_files_per_rank
        self.num_files_per_rank = num_files_per_rank
        self.data_paths_per_rank = data_paths[local_start:local_end]

    def get_data_paths_per_worker(self):
        if self.data_paths is None:
            return None

        info = torch.utils.data.get_worker_info()
        if info is None:
            # Single worker: Use all files assigned to the rank
            if not self.data_paths_per_rank:
                raise ValueError(f"Dataset {self.dataset_name!r} has {len(self.data_paths)} entries for "
                                 f"{self.world_size} ranks; rank {self.local_rank} has no entries")
            return self.data_paths_per_rank, 0

        worker_id = info.id
        num_files_per_worker = self.num_files_per_rank // info.num_workers
        start = num_files_per_worker * worker_id
        end = num_files_per_worker * (worker_id + 1)
        data_paths_per_worker = self.data_paths_per_rank[start:end]
        if not data_paths_per_worker:
            raise ValueError(
                f"Dataset {self.dataset_name!r} has {self.num_files_per_rank} entries for "
                f"{info.num_workers} workers on rank {self.local_rank}; reduce num_workers or add data"
            )

        return data_paths_per_worker[::-1], worker_id

    def __iter__(self):
        raise NotImplementedError
