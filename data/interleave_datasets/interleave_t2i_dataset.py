# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# Modified for Agima: validate JSONL partitions, row limits and resume cursors.

import pyarrow.parquet as pq

from ..distributed_iterable_dataset import DistributedIterableDataset
from ..parquet_utils import get_parquet_data_paths, init_arrow_pf_fs
from ..data_utils import pil_img2rgb
import gzip
import json
import os
import traceback


def _open_jsonl(path):
    """Open a .jsonl or .jsonl.gz file in text mode."""
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')


def _worker_resume_cursor(data_status, worker_id):
    if data_status is None:
        return None
    if not isinstance(data_status, dict):
        raise ValueError("JSONL data_status must map worker IDs to saved row cursors")
    if worker_id in data_status:
        return data_status[worker_id]
    if str(worker_id) in data_status:
        return data_status[str(worker_id)]
    raise ValueError(f"JSONL checkpoint has no cursor for worker {worker_id}; keep the original worker configuration")


def _jsonl_resume_start(data_status, worker_id, row_count):
    """Resume after the saved worker-local row; accept legacy int and saved [int]."""
    cursor = _worker_resume_cursor(data_status, worker_id)
    if data_status is None:
        return 0
    if isinstance(cursor, (list, tuple)) and len(cursor) == 1:
        cursor = cursor[0]
    if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor < row_count:
        raise ValueError(
            f"Invalid JSONL resume cursor {cursor!r} for worker {worker_id} with {row_count} rows; "
            "expected an integer or [integer] within the original worker partition"
        )
    return cursor + 1


import re as _re
# Each ordered-pair row begins with a unique, fixed-width, zero-padded
# "__bagel_order_key" as its FIRST JSON field. We parse it cheaply (string
# prefix, no full json.loads) to build a lightweight row index so the
# non-streaming loader does not have to hold every base64-image line in RAM.
_ORDER_KEY_RE = _re.compile(r'^\s*\{\s*"__bagel_order_key"\s*:\s*"([^"]+)"')


def _extract_order_key(line):
    """Return the __bagel_order_key prefix of a JSONL line, or None if absent."""
    m = _ORDER_KEY_RE.match(line)
    return m.group(1) if m else None
from PIL import Image, ImageFile, PngImagePlugin
from pathlib import Path
import math
import torch
import torch.distributed as dist

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


# -----------------------------------------------------------------------------
# DEBUG MODE: dump the exact inputs handed to the tokenizer / image transforms
# BEFORE tokenization, for the first N samples on rank 0. Pure opt-in: when the
# env switch is off this adds nothing to the data path. Toggle via env vars:
#   BAGEL_DEBUG_INPUTS=1            enable the dump
#   BAGEL_DEBUG_INPUTS_N=3          how many samples to dump (default 3)
#   BAGEL_DEBUG_INPUTS_MAXCHARS=4000  cap per-text chars (default 4000; <=0 = no cap)
#   BAGEL_DEBUG_INPUTS_NIDS=16      also print post-encode token ids: how many head ids
#                                   to show (default 16; 0 = count only; <0 = all ids)
# A fresh sample is detected when sequence_plan is empty (parse_row calls _init_data
# first). The dump shows, in call order, every text segment's raw string (pre-encode)
# with its loss/bos/eos flags, the token ids it encodes to, and every image segment's
# vae/vit/loss plan + PIL size. (Use num_workers<=1 for a clean, non-interleaved dump.)
# -----------------------------------------------------------------------------
_DBG_INPUTS = os.environ.get("BAGEL_DEBUG_INPUTS", "0").lower() not in ("0", "", "false", "no")
_DBG_INPUTS_N = int(os.environ.get("BAGEL_DEBUG_INPUTS_N", "3"))
_DBG_INPUTS_MAXCHARS = int(os.environ.get("BAGEL_DEBUG_INPUTS_MAXCHARS", "4000"))
_DBG_INPUTS_NIDS = int(os.environ.get("BAGEL_DEBUG_INPUTS_NIDS", "16"))
_dbg_inputs_state = {"sample": 0}


def _dbg_inputs_on():
    return _DBG_INPUTS and os.environ.get("RANK", "0") in ("0", "")


def _dbg_inputs_new_sample(data):
    """If this is a sample's first segment, bump the counter + print a header.
    Returns True while the current sample is still within the dump budget."""
    if len(data.get("sequence_plan", [])) == 0:
        _dbg_inputs_state["sample"] += 1
        if _dbg_inputs_state["sample"] <= _DBG_INPUTS_N:
            print(f"\n===== [BAGEL_DBG_INPUTS] SAMPLE #{_dbg_inputs_state['sample']} "
                  f"(pre-tokenizer input dump, rank={os.environ.get('RANK', '0')}) =====", flush=True)
    return _dbg_inputs_state["sample"] <= _DBG_INPUTS_N


def _dbg_inputs_trunc(text):
    if _DBG_INPUTS_MAXCHARS > 0 and len(text) > _DBG_INPUTS_MAXCHARS:
        return text[:_DBG_INPUTS_MAXCHARS] + f"...<+{len(text) - _DBG_INPUTS_MAXCHARS} more chars>"
    return text


class InterleavedBaseIterableDataset(DistributedIterableDataset):

    def _init_data(self):
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'num_tokens': 0,
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True,
                  prepend_bos=True, append_eos=True):
        """Add a text segment to the sequence_plan.

        prepend_bos / append_eos default to True (single-segment turn). Set
        either to False when this text run is part of a multi-segment turn —
        e.g. interleaved CoT where a single assistant turn has text → image →
        text → image → text. In that case the FIRST run uses prepend_bos=True,
        append_eos=False; MIDDLE runs use prepend_bos=False, append_eos=False;
        the LAST run uses prepend_bos=False, append_eos=True.
        """
        _dbg = _dbg_inputs_on() and _dbg_inputs_new_sample(data)
        if _dbg:
            _tag = "OUTPUT/CE-target" if need_loss else "INPUT/context"
            print(f"[BAGEL_DBG_INPUTS] TEXT  seg#{len(data['sequence_plan'])}  {_tag}  "
                  f"loss={int(need_loss)} prepend_bos={bool(prepend_bos)} append_eos={bool(append_eos)} "
                  f"enable_cfg={int(enable_cfg)}\n  text={_dbg_inputs_trunc(text)!r}", flush=True)
        text_ids = self.tokenizer.encode(text)
        if _dbg:
            if _DBG_INPUTS_NIDS == 0:
                print(f"    -> tokenized: {len(text_ids)} token ids (raw encode, before bos/eos)", flush=True)
            elif _DBG_INPUTS_NIDS < 0 or len(text_ids) <= _DBG_INPUTS_NIDS:
                print(f"    -> tokenized: {len(text_ids)} token ids = {text_ids}", flush=True)
            else:
                print(f"    -> tokenized: {len(text_ids)} token ids; "
                      f"first {_DBG_INPUTS_NIDS}={text_ids[:_DBG_INPUTS_NIDS]} ... last 8={text_ids[-8:]}", flush=True)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
                'prepend_bos': bool(prepend_bos),
                'append_eos': bool(append_eos),
            }
        )
        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
        assert need_loss or need_vae or need_vit

        if _dbg_inputs_on() and _dbg_inputs_new_sample(data):
            try:
                _w, _h = image.size
            except Exception:
                _w = _h = "?"
            print(f"[BAGEL_DBG_INPUTS] IMAGE seg#{len(data['sequence_plan'])}  "
                  f"need_loss={int(need_loss)} need_vae={int(need_vae)} need_vit={int(need_vit)} "
                  f"enable_cfg={int(enable_cfg)}  PIL_size={_w}x{_h}", flush=True)

        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)

        if need_vae:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.vit_transform.stride ** 2
            data['image_tensor_list'].append(vit_image_tensor)

        return data

    def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True):
        assert int(need_loss) + int(need_vae) == 1

        if need_loss:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        elif need_vae:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        return data
        
class ParquetStandardIterableDataset(DistributedIterableDataset):

    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform, 
        data_dir_list, num_used_data, parquet_info,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        vit_transform: input transform for vit model.
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data, parquet_info)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data, parquet_info):
        row_groups = []
        for data_dir, num_data_path in zip(data_dir_list, num_used_data):
            data_paths = get_parquet_data_paths([data_dir], [num_data_path])
            for data_path in data_paths:
                if data_path in parquet_info.keys():
                    num_row_groups = parquet_info[data_path]['num_row_groups']
                    for rg_idx in range(num_row_groups):
                        row_groups.append((data_path, rg_idx))
        return row_groups

    def parse_row(self, row):
        raise NotImplementedError

    def __iter__(self):
        file_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            global_row_group_start_id = self.data_status[worker_id][0]
            row_start_id = self.data_status[worker_id][1] + 1
        else:
            global_row_group_start_id = 0
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at global_rg#{global_row_group_start_id}, row#{row_start_id}"
        )

        while True:
            file_paths_per_worker_ = file_paths_per_worker[global_row_group_start_id:]
            for global_row_group_idx, (parquet_file_path, row_group_id) in enumerate(
                file_paths_per_worker_, start=global_row_group_start_id
            ):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    try:
                        fr = pq.ParquetFile(f)
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]
                    except Exception as e:
                        print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                        continue

                    for row_idx, row in df.iterrows():
                        try:
                            data = self.parse_row(row)
                            if len(data) == 0:
                                continue
                            data['data_indexes'] = {
                                "data_indexes": [global_row_group_idx, row_idx],
                                "worker_id": worker_id,
                                "dataset_name": self.dataset_name,
                            }
                        except Exception as e:
                            print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                            continue
                        yield data

                    row_start_id = 0
            global_row_group_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")

class JsonLIterableDataset(DistributedIterableDataset):

    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform, 
        data_dir_list, num_used_data, jsonl_path_list,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        **kwargs,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.jsonl_streaming = os.environ.get("BAGEL_JSONL_STREAMING", "0") == "1"
        self.streaming_num_used_data = int(num_used_data[0]) if num_used_data else None
        self.data_paths = self.get_data_paths(
            jsonl_path_list, 
            num_used_data, 
        )
        self.set_epoch()

    def _expand_jsonl_paths(self, jsonl_path_list):
        if not jsonl_path_list:
            raise ValueError(f"Dataset {self.dataset_name!r} has no JSONL paths")
        if os.path.isdir(jsonl_path_list[0]):
            base = jsonl_path_list[0]
            files = sorted([fn for fn in os.listdir(base) if fn.endswith('.jsonl') or fn.endswith('.jsonl.gz')])
            return [os.path.join(base, fn) for fn in files]
        return jsonl_path_list

    def get_data_paths(
        self, 
        jsonl_path_list, 
        num_used_data
    ):
        data_paths = []

        jsonl_path_list = self._expand_jsonl_paths(jsonl_path_list)
        if self.jsonl_streaming:
            return jsonl_path_list

        # Memory-safe index mode: if rows carry a unique "__bagel_order_key"
        # prefix (the ordered-pair datasets), index each row as a lightweight
        # (order_key, file_idx, line_no) tuple instead of holding the full
        # base64-image line in RAM on every rank. The base set_epoch sorts
        # tuples by (order_key, file_idx) -- identical to sorting the raw lines
        # (key is the unique prefix) -- then shuffles with the same seed and
        # slices per rank, so the per-rank/per-worker row order is byte-for-byte
        # the SAME as the old raw-line path. Content is materialized lazily,
        # only for the rows this rank/worker owns (see __iter__).
        self._index_mode = False
        self._jsonl_files = jsonl_path_list
        limit = num_used_data[0]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("JSONL num_used_data must contain a positive integer row limit")
        index = []
        for file_idx, jsonl_path in enumerate(jsonl_path_list):
            if len(index) >= limit:
                break
            with _open_jsonl(jsonl_path) as f:
                for line_no, line in enumerate(f):
                    if not line.strip():
                        continue
                    if len(index) >= limit:
                        break
                    order_key = _extract_order_key(line)
                    if order_key is None:
                        # Dataset without order keys: keep raw lines while
                        # applying the same nonblank global row limit.
                        index = None
                        break
                    index.append((order_key, file_idx, line_no))
            if index is None:
                break

        if index is not None:
            self._index_mode = True
            return index

        # Raw-line fallback uses the same global cap as the indexed path.
        for jsonl_path in jsonl_path_list:
            if len(data_paths) >= limit:
                break
            with _open_jsonl(jsonl_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data_paths.append(line)
                    if len(data_paths) >= limit:
                        break
        return data_paths

    def _materialize_lines(self, index_rows):
        """Read the raw JSONL lines for ``index_rows`` ((key, file_idx, line_no)
        tuples) in one sequential pass per file. Only the rows this rank/worker
        owns are kept in RAM. Returns raw lines in the SAME order as index_rows."""
        wanted = {}
        for key, file_idx, line_no in index_rows:
            wanted.setdefault(file_idx, {})[line_no] = None
        for file_idx, line_map in wanted.items():
            with _open_jsonl(self._jsonl_files[file_idx]) as f:
                for line_no, line in enumerate(f):
                    if line_no in line_map:
                        line_map[line_no] = line
        return [wanted[fi][ln] for (key, fi, ln) in index_rows]

    def set_epoch(self, seed=42):
        if not self.jsonl_streaming:
            return super().set_epoch(seed)

        data_paths = sorted(self.data_paths)
        self.rng.seed(seed)
        self.rng.shuffle(data_paths)
        self.data_paths_per_rank = data_paths[self.local_rank::self.world_size]

    def _get_streaming_paths_per_worker(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            paths, worker_id, worker_count = self.data_paths_per_rank, 0, 1
        else:
            paths = self.data_paths_per_rank[info.id::info.num_workers]
            worker_id, worker_count = info.id, info.num_workers
        if not paths:
            raise ValueError(
                f"Dataset {self.dataset_name!r} has no JSONL shards for rank {self.local_rank} "
                f"worker {worker_id}; provide at least one shard per rank and worker"
            )
        return paths, worker_id, worker_count

    def _streaming_resume_start(self, paths, worker_id):
        cursor = _worker_resume_cursor(self.data_status, worker_id)
        if self.data_status is None:
            return 0
        if isinstance(cursor, int) and not isinstance(cursor, bool) and cursor >= 0:
            return cursor + 1
        if (not isinstance(cursor, (list, tuple)) or len(cursor) != 2
                or not isinstance(cursor[0], str) or isinstance(cursor[1], bool)
                or not isinstance(cursor[1], int) or cursor[1] < 0 or cursor[0] not in paths):
            raise ValueError(f"Invalid streaming JSONL resume cursor {cursor!r} for worker {worker_id}; "
                             "expected [assigned_shard_path, line_index] or a legacy integer")
        seen = 0
        for path in paths:
            with _open_jsonl(path) as stream:
                for line_index, line in enumerate(stream):
                    if line.strip():
                        seen += 1
                    if path == cursor[0] and line_index == cursor[1]:
                        if not line.strip():
                            raise ValueError(f"Streaming JSONL resume cursor points to a blank line: {cursor!r}")
                        return seen
            if path == cursor[0]:
                break
        raise ValueError(f"Streaming JSONL resume cursor is past the end of its shard: {cursor!r}")

    def _iter_streaming_rows(self, file_paths_per_worker, worker_id, worker_count, row_start_id):
        rank_worker_count = max(1, self.world_size * worker_count)
        if self.streaming_num_used_data is None:
            worker_row_limit = None
        else:
            if self.streaming_num_used_data < rank_worker_count:
                raise ValueError(
                    f"Streaming JSONL row limit {self.streaming_num_used_data} is smaller than "
                    f"{rank_worker_count} rank/worker partitions; increase num_used_data or reduce parallelism"
                )
            worker_row_limit = max(1, self.streaming_num_used_data // rank_worker_count)

        while True:
            yielded_rows = 0
            valid_rows = 0
            for jsonl_path in file_paths_per_worker:
                with _open_jsonl(jsonl_path) as f:
                    for row_idx, row in enumerate(f):
                        if not row.strip():
                            continue
                        if worker_row_limit is not None and yielded_rows >= worker_row_limit:
                            break
                        if yielded_rows < row_start_id:
                            yielded_rows += 1
                            continue
                        data = self.parse_row(row)
                        yielded_rows += 1
                        if len(data) == 0:
                            continue
                        valid_rows += 1
                        data['data_indexes'] = {
                            "data_indexes": [jsonl_path, row_idx],
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                        yield data
                if worker_row_limit is not None and yielded_rows >= worker_row_limit:
                    break

            if row_start_id > yielded_rows:
                raise ValueError(f"Streaming JSONL resume cursor {row_start_id - 1} exceeds the worker's row limit")
            if valid_rows == 0 and row_start_id == 0:
                raise ValueError(f"Dataset {self.dataset_name!r} produced no valid rows on rank "
                                 f"{self.local_rank} worker {worker_id}; check the assigned JSONL shards and schema")
            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
    
    def parse_row(self, row):
        raise NotImplementedError

    def __iter__(self):
        if self.jsonl_streaming:
            file_paths_per_worker, worker_id, worker_count = self._get_streaming_paths_per_worker()
            row_start_id = self._streaming_resume_start(file_paths_per_worker, worker_id)
            print(
                f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
                f"streaming {len(file_paths_per_worker)} jsonl files from row#{row_start_id}"
            )
            yield from self._iter_streaming_rows(file_paths_per_worker, worker_id, worker_count, row_start_id)
            return

        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if getattr(self, "_index_mode", False):
            # data_paths_per_worker are lightweight (key, file_idx, line_no)
            # tuples; read the raw lines for ONLY this worker's slice, once,
            # in the exact (already sorted+shuffled+sharded) order.
            data_paths_per_worker = self._materialize_lines(data_paths_per_worker)
        row_start_id = _jsonl_resume_start(self.data_status, worker_id, len(data_paths_per_worker))
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            skipped_rows = 0
            valid_rows = 0
            for row_idx, data in enumerate(data_paths_per_worker_, start=row_start_id):
                num_tokens = 0
                image_tensor_list = []
                text_ids_list = []
                sequence_plan = []

                # try:
                data = self.parse_row(data)
                if len(data) == 0:
                    skipped_rows += 1
                    if skipped_rows % 1000 == 0:
                        print(f"[Rank {self.local_rank}] Skipped {skipped_rows} total rows due to empty parse_row.", flush=True)
                    continue
                skipped_rows = 0 # reset if we found a valid row
                valid_rows += 1
                # except Exception as e:
                #     print(f'Error {e} in row#{row_idx}')
                    # continue
                data['data_indexes'] = {
                    "data_indexes": [row_idx],
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                yield data

            if valid_rows == 0 and row_start_id == 0:
                raise ValueError(f"Dataset {self.dataset_name!r} produced no valid rows on rank "
                                 f"{self.local_rank} worker {worker_id}; check the JSONL schema")
            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class JsonLIterableDatasetEvalOnce(JsonLIterableDataset):
    def get_data_paths(self, jsonl_path_list, num_used_data):
        path0 = jsonl_path_list[0]
        if os.path.isdir(path0):
            files = sorted([fn for fn in os.listdir(path0) if fn.endswith('.jsonl') or fn.endswith('.jsonl.gz')])
            assert len(files) > 0, f"Empty jsonl dir: {path0}"
            path0 = os.path.join(path0, files[0])

        data_paths = []
        with _open_jsonl(path0) as f:
            for line in f:
                if line.strip():
                    data_paths.append(line)
        return data_paths

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        row_start_id = _jsonl_resume_start(self.data_status, worker_id, len(data_paths_per_worker))

        print(
            f"[EVAL-ONCE] rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"start at row#{row_start_id}, total_rows(worker_split)={len(data_paths_per_worker)}"
        )

        for row_idx, row in enumerate(data_paths_per_worker[row_start_id:], start=row_start_id):
            data = self.parse_row(row)
            if len(data) == 0:
                continue
            data["data_indexes"] = {
                "data_indexes": [row_idx],
                "worker_id": worker_id,
                "dataset_name": self.dataset_name,
            }
            yield data


class JsonLIterableDatasetEvalOnceShardPad(JsonLIterableDatasetEvalOnce):
    def __init__(self, *args, eval_log_dir=None, eval_run_tag="run0", **kwargs):
        super().__init__(*args, **kwargs)
        # print(f"[SHARDPAD] rank{dist.get_rank()} N={len(self.data_paths)} steps={(len(self.data_paths)+dist.get_world_size()-1)//dist.get_world_size()} eval_log_dir={self.eval_log_dir}", flush=True)

        self.eval_log_dir = eval_log_dir
        self.eval_run_tag = eval_run_tag
    def __iter__(self):
        # IMPORTANT: eval should run with DataLoader(num_workers=0)
        rows = self.data_paths
        N = len(rows)

        rank = dist.get_rank()
        world = dist.get_world_size()
        steps = (N + world - 1) // world  # ceil(N/world)

        # choose a dummy row that is guaranteed parseable & not forbidden
        dummy_row = None
        for r in rows:
            d = self.parse_row(r)
            if len(d) != 0:
                dummy_row = r
                break
        if dummy_row is None:
            print(f"[SHARDPAD] rank{dist.get_rank()} No valid row found for dummy_row", flush=True)
            raise RuntimeError("[EVAL] No valid row found for dummy_row")
        log_f = None
        if self.eval_log_dir is not None:
            os.makedirs(self.eval_log_dir, exist_ok=True)
            log_path = os.path.join(self.eval_log_dir, f"eval_manifest_{self.eval_run_tag}_rank{rank}.jsonl")
            log_f = open(log_path, "w", encoding="utf-8")

        for t in range(steps):
            idx = rank + t * world
            data_json = {}
            if idx < N:
                row = rows[idx]
                data = self.parse_row(row)
                data_json = json.loads(row)
                if len(data) == 0:
                    # treat forbidden/bad rows as dummy (do NOT skip)
                    data = self.parse_row(dummy_row)
                    data_json = json.loads(dummy_row)
                    weight = 0.0
                else:
                    weight = 1.0
            else:
                data = self.parse_row(dummy_row)
                data_json = json.loads(dummy_row)
                weight = 0.0
            if log_f is not None:
                log_f.write(json.dumps(data_json) + "\n")

            data["eval_weight"] = weight
            data["data_indexes"] = {"data_indexes":[idx], "worker_id":0, "dataset_name": self.dataset_name}
            yield data
        
        if log_f is not None:
            log_f.close()
        return


# class JsonLIterableDatasetEvalOnceShardPad(JsonLIterableDatasetEvalOnce):
#     def __init__(self, *args, eval_log_dir=None, eval_run_tag="run0", **kwargs):
#         super().__init__(*args, **kwargs)
#         self.eval_log_dir = eval_log_dir
#         self.eval_run_tag = eval_run_tag

#     def __iter__(self):
#         rows = self.data_paths
#         N = len(rows)

#         rank = dist.get_rank()
#         world = dist.get_world_size()
#         steps = (N + world - 1) // world  # ceil

#         # pick a dummy row that is parseable (and ideally not forbidden)
#         dummy_row = None
#         for r in rows:
#             d = self.parse_row(r)
#             if len(d) != 0:
#                 dummy_row = r
#                 break
#         if dummy_row is None:
#             raise RuntimeError("[EVAL] No valid row found for dummy_row")

#         # open per-rank log file
#         log_f = None
#         if self.eval_log_dir is not None:
#             os.makedirs(self.eval_log_dir, exist_ok=True)
#             log_path = os.path.join(self.eval_log_dir, f"eval_manifest_{self.eval_run_tag}_rank{rank}.jsonl")
#             log_f = open(log_path, "w", encoding="utf-8")

#         def _row_hash(row_str: str):
#             try:
#                 j = json.loads(row_str)
#                 return j.get("hash", None)
#             except Exception:
#                 return None

#         for t in range(steps):
#             idx = rank + t * world
#             if idx < N:
#                 row = rows[idx]
#                 data = self.parse_row(row)
#                 if len(data) == 0:
#                     # forbidden/bad -> use dummy but weight=0
#                     row = dummy_row
#                     data = self.parse_row(row)
#                     weight = 0.0
#                     is_dummy = True
#                     reason = "invalid_or_forbidden"
#                 else:
#                     weight = 1.0
#                     is_dummy = False
#                     reason = "real"
#             else:
#                 row = dummy_row
#                 data = self.parse_row(row)
#                 weight = 0.0
#                 is_dummy = True
#                 reason = "pad"

#             # attach meta for downstream (optional but handy)
#             meta = {
#                 "global_idx": int(idx),
#                 "t": int(t),
#                 "rank": int(rank),
#                 "world": int(world),
#                 "weight": float(weight),
#                 "is_dummy": bool(is_dummy),
#                 "reason": reason,
#                 "row_hash": _row_hash(row),
#             }
#             data["eval_weight"] = weight
#             data["eval_meta"] = meta

#             # write manifest
#             if log_f is not None:
#                 log_f.write(json.dumps(meta) + "\n")

#             yield data

#         if log_f is not None:
#             log_f.close()
#         return
