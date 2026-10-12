# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# Modified for Agima: portable asset-path defaults and documentation.

import io
import os
import random
from pathlib import Path
from PIL import Image, ImageFile, PngImagePlugin
import json
import base64
from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset, JsonLIterableDataset, JsonLIterableDatasetEvalOnceShardPad
from ..data_utils import pil_img2rgb
from ..instruction_paraphraser import InstructionParaphraser


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INFERENCE_HASH_FILE = str(Path(os.environ.get(
    "AGIMA_INFERENCE_HASH_FILE", _REPO_ROOT / "datasets" / "inference_hashes.json"
)).expanduser().resolve())
_DEFAULT_INSTRUCTIONS_DIR = str(Path(os.environ.get(
    "AGIMA_INSTRUCTIONS_DIR", _REPO_ROOT / "data" / "instructions"
)).expanduser().resolve())


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        image_num = len(row["image_list"])
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True, 
        )

        if end_idx - start_idx > 1 and random.random() < 0.5: # concat multiple insturction
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(row["instruction_list"][idx-1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(row["instruction_list"][idx-1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=True, 
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=False, 
                        need_vit=False,
                    )
        return data


class VLMGymDataset(InterleavedBaseIterableDataset, JsonLIterableDataset):
    def __init__(self, *args,  **kwargs):
        self.inference_hash_file = kwargs.pop('inference_hash_file', _DEFAULT_INFERENCE_HASH_FILE)
        # Instruction paraphrasing options
        self.use_paraphrases = kwargs.pop('use_paraphrases', True)
        self.instructions_dir = kwargs.pop('instructions_dir', _DEFAULT_INSTRUCTIONS_DIR)
        self.task_name = kwargs.pop('task_name', None)
        super().__init__(*args, **kwargs)
        self.forbidden_hashes = self._load_forbidden_hashes()
        # Initialize paraphraser
        self.paraphraser = InstructionParaphraser(
            instructions_dir=self.instructions_dir,
            use_paraphrases=self.use_paraphrases,
        ) if self.use_paraphrases else None

    def _load_hash_file(self, abs_path: str):
        with open(abs_path, 'r') as jf:
            return json.load(jf)

    def _load_forbidden_hashes(self):
        forbidden_hashes = set()
        try:
            merged_map = self._load_hash_file(self.inference_hash_file)
            for values in merged_map.values():
                if isinstance(values, list):
                    forbidden_hashes.update(values)
        except Exception as e:
            print(f"Warning: failed loading hashes from {self.inference_hash_file}: {e}")
        return forbidden_hashes

    def should_skip_row(self, row):
        # Check inference contamination via hash
        if self.forbidden_hashes is not None and len(self.forbidden_hashes) > 0:
            try:
                data_json = json.loads(row)
                sample_hash = data_json.get('hash')
                if sample_hash in self.forbidden_hashes:
                    # print(f"Skipping row with hash {sample_hash}", flush=True)
                    return True
            except Exception:
                pass
        return False

    def parse_row(self, row):
        # print(row)
        if self.should_skip_row(row):
            return []
        try:
            data_json = json.loads(row)
            history = data_json["history"]
        except Exception as e:
            # print(f"JSON Decode Error: {e}")
            # print(f"Problematic row: {row}")
            return []

        image_prev_list = []
        image_next_list = []
        instruction_list = []
        gt_list = []
        for step in history:
            image_prev_list.append(step["image_prev"] if step["image_prev"] is not None else step["image"])
            image_next_list.append(step["image_next"])
            instruction_list.append(step["prompt"])
            gt_list.append(step["vlm_output"])
        image_num = len(image_prev_list)
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))
        data = self._init_data()
        for idx in range(start_idx, end_idx + 1):
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_prev_list[idx])))),
                need_loss=False if idx == start_idx else True,
                need_vae=True,
                need_vit=True,
            )
            instruction = instruction_list[idx]
            # Apply instruction paraphrasing (only for first instruction which defines the task)
            if idx == start_idx and self.paraphraser is not None and self.task_name:
                instruction = self.paraphraser.get_instruction(self.task_name, instruction)
            gt = gt_list[idx].replace("<think></think>", "")
            data = self._add_text(data, instruction + gt, need_loss=False)
            if idx == end_idx:
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_next_list[idx])))),
                    need_loss=True,
                    need_vae=False,
                    need_vit=False,
                )

        return data


class VLMGymDatasetJigsaw(InterleavedBaseIterableDataset, JsonLIterableDataset):

    def __init__(self, *args,  **kwargs):
        self.inference_hash_file = kwargs.pop('inference_hash_file', _DEFAULT_INFERENCE_HASH_FILE)
        # Instruction paraphrasing options
        self.use_paraphrases = kwargs.pop('use_paraphrases', True)
        self.instructions_dir = kwargs.pop('instructions_dir', _DEFAULT_INSTRUCTIONS_DIR)
        self.task_name = kwargs.pop('task_name', None)
        super().__init__(*args, **kwargs)
        self.forbidden_hashes = self._load_forbidden_hashes()
        # Initialize paraphraser
        self.paraphraser = InstructionParaphraser(
            instructions_dir=self.instructions_dir,
            use_paraphrases=self.use_paraphrases,
        ) if self.use_paraphrases else None

    def _load_hash_file(self, abs_path: str):
        with open(abs_path, 'r') as jf:
            return json.load(jf)

    def _load_forbidden_hashes(self):
        forbidden_hashes = set()
        try:
            merged_map = self._load_hash_file(self.inference_hash_file)
            for values in merged_map.values():
                if isinstance(values, list):
                    forbidden_hashes.update(values)
        except Exception as e:
            print(f"Warning: failed loading hashes from {self.inference_hash_file}: {e}")
        return forbidden_hashes

    def should_skip_row(self, row):
        # Check inference contamination via hash
        if self.forbidden_hashes is not None and len(self.forbidden_hashes) > 0:
            try:
                data_json = json.loads(row)
                sample_hash = data_json.get('hash')
                if sample_hash in self.forbidden_hashes:
                    # print(f"Skipping row with hash {sample_hash}", flush=True)
                    return True
            except Exception:
                pass
        return False

    def parse_row(self, row):
        if self.should_skip_row(row):
            return []
        try:
            data_json = json.loads(row)
            history = data_json["history"]
        except Exception as e:
            # print(f"JSON Decode Error: {e}")
            # print(f"Problematic row: {row}")
            return []
        image_prev_list = []
        image_next_list = []
        instruction_list = []
        gt_list = []
        for step in history:
            image_prev_list.append(step["image_prev"] if step["image_prev"] is not None else step["image"])
            image_next_list.append(step["image_next"])
            instruction_list.append(step["prompt"])
            gt_list.append(step["action"])
        image_num = len(image_prev_list)
        data = self._init_data()
        instruction = instruction_list[0]
        instruction = instruction.split("Verbalize your reasoning")[0]
        # Apply instruction paraphrasing
        if self.paraphraser is not None and self.task_name:
            instruction = self.paraphraser.get_instruction(self.task_name, instruction)
        IMG_GENERATION_PROMPT_LAST = "Generate an image of what you think the next state would be after taking the action."
        data = self._add_text(data, instruction, need_loss=False)
        # randomly decide if we want to learn stop of not
        if random.random() < 0.95:
            # for jigsaw img1 -> img2 and img2 -> img1 need the same action, we randomly choose which direction to take.
            idx = random.choice(range(len(history)))
            if random.random() < 0.8:
                data = self._add_image(
                    data, 
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_prev_list[idx])))),
                    need_loss=False, 
                    need_vae=False, 
                    need_vit=True, 
                )
                # only use the first instruction, which defines the task, we don't need the "you are allowed to take xxx more steps prompt"
                gt = gt_list[idx].replace("<think></think>", "")
                data = self._add_text(data, gt + "\n" + IMG_GENERATION_PROMPT_LAST, need_loss=False)
                data = self._add_image(
                    data, 
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_next_list[idx])))),
                    need_loss=True, 
                    need_vae=True, 
                    need_vit=True,
                )
            else:
                data = self._add_image(
                    data, 
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_next_list[idx])))),
                    need_loss=False, 
                    need_vae=False, 
                    need_vit=True, 
                )
                # only use the first instruction, which defines the task, we don't need the "you are allowed to take xxx more steps prompt"
                gt = gt_list[idx].replace("<think></think>", "")
                data = self._add_text(data, gt + "\n" + IMG_GENERATION_PROMPT_LAST, need_loss=False)
                data = self._add_image(
                    data, 
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_prev_list[idx])))),
                    need_loss=True, 
                    need_vae=True, 
                    need_vit=True,
                )
        else:
            # in this case we give the stop action
            gt = gt_list[-1].replace("<think></think>", "") # this should be the stop action
            idx = random.choice(range(len(history)))
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_next_list[idx])))),
                need_loss=False, 
                need_vae=False, 
                need_vit=True, 
            )
            # only use the first instruction, which defines the task, we don't need the "you are allowed to take xxx more steps prompt"
            data = self._add_text(data, gt + "\n" + IMG_GENERATION_PROMPT_LAST, need_loss=False)
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_next_list[idx])))),
                need_loss=True, 
                need_vae=True, 
                need_vit=True,
            )
        return data


class VLMGymMultiactionDataset(InterleavedBaseIterableDataset, JsonLIterableDataset):

    def __init__(self, *args,  **kwargs):
        self.inference_hash_file = kwargs.pop('inference_hash_file', _DEFAULT_INFERENCE_HASH_FILE)
        # Instruction paraphrasing options
        self.use_paraphrases = kwargs.pop('use_paraphrases', True)
        self.instructions_dir = kwargs.pop('instructions_dir', _DEFAULT_INSTRUCTIONS_DIR)
        self.task_name = kwargs.pop('task_name', None)
        super().__init__(*args, **kwargs)
        self.forbidden_hashes = self._load_forbidden_hashes()
        # Initialize paraphraser
        self.paraphraser = InstructionParaphraser(
            instructions_dir=self.instructions_dir,
            use_paraphrases=self.use_paraphrases,
        ) if self.use_paraphrases else None

    def _load_hash_file(self, abs_path: str):
        with open(abs_path, 'r') as jf:
            return json.load(jf)

    def _load_forbidden_hashes(self):
        forbidden_hashes = set()
        try:
            merged_map = self._load_hash_file(self.inference_hash_file)
            for values in merged_map.values():
                if isinstance(values, list):
                    forbidden_hashes.update(values)
        except Exception as e:
            print(f"Warning: failed loading hashes from {self.inference_hash_file}: {e}")
        return forbidden_hashes

    def should_skip_row(self, row):
        # Check inference contamination via hash
        if self.forbidden_hashes is not None and len(self.forbidden_hashes) > 0:
            try:
                data_json = json.loads(row)
                sample_hash = data_json.get('hash')
                if sample_hash in self.forbidden_hashes:
                    # print(f"Skipping row with hash {sample_hash}", flush=True)
                    return True
            except Exception:
                pass
        return False

    def parse_row(self, row):
        if self.should_skip_row(row):
            return []
        try:
            data_json = json.loads(row)
            history = data_json["history"]
        except Exception as e:
            # print(f"JSON Decode Error: {e}")
            # print(f"Problematic row: {row}")
            return []
        history = data_json["history"]
        image_prev_list = []
        image_next_list = []
        instruction_list = []
        gt_list = []
        for step in history:
            image_prev_list.append(step["image_prev"] if step["image_prev"] is not None else step["image"])
            image_next_list.append(step["image_next"])
            instruction_list.append(step["prompt"])
            gt_list.append(step["vlm_output"])
        image_num = len(image_prev_list)
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = 0
        max_end = min(start_idx + 5, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))
        data = self._init_data()
        for idx in range(start_idx, end_idx + 1):
            if idx == start_idx:
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_prev_list[idx])))),
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                )
            instruction = instruction_list[idx]
            # Apply instruction paraphrasing (only for first instruction which defines the task)
            if idx == start_idx and self.paraphraser is not None and self.task_name:
                instruction = self.paraphraser.get_instruction(self.task_name, instruction)
            gt = gt_list[idx].replace("<think></think>", "")
            data = self._add_text(data, instruction + gt, need_loss=False)
            if idx == end_idx:
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_next_list[idx])))),
                    need_loss=True,
                    need_vae=False,
                    need_vit=False,
                )
            
        return data

class VLMGymDatasetCELOSS(InterleavedBaseIterableDataset, JsonLIterableDataset):
    def __init__(self, *args,  **kwargs):
        self.inference_hash_file = kwargs.pop('inference_hash_file', _DEFAULT_INFERENCE_HASH_FILE)
        # Instruction paraphrasing options
        self.use_paraphrases = kwargs.pop('use_paraphrases', True)
        self.instructions_dir = kwargs.pop('instructions_dir', _DEFAULT_INSTRUCTIONS_DIR)
        self.task_name = kwargs.pop('task_name', None)
        super().__init__(*args, **kwargs)
        self.forbidden_hashes = self._load_forbidden_hashes()
        # Initialize paraphraser
        self.paraphraser = InstructionParaphraser(
            instructions_dir=self.instructions_dir,
            use_paraphrases=self.use_paraphrases,
        ) if self.use_paraphrases else None

    def _load_hash_file(self, abs_path: str):
        with open(abs_path, 'r') as jf:
            return json.load(jf)

    def _load_forbidden_hashes(self):
        forbidden_hashes = set()
        try:
            merged_map = self._load_hash_file(self.inference_hash_file)
            for values in merged_map.values():
                if isinstance(values, list):
                    forbidden_hashes.update(values)
        except Exception as e:
            print(f"Warning: failed loading hashes from {self.inference_hash_file}: {e}")
        return forbidden_hashes

    def should_skip_row(self, row):
        # Check inference contamination via hash
        if self.forbidden_hashes is not None and len(self.forbidden_hashes) > 0:
            try:
                data_json = json.loads(row)
                sample_hash = data_json.get('hash')
                if sample_hash in self.forbidden_hashes:
                    # print(f"Skipping row with hash {sample_hash}", flush=True)
                    return True
            except Exception:
                pass
        return False

    def parse_row(self, row):
        if self.should_skip_row(row):
            return []
        try:
            data_json = json.loads(row)
            history = data_json["history"]
        except Exception as e:
            # print(f"JSON Decode Error: {e}")
            # print(f"Problematic row: {row}")
            return []
        history = data_json["history"]
        image_prev_list = []
        image_next_list = []
        instruction_list = []
        gt_list = []
        for step in history:
            image_prev_list.append(step["image_prev"] if step["image_prev"] is not None else step["image"])
            image_next_list.append(step["image_next"])
            instruction_list.append(step["prompt"])
            gt_list.append(step["vlm_output"])
        image_num = len(image_prev_list)
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = 0
        end_idx = image_num-1
        data = self._init_data()
        for idx in range(start_idx, end_idx + 1):
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_prev_list[idx])))),
                need_loss=False if idx == start_idx else True,
                need_vae=True,
                need_vit=True,
            )
            instruction = instruction_list[idx]
            # Apply instruction paraphrasing (only for first instruction which defines the task)
            if idx == start_idx and self.paraphraser is not None and self.task_name:
                instruction = self.paraphraser.get_instruction(self.task_name, instruction)
                # print(f"Instruction: {instruction}")

            gt = gt_list[idx].replace("<think></think>", "")
            data = self._add_text(data, instruction, need_loss=False)
            data = self._add_text(data, gt, need_loss=True)
            if idx == end_idx:
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_next_list[idx])))),
                    need_loss=True,
                    need_vae=False,
                    need_vit=False,
                )

        return data

class VLMGymDatasetCELOSSNoMSE(InterleavedBaseIterableDataset, JsonLIterableDataset):
    def __init__(self, *args,  **kwargs):
        self.inference_hash_file = kwargs.pop('inference_hash_file', _DEFAULT_INFERENCE_HASH_FILE)
        # Instruction paraphrasing options
        self.use_paraphrases = kwargs.pop('use_paraphrases', True)
        self.instructions_dir = kwargs.pop('instructions_dir', _DEFAULT_INSTRUCTIONS_DIR)
        self.task_name = kwargs.pop('task_name', None)  # Will be auto-detected if not provided
        super().__init__(*args, **kwargs)
        self.forbidden_hashes = self._load_forbidden_hashes()
        # Initialize paraphraser
        self.paraphraser = InstructionParaphraser(
            instructions_dir=self.instructions_dir,
            use_paraphrases=self.use_paraphrases,
        ) if self.use_paraphrases else None

    def _load_hash_file(self, abs_path: str):
        with open(abs_path, 'r') as jf:
            return json.load(jf)

    def _load_forbidden_hashes(self):
        forbidden_hashes = set()
        try:
            merged_map = self._load_hash_file(self.inference_hash_file)
            for values in merged_map.values():
                if isinstance(values, list):
                    forbidden_hashes.update(values)
        except Exception as e:
            print(f"Warning: failed loading hashes from {self.inference_hash_file}: {e}")
        return forbidden_hashes

    def should_skip_row(self, row):
        # Check inference contamination via hash
        if self.forbidden_hashes is not None and len(self.forbidden_hashes) > 0:
            try:
                data_json = json.loads(row)
                sample_hash = data_json.get('hash')
                if sample_hash in self.forbidden_hashes:
                    # print(f"Skipping row with hash {sample_hash}", flush=True)
                    return True
            except Exception:
                pass
        return False

    def parse_row(self, row):
        if self.should_skip_row(row):
            return []
        try:
            data_json = json.loads(row)
            history = data_json["history"]
        except Exception as e:
            # print(f"JSON Decode Error: {e}")
            # print(f"Problematic row: {row}")
            return []
        history = data_json["history"]
        image_prev_list = []
        image_next_list = []
        instruction_list = []
        gt_list = []
        for step in history:
            image_prev_list.append(step["image_prev"] if step["image_prev"] is not None else step["image"])
            image_next_list.append(step["image_next"])
            instruction_list.append(step["prompt"])
            gt_list.append(step["vlm_output"])
        image_num = len(image_prev_list)
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = 0
        end_idx = image_num-1
        data = self._init_data()
        for idx in range(start_idx, end_idx + 1):
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_prev_list[idx])))),
                need_loss=False,
                need_vae=False,
                need_vit=True,
            )
            instruction = instruction_list[idx]
            # Apply instruction paraphrasing (only for first instruction which defines the task)
            if idx == start_idx and self.paraphraser is not None and self.task_name:
                instruction = self.paraphraser.get_instruction(self.task_name, instruction)
            gt = gt_list[idx].replace("<think></think>", "")
            data = self._add_text(data, instruction, need_loss=False)
            data = self._add_text(data, gt, need_loss=True)

        return data


class VLMGymDatasetCELOSSTextOnly(InterleavedBaseIterableDataset, JsonLIterableDataset):
    """Like CELOSSNoMSE but only the first frame is shown as image.

    Sequence layout:
      [image_0, prompt_0, action_0,         (image at step 0 only)
                prompt_1, action_1,
                prompt_2, action_2,
                ...]

    Forces the model to track state purely via text history after step 0
    -- effectively training an internal text-form world model.
    """

    def __init__(self, *args, **kwargs):
        self.inference_hash_file = kwargs.pop(
            'inference_hash_file',
            _DEFAULT_INFERENCE_HASH_FILE,
        )
        self.use_paraphrases = kwargs.pop('use_paraphrases', True)
        self.instructions_dir = kwargs.pop(
            'instructions_dir',
            _DEFAULT_INSTRUCTIONS_DIR,
        )
        self.task_name = kwargs.pop('task_name', None)
        super().__init__(*args, **kwargs)
        self.forbidden_hashes = self._load_forbidden_hashes()
        self.paraphraser = InstructionParaphraser(
            instructions_dir=self.instructions_dir,
            use_paraphrases=self.use_paraphrases,
        ) if self.use_paraphrases else None

    def _load_hash_file(self, abs_path: str):
        with open(abs_path, 'r') as jf:
            return json.load(jf)

    def _load_forbidden_hashes(self):
        forbidden_hashes = set()
        try:
            merged_map = self._load_hash_file(self.inference_hash_file)
            for values in merged_map.values():
                if isinstance(values, list):
                    forbidden_hashes.update(values)
        except Exception as e:
            print(f"Warning: failed loading hashes from {self.inference_hash_file}: {e}")
        return forbidden_hashes

    def should_skip_row(self, row):
        if self.forbidden_hashes is not None and len(self.forbidden_hashes) > 0:
            try:
                data_json = json.loads(row)
                sample_hash = data_json.get('hash')
                if sample_hash in self.forbidden_hashes:
                    return True
            except Exception:
                pass
        return False

    def parse_row(self, row):
        if self.should_skip_row(row):
            return []
        try:
            data_json = json.loads(row)
            history = data_json["history"]
        except Exception:
            return []
        image_prev_list, instruction_list, gt_list = [], [], []
        for step in history:
            image_prev_list.append(step["image_prev"] if step["image_prev"] is not None else step["image"])
            instruction_list.append(step["prompt"])
            gt_list.append(step["vlm_output"])
        start_idx, end_idx = 0, len(image_prev_list) - 1
        data = self._init_data()
        for idx in range(start_idx, end_idx + 1):
            if idx == start_idx:
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_prev_list[idx])))),
                    need_loss=False,
                    need_vae=False,
                    need_vit=True,
                )
            instruction = instruction_list[idx]
            if idx == start_idx and self.paraphraser is not None and self.task_name:
                instruction = self.paraphraser.get_instruction(self.task_name, instruction)
            gt = gt_list[idx].replace("<think></think>", "")
            data = self._add_text(data, instruction, need_loss=False)
            data = self._add_text(data, gt, need_loss=True)
        return data


class VLMGymDatasetMSELOSSOnly(InterleavedBaseIterableDataset, JsonLIterableDataset):
    def __init__(self, *args,  **kwargs):
        self.inference_hash_file = kwargs.pop('inference_hash_file', _DEFAULT_INFERENCE_HASH_FILE)
        # Instruction paraphrasing options
        self.use_paraphrases = kwargs.pop('use_paraphrases', True)
        self.instructions_dir = kwargs.pop('instructions_dir', _DEFAULT_INSTRUCTIONS_DIR)
        self.task_name = kwargs.pop('task_name', None)
        super().__init__(*args, **kwargs)
        self.forbidden_hashes = self._load_forbidden_hashes()
        # Initialize paraphraser
        self.paraphraser = InstructionParaphraser(
            instructions_dir=self.instructions_dir,
            use_paraphrases=self.use_paraphrases,
        ) if self.use_paraphrases else None

    def _load_hash_file(self, abs_path: str):
        with open(abs_path, 'r') as jf:
            return json.load(jf)

    def _load_forbidden_hashes(self):
        forbidden_hashes = set()
        try:
            merged_map = self._load_hash_file(self.inference_hash_file)
            for values in merged_map.values():
                if isinstance(values, list):
                    forbidden_hashes.update(values)
        except Exception as e:
            print(f"Warning: failed loading hashes from {self.inference_hash_file}: {e}")
        return forbidden_hashes

    def should_skip_row(self, row):
        # Check inference contamination via hash
        if self.forbidden_hashes is not None and len(self.forbidden_hashes) > 0:
            try:
                data_json = json.loads(row)
                sample_hash = data_json.get('hash')
                if sample_hash in self.forbidden_hashes:
                    # print(f"Skipping row with hash {sample_hash}", flush=True)
                    return True
            except Exception:
                pass
        return False

    def parse_row(self, row):
        if self.should_skip_row(row):
            return []
        try:
            data_json = json.loads(row)
            history = data_json["history"]
        except Exception as e:
            # print(f"JSON Decode Error: {e}")
            # print(f"Problematic row: {row}")
            return []
        history = data_json["history"]
        image_prev_list = []
        image_next_list = []
        instruction_list = []
        gt_list = []
        for step in history:
            image_prev_list.append(step["image_prev"] if step["image_prev"] is not None else step["image"])
            image_next_list.append(step["image_next"])
            instruction_list.append(step["prompt"])
            gt_list.append(step["vlm_output"])
        image_num = len(image_prev_list)
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = 0
        end_idx = image_num-1
        data = self._init_data()
        for idx in range(start_idx, end_idx + 1):
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_prev_list[idx])))),
                need_loss=False if idx == start_idx else True,
                need_vae=True,
                need_vit=True,
            )
            instruction = instruction_list[idx]
            # Apply instruction paraphrasing (only for first instruction which defines the task)
            if idx == start_idx and self.paraphraser is not None and self.task_name:
                instruction = self.paraphraser.get_instruction(self.task_name, instruction)
            gt = gt_list[idx].replace("<think></think>", "")
            data = self._add_text(data, instruction, need_loss=False)
            data = self._add_text(data, gt, need_loss=False)
            if idx == end_idx:
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_next_list[idx])))),
                    need_loss=True,
                    need_vae=False,
                    need_vit=False,
                )

        return data

class VLMGymDatasetCELOSS_EvalOnce(JsonLIterableDatasetEvalOnceShardPad, InterleavedBaseIterableDataset):
    def __init__(self, *args,  **kwargs):
        self.inference_hash_file = kwargs.pop('inference_hash_file', _DEFAULT_INFERENCE_HASH_FILE)
        # Instruction paraphrasing options (disabled by default for eval)
        self.use_paraphrases = kwargs.pop('use_paraphrases', False)
        self.instructions_dir = kwargs.pop('instructions_dir', _DEFAULT_INSTRUCTIONS_DIR)
        self.task_name = kwargs.pop('task_name', None)
        super().__init__(*args, **kwargs)
        self.forbidden_hashes = self._load_forbidden_hashes()
        # Initialize paraphraser (uses deterministic=True for eval consistency)
        self.paraphraser = InstructionParaphraser(
            instructions_dir=self.instructions_dir,
            use_paraphrases=self.use_paraphrases,
        ) if self.use_paraphrases else None

    def _load_hash_file(self, abs_path: str):
        with open(abs_path, 'r') as jf:
            return json.load(jf)

    def _load_forbidden_hashes(self):
        forbidden_hashes = set()
        try:
            merged_map = self._load_hash_file(self.inference_hash_file)
            for values in merged_map.values():
                if isinstance(values, list):
                    forbidden_hashes.update(values)
        except Exception as e:
            print(f"Warning: failed loading hashes from {self.inference_hash_file}: {e}")
        return forbidden_hashes

    def should_skip_row(self, row):
        # Check inference contamination via hash
        if self.forbidden_hashes is not None and len(self.forbidden_hashes) > 0:
            try:
                data_json = json.loads(row)
                sample_hash = data_json.get('hash')
                if sample_hash in self.forbidden_hashes:
                    # print(f"Skipping row with hash {sample_hash}", flush=True)
                    return True
            except Exception:
                pass
        return False

    def parse_row(self, row):
        if self.should_skip_row(row):
            return []
        try:
            data_json = json.loads(row)
            history = data_json["history"]
        except Exception as e:
            # print(f"JSON Decode Error: {e}")
            # print(f"Problematic row: {row}")
            return []
        history = data_json["history"]
        image_prev_list = []
        image_next_list = []
        instruction_list = []
        gt_list = []
        for step in history:
            image_prev_list.append(step["image_prev"] if step["image_prev"] is not None else step["image"])
            image_next_list.append(step["image_next"])
            instruction_list.append(step["prompt"])
            gt_list.append(step["vlm_output"])
        image_num = len(image_prev_list)
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = 0
        end_idx = image_num-1
        data = self._init_data()
        for idx in range(start_idx, end_idx + 1):
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_prev_list[idx])))),
                need_loss=False if idx == start_idx else True,
                need_vae=True,
                need_vit=True,
            )
            instruction = instruction_list[idx]
            # Apply instruction paraphrasing (deterministic for eval)
            # if idx == start_idx and self.paraphraser is not None and self.task_name:
            #     instruction = self.paraphraser.get_instruction(self.task_name, instruction, deterministic=True)
            gt = gt_list[idx].replace("<think></think>", "")
            data = self._add_text(data, instruction, need_loss=False)
            data = self._add_text(data, gt, need_loss=True)
            if idx == end_idx:
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_next_list[idx])))),
                    need_loss=True,
                    need_vae=False,
                    need_vit=False,
                )
            
        return data

class VLMGymDatasetCELOSSNoMSE_EvalOnce(JsonLIterableDatasetEvalOnceShardPad, InterleavedBaseIterableDataset):
    def __init__(self, *args,  **kwargs):
        self.inference_hash_file = kwargs.pop('inference_hash_file', _DEFAULT_INFERENCE_HASH_FILE)
        # Instruction paraphrasing options (disabled by default for eval)
        self.use_paraphrases = kwargs.pop('use_paraphrases', False)
        self.instructions_dir = kwargs.pop('instructions_dir', _DEFAULT_INSTRUCTIONS_DIR)
        self.task_name = kwargs.pop('task_name', None)
        super().__init__(*args, **kwargs)
        self.forbidden_hashes = self._load_forbidden_hashes()
        # Initialize paraphraser (uses deterministic=True for eval consistency)
        self.paraphraser = InstructionParaphraser(
            instructions_dir=self.instructions_dir,
            use_paraphrases=self.use_paraphrases,
        ) if self.use_paraphrases else None

    def _load_hash_file(self, abs_path: str):
        with open(abs_path, 'r') as jf:
            return json.load(jf)

    def _load_forbidden_hashes(self):
        forbidden_hashes = set()
        try:
            merged_map = self._load_hash_file(self.inference_hash_file)
            for values in merged_map.values():
                if isinstance(values, list):
                    forbidden_hashes.update(values)
        except Exception as e:
            print(f"Warning: failed loading hashes from {self.inference_hash_file}: {e}")
        return forbidden_hashes

    def should_skip_row(self, row):
        # Check inference contamination via hash
        if self.forbidden_hashes is not None and len(self.forbidden_hashes) > 0:
            try:
                data_json = json.loads(row)
                sample_hash = data_json.get('hash')
                if sample_hash in self.forbidden_hashes:
                    # print(f"Skipping row with hash {sample_hash}", flush=True)
                    return True
            except Exception:
                pass
        return False

    def parse_row(self, row):
        if self.should_skip_row(row):
            return []
        try:
            data_json = json.loads(row)
            history = data_json["history"]
        except Exception as e:
            # print(f"JSON Decode Error: {e}")
            # print(f"Problematic row: {row}")
            return []
        history = data_json["history"]
        image_prev_list = []
        image_next_list = []
        instruction_list = []
        gt_list = []
        for step in history:
            image_prev_list.append(step["image_prev"] if step["image_prev"] is not None else step["image"])
            image_next_list.append(step["image_next"])
            instruction_list.append(step["prompt"])
            gt_list.append(step["vlm_output"])
        image_num = len(image_prev_list)
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = 0
        end_idx = image_num-1
        data = self._init_data()
        for idx in range(start_idx, end_idx + 1):
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_prev_list[idx])))),
                need_loss=False,
                need_vae=False,
                need_vit=True,
            )
            instruction = instruction_list[idx]
            # Apply instruction paraphrasing (deterministic for eval)
            # if idx == start_idx and self.paraphraser is not None and self.task_name:
            #     instruction = self.paraphraser.get_instruction(self.task_name, instruction, deterministic=True)
            gt = gt_list[idx].replace("<think></think>", "")
            data = self._add_text(data, instruction, need_loss=False)
            data = self._add_text(data, gt, need_loss=True)

        return data


class VLMGymDatasetCELOSSTextOnly_EvalOnce(JsonLIterableDatasetEvalOnceShardPad, InterleavedBaseIterableDataset):
    """EvalOnce version: only first frame shown as image."""

    def __init__(self, *args, **kwargs):
        self.inference_hash_file = kwargs.pop(
            'inference_hash_file',
            _DEFAULT_INFERENCE_HASH_FILE,
        )
        self.use_paraphrases = kwargs.pop('use_paraphrases', False)
        self.instructions_dir = kwargs.pop(
            'instructions_dir',
            _DEFAULT_INSTRUCTIONS_DIR,
        )
        self.task_name = kwargs.pop('task_name', None)
        super().__init__(*args, **kwargs)
        self.forbidden_hashes = self._load_forbidden_hashes()
        self.paraphraser = InstructionParaphraser(
            instructions_dir=self.instructions_dir,
            use_paraphrases=self.use_paraphrases,
        ) if self.use_paraphrases else None

    def _load_hash_file(self, abs_path: str):
        with open(abs_path, 'r') as jf:
            return json.load(jf)

    def _load_forbidden_hashes(self):
        forbidden_hashes = set()
        try:
            merged_map = self._load_hash_file(self.inference_hash_file)
            for values in merged_map.values():
                if isinstance(values, list):
                    forbidden_hashes.update(values)
        except Exception as e:
            print(f"Warning: failed loading hashes from {self.inference_hash_file}: {e}")
        return forbidden_hashes

    def should_skip_row(self, row):
        if self.forbidden_hashes is not None and len(self.forbidden_hashes) > 0:
            try:
                data_json = json.loads(row)
                sample_hash = data_json.get('hash')
                if sample_hash in self.forbidden_hashes:
                    return True
            except Exception:
                pass
        return False

    def parse_row(self, row):
        if self.should_skip_row(row):
            return []
        try:
            data_json = json.loads(row)
            history = data_json["history"]
        except Exception:
            return []
        image_prev_list, instruction_list, gt_list = [], [], []
        for step in history:
            image_prev_list.append(step["image_prev"] if step["image_prev"] is not None else step["image"])
            instruction_list.append(step["prompt"])
            gt_list.append(step["vlm_output"])
        start_idx, end_idx = 0, len(image_prev_list) - 1
        data = self._init_data()
        for idx in range(start_idx, end_idx + 1):
            if idx == start_idx:
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_prev_list[idx])))),
                    need_loss=False,
                    need_vae=False,
                    need_vit=True,
                )
            instruction = instruction_list[idx]
            gt = gt_list[idx].replace("<think></think>", "")
            data = self._add_text(data, instruction, need_loss=False)
            data = self._add_text(data, gt, need_loss=True)
        return data


class VLMGymDatasetMSELOSSOnly_EvalOnce(JsonLIterableDatasetEvalOnceShardPad, InterleavedBaseIterableDataset):
    def __init__(self, *args,  **kwargs):
        self.inference_hash_file = kwargs.pop('inference_hash_file', _DEFAULT_INFERENCE_HASH_FILE)
        # Instruction paraphrasing options (disabled by default for eval)
        self.use_paraphrases = kwargs.pop('use_paraphrases', False)
        self.instructions_dir = kwargs.pop('instructions_dir', _DEFAULT_INSTRUCTIONS_DIR)
        self.task_name = kwargs.pop('task_name', None)
        super().__init__(*args, **kwargs)
        self.forbidden_hashes = self._load_forbidden_hashes()
        # Initialize paraphraser (uses deterministic=True for eval consistency)
        self.paraphraser = InstructionParaphraser(
            instructions_dir=self.instructions_dir,
            use_paraphrases=self.use_paraphrases,
        ) if self.use_paraphrases else None

    def _load_hash_file(self, abs_path: str):
        with open(abs_path, 'r') as jf:
            return json.load(jf)

    def _load_forbidden_hashes(self):
        forbidden_hashes = set()
        try:
            merged_map = self._load_hash_file(self.inference_hash_file)
            for values in merged_map.values():
                if isinstance(values, list):
                    forbidden_hashes.update(values)
        except Exception as e:
            print(f"Warning: failed loading hashes from {self.inference_hash_file}: {e}")
        return forbidden_hashes

    def should_skip_row(self, row):
        # Check inference contamination via hash
        if self.forbidden_hashes is not None and len(self.forbidden_hashes) > 0:
            try:
                data_json = json.loads(row)
                sample_hash = data_json.get('hash')
                if sample_hash in self.forbidden_hashes:
                    # print(f"Skipping row with hash {sample_hash}", flush=True)
                    return True
            except Exception:
                pass
        return False

    def parse_row(self, row):
        if self.should_skip_row(row):
            return []
        try:
            data_json = json.loads(row)
            history = data_json["history"]
        except Exception as e:
            # print(f"JSON Decode Error: {e}")
            # print(f"Problematic row: {row}")
            return []
        history = data_json["history"]
        image_prev_list = []
        image_next_list = []
        instruction_list = []
        gt_list = []
        for step in history:
            image_prev_list.append(step["image_prev"] if step["image_prev"] is not None else step["image"])
            image_next_list.append(step["image_next"])
            instruction_list.append(step["prompt"])
            gt_list.append(step["vlm_output"])
        image_num = len(image_prev_list)
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = 0
        end_idx = image_num-1
        data = self._init_data()
        for idx in range(start_idx, end_idx + 1):
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_prev_list[idx])))),
                need_loss=False if idx == start_idx else True,
                need_vae=True,
                need_vit=True,
            )
            instruction = instruction_list[idx]
            # Apply instruction paraphrasing (deterministic for eval)
            # if idx == start_idx and self.paraphraser is not None and self.task_name:
            #     instruction = self.paraphraser.get_instruction(self.task_name, instruction, deterministic=True)
            gt = gt_list[idx].replace("<think></think>", "")
            data = self._add_text(data, instruction, need_loss=False)
            data = self._add_text(data, gt, need_loss=False)
            if idx == end_idx:
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_next_list[idx])))),
                    need_loss=True,
                    need_vae=False,
                    need_vit=False,
                )
            
        return data


class VLMGymDatasetSokobanInterleavedCoT(InterleavedBaseIterableDataset, JsonLIterableDataset):
    """Sokoban imagined-rollout interleaved CoT.

    Schema: sokoban_interleaved_trajectory_cot_v1. Each row is a full successful
    sokoban_easy trajectory in `messages` format (alternating user/assistant).
    One assistant turn (`selected_step`) interleaves <think> text with imagined
    one-step branch images. Assistant-side loss=False current-state copies are
    skipped; they are context gaps in older schemas and not model targets.

    Loss policy follows the ordered allstep CoT recipe:
      * User text: no CE loss.
      * Assistant text: CE loss.
      * User/context images: VAE + ViT, with MSE on all but the first image.
      * Assistant generated branch images: VAE + ViT with MSE loss.
      * The LM is trained to emit VISION_START before generated branch images.
    """

    VISION_START = "<|vision_start|>"
    VISION_END = "<|vision_end|>"

    def __init__(self, *args, **kwargs):
        # These kwargs are accepted but unused; kept so the registry can pass
        # the same interface as VLMGymDatasetCELOSS.
        kwargs.pop('inference_hash_file', None)
        kwargs.pop('use_paraphrases', None)
        kwargs.pop('instructions_dir', None)
        kwargs.pop('task_name', None)
        super().__init__(*args, **kwargs)

    def _decode_b64_image(self, b64s):
        return pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(b64s))))

    def parse_row(self, row):
        """Parse each JSONL row as alternating user/assistant turns.

        For the selected assistant reasoning turn, keep the full turn as one
        logical sequence: text before each imagined branch image includes the
        VISION_START trigger, then the image is appended as a model target.
        The explicit VISION_END token is intentionally not added in this train
        path; the image packer supplies the modality boundary.
        """
        try:
            d = json.loads(row)
        except Exception:
            return []
        msgs = d.get('messages', [])
        if not msgs:
            return []

        data = self._init_data()
        is_first_image = True

        for msg in msgs:
            role = msg.get('role')
            is_assistant = (role == 'assistant')
            content = msg.get('content', [])
            if not isinstance(content, list):
                continue

            entries = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                ctype = c.get('type')
                if ctype == 'text':
                    txt = c.get('text', '')
                    if txt:
                        entries.append(('text', txt))
                elif ctype == 'image':
                    b64 = c.get('image_b64', '') or c.get('image', '')
                    if not b64:
                        continue
                    src_loss_flag = bool(c.get('loss', False))
                    is_gen_target = src_loss_flag and is_assistant
                    if is_assistant and not src_loss_flag:
                        continue
                    entries.append(('image', b64, is_gen_target))

            if not entries:
                continue

            assistant_has_images = is_assistant and any(e[0] == 'image' for e in entries)

            if assistant_has_images:
                text_segs = []
                img_segs = []
                cur_text = ""
                for e in entries:
                    if e[0] == 'text':
                        cur_text += e[1]
                    else:
                        text_segs.append(cur_text)
                        cur_text = ""
                        img_segs.append((e[1], e[2]))
                text_segs.append(cur_text)

                text_segs[0] = "<think>" + text_segs[0]
                if text_segs[-1].lstrip().startswith("<think>"):
                    s = text_segs[-1].lstrip()
                    s = s[len("<think>"):]
                    text_segs[-1] = s

                for i, ((b64, is_gen), txt_before) in enumerate(zip(img_segs, text_segs[:-1])):
                    # PER-SEGMENT (reverted the single-span override): default prepend_bos=append_eos=True so
                    # EACH text run is its own <|im_start|>..<|im_end|> turn (the established per-segment structure).
                    self._add_text(
                        data, txt_before + self.VISION_START,
                        need_loss=True,
                    )
                    img = self._decode_b64_image(b64)
                    self._add_image(
                        data, img,
                        need_loss=is_gen,
                        need_vae=True, need_vit=True,
                    )
                    is_first_image = False
                self._add_text(
                    data, text_segs[-1],
                    need_loss=True,
                )
            else:
                for e in entries:
                    if e[0] == 'text':
                        self._add_text(
                            data, e[1],
                            need_loss=is_assistant,
                        )
                    else:
                        b64, is_gen = e[1], e[2]
                        img = self._decode_b64_image(b64)
                        if is_gen:
                            self._add_image(
                                data, img,
                                need_loss=True,
                                need_vae=True, need_vit=True,
                            )
                        else:
                            self._add_image(
                                data, img,
                                need_loss=(not is_first_image),
                                need_vae=True, need_vit=True,
                            )
                        is_first_image = False
        return data


class VLMGymDatasetSokobanInterleavedCoT_EvalOnce(
    JsonLIterableDatasetEvalOnceShardPad, InterleavedBaseIterableDataset
):
    """Eval-once variant of VLMGymDatasetSokobanInterleavedCoT (same parser)."""

    VISION_START = "<|vision_start|>"
    VISION_END = "<|vision_end|>"

    def __init__(self, *args, **kwargs):
        kwargs.pop('inference_hash_file', None)
        kwargs.pop('use_paraphrases', None)
        kwargs.pop('instructions_dir', None)
        kwargs.pop('task_name', None)
        super().__init__(*args, **kwargs)

    def _decode_b64_image(self, b64s):
        return pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(b64s))))

    def parse_row(self, row):
        try:
            d = json.loads(row)
        except Exception:
            return []
        msgs = d.get('messages', [])
        if not msgs:
            return []
        data = self._init_data()
        is_first_image = True
        text_buffer = ""
        text_buffer_loss = False

        def flush():
            nonlocal text_buffer
            if text_buffer:
                self._add_text(data, text_buffer, need_loss=text_buffer_loss)
                text_buffer = ""

        for msg in msgs:
            role = msg.get('role')
            is_assistant = (role == 'assistant')
            content = msg.get('content', [])
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                ctype = c.get('type')
                if ctype == 'text':
                    txt = c.get('text', '')
                    if txt == '':
                        continue
                    if text_buffer and text_buffer_loss != is_assistant:
                        flush()
                    text_buffer += txt
                    text_buffer_loss = is_assistant
                elif ctype == 'image':
                    b64 = c.get('image_b64', '') or c.get('image', '')
                    if not b64:
                        continue
                    src_loss_flag = bool(c.get('loss', False))
                    is_gen_target = src_loss_flag and is_assistant
                    if is_gen_target:
                        if text_buffer and text_buffer_loss is not True:
                            flush()
                        text_buffer += self.VISION_START
                        text_buffer_loss = True
                        flush()
                        img = self._decode_b64_image(b64)
                        self._add_image(
                            data, img,
                            need_loss=True, need_vae=True, need_vit=True,
                        )
                        text_buffer = self.VISION_END
                        text_buffer_loss = True
                        is_first_image = False
                    elif is_assistant and not src_loss_flag:
                        continue
                    else:
                        flush()
                        img = self._decode_b64_image(b64)
                        self._add_image(
                            data, img,
                            need_loss=(not is_first_image),
                            need_vae=True, need_vit=True,
                        )
                        is_first_image = False
        flush()
        return data
