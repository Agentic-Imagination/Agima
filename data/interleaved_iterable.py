# Create a dataset, takes in template + prompt; process the sequence with interleaved image text, can self-trigger.
# data/dataset_base.py
# data/interleaved_iterable.py
# data/interleaved_iterable.py
import json, os, traceback
from PIL import Image, ImageFile
import torch
from .data_utils import pil_img2rgb, add_special_tokens
from .distributed_iterable_dataset import DistributedIterableDataset

Image.MAX_IMAGE_PIXELS = 200_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

class InterleavedJSONLIterableDataset(DistributedIterableDataset):
    def __init__(
        self,
        dataset_name: str,
        transform,
        tokenizer,
        jsonl_path_list: list,
        data_dir_list: list,
        num_used_data: list,
        template: list = ['text', 'vision', 'text', 'vision', 'text'],
        local_rank=0,
        world_size=1,
        num_workers=1,
        data_status=None,
        shuffle_lines=False,
        shuffle_seed=0,
    ):
        """
        template: e.g. ['vision','text','vision']  
        jsonl entries must have keys "prompt", "images": [...], "texts": [...]
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform   = transform
        self.tokenizer   = tokenizer
        self.template    = template
        self.data_status = data_status

        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        self.new_token_ids = new_token_ids
        # build (raw_json_line, image_dir) list exactly as in SftJSONLIterableDataset
        self.data_paths = []
        for path, img_dir, n in zip(jsonl_path_list, data_dir_list, num_used_data):
            with open(path,'r') as f:
                lines = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(lines)
            lines = lines[:n]
            self.data_paths += [(l, img_dir) for l in lines]

        self.set_epoch()

    def change_format(self, ex: dict):
        """
        Build sequence_plan plus split text vs. vision slots.
        """
        seq_plan = []
        num_slots = len(self.template)
        for idx, slot in enumerate(self.template):
            is_first  = (idx == 0)
            is_last   = (idx == num_slots - 1)
            next_slot = (self.template[idx+1] if idx+1 < num_slots else None)

            if slot == "vision":
                seq_plan.append({
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,                   # MSE loss on latents
                    "special_token_loss": 1,
                    "special_token_label": None
                })

            elif slot == "text":
                # Just a plain text slot: CE loss on the actual text tokens
                seq_plan.append({
                    "type": "text",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 1,
                    "special_token_label": None,
                    "last": is_last,
                    "next_text": False,
                })

            else:
                raise ValueError(f"Unknown slot {slot!r}")
        return seq_plan

    def __iter__(self):
        data_paths, worker_id = self.get_data_paths_per_worker()
        start = self.data_status[worker_id] + 1 if self.data_status else 0
        stride = self.transform.stride
        while True:
            for idx, (raw, img_dir) in enumerate(data_paths[start:], start=start):
                try:
                    item = json.loads(raw)
                    prompt = item.get('prompt','')
                    imgs   = item.get('images', [])
                    texts  = item.get('texts',  [])
                except:
                    traceback.print_exc()
                    continue

                # load and transform images
                image_tensor_list = []
                num_tokens = 0
                for im in imgs:
                    img = pil_img2rgb(Image.open(os.path.join(img_dir, im)))
                    t = self.transform(img, img_num=len(imgs))
                    image_tensor_list.append(t)
                    h,w = t.shape[1:]
                    num_tokens += (h*w)//(stride*stride)

                # tokenize each text slot
                text_ids_list = []
                texts = [prompt] + texts
                for txt in texts:
                    ids = self.tokenizer.encode(txt)
                    if ids:
                        text_ids_list.append(ids)
                        num_tokens += len(ids)

                # build sequence_plan
                seq_plan = self.change_format(item)

                # skip if no losses
                if not any(e['loss'] for e in seq_plan):
                    continue

                yield dict(
                    image_tensor_list=image_tensor_list,
                    text_ids_list=text_ids_list,
                    sequence_plan=seq_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        'data_indexes': idx,
                        'worker_id':    worker_id,
                        'dataset_name': self.dataset_name
                    }
                )

            start = 0
