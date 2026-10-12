# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# Modified for Agima: documentation-only removal of machine-specific references.
"""Env-Feedback CoT-chunk loader (GT re-grounding).

Variant of VLMGymDatasetSokobanInterleavedCoT for the "Env Feedback: <img>"
design: between chunks the ENV (a user turn) provides the REAL chunk-end frame as
a loss-0 CONTEXT INPUT (the model RE-SEES it; it does NOT generate it), instead of
the text-only "Action executed.".

The ONLY behavioral change vs the base train parse_row is in the user-turn image
branch: a user image honors its explicit `loss` flag (loss=False -> need_loss=0,
context) rather than the base's `need_loss=(not is_first_image)` (which would turn
the 2nd+ user image into an MSE target). The base class is documented to keep that
older-schema behavior, so we subclass instead of editing it. Backward-compatible:
frame0 (loss=False) stays loss-0; the new Env-Feedback frame (loss=False) is loss-0.

Assistant turns (the <think> per-step imagined frames) are UNCHANGED: still
need_loss=is_gen (MSE targets) + vae/vit context.
"""
import json

from .edit_dataset import (
    VLMGymDatasetSokobanInterleavedCoT,
    VLMGymDatasetSokobanInterleavedCoT_EvalOnce,
)


class _EnvFBParseRowMixin:
    """parse_row that honors the explicit per-image `loss` flag for USER-turn images.

    Identical to VLMGymDatasetSokobanInterleavedCoT.parse_row except:
      (1) `entries` carries the raw src_loss_flag for images;
      (2) in the user-turn (else) image branch, need_loss = bool(raw_loss_flag)
          instead of (not is_first_image).
    """

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
                    # carry raw loss flag (4th elem) so user-turn images can honor it
                    entries.append(('image', b64, is_gen_target, src_loss_flag))

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
                        img_segs.append((e[1], e[2]))   # (b64, is_gen)
                text_segs.append(cur_text)

                text_segs[0] = "<think>" + text_segs[0]
                if text_segs[-1].lstrip().startswith("<think>"):
                    s = text_segs[-1].lstrip()
                    s = s[len("<think>"):]
                    text_segs[-1] = s

                for i, ((b64, is_gen), txt_before) in enumerate(zip(img_segs, text_segs[:-1])):
                    # PER-SEGMENT (reverted the single-span override): default prepend_bos=append_eos=True so
                    # EACH text run is its own <|im_start|>..<|im_end|> turn (the established per-segment structure). The
                    # per-step im_end re-anchors self-rollout, avoiding the single-span exposure-bias collapse.
                    self._add_text(data, txt_before + self.VISION_START, need_loss=True)
                    img = self._decode_b64_image(b64)
                    self._add_image(data, img, need_loss=is_gen, need_vae=True, need_vit=True)
                    is_first_image = False
                self._add_text(data, text_segs[-1], need_loss=True)
            else:
                for e in entries:
                    if e[0] == 'text':
                        self._add_text(data, e[1], need_loss=is_assistant)
                    else:
                        b64, is_gen, raw_loss = e[1], e[2], e[3]
                        img = self._decode_b64_image(b64)
                        # KEY DIFF: user-turn image honors its explicit loss flag.
                        # frame0 (loss=False) -> context; Env-Feedback frame (loss=False) -> context.
                        self._add_image(data, img, need_loss=bool(raw_loss),
                                        need_vae=True, need_vit=True)
                        is_first_image = False
        return data


class VLMGymDatasetSokobanInterleavedCoTEnvFB(
    _EnvFBParseRowMixin, VLMGymDatasetSokobanInterleavedCoT
):
    """Train loader: GT Env-Feedback frame is a loss-0 user-turn context input."""
    pass


class VLMGymDatasetSokobanInterleavedCoTEnvFB_EvalOnce(
    _EnvFBParseRowMixin, VLMGymDatasetSokobanInterleavedCoT_EvalOnce
):
    """Eval-once variant (same envfb parser; sharding/padding from the base)."""
    pass
