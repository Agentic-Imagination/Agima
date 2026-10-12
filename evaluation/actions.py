# Modified for Agima: package imports and portable resource configuration.
"""Shared action parsing and observation encoding, preserved from the original evaluator."""

from __future__ import annotations

import ast
from io import BytesIO

import numpy as np
from PIL import Image

from data.data_utils import pil_img2rgb
from evaluation import base


def _jpeg_roundtrip(arr_or_pil, quality: int, image_format: str = "JPEG") -> Image.Image:
    """Reproduce the per-step eval's frame preprocessing: the env render is
    JPEG-encoded at `quality` (q95 for sokoban v8) then decoded, so the model
    sees the same JPEG artifacts as in the training data (also q95)."""
    pil = arr_or_pil if isinstance(arr_or_pil, Image.Image) else Image.fromarray(np.asarray(arr_or_pil))
    pil = pil.convert("RGB")
    buf = BytesIO()
    if str(image_format).upper() == "PNG":
        pil.save(buf, format="PNG")
    else:
        pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return pil_img2rgb(Image.open(buf))


def parse_action_chunk(text: str) -> list[str]:
    """Extract EVERY action tuple, IN ORDER, from the one-shot generated chunk
    (one `('move', id)` / `('stop','stop')` per line). Uses the same balanced-paren
    + ast scan as base.Interactor.parse_output, but collects ALL candidates rather
    than just the first/last. Stops at the first ('stop','stop') — anything after
    a stop is moot because stop terminates the episode."""
    out: list[str] = []
    s = base.Interactor._normalize(text or "")
    n = len(s)
    i = 0
    while i < n:
        if s[i] != "(":
            i += 1
            continue
        depth = 0
        j = i
        while j < n:
            ch = s[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    raw = s[i:j + 1]
                    try:
                        attempt = base.Interactor._quote_first_identifier(raw)
                        node = ast.literal_eval(attempt)
                        if base.Interactor._is_action_tuple(node):
                            out.append(attempt)
                            if node == ("stop", "stop"):
                                return out
                    except Exception:
                        pass
                    break
            j += 1
        i += 1
    return out
