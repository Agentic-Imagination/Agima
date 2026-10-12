"""Encode RGB arrays to JPEG b64 using a non-turbo (IJG) libjpeg encoder.

Training data was generated with PIL linked against IJG libjpeg (non-turbo);
inference defaults to libjpeg-turbo, producing different bytes (1-3 RGB
diff per pixel after roundtrip). To match training distribution, route
the JPEG encode step through a separate Python process whose PIL is linked
against non-turbo libjpeg.

Usage:
    # Set NONTURBO_PYTHON to a Python interpreter with non-turbo Pillow.
    from gymnasium.vlm.nonturbo_jpeg import NonTurboJpegEncoder
    enc = NonTurboJpegEncoder()
    b64 = enc.encode(rgb_uint8, quality=90)
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import struct
import subprocess
import threading
from typing import Optional

import numpy as np


_WORKER_SCRIPT = r"""
import sys, base64, io, json, struct
import numpy as np
from PIL import Image, features

# The parent verifies the selected codec before sending any pixels.
print(json.dumps({"libjpeg_turbo": features.check_feature('libjpeg_turbo'),
                  "jpg": features.version('jpg')}), file=sys.stderr, flush=True)

stdin = sys.stdin.buffer
stdout = sys.stdout.buffer

while True:
    hdr = stdin.read(16)
    if not hdr or len(hdr) < 16:
        break
    h, w, c, q = struct.unpack('<IIII', hdr)
    nbytes = h * w * c
    raw = stdin.read(nbytes)
    if len(raw) < nbytes:
        break
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, c)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=int(q))
    out = base64.b64encode(buf.getvalue())
    stdout.write(struct.pack('<I', len(out)))
    stdout.write(out)
    stdout.flush()
"""


class NonTurboJpegEncoder:
    """Persistent subprocess that encodes RGB arrays via non-turbo libjpeg."""

    def __init__(self, python_path: Optional[str] = None):
        self.python_path = python_path or os.environ.get("NONTURBO_PYTHON")
        if not self.python_path:
            raise ValueError(
                "JPEG-NONTURBO requires NONTURBO_PYTHON or python_path pointing to "
                "a Python interpreter whose Pillow uses non-turbo libjpeg."
            )
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._start()
        atexit.register(self.close)

    def _start(self) -> None:
        self._stop()
        self._proc = subprocess.Popen(
            [self.python_path, "-u", "-c", _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=-1,
        )
        try:
            line = self._proc.stderr.readline()
            try:
                codec = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise RuntimeError(
                    "Non-turbo JPEG worker failed to start; its interpreter must have "
                    "NumPy and Pillow with non-turbo libjpeg. "
                    + line.decode("utf-8", errors="replace").strip()
                ) from exc
            if not isinstance(codec, dict) or codec.get("libjpeg_turbo") is not False or not codec.get("jpg"):
                raise RuntimeError(
                    "JPEG-NONTURBO requires Pillow built with non-turbo libjpeg; "
                    "the configured interpreter reports libjpeg-turbo or no JPEG support."
                )
        except BaseException:
            self._stop()
            raise

    @staticmethod
    def _read_exact(stream, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise RuntimeError("Non-turbo JPEG worker closed its output before completing a frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                stream.close()
            except OSError:
                pass

    def encode(self, rgb: np.ndarray, quality: int = 90) -> str:
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        h, w, c = rgb.shape
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            payload = struct.pack("<IIII", h, w, c, int(quality)) + rgb.tobytes()
            try:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
                (n,) = struct.unpack("<I", self._read_exact(self._proc.stdout, 4))
                b64 = self._read_exact(self._proc.stdout, n).decode("utf-8")
            except (OSError, RuntimeError, UnicodeError) as exc:
                self._stop()
                raise RuntimeError(f"Non-turbo JPEG encoding failed: {exc}") from exc
        return b64

    def close(self) -> None:
        with self._lock:
            self._stop()


_singleton: Optional[NonTurboJpegEncoder] = None
_singleton_lock = threading.Lock()


def get_encoder() -> NonTurboJpegEncoder:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = NonTurboJpegEncoder()
    return _singleton
