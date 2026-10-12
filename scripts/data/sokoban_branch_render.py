#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""SK3 - byte-compatible Sokoban counterfactual branch-frame renderer.

Exp2 (branching) splices a single counterfactual board state into a CoT-inf
single-turn row. Those spliced "branch" frames must be visually indistinguishable
from the expert frames the model was trained on - not just pixel-similar, but
carrying the SAME JPEG byte distribution, so a model/discriminator cannot pick out
the counterfactual by a JPEG artifact ("no JPEG tell").

This module renders a Sokoban board (player + boxes + room_fixed) to a 256x256
q95 base64 JPEG using the *exact same code path* that generated the v8 CoT-inf
source frames: `render_state_b64` from the v8 state-replay renderer
(the sibling sokoban_v8_state_replay_res256.py) with the byte-identical
surface assets. That guarantees byte-identity for identical states.

ENCODER FINDING (verified, do not "fix" to non-turbo):
  The scoping note asked for the "non-turbo IJG encoder". That is WRONG for the v8
  CoT-inf dataset. The v8 res256 frames were regenerated in the bagel env, whose
  PIL is linked against libjpeg-TURBO 3.1.1. Empirically, over 65 image_next frames
  from 5 CoT-inf rows:
      turbo (default-PIL render_state_b64):  65/65 byte-EXACT match to source
      non-turbo IJG (gymnasium.vlm.nonturbo_jpeg): 0/65 (up to 26-level pixel diff,
                                                    different byte length)
  So "byte-distribution-identical to source" REQUIRES the default-PIL/turbo encoder.
  Non-turbo would stamp a JPEG tell on every branch frame. `encoder='source'`
  (default) uses that turbo path. `encoder='nonturbo'` is retained ONLY for legacy
  pre-v8 datasets whose frames were genuinely IJG-encoded.

LOADER CONTRACT (caller's responsibility, not this renderer's):
  In interleaved_cot_envfb (data/interleave_datasets/edit_dataset_envfb.py) an
  assistant image with loss=False is DROPPED. A spliced branch frame lives in the
  assistant turn, so the caller MUST emit it with loss=True (MSE target). This
  module only returns the base64 string.

Public API:
  render_branch_frame(player, boxes, room_fixed, *, action_mode='move_only',
                      encoder='source') -> str   # 256px q95 base64 JPEG
  render_branch_state(state, encoder='source') -> str
  make_branch_state(player, boxes, room_fixed, action_mode='move_only') -> ReplayState
  ReplayState, ReplayError  (re-exported from the v8 replay module)

Self-test: `python scripts/data/sokoban_branch_render.py --input /path/to/shard.jsonl.gz`.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Config (mirrors rebuild_sokoban_allstep_cot_v8_res256.py DEFAULT_REPLAY_SCRIPT
# / DEFAULT_ASSET_ROOT). Surface assets default to the repo-local copy, which is
# md5-identical to the VisGym_3d assets that produced the v8 source frames.
# ---------------------------------------------------------------------------
DEFAULT_REPLAY_SCRIPT = os.environ.get(
    "SOKOBAN_REPLAY_SCRIPT",
    str(Path(__file__).resolve().with_name("sokoban_v8_state_replay_res256.py")),
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_ROOT = os.environ.get(
    "SOKOBAN_SURFACE_ASSET_ROOT",
    str(_REPO_ROOT / "gymnasium" / "envs" / "sokoban" / "surface"),
)
DEFAULT_SELFTEST_DATASET = os.environ.get(
    "SOKOBAN_COT_INF_DATASET",
    None,
)

_RENDER_SIZE = 256
_JPEG_QUALITY = 95

_REPLAY: Any = None
_SURFACES_READY = False
_NT_ENCODER: Any = None


def _load_replay_module(path: str | None = None) -> Any:
    """Import the v8 state-replay module (single source of truth for state +
    render logic), shimming orjson if the active interpreter lacks it."""
    global _REPLAY
    if _REPLAY is not None:
        return _REPLAY
    path = path or DEFAULT_REPLAY_SCRIPT
    if "orjson" not in sys.modules:
        try:
            import orjson  # type: ignore  # noqa: F401
        except Exception:
            shim = types.ModuleType("orjson")
            shim.loads = lambda b: json.loads(  # type: ignore[attr-defined]
                b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else b
            )
            shim.dumps = lambda o: json.dumps(  # type: ignore[attr-defined]
                o, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            sys.modules["orjson"] = shim
    spec = importlib.util.spec_from_file_location(
        "sokoban_v8_state_replay_res256_sk3", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load replay module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _REPLAY = mod
    return mod


def _ensure_surfaces(asset_root: str | None = None) -> Any:
    """Load surface assets once and pin render size=256 / quality=95 on the
    replay module globals."""
    global _SURFACES_READY
    mod = _load_replay_module()
    if not _SURFACES_READY:
        mod.init_worker(asset_root or DEFAULT_ASSET_ROOT, _RENDER_SIZE, _JPEG_QUALITY, False)
        _SURFACES_READY = True
    return mod


def make_branch_state(
    player: Iterable[int],
    boxes: Iterable[Iterable[int]],
    room_fixed: Iterable[Iterable[int]],
    action_mode: str = "move_only",
) -> Any:
    """Build a ReplayState from raw counterfactual components. room_fixed is
    normalized (defensively) exactly like ReplayState.from_row so callers may
    pass either fixed ids {WALL,FLOOR,TARGET} or full room_state ids. Inputs are
    copied - the caller's structures are never mutated. No position validation is
    run: branch states are deliberately off-policy (a counterfactual push may
    strand a box), which is still a legal board to render."""
    mod = _load_replay_module()
    rf = [[mod.normalize_fixed_cell(cell) for cell in row] for row in room_fixed]
    pl = (int(player[0]), int(player[1]))  # type: ignore[index]
    bx = {(int(b[0]), int(b[1])) for b in boxes}  # type: ignore[index]
    return mod.ReplayState(room_fixed=rf, player=pl, boxes=bx, action_mode=str(action_mode))


def _build_canvas(state: Any):
    """Replicate render_state_b64's canvas construction (needed only for the
    nonturbo path, which must feed raw pixels to the IJG subprocess encoder).
    Kept byte-for-byte in step with render_state_b64's paste loop."""
    from PIL import Image

    mod = _REPLAY
    canvas = Image.new("RGB", (_RENDER_SIZE, _RENDER_SIZE))
    grid = state.room_state()
    for r, row in enumerate(grid):
        y0 = round(r * _RENDER_SIZE / state.rows)
        y1 = round((r + 1) * _RENDER_SIZE / state.rows)
        for c, cell in enumerate(row):
            x0 = round(c * _RENDER_SIZE / state.cols)
            x1 = round((c + 1) * _RENDER_SIZE / state.cols)
            canvas.paste(mod.resized_surface(int(cell), x1 - x0, y1 - y0), (x0, y0))
    return canvas


def render_branch_state(state: Any, encoder: str = "source") -> str:
    """Render a ReplayState to a 256px q95 base64 JPEG string.

    encoder='source' (default): the exact render_state_b64 path that produced the
      v8 CoT-inf source frames -> byte-identical for identical states (turbo in
      the bagel env). USE THIS for anything the model consumes as a "real" frame.
    encoder='nonturbo': re-encode via the IJG subprocess (gymnasium.vlm.nonturbo_jpeg).
      Only for legacy pre-v8 IJG datasets; does NOT match v8 source frames.
    """
    mod = _ensure_surfaces()
    if encoder == "source":
        return mod.render_state_b64(state)
    if encoder == "nonturbo":
        import numpy as np

        global _NT_ENCODER
        if _NT_ENCODER is None:
            from gymnasium.vlm.nonturbo_jpeg import get_encoder

            _NT_ENCODER = get_encoder()
        canvas = _build_canvas(state)
        return _NT_ENCODER.encode(np.asarray(canvas, dtype=np.uint8), quality=_JPEG_QUALITY)
    raise ValueError(f"unknown encoder {encoder!r}; use 'source' or 'nonturbo'")


def render_branch_frame(
    player: Iterable[int],
    boxes: Iterable[Iterable[int]],
    room_fixed: Iterable[Iterable[int]],
    *,
    action_mode: str = "move_only",
    encoder: str = "source",
) -> str:
    """Render a counterfactual Sokoban board to a base64 JPEG (256x256, q95).

    Uses the SAME encoder as the v8 CoT-inf expert frames (default-PIL / turbo)
    so branch frames are byte-distribution-identical to expert frames - no JPEG
    tell. See module docstring for the turbo-vs-nonturbo evidence.
    """
    state = make_branch_state(player, boxes, room_fixed, action_mode=action_mode)
    return render_branch_state(state, encoder=encoder)


# Re-exports (lazy attribute access keeps import side-effect-free until first use).
def __getattr__(name: str) -> Any:  # pragma: no cover - thin passthrough
    if name in ("ReplayState", "ReplayError", "render_state_b64", "normalize_fixed_cell"):
        return getattr(_load_replay_module(), name)
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _qtable0(b64: str):
    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    img.load()
    return img.size, img.mode, list(img.quantization[0])


def _plan_actions(assistant_content: list) -> list[str]:
    text_segs = [c["text"] for c in assistant_content if c.get("type") == "text"]
    tail = text_segs[-1].split("</think>")[-1].strip()
    return [ln.strip() for ln in tail.splitlines() if ln.strip()]


def _selftest(dataset: str = DEFAULT_SELFTEST_DATASET, n_rows: int = 3) -> int:
    import gzip

    if not dataset:
        raise ValueError("Supply a dataset file with --input or SOKOBAN_COT_INF_DATASET")
    mod = _ensure_surfaces()
    print(f"[SK3 selftest] replay={DEFAULT_REPLAY_SCRIPT}")
    print(f"[SK3 selftest] assets={DEFAULT_ASSET_ROOT}")
    print(f"[SK3 selftest] dataset={dataset}")

    frame0_qtable_ok = frame0_byte_ok = 0
    total_frames = byte_exact = size_ok = 0
    rows = 0
    with gzip.open(dataset, "rt") as fh:
        for _ in range(n_rows):
            line = fh.readline()
            if not line:
                break
            row = json.loads(line)
            user_c = row["messages"][0]["content"]
            asst_c = row["messages"][1]["content"]
            src_frame0 = user_c[1]["image"]
            src_imgs = [c["image"] for c in asst_c if c.get("type") == "image"]

            # ---- frame0 via the PUBLIC API (player/boxes/room_fixed) ----
            st = mod.ReplayState.from_row(row)
            api_f0 = render_branch_frame(
                st.player, st.boxes, st.room_fixed, action_mode=st.action_mode
            )
            (w, h), mode, qt = _qtable0(api_f0)
            (sw, sh), smode, sqt = _qtable0(src_frame0)
            assert (w, h) == (256, 256) and mode == "RGB", f"bad decode {(w,h)} {mode}"
            if qt == sqt:
                frame0_qtable_ok += 1
            if api_f0 == src_frame0:
                frame0_byte_ok += 1

            # ---- every image_next via the PUBLIC API ----
            actions = _plan_actions(asst_c)
            for j, act in enumerate(actions):
                st.apply_action(act)
                api_next = render_branch_frame(
                    st.player, st.boxes, st.room_fixed, action_mode=st.action_mode
                )
                (nw, nh), nmode, _ = _qtable0(api_next)
                if (nw, nh) == (256, 256) and nmode == "RGB":
                    size_ok += 1
                if j < len(src_imgs):
                    total_frames += 1
                    if api_next == src_imgs[j]:
                        byte_exact += 1
            rows += 1

    print(f"[SK3 selftest] rows={rows}")
    print(f"[SK3 selftest] frame0 qtable[0] == source: {frame0_qtable_ok}/{rows}")
    print(f"[SK3 selftest] frame0 byte-identical to source: {frame0_byte_ok}/{rows}")
    print(f"[SK3 selftest] image_next decoded 256x256 RGB: {size_ok}")
    print(f"[SK3 selftest] image_next byte-identical to source: {byte_exact}/{total_frames}")

    # PASS = correct geometry AND byte-distribution identity (qtable) AND, since
    # the encoder matches source, full byte identity on every compared frame.
    ok = (
        rows > 0
        and frame0_qtable_ok == rows
        and frame0_byte_ok == rows
        and total_frames > 0
        and byte_exact == total_frames
        and size_ok >= total_frames  # every compared next frame decoded to 256x256 RGB
    )
    print(f"[SK3 selftest] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify branch rendering against recorded frames.")
    parser.add_argument("--input", required=True, help="CoT JSONL.gz shard to verify")
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()
    raise SystemExit(_selftest(args.input, args.rows))
