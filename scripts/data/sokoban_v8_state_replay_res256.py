#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""Regenerate Sokoban v7 JSONL images from state at 256x256.

This converts an existing v7 dataset without using its old 112/160px image
fields as image sources. For each episode, it reconstructs the Sokoban state
from init_state, replays history[*].action, and replaces image_prev/image/
image_next with freshly rendered 256x256 JPEG base64 strings.
"""

from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures as futures
import dataclasses
import datetime as dt
import gzip
import html
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import orjson
from PIL import Image, ImageDraw


WALL = 0
FLOOR = 1
TARGET = 2
BOX_ON_TARGET = 3
BOX = 4
PLAYER = 5
PLAYER_ON_TARGET = 6

SURFACE_NAMES = {
    WALL: "wall.png",
    FLOOR: "floor.png",
    TARGET: "box_target.png",
    BOX_ON_TARGET: "box_on_target.png",
    BOX: "box.png",
    PLAYER: "player.png",
    PLAYER_ON_TARGET: "player_on_target.png",
}

PAYLOAD_TO_DELTA = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

LEGACY_MOVE_TO_PAYLOAD = {
    5: 1,
    6: 2,
    7: 3,
    8: 4,
}


class ReplayError(RuntimeError):
    pass


@dataclasses.dataclass
class ReplayState:
    room_fixed: list[list[int]]
    player: tuple[int, int]
    boxes: set[tuple[int, int]]
    action_mode: str

    @property
    def rows(self) -> int:
        return len(self.room_fixed)

    @property
    def cols(self) -> int:
        return len(self.room_fixed[0]) if self.room_fixed else 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ReplayState":
        init_state = row.get("init_state")
        if not isinstance(init_state, dict):
            raise ReplayError("row is missing init_state")

        room_fixed_raw = init_state.get("room_fixed")
        if not isinstance(room_fixed_raw, list) or not room_fixed_raw:
            raise ReplayError("init_state.room_fixed is missing or empty")
        room_fixed = [
            [normalize_fixed_cell(cell) for cell in raw_row]
            for raw_row in room_fixed_raw
        ]

        player_raw = init_state.get("player_position")
        if not isinstance(player_raw, (list, tuple)) or len(player_raw) != 2:
            raise ReplayError("init_state.player_position is missing or invalid")
        player = (int(player_raw[0]), int(player_raw[1]))

        boxes_raw = init_state.get("box_positions")
        if boxes_raw is None and "box_mapping" in init_state:
            mapping = init_state["box_mapping"]
            if not isinstance(mapping, dict):
                raise ReplayError("init_state.box_mapping is not a dict")
            boxes_raw = list(mapping.values())
        if boxes_raw is None and isinstance(init_state.get("room_state"), list):
            boxes_raw = []
            for r, raw_row in enumerate(init_state["room_state"]):
                for c, value in enumerate(raw_row):
                    if int(value) in {BOX, BOX_ON_TARGET}:
                        boxes_raw.append([r, c])
        if boxes_raw is None:
            raise ReplayError("init_state must contain box_positions or box_mapping")

        boxes: set[tuple[int, int]] = set()
        for box in boxes_raw:
            if not isinstance(box, (list, tuple)) or len(box) != 2:
                raise ReplayError(f"invalid box position: {box!r}")
            boxes.add((int(box[0]), int(box[1])))

        action_mode = (
            init_state.get("action_mode")
            or (row.get("env_kwargs") or {}).get("action_mode")
            or ("move_only" if "move_only" in str(row.get("action_schema_version", "")) else "move_push")
        )
        action_mode = str(action_mode)
        if action_mode not in {"move_only", "move_push", "move_and_push"}:
            raise ReplayError(f"unsupported action_mode={action_mode!r}")
        if action_mode == "move_and_push":
            action_mode = "move_push"

        state = cls(
            room_fixed=room_fixed,
            player=player,
            boxes=boxes,
            action_mode=action_mode,
        )
        state._validate_positions()
        return state

    def _validate_positions(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ReplayError("empty room")
        for pos_name, pos in [("player", self.player), *[(f"box[{i}]", b) for i, b in enumerate(self.boxes)]]:
            r, c = pos
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                raise ReplayError(f"{pos_name} position out of bounds: {pos}")
            if self.room_fixed[r][c] == WALL:
                raise ReplayError(f"{pos_name} position is on a wall: {pos}")

    def room_state(self) -> list[list[int]]:
        grid = [row.copy() for row in self.room_fixed]
        for r, c in sorted(self.boxes):
            grid[r][c] = BOX_ON_TARGET if self.room_fixed[r][c] == TARGET else BOX
        pr, pc = self.player
        grid[pr][pc] = PLAYER_ON_TARGET if self.room_fixed[pr][pc] == TARGET else PLAYER
        return grid

    def is_floor(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return False
        if self.room_fixed[r][c] == WALL:
            return False
        return pos not in self.boxes

    def apply_action(self, action_text: str) -> None:
        try:
            parsed = ast.literal_eval(action_text)
        except Exception as exc:
            raise ReplayError(f"cannot parse action {action_text!r}: {exc}") from exc

        if not isinstance(parsed, tuple) or len(parsed) != 2:
            raise ReplayError(f"action must be a tuple(branch, payload): {action_text!r}")

        branch, payload = parsed
        if branch == "stop":
            if payload != "stop":
                raise ReplayError(f"invalid stop payload: {payload!r}")
            return

        if branch not in {"move", "push"}:
            raise ReplayError(f"unknown action branch: {branch!r}")
        if not isinstance(payload, int):
            raise ReplayError(f"action payload must be int for {branch!r}: {payload!r}")

        if self.action_mode == "move_only":
            if branch != "move":
                raise ReplayError(f"move_only mode does not support branch {branch!r}")
            if payload not in PAYLOAD_TO_DELTA:
                raise ReplayError(f"invalid move_only payload: {payload!r}")
            self._apply_move_only(payload)
            return

        if branch == "move":
            if payload == 0:
                return
            if payload in LEGACY_MOVE_TO_PAYLOAD:
                self._apply_plain_move(LEGACY_MOVE_TO_PAYLOAD[payload])
                return
            if payload in PAYLOAD_TO_DELTA:
                self._apply_plain_move(payload)
                return
            raise ReplayError(f"invalid move payload: {payload!r}")

        if payload not in PAYLOAD_TO_DELTA:
            raise ReplayError(f"invalid push payload: {payload!r}")
        self._apply_push(payload)

    def _apply_move_only(self, payload: int) -> None:
        dr, dc = PAYLOAD_TO_DELTA[payload]
        nr, nc = self.player[0] + dr, self.player[1] + dc
        if (nr, nc) in self.boxes:
            self._apply_push(payload)
        else:
            self._apply_plain_move(payload)

    def _apply_plain_move(self, payload: int) -> None:
        dr, dc = PAYLOAD_TO_DELTA[payload]
        next_pos = (self.player[0] + dr, self.player[1] + dc)
        if not self.is_floor(next_pos):
            raise ReplayError(f"move payload {payload} is blocked at {next_pos}")
        self.player = next_pos

    def _apply_push(self, payload: int) -> None:
        dr, dc = PAYLOAD_TO_DELTA[payload]
        box_pos = (self.player[0] + dr, self.player[1] + dc)
        if box_pos not in self.boxes:
            raise ReplayError(f"push payload {payload} has no box at {box_pos}")
        target_pos = (box_pos[0] + dr, box_pos[1] + dc)
        if not self.is_floor(target_pos):
            raise ReplayError(f"push payload {payload} target is blocked at {target_pos}")
        self.boxes.remove(box_pos)
        self.boxes.add(target_pos)
        self.player = box_pos


def normalize_fixed_cell(value: Any) -> int:
    cell = int(value)
    if cell in {WALL, FLOOR, TARGET}:
        return cell
    if cell in {BOX_ON_TARGET, PLAYER_ON_TARGET}:
        return TARGET
    if cell in {BOX, PLAYER}:
        return FLOOR
    raise ReplayError(f"unknown room cell id: {cell!r}")


_SURFACES: dict[int, Image.Image] = {}
_RESIZED_SURFACES: dict[tuple[int, int, int], Image.Image] = {}
_RENDER_SIZE = 256
_JPEG_QUALITY = 95
_DECODE_VALIDATE = False


def init_worker(asset_root: str, render_size: int, jpeg_quality: int, decode_validate: bool) -> None:
    global _SURFACES, _RESIZED_SURFACES, _RENDER_SIZE, _JPEG_QUALITY, _DECODE_VALIDATE
    root = Path(asset_root)
    surfaces: dict[int, Image.Image] = {}
    for surface_id, name in SURFACE_NAMES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing surface asset: {path}")
        surfaces[surface_id] = Image.open(path).convert("RGB")
    _SURFACES = surfaces
    _RESIZED_SURFACES = {}
    _RENDER_SIZE = int(render_size)
    _JPEG_QUALITY = int(jpeg_quality)
    _DECODE_VALIDATE = bool(decode_validate)


def resized_surface(surface_id: int, width: int, height: int) -> Image.Image:
    key = (surface_id, width, height)
    cached = _RESIZED_SURFACES.get(key)
    if cached is not None:
        return cached
    img = _SURFACES[surface_id].resize((width, height), Image.Resampling.NEAREST)
    _RESIZED_SURFACES[key] = img
    return img


def render_state_b64(state: ReplayState) -> str:
    if not _SURFACES:
        raise ReplayError("surface assets are not initialized")
    canvas = Image.new("RGB", (_RENDER_SIZE, _RENDER_SIZE))
    grid = state.room_state()
    for r, row in enumerate(grid):
        y0 = round(r * _RENDER_SIZE / state.rows)
        y1 = round((r + 1) * _RENDER_SIZE / state.rows)
        for c, cell in enumerate(row):
            x0 = round(c * _RENDER_SIZE / state.cols)
            x1 = round((c + 1) * _RENDER_SIZE / state.cols)
            canvas.paste(resized_surface(int(cell), x1 - x0, y1 - y0), (x0, y0))

    if canvas.size != (_RENDER_SIZE, _RENDER_SIZE) or canvas.mode != "RGB":
        raise ReplayError(f"rendered invalid image: size={canvas.size} mode={canvas.mode}")

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    payload = buf.getvalue()
    if not (payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")):
        raise ReplayError("encoded image is not a complete JPEG")
    if _DECODE_VALIDATE:
        decoded = Image.open(io.BytesIO(payload)).convert("RGB")
        if decoded.size != (_RENDER_SIZE, _RENDER_SIZE) or decoded.mode != "RGB":
            raise ReplayError(f"decoded invalid image: size={decoded.size} mode={decoded.mode}")
    return base64.b64encode(payload).decode("ascii")


def maybe_validate_info_state(step: dict[str, Any], state: ReplayState) -> None:
    info_raw = step.get("info")
    if not isinstance(info_raw, str) or not info_raw:
        return
    if '"room_state"' not in info_raw and '"player_position"' not in info_raw:
        return
    try:
        info = json.loads(info_raw)
    except Exception:
        return
    expected_player = info.get("player_position")
    if isinstance(expected_player, list) and len(expected_player) == 2:
        if [state.player[0], state.player[1]] != expected_player:
            raise ReplayError(
                f"info.player_position mismatch: replay={state.player} info={expected_player}"
            )
    expected_room_state = info.get("room_state")
    if isinstance(expected_room_state, list):
        if not room_states_match_for_info(state.room_state(), expected_room_state, state.room_fixed):
            raise ReplayError("info.room_state mismatch after replay")


def room_states_match_for_info(
    replay_room_state: list[list[int]],
    info_room_state: list[list[Any]],
    room_fixed: list[list[int]],
) -> bool:
    if len(replay_room_state) != len(info_room_state):
        return False
    for r, (replay_row, info_row) in enumerate(zip(replay_room_state, info_room_state)):
        if len(replay_row) != len(info_row):
            return False
        for c, (actual, expected_raw) in enumerate(zip(replay_row, info_row)):
            try:
                expected = int(expected_raw)
            except Exception:
                return False
            if int(actual) == expected:
                continue
            # Some upstream easy rows record player-on-target as PLAYER in
            # info.room_state, while VisGym rendering maps it to PLAYER_ON_TARGET.
            if (
                room_fixed[r][c] == TARGET
                and {int(actual), expected} == {PLAYER, PLAYER_ON_TARGET}
            ):
                continue
            return False
    return True


@dataclasses.dataclass
class FileTask:
    source_root: str
    output_root: str
    gzip_root: str
    relative_path: str
    gzip_level: int
    max_rows: int | None


@dataclasses.dataclass
class FileResult:
    relative_path: str
    rows: int
    steps: int
    rendered_frames: int
    image_fields_written: int
    source_bytes: int
    output_bytes: int
    gzip_bytes: int
    elapsed_s: float


def process_file(task: FileTask) -> dict[str, Any]:
    start = time.time()
    source_path = Path(task.source_root) / task.relative_path
    output_path = Path(task.output_root) / task.relative_path
    gzip_path = Path(task.gzip_root) / (task.relative_path + ".gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gzip_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_output = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    tmp_gzip = gzip_path.with_name(f"{gzip_path.name}.tmp.{os.getpid()}")

    rows = 0
    steps = 0
    rendered_frames = 0
    image_fields_written = 0

    try:
        with source_path.open("rb") as src, tmp_output.open("wb") as out, gzip.open(
            tmp_gzip, "wb", compresslevel=task.gzip_level
        ) as gz:
            for line_number, line in enumerate(src, start=1):
                if task.max_rows is not None and rows >= task.max_rows:
                    break
                stripped = line.strip()
                if not stripped:
                    continue

                row = orjson.loads(stripped)
                history = row.get("history")
                if not isinstance(history, list):
                    raise ReplayError(f"{task.relative_path}:{line_number}: row history is missing")

                state = ReplayState.from_row(row)
                current_b64 = render_state_b64(state)
                rendered_frames += 1

                for step_index, step in enumerate(history):
                    if not isinstance(step, dict):
                        raise ReplayError(f"{task.relative_path}:{line_number}: invalid history item")
                    action = step.get("action")
                    prompt = step.get("prompt")
                    if not isinstance(action, str):
                        raise ReplayError(
                            f"{task.relative_path}:{line_number}: step {step_index} missing action"
                        )
                    if not isinstance(prompt, str):
                        raise ReplayError(
                            f"{task.relative_path}:{line_number}: step {step_index} missing prompt"
                        )

                    step["image_prev"] = current_b64
                    step["image"] = current_b64
                    state.apply_action(action)
                    next_b64 = render_state_b64(state)
                    step["image_next"] = next_b64
                    maybe_validate_info_state(step, state)

                    current_b64 = next_b64
                    steps += 1
                    rendered_frames += 1
                    image_fields_written += 3

                payload = orjson.dumps(row)
                out.write(payload)
                out.write(b"\n")
                gz.write(payload)
                gz.write(b"\n")
                rows += 1

        tmp_output.replace(output_path)
        tmp_gzip.replace(gzip_path)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        tmp_gzip.unlink(missing_ok=True)
        raise

    result = FileResult(
        relative_path=task.relative_path,
        rows=rows,
        steps=steps,
        rendered_frames=rendered_frames,
        image_fields_written=image_fields_written,
        source_bytes=source_path.stat().st_size,
        output_bytes=output_path.stat().st_size,
        gzip_bytes=gzip_path.stat().st_size,
        elapsed_s=time.time() - start,
    )
    return dataclasses.asdict(result)


def list_jsonl_files(source_root: Path) -> list[str]:
    files = sorted((source_root / "trajectories").rglob("*.jsonl"))
    return [str(path.relative_to(source_root)) for path in files]


def split_key(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) >= 4 and parts[0] == "trajectories":
        return f"{parts[1]}/{parts[2]}"
    return "unknown"


def copy_static_files(source_root: Path, output_root: Path, gzip_root: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source_root)
        if rel.parts and rel.parts[0] == "trajectories":
            continue
        for root in (output_root, gzip_root):
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
        copied.append(str(rel))
    return copied


def git_info(path: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(path), *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True,
            )
            return proc.stdout.strip()
        except Exception:
            return None

    return {
        "path": str(path),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": run(["rev-parse", "HEAD"]),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_conversion_note(output_root: Path, gzip_root: Path, summary_path: Path) -> None:
    text = (
        "# Sokoban v8 state replay res256\n\n"
        "Images in this local copy were regenerated from init_state plus "
        "history actions at 256x256. The source v7 image bytes were not used "
        "as image inputs.\n\n"
        f"Conversion summary: {summary_path.relative_to(output_root)}\n"
        f"Gzip mirror: {gzip_root}\n"
    )
    for root in (output_root, gzip_root):
        (root / "STATE_REPLAY_RES256_README.md").write_text(text, encoding="utf-8")


def aggregate_results(
    *,
    args: argparse.Namespace,
    files: list[str],
    results: list[dict[str, Any]],
    copied_static_files: list[str],
    start_time: float,
) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {}
    for relative_path in files:
        by_split.setdefault(split_key(relative_path), {"source_files": 0, "output_files": 0, "rows": 0, "steps": 0})
        by_split[split_key(relative_path)]["source_files"] += 1
    for result in results:
        entry = by_split.setdefault(
            split_key(result["relative_path"]),
            {"source_files": 0, "output_files": 0, "rows": 0, "steps": 0},
        )
        entry["output_files"] += 1
        entry["rows"] += int(result["rows"])
        entry["steps"] += int(result["steps"])

    totals = {
        "source_files": len(files),
        "output_files": len(results),
        "rows": sum(int(r["rows"]) for r in results),
        "steps": sum(int(r["steps"]) for r in results),
        "rendered_frames": sum(int(r["rendered_frames"]) for r in results),
        "image_fields_written": sum(int(r["image_fields_written"]) for r in results),
        "source_bytes": sum(int(r["source_bytes"]) for r in results),
        "output_bytes": sum(int(r["output_bytes"]) for r in results),
        "gzip_bytes": sum(int(r["gzip_bytes"]) for r in results),
        "failures": 0,
    }

    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_s": round(time.time() - start_time, 3),
        "source_root": str(Path(args.source_root).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "gzip_root": str(Path(args.gzip_root).resolve()),
        "script": str(Path(__file__).resolve()),
        "visgym_3d": git_info(Path(args.visgym_root).resolve()),
        "render": {
            "size": int(args.render_size),
            "mode": "state_replay_surface_assets",
            "surface_asset_root": str(Path(args.surface_asset_root).resolve()),
            "cell_boundary_policy": "round(i * render_size / dim) to round((i + 1) * render_size / dim)",
            "surface_resize_resampling": "PIL.Image.Resampling.NEAREST",
            "jpeg_quality": int(args.jpeg_quality),
            "base64_data_url_header": False,
        },
        "replay": {
            "state_source": "init_state.room_fixed + init_state.player_position + box_positions_or_box_mapping",
            "action_source": "history[*].action",
            "image_prev_policy": "pre_action_state",
            "image_policy": "pre_action_state",
            "image_next_policy": "post_action_state",
            "stop_policy": "does_not_change_state",
            "action_parse_failures": 0,
            "invalid_transition_failures": 0,
        },
        "metadata_policy": {
            "preserved_source_fields": "all non-image fields are preserved in place",
            "preserved_hash_fields": ["hash", "init_state_hash"],
            "hash_recomputed": False,
        },
        "limits": {
            "max_files_per_split": args.max_files_per_split,
            "max_rows_per_file": args.max_rows_per_file,
        },
        "workers": int(args.workers),
        "gzip_level": int(args.gzip_level),
        "copied_static_files": copied_static_files,
        "by_split": by_split,
        "totals": totals,
        "files": results,
    }


def filter_files(files: list[str], max_files_per_split: int | None) -> list[str]:
    if max_files_per_split is None:
        return files
    kept: list[str] = []
    counts: dict[str, int] = {}
    for relative_path in files:
        key = split_key(relative_path)
        count = counts.get(key, 0)
        if count < max_files_per_split:
            kept.append(relative_path)
            counts[key] = count + 1
    return kept


def run_convert(args: argparse.Namespace) -> int:
    start_time = time.time()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    gzip_root = Path(args.gzip_root).resolve()
    if not (source_root / "trajectories").is_dir():
        raise FileNotFoundError(f"missing source trajectories dir: {source_root / 'trajectories'}")
    if output_root == source_root or gzip_root == source_root:
        raise ValueError("output roots must differ from source root")

    files = filter_files(list_jsonl_files(source_root), args.max_files_per_split)
    if not files:
        raise RuntimeError("no source JSONL files found")

    output_root.mkdir(parents=True, exist_ok=True)
    gzip_root.mkdir(parents=True, exist_ok=True)
    copied_static_files = copy_static_files(source_root, output_root, gzip_root)

    tasks = [
        FileTask(
            source_root=str(source_root),
            output_root=str(output_root),
            gzip_root=str(gzip_root),
            relative_path=relative_path,
            gzip_level=args.gzip_level,
            max_rows=args.max_rows_per_file,
        )
        for relative_path in files
    ]

    print(
        json.dumps(
            {
                "event": "convert_start",
                "source_root": str(source_root),
                "output_root": str(output_root),
                "gzip_root": str(gzip_root),
                "files": len(tasks),
                "workers": args.workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    asset_root = str(Path(args.surface_asset_root).resolve())

    if args.workers == 1:
        init_worker(asset_root, args.render_size, args.jpeg_quality, args.decode_validate)
        for index, task in enumerate(tasks, start=1):
            try:
                result = process_file(task)
                results.append(result)
                print_progress(index, len(tasks), result)
            except Exception as exc:
                failures.append(
                    {
                        "relative_path": task.relative_path,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                break
    else:
        with futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(asset_root, args.render_size, args.jpeg_quality, args.decode_validate),
        ) as pool:
            future_to_task = {pool.submit(process_file, task): task for task in tasks}
            for index, future in enumerate(futures.as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    print_progress(index, len(tasks), result)
                except Exception as exc:
                    failures.append(
                        {
                            "relative_path": task.relative_path,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    for pending in future_to_task:
                        pending.cancel()
                    break

    if failures:
        failure_path = output_root / "metadata" / "state_replay_res256_failures.json"
        write_json(failure_path, failures)
        print(json.dumps({"event": "convert_failed", "failures": failures[:3], "failure_path": str(failure_path)}))
        return 2

    results.sort(key=lambda item: item["relative_path"])
    summary = aggregate_results(
        args=args,
        files=files,
        results=results,
        copied_static_files=copied_static_files,
        start_time=start_time,
    )
    summary_path = output_root / "metadata" / "state_replay_res256_summary.json"
    write_json(summary_path, summary)
    write_json(gzip_root / "metadata" / "state_replay_res256_summary.json", summary)
    write_conversion_note(output_root, gzip_root, summary_path)

    if args.visualize_samples > 0:
        write_visualizations(
            source_root=source_root,
            output_root=output_root,
            n_per_task=args.visualize_samples,
            render_size=args.render_size,
        )

    print(json.dumps({"event": "convert_done", "summary_path": str(summary_path), "totals": summary["totals"]}, sort_keys=True))
    return 0


def print_progress(index: int, total: int, result: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "event": "file_done",
                "index": index,
                "total": total,
                "relative_path": result["relative_path"],
                "rows": result["rows"],
                "steps": result["steps"],
                "elapsed_s": round(float(result["elapsed_s"]), 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def collect_sample_pairs(source_root: Path, output_root: Path, n_per_task: int) -> list[tuple[str, int, dict[str, Any], dict[str, Any]]]:
    samples: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
    task_counts: dict[str, int] = {"sokoban_easy": 0, "sokoban_hard": 0}
    output_files = sorted((output_root / "trajectories").rglob("*.jsonl"))
    output_files.sort(key=lambda p: (0 if "train" in p.parts else 1, str(p)))

    for output_path in output_files:
        rel = output_path.relative_to(output_root)
        parts = rel.parts
        if len(parts) < 4:
            continue
        task = parts[1]
        if task not in task_counts or task_counts[task] >= n_per_task:
            continue
        source_path = source_root / rel
        if not source_path.is_file():
            continue
        with source_path.open("rb") as src, output_path.open("rb") as out:
            for line_no, (src_line, out_line) in enumerate(zip(src, out), start=1):
                if task_counts[task] >= n_per_task:
                    break
                source_row = orjson.loads(src_line)
                output_row = orjson.loads(out_line)
                samples.append((str(rel), line_no, source_row, output_row))
                task_counts[task] += 1
        if all(count >= n_per_task for count in task_counts.values()):
            break
    return samples


def decode_b64_image(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def write_visualizations(source_root: Path, output_root: Path, n_per_task: int, render_size: int) -> None:
    vis_root = output_root / "metadata" / "state_replay_res256_visualizations"
    vis_root.mkdir(parents=True, exist_ok=True)
    samples = collect_sample_pairs(source_root, output_root, n_per_task)
    index_entries: list[str] = []

    for sample_index, (rel, line_no, source_row, output_row) in enumerate(samples):
        task = output_row.get("task", "unknown")
        episode = output_row.get("episode", sample_index)
        stem = f"{sample_index:02d}_{task}_episode_{episode}_line_{line_no}"
        png_path = vis_root / f"{stem}.png"
        html_path = vis_root / f"{stem}.html"
        write_contact_sheet(output_row, png_path)
        write_episode_html(source_row, output_row, rel, line_no, png_path.name, html_path, render_size)
        index_entries.append(f'<li><a href="{html.escape(html_path.name)}">{html.escape(stem)}</a></li>')

    index_html = (
        "<!doctype html><meta charset='utf-8'><title>Sokoban v8 res256 samples</title>"
        "<h1>Sokoban v8 state replay res256 samples</h1><ul>"
        + "\n".join(index_entries)
        + "</ul>"
    )
    (vis_root / "index.html").write_text(index_html, encoding="utf-8")


def frame_sequence(row: dict[str, Any]) -> list[tuple[str, str]]:
    history = row.get("history") or []
    if not history:
        return []
    frames: list[tuple[str, str]] = [("start", history[0]["image"])]
    for idx, step in enumerate(history):
        frames.append((f"{idx}: {step.get('action', '')}", step["image_next"]))
    return frames


def write_contact_sheet(row: dict[str, Any], path: Path) -> None:
    frames = frame_sequence(row)
    if not frames:
        return
    thumb = 128
    label_h = 34
    cols = 5
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, b64) in enumerate(frames):
        img = decode_b64_image(b64).resize((thumb, thumb), Image.Resampling.NEAREST)
        x = (idx % cols) * thumb
        y = (idx // cols) * (thumb + label_h)
        sheet.paste(img, (x, y))
        draw.text((x + 3, y + thumb + 3), label[:24], fill=(0, 0, 0))
    sheet.save(path)


def write_episode_html(
    source_row: dict[str, Any],
    output_row: dict[str, Any],
    rel: str,
    line_no: int,
    png_name: str,
    path: Path,
    render_size: int,
) -> None:
    source_history = source_row.get("history") or []
    output_history = output_row.get("history") or []
    actions_match = [s.get("action") for s in source_history] == [o.get("action") for o in output_history]
    prompts_match = [s.get("prompt") for s in source_history] == [o.get("prompt") for o in output_history]
    first_prompt = source_history[0].get("prompt", "") if source_history else ""
    cards = []
    for label, b64 in frame_sequence(output_row):
        cards.append(
            "<figure>"
            f"<img src='data:image/jpeg;base64,{b64}' width='160' height='160'>"
            f"<figcaption>{html.escape(label)}</figcaption>"
            "</figure>"
        )
    body = f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(str(output_row.get("task")))} episode {html.escape(str(output_row.get("episode")))}</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
figure {{ margin: 0; width: 160px; }}
figcaption {{ font-size: 12px; overflow-wrap: anywhere; }}
pre {{ white-space: pre-wrap; background: #f4f4f4; padding: 12px; }}
</style>
<h1>{html.escape(str(output_row.get("task")))} episode {html.escape(str(output_row.get("episode")))}</h1>
<p>source row: {html.escape(rel)}:{line_no}</p>
<p>render: {render_size}x{render_size} RGB JPEG q95. Actions match source: {actions_match}. Prompts match source: {prompts_match}.</p>
<p><a href="{html.escape(png_name)}">contact sheet png</a></p>
<h2>v7 prompt text</h2>
<pre>{html.escape(first_prompt)}</pre>
<h2>new 256 state frames</h2>
<div class="grid">
{''.join(cards)}
</div>
"""
    path.write_text(body, encoding="utf-8")


def run_validate(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    files = filter_files(list_jsonl_files(source_root), args.max_files_per_split)
    checked = {
        "files": 0,
        "rows": 0,
        "steps": 0,
        "decoded_image_fields": 0,
    }
    errors: list[str] = []
    for relative_path in files:
        source_path = source_root / relative_path
        output_path = output_root / relative_path
        if not output_path.is_file():
            errors.append(f"missing output file {output_path}")
            break
        with source_path.open("rb") as src, output_path.open("rb") as out:
            for line_no, pair in enumerate(zip(src, out), start=1):
                if args.max_rows_per_file is not None and line_no > args.max_rows_per_file:
                    break
                source_row = orjson.loads(pair[0])
                output_row = orjson.loads(pair[1])
                source_history = source_row.get("history") or []
                output_history = output_row.get("history") or []
                if len(source_history) != len(output_history):
                    errors.append(f"{relative_path}:{line_no}: history length mismatch")
                    break
                for idx, (src_step, out_step) in enumerate(zip(source_history, output_history)):
                    if src_step.get("action") != out_step.get("action"):
                        errors.append(f"{relative_path}:{line_no}: step {idx} action mismatch")
                        break
                    if src_step.get("prompt") != out_step.get("prompt"):
                        errors.append(f"{relative_path}:{line_no}: step {idx} prompt mismatch")
                        break
                    if args.decode_images:
                        for key in ("image_prev", "image", "image_next"):
                            img = decode_b64_image(out_step[key])
                            if img.size != (args.render_size, args.render_size) or img.mode != "RGB":
                                errors.append(
                                    f"{relative_path}:{line_no}: step {idx} {key} invalid {img.size} {img.mode}"
                                )
                                break
                            checked["decoded_image_fields"] += 1
                checked["rows"] += 1
                checked["steps"] += len(output_history)
                if errors:
                    break
        checked["files"] += 1
        if errors:
            break
    print(json.dumps({"event": "validate_done", "checked": checked, "errors": errors[:5]}, indent=2, sort_keys=True))
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    default_visgym = str(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert")
    convert.add_argument("--source-root", required=True)
    convert.add_argument("--output-root", required=True)
    convert.add_argument("--gzip-root", required=True)
    convert.add_argument("--visgym-root", default=default_visgym)
    convert.add_argument(
        "--surface-asset-root",
        default=f"{default_visgym}/gymnasium/envs/sokoban/surface",
    )
    convert.add_argument("--workers", type=int, default=max(1, min(48, os.cpu_count() or 1)))
    convert.add_argument("--render-size", type=int, default=256)
    convert.add_argument("--jpeg-quality", type=int, default=95)
    convert.add_argument("--gzip-level", type=int, default=6)
    convert.add_argument("--max-files-per-split", type=int)
    convert.add_argument("--max-rows-per-file", type=int)
    convert.add_argument("--decode-validate", action="store_true")
    convert.add_argument("--visualize-samples", type=int, default=3)
    convert.set_defaults(func=run_convert)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--source-root", required=True)
    validate.add_argument("--output-root", required=True)
    validate.add_argument("--render-size", type=int, default=256)
    validate.add_argument("--max-files-per-split", type=int)
    validate.add_argument("--max-rows-per-file", type=int)
    validate.add_argument("--decode-images", action="store_true")
    validate.set_defaults(func=run_validate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
