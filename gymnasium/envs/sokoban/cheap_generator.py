from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


WALL = 0
FLOOR = 1
TARGET = 2


Coord = Tuple[int, int]


@dataclass(frozen=True)
class ReverseAction:
    direction_idx: int
    pull: bool


_CARVE_MASKS: Tuple[np.ndarray, ...] = (
    np.asarray(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.int8,
    ),
    np.asarray(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.int8,
    ),
    np.asarray(
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.int8,
    ),
    np.asarray(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.int8,
    ),
    np.asarray(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.int8,
    ),
)

_DIRS: Tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def generate_level(
    *,
    dim_room: Tuple[int, int],
    num_boxes: int,
    num_gen_steps: int,
    rng: np.random.Generator,
    max_topology_attempts: int = 64,
) -> Tuple[np.ndarray, Coord, Tuple[Coord, ...], Tuple[str, ...]]:
    min_floor_tiles = max(num_boxes * 3 + 2, num_boxes + 6)

    for _ in range(max_topology_attempts):
        room = _generate_connected_topology(dim_room=dim_room, num_steps=num_gen_steps, rng=rng)
        floor_cells = _largest_component(room)
        if len(floor_cells) < min_floor_tiles:
            continue

        room = _walls_from_component(dim_room=dim_room, component=floor_cells)
        targets = _sample_targets(room=room, num_boxes=num_boxes, rng=rng)
        if targets is None:
            continue
        room_fixed = room.copy()
        for target in targets:
            room_fixed[target] = TARGET

        scrambled = _scramble_from_solved(room_fixed=room_fixed, targets=targets, num_boxes=num_boxes, rng=rng)
        if scrambled is None:
            continue
        player, boxes, reverse_history = scrambled
        return room_fixed, player, tuple(sorted(boxes)), _construct_solution_from_reverse_history(reverse_history)

    raise RuntimeError("Unable to generate a Sokoban level with the cheap native generator.")


def _generate_connected_topology(
    *,
    dim_room: Tuple[int, int],
    num_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = dim_room
    room = np.zeros((height, width), dtype=np.int8)
    position = np.asarray(
        [
            int(rng.integers(1, height - 1)),
            int(rng.integers(1, width - 1)),
        ],
        dtype=np.int64,
    )
    direction = np.asarray(_DIRS[int(rng.integers(0, len(_DIRS)))], dtype=np.int64)

    for _ in range(max(8, int(num_steps))):
        if float(rng.random()) < 0.35:
            direction = np.asarray(_DIRS[int(rng.integers(0, len(_DIRS)))], dtype=np.int64)
        position = position + direction
        position[0] = int(np.clip(position[0], 1, height - 2))
        position[1] = int(np.clip(position[1], 1, width - 2))

        mask = _CARVE_MASKS[int(rng.integers(0, len(_CARVE_MASKS)))]
        row = int(position[0]) - 1
        col = int(position[1]) - 1
        room[row : row + 3, col : col + 3] = np.maximum(room[row : row + 3, col : col + 3], mask)

    room[0, :] = WALL
    room[-1, :] = WALL
    room[:, 0] = WALL
    room[:, -1] = WALL
    return room


def _largest_component(room: np.ndarray) -> List[Coord]:
    visited = np.zeros_like(room, dtype=bool)
    best: List[Coord] = []
    for row, col in zip(*np.where(room == FLOOR)):
        if visited[row, col]:
            continue
        component: List[Coord] = []
        queue: deque[Coord] = deque([(int(row), int(col))])
        visited[row, col] = True
        while queue:
            current = queue.popleft()
            component.append(current)
            for nxt in _neighbors(current):
                nr, nc = nxt
                if nr < 0 or nr >= room.shape[0] or nc < 0 or nc >= room.shape[1]:
                    continue
                if visited[nr, nc] or room[nr, nc] != FLOOR:
                    continue
                visited[nr, nc] = True
                queue.append((nr, nc))
        if len(component) > len(best):
            best = component
    return best


def _walls_from_component(*, dim_room: Tuple[int, int], component: Sequence[Coord]) -> np.ndarray:
    room = np.zeros(dim_room, dtype=np.int8)
    for row, col in component:
        room[row, col] = FLOOR
    room[0, :] = WALL
    room[-1, :] = WALL
    room[:, 0] = WALL
    room[:, -1] = WALL
    return room


def _sample_targets(room: np.ndarray, num_boxes: int, rng: np.random.Generator) -> Optional[Tuple[Coord, ...]]:
    candidates = _floor_cells(room)
    if len(candidates) < num_boxes:
        return None
    indices = rng.choice(len(candidates), size=num_boxes, replace=False)
    return tuple(sorted(candidates[int(idx)] for idx in np.atleast_1d(indices)))


def _scramble_from_solved(
    *,
    room_fixed: np.ndarray,
    targets: Sequence[Coord],
    num_boxes: int,
    rng: np.random.Generator,
) -> Optional[Tuple[Coord, Tuple[Coord, ...], Tuple[ReverseAction, ...]]]:
    boxes = tuple(sorted(targets))
    target_set = frozenset(targets)
    player_candidates = [cell for cell in _floor_cells(room_fixed) if cell not in set(boxes)]
    if not player_candidates:
        return None
    player, boxes, reverse_history = _initialize_reverse_scramble(
        room_fixed=room_fixed,
        player_candidates=player_candidates,
        boxes=boxes,
        rng=rng,
    )
    if player is None or boxes is None or reverse_history is None:
        return None

    target_steps = _target_scramble_steps(room_fixed=room_fixed, num_boxes=num_boxes, rng=rng)
    attempts = max(128, target_steps * 16)
    successful_steps = len(reverse_history)
    previous_signature: Optional[Tuple[Coord, Tuple[Coord, ...]]] = None

    for _ in range(attempts):
        actions = _enumerate_reverse_actions(room_fixed=room_fixed, player=player, boxes=boxes)
        if not actions:
            break

        current_boxes_on_target = _boxes_on_target_count(target_set=target_set, boxes=boxes)
        weighted_actions: List[Tuple[ReverseAction, Coord, Tuple[Coord, ...]]] = []
        for action in actions:
            next_player, next_boxes = _apply_reverse_action(
                room_fixed=room_fixed,
                player=player,
                boxes=boxes,
                action=action,
            )
            signature = (next_player, next_boxes)
            if signature == previous_signature:
                continue
            next_boxes_on_target = _boxes_on_target_count(target_set=target_set, boxes=next_boxes)
            weight = 1
            if action.pull:
                weight += 5
                if next_boxes_on_target < current_boxes_on_target:
                    weight += 12 * (current_boxes_on_target - next_boxes_on_target)
                elif next_boxes_on_target == current_boxes_on_target:
                    weight += 3
            weighted_actions.extend([(action, next_player, next_boxes)] * weight)
        if not weighted_actions:
            continue
        action, next_player, next_boxes = weighted_actions[int(rng.integers(0, len(weighted_actions)))]

        previous_signature = (player, boxes)
        player, boxes = next_player, next_boxes
        successful_steps += 1
        reverse_history.append(action)

        if successful_steps >= target_steps:
            if _is_scramble_nontrivial(target_set=target_set, boxes=boxes):
                return player, boxes, tuple(reverse_history)
            break

    if _is_scramble_nontrivial(target_set=target_set, boxes=boxes):
        return player, boxes, tuple(reverse_history)
    return None


def _initialize_reverse_scramble(
    *,
    room_fixed: np.ndarray,
    player_candidates: Sequence[Coord],
    boxes: Tuple[Coord, ...],
    rng: np.random.Generator,
) -> Tuple[Optional[Coord], Optional[Tuple[Coord, ...]], Optional[List[ReverseAction]]]:
    shuffled_indices = rng.permutation(len(player_candidates))
    for idx in np.atleast_1d(shuffled_indices):
        candidate_player = player_candidates[int(idx)]
        pull_actions = [
            action
            for action in _enumerate_reverse_actions(room_fixed=room_fixed, player=candidate_player, boxes=boxes)
            if action.pull
        ]
        if not pull_actions:
            continue
        first_action = pull_actions[int(rng.integers(0, len(pull_actions)))]
        next_player, next_boxes = _apply_reverse_action(
            room_fixed=room_fixed,
            player=candidate_player,
            boxes=boxes,
            action=first_action,
        )
        return next_player, next_boxes, [first_action]
    return None, None, None


def _target_scramble_steps(
    *,
    room_fixed: np.ndarray,
    num_boxes: int,
    rng: np.random.Generator,
) -> int:
    area = int(np.count_nonzero(room_fixed != WALL))
    if room_fixed.shape[0] <= 7:
        return int(rng.integers(8, 17))
    base_low = max(18, num_boxes * 4)
    base_high = min(39, max(base_low + 1, area // 2))
    return int(rng.integers(base_low, base_high + 1))


def _enumerate_reverse_actions(
    *,
    room_fixed: np.ndarray,
    player: Coord,
    boxes: Tuple[Coord, ...],
) -> List[ReverseAction]:
    box_set = set(boxes)
    out: List[ReverseAction] = []
    for direction_idx, (dr, dc) in enumerate(_DIRS):
        ahead = (player[0] + dr, player[1] + dc)
        if not _is_open(room_fixed=room_fixed, pos=ahead, boxes=box_set):
            continue
        out.append(ReverseAction(direction_idx=direction_idx, pull=False))
        behind = (player[0] - dr, player[1] - dc)
        if behind in box_set:
            out.append(ReverseAction(direction_idx=direction_idx, pull=True))
    return out


def _apply_reverse_action(
    *,
    room_fixed: np.ndarray,
    player: Coord,
    boxes: Tuple[Coord, ...],
    action: ReverseAction,
) -> Tuple[Coord, Tuple[Coord, ...]]:
    dr, dc = _DIRS[action.direction_idx]
    next_player = (player[0] + dr, player[1] + dc)
    next_boxes = set(boxes)
    if action.pull:
        behind = (player[0] - dr, player[1] - dc)
        next_boxes.remove(behind)
        next_boxes.add(player)
    return next_player, tuple(sorted(next_boxes))


def _is_scramble_nontrivial(*, target_set: Sequence[Coord], boxes: Tuple[Coord, ...]) -> bool:
    return _boxes_on_target_count(target_set=target_set, boxes=boxes) == 0


def _construct_solution_from_reverse_history(reverse_history: Sequence[ReverseAction]) -> Tuple[str, ...]:
    if not reverse_history or not reverse_history[0].pull:
        raise RuntimeError("Constructive Sokoban solutions require the first reverse action to be a pull.")
    out: List[str] = []
    for action in reversed(reverse_history):
        inverse_direction = _opposite_direction(action.direction_idx)
        action_id = inverse_direction + (1 if action.pull else 5)
        out.append(str(("move", int(action_id))))
    out.append(str(("stop", "stop")))
    return tuple(out)


def _boxes_on_target_count(*, target_set: Sequence[Coord], boxes: Sequence[Coord]) -> int:
    targets = set(target_set)
    return sum(1 for box in boxes if box in targets)


def _opposite_direction(direction_idx: int) -> int:
    return {0: 1, 1: 0, 2: 3, 3: 2}[int(direction_idx)]


def _floor_cells(room: np.ndarray) -> List[Coord]:
    return [(int(row), int(col)) for row, col in zip(*np.where(room != WALL))]


def _neighbors(cell: Coord) -> Iterable[Coord]:
    row, col = cell
    for dr, dc in _DIRS:
        yield row + dr, col + dc


def _is_open(*, room_fixed: np.ndarray, pos: Coord, boxes: Sequence[Coord]) -> bool:
    row, col = pos
    if row < 0 or row >= room_fixed.shape[0] or col < 0 or col >= room_fixed.shape[1]:
        return False
    if room_fixed[row, col] == WALL:
        return False
    return pos not in boxes