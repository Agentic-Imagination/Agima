"""Independent simulator copy for per-action imagined-frame replacement."""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np


def render_rgb(env) -> np.ndarray:
    frame = np.asarray(env.render())
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Unexpected environment render: shape={frame.shape}, dtype={frame.dtype}")
    return frame.copy()


def physical_state(env, task: str) -> str:
    """Dynamic visible state, excluding the action counter for invalid-action counts."""
    state = env.unwrapped.get_init_state()
    keys = ("player_position", "box_positions") if task == "sokoban" else (
        "pacman_pos", "ghost_positions", "food", "power_pellets", "alive", "won")
    return json.dumps([state.get(key) for key in keys], sort_keys=True)


def invalid_action(task: str, before: str, after: str, info: dict, is_stop: bool) -> bool:
    if is_stop:
        return False
    if task == "pacman":
        # A wall bump still advances the ghost, so whole-state equality would miss it.
        feedback = str((info or {}).get("env_feedback") or "")
        if "Cannot move into a wall" in feedback:
            return True
    return before == after


class OracleReplica:
    def __init__(self, real_env):
        # The eval RenderResizeEnv forwards unknown attributes through __getattr__, which makes
        # deepcopy recurse before _env is initialized. Copy the physics env (including its RNG)
        # and recreate only that deterministic render wrapper.
        physics = deepcopy(real_env.unwrapped)
        if hasattr(real_env, "size") and hasattr(real_env, "resample"):
            self.env = type(real_env)(physics, size=real_env.size, resample=real_env.resample)
        else:
            self.env = physics
        self.terminal = False
        self.last_frame = render_rgb(self.env)

    def step(self, action: str) -> np.ndarray:
        if not self.terminal:
            _, _, terminated, truncated, _ = self.env.step(action)
            self.last_frame = render_rgb(self.env)
            self.terminal = bool(terminated or truncated)
        return self.last_frame.copy()

    def close(self) -> None:
        self.env.close()
