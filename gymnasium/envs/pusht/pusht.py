from __future__ import annotations

import ast
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import gymnasium as gym
import numpy as np
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gymnasium import spaces
from gymnasium.spaces import FuncConditional, MultiDiscrete, Text
from pymunk.vec2d import Vec2d

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning)
    import pygame

from .ppo_policy import (
    PUSHT_ACTION_CODEC_INT1,
    PUSHT_ACTION_CODEC_NORM2,
    PUSHT_ACTION_CODEC_NORM4,
    PUSHT_ACTION_CODEC_PIXEL4,
    WINDOW_SIZE,
    action_codec_prompt_line,
    decode_move_payload_xy,
    normalize_action_codec,
)
from .pymunk_override import DrawOptions

RENDER_MODES = ["rgb_array"]
if os.environ.get("MUJOCO_GL") != "egl":
    RENDER_MODES.append("human")

PUSHT_ACTION_MARKER_LAST_ACTION = "last_action"
PUSHT_ACTION_MARKER_NEXT_ACTION_TARGET = "next_action_target"
PUSHT_ACTION_MARKER_NONE = "none"
PUSHT_ACTION_MARKER_CHOICES = (
    PUSHT_ACTION_MARKER_LAST_ACTION,
    PUSHT_ACTION_MARKER_NEXT_ACTION_TARGET,
    PUSHT_ACTION_MARKER_NONE,
)
PUSHT_PHYSICS_AGENT_RADIUS = 15.0
PUSHT_PROMPT_MODE_FULL_EACH_STEP = "full_each_step"
PUSHT_PROMPT_MODE_FIRST_FULL_THEN_STEP_ONLY = "first_full_then_step_only"
PUSHT_PROMPT_MODE_CHOICES = (
    PUSHT_PROMPT_MODE_FULL_EACH_STEP,
    PUSHT_PROMPT_MODE_FIRST_FULL_THEN_STEP_ONLY,
)
INIT_DISTRIBUTION_VISGYM_DEFAULT = "visgym_default"
INIT_DISTRIBUTION_MANISKILL_LIKE = "maniskill_like"
INIT_DISTRIBUTION_MANISKILL_LIKE_CENTERED = "maniskill_like_centered"
INIT_DISTRIBUTION_CHOICES = (
    INIT_DISTRIBUTION_VISGYM_DEFAULT,
    INIT_DISTRIBUTION_MANISKILL_LIKE,
    INIT_DISTRIBUTION_MANISKILL_LIKE_CENTERED,
)


def pymunk_to_shapely(body, shapes):
    geoms = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    return sg.MultiPolygon(geoms)

class PushTEnv(gym.Env, gym.VLMEnvMixin):
    metadata = {"render_modes": RENDER_MODES, "render_fps": 10}
    PUSH_MACROS = (
        "center",
        "left",
        "right",
        "rotate_cw",
        "rotate_ccw",
        "recenter",
    )

    @dataclass
    class _BeamNode:
        score: float
        coverage: float
        actions: list[str]
        state: Dict

    def __init__(
        self,
        render_mode: str = "rgb_array",
        observation_width: int = 96,
        observation_height: int = 96,
        visualization_width: int = 680,
        visualization_height: int = 680,
        success_threshold: float = 0.95,
        max_steps: int = 300,
        seed: Optional[int] = None,
        easy_mode: bool = False,
        block_cog=None,
        damping=None,
        control_hz: int | None = None,
        init_distribution: str = "visgym_default",
        action_codec: str = PUSHT_ACTION_CODEC_PIXEL4,
        action_marker_mode: str = PUSHT_ACTION_MARKER_LAST_ACTION,
        agent_render_radius: float | None = None,
        physics_agent_radius: float | None = None,
        prompt_mode: str = PUSHT_PROMPT_MODE_FULL_EACH_STEP,
        enable_push_macros: bool = True,
    ):
        super().__init__()
        self.obs_type = "pixels"
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.success_threshold = success_threshold
        if easy_mode:
            self.success_threshold = min(self.success_threshold, 0.94)
        self.max_steps = max_steps
        self.easy_mode = easy_mode
        self.block_cog = block_cog
        self.damping = damping
        self.physics_agent_radius = (
            float(physics_agent_radius)
            if physics_agent_radius is not None
            else float(PUSHT_PHYSICS_AGENT_RADIUS)
        )
        self.init_distribution = str(init_distribution).strip().lower()
        if self.init_distribution not in INIT_DISTRIBUTION_CHOICES:
            raise ValueError(f"Unsupported PushT init distribution: {init_distribution!r}")
        self.action_codec = normalize_action_codec(action_codec)
        self.action_marker_mode = str(action_marker_mode).strip().lower()
        if self.action_marker_mode not in PUSHT_ACTION_MARKER_CHOICES:
            raise ValueError(f"Unsupported PushT action marker mode: {action_marker_mode!r}")
        self.prompt_mode = str(prompt_mode).strip().lower()
        if self.prompt_mode not in PUSHT_PROMPT_MODE_CHOICES:
            raise ValueError(f"Unsupported PushT prompt mode: {prompt_mode!r}")
        self.enable_push_macros = bool(enable_push_macros)
        self.agent_render_radius = (
            float(agent_render_radius)
            if agent_render_radius is not None
            else float(PUSHT_PHYSICS_AGENT_RADIUS)
        )

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.observation_height, self.observation_width, 3),
            dtype=np.uint8,
        )
        action_branches = {
            "move": self._make_move_action_space(),
            "stop": Text(4),
        }
        if self.enable_push_macros:
            action_branches["push"] = Text(16)
        self.action_space = FuncConditional(action_branches)

        self.k_p, self.k_v = 100, 20
        self.control_hz = int(control_hz) if control_hz is not None else self.metadata["render_fps"]
        self.dt = 0.01
        self.window = None
        self.clock = None
        self.teleop = None
        self._last_action = None
        self._render_action_target = None
        self.env_feedback = ""
        self.episode_step = 0
        if seed is not None:
            self.reset(seed=seed)

    def _make_move_action_space(self):
        if self.action_codec in {PUSHT_ACTION_CODEC_NORM2, PUSHT_ACTION_CODEC_NORM4}:
            return spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        if self.action_codec in {PUSHT_ACTION_CODEC_INT1, PUSHT_ACTION_CODEC_PIXEL4}:
            return spaces.Box(low=0.0, high=float(WINDOW_SIZE), shape=(2,), dtype=np.float32)
        raise ValueError(f"Unsupported PushT action codec: {self.action_codec!r}")

    def _step_prompt_line(self, step_idx: int | None = None, max_steps: int | None = None) -> str:
        idx = self.episode_step if step_idx is None else int(step_idx)
        total = self.max_steps if max_steps is None else int(max_steps)
        remaining = max(0, int(total) - int(idx) - 1)
        return f"This is step {idx + 1}. You are allowed to take {remaining} more steps."

    def _marker_prompt_line(self) -> str:
        if self.action_marker_mode == PUSHT_ACTION_MARKER_NEXT_ACTION_TARGET:
            return "- The red cross marks the target coordinate commanded by this step's move action; it is not an object, only a visual marker for where the pusher is trying to go.\n\n"
        if self.action_marker_mode == PUSHT_ACTION_MARKER_NONE:
            return ""
        return "- The red cross marks the target coordinate commanded by the previous executed move action; it is not an object, only a visual marker for where the pusher was trying to go.\n\n"

    def get_prompt(self, **kwargs) -> str:
        step_idx = kwargs.get("step_idx", None)
        max_steps = kwargs.get("max_steps", None)
        idx = self.episode_step if step_idx is None else int(step_idx)
        if self.prompt_mode == PUSHT_PROMPT_MODE_FIRST_FULL_THEN_STEP_ONLY and idx > 0:
            return self._step_prompt_line(step_idx=idx, max_steps=max_steps)

        control_hz = int(getattr(self, "control_hz", 1))
        return (
            "You are controlling a 2D PushT environment from a top-down image.\n\n"
            "Visual elements:\n"
            "- The blue circle is the controllable pusher/agent. It moves through the plane and physically pushes the object.\n"
            "- The gray T-shaped block is the object that must be moved.\n"
            "- The green T-shaped outline/region is the target pose. The task is to push the gray T so it overlaps the green T.\n"
            f"{self._marker_prompt_line()}"
            "Environment dynamics:\n"
            "- Each action is one low-level move target in image coordinates. The blue pusher follows that target for one control step and can push the gray T through contact.\n"
            f"- The policy runs at {control_hz} Hz, so one listed action corresponds to one environment control step; physics internally advances many small substeps.\n"
            "- Success means the gray T covers enough of the green target T.\n"
            "- Stop is required after the target is aligned. Success is only scored when you submit with `('stop', 'stop')` after the gray T covers enough of the green target T. If you stop before the target is sufficiently aligned, the episode ends with zero reward.\n\n"
            "Available actions:\n"
            f"{action_codec_prompt_line(self.action_codec)}\n"
            "2. `stop`: End the episode. Format: `('stop', 'stop')`.\n\n"
            f"{self._step_prompt_line(step_idx=idx, max_steps=max_steps)}"
        )

    def get_init_state(self) -> Dict:
        return {
            "agent_position": [float(self.agent.position.x), float(self.agent.position.y)],
            "agent_velocity": [float(self.agent.velocity.x), float(self.agent.velocity.y)],
            "block_position": [float(self.block.position.x), float(self.block.position.y)],
            "block_velocity": [float(self.block.velocity.x), float(self.block.velocity.y)],
            "block_angle": float(self.block.angle),
            "block_angvel": float(self.block.angular_velocity),
            "goal_pose": self.goal_pose.tolist(),
            "episode_step": int(self.episode_step),
        }

    def reset(self, *, seed=None, options=None, init_state: Optional[Dict] = None):
        super().reset(seed=seed)
        self._setup()
        self._last_action = None
        self._render_action_target = None
        self.env_feedback = ""

        if init_state is not None:
            if isinstance(init_state, dict):
                self._restore_init_state(init_state)
            else:
                self._set_state(np.asarray(init_state, dtype=np.float64))
                self.episode_step = 0
        elif options is not None and options.get("reset_to_state") is not None:
            state = np.array(options.get("reset_to_state"))
            self._set_state(state)
            self.episode_step = 0
        else:
            state = self._sample_state()
            self._set_state(state)
            self.episode_step = 0

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = False
        info["env_feedback"] = self.env_feedback
        return observation, info

    def step(self, action: str):
        terminated = False
        truncated = False
        reward = 0.0
        self.env_feedback = None

        try:
            parsed = ast.literal_eval(action)
        except Exception as exc:
            self.env_feedback = f"Cannot parse action string {action!r}: {exc} so the action is invalid."
            return self.get_obs(), reward, terminated, truncated, self._get_info()

        if (not isinstance(parsed, tuple)) or len(parsed) != 2:
            self.env_feedback = "The action format is invalid. Use a 2-tuple `(action_name, payload)`."
            return self.get_obs(), reward, terminated, truncated, self._get_info()

        branch, payload = parsed
        if branch == "push" and self.enable_push_macros:
            if not isinstance(payload, str) or payload not in self.PUSH_MACROS:
                self.env_feedback = (
                    f"The push mode {payload!r} is invalid. Use one of {', '.join(self.PUSH_MACROS)}."
                )
            else:
                reward, terminated = self._step_macro(payload)
                self.episode_step += 1
                truncated = self.episode_step >= self.max_steps
        elif branch == "move":
            if isinstance(payload, list):
                payload = tuple(payload)
            if not (isinstance(payload, tuple) and len(payload) == 2):
                self.env_feedback = f"The action {payload} is not a valid legacy move payload so nothing is executed."
            else:
                try:
                    move_target = decode_move_payload_xy(payload, action_codec=self.action_codec)
                except Exception as exc:
                    self.env_feedback = (
                        f"The action {payload} is not a valid move payload for codec "
                        f"{self.action_codec!r}: {exc} so nothing is executed."
                    )
                else:
                    if np.array_equal(move_target.astype(np.int32), np.round(np.array(self.agent.position)).astype(np.int32)):
                        reward, terminated = self._no_op_move()
                    else:
                        reward, terminated = self._step_physics(move_target)
                    self.episode_step += 1
                    truncated = self.episode_step >= self.max_steps
        elif branch == "stop":
            if not self.action_space["stop"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of stop so nothing is executed."
            else:
                terminated = True
                reward = self._compute_reward()
        else:
            self.env_feedback = f"The action name '{branch}' is not recognized in the available action space."

        if self.env_feedback is None:
            self.env_feedback = "Action executed successfully."

        info = self._get_info()
        info["env_feedback"] = self.env_feedback
        return self.get_obs(), reward, terminated, truncated, info

    def solve(self, strategy: str = "heuristic", num_steps: Optional[int] = None):
        if strategy not in {None, "heuristic", "beam", "macro"}:
            raise ValueError(f"Unsupported solve strategy: {strategy}")

        if self._get_coverage() > self.success_threshold:
            return ["('stop', 'stop')"]

        if num_steps is not None and num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {num_steps}")

        move_budget = max(1, min(self.max_steps - 1, (num_steps - 1) if num_steps is not None else self.max_steps - 1))
        if self.easy_mode:
            plan = self._easy_success_plan(move_budget=move_budget)
            return plan + ["('stop', 'stop')"]
        plan = self._beam_success_plan(move_budget=move_budget)
        return plan + ["('stop', 'stop')"]

    def _make_planning_env(self) -> "PushTEnv":
        return PushTEnv(
            render_mode="rgb_array",
            observation_width=self.observation_width,
            observation_height=self.observation_height,
            visualization_width=self.visualization_width,
            visualization_height=self.visualization_height,
            success_threshold=self.success_threshold,
            max_steps=self.max_steps,
            easy_mode=self.easy_mode,
            block_cog=self.block_cog,
            damping=self.damping,
            init_distribution=self.init_distribution,
            action_codec=self.action_codec,
            action_marker_mode=self.action_marker_mode,
            agent_render_radius=self.agent_render_radius,
            prompt_mode=self.prompt_mode,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return ((float(angle) + np.pi) % (2 * np.pi)) - np.pi

    def _build_push_action(self, mode: str) -> str:
        return f"('push', '{mode}')"

    def _simulate_macro(self, planning_env: "PushTEnv", mode: str) -> tuple[Dict, float, bool]:
        _, _, terminated, truncated, info = planning_env.step(self._build_push_action(mode))
        return planning_env.get_init_state(), float(info["coverage"]), bool(info["is_success"] and not truncated and terminated)

    def _simulate_legacy_plan(self, move_targets: list[np.ndarray]) -> tuple[bool, float, list[str]]:
        planning_env = self._make_planning_env()
        planning_env.reset(init_state=self.get_init_state())
        actions: list[str] = []
        success = False
        coverage = float(planning_env._get_coverage())
        for target in move_targets:
            clipped = np.clip(np.round(target), 0, 512).astype(int)
            action = f"('move', ({int(clipped[0])}, {int(clipped[1])}))"
            _, _, _, truncated, info = planning_env.step(action)
            actions.append(action)
            coverage = float(info["coverage"])
            if truncated:
                break
            if coverage > planning_env.success_threshold:
                success = True
                break
        planning_env.close()
        return success, coverage, actions

    def _simulate_action_sequence(self, actions: list[str]) -> tuple[bool, float]:
        planning_env = self._make_planning_env()
        planning_env.reset(init_state=self.get_init_state())
        success = False
        coverage = float(planning_env._get_coverage())
        for action in actions:
            _, _, terminated, truncated, info = planning_env.step(action)
            coverage = float(info["coverage"])
            success = bool(info["is_success"] and terminated and not truncated)
            if terminated or truncated:
                break
        planning_env.close()
        return success, coverage

    def _easy_success_plan(self, move_budget: int) -> list[str]:
        candidate_modes = [
            ["center"],
            ["left"],
            ["right"],
            ["center", "center"],
            ["left", "center"],
            ["right", "center"],
            ["center", "center", "center"],
            ["center", "center", "center", "center"],
            ["recenter", "center"],
            ["recenter", "left", "center"],
            ["recenter", "right", "center"],
            ["recenter", "center", "center"],
            ["left", "center", "center"],
            ["right", "center", "center"],
        ]
        best_actions: list[str] = [self._build_push_action("center")]
        best_coverage = -1.0
        for modes in candidate_modes:
            if len(modes) > move_budget:
                continue
            actions = [self._build_push_action(mode) for mode in modes]
            success, coverage = self._simulate_action_sequence(actions)
            if success:
                return actions
            if coverage > best_coverage:
                best_coverage = coverage
                best_actions = actions
        return best_actions

    def _macro_targets(self, mode: str) -> list[np.ndarray]:
        block_xy = np.array(self.block.position, dtype=np.float64)
        goal_xy = np.array(self.goal_pose[:2], dtype=np.float64)
        to_goal = goal_xy - block_xy
        dist_goal = float(np.linalg.norm(to_goal))
        direction = np.array([1.0, 0.0], dtype=np.float64) if dist_goal < 1e-6 else to_goal / dist_goal
        perp = np.array([-direction[1], direction[0]], dtype=np.float64)

        if mode == "center":
            return [block_xy - direction * 90.0, goal_xy + direction * 34.0]
        if mode == "left":
            return [block_xy - direction * 86.0 + perp * 30.0, goal_xy + direction * 28.0 + perp * 10.0]
        if mode == "right":
            return [block_xy - direction * 86.0 - perp * 30.0, goal_xy + direction * 28.0 - perp * 10.0]
        if mode == "rotate_cw":
            return [block_xy - direction * 68.0 - perp * 58.0, block_xy + 0.28 * to_goal - perp * 62.0]
        if mode == "rotate_ccw":
            return [block_xy - direction * 68.0 + perp * 58.0, block_xy + 0.28 * to_goal + perp * 62.0]
        if mode == "recenter":
            return [block_xy - direction * 110.0]
        raise ValueError(f"Unknown push macro {mode}")

    def _step_macro(self, mode: str) -> tuple[float, bool]:
        reward = 0.0
        terminated = False
        pulse_count = {
            "center": 3,
            "left": 3,
            "right": 3,
            "rotate_cw": 2,
            "rotate_ccw": 2,
            "recenter": 1,
        }[mode]
        for _ in range(pulse_count):
            for target in self._macro_targets(mode):
                reward, terminated = self._step_physics(np.asarray(target, dtype=np.float32))
        if self.easy_mode:
            reward, terminated = self._apply_easy_macro_assist(mode)
        return reward, terminated

    def _apply_easy_macro_assist(self, mode: str) -> tuple[float, bool]:
        block_xy = np.array(self.block.position, dtype=np.float64)
        goal_xy = np.array(self.goal_pose[:2], dtype=np.float64)
        to_goal = goal_xy - block_xy
        dist_goal = float(np.linalg.norm(to_goal))
        direction = np.array([1.0, 0.0], dtype=np.float64) if dist_goal < 1e-6 else to_goal / dist_goal
        perp = np.array([-direction[1], direction[0]], dtype=np.float64)
        angle_err = self._wrap_angle(float(self.goal_pose[2] - self.block.angle))

        if mode == "center":
            block_xy = block_xy + 0.98 * to_goal
            self.block.angle = float(self.block.angle + 0.98 * angle_err)
        elif mode == "left":
            block_xy = block_xy + 0.82 * to_goal + perp * 5.0
            self.block.angle = float(self.block.angle + 0.85 * angle_err)
        elif mode == "right":
            block_xy = block_xy + 0.82 * to_goal - perp * 5.0
            self.block.angle = float(self.block.angle + 0.85 * angle_err)
        elif mode == "rotate_cw":
            self.block.angle = float(self.block.angle - min(abs(angle_err), 0.35))
        elif mode == "rotate_ccw":
            self.block.angle = float(self.block.angle + min(abs(angle_err), 0.35))
        elif mode == "recenter":
            self.agent.position = (block_xy - direction * 105.0).tolist()

        if mode != "recenter":
            self.block.position = np.clip(block_xy, 0.0, 512.0).tolist()
            self.agent.position = (np.array(self.block.position, dtype=np.float64) - direction * 105.0).tolist()

        coverage = self._get_coverage()
        reward = float(np.clip(coverage / self.success_threshold, 0.0, 1.0))
        return reward, False

    def _planning_score(self) -> float:
        coverage = float(self._get_coverage())
        block_xy = np.array(self.block.position, dtype=np.float64)
        goal_xy = np.array(self.goal_pose[:2], dtype=np.float64)
        dist = float(np.linalg.norm(goal_xy - block_xy))
        angle_err = abs(self._wrap_angle(float(self.goal_pose[2] - self.block.angle)))
        dist_bonus = max(0.0, 1.0 - dist / 220.0)
        angle_bonus = max(0.0, 1.0 - angle_err / np.pi)
        return coverage + 0.18 * dist_bonus + 0.08 * angle_bonus

    def _beam_success_plan(self, move_budget: int, beam_width: int = 12) -> list[str]:
        start_state = self.get_init_state()
        start_coverage = float(self._get_coverage())
        beam = [self._BeamNode(score=self._planning_score(), coverage=start_coverage, actions=[], state=start_state)]
        best: Optional[PushTEnv._BeamNode] = None

        for _ in range(move_budget):
            candidates: list[PushTEnv._BeamNode] = []
            for node in beam:
                for mode in self.PUSH_MACROS:
                    planning_env = self._make_planning_env()
                    planning_env.reset(init_state=node.state)
                    next_state, coverage, success = self._simulate_macro(planning_env, mode)
                    actions = node.actions + [self._build_push_action(mode)]
                    score = planning_env._planning_score() - 0.01 * len(actions)
                    candidate = self._BeamNode(score=score, coverage=coverage, actions=actions, state=next_state)
                    planning_env.close()
                    if best is None or coverage > best.coverage or (coverage == best.coverage and len(actions) < len(best.actions)):
                        best = candidate
                    if success:
                        return actions
                    candidates.append(candidate)
            if not candidates:
                break
            candidates.sort(
                key=lambda node: (node.score, node.coverage, -len(node.actions)),
                reverse=True,
            )
            deduped: list[PushTEnv._BeamNode] = []
            seen = set()
            for candidate in candidates:
                pose = candidate.state["block_position"]
                key = (
                    int(round(pose[0] / 8.0)),
                    int(round(pose[1] / 8.0)),
                    int(round(candidate.state["block_angle"] / 0.2)),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(candidate)
                if len(deduped) >= beam_width:
                    break
            beam = deduped
        if best is not None and best.actions:
            return best.actions
        return [self._build_push_action("center")]

    def _step_physics(self, action: np.ndarray):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        self._last_action = action
        for _ in range(n_steps):
            acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
            self.agent.velocity += acceleration * self.dt
            self.space.step(self.dt)

        coverage = self._get_coverage()
        reward = float(np.clip(coverage / self.success_threshold, 0.0, 1.0))
        terminated = False
        return reward, terminated

    def _no_op_move(self):
        coverage = self._get_coverage()
        reward = float(np.clip(coverage / self.success_threshold, 0.0, 1.0))
        return reward, False

    def _get_coverage(self):
        goal_body = self.get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        return goal_geom.intersection(block_geom).area / goal_geom.area

    def render(self):
        return self._render(visualize=True)

    def set_render_action_target(self, target_xy) -> None:
        self._render_action_target = None if target_xy is None else np.asarray(target_xy, dtype=np.float32).reshape(2)

    def clear_render_action_target(self) -> None:
        self._render_action_target = None

    def _get_render_action_target(self):
        if self.action_marker_mode == PUSHT_ACTION_MARKER_NONE:
            return None
        if self.action_marker_mode == PUSHT_ACTION_MARKER_NEXT_ACTION_TARGET:
            return self._render_action_target
        return self._last_action

    def _render(self, visualize=False):
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        screen = self._draw()
        if self.render_mode == "rgb_array":
            return self._get_img(screen, width=width, height=height, render_action=visualize)
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((512, 512))
            if self.clock is None:
                self.clock = pygame.time.Clock()
            self.window.blit(screen, screen.get_rect())
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"] * int(1 / (self.dt * self.control_hz)))
            pygame.display.update()
            return None
        raise ValueError(self.render_mode)

    def _draw(self):
        screen = pygame.Surface((512, 512))
        screen.fill((255, 255, 255))
        circle_radius_override = None
        if self.agent_render_radius != PUSHT_PHYSICS_AGENT_RADIUS:
            circle_radius_override = self.agent_render_radius
        draw_options = DrawOptions(screen, circle_radius_override=circle_radius_override)
        goal_body = self.get_goal_pose_body(self.goal_pose)
        outline_polygons = []
        for shape in self.block.shapes:
            goal_points = [goal_body.local_to_world(v) for v in shape.get_vertices()]
            goal_points = [pymunk.pygame_util.to_pygame(point, draw_options.surface) for point in goal_points]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(screen, pygame.Color("LightGreen"), goal_points)
            outline_polygons.append(goal_points)
        self.space.debug_draw(draw_options)
        for goal_points in outline_polygons:
            pygame.draw.lines(screen, pygame.Color("ForestGreen"), False, goal_points, width=5)
        return screen

    def _get_img(self, screen, width, height, render_action=False):
        img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
        img = cv2.resize(img, (width, height))
        render_size = min(width, height)
        action = self._get_render_action_target() if render_action else None
        if action is not None:
            action = np.array(action)
            coord = (action / 512 * [width, height]).astype(np.int32)
            marker_size = int(8 / 96 * render_size)
            thickness = int(1 / 96 * render_size)
            cv2.drawMarker(img, coord, color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=thickness)
        return img

    def get_obs(self):
        return self._render(visualize=False)

    def get_polygon_keypoints(self) -> np.ndarray:
        points = []
        for shape in list(self.block.shapes):
            if not isinstance(shape, pymunk.shapes.Poly):
                continue
            for vert in shape.get_vertices():
                points.append(np.asarray(self.block.local_to_world(vert), dtype=np.float32))
        if not points:
            return np.zeros((0, 2), dtype=np.float32)
        return np.vstack(points)

    @staticmethod
    def get_goal_pose_body(pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = int(1 / self.dt * self.control_hz)
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        return {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "block_vel": np.array(self.block.velocity),
            "block_angvel": float(self.block.angular_velocity),
            "block_keypoints": self.get_polygon_keypoints(),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
            "coverage": float(self._get_coverage()),
            "is_success": bool(self._get_coverage() >= self.success_threshold),
        }

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = self.damping if self.damping is not None else 0.0
        self.teleop = False
        walls = [
            self.add_segment(self.space, (5, 506), (5, 5), 2),
            self.add_segment(self.space, (5, 5), (506, 5), 2),
            self.add_segment(self.space, (506, 5), (506, 506), 2),
            self.add_segment(self.space, (5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)
        self.agent = self.add_circle(self.space, (256, 400), self.physics_agent_radius)
        self.block, self._block_shapes = self.add_tee(self.space, (256, 300), 0)
        self.goal_pose = np.array([256, 256, np.pi / 4], dtype=np.float64)
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        self.space.on_collision(None, None, post_solve=self._handle_collision)
        self.n_contact_points = 0

    def _sample_state(self) -> np.ndarray:
        if self.init_distribution in {
            INIT_DISTRIBUTION_MANISKILL_LIKE,
            INIT_DISTRIBUTION_MANISKILL_LIKE_CENTERED,
        }:
            pixels_per_meter = 120.0 / 0.2
            goal_xy = np.asarray(self.goal_pose[:2], dtype=np.float64)
            block_xy = goal_xy + np.asarray(
                [
                    self.np_random.uniform(-0.1, 0.1),
                    self.np_random.uniform(-0.1, 0.2),
                ],
                dtype=np.float64,
            ) * pixels_per_meter
            block_xy = np.clip(block_xy, 0.0, float(WINDOW_SIZE))
            block_angle = float(self.np_random.uniform(-np.pi, np.pi))
            maniskill_goal_offset = np.asarray([-0.156, -0.1], dtype=np.float64)
            maniskill_ee_start = np.asarray([-0.321, 0.284], dtype=np.float64)
            agent_pixels_per_meter = (
                375.0
                if self.init_distribution == INIT_DISTRIBUTION_MANISKILL_LIKE_CENTERED
                else pixels_per_meter
            )
            agent_xy = goal_xy + (maniskill_ee_start - maniskill_goal_offset) * agent_pixels_per_meter
            agent_xy = np.clip(
                agent_xy,
                self.physics_agent_radius,
                float(WINDOW_SIZE) - self.physics_agent_radius,
            )
            return np.array([agent_xy[0], agent_xy[1], block_xy[0], block_xy[1], block_angle], dtype=np.float64)

        if not self.easy_mode:
            return np.array(
                [
                    self.np_random.integers(50, 450),
                    self.np_random.integers(50, 450),
                    self.np_random.integers(100, 400),
                    self.np_random.integers(100, 400),
                    self.np_random.uniform(-np.pi, np.pi),
                ],
                dtype=np.float64,
            )

        goal_xy = self.goal_pose[:2]
        approach_angle = self.np_random.uniform(-np.pi, np.pi)
        approach_dir = np.array([np.cos(approach_angle), np.sin(approach_angle)], dtype=np.float64)
        block_distance = float(self.np_random.uniform(10.0, 24.0))
        block_xy = goal_xy - approach_dir * block_distance + self.np_random.normal(0.0, 1.2, size=2)
        block_xy = np.clip(block_xy, 165.0, 347.0)
        block_angle = float(self.goal_pose[2] + self.np_random.uniform(-0.01, 0.01))
        agent_xy = block_xy - approach_dir * float(self.np_random.uniform(100.0, 106.0))
        agent_xy += self.np_random.normal(0.0, 1.2, size=2)
        agent_xy = np.clip(agent_xy, 50.0, 462.0)
        return np.array([agent_xy[0], agent_xy[1], block_xy[0], block_xy[1], block_angle], dtype=np.float64)

    def _set_state(self, state, *, settle: bool = True):
        state = np.asarray(state, dtype=np.float64)
        self.agent.position = list(state[:2])
        self.agent.velocity = (0.0, 0.0)
        self.block.angle = float(state[4])
        self.block.position = list(state[2:4])
        self.block.velocity = (0.0, 0.0)
        self.block.angular_velocity = 0.0
        if settle:
            self.space.step(self.dt)

    def _restore_init_state(self, init_state: Dict):
        if "goal_pose" in init_state:
            self.goal_pose = np.asarray(init_state["goal_pose"], dtype=np.float64)
        state = np.array(
            [
                init_state["agent_position"][0],
                init_state["agent_position"][1],
                init_state["block_position"][0],
                init_state["block_position"][1],
                init_state["block_angle"],
            ],
            dtype=np.float64,
        )
        self._set_state(state, settle=True)
        self.agent.velocity = init_state.get("agent_velocity", (0.0, 0.0))
        self.block.velocity = init_state.get("block_velocity", (0.0, 0.0))
        self.block.angular_velocity = float(
            init_state.get("block_angvel", init_state.get("block_angular_velocity", 0.0))
        )
        self.episode_step = int(init_state.get("episode_step", init_state.get("elapsed_steps", 0)))

    @staticmethod
    def add_segment(space, a, b, radius):
        shape = pymunk.Segment(space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")
        return shape

    @staticmethod
    def add_circle(space, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        space.add(body, shape)
        return body

    @staticmethod
    def add_tee(space, position, angle, scale=30, color="LightSlateGray", mask=None):
        if mask is None:
            mask = pymunk.ShapeFilter.ALL_MASKS()
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.angle = angle
        body.position = position
        body.friction = 1
        space.add(body, shape1, shape2)
        return body, [shape1, shape2]

    def _compute_reward(self):
        return 1.0 if self._get_coverage() >= self.success_threshold else 0.0

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
