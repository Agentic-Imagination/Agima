from __future__ import annotations

import ast
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Discrete, FuncConditional, Text
from PIL import Image, ImageDraw


TILE_SIZE = 24
ASSET_DIR = Path(__file__).resolve().parent / "assets" / "greyblue9_pacman_python"

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_TO_DELTA = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
ACTION_NAMES = {
    UP: "up",
    DOWN: "down",
    LEFT: "left",
    RIGHT: "right",
}
PACMAN_SPRITE_PREFIX = {
    UP: "u",
    DOWN: "d",
    LEFT: "l",
    RIGHT: "r",
}
ACTION_ORDER = (UP, DOWN, LEFT, RIGHT)
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# Deterministic, step-independent ghost personalities (classic Pac-Man targeting):
#   blinky - targets Pacman's current cell (direct chase)
#   pinky  - targets PINKY_AHEAD cells ahead of Pacman's heading (ambush/cut-off)
#   inky   - targets the cell reflected through Blinky about a point INKY_AHEAD ahead of Pacman
#   clyde  - direct chase when far, retreats to its home corner when within CLYDE_NEAR
PINKY_AHEAD = 4
INKY_AHEAD = 2
CLYDE_NEAR = 8
# Difficulty ladder: which personalities are active at each level (index 0 is always Blinky, which
# Inky's targeting references). Level N => N ghosts. Only the ghosts change across levels.
LEVEL_PERSONAS = {
    1: ["blinky"],
    2: ["blinky", "pinky"],
    3: ["blinky", "pinky", "clyde"],
    4: ["blinky", "pinky", "clyde", "inky"],
}
# Per-personality render tint ("blinky" == default red sprite, no tint).
GHOST_COLORS = {
    "pinky": (255, 150, 210),   # pink
    "clyde": (255, 165, 60),    # orange
    "inky": (0, 225, 225),      # cyan
}


class Pacman2DEnv(gym.Env, gym.VLMEnvMixin):
    """Custom deterministic Pacman puzzle rendered with pacman-python assets."""

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}

    def __init__(
        self,
        difficulty: str = "easy",
        grid_size: Optional[int] = None,
        food_count: Optional[int] = None,
        num_ghosts: Optional[int] = None,
        wall_density: Optional[float] = None,
        max_steps: Optional[int] = None,
        horizon: int = 8,
        render_mode: str = "rgb_array",
        draw_grid: bool = False,
        randomize_layout: bool = True,
        seed: Optional[int] = None,
        tile_size: int = TILE_SIZE,
        render_size: Optional[Any] = None,
        num_power_pellets: Optional[int] = None,
        frightened_steps: int = 12,
        ghost_spawn_dist: Optional[int] = None,
        diverse_ghosts: bool = False,
        level: Optional[int] = None,
        braid: bool = False,
        thin_corridors: bool = False,
    ):
        super().__init__()
        if difficulty not in {"easy", "hard"}:
            raise ValueError(f"Unsupported difficulty: {difficulty}")

        self.difficulty = difficulty
        self.grid_size = int(grid_size if grid_size is not None else (9 if difficulty == "easy" else 11))
        # Upper bound raised 12 -> 16 for the larger braided sets. Nothing else in the env assumed
        # 12; note that `max_steps` does NOT scale with grid_size, so a big grid with lots of food
        # needs an explicit max_steps or the oracle plan runs past the horizon.
        if not 7 <= self.grid_size <= 16:
            raise ValueError(f"grid_size must be between 7 and 16, got {self.grid_size}")
        self.food_count = int(food_count if food_count is not None else (4 if difficulty == "easy" else 8))
        if self.food_count < 1:
            raise ValueError(f"food_count must be positive, got {self.food_count}")
        # Number of ghosts (all deterministic optimal-chasers). Default 1 == the original single-ghost
        # env (backward compatible). Ghosts PASS THROUGH each other (may co-occupy a cell); only
        # Pacman-vs-ghost contact/swap is fatal.
        self.num_ghosts = int(num_ghosts if num_ghosts is not None else 1)
        if not 1 <= self.num_ghosts <= 4:
            raise ValueError(f"num_ghosts must be between 1 and 4, got {self.num_ghosts}")
        default_density = 0.22 if difficulty == "easy" else 0.16
        self.wall_density = float(wall_density if wall_density is not None else default_density)
        if not 0.0 <= self.wall_density <= 0.35:
            raise ValueError(f"wall_density must be between 0 and 0.35, got {self.wall_density}")
        self.max_steps = int(max_steps if max_steps is not None else (80 if difficulty == "easy" else 120))
        # ---- Arcade mechanics (opt-in) -------------------------------------------------------
        # num_power_pellets > 0 turns on the "real Pacman" bundle: power pellets that make ghosts
        # edible ("frightened") for `frightened_steps` steps, ghosts that CANNOT reverse 180
        # (classic no-reverse rule), and a MODERATE ghost spawn distance. Default 0 => legacy env,
        # byte-for-byte identical to the pre-arcade behavior.
        self.num_power_pellets = int(num_power_pellets if num_power_pellets is not None else 0)
        if self.num_power_pellets < 0:
            raise ValueError(f"num_power_pellets must be >= 0, got {self.num_power_pellets}")
        self.arcade = self.num_power_pellets > 0
        self.frightened_steps = int(frightened_steps)
        if self.frightened_steps < 1:
            raise ValueError(f"frightened_steps must be >= 1, got {self.frightened_steps}")
        self.ghost_spawn_dist = int(ghost_spawn_dist) if ghost_spawn_dist is not None else (4 if self.arcade else 0)
        # Difficulty ladder: `level` (1-4) selects the active ghost personalities and forces
        # num_ghosts=level (L1 Blinky -> L2 +Pinky -> L3 +Clyde -> L4 +Inky). `self.ghost_personas`
        # is the single source of truth for per-ghost AI; deterministic diverse targeting is
        # step-independent, so the oracle BFS is unaffected. Default == legacy all-Blinky behavior.
        self.level: Optional[int] = None
        if level is not None:
            if int(level) not in LEVEL_PERSONAS:
                raise ValueError(f"level must be one of {sorted(LEVEL_PERSONAS)}, got {level}")
            self.level = int(level)
            self.num_ghosts = self.level
            self.ghost_personas = list(LEVEL_PERSONAS[self.level])
        elif bool(diverse_ghosts):
            self.ghost_personas = ["blinky", "pinky", "inky", "clyde"][: self.num_ghosts]
        else:
            self.ghost_personas = ["blinky"] * self.num_ghosts
        self.diverse_ghosts = any(p != "blinky" for p in self.ghost_personas)
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {self.horizon}")
        self.render_mode = render_mode
        self.draw_grid = bool(draw_grid)
        self.randomize_layout = bool(randomize_layout)
        # Braided layouts (opt-in; default False == byte-for-byte legacy sampling). Recursive
        # division gives every dividing wall EXACTLY ONE gap, so the room-adjacency graph is a
        # TREE and every gap is an articulation point -- a single ghost can seal Pacman into a
        # region. Braiding fills dead ends and carves extra openings until the open region is
        # 2-vertex-connected (every cell has >=2 independent ways out). See _braid_walls.
        self.braid = bool(braid)
        # Best-effort Pac-Man-style corridor thinning: reduce 2x2 fully-open blocks ("prevent fat
        # paths" in shaunlebron/pacman-mazegen's ASP ruleset) without reintroducing a chokepoint or
        # dead end. Some residual 2x2 rooms are intentionally retained when every possible fill
        # would violate those stronger topology guarantees. Requires braid.
        self.thin_corridors = bool(thin_corridors)
        if self.thin_corridors and not self.braid:
            raise ValueError("thin_corridors=True requires braid=True")
        self.tile_size = int(tile_size)
        if self.tile_size < 8:
            raise ValueError(f"tile_size must be at least 8, got {self.tile_size}")

        self.action_space = FuncConditional({"move": Discrete(4), "stop": Text(4)})
        self.render_size: Optional[tuple[int, int]] = None
        self._set_render_size(render_size)

        self.walls = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.food: frozenset[tuple[int, int]] = frozenset()
        # Power pellets (arcade only): eating one frightens all ghosts. Separate from `food`; NOT
        # required for the win condition (win == all `food` collected). Frightened timer in steps.
        self.power_pellets: frozenset[tuple[int, int]] = frozenset()
        self.frightened = 0
        self.pacman_pos = (1, 1)
        # Ghosts are stored as parallel lists (positions + facing dirs); index 0 is the legacy ghost.
        self.ghost_positions: list[tuple[int, int]] = [(self.grid_size - 2, self.grid_size - 2)]
        self.ghost_dirs: list[int] = [LEFT]
        # Ghost home cells (respawn targets when eaten during frightened mode); == start positions.
        self.ghost_homes: list[tuple[int, int]] = [(self.grid_size - 2, self.grid_size - 2)]
        self.pacman_dir = RIGHT
        self.steps = 0
        self.alive = True
        self.won = False
        self.env_feedback = ""
        self.last_solve_info: dict[str, Any] = {}
        self._distance_cache: dict[tuple[bytes, tuple[int, int]], np.ndarray] = {}
        self._wall_cache_key: Optional[bytes] = None

        self.seed(seed)

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    @property
    def ghost_pos(self) -> tuple[int, int]:
        """Legacy alias: position of the first ghost."""
        return self.ghost_positions[0]

    @property
    def ghost_dir(self) -> int:
        """Legacy alias: facing direction of the first ghost."""
        return self.ghost_dirs[0]

    def get_prompt(self, **kwargs) -> str:
        n = self.num_ghosts
        ghost_noun = "the single red ghost" if n == 1 else f"all {n} red ghosts"
        pellet_clause = f" There are also {self.num_power_pellets} large power pellets." if self.arcade else ""
        prompt = (
            f"You are Pacman in a deterministic {self.grid_size}x{self.grid_size} maze. "
            f"Collect every food pellet, avoid {ghost_noun}, and stop only after all food is gone.{pellet_clause} "
            "The rendering uses pacman-python Pacman, ghost, pellet, and maze tile visuals, but the environment physics are custom."
        )
        if self.render_mode == "ansi":
            ghost_glyph = "G is the ghost" if n == 1 else "G is a ghost"
            pellet_glyph = ", O is a power pellet, g is a frightened (edible) ghost" if self.arcade else ""
            prompt += (
                f" In the text grid, # is a wall, o is food, P is Pacman, {ghost_glyph}{pellet_glyph}, "
                "and X means Pacman and a ghost collided."
            )
        prompt += "\n\nAvailable actions:\n"
        prompt += "1. `move`: Move Pacman one cell. Format: `('move', action_id)`\n"
        prompt += "   - 0=up, 1=down, 2=left, 3=right\n"
        prompt += "2. `stop`: End the episode. Format: `('stop', 'stop')`"
        chase = "takes a shortest-path step toward Pacman" if n == 1 else "take a shortest-path step toward Pacman"
        if self.diverse_ghosts and n >= 2:
            persona_desc = {
                "blinky": "the red ghost targets Pacman's current cell",
                "pinky": f"the pink ghost targets {PINKY_AHEAD} cells ahead of Pacman's heading (to cut you off)",
                "clyde": "the orange ghost chases when far but retreats to a corner when close",
                "inky": "the cyan ghost flanks using the red ghost's position",
            }
            personas = [persona_desc.get(p, f"the {p} ghost chases Pacman") for p in self.ghost_personas]
            prompt += (
                "\n\nRules: Pacman moves first. Moving into a wall keeps Pacman in place, then every ghost still moves. "
                "Each ghost deterministically takes one step along the shortest path toward its own target cell, "
                "tie-breaking up, down, left, right, and never reverses direction unless trapped: "
                + "; ".join(personas)
                + ". Ghosts ignore each other and may share a cell. Touching any ghost or swapping cells with any ghost is fatal."
            )
        elif n == 1:
            prompt += (
                "\n\nRules: Pacman moves first. Moving into a wall keeps Pacman in place, then the ghost still moves. "
                f"The ghost deterministically {chase}, tie-breaking up, down, left, right. "
                "Touching the ghost or swapping cells with it is fatal."
            )
        else:
            prompt += (
                "\n\nRules: Pacman moves first. Moving into a wall keeps Pacman in place, then every ghost still moves. "
                f"Each of the {n} ghosts deterministically {chase}, tie-breaking up, down, left, right; "
                "ghosts ignore each other and may share a cell. Touching any ghost or swapping cells with any ghost is fatal."
            )
        if self.arcade:
            ghost_word = "the ghost" if n == 1 else "every ghost"
            it_them = "it" if n == 1 else "them"
            prompt += (
                f" Ghosts never reverse direction unless trapped. Eating a power pellet turns {ghost_word} "
                f"'frightened' (blue) for {self.frightened_steps} steps: while frightened they flee from Pacman, "
                f"and touching {it_them} then eats the ghost (it respawns at its start cell) instead of being fatal."
            )
        prompt += "\n\nSuccess: all food is collected, Pacman is alive, and you then output `('stop', 'stop')`."
        return prompt

    def get_init_state(self) -> Dict[str, Any]:
        state = {
            "backend": "custom_grid_pacman_python_assets",
            "difficulty": self.difficulty,
            "grid_size": int(self.grid_size),
            "food_count": int(self.food_count),
            "wall_density": float(self.wall_density),
            "max_steps": int(self.max_steps),
            "horizon": int(self.horizon),
            "tile_size": int(self.tile_size),
            "walls": self.walls.astype(np.uint8).tolist(),
            "food": [list(pos) for pos in sorted(self.food)],
            "power_pellets": [list(pos) for pos in sorted(self.power_pellets)],
            "num_power_pellets": int(self.num_power_pellets),
            "frightened": int(self.frightened),
            "frightened_steps": int(self.frightened_steps),
            "diverse_ghosts": bool(self.diverse_ghosts),
            "level": self.level,
            "ghost_personas": list(self.ghost_personas),
            "pacman_pos": list(self.pacman_pos),
            "num_ghosts": int(self.num_ghosts),
            "ghost_positions": [list(pos) for pos in self.ghost_positions],
            "ghost_homes": [list(pos) for pos in self.ghost_homes],
            "ghost_dirs": [int(d) for d in self.ghost_dirs],
            # legacy single-ghost fields (first ghost) for backward compatibility with old readers.
            "ghost_pos": list(self.ghost_positions[0]),
            "pacman_dir": int(self.pacman_dir),
            "ghost_dir": int(self.ghost_dirs[0]),
            "steps": int(self.steps),
            "alive": bool(self.alive),
            "won": bool(self.won),
        }
        # Layout-shape flags are emitted ONLY when enabled. init_state feeds stable_hash(), so an
        # unconditional key would change init_state_hash for every legacy episode too and silently
        # break hash continuity with already-generated datasets (train/test forbidding,
        # hashes_test_set_v10.json). Absent == legacy == the pre-braid hash.
        if self.braid:
            state["braid"] = True
        if self.thin_corridors:
            state["thin_corridors"] = True
        if self.render_size is not None:
            width, height = self.render_size
            state["render_size"] = int(width) if width == height else [int(width), int(height)]
        return state

    def reset(self, *, seed: Optional[int] = None, options=None, init_state: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        if isinstance(init_state, dict) and "render_size" in init_state:
            self._set_render_size(init_state["render_size"])

        # Absorb episode-shaping config from a loaded init_state so eval behaves like data-gen even
        # if this env was constructed with defaults. Only on reset (not per-step) so it persists
        # across the episode. `ghost_personas` is authoritative for per-ghost AI + tint.
        if isinstance(init_state, dict):
            # These flags are serialized only when enabled. Absence explicitly means
            # legacy/unbraided, which also prevents a reusable braided env from leaking its config
            # into a legacy episode loaded later.
            self.braid = bool(init_state.get("braid", False))
            self.thin_corridors = bool(init_state.get("thin_corridors", False))
            if self.thin_corridors and not self.braid:
                raise ValueError("init_state thin_corridors=True requires braid=True")
            npp = init_state.get("num_power_pellets")
            if npp is None and init_state.get("power_pellets"):
                npp = len(init_state["power_pellets"])
            if npp is not None:
                self.num_power_pellets = int(npp)
                self.arcade = self.num_power_pellets > 0
            if init_state.get("frightened_steps") is not None:
                self.frightened_steps = int(init_state["frightened_steps"])
            if init_state.get("diverse_ghosts") is not None:
                self.diverse_ghosts = bool(init_state["diverse_ghosts"])
            if init_state.get("ghost_personas"):
                self.ghost_personas = [str(p) for p in init_state["ghost_personas"]]
                self.diverse_ghosts = any(p != "blinky" for p in self.ghost_personas)
            if init_state.get("level") is not None:
                self.level = int(init_state["level"])

        self.env_feedback = "Environment reset."
        state = self._generate_initial_state() if init_state is None else self._coerce_state(init_state)
        self._apply_state(state)
        return self.render(), self._get_info()

    def step(self, action: str):
        terminated = False
        truncated = False
        reward = 0.0
        self.env_feedback = None

        try:
            parsed = ast.literal_eval(action)
        except Exception as exc:
            self.env_feedback = f"Cannot parse action string {action!r}: {exc}."
            return self.render(), reward, terminated, truncated, self._get_info()

        if not isinstance(parsed, tuple) or len(parsed) != 2:
            self.env_feedback = "The action format is invalid. Use a 2-tuple `(action_name, payload)`."
            return self.render(), reward, terminated, truncated, self._get_info()

        branch, payload = parsed
        if branch == "stop":
            if not self.action_space["stop"].contains(payload):
                self.env_feedback = f"The stop payload {payload!r} is invalid."
            else:
                terminated = True
                reward = self._compute_stop_reward()
                self.env_feedback = "Success." if reward > 0 else "Stopped before all food was collected."
            return self.render(), reward, terminated, truncated, self._get_info()

        if branch not in {"act", "move"}:
            self.env_feedback = f"The action name '{branch}' is not recognized."
            return self.render(), reward, terminated, truncated, self._get_info()
        if not self.action_space["move"].contains(payload):
            self.env_feedback = f"The action {payload!r} is not in the valid action space."
            return self.render(), reward, terminated, truncated, self._get_info()

        state, reward, terminated, truncated, feedback = self._transition_state(self._state_dict(), int(payload))
        self._apply_state(state)
        self.env_feedback = feedback
        return self.render(), reward, terminated, truncated, self._get_info()

    def render(self, mode: Optional[str] = None):
        return self.render_state(self.get_init_state(), mode=mode or self.render_mode)

    def render_state(self, state: Dict[str, Any], mode: Optional[str] = None):
        mode = mode or self.render_mode
        if isinstance(state, dict) and "render_size" in state:
            render_size = self._normalize_render_size(state.get("render_size"))
        else:
            render_size = self.render_size
        coerced = self._coerce_state(state)
        if mode == "ansi":
            return self._render_ansi(coerced)
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        return self._render_rgb(coerced, render_size=render_size)

    def close(self):
        return None

    @staticmethod
    def _normalize_render_size(render_size: Optional[Any]) -> Optional[tuple[int, int]]:
        if render_size is None:
            return None
        if isinstance(render_size, (int, np.integer)):
            width = height = int(render_size)
        elif isinstance(render_size, (tuple, list)) and len(render_size) == 2:
            width = int(render_size[0])
            height = int(render_size[1])
        else:
            raise ValueError(f"render_size must be an int or a 2-item sequence, got {render_size!r}")
        if width < 8 or height < 8:
            raise ValueError(f"render_size must be at least 8 pixels per side, got {(width, height)}")
        return width, height

    def _set_render_size(self, render_size: Optional[Any]) -> None:
        self.render_size = self._normalize_render_size(render_size)
        if self.render_size is None:
            width = height = self.grid_size * self.tile_size
        else:
            width, height = self.render_size
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)

    def solve(self, strategy: str = "receding_horizon", horizon: Optional[int] = None, num_steps: Optional[int] = None):
        if strategy not in {None, "receding_horizon", "tree_search"}:
            raise ValueError(f"Unsupported solve strategy: {strategy}")
        search_horizon = int(horizon if horizon is not None else self.horizon)
        if search_horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {search_horizon}")
        if num_steps is not None and num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {num_steps}")

        state = self._state_dict()
        actions: list[str] = []
        max_moves = min(self.max_steps - int(state["steps"]), (int(num_steps) - 1) if num_steps is not None else self.max_steps)
        shortest_plan = self._plan_safe(state, max_depth=max_moves)
        if shortest_plan is not None:
            self.last_solve_info = {
                "strategy_requested": strategy or "receding_horizon",
                "planner_used": "greedy_safe" if (self.arcade or self.num_ghosts >= 3 or self.food_count >= 10) else "shortest_safe",
                "fallback_planner": "receding_horizon",
                "horizon": int(search_horizon),
                "max_moves": int(max_moves),
                "plan_length": int(len(shortest_plan)),
            }
            actions.extend(str(("move", action_id)) for action_id in shortest_plan[:max_moves])
            if len(shortest_plan) <= max_moves:
                actions.append("('stop', 'stop')")
            return actions

        fallback_steps = 0
        visited = {self._state_key(state)}
        for _ in range(max_moves):
            if state["alive"] and not state["food"]:
                actions.append("('stop', 'stop')")
                self.last_solve_info = {
                    "strategy_requested": strategy or "receding_horizon",
                    "planner_used": "receding_horizon",
                    "fallback_from": "shortest_safe",
                    "horizon": int(search_horizon),
                    "max_moves": int(max_moves),
                    "plan_length": int(fallback_steps),
                }
                return actions
            action_id = self._oracle_action(state, search_horizon, avoid_keys=visited)
            if action_id is None:
                break
            child, _, terminated, truncated, _ = self._transition_state(state, action_id)
            if terminated or truncated:
                break
            actions.append(str(("move", action_id)))
            fallback_steps += 1
            state = child
            visited.add(self._state_key(state))

        actions.append("('stop', 'stop')")
        self.last_solve_info = {
            "strategy_requested": strategy or "receding_horizon",
            "planner_used": "receding_horizon",
            "fallback_from": "shortest_safe",
            "horizon": int(search_horizon),
            "max_moves": int(max_moves),
            "plan_length": int(fallback_steps),
        }
        return actions

    def _shortest_safe_plan(
        self,
        state: Dict[str, Any],
        max_depth: int,
        max_expansions: int = 200_000,
    ) -> Optional[list[int]]:
        state = self._coerce_state(state)
        if not state["alive"]:
            return None
        if not state["food"]:
            return []

        start_key = self._state_key(state)
        visited = {start_key}
        queue = deque([(state, [])])
        expansions = 0
        while queue and expansions < max_expansions:
            node, path = queue.popleft()
            expansions += 1
            if len(path) >= max_depth:
                continue
            for action_id in ACTION_ORDER:
                child, _, terminated, truncated, _ = self._transition_state(node, action_id)
                if terminated or truncated:
                    continue
                if tuple(child["pacman_pos"]) == tuple(node["pacman_pos"]):
                    continue
                child_key = self._state_key(child)
                if child_key in visited:
                    continue
                child_path = [*path, action_id]
                if child["alive"] and not child["food"]:
                    return child_path
                visited.add(child_key)
                queue.append((child, child_path))
        return None

    def _plan_safe(self, state: Dict[str, Any], max_depth: int, max_expansions: int = 200_000) -> Optional[list[int]]:
        """Dispatch: greedy food-by-food safe planner for arcade or >=3 ghosts (the all-food joint BFS
        explodes with many ghosts OR many food -- e.g. food12 has 2**12 food-subsets); exact
        shortest-safe BFS otherwise (legacy small-food layouts, already validated). Both are
        deterministic and return a plan that collects all food without ever being caught, or None."""
        if self.arcade or self.num_ghosts >= 3 or self.food_count >= 10:
            return self._greedy_safe_plan(state, max_depth)
        return self._shortest_safe_plan(state, max_depth, max_expansions=max_expansions)

    def _greedy_safe_plan(
        self,
        state: Dict[str, Any],
        max_depth: int,
        per_food_expansions: int = 30_000,
    ) -> Optional[list[int]]:
        """Deterministic greedy oracle for many-ghost levels: repeatedly find a SHORT safe path that
        collects one more food (nearest reachable, ghosts simulated forward), re-planning after each.
        Decomposes the exponential all-food joint search into a sequence of shallow single-food searches.
        Not provably shortest, but safe + collects all food; returns None if some food becomes unreachable."""
        node = self._coerce_state(state)
        if not node["alive"]:
            return None
        if not node["food"]:
            return []
        plan: list[int] = []
        while node["food"]:
            seg = self._safe_plan_to_next_food(node, max_depth - len(plan), per_food_expansions)
            if seg is None:
                return None
            for action_id in seg:
                node, _, terminated, truncated, _ = self._transition_state(node, action_id)
                if terminated or truncated:
                    return None
            plan.extend(seg)
            if len(plan) > max_depth:
                return None
        return plan

    def _safe_plan_to_next_food(
        self,
        state: Dict[str, Any],
        max_depth: int,
        max_expansions: int,
    ) -> Optional[list[int]]:
        """Shortest safe move sequence (deterministic, ACTION_ORDER tie-break) that eats >=1 food while
        staying alive. Shallow (nearest food is close) so cheap even with 3-4 ghosts."""
        state = self._coerce_state(state)
        start_food = len(state["food"])
        if start_food == 0:
            return []
        visited = {self._state_key(state)}
        queue = deque([(state, [])])
        expansions = 0
        while queue and expansions < max_expansions:
            node, path = queue.popleft()
            expansions += 1
            if len(path) >= max_depth:
                continue
            for action_id in ACTION_ORDER:
                child, _, terminated, truncated, _ = self._transition_state(node, action_id)
                if terminated or truncated:
                    continue
                if tuple(child["pacman_pos"]) == tuple(node["pacman_pos"]):
                    continue
                child_path = [*path, action_id]
                if len(child["food"]) < start_food:  # ate a food this segment, still alive
                    return child_path
                child_key = self._state_key(child)
                if child_key in visited:
                    continue
                visited.add(child_key)
                queue.append((child, child_path))
        return None

    def simulate_action_from_state(self, state: Dict[str, Any], action_id: int, include_observation: bool = False) -> Dict[str, Any]:
        next_state, reward, terminated, truncated, feedback = self._transition_state(self._coerce_state(state), int(action_id))
        result = {
            "state": self._json_state(next_state),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": self._info_from_state(next_state, feedback),
        }
        if include_observation:
            result["observation"] = self.render_state(result["state"], mode="rgb_array")
        return result

    def simulate_stop_from_state(self, state: Dict[str, Any], include_observation: bool = False) -> Dict[str, Any]:
        coerced = self._coerce_state(state)
        reward = 1.0 if coerced["alive"] and not coerced["food"] else 0.0
        feedback = "Success." if reward > 0 else "Stopped before all food was collected."
        result = {
            "state": self._json_state(coerced),
            "reward": float(reward),
            "terminated": True,
            "truncated": False,
            "info": self._info_from_state(coerced, feedback),
        }
        if include_observation:
            result["observation"] = self.render_state(result["state"], mode="rgb_array")
        return result

    def counterfactual_transitions(
        self,
        state: Optional[Dict[str, Any]] = None,
        actions: Optional[Iterable[int]] = None,
        include_stop: bool = True,
    ) -> list[Dict[str, Any]]:
        state = self.get_init_state() if state is None else state
        records = [
            {"action": str(("move", int(action_id))), **self.simulate_action_from_state(state, int(action_id))}
            for action_id in tuple(ACTION_ORDER if actions is None else actions)
        ]
        if include_stop:
            records.append({"action": "('stop', 'stop')", **self.simulate_stop_from_state(state)})
        return records

    def _generate_initial_state(self) -> Dict[str, Any]:
        if not self.randomize_layout:
            walls = self._template_walls()
            state = self._sample_entities(walls)
            if state is None:
                raise RuntimeError("Failed to sample a valid Pacman2D initial state on the template maze.")
            return state

        for _ in range(512):
            walls = self._sample_connected_walls()
            # Braiding trades wall density for topology (it fills dead ends and carves openings),
            # so the pre-braid wall floor no longer applies -- _sample_connected_walls already
            # enforced it before braiding.
            if not self.braid and self._inner_wall_count(walls) < self._min_inner_wall_count():
                continue
            state = self._sample_entities(walls)
            if state is not None:
                return state
        raise RuntimeError("Failed to sample a valid Pacman2D initial state.")

    def _sample_connected_walls(self) -> np.ndarray:
        for _ in range(64):
            walls = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            walls[0, :] = True
            walls[-1, :] = True
            walls[:, 0] = True
            walls[:, -1] = True

            depth = max(2, min(5, int(round(self.wall_density * 18))))
            if self.difficulty == "hard":
                depth = min(5, depth + 1)
            self._divide_region(walls, 1, self.grid_size - 2, 1, self.grid_size - 2, depth)
            self._add_wall_segments(walls, target_inner_walls=self._min_inner_wall_count())
            # Braid AFTER the wall top-up: _add_wall_segments only tests connectivity, so running
            # it later would re-introduce the dead ends/chokepoints braiding just removed.
            if self.braid:
                self._braid_walls(walls)
                if self.thin_corridors:
                    self._thin_corridors(walls)

            largest = self._largest_open_component(walls)
            if largest is None or len(largest) < self.food_count + 6:
                continue
            open_count = int(np.count_nonzero(~walls))
            if len(largest) == open_count:
                return walls
        return walls

    def _divide_region(self, walls: np.ndarray, top: int, bottom: int, left: int, right: int, depth: int) -> None:
        height = bottom - top + 1
        width = right - left + 1
        if depth <= 0 or height < 3 or width < 3:
            return

        if width > height:
            orientation = "vertical"
        elif height > width:
            orientation = "horizontal"
        else:
            orientation = "vertical" if self.np_random.random() < 0.5 else "horizontal"

        if orientation == "vertical":
            candidates = [col for col in range(left + 1, right) if not walls[top: bottom + 1, col].all()]
            if not candidates:
                return
            wall_col = int(candidates[int(self.np_random.integers(len(candidates)))])
            gap_row = int(self.np_random.integers(top, bottom + 1))
            for row in range(top, bottom + 1):
                if row != gap_row:
                    walls[row, wall_col] = True
            self._divide_region(walls, top, bottom, left, wall_col - 1, depth - 1)
            self._divide_region(walls, top, bottom, wall_col + 1, right, depth - 1)
        else:
            candidates = [row for row in range(top + 1, bottom) if not walls[row, left: right + 1].all()]
            if not candidates:
                return
            wall_row = int(candidates[int(self.np_random.integers(len(candidates)))])
            gap_col = int(self.np_random.integers(left, right + 1))
            for col in range(left, right + 1):
                if col != gap_col:
                    walls[wall_row, col] = True
            self._divide_region(walls, top, wall_row - 1, left, right, depth - 1)
            self._divide_region(walls, wall_row + 1, bottom, left, right, depth - 1)

    def _add_wall_segments(self, walls: np.ndarray, target_inner_walls: int) -> None:
        attempts = 0
        while self._inner_wall_count(walls) < target_inner_walls and attempts < 256:
            attempts += 1
            candidate = walls.copy()
            horizontal = bool(self.np_random.integers(2))
            length = int(self.np_random.integers(2, min(5, self.grid_size - 2) + 1))
            if horizontal:
                row = int(self.np_random.integers(1, self.grid_size - 1))
                col = int(self.np_random.integers(1, self.grid_size - length))
                candidate[row, col: col + length] = True
            else:
                row = int(self.np_random.integers(1, self.grid_size - length))
                col = int(self.np_random.integers(1, self.grid_size - 1))
                candidate[row: row + length, col] = True
            if self._open_region_is_connected(candidate):
                walls[:, :] = candidate

    # ---- Braiding -----------------------------------------------------------------------
    # Recursive division is a TREE generator: each dividing wall keeps exactly one gap, so that
    # gap is a cut vertex and one ghost can seal Pacman in. Braiding repairs the topology in two
    # phases, both deterministic (no RNG -> the entity-sampling RNG stream is untouched):
    #   1. open dead ends  -- for each leaf, carve a wall neighbour that leads on somewhere else,
    #      turning the stub into a through-route. Only if no such carve exists is the leaf FILLED
    #      (walling a leaf can never disconnect the map). Carving is tried first because filling
    #      cascades -- every fill creates a new leaf -- and eats whole corridors, which collapses
    #      the map diameter and then breaks the legacy far-ghost spawn rule.
    #   2. carve openings  -- greedily open the inner wall cell that most reduces the articulation
    #      -point count, restricted to walls with >=2 open neighbours so a carve always joins two
    #      existing passages (creating a cycle) and can never introduce a new dead end.
    # Terminates when the open region is 2-vertex-connected, or when the carve budget runs out.

    def _dead_end_cells(self, walls: np.ndarray) -> list[tuple[int, int]]:
        """Open cells with at most one open neighbour (leaves of the open-cell graph)."""
        out: list[tuple[int, int]] = []
        for row in range(1, self.grid_size - 1):
            for col in range(1, self.grid_size - 1):
                if walls[row, col]:
                    continue
                degree = sum(
                    1 for dr, dc in ACTION_TO_DELTA.values()
                    if self._is_open(walls, (row + dr, col + dc))
                )
                if degree <= 1:
                    out.append((row, col))
        return out

    def _articulation_points(self, walls: np.ndarray) -> set[tuple[int, int]]:
        """Cut vertices of the open-cell graph: cells whose removal splits the map in two.
        Iterative Tarjan (the grid is tiny, but recursion depth is avoided on principle)."""
        cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if not walls[r, c]
        ]
        adj = {
            cell: [
                nb for nb in (
                    (cell[0] + dr, cell[1] + dc) for dr, dc in ACTION_TO_DELTA.values()
                )
                if self._is_open(walls, nb)
            ]
            for cell in cells
        }
        disc: dict[tuple[int, int], int] = {}
        low: dict[tuple[int, int], int] = {}
        cuts: set[tuple[int, int]] = set()
        timer = 0
        for root in cells:
            if root in disc:
                continue
            disc[root] = low[root] = timer
            timer += 1
            root_children = 0
            stack = [(root, None, iter(adj[root]))]
            while stack:
                node, parent, neighbours = stack[-1]
                descended = False
                for nb in neighbours:
                    if nb == parent:
                        continue
                    if nb in disc:
                        low[node] = min(low[node], disc[nb])
                    else:
                        disc[nb] = low[nb] = timer
                        timer += 1
                        stack.append((nb, node, iter(adj[nb])))
                        if node == root:
                            root_children += 1
                        descended = True
                        break
                if not descended:
                    stack.pop()
                    if stack:
                        parent_node = stack[-1][0]
                        low[parent_node] = min(low[parent_node], low[node])
                        if parent_node != root and low[node] >= disc[parent_node]:
                            cuts.add(parent_node)
            if root_children > 1:
                cuts.add(root)
        return cuts

    def _braid_walls(self, walls: np.ndarray, max_carves: Optional[int] = None) -> None:
        """In-place: fill dead ends, then carve until no articulation points remain (or budget)."""
        budget = max_carves if max_carves is not None else self.grid_size * 4

        # Phase 1: open each dead end into a through-route; fill it only if that is impossible.
        for _ in range(self.grid_size * self.grid_size):
            leaves = self._dead_end_cells(walls)
            if not leaves:
                break
            for cell in leaves:
                if walls[cell]:
                    continue   # an earlier carve in this sweep already resolved it
                carve = None
                for dr, dc in ACTION_TO_DELTA.values():
                    nb = (cell[0] + dr, cell[1] + dc)
                    if not self._in_bounds(nb) or not walls[nb]:
                        continue
                    if nb[0] in (0, self.grid_size - 1) or nb[1] in (0, self.grid_size - 1):
                        continue   # never breach the border
                    # the carved cell must lead on somewhere that is not the dead end itself
                    onward = [
                        (nb[0] + r2, nb[1] + c2) for r2, c2 in ACTION_TO_DELTA.values()
                        if self._is_open(walls, (nb[0] + r2, nb[1] + c2)) and (nb[0] + r2, nb[1] + c2) != cell
                    ]
                    if onward:
                        carve = nb
                        break
                if carve is not None:
                    walls[carve] = False
                else:
                    walls[cell] = True   # hopeless stub: wall it off (a leaf is never a cut vertex)

        # Phase 2: greedy carving. Each iteration opens the single wall cell that leaves the fewest
        # cut vertices (ties broken by dead-end count, then position, so the result is seed-stable).
        for _ in range(budget):
            if not self._articulation_points(walls):
                break
            best: Optional[tuple[tuple[int, int, int], tuple[int, int]]] = None
            for row in range(1, self.grid_size - 1):
                for col in range(1, self.grid_size - 1):
                    if not walls[row, col]:
                        continue
                    open_neighbours = sum(
                        1 for dr, dc in ACTION_TO_DELTA.values()
                        if self._is_open(walls, (row + dr, col + dc))
                    )
                    if open_neighbours < 2:
                        continue   # would dangle a new dead end instead of joining two passages
                    walls[row, col] = False
                    score = (
                        len(self._articulation_points(walls)),
                        len(self._dead_end_cells(walls)),
                        row * self.grid_size + col,
                    )
                    walls[row, col] = True
                    if best is None or score < best[0]:
                        best = (score, (row, col))
            if best is None:
                break
            walls[best[1]] = False

    def _open_squares(self, walls: np.ndarray) -> list[tuple[int, int]]:
        """Top-left corners of every 2x2 block that is fully open (a 'fat path')."""
        out: list[tuple[int, int]] = []
        for row in range(self.grid_size - 1):
            for col in range(self.grid_size - 1):
                if (not walls[row, col] and not walls[row + 1, col]
                        and not walls[row, col + 1] and not walls[row + 1, col + 1]):
                    out.append((row, col))
        return out

    def _thin_corridors(self, walls: np.ndarray, max_sweeps: Optional[int] = None) -> None:
        """In-place: reduce fully-open 2x2 blocks while preserving braided-map guarantees.

        A fill is only accepted if the map stays connected with NO articulation point and NO dead
        end -- i.e. thinning may never undo braiding. Blocks with no acceptable fill are left open
        (better a residual room than a chokepoint). Deterministic: candidates are scored by
        (remaining fat blocks, position), so the result is seed-stable."""
        sweeps = max_sweeps if max_sweeps is not None else self.grid_size
        for _ in range(sweeps):
            squares = self._open_squares(walls)
            if not squares:
                break
            changed = False
            for (row, col) in squares:
                quad = [(row, col), (row + 1, col), (row, col + 1), (row + 1, col + 1)]
                if any(walls[cell] for cell in quad):
                    continue   # an earlier fill in this sweep already broke this block up
                best: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
                for cell in quad:
                    walls[cell] = True
                    acceptable = (not self._articulation_points(walls)
                                  and not self._dead_end_cells(walls)
                                  and self._open_region_is_connected(walls))
                    score = (len(self._open_squares(walls)),
                             cell[0] * self.grid_size + cell[1]) if acceptable else None
                    walls[cell] = False
                    if score is not None and (best is None or score < best[0]):
                        best = (score, cell)
                if best is not None:
                    walls[best[1]] = True
                    changed = True
            if not changed:
                break

    def _open_region_is_connected(self, walls: np.ndarray) -> bool:
        largest = self._largest_open_component(walls)
        if largest is None:
            return False
        return len(largest) == int(np.count_nonzero(~walls))

    def _inner_wall_count(self, walls: np.ndarray) -> int:
        inner = walls[1:-1, 1:-1]
        return int(np.count_nonzero(inner))

    def _min_inner_wall_count(self) -> int:
        if self.wall_density <= 0.0:
            return 0
        if self.difficulty == "easy":
            return max(12, self.grid_size + 3)
        return max(14, self.grid_size + 4)

    def _template_walls(self) -> np.ndarray:
        walls = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        walls[0, :] = True
        walls[-1, :] = True
        walls[:, 0] = True
        walls[:, -1] = True
        for row in range(2, self.grid_size - 2, 3):
            for col in range(2, self.grid_size - 2):
                if col % 4 != 0:
                    walls[row, col] = True
        return walls

    def _sample_entities(self, walls: np.ndarray) -> Optional[Dict[str, Any]]:
        component = self._largest_open_component(walls)
        if component is None or len(component) < self.food_count + self.num_ghosts + self.num_power_pellets + 5:
            return None
        cells = sorted(component)
        pacman_candidates = list(cells)
        self.np_random.shuffle(pacman_candidates)
        # Moderate spawn band (a live threat) whenever ghost_spawn_dist>0 -- arcade default 4, OR an
        # explicit override for the legacy single-ghost env (harder OOD test where the ghost is close);
        # otherwise legacy far spawn (max(5, grid-2), farthest-first).
        use_band = self.ghost_spawn_dist > 0
        min_ghost_dist = self.ghost_spawn_dist if use_band else max(5, self.grid_size - 2)
        band_max = self.ghost_spawn_dist + 3 if use_band else self.grid_size * self.grid_size
        food_radius = max(4, min(self.grid_size - 2, self.food_count + 2))
        ghost_gap = 3 if self.difficulty == "easy" else 4
        ghost_sep = 2   # min BFS separation between two ghosts' start cells (avoid trivial overlap)

        for pacman_pos in pacman_candidates:
            pacman_pos = tuple(pacman_pos)
            pacman_distances = self._bfs_distances(walls, pacman_pos)
            far_cells = [pos for pos in cells if min_ghost_dist <= int(pacman_distances[pos]) <= band_max]
            if len(far_cells) < self.num_ghosts and use_band:
                far_cells = [pos for pos in cells if int(pacman_distances[pos]) >= min_ghost_dist]
            if len(far_cells) < self.num_ghosts:
                continue
            if use_band:
                self.np_random.shuffle(far_cells)
                ghost_pool = far_cells[: max(12, self.num_ghosts * 6)]
            else:
                far_cells.sort(key=lambda pos: (int(pacman_distances[pos]), pos[0], pos[1]), reverse=True)
                ghost_pool = far_cells[: min(len(far_cells), max(12, self.num_ghosts * 6))]

            for _ in range(6):
                self.np_random.shuffle(ghost_pool)
                ghosts = self._pick_ghost_set(ghost_pool, walls, ghost_sep)
                if ghosts is None:
                    continue
                ghost_distances = [self._bfs_distances(walls, gp) for gp in ghosts]
                ghost_set = set(ghosts)
                food_candidates = []
                for pos in cells:
                    if pos == pacman_pos or pos in ghost_set:
                        continue
                    pacman_distance = int(pacman_distances[pos])
                    if pacman_distance < 1:
                        continue
                    reachable = [int(gd[pos]) for gd in ghost_distances if int(gd[pos]) >= 0]
                    nearest_ghost = min(reachable) if reachable else self.grid_size * self.grid_size
                    if use_band:
                        # Moderate/close spawn: whole-episode solvability is guaranteed by the final
                        # _plan_safe check, so the strict per-cell food-safety heuristic below is
                        # redundant and, with a close ghost + many food, makes layout sampling thrash.
                        # Only forbid food on a ghost's own spawn cell; allow food anywhere reachable
                        # (this also makes the task harder — food can sit near the ghost — as desired).
                        if nearest_ghost < 1:
                            continue
                    else:
                        # Legacy: keep the conservative placement (food near Pacman, safe from ghosts).
                        if pacman_distance > food_radius:
                            continue
                        if nearest_ghost <= 2 or nearest_ghost <= pacman_distance + ghost_gap:
                            continue
                    food_candidates.append(pos)

                if len(food_candidates) < self.food_count:
                    continue
                food_candidates.sort(key=lambda pos: (int(pacman_distances[pos]), pos[0], pos[1]))
                pool = food_candidates[: min(len(food_candidates), max(self.food_count + 4, self.food_count * 3))]
                self.np_random.shuffle(pool)
                food = frozenset(tuple(pos) for pos in pool[: self.food_count])

                # Power pellets (arcade only): free cells (not Pacman/ghost/food), >=2 from Pacman.
                power_pellets: frozenset[tuple[int, int]] = frozenset()
                if self.arcade and self.num_power_pellets > 0:
                    used = {pacman_pos, *ghost_set} | set(food)
                    pellet_cands = [pos for pos in cells if pos not in used and int(pacman_distances[pos]) >= 2]
                    if len(pellet_cands) < self.num_power_pellets:
                        continue
                    self.np_random.shuffle(pellet_cands)
                    power_pellets = frozenset(tuple(p) for p in pellet_cands[: self.num_power_pellets])

                candidate_state = {
                    "walls": walls,
                    "food": food,
                    "power_pellets": power_pellets,
                    "frightened": 0,
                    "pacman_pos": pacman_pos,
                    "ghost_positions": [tuple(gp) for gp in ghosts],
                    "ghost_homes": [tuple(gp) for gp in ghosts],
                    "ghost_dirs": [LEFT] * len(ghosts),
                    "pacman_dir": RIGHT,
                    "steps": 0,
                    "alive": True,
                    "won": False,
                }
                plan_depth = min(self.max_steps, max(18, self.grid_size * 3))
                if self._plan_safe(candidate_state, max_depth=plan_depth, max_expansions=50_000) is None:
                    continue
                return candidate_state
        return None

    def _pick_ghost_set(
        self, ghost_pool: list[tuple[int, int]], walls: np.ndarray, ghost_sep: int
    ) -> Optional[list[tuple[int, int]]]:
        """Greedily pick `num_ghosts` start cells from the (shuffled) pool, keeping them at least
        `ghost_sep` apart so the ghosts do not start stacked."""
        chosen: list[tuple[int, int]] = []
        for pos in ghost_pool:
            pos = tuple(pos)
            if all(self._bfs_distance(walls, pos, c) >= ghost_sep for c in chosen):
                chosen.append(pos)
            if len(chosen) == self.num_ghosts:
                return chosen
        return None

    def _largest_open_component(self, walls: np.ndarray) -> Optional[list[tuple[int, int]]]:
        visited = np.zeros_like(walls, dtype=bool)
        best: Optional[list[tuple[int, int]]] = None
        for row in range(1, self.grid_size - 1):
            for col in range(1, self.grid_size - 1):
                if visited[row, col] or walls[row, col]:
                    continue
                component: list[tuple[int, int]] = []
                queue = deque([(row, col)])
                visited[row, col] = True
                while queue:
                    cur = queue.popleft()
                    component.append(cur)
                    for dr, dc in ACTION_TO_DELTA.values():
                        nxt = (cur[0] + dr, cur[1] + dc)
                        if not self._in_bounds(nxt) or visited[nxt] or walls[nxt]:
                            continue
                        visited[nxt] = True
                        queue.append(nxt)
                if best is None or len(component) > len(best):
                    best = component
        return best

    def _transition_state(self, state: Dict[str, Any], action_id: int):
        state = self._coerce_state(state)
        walls = state["walls"]
        food = set(state["food"])
        power = set(state["power_pellets"])
        ghost_positions = list(state["ghost_positions"])
        ghost_dirs = list(state["ghost_dirs"])
        ghost_homes = list(state["ghost_homes"])
        n_ghosts = len(ghost_positions)
        pacman_pos = tuple(state["pacman_pos"])
        pacman_dir = int(action_id)
        steps = int(state["steps"]) + 1

        if not state["alive"]:
            next_state = dict(state, steps=steps, alive=False, won=False, pacman_dir=pacman_dir)
            return next_state, -1.0, True, False, "Pacman is already dead."
        if action_id not in ACTION_TO_DELTA:
            next_state = dict(state, steps=steps)
            return next_state, 0.0, False, steps >= self.max_steps, f"Invalid move id {action_id!r}."

        dr, dc = ACTION_TO_DELTA[action_id]
        attempted_pos = (pacman_pos[0] + dr, pacman_pos[1] + dc)
        bumped_wall = not self._is_open(walls, attempted_pos)
        new_pacman_pos = pacman_pos if bumped_wall else attempted_pos

        food_eaten = 0
        if new_pacman_pos in food:
            food.remove(new_pacman_pos)
            food_eaten = 1

        # Power pellet -> frighten all ghosts (arcade only; `power` is empty otherwise).
        ate_power = False
        if new_pacman_pos in power:
            power.remove(new_pacman_pos)
            ate_power = True
        frightened_now = self.frightened_steps if ate_power else int(state["frightened"])
        edible = self.arcade and frightened_now > 0

        new_ghost_positions = list(ghost_positions)
        new_ghost_dirs = list(ghost_dirs)
        alive = True
        eaten: set[int] = set()
        feedback = "Action executed successfully."

        # Phase A: Pacman steps onto ghost cell(s). Edible -> eat; else fatal.
        on_pacman = [i for i, gp in enumerate(ghost_positions) if gp == new_pacman_pos]
        if on_pacman:
            if edible:
                eaten.update(on_pacman)
            else:
                alive = False
                feedback = "Pacman ran into the ghost." if n_ghosts == 1 else "Pacman ran into a ghost."

        no_reverse = self.arcade or self.diverse_ghosts
        blinky_pos = ghost_positions[0] if ghost_positions else new_pacman_pos
        if alive:
            # Phase B: ghosts move. Eaten -> respawn home; others move toward their target (frightened
            # -> flee from Pacman; diverse -> per-personality target; else direct chase), never
            # reversing 180 unless trapped. Ghosts pass through each other, computed independently.
            for i in range(n_ghosts):
                if i in eaten:
                    new_ghost_positions[i] = ghost_homes[i]
                    new_ghost_dirs[i] = LEFT
                    continue
                gp, gd = ghost_positions[i], ghost_dirs[i]
                if edible:
                    dest, flee = new_pacman_pos, True
                elif self.diverse_ghosts:
                    persona = self.ghost_personas[i] if i < len(self.ghost_personas) else "blinky"
                    dest, flee = self._ghost_target(walls, persona, new_pacman_pos, pacman_dir, gp, blinky_pos), False
                else:
                    dest, flee = new_pacman_pos, False
                ngp = self._next_ghost_pos(walls, gp, gd, dest, flee=flee, no_reverse=no_reverse)
                new_ghost_positions[i] = ngp
                new_ghost_dirs[i] = self._direction_between(gp, ngp, gd)
            # Phase C: a ghost lands on Pacman or swaps cells. Edible -> eat; else fatal.
            for i in range(n_ghosts):
                if i in eaten:
                    continue
                ngp, gp = new_ghost_positions[i], ghost_positions[i]
                contact = (ngp == new_pacman_pos) or (ngp == pacman_pos and new_pacman_pos == gp)
                if not contact:
                    continue
                if edible:
                    eaten.add(i)
                    new_ghost_positions[i] = ghost_homes[i]
                    new_ghost_dirs[i] = LEFT
                else:
                    alive = False
                    if ngp == new_pacman_pos:
                        feedback = "The ghost caught Pacman." if n_ghosts == 1 else "A ghost caught Pacman."
                    else:
                        feedback = ("Pacman and the ghost swapped cells." if n_ghosts == 1
                                    else "Pacman and a ghost swapped cells.")
                    break

        if alive:
            if eaten and ate_power:
                feedback = "Ate a power pellet and a frightened ghost."
            elif eaten:
                feedback = "Ate a frightened ghost."
            elif ate_power:
                feedback = "Ate a power pellet - ghosts are frightened."
            elif food_eaten:
                feedback = "Food consumed."
            elif bumped_wall:
                feedback = "Cannot move into a wall."

        new_frightened = self.frightened_steps if ate_power else max(0, int(state["frightened"]) - 1)
        won = alive and not food
        terminated = not alive
        truncated = bool(steps >= self.max_steps and not terminated)
        reward = -1.0 if terminated else 0.0
        next_state = {
            "walls": walls,
            "food": frozenset(food),
            "power_pellets": frozenset(power),
            "frightened": int(new_frightened),
            "pacman_pos": tuple(new_pacman_pos),
            "ghost_positions": [tuple(gp) for gp in new_ghost_positions],
            "ghost_homes": [tuple(gp) for gp in ghost_homes],
            "ghost_dirs": [int(d) for d in new_ghost_dirs],
            "pacman_dir": pacman_dir,
            "steps": steps,
            "alive": alive,
            "won": won,
        }
        return next_state, reward, terminated, truncated, feedback

    def _oracle_action(
        self,
        state: Dict[str, Any],
        horizon: int,
        avoid_keys: Optional[set[tuple]] = None,
    ) -> Optional[int]:
        state = self._coerce_state(state)
        initial_food_count = len(state["food"])
        avoid_keys = set() if avoid_keys is None else set(avoid_keys)

        @lru_cache(maxsize=None)
        def search(key: tuple, depth: int) -> float:
            node = self._state_from_key(state["walls"], key)
            if not node["alive"]:
                return float("-inf")
            if depth == 0 or not node["food"] or node["steps"] >= self.max_steps:
                eaten = initial_food_count - len(node["food"])
                nearest_food = self._nearest_food_distance(node["walls"], node["pacman_pos"], node["food"])
                ghost_dists = [self._bfs_distance(node["walls"], node["pacman_pos"], gp) for gp in node["ghost_positions"]]
                reachable_ghosts = [d for d in ghost_dists if d >= 0]
                ghost_distance = min(reachable_ghosts) if reachable_ghosts else self.grid_size * self.grid_size
                remaining_food_penalty = sum(
                    max(0, int(self._bfs_distance(node["walls"], node["pacman_pos"], food_pos)))
                    for food_pos in node["food"]
                )
                return (
                    1000.0 * eaten
                    - float(nearest_food)
                    - 0.05 * float(remaining_food_penalty)
                    + 0.02 * float(max(0, ghost_distance))
                    + (0.1 * float(depth) if not node["food"] else 0.0)
                    - 0.001 * float(node["steps"])
                )
            best = float("-inf")
            for action_id in ACTION_ORDER:
                child, _, terminated, truncated, _ = self._transition_state(node, action_id)
                if terminated or truncated:
                    continue
                score = search(self._state_key(child), depth - 1)
                food_eaten_now = len(node["food"]) - len(child["food"])
                score += 0.05 * float(food_eaten_now) * float(depth)
                if tuple(child["pacman_pos"]) == tuple(node["pacman_pos"]):
                    score -= 5.0
                best = max(best, score)
            return best

        best_action: Optional[int] = None
        best_score = float("-inf")
        for action_id in ACTION_ORDER:
            child, _, terminated, truncated, _ = self._transition_state(state, action_id)
            if terminated or truncated:
                continue
            score = search(self._state_key(child), horizon - 1)
            if self._state_key(child) in avoid_keys:
                score -= 50.0
            if tuple(child["pacman_pos"]) == tuple(state["pacman_pos"]):
                score -= 5.0
            if score > best_score:
                best_score = score
                best_action = action_id
        return best_action

    def _next_ghost_pos(
        self,
        walls: np.ndarray,
        ghost_pos: tuple[int, int],
        ghost_dir: int,
        dest: tuple[int, int],
        flee: bool = False,
        no_reverse: bool = False,
    ) -> tuple[int, int]:
        """One deterministic ghost step relative to a destination tile `dest`. Chase = minimize BFS
        distance to `dest` (its personality target); flee = maximize BFS distance from `dest` (Pacman,
        when frightened). With no_reverse the ghost may not turn 180 from `ghost_dir` unless that is the
        only legal move (trapped). Tie-break by ACTION_ORDER (up, down, left, right)."""
        if ghost_pos == dest and not flee:
            return ghost_pos
        distances = self._bfs_distances(walls, dest)
        reverse = OPPOSITE.get(ghost_dir)

        def collect(allow_reverse: bool) -> list[tuple[int, tuple[int, int], int]]:
            opts: list[tuple[int, tuple[int, int], int]] = []
            for action_id in ACTION_ORDER:
                if no_reverse and not allow_reverse and action_id == reverse:
                    continue
                dr, dc = ACTION_TO_DELTA[action_id]
                candidate = (ghost_pos[0] + dr, ghost_pos[1] + dc)
                if not self._is_open(walls, candidate):
                    continue
                dist = int(distances[candidate])
                if dist < 0:
                    continue
                opts.append((action_id, candidate, dist))
            return opts

        opts = collect(allow_reverse=False)
        if not opts:  # trapped: reversing is the only legal move
            opts = collect(allow_reverse=True)
        if not opts:
            return ghost_pos
        if flee:
            best = max(opts, key=lambda o: (o[2], -o[0]))  # farthest cell, tie-break lowest action_id
        else:
            best = min(opts, key=lambda o: (o[2], o[0]))   # nearest cell, tie-break lowest action_id
        return best[1]

    def _project_ahead(self, walls: np.ndarray, pos: tuple[int, int], direction: int, ahead: int) -> tuple[int, int]:
        """The farthest OPEN cell up to `ahead` steps in `direction` from `pos` (so the target is always
        reachable for BFS). Falls back to `pos` if none is open."""
        dr, dc = ACTION_TO_DELTA.get(direction, (0, 0))
        for k in range(ahead, 0, -1):
            cand = (pos[0] + dr * k, pos[1] + dc * k)
            if self._is_open(walls, cand):
                return cand
        return pos

    def _ghost_target(
        self,
        walls: np.ndarray,
        persona: str,
        pacman_pos: tuple[int, int],
        pacman_dir: int,
        ghost_pos: tuple[int, int],
        blinky_pos: tuple[int, int],
    ) -> tuple[int, int]:
        """Deterministic per-ghost target tile (classic Blinky/Pinky/Inky/Clyde flavors). Every target
        is projected to an open, reachable cell; step-independent (pure function of positions + heading)."""
        if persona == "blinky":  # chase Pacman's current cell
            return pacman_pos
        if persona == "pinky":   # ambush PINKY_AHEAD cells ahead of Pacman's heading
            return self._project_ahead(walls, pacman_pos, pacman_dir, PINKY_AHEAD)
        if persona == "inky":    # reflect an INKY_AHEAD pivot through Blinky, then project to open
            pivot = self._project_ahead(walls, pacman_pos, pacman_dir, INKY_AHEAD)
            raw = (2 * pivot[0] - blinky_pos[0], 2 * pivot[1] - blinky_pos[1])
            return self._nearest_open(walls, raw, pacman_pos)
        if persona == "clyde":   # chase when far, retreat to home (bottom-left) corner when close
            if int(self._bfs_distance(walls, ghost_pos, pacman_pos)) > CLYDE_NEAR:
                return pacman_pos
            return self._nearest_open(walls, (self.grid_size - 2, 1), pacman_pos)
        return pacman_pos        # unknown persona -> safe default (direct chase)

    def _nearest_open(self, walls: np.ndarray, target: tuple[int, int], fallback: tuple[int, int]) -> tuple[int, int]:
        """Clamp `target` into bounds and return the closest open cell by Manhattan distance
        (deterministic lexicographic tie-break); else `fallback`. Keeps Inky/Clyde targets reachable
        when the raw geometric target lands on a wall/off-grid."""
        r = min(max(target[0], 0), self.grid_size - 1)
        c = min(max(target[1], 0), self.grid_size - 1)
        if self._is_open(walls, (r, c)):
            return (r, c)
        for radius in range(1, self.grid_size):
            best = None
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    cand = (r + dr, c + dc)
                    if self._is_open(walls, cand):
                        d = abs(dr) + abs(dc)  # Manhattan: prefer orthogonally-adjacent open cells
                        if best is None or (d, cand) < best[0]:
                            best = ((d, cand), cand)
            if best is not None:
                return best[1]
        return fallback

    def _bfs_distance(self, walls: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
        distances = self._bfs_distances(walls, start)
        return int(distances[goal])

    def _bfs_distances(self, walls: np.ndarray, start: tuple[int, int]) -> np.ndarray:
        wall_key = walls.tobytes()
        cache_key = (wall_key, tuple(start))
        cached = self._distance_cache.get(cache_key)
        if cached is not None:
            return cached
        distances = np.full((self.grid_size, self.grid_size), -1, dtype=np.int16)
        if not self._is_open(walls, start):
            return distances
        distances[start] = 0
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            for action_id in ACTION_ORDER:
                dr, dc = ACTION_TO_DELTA[action_id]
                nxt = (row + dr, col + dc)
                if not self._is_open(walls, nxt) or distances[nxt] >= 0:
                    continue
                distances[nxt] = distances[row, col] + 1
                queue.append(nxt)
        self._distance_cache[cache_key] = distances
        return distances

    def _nearest_food_distance(self, walls: np.ndarray, pacman_pos: tuple[int, int], food: frozenset[tuple[int, int]]) -> int:
        if not food:
            return 0
        distances = self._bfs_distances(walls, pacman_pos)
        reachable = [int(distances[pos]) for pos in food if distances[pos] >= 0]
        return min(reachable) if reachable else self.grid_size * self.grid_size

    def _is_open(self, walls: np.ndarray, pos: tuple[int, int]) -> bool:
        return self._in_bounds(pos) and not bool(walls[pos])

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        row, col = pos
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def _direction_between(self, src: tuple[int, int], dst: tuple[int, int], fallback: int) -> int:
        delta = (dst[0] - src[0], dst[1] - src[1])
        for action_id, action_delta in ACTION_TO_DELTA.items():
            if delta == action_delta:
                return action_id
        return fallback

    def _compute_stop_reward(self) -> float:
        return 1.0 if self.alive and not self.food else 0.0

    def _state_dict(self) -> Dict[str, Any]:
        return {
            "walls": self.walls.copy(),
            "food": frozenset(self.food),
            "power_pellets": frozenset(self.power_pellets),
            "frightened": int(self.frightened),
            "pacman_pos": tuple(self.pacman_pos),
            "ghost_positions": [tuple(gp) for gp in self.ghost_positions],
            "ghost_homes": [tuple(gp) for gp in self.ghost_homes],
            "ghost_dirs": [int(d) for d in self.ghost_dirs],
            "pacman_dir": int(self.pacman_dir),
            "steps": int(self.steps),
            "alive": bool(self.alive),
            "won": bool(self.won),
        }

    def _coerce_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ghost_positions, ghost_dirs = self._normalize_ghosts(state)
        raw_homes = state.get("ghost_homes")
        if raw_homes is None:
            ghost_homes = list(ghost_positions)  # legacy/fresh states: homes default to start cells
        else:
            ghost_homes = [tuple(int(v) for v in pos) for pos in raw_homes]
        if len(ghost_homes) < len(ghost_positions):
            ghost_homes = ghost_homes + list(ghost_positions[len(ghost_homes):])
        ghost_homes = ghost_homes[: len(ghost_positions)]
        return {
            "walls": np.array(state["walls"], dtype=bool),
            "food": frozenset(tuple(int(v) for v in pos) for pos in state.get("food", [])),
            "power_pellets": frozenset(tuple(int(v) for v in pos) for pos in state.get("power_pellets", [])),
            "frightened": int(state.get("frightened", 0)),
            "pacman_pos": tuple(int(v) for v in state["pacman_pos"]),
            "ghost_positions": ghost_positions,
            "ghost_homes": ghost_homes,
            "ghost_dirs": ghost_dirs,
            "pacman_dir": int(state.get("pacman_dir", RIGHT)),
            "steps": int(state.get("steps", 0)),
            "alive": bool(state.get("alive", True)),
            "won": bool(state.get("won", False)),
        }

    @staticmethod
    def _normalize_ghosts(state: Dict[str, Any]) -> tuple[list[tuple[int, int]], list[int]]:
        """Read ghosts from either the new (`ghost_positions`/`ghost_dirs`) or legacy
        (`ghost_pos`/`ghost_dir`) schema and return parallel lists of tuples/ints."""
        raw_pos = state.get("ghost_positions")
        if raw_pos is None:
            legacy = state.get("ghost_pos")
            raw_pos = [legacy] if legacy is not None else []
        positions = [tuple(int(v) for v in pos) for pos in raw_pos]
        raw_dirs = state.get("ghost_dirs")
        if raw_dirs is None:
            legacy_dir = state.get("ghost_dir", LEFT)
            raw_dirs = [legacy_dir] * len(positions)
        dirs = [int(d) for d in raw_dirs]
        if len(dirs) < len(positions):
            dirs = dirs + [LEFT] * (len(positions) - len(dirs))
        return positions, dirs[: len(positions)]

    def _json_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state = self._coerce_state(state)
        out = self.get_init_state()
        positions = state["ghost_positions"]
        dirs = state["ghost_dirs"]
        out.update(
            {
                "walls": state["walls"].astype(np.uint8).tolist(),
                "food": [list(pos) for pos in sorted(state["food"])],
                "power_pellets": [list(pos) for pos in sorted(state["power_pellets"])],
                "frightened": int(state["frightened"]),
                "pacman_pos": list(state["pacman_pos"]),
                "num_ghosts": len(positions),
                "ghost_positions": [list(pos) for pos in positions],
                "ghost_homes": [list(pos) for pos in state["ghost_homes"]],
                "ghost_dirs": [int(d) for d in dirs],
                "ghost_pos": list(positions[0]) if positions else None,
                "pacman_dir": int(state["pacman_dir"]),
                "ghost_dir": int(dirs[0]) if dirs else LEFT,
                "steps": int(state["steps"]),
                "alive": bool(state["alive"]),
                "won": bool(state["won"]),
            }
        )
        return out

    def _apply_state(self, state: Dict[str, Any]) -> None:
        state = self._coerce_state(state)
        self.walls = state["walls"].copy()
        wall_key = self.walls.tobytes()
        if self._wall_cache_key != wall_key:
            self._distance_cache.clear()
            self._wall_cache_key = wall_key
        self.food = frozenset(state["food"])
        self.power_pellets = frozenset(state["power_pellets"])
        self.frightened = int(state["frightened"])
        self.pacman_pos = tuple(state["pacman_pos"])
        self.ghost_positions = [tuple(gp) for gp in state["ghost_positions"]]
        self.ghost_homes = [tuple(gp) for gp in state["ghost_homes"]]
        self.ghost_dirs = [int(d) for d in state["ghost_dirs"]]
        self.num_ghosts = len(self.ghost_positions)
        self.pacman_dir = int(state["pacman_dir"])
        self.steps = int(state["steps"])
        self.alive = bool(state["alive"])
        self.won = bool(state["won"])

    def _state_key(self, state: Dict[str, Any]) -> tuple:
        positions, dirs = self._normalize_ghosts(state)
        homes = state.get("ghost_homes") or list(positions)
        return (
            tuple(state["pacman_pos"]),
            tuple(positions),
            tuple(sorted(state["food"])),
            int(state["pacman_dir"]),
            tuple(dirs),
            bool(state["alive"]),
            tuple(sorted(state.get("power_pellets", ()))),
            int(state.get("frightened", 0)),
            tuple(tuple(int(v) for v in h) for h in homes),
        )

    def _state_from_key(self, walls: np.ndarray, key: tuple) -> Dict[str, Any]:
        (pacman_pos, ghost_positions, food, pacman_dir, ghost_dirs, alive,
         power_pellets, frightened, ghost_homes) = key
        return {
            "walls": walls,
            "food": frozenset(tuple(pos) for pos in food),
            "power_pellets": frozenset(tuple(pos) for pos in power_pellets),
            "frightened": int(frightened),
            "pacman_pos": tuple(pacman_pos),
            "ghost_positions": [tuple(gp) for gp in ghost_positions],
            "ghost_homes": [tuple(gp) for gp in ghost_homes],
            "ghost_dirs": [int(d) for d in ghost_dirs],
            "pacman_dir": int(pacman_dir),
            "steps": 0,
            "alive": bool(alive),
            "won": bool(alive) and not food,
        }

    def _get_info(self) -> Dict[str, Any]:
        return self._info_from_state(self._state_dict(), self.env_feedback)

    def _info_from_state(self, state: Dict[str, Any], feedback: Optional[str]) -> Dict[str, Any]:
        state = self._coerce_state(state)
        positions = state["ghost_positions"]
        dirs = state["ghost_dirs"]
        ghost_dists = [int(self._bfs_distance(state["walls"], state["pacman_pos"], gp)) for gp in positions]
        # nearest ghost by BFS (ignore unreachable -1); fall back to -1 if none reachable.
        reachable = [d for d in ghost_dists if d >= 0]
        distance_to_ghost = min(reachable) if reachable else -1
        return {
            "backend": "custom_grid_pacman_python_assets",
            "steps": int(state["steps"]),
            "pacman_pos": list(state["pacman_pos"]),
            "num_ghosts": len(positions),
            "ghost_positions": [list(gp) for gp in positions],
            "ghost_dirs": [ACTION_NAMES.get(int(d), str(d)) for d in dirs],
            "ghost_pos": list(positions[0]) if positions else None,
            "pacman_dir": ACTION_NAMES.get(int(state["pacman_dir"]), str(state["pacman_dir"])),
            "ghost_dir": ACTION_NAMES.get(int(dirs[0]), str(dirs[0])) if dirs else None,
            "food": [list(pos) for pos in sorted(state["food"])],
            "food_remaining": int(len(state["food"])),
            "power_pellets": [list(pos) for pos in sorted(state["power_pellets"])],
            "power_pellets_remaining": int(len(state["power_pellets"])),
            "frightened": int(state["frightened"]),
            "alive": bool(state["alive"]),
            "is_success": bool(state["alive"] and not state["food"]),
            "won": bool(state["won"]),
            "distance_to_ghost": int(distance_to_ghost),
            "nearest_food_distance": int(self._nearest_food_distance(state["walls"], state["pacman_pos"], state["food"])),
            "env_feedback": feedback,
        }

    def _render_ansi(self, state: Dict[str, Any]) -> str:
        state = self._coerce_state(state)
        grid = np.full((self.grid_size, self.grid_size), " ", dtype="<U1")
        grid[state["walls"]] = "#"
        for row, col in state["food"]:
            grid[row, col] = "o"
        for row, col in state["power_pellets"]:
            grid[row, col] = "O"
        ghost_glyph = "g" if int(state["frightened"]) > 0 else "G"
        for gp in state["ghost_positions"]:
            grid[gp] = ghost_glyph
        pac = tuple(state["pacman_pos"])
        if not state["alive"] or any(pac == tuple(gp) for gp in state["ghost_positions"]):
            grid[pac] = "X"
        else:
            grid[pac] = "P"
        return "\n".join("".join(row) for row in grid)

    def _render_rgb(self, state: Dict[str, Any], render_size: Optional[tuple[int, int]] = None) -> np.ndarray:
        state = self._coerce_state(state)
        side = self.grid_size * self.tile_size
        image = Image.new("RGBA", (side, side), (0, 0, 0, 255))
        background = self._sprite("tiles/blank.gif")
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                image.paste(background, (col * self.tile_size, row * self.tile_size), background)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if state["walls"][row, col]:
                    self._paste_wall(image, state["walls"], row, col)
        for row, col in state["food"]:
            self._paste_food(image, row, col)
        for row, col in state["power_pellets"]:
            self._paste_power_pellet(image, row, col)
        frightened = int(state["frightened"]) > 0
        for i, (gp, gd) in enumerate(zip(state["ghost_positions"], state["ghost_dirs"])):
            persona = self.ghost_personas[i] if i < len(self.ghost_personas) else "blinky"
            self._paste_actor(image, "ghost", gp, int(gd), int(state["steps"]),
                              frightened=frightened, persona=persona)
        self._paste_actor(image, "pacman", state["pacman_pos"], int(state["pacman_dir"]), int(state["steps"]))
        rgb = image.convert("RGB")
        if render_size is not None and rgb.size != render_size:
            rgb = rgb.resize(render_size, Image.Resampling.BICUBIC)
        return np.array(rgb, dtype=np.uint8)

    @staticmethod
    @lru_cache(maxsize=None)
    def _asset(name: str) -> Image.Image:
        with Image.open(ASSET_DIR / name) as source:
            return source.convert("RGBA")

    def _sprite(self, name: str) -> Image.Image:
        sprite = self._asset(name)
        if sprite.size != (self.tile_size, self.tile_size):
            sprite = sprite.resize((self.tile_size, self.tile_size), Image.Resampling.NEAREST)
        return sprite

    def _paste_sprite(self, image: Image.Image, name: str, row: int, col: int) -> None:
        sprite = self._sprite(name)
        image.paste(sprite, (col * self.tile_size, row * self.tile_size), sprite)

    def _paste_food(self, image: Image.Image, row: int, col: int) -> None:
        draw = ImageDraw.Draw(image)
        radius = max(3, self.tile_size // 5)
        cx = col * self.tile_size + self.tile_size // 2
        cy = row * self.tile_size + self.tile_size // 2
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(245, 245, 245, 255))

    def _paste_power_pellet(self, image: Image.Image, row: int, col: int) -> None:
        """Larger, distinct pickup (bright white disc with a cyan ring) so it reads clearly vs food."""
        draw = ImageDraw.Draw(image)
        radius = max(5, self.tile_size // 3)
        cx = col * self.tile_size + self.tile_size // 2
        cy = row * self.tile_size + self.tile_size // 2
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 255, 255, 255))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     outline=(120, 220, 255, 255), width=max(2, self.tile_size // 12))

    def _paste_actor(self, image: Image.Image, actor: str, pos: tuple[int, int], direction: int, steps: int,
                     frightened: bool = False, persona: str = "blinky") -> None:
        if actor == "pacman":
            direction_prefix = PACMAN_SPRITE_PREFIX.get(direction, "r")
            frame = 1 + (steps % 8)
            name = f"sprite/pacman-{direction_prefix} {frame}.gif"
            self._paste_sprite(image, name, pos[0], pos[1])
        else:
            frame = 1 + (steps % 2)
            name = f"sprite/ghost {frame}.gif"
            sprite = self._sprite(name)
            if frightened:
                sprite = self._tint_color(sprite, (40, 60, 230))  # frightened blue (all personas)
            else:
                color = GHOST_COLORS.get(persona)  # "blinky" -> None -> default red sprite (no tint)
                if color is not None:
                    sprite = self._tint_color(sprite, color)
            image.paste(sprite, (pos[1] * self.tile_size, pos[0] * self.tile_size), sprite)

    @staticmethod
    def _tint_color(sprite: Image.Image, color: tuple[int, int, int]) -> Image.Image:
        """Recolor a ghost sprite's opaque pixels toward `color`, preserving shape/alpha and some
        original luminance for shading."""
        arr = np.array(sprite.convert("RGBA"), dtype=np.uint8)
        opaque = arr[..., 3] > 0
        tint = np.array(color, dtype=np.uint16)
        rgb = arr[..., :3].astype(np.uint16)
        arr[..., :3][opaque] = ((rgb[opaque] // 4) + (tint * 3) // 4).astype(np.uint8)
        return Image.fromarray(arr, "RGBA")

    def _paste_wall(self, image: Image.Image, walls: np.ndarray, row: int, col: int) -> None:
        up = row > 0 and bool(walls[row - 1, col])
        down = row < self.grid_size - 1 and bool(walls[row + 1, col])
        left = col > 0 and bool(walls[row, col - 1])
        right = col < self.grid_size - 1 and bool(walls[row, col + 1])
        name: str
        if up and down and left and right:
            name = "wall-x.gif"
        elif left and right and down and not up:
            name = "wall-t-bottom.gif"
        elif left and right and up and not down:
            name = "wall-t-top.gif"
        elif up and down and right and not left:
            name = "wall-t-right.gif"
        elif up and down and left and not right:
            name = "wall-t-left.gif"
        elif left and right and not (up or down):
            name = "wall-straight-horiz.gif"
        elif up and down and not (left or right):
            name = "wall-straight-vert.gif"
        elif right and down and not left and not up:
            name = "wall-corner-ul.gif"
        elif left and down and not right and not up:
            name = "wall-corner-ur.gif"
        elif right and up and not left and not down:
            name = "wall-corner-ll.gif"
        elif left and up and not right and not down:
            name = "wall-corner-lr.gif"
        elif up and not (down or left or right):
            name = "wall-end-b.gif"
        elif down and not (up or left or right):
            name = "wall-end-t.gif"
        elif left and not (up or down or right):
            name = "wall-end-r.gif"
        elif right and not (up or down or left):
            name = "wall-end-l.gif"
        else:
            name = "wall-nub.gif"
        self._paste_sprite(image, f"tiles/{name}", row, col)
