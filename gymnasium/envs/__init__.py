"""Register the task environments shipped with Agima."""

from gymnasium.envs.registration import make, pprint_registry, register, registry, spec

# Sokoban is registered by registration._register_visgym_local_envs().
# Keep the existing task kwargs and wrapper options unchanged.

register(
    id="maze_2d/easy",
    entry_point="gymnasium.envs.maze_2d.maze_2d:Maze2DEnv",
    kwargs={"maze_width": 11, "maze_height": 11, "target_in_center": False, "render_size": 256},
    disable_env_checker=True,
)

register(
    id="maze_2d/hard",
    entry_point="gymnasium.envs.maze_2d.maze_2d:Maze2DEnv",
    kwargs={"maze_width": 15, "maze_height": 15, "target_in_center": False, "render_size": 256},
    disable_env_checker=True,
)

register(
    id="pacman_2d/easy",
    entry_point="gymnasium.envs.pacman_2d.pacman_2d:Pacman2DEnv",
    kwargs={"difficulty": "easy", "grid_size": 9, "food_count": 4, "horizon": 8},
    disable_env_checker=True,
    order_enforce=False,
)

register(
    id="pacman_2d/hard",
    entry_point="gymnasium.envs.pacman_2d.pacman_2d:Pacman2DEnv",
    kwargs={"difficulty": "hard", "grid_size": 11, "food_count": 8, "horizon": 8},
    disable_env_checker=True,
    order_enforce=False,
)

register(
    id="pusht/easy",
    entry_point="gymnasium.envs.pusht.pusht:PushTEnv",
    disable_env_checker=True,
)
