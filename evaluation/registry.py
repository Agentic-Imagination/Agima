# Modified for Agima: package imports and portable resource configuration.
"""Default metadata for the four released task families.

Environment parameters and initial states come from the supplied episode records.
No external image or video datasets are needed for these procedural environments.
"""

ENV_REGISTRY = {
    env_id: {
        "env_kwargs": {},
        "seed_key": None,
        "gt_strategy": None,
        "gt_kwargs": {},
        "extra_state": None,
    }
    for env_id in (
        "sokoban/easy", "sokoban/hard",
        "pusht/easy", "pusht/hard",
        "maze_2d/easy", "maze_2d/hard",
        "pacman_2d/easy", "pacman_2d/hard",
    )
}
