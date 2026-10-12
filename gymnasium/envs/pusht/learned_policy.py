from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class PushTLearnedPolicy(Protocol):
    @property
    def sequence_length(self) -> int: ...

    @property
    def observation_kind(self) -> str: ...

    @property
    def observation_dim(self) -> int: ...

    def action(self, observation_history: Any, *, first_step: bool) -> np.ndarray: ...


def load_pusht_policy(
    strategy: str,
    *,
    checkpoint_root: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_env_var: str | None = None,
    num_inference_steps: int | None = None,
    device: str | None = None,
) -> PushTLearnedPolicy:
    del num_inference_steps
    strategy_name = str(strategy).strip().lower()
    if strategy_name == "ppo":
        from gymnasium.envs.pusht.ppo_policy import PushTPPOPolicy

        return PushTPPOPolicy.load(
            checkpoint_root=checkpoint_root,
            checkpoint_path=checkpoint_path,
            checkpoint_env_var="PUSHT_PPO_ROOT" if checkpoint_env_var is None else checkpoint_env_var,
            device=device,
        )
    raise ValueError(f"Unsupported learned PushT policy strategy: {strategy}")
