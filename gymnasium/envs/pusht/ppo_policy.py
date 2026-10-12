from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal


WINDOW_SIZE = 512.0
PUSHT_PPO_POSE_OBS_KIND = "pusht_ppo_lowdim"
PUSHT_PPO_GEOM_DYN_OBS_KIND = "pusht_ppo_geomdyn_v1"
PUSHT_PPO_OFFICIAL_LOWDIM_OBS_KIND = "pusht_ppo_official_lowdim"
PUSHT_PPO_POSE_OBS_DIM = 10
PUSHT_PPO_GEOM_DYN_OBS_DIM = 23
PUSHT_PPO_OFFICIAL_NUM_KEYPOINTS = 9
PUSHT_PPO_OFFICIAL_KEYPOINT_DIM = PUSHT_PPO_OFFICIAL_NUM_KEYPOINTS * 2
PUSHT_PPO_OFFICIAL_LOWDIM_OBS_DIM = PUSHT_PPO_OFFICIAL_KEYPOINT_DIM + 2
PUSHT_PPO_OBS_DIM = PUSHT_PPO_POSE_OBS_DIM
PUSHT_PPO_ACTION_DIM = 2
PUSHT_PPO_CHECKPOINT_FORMAT = "visgym_pusht_ppo_torch_v1"
PUSHT_PPO_ABSOLUTE_ACTION_CONTROL = "absolute"
PUSHT_PPO_DELTA_ACTION_CONTROL = "delta"
PUSHT_PPO_DELTA_INTERPOLATE_ACTION_CONTROL = "delta_interpolate"
PUSHT_PPO_ACTION_CONTROL_CHOICES = (
    PUSHT_PPO_ABSOLUTE_ACTION_CONTROL,
    PUSHT_PPO_DELTA_ACTION_CONTROL,
    PUSHT_PPO_DELTA_INTERPOLATE_ACTION_CONTROL,
)
PUSHT_ACTION_CODEC_PIXEL4 = "pixel4"
PUSHT_ACTION_CODEC_INT1 = "int1"
PUSHT_ACTION_CODEC_NORM2 = "norm2"
PUSHT_ACTION_CODEC_NORM4 = "norm4"
PUSHT_ACTION_CODEC_CHOICES = (
    PUSHT_ACTION_CODEC_PIXEL4,
    PUSHT_ACTION_CODEC_INT1,
    PUSHT_ACTION_CODEC_NORM2,
    PUSHT_ACTION_CODEC_NORM4,
)


def parse_net_arch(spec: str) -> tuple[int, ...]:
    values = [chunk.strip() for chunk in str(spec).split(",") if chunk.strip()]
    if not values:
        raise ValueError("`--net-arch` must contain at least one hidden layer size.")
    return tuple(int(value) for value in values)


def build_pusht_ppo_observation(
    pos_agent: np.ndarray | Sequence[float],
    block_pose: np.ndarray | Sequence[float],
    goal_pose: np.ndarray | Sequence[float],
) -> np.ndarray:
    pos_agent = np.asarray(pos_agent, dtype=np.float32).reshape(2)
    block_pose = np.asarray(block_pose, dtype=np.float32).reshape(3)
    goal_pose = np.asarray(goal_pose, dtype=np.float32).reshape(3)
    return np.asarray(
        [
            pos_agent[0] / WINDOW_SIZE,
            pos_agent[1] / WINDOW_SIZE,
            block_pose[0] / WINDOW_SIZE,
            block_pose[1] / WINDOW_SIZE,
            math.cos(float(block_pose[2])),
            math.sin(float(block_pose[2])),
            goal_pose[0] / WINDOW_SIZE,
            goal_pose[1] / WINDOW_SIZE,
            math.cos(float(goal_pose[2])),
            math.sin(float(goal_pose[2])),
        ],
        dtype=np.float32,
    )


def build_pusht_ppo_geom_dyn_observation(
    pos_agent: np.ndarray | Sequence[float],
    vel_agent: np.ndarray | Sequence[float],
    block_keypoints: np.ndarray | Sequence[float],
    block_vel: np.ndarray | Sequence[float],
    block_angvel: float,
) -> np.ndarray:
    pos_agent = np.asarray(pos_agent, dtype=np.float32).reshape(2)
    vel_agent = np.asarray(vel_agent, dtype=np.float32).reshape(2)
    block_keypoints = np.asarray(block_keypoints, dtype=np.float32).reshape(-1)
    if block_keypoints.shape[0] != 16:
        raise ValueError(f"Expected 16 block keypoint values, got {block_keypoints.shape[0]}.")
    block_vel = np.asarray(block_vel, dtype=np.float32).reshape(2)
    return np.asarray(
        [
            pos_agent[0] / WINDOW_SIZE,
            pos_agent[1] / WINDOW_SIZE,
            *list(block_keypoints / WINDOW_SIZE),
            math.tanh(float(vel_agent[0]) / WINDOW_SIZE),
            math.tanh(float(vel_agent[1]) / WINDOW_SIZE),
            math.tanh(float(block_vel[0]) / WINDOW_SIZE),
            math.tanh(float(block_vel[1]) / WINDOW_SIZE),
            math.tanh(float(block_angvel) / math.pi),
        ],
        dtype=np.float32,
    )


def build_pusht_ppo_official_lowdim_observation(
    pos_agent: np.ndarray | Sequence[float],
    official_keypoints: np.ndarray | Sequence[float],
) -> np.ndarray:
    pos_agent = np.asarray(pos_agent, dtype=np.float32).reshape(2)
    keypoints = np.asarray(official_keypoints, dtype=np.float32).reshape(-1)
    if keypoints.shape[0] != PUSHT_PPO_OFFICIAL_KEYPOINT_DIM:
        raise ValueError(
            f"Expected {PUSHT_PPO_OFFICIAL_KEYPOINT_DIM} official keypoint values, got {keypoints.shape[0]}."
        )
    return np.asarray(
        [
            *list(keypoints / WINDOW_SIZE),
            pos_agent[0] / WINDOW_SIZE,
            pos_agent[1] / WINDOW_SIZE,
        ],
        dtype=np.float32,
    )


def build_pusht_ppo_observation_for_kind(
    observation_kind: str,
    *,
    pos_agent: np.ndarray | Sequence[float],
    block_pose: np.ndarray | Sequence[float] | None = None,
    goal_pose: np.ndarray | Sequence[float] | None = None,
    vel_agent: np.ndarray | Sequence[float] | None = None,
    block_keypoints: np.ndarray | Sequence[float] | None = None,
    official_keypoints: np.ndarray | Sequence[float] | None = None,
    block_vel: np.ndarray | Sequence[float] | None = None,
    block_angvel: float | None = None,
) -> np.ndarray:
    kind = str(observation_kind).strip().lower()
    if kind == PUSHT_PPO_OFFICIAL_LOWDIM_OBS_KIND:
        if official_keypoints is None:
            raise ValueError(f"`{PUSHT_PPO_OFFICIAL_LOWDIM_OBS_KIND}` observation requires official keypoints.")
        return build_pusht_ppo_official_lowdim_observation(
            pos_agent=pos_agent,
            official_keypoints=official_keypoints,
        )
    if kind == PUSHT_PPO_GEOM_DYN_OBS_KIND:
        if vel_agent is None or block_keypoints is None or block_vel is None or block_angvel is None:
            raise ValueError(f"`{PUSHT_PPO_GEOM_DYN_OBS_KIND}` observation requires geometry and dynamics inputs.")
        return build_pusht_ppo_geom_dyn_observation(
            pos_agent=pos_agent,
            vel_agent=vel_agent,
            block_keypoints=block_keypoints,
            block_vel=block_vel,
            block_angvel=float(block_angvel),
        )
    if kind not in {"", PUSHT_PPO_POSE_OBS_KIND}:
        raise ValueError(f"Unsupported PushT PPO observation kind: {observation_kind!r}")
    if block_pose is None or goal_pose is None:
        raise ValueError(f"`{PUSHT_PPO_POSE_OBS_KIND}` observation requires block and goal pose inputs.")
    return build_pusht_ppo_observation(
        pos_agent=pos_agent,
        block_pose=block_pose,
        goal_pose=goal_pose,
    )


def observation_dim_for_kind(observation_kind: str) -> int:
    kind = str(observation_kind).strip().lower()
    if kind == PUSHT_PPO_OFFICIAL_LOWDIM_OBS_KIND:
        return PUSHT_PPO_OFFICIAL_LOWDIM_OBS_DIM
    if kind == PUSHT_PPO_GEOM_DYN_OBS_KIND:
        return PUSHT_PPO_GEOM_DYN_OBS_DIM
    return PUSHT_PPO_POSE_OBS_DIM


def normalize_action_control(action_control: str | None) -> str:
    value = PUSHT_PPO_ABSOLUTE_ACTION_CONTROL if action_control is None else str(action_control).strip().lower()
    if value in {"abs", "absolute_target", "target"}:
        value = PUSHT_PPO_ABSOLUTE_ACTION_CONTROL
    if value in {"relative", "delta_xy", "ee_delta"}:
        value = PUSHT_PPO_DELTA_ACTION_CONTROL
    if value in {"delta_interp", "interpolate", "interpolated_delta", "ee_delta_interpolate"}:
        value = PUSHT_PPO_DELTA_INTERPOLATE_ACTION_CONTROL
    if value not in PUSHT_PPO_ACTION_CONTROL_CHOICES:
        raise ValueError(f"Unsupported PushT PPO action control mode: {action_control!r}")
    return value


def layer_init(layer: nn.Module, *, std: float = 1.0, bias_const: float = 0.0) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    return layer


class PushTPPOActorCritic(nn.Module):
    def __init__(self, *, observation_dim: int, action_dim: int, net_arch: Sequence[int], log_std_init: float):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.net_arch = tuple(int(value) for value in net_arch)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init), dtype=torch.float32))

        self.actor_body = self._build_body(self.observation_dim, self.net_arch)
        self.critic_body = self._build_body(self.observation_dim, self.net_arch)
        last_dim = self.net_arch[-1] if self.net_arch else self.observation_dim
        self.actor_mean = layer_init(nn.Linear(last_dim, self.action_dim), std=0.01)
        self.critic_value = layer_init(nn.Linear(last_dim, 1), std=1.0)

    @staticmethod
    def _build_body(input_dim: int, net_arch: Sequence[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for hidden_dim in net_arch:
            layers.append(layer_init(nn.Linear(current_dim, int(hidden_dim)), std=math.sqrt(2.0)))
            layers.append(nn.Tanh())
            current_dim = int(hidden_dim)
        return nn.Sequential(*layers)

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic_value(self.critic_body(observation))

    def get_action_dist(self, observation: torch.Tensor) -> Normal:
        mean = self.actor_mean(self.actor_body(observation))
        log_std = self.log_std.expand_as(mean)
        return Normal(mean, log_std.exp())

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_action_dist(observation)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.get_value(observation).squeeze(-1)
        return action, log_prob, entropy, value

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.get_action_dist(observation).mean, -1.0, 1.0)

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.get_action_and_value(observation, action)


def clip_action_xy(action_xy: np.ndarray | Sequence[float]) -> np.ndarray:
    action = np.asarray(action_xy, dtype=np.float32).reshape(-1)
    if action.shape[0] < PUSHT_PPO_ACTION_DIM:
        raise ValueError(f"Expected at least {PUSHT_PPO_ACTION_DIM} action values, got {action.shape[0]}.")
    return np.clip(action[:PUSHT_PPO_ACTION_DIM], 0.0, WINDOW_SIZE)


def denormalize_train_action(train_action: np.ndarray | Sequence[float]) -> np.ndarray:
    action = np.asarray(train_action, dtype=np.float32).reshape(-1)
    if action.shape[0] < PUSHT_PPO_ACTION_DIM:
        raise ValueError(f"Expected at least {PUSHT_PPO_ACTION_DIM} action values, got {action.shape[0]}.")
    normalized = np.clip(action[:PUSHT_PPO_ACTION_DIM], -1.0, 1.0)
    return clip_action_xy(((normalized + 1.0) * 0.5) * WINDOW_SIZE)


def normalize_action_codec(action_codec: str | None) -> str:
    value = PUSHT_ACTION_CODEC_PIXEL4 if action_codec is None else str(action_codec).strip().lower()
    if value in {"", "legacy", "pixel", "pixels", "px", "px4", "float4"}:
        value = PUSHT_ACTION_CODEC_PIXEL4
    if value in {"integer1", "int", "pixel1", "px1"}:
        value = PUSHT_ACTION_CODEC_INT1
    if value in {"normalized2", "normalised2"}:
        value = PUSHT_ACTION_CODEC_NORM2
    if value in {"normalized", "normalised", "normalized4", "normalised4", "norm"}:
        value = PUSHT_ACTION_CODEC_NORM4
    if value not in PUSHT_ACTION_CODEC_CHOICES:
        raise ValueError(f"Unsupported PushT action codec: {action_codec!r}")
    return value


def quantize_action_target_xy(
    target_xy: np.ndarray | Sequence[float],
    *,
    action_codec: str | None = None,
) -> np.ndarray:
    codec = normalize_action_codec(action_codec)
    target = clip_action_xy(target_xy)
    if codec == PUSHT_ACTION_CODEC_INT1:
        return np.asarray([round(float(target[0]), 1), round(float(target[1]), 1)], dtype=np.float32)
    if codec == PUSHT_ACTION_CODEC_NORM2:
        normalized = np.clip(target / WINDOW_SIZE, 0.0, 1.0)
        encoded = np.asarray([round(float(normalized[0]), 2), round(float(normalized[1]), 2)], dtype=np.float32)
        return clip_action_xy(encoded * WINDOW_SIZE)
    if codec == PUSHT_ACTION_CODEC_NORM4:
        normalized = np.clip(target / WINDOW_SIZE, 0.0, 1.0)
        encoded = np.asarray([round(float(normalized[0]), 4), round(float(normalized[1]), 4)], dtype=np.float32)
        return clip_action_xy(encoded * WINDOW_SIZE)
    return np.asarray([round(float(target[0]), 4), round(float(target[1]), 4)], dtype=np.float32)


def format_move_action(
    target_xy: np.ndarray | Sequence[float],
    *,
    action_codec: str | None = None,
) -> str:
    codec = normalize_action_codec(action_codec)
    executed_xy = quantize_action_target_xy(target_xy, action_codec=codec)
    if codec == PUSHT_ACTION_CODEC_INT1:
        return f"('move', ({float(executed_xy[0]):.1f}, {float(executed_xy[1]):.1f}))"
    if codec == PUSHT_ACTION_CODEC_NORM2:
        normalized = np.clip(executed_xy / WINDOW_SIZE, 0.0, 1.0)
        return f"('move', ({float(normalized[0]):.2f}, {float(normalized[1]):.2f}))"
    if codec == PUSHT_ACTION_CODEC_NORM4:
        normalized = np.clip(executed_xy / WINDOW_SIZE, 0.0, 1.0)
        return f"('move', ({float(normalized[0]):.4f}, {float(normalized[1]):.4f}))"
    return f"('move', ({float(executed_xy[0]):.4f}, {float(executed_xy[1]):.4f}))"


def decode_move_payload_xy(
    payload: Sequence[float],
    *,
    action_codec: str | None = None,
) -> np.ndarray:
    codec = normalize_action_codec(action_codec)
    values = np.asarray(payload, dtype=np.float32).reshape(-1)
    if values.shape[0] != PUSHT_PPO_ACTION_DIM:
        raise ValueError(f"Expected exactly {PUSHT_PPO_ACTION_DIM} move payload values, got {values.shape[0]}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Move payload values must be finite.")
    if codec in {PUSHT_ACTION_CODEC_NORM2, PUSHT_ACTION_CODEC_NORM4}:
        if bool(np.any(values < 0.0)) or bool(np.any(values > 1.0)):
            raise ValueError("Normalized PushT move payload values must be in [0, 1].")
        return quantize_action_target_xy(values * WINDOW_SIZE, action_codec=codec)
    if bool(np.any(values < 0.0)) or bool(np.any(values > WINDOW_SIZE)):
        raise ValueError(f"PushT move payload values must be in [0, {WINDOW_SIZE:g}].")
    return quantize_action_target_xy(values, action_codec=codec)


def action_codec_prompt_line(action_codec: str | None = None) -> str:
    codec = normalize_action_codec(action_codec)
    if codec == PUSHT_ACTION_CODEC_INT1:
        return (
            "1. `move`: Move the blue pusher toward a target coordinate. Format: `('move', (x, y))`, "
            "where x and y are 512x512 world/image coordinates with exactly one decimal place, e.g. `225.1`."
        )
    if codec == PUSHT_ACTION_CODEC_NORM2:
        return (
            "1. `move`: Move the blue pusher toward a target coordinate. Format: `('move', (x, y))`, "
            "where x and y are normalized coordinates in [0, 1] with exactly two decimal places, e.g. `0.78`."
        )
    if codec == PUSHT_ACTION_CODEC_NORM4:
        return (
            "1. `move`: Move the blue pusher toward a target coordinate. Format: `('move', (x, y))`, "
            "where x and y are normalized coordinates in [0, 1] with exactly four decimal places, e.g. `0.7801`."
        )
    return (
        "1. `move`: Move the blue pusher toward a target coordinate. Format: `('move', (x, y))`, "
        "where x and y are coordinates in the 512x512 world/image coordinate system."
    )


def train_action_to_target_xy(
    train_action: np.ndarray | Sequence[float],
    *,
    agent_xy: np.ndarray | Sequence[float] | None = None,
    action_control: str | None = None,
    delta_action_scale: float = 64.0,
) -> np.ndarray:
    mode = normalize_action_control(action_control)
    normalized = np.asarray(train_action, dtype=np.float32).reshape(-1)
    if normalized.shape[0] < PUSHT_PPO_ACTION_DIM:
        raise ValueError(f"Expected at least {PUSHT_PPO_ACTION_DIM} action values, got {normalized.shape[0]}.")
    normalized = np.clip(normalized[:PUSHT_PPO_ACTION_DIM], -1.0, 1.0)
    if mode == PUSHT_PPO_ABSOLUTE_ACTION_CONTROL:
        return denormalize_train_action(normalized)
    if agent_xy is None:
        raise ValueError("Delta PushT PPO action control requires the current agent xy position.")
    agent = np.asarray(agent_xy, dtype=np.float32).reshape(2)
    return clip_action_xy(agent + normalized * float(delta_action_scale))


def load_pusht_ppo_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[PushTPPOActorCritic, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser()
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    if payload.get("format") != PUSHT_PPO_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported PushT PPO checkpoint format {payload.get('format')!r} at {checkpoint_path}"
        )
    model_config = dict(payload.get("model_config", {}))
    model = PushTPPOActorCritic(
        observation_dim=int(model_config.get("observation_dim", PUSHT_PPO_OBS_DIM)),
        action_dim=int(model_config.get("action_dim", PUSHT_PPO_ACTION_DIM)),
        net_arch=tuple(model_config.get("net_arch", (256, 256, 256))),
        log_std_init=float(model_config.get("log_std_init", -0.5)),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def load_policy_metadata(root: str | Path) -> dict[str, Any]:
    metadata_path = Path(root).expanduser() / "policy_config.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_policy_root(
    checkpoint_root: str | os.PathLike[str] | None = None,
    checkpoint_env_var: str | None = None,
) -> Path | None:
    if checkpoint_root:
        return Path(checkpoint_root).expanduser()
    env_var = "PUSHT_PPO_ROOT" if checkpoint_env_var is None else str(checkpoint_env_var)
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser()
    default_root = Path("visgym_training") / "ppo_pusht" / "runs" / "pusht_ppo_lowdim_1hz"
    if default_root.exists():
        return default_root
    return None


def resolve_checkpoint_path(
    root: str | Path,
    *,
    explicit_checkpoint_path: str | os.PathLike[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    if explicit_checkpoint_path:
        path = Path(explicit_checkpoint_path).expanduser()
        if path.is_dir():
            root = path
        else:
            return path
    root_path = Path(root).expanduser()
    meta = metadata or {}
    for key in ("latest_checkpoint", "checkpoint_path"):
        value = meta.get(key)
        if not value:
            continue
        path = Path(value).expanduser()
        if path.exists():
            return path
        rooted = root_path / path
        if rooted.exists():
            return rooted
    for candidate in ("best_model.pt", "latest_model.pt", "final_model.pt", "model.pt"):
        path = root_path / candidate
        if path.exists():
            return path
    return None


@dataclass
class _LoadedPPO:
    model: PushTPPOActorCritic
    checkpoint_path: Path
    metadata: dict[str, Any]
    device: str


class PushTPPOPolicy:
    _CACHE: ClassVar[dict[tuple[str, str], _LoadedPPO]] = {}

    def __init__(self, *, loaded: _LoadedPPO):
        self._model = loaded.model
        self._checkpoint_path = loaded.checkpoint_path
        self._metadata = dict(loaded.metadata)
        self._device = str(loaded.device)
        self.sequence_length = int(self._metadata.get("sequence_length", 1))
        self.observation_kind = str(self._metadata.get("observation_kind", PUSHT_PPO_POSE_OBS_KIND))
        self.observation_dim = int(
            self._metadata.get("observation_dim", observation_dim_for_kind(self.observation_kind))
        )
        self.control_hz = int(self._metadata.get("control_hz", 1))
        self.deterministic = bool(self._metadata.get("deterministic", True))
        self.action_control = normalize_action_control(
            self._metadata.get("action_control", PUSHT_PPO_ABSOLUTE_ACTION_CONTROL)
        )
        self.action_codec = normalize_action_codec(self._metadata.get("action_codec", PUSHT_ACTION_CODEC_PIXEL4))
        self.delta_action_scale = float(self._metadata.get("delta_action_scale", 64.0))
        self.init_distribution = str(self._metadata.get("init_distribution", "visgym_default"))
        self.terminate_on_success = bool(self._metadata.get("terminate_on_success", True))
        self.max_episode_steps = int(self._metadata.get("max_episode_steps", 100))
        self.success_threshold = float(self._metadata.get("success_threshold", 0.95))

    @classmethod
    def load(
        cls,
        *,
        checkpoint_root: str | None = None,
        checkpoint_path: str | None = None,
        checkpoint_env_var: str | None = None,
        device: str | None = None,
    ) -> "PushTPPOPolicy":
        root = resolve_policy_root(checkpoint_root=checkpoint_root, checkpoint_env_var=checkpoint_env_var)
        if checkpoint_path is not None:
            checkpoint_candidate = Path(checkpoint_path).expanduser()
            root = checkpoint_candidate.parent if checkpoint_candidate.is_file() else checkpoint_candidate
        if root is None:
            raise FileNotFoundError(
                "Could not resolve a PushT PPO checkpoint root. Pass `checkpoint_root` or set `PUSHT_PPO_ROOT`."
            )
        root = Path(root).expanduser()
        metadata = load_policy_metadata(root)
        checkpoint = resolve_checkpoint_path(root, explicit_checkpoint_path=checkpoint_path, metadata=metadata)
        if checkpoint is None:
            raise FileNotFoundError(f"Could not find a PushT PPO checkpoint under {root}.")

        chosen_device = "cpu" if device is None else str(device)
        cache_key = (str(checkpoint.resolve()), chosen_device)
        loaded = cls._CACHE.get(cache_key)
        if loaded is None:
            model, payload = load_pusht_ppo_checkpoint(checkpoint, device=chosen_device)
            merged_metadata = dict(metadata)
            train_args = dict(payload.get("train_args", {}))
            merged_metadata.setdefault("sequence_length", 1)
            default_kind = str(train_args.get("observation_kind", PUSHT_PPO_POSE_OBS_KIND))
            merged_metadata.setdefault("observation_kind", default_kind)
            merged_metadata.setdefault(
                "observation_dim",
                int(
                    payload.get("model_config", {}).get(
                        "observation_dim",
                        observation_dim_for_kind(merged_metadata["observation_kind"]),
                    )
                ),
            )
            merged_metadata.setdefault("control_hz", int(train_args.get("control_hz", 1)))
            merged_metadata.setdefault(
                "action_control",
                str(train_args.get("action_control", PUSHT_PPO_ABSOLUTE_ACTION_CONTROL)),
            )
            merged_metadata.setdefault("delta_action_scale", float(train_args.get("delta_action_scale", 64.0)))
            merged_metadata.setdefault("init_distribution", str(train_args.get("init_distribution", "visgym_default")))
            merged_metadata.setdefault("terminate_on_success", bool(train_args.get("terminate_on_success", True)))
            merged_metadata.setdefault("max_episode_steps", int(train_args.get("max_episode_steps", 100)))
            merged_metadata.setdefault("success_threshold", float(train_args.get("success_threshold", 0.95)))
            merged_metadata.setdefault("deterministic", True)
            merged_metadata.setdefault("action_codec", str(train_args.get("action_codec", PUSHT_ACTION_CODEC_PIXEL4)))
            loaded = _LoadedPPO(
                model=model,
                checkpoint_path=Path(checkpoint).expanduser(),
                metadata=merged_metadata,
                device=chosen_device,
            )
            cls._CACHE[cache_key] = loaded
        return cls(loaded=loaded)

    def _coerce_observation(self, observation: Any) -> np.ndarray:
        if isinstance(observation, list):
            if not observation:
                raise ValueError("Observation history is empty.")
            observation = observation[-1]
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.observation_dim:
            raise ValueError(f"Expected observation dim {self.observation_dim}, got {obs.shape[0]}.")
        return obs

    def _infer_agent_xy(self, obs: np.ndarray) -> np.ndarray | None:
        if self.observation_kind == PUSHT_PPO_OFFICIAL_LOWDIM_OBS_KIND:
            return obs[-2:] * WINDOW_SIZE
        if self.observation_kind in {PUSHT_PPO_POSE_OBS_KIND, PUSHT_PPO_GEOM_DYN_OBS_KIND}:
            return obs[:2] * WINDOW_SIZE
        return None

    def train_action(self, observation_history: Any) -> np.ndarray:
        obs = self._coerce_observation(observation_history)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self._device).reshape(1, -1)
        with torch.no_grad():
            train_action = self._model.deterministic_action(obs_tensor).detach().cpu().numpy().reshape(-1)
        return np.clip(train_action[:PUSHT_PPO_ACTION_DIM], -1.0, 1.0).astype(np.float32, copy=False)

    def action(
        self,
        observation_history: Any,
        *,
        first_step: bool,
        agent_xy: np.ndarray | Sequence[float] | None = None,
    ) -> np.ndarray:
        del first_step
        train_action = self.train_action(observation_history)
        obs = self._coerce_observation(observation_history)
        target_xy = train_action_to_target_xy(
            train_action,
            agent_xy=self._infer_agent_xy(obs) if agent_xy is None else agent_xy,
            action_control=self.action_control,
            delta_action_scale=self.delta_action_scale,
        )
        return quantize_action_target_xy(target_xy, action_codec=self.action_codec)
