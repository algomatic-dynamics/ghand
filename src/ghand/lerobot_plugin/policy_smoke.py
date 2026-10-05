"""One-step policy smoke helpers for LEAP V1 KAIZEN LeRobot integration."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import RobotAction, make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

from ghand.hand import JOINT_CURRENT_LIMIT_KEY
from ghand.lerobot_plugin.config_ghand import GHandRobotConfig
from ghand.lerobot_plugin.robot import GHandRobot

GHandPolicySmokeActionSource = Literal["policy", "json", "default"]


@dataclass(frozen=True)
class GHandPolicySmokeResult:
    """Summary of one policy/action preflight step."""

    robot_type: str
    backend: str
    dataset_schema: str
    action_source: GHandPolicySmokeActionSource
    observation_feature_count: int
    action_feature_count: int
    dataset_feature_keys: tuple[str, ...]
    sent_action: dict[str, float]
    scalar_observation: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_ghand_policy_smoke(
    config: GHandRobotConfig,
    *,
    policy_path: str | Path | None = None,
    action_values: Mapping[str, float] | Sequence[float] | None = None,
    task: str = "ghand policy smoke",
    use_videos: bool = True,
) -> GHandPolicySmokeResult:
    """Load an optional policy, run one observation/action step, and disconnect."""

    robot = GHandRobot(config)
    dataset_features = make_dataset_features(robot, use_videos=use_videos)
    action_source: GHandPolicySmokeActionSource

    robot.connect()
    try:
        observation = robot.get_observation()
        if policy_path is not None:
            action = _predict_policy_action(
                robot=robot,
                dataset_features=dataset_features,
                observation=observation,
                policy_path=policy_path,
                task=task,
                fps=int(round(config.target_fps)),
                use_videos=use_videos,
            )
            action_source = "policy"
        else:
            action = named_action_from_values(
                action_values,
                action_names=tuple(robot.action_features),
                default_current_limit_ma=config.default_current_limit_ma,
            )
            action_source = "default" if action_values is None else "json"
        sent_action = robot.send_action(action)
        post_observation = robot.get_observation()
        backend = robot.runtime.backend if robot.runtime is not None else config.backend
    finally:
        robot.disconnect()

    return GHandPolicySmokeResult(
        robot_type=robot.robot_type,
        backend=backend,
        dataset_schema=config.dataset_schema,
        action_source=action_source,
        observation_feature_count=len(robot.observation_features),
        action_feature_count=len(robot.action_features),
        dataset_feature_keys=tuple(dataset_features),
        sent_action={str(name): float(value) for name, value in sent_action.items()},
        scalar_observation=_scalar_observation(post_observation),
    )


def make_dataset_features(robot: GHandRobot, *, use_videos: bool = True) -> dict[str, dict]:
    """Build LeRobot dataset features exactly like ``lerobot-record`` does."""

    teleop_action_processor, _robot_action_processor, robot_observation_processor = make_default_processors()
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=use_videos,
        ),
    )


def parse_action_json(raw_action: str | None) -> Mapping[str, float] | Sequence[float] | None:
    """Parse a JSON action mapping or vector from the CLI."""

    if raw_action is None:
        return None
    try:
        parsed = json.loads(raw_action)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--action-json must be valid JSON: {exc}") from exc
    if isinstance(parsed, Mapping):
        return {str(name): float(value) for name, value in parsed.items()}
    if isinstance(parsed, Sequence) and not isinstance(parsed, str):
        return [float(value) for value in parsed]
    raise ValueError("--action-json must be either a JSON object or a JSON array.")


def named_action_from_values(
    action_values: Mapping[str, float] | Sequence[float] | None,
    *,
    action_names: Sequence[str],
    default_current_limit_ma: float | None,
) -> RobotAction:
    """Return a complete named robot action from sparse mapping, vector, or defaults."""

    defaults = {
        name: _default_action_value(name, default_current_limit_ma=default_current_limit_ma)
        for name in action_names
    }
    if action_values is None:
        return defaults
    if isinstance(action_values, Mapping):
        unknown = tuple(sorted(set(action_values) - set(action_names)))
        if unknown:
            raise ValueError(f"Unknown action key(s): {', '.join(unknown)}.")
        merged = dict(defaults)
        for name, value in action_values.items():
            merged[str(name)] = float(value)
        return merged

    values = tuple(float(value) for value in action_values)
    if len(values) != len(action_names):
        raise ValueError(f"Expected action vector length {len(action_names)}, got {len(values)}.")
    ds_features = {
        ACTION: {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": list(action_names),
        }
    }
    return make_robot_action(torch.tensor([values], dtype=torch.float32), ds_features)


def _predict_policy_action(
    *,
    robot: GHandRobot,
    dataset_features: dict[str, dict],
    observation: Mapping[str, Any],
    policy_path: str | Path,
    task: str,
    fps: int,
    use_videos: bool,
) -> RobotAction:
    policy_path_str = str(policy_path)
    with tempfile.TemporaryDirectory(prefix="ghand_policy_smoke_") as tempdir:
        ds_meta = LeRobotDatasetMetadata.create(
            "local/ghand-policy-smoke",
            fps=max(fps, 1),
            features=dataset_features,
            robot_type=robot.name,
            root=Path(tempdir) / "dataset",
            use_videos=use_videos,
        )
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path_str)
        policy_cfg.pretrained_path = policy_path_str
        policy = make_policy(policy_cfg, ds_meta=ds_meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            dataset_stats=ds_meta.stats,
        )
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        observation_frame = build_dataset_frame(dataset_features, dict(observation), prefix=OBS_STR)
        action_tensor = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=task,
            robot_type=robot.robot_type,
        )
        return make_robot_action(action_tensor, dataset_features)


def _scalar_observation(observation: Mapping[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for name, value in observation.items():
        try:
            scalars[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return scalars


def _default_action_value(name: str, *, default_current_limit_ma: float | None) -> float:
    if name.rsplit(".", 1)[-1] == JOINT_CURRENT_LIMIT_KEY:
        return 0.0 if default_current_limit_ma is None else float(default_current_limit_ma)
    return 0.0
