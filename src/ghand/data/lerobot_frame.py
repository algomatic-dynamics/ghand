"""Adapters from runtime snapshots into LeRobot frame dictionaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from ghand.data.lerobot_export import FrameDict
from ghand.data.lerobot_schema import (
    ACTION_KEY,
    AUX_TELEOP_HANDEDNESS_KEY,
    AUX_TELEOP_LANDMARKS_KEY,
    AUX_TELEOP_SCORE_KEY,
    AUX_TELEOP_VALID_KEY,
    AUX_TELEOP_WORLD_LANDMARKS_KEY,
    FeatureDict,
    HAND_LANDMARK_COUNT,
    HANDEDNESS_LABEL_TO_ID,
    LANDMARK_DIMS,
    OBSERVATION_IMAGES_PREFIX,
    OBSERVATION_STATE_KEY,
    TELEOP_HAND_SLOT_NAMES,
    TELEOP_HAND_SLOTS,
    camera_feature_key,
)
from ghand.hand import (
    JOINT_CURRENT_LIMIT_KEY,
    JOINT_CONTACT_KEY,
    JOINT_CURRENT_RESIDUAL_KEY,
    JOINT_POSITION_KEY,
    JOINT_PRESENT_CURRENT_KEY,
)
from ghand.teleop.hand_pose import HandDetection, HandFrame, Landmark

ImageColorFormat = Literal["bgr", "rgb"]
RuntimeActionSource = Literal["target", "commanded"]
CurrentLimitAction = Mapping[str, float]
DEFAULT_CURRENT_CONTACT_THRESHOLD_MA: Final[float] = 50.0


@runtime_checkable
class RuntimeJointStateLike(Protocol):
    """Subset of JointRuntimeState used by the dataset adapter."""

    joint_name: str
    target: float
    measured: float
    commanded: float
    current_limit_ma: float | None
    present_current_ma: float | None


@runtime_checkable
class RuntimeSnapshotLike(Protocol):
    """Subset of OperatorRuntimeSnapshot used by the dataset adapter."""

    joints: Sequence[RuntimeJointStateLike]


@runtime_checkable
class RadianTargetLike(Protocol):
    """Subset of RadianJointTarget used by the dataset adapter."""

    joints: Mapping[str, float]


def make_lerobot_frame_from_runtime_snapshot(
    *,
    snapshot: RuntimeSnapshotLike,
    features: FeatureDict,
    task: str,
    camera_frames: Mapping[str, object] | None = None,
    hand_frame: HandFrame | None = None,
    action: RadianTargetLike | None = None,
    current_limit_action: CurrentLimitAction | None = None,
    default_current_limit_ma: float | None = None,
    current_contact_threshold_ma: float = DEFAULT_CURRENT_CONTACT_THRESHOLD_MA,
    action_source: RuntimeActionSource = "target",
    timestamp_ns: int | None = None,
    image_color_format: ImageColorFormat = "bgr",
) -> FrameDict:
    """Build one LeRobot frame from a hand runtime snapshot and camera images.

    `observation.state` is measured joint radians, plus present motor current
    when the feature schema includes current channels. `action` is target joint
    radians by default, or commanded radians when `action_source="commanded"`.
    If an explicit radian action is provided, it overrides the snapshot action
    values for the joints it contains. Current-limit action values are only
    emitted when the feature schema includes `*.current_limit_ma` channels.
    """

    if not task:
        raise ValueError("task must not be empty.")
    if action_source not in {"target", "commanded"}:
        raise ValueError("action_source must be 'target' or 'commanded'.")
    if image_color_format not in {"bgr", "rgb"}:
        raise ValueError("image_color_format must be 'bgr' or 'rgb'.")
    current_contact_threshold_ma = _validate_non_negative_finite(
        current_contact_threshold_ma,
        "current_contact_threshold_ma",
    )

    state_names = _feature_names(features, OBSERVATION_STATE_KEY)
    action_names = _feature_names(features, ACTION_KEY)
    joints_by_name = _snapshot_joints_by_name(snapshot)
    state_values = _state_values_from_snapshot(joints_by_name)
    state_values.update(
        _current_diagnostic_state_values(
            joints_by_name=joints_by_name,
            state_names=state_names,
            current_contact_threshold_ma=current_contact_threshold_ma,
        )
    )
    snapshot_action_values = {
        joint_name: getattr(joint, action_source)
        for joint_name, joint in joints_by_name.items()
    }
    if action is not None:
        snapshot_action_values.update(
            {joint_name: float(value) for joint_name, value in action.joints.items()}
        )
    snapshot_action_values.update(
        _current_limit_action_values(
            current_limit_action=current_limit_action,
            default_current_limit_ma=default_current_limit_ma,
            joints_by_name=joints_by_name,
            action_names=action_names,
        )
    )

    frame: FrameDict = {
        "task": task,
        OBSERVATION_STATE_KEY: _vector_from_joint_mapping(
            state_values,
            state_names,
            value_label=OBSERVATION_STATE_KEY,
        ),
        ACTION_KEY: _vector_from_joint_mapping(
            snapshot_action_values,
            action_names,
            value_label=ACTION_KEY,
        ),
    }
    resolved_timestamp_ns = (
        hand_frame.timestamp_ns if timestamp_ns is None and hand_frame is not None else timestamp_ns
    )
    if resolved_timestamp_ns is not None:
        frame["timestamp"] = float(resolved_timestamp_ns) / 1_000_000_000.0

    frame.update(
        _make_camera_frame_entries(
            features=features,
            camera_frames=camera_frames or {},
            image_color_format=image_color_format,
        )
    )
    frame.update(_make_aux_teleop_entries(features=features, hand_frame=hand_frame))
    return frame


def _feature_names(features: FeatureDict, key: str) -> tuple[str, ...]:
    try:
        raw_names = features[key]["names"]
    except KeyError as exc:
        raise ValueError(f"features must include {key!r}.") from exc
    if not isinstance(raw_names, list) or not all(isinstance(name, str) for name in raw_names):
        raise ValueError(f"{key!r} feature must define string names.")
    return tuple(raw_names)


def _snapshot_joints_by_name(snapshot: RuntimeSnapshotLike) -> dict[str, RuntimeJointStateLike]:
    joints_by_name: dict[str, RuntimeJointStateLike] = {}
    for joint in snapshot.joints:
        if joint.joint_name in joints_by_name:
            raise ValueError(f"Duplicate runtime joint in snapshot: {joint.joint_name!r}.")
        joints_by_name[joint.joint_name] = joint
    return joints_by_name


def _state_values_from_snapshot(
    joints_by_name: Mapping[str, RuntimeJointStateLike],
) -> dict[str, float]:
    values = {
        joint_name: float(joint.measured) for joint_name, joint in joints_by_name.items()
    }
    for joint_name, joint in joints_by_name.items():
        current = getattr(joint, "present_current_ma", None)
        if current is None:
            continue
        values[_replace_joint_value_key(joint_name, JOINT_PRESENT_CURRENT_KEY)] = float(current)
    return values


def _current_limit_action_values(
    *,
    current_limit_action: CurrentLimitAction | None,
    default_current_limit_ma: float | None,
    joints_by_name: Mapping[str, RuntimeJointStateLike],
    action_names: Sequence[str],
) -> dict[str, float]:
    current_limit_names = tuple(
        name for name in action_names if _joint_value_key(name) == JOINT_CURRENT_LIMIT_KEY
    )
    if not current_limit_names:
        return {}

    values: dict[str, float] = {}
    for joint_name, joint in joints_by_name.items():
        current_limit = getattr(joint, "current_limit_ma", None)
        if current_limit is None:
            continue
        values[_replace_joint_value_key(joint_name, JOINT_CURRENT_LIMIT_KEY)] = float(
            current_limit
        )

    if default_current_limit_ma is not None:
        for name in current_limit_names:
            values.setdefault(name, float(default_current_limit_ma))

    if current_limit_action is None:
        return values

    position_name_set = set(joints_by_name)
    current_limit_name_set = set(current_limit_names)
    for raw_name, value in current_limit_action.items():
        name = str(raw_name)
        if name in position_name_set:
            values[_replace_joint_value_key(name, JOINT_CURRENT_LIMIT_KEY)] = float(value)
        elif name in current_limit_name_set:
            values[name] = float(value)
        else:
            raise ValueError(f"Current-limit action joint is not configured: {name}.")
    return values


def _current_diagnostic_state_values(
    *,
    joints_by_name: Mapping[str, RuntimeJointStateLike],
    state_names: Sequence[str],
    current_contact_threshold_ma: float,
) -> dict[str, float]:
    residual_names = tuple(
        name for name in state_names if _joint_value_key(name) == JOINT_CURRENT_RESIDUAL_KEY
    )
    contact_names = tuple(
        name for name in state_names if _joint_value_key(name) == JOINT_CONTACT_KEY
    )
    if not residual_names and not contact_names:
        return {}

    values: dict[str, float] = {}
    for joint_name, joint in joints_by_name.items():
        present_current_ma = getattr(joint, "present_current_ma", None)
        if present_current_ma is None:
            continue
        current_magnitude_ma = abs(float(present_current_ma))
        residual_ma = max(current_magnitude_ma - current_contact_threshold_ma, 0.0)
        if residual_names:
            values[_replace_joint_value_key(joint_name, JOINT_CURRENT_RESIDUAL_KEY)] = residual_ma
        if contact_names:
            values[_replace_joint_value_key(joint_name, JOINT_CONTACT_KEY)] = (
                1.0 if residual_ma > 0.0 else 0.0
            )
    return values


def _vector_from_joint_mapping(
    values: Mapping[str, float],
    joint_names: Sequence[str],
    *,
    value_label: str,
) -> np.ndarray:
    missing = tuple(joint_name for joint_name in joint_names if joint_name not in values)
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing {value_label} joint value(s): {missing_text}.")
    return np.asarray([float(values[joint_name]) for joint_name in joint_names], dtype=np.float32)


def _joint_value_key(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _replace_joint_value_key(name: str, value_key: str) -> str:
    prefix, separator, key = name.rpartition(".")
    if separator == "" or key != JOINT_POSITION_KEY:
        raise ValueError(f"Expected a joint position channel ending with '.{JOINT_POSITION_KEY}': {name}.")
    return f"{prefix}.{value_key}"


def _validate_non_negative_finite(value: float, label: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be finite.")
    if resolved < 0.0:
        raise ValueError(f"{label} must be non-negative.")
    return resolved


def _make_camera_frame_entries(
    *,
    features: FeatureDict,
    camera_frames: Mapping[str, object],
    image_color_format: ImageColorFormat,
) -> dict[str, object]:
    image_feature_keys = tuple(
        key for key in features if key.startswith(f"{OBSERVATION_IMAGES_PREFIX}.")
    )
    if not image_feature_keys:
        return {}

    entries: dict[str, object] = {}
    missing: list[str] = []
    for feature_key in image_feature_keys:
        camera_name = feature_key.removeprefix(f"{OBSERVATION_IMAGES_PREFIX}.")
        if feature_key in camera_frames:
            raw_frame = camera_frames[feature_key]
        elif camera_name in camera_frames:
            raw_frame = camera_frames[camera_name]
        else:
            missing.append(camera_name)
            continue
        expected_shape = features[feature_key].get("shape")
        entries[feature_key] = _prepare_camera_frame(
            raw_frame,
            feature_key=feature_key,
            expected_shape=expected_shape,
            image_color_format=image_color_format,
        )

    if missing:
        raise ValueError(f"Missing camera frame(s): {', '.join(missing)}.")
    return entries


def _prepare_camera_frame(
    frame: object,
    *,
    feature_key: str,
    expected_shape: object,
    image_color_format: ImageColorFormat,
) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{feature_key} frame must have shape (height, width, 3).")
    if expected_shape is not None and tuple(image.shape) != tuple(expected_shape):
        raise ValueError(
            f"{feature_key} frame shape {tuple(image.shape)} does not match "
            f"feature shape {tuple(expected_shape)}."
        )
    if image.dtype != np.uint8:
        raise ValueError(f"{feature_key} frame must use dtype uint8.")
    if image_color_format == "bgr":
        return np.ascontiguousarray(image[..., ::-1])
    return np.ascontiguousarray(image)


def _make_aux_teleop_entries(
    *,
    features: FeatureDict,
    hand_frame: HandFrame | None,
) -> dict[str, object]:
    aux_keys = {
        AUX_TELEOP_LANDMARKS_KEY,
        AUX_TELEOP_WORLD_LANDMARKS_KEY,
        AUX_TELEOP_HANDEDNESS_KEY,
        AUX_TELEOP_SCORE_KEY,
        AUX_TELEOP_VALID_KEY,
    }
    if not aux_keys.intersection(features):
        return {}

    landmarks = np.zeros(
        (TELEOP_HAND_SLOTS, HAND_LANDMARK_COUNT, LANDMARK_DIMS),
        dtype=np.float32,
    )
    world_landmarks = np.zeros_like(landmarks)
    handedness = np.zeros((TELEOP_HAND_SLOTS,), dtype=np.int64)
    scores = np.zeros((TELEOP_HAND_SLOTS,), dtype=np.float32)
    valid = np.zeros((TELEOP_HAND_SLOTS,), dtype=np.int64)

    if hand_frame is not None:
        for hand in hand_frame.hands:
            slot = _hand_slot(hand)
            if slot is None:
                continue
            score = float(hand.score)
            if valid[slot] and score <= scores[slot]:
                continue
            landmarks[slot] = _landmarks_to_array(hand.landmarks)
            if hand.world_landmarks is not None:
                world_landmarks[slot] = _landmarks_to_array(hand.world_landmarks)
            handedness[slot] = HANDEDNESS_LABEL_TO_ID.get(hand.handedness, 0)
            scores[slot] = score
            valid[slot] = 1

    entries: dict[str, object] = {}
    if AUX_TELEOP_LANDMARKS_KEY in features:
        entries[AUX_TELEOP_LANDMARKS_KEY] = landmarks
    if AUX_TELEOP_WORLD_LANDMARKS_KEY in features:
        entries[AUX_TELEOP_WORLD_LANDMARKS_KEY] = world_landmarks
    if AUX_TELEOP_HANDEDNESS_KEY in features:
        entries[AUX_TELEOP_HANDEDNESS_KEY] = handedness
    if AUX_TELEOP_SCORE_KEY in features:
        entries[AUX_TELEOP_SCORE_KEY] = scores
    if AUX_TELEOP_VALID_KEY in features:
        entries[AUX_TELEOP_VALID_KEY] = valid
    return entries


def _hand_slot(hand: HandDetection) -> int | None:
    label = hand.handedness.casefold()
    for slot, slot_name in enumerate(TELEOP_HAND_SLOT_NAMES):
        if label == slot_name:
            return slot
    return None


def _landmarks_to_array(landmarks: Sequence[Landmark]) -> np.ndarray:
    values = np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS), dtype=np.float32)
    for index, landmark in enumerate(landmarks[:HAND_LANDMARK_COUNT]):
        values[index] = (float(landmark.x), float(landmark.y), float(landmark.z))
    return values


def make_camera_frame_mapping(name: str, frame: object) -> dict[str, object]:
    """Return a camera frame mapping for one camera name."""

    return {camera_feature_key(name): frame}
