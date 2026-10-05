"""Calibration contracts for human-hand retargeting features."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Final

from ghand.hand import GHAND_HAND_JOINTS
from ghand.retargeting.features import (
    FINGER_CURL_JOINT_NAMES,
    FINGER_NAMES,
    PINCH_FINGER_NAMES,
    SPREAD_FINGER_NAMES,
    THUMB_POSE_FEATURE_NAMES,
    HandFeatures,
)


RETARGET_CALIBRATION_SCHEMA_VERSION: Final[int] = 1
DEFAULT_RETARGET_CALIBRATION_RESOURCE: Final[str] = (
    "calibration/default_retarget_calibration_v1.json"
)
RETARGET_CALIBRATION_POSES: Final[tuple[str, ...]] = (
    "flat_closed",
    "fist",
    "flat_spread",
    "scissors",
    "index_pinch",
)
THUMB_TARGET_JOINTS: Final[tuple[str, ...]] = (
    "thumb_cmc_side",
    "thumb_cmc_flex",
    "thumb_mcp_flex",
    "thumb_ip_flex",
)
ROBOT_TARGET_JOINTS: Final[tuple[str, ...]] = GHAND_HAND_JOINTS
_EPSILON: Final[float] = 1e-9
_MIN_SPREAD_RANGE: Final[float] = 0.05
_SPREAD_DEADBAND: Final[float] = 0.04
_MIN_THUMB_ABDUCTION_RANGE: Final[float] = 0.05


@dataclass(frozen=True)
class RetargetPoseStats:
    """Metadata for one captured calibration pose."""

    samples: int

    def __post_init__(self) -> None:
        if self.samples < 0:
            raise ValueError("pose samples must be non-negative.")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RetargetPoseStats:
        return cls(samples=_required_int(data, "samples"))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CurlCalibration:
    """Raw feature values for an extended and curled finger."""

    open: float
    closed: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "open", _validate_float(self.open, 0.0, 1.0, "curl.open")
        )
        object.__setattr__(
            self,
            "closed",
            _validate_float(self.closed, 0.0, 1.0, "curl.closed"),
        )
        if self.closed - self.open <= _EPSILON:
            raise ValueError("curl.closed must be greater than curl.open.")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CurlCalibration:
        return cls(
            open=_required_float(data, "open"), closed=_required_float(data, "closed")
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SpreadCalibration:
    """Raw feature values for neutral and spread fingers."""

    neutral: float
    spread: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "neutral",
            _validate_float(self.neutral, -1.0, 1.0, "spread.neutral"),
        )
        object.__setattr__(
            self,
            "spread",
            _validate_float(self.spread, -1.0, 1.0, "spread.spread"),
        )
        if abs(self.spread - self.neutral) <= _EPSILON:
            raise ValueError("spread.spread must differ from spread.neutral.")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SpreadCalibration:
        return cls(
            neutral=_required_float(data, "neutral"),
            spread=_required_float(data, "spread"),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PinchCalibration:
    """Raw feature values for open and closed pinch distances."""

    open: float
    closed: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "open", _validate_float(self.open, 0.0, 1.0, "pinch.open")
        )
        object.__setattr__(
            self,
            "closed",
            _validate_float(self.closed, 0.0, 1.0, "pinch.closed"),
        )
        if self.closed - self.open <= _EPSILON:
            raise ValueError("pinch.closed must be greater than pinch.open.")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PinchCalibration:
        return cls(
            open=_required_float(data, "open"), closed=_required_float(data, "closed")
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ThumbAbductionCalibration:
    """Raw thumb lateral-open values for closed and spread flat-hand poses."""

    closed: float
    spread: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "closed",
            _validate_float(self.closed, 0.0, 1.0, "thumb_abduction.closed"),
        )
        object.__setattr__(
            self,
            "spread",
            _validate_float(self.spread, 0.0, 1.0, "thumb_abduction.spread"),
        )
        if abs(self.spread - self.closed) <= _EPSILON:
            raise ValueError(
                "thumb_abduction.spread must differ from thumb_abduction.closed."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ThumbAbductionCalibration:
        return cls(
            closed=_required_float(data, "closed"),
            spread=_required_float(data, "spread"),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ThumbJointTargets:
    """Normalized LEAP V1 KAIZEN thumb joint targets for one canonical hand pose."""

    thumb_cmc_side: float
    thumb_cmc_flex: float
    thumb_mcp_flex: float
    thumb_ip_flex: float

    def __post_init__(self) -> None:
        for joint_name in THUMB_TARGET_JOINTS:
            object.__setattr__(
                self,
                joint_name,
                _validate_float(
                    getattr(self, joint_name),
                    -1.0,
                    1.0,
                    f"thumb_targets.{joint_name}",
                ),
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ThumbJointTargets:
        return cls(
            thumb_cmc_side=_required_float(data, "thumb_cmc_side"),
            thumb_cmc_flex=_required_float(data, "thumb_cmc_flex"),
            thumb_mcp_flex=_required_float(data, "thumb_mcp_flex"),
            thumb_ip_flex=_required_float(data, "thumb_ip_flex"),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def blend(self, other: ThumbJointTargets, weight: float) -> ThumbJointTargets:
        """Linearly blend this pose target toward another target."""

        amount = _clamp(float(weight), 0.0, 1.0)
        return ThumbJointTargets(
            **{
                joint_name: _clamp(
                    _lerp(
                        getattr(self, joint_name), getattr(other, joint_name), amount
                    ),
                    -1.0,
                    1.0,
                )
                for joint_name in THUMB_TARGET_JOINTS
            },
        )


@dataclass(frozen=True)
class ThumbPoseAnchor:
    """Raw human thumb feature vector captured for one canonical pose."""

    features: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "features",
            _validate_thumb_pose_features(self.features),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ThumbPoseAnchor:
        return cls(
            features={
                feature_name: _required_float(data, feature_name)
                for feature_name in THUMB_POSE_FEATURE_NAMES
            },
        )

    def to_dict(self) -> dict[str, object]:
        return dict(self.features)


@dataclass(frozen=True)
class PoseFeatureAnchor:
    """Median raw human feature vector captured for one calibration pose."""

    finger_curl: Mapping[str, float]
    finger_joint_curl: Mapping[str, Mapping[str, float]]
    finger_spread: Mapping[str, float]
    pinch: Mapping[str, float]
    thumb_abduction: float
    thumb_pose: Mapping[str, float]
    palm_size: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "finger_curl",
            _validate_pose_feature_metric_map(
                self.finger_curl,
                FINGER_NAMES,
                lower=0.0,
                upper=1.0,
                label="pose_feature_anchors.finger_curl",
            ),
        )
        object.__setattr__(
            self,
            "finger_joint_curl",
            _validate_pose_feature_finger_joint_curl(self.finger_joint_curl),
        )
        object.__setattr__(
            self,
            "finger_spread",
            _validate_pose_feature_metric_map(
                self.finger_spread,
                SPREAD_FINGER_NAMES,
                lower=-1.0,
                upper=1.0,
                label="pose_feature_anchors.finger_spread",
            ),
        )
        object.__setattr__(
            self,
            "pinch",
            _validate_pose_feature_metric_map(
                self.pinch,
                PINCH_FINGER_NAMES,
                lower=0.0,
                upper=1.0,
                label="pose_feature_anchors.pinch",
            ),
        )
        object.__setattr__(
            self,
            "thumb_abduction",
            _validate_float(
                self.thumb_abduction,
                0.0,
                1.0,
                "pose_feature_anchors.thumb_abduction",
            ),
        )
        object.__setattr__(
            self,
            "thumb_pose",
            _validate_thumb_pose_features(self.thumb_pose),
        )
        object.__setattr__(
            self,
            "palm_size",
            _validate_float(
                self.palm_size,
                0.0,
                math.inf,
                "pose_feature_anchors.palm_size",
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PoseFeatureAnchor:
        return cls(
            finger_curl={
                finger: _required_float(_required_mapping(data, "finger_curl"), finger)
                for finger in FINGER_NAMES
            },
            finger_joint_curl=_raw_finger_joint_curl_from_dict(
                _required_mapping(data, "finger_joint_curl")
            ),
            finger_spread={
                finger: _required_float(
                    _required_mapping(data, "finger_spread"), finger
                )
                for finger in SPREAD_FINGER_NAMES
            },
            pinch={
                finger: _required_float(_required_mapping(data, "pinch"), finger)
                for finger in PINCH_FINGER_NAMES
            },
            thumb_abduction=_required_float(data, "thumb_abduction"),
            thumb_pose={
                feature_name: _required_float(
                    _required_mapping(data, "thumb_pose"), feature_name
                )
                for feature_name in THUMB_POSE_FEATURE_NAMES
            },
            palm_size=(
                _required_float(data, "palm_size") if "palm_size" in data else 0.0
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "finger_curl": dict(self.finger_curl),
            "finger_joint_curl": {
                finger: dict(joints)
                for finger, joints in self.finger_joint_curl.items()
            },
            "finger_spread": dict(self.finger_spread),
            "pinch": dict(self.pinch),
            "thumb_abduction": self.thumb_abduction,
            "thumb_pose": dict(self.thumb_pose),
            "palm_size": self.palm_size,
        }


@dataclass(frozen=True)
class RobotPoseTargets:
    """Normalized LEAP V1 KAIZEN joint targets for one canonical retargeting pose."""

    joints: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "joints", _validate_robot_pose_joints(self.joints))

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RobotPoseTargets:
        return cls(
            joints={
                joint_name: _required_float(data, joint_name)
                for joint_name in ROBOT_TARGET_JOINTS
            },
        )

    def to_dict(self) -> dict[str, object]:
        return dict(self.joints)

    def target(self, joint_name: str) -> float:
        return self.joints[joint_name]


def default_thumb_targets() -> dict[str, ThumbJointTargets]:
    """Return mechanically-informed default targets for the LEAP V1 KAIZEN thumb."""

    return _thumb_targets_from_robot_pose_targets(default_robot_pose_targets())


def default_thumb_pose_anchors() -> dict[str, ThumbPoseAnchor]:
    """Return generic thumb feature anchors for the calibration poses."""

    return {
        "flat_closed": _thumb_pose_anchor(),
        "fist": _thumb_pose_anchor(cmc_curl=1.0, mcp_curl=1.0, ip_curl=1.0),
        "flat_spread": _thumb_pose_anchor(lateral=1.0),
        "scissors": _thumb_pose_anchor(lateral=0.5, opposition=0.3),
        "index_pinch": _thumb_pose_anchor(index_pinch=1.0),
    }


def default_robot_pose_targets() -> dict[str, RobotPoseTargets]:
    """Return default LEAP V1 KAIZEN target anchors for the calibration poses."""

    return {
        "flat_closed": _robot_pose(
            thumb_cmc_side=-60.0 / 180.0,
            thumb_mcp_flex=-80.0 / 180.0,
        ),
        "fist": _robot_pose(
            ring_mcp_flex=-70.0 / 180.0,
            ring_pip_flex=-70.0 / 180.0,
            ring_dip_flex=-70.0 / 180.0,
            middle_mcp_flex=-70.0 / 180.0,
            middle_pip_flex=-70.0 / 180.0,
            middle_dip_flex=-70.0 / 180.0,
            index_mcp_flex=-70.0 / 180.0,
            index_pip_flex=-70.0 / 180.0,
            index_dip_flex=-70.0 / 180.0,
            thumb_cmc_side=-88.0 / 180.0,
            thumb_cmc_flex=-90.0 / 180.0,
            thumb_mcp_flex=-56.0 / 180.0,
            thumb_ip_flex=-62.0 / 180.0,
        ),
        "flat_spread": _robot_pose(
            ring_mcp_side=-30.0 / 180.0,
            index_mcp_side=30.0 / 180.0,
            thumb_cmc_side=-30.0 / 180.0,
        ),
        "scissors": _robot_pose(
            ring_mcp_flex=-70.0 / 180.0,
            ring_pip_flex=-70.0 / 180.0,
            ring_dip_flex=-70.0 / 180.0,
            middle_mcp_side=-20.0 / 180.0,
            index_mcp_side=20.0 / 180.0,
            thumb_cmc_side=-95.0 / 180.0,
            thumb_cmc_flex=-65.0 / 180.0,
            thumb_mcp_flex=-28.0 / 180.0,
            thumb_ip_flex=-48.0 / 180.0,
        ),
        "three_finger_extend": _robot_pose(
            ring_mcp_side=-20.0 / 180.0,
            index_mcp_side=20.0 / 180.0,
            thumb_cmc_side=-95.0 / 180.0,
            thumb_cmc_flex=-65.0 / 180.0,
            thumb_mcp_flex=-28.0 / 180.0,
            thumb_ip_flex=-48.0 / 180.0,
        ),
        "index_pinch": _robot_pose(
            index_mcp_flex=-54.0 / 180.0,
            index_pip_flex=-54.0 / 180.0,
            index_dip_flex=-37.0 / 180.0,
            thumb_cmc_side=-65.0 / 180.0,
            thumb_cmc_flex=-34.0 / 180.0,
            thumb_mcp_flex=14.0 / 180.0,
            thumb_ip_flex=-88.0 / 180.0,
        ),
        "middle_pinch": _robot_pose(
            middle_mcp_side=35.0 / 180.0,
            middle_mcp_flex=-45.0 / 180.0,
            middle_pip_flex=-54.0 / 180.0,
            middle_dip_flex=-45.0 / 180.0,
            thumb_cmc_side=-65.0 / 180.0,
            thumb_cmc_flex=-34.0 / 180.0,
            thumb_mcp_flex=-20.0 / 180.0,
            thumb_ip_flex=-66.0 / 180.0,
        ),
        "ring_pinch": _robot_pose(
            ring_mcp_side=54.0 / 180.0,
            ring_mcp_flex=-45.0 / 180.0,
            ring_pip_flex=-54.0 / 180.0,
            ring_dip_flex=-45.0 / 180.0,
            thumb_cmc_side=-65.0 / 180.0,
            thumb_cmc_flex=-54.0 / 180.0,
            thumb_mcp_flex=-48.0 / 180.0,
            thumb_ip_flex=-48.0 / 180.0,
        ),
    }


def _robot_pose(**overrides: float) -> RobotPoseTargets:
    values = {joint_name: 0.0 for joint_name in ROBOT_TARGET_JOINTS}
    values.update(overrides)
    return RobotPoseTargets(joints=values)


def _thumb_pose_anchor(**overrides: float) -> ThumbPoseAnchor:
    values = {feature_name: 0.0 for feature_name in THUMB_POSE_FEATURE_NAMES}
    values.update(overrides)
    return ThumbPoseAnchor(features=values)


def _thumb_targets_from_robot_pose_targets(
    robot_pose_targets: Mapping[str, RobotPoseTargets],
) -> dict[str, ThumbJointTargets]:
    return {
        pose_name: ThumbJointTargets(
            thumb_cmc_side=pose.target("thumb_cmc_side"),
            thumb_cmc_flex=pose.target("thumb_cmc_flex"),
            thumb_mcp_flex=pose.target("thumb_mcp_flex"),
            thumb_ip_flex=pose.target("thumb_ip_flex"),
        )
        for pose_name, pose in robot_pose_targets.items()
    }


def _with_default_pose_stats(
    poses: Mapping[str, RetargetPoseStats],
) -> dict[str, RetargetPoseStats]:
    updated = {
        pose: RetargetPoseStats(samples=0) for pose in RETARGET_CALIBRATION_POSES
    }
    updated.update(poses)
    return updated


def _with_default_robot_pose_targets(
    robot_pose_targets: Mapping[str, RobotPoseTargets],
) -> dict[str, RobotPoseTargets]:
    updated = default_robot_pose_targets()
    updated.update(robot_pose_targets)
    return updated


def _with_default_thumb_targets(
    thumb_targets: Mapping[str, ThumbJointTargets],
) -> dict[str, ThumbJointTargets]:
    updated = default_thumb_targets()
    updated.update(thumb_targets)
    return updated


def _with_default_thumb_pose_anchors(
    thumb_pose_anchors: Mapping[str, ThumbPoseAnchor],
) -> dict[str, ThumbPoseAnchor]:
    updated = default_thumb_pose_anchors()
    updated.update(thumb_pose_anchors)
    return updated


@dataclass(frozen=True)
class HandRetargetCalibration:
    """JSON-serializable calibration for MediaPipe-derived retargeting features."""

    version: int
    handedness: str
    poses: Mapping[str, RetargetPoseStats]
    finger_curl: Mapping[str, CurlCalibration]
    finger_spread: Mapping[str, SpreadCalibration]
    pinch: Mapping[str, PinchCalibration]
    thumb_abduction: ThumbAbductionCalibration = field(
        default_factory=lambda: ThumbAbductionCalibration(closed=0.0, spread=1.0),
    )
    robot_pose_targets: Mapping[str, RobotPoseTargets] = field(
        default_factory=default_robot_pose_targets,
    )
    thumb_targets: Mapping[str, ThumbJointTargets] = field(
        default_factory=default_thumb_targets
    )
    finger_joint_curl: Mapping[str, Mapping[str, CurlCalibration]] | None = None
    thumb_pose_anchors: Mapping[str, ThumbPoseAnchor] = field(
        default_factory=default_thumb_pose_anchors
    )
    pose_feature_anchors: Mapping[str, PoseFeatureAnchor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version != RETARGET_CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported retarget calibration version: {self.version}; "
                f"expected {RETARGET_CALIBRATION_SCHEMA_VERSION}."
            )
        if self.handedness not in {"Left", "Right"}:
            raise ValueError("handedness must be 'Left' or 'Right'.")

        object.__setattr__(
            self,
            "poses",
            _validate_mapping(
                _with_default_pose_stats(self.poses),
                RETARGET_CALIBRATION_POSES,
                RetargetPoseStats,
                label="poses",
                allow_extra=True,
            ),
        )
        object.__setattr__(
            self,
            "finger_curl",
            _validate_mapping(
                self.finger_curl,
                FINGER_NAMES,
                CurlCalibration,
                label="finger_curl",
            ),
        )
        object.__setattr__(
            self,
            "finger_joint_curl",
            _validate_finger_joint_curl_calibration(
                self.finger_joint_curl
                if self.finger_joint_curl is not None
                else _finger_joint_curl_calibration_from_aggregate(self.finger_curl),
            ),
        )
        object.__setattr__(
            self,
            "finger_spread",
            _validate_mapping(
                self.finger_spread,
                SPREAD_FINGER_NAMES,
                SpreadCalibration,
                label="finger_spread",
            ),
        )
        object.__setattr__(
            self,
            "pinch",
            _validate_mapping(
                self.pinch, PINCH_FINGER_NAMES, PinchCalibration, label="pinch"
            ),
        )
        object.__setattr__(
            self,
            "robot_pose_targets",
            _validate_mapping(
                _with_default_robot_pose_targets(self.robot_pose_targets),
                RETARGET_CALIBRATION_POSES,
                RobotPoseTargets,
                label="robot_pose_targets",
                allow_extra=True,
            ),
        )
        object.__setattr__(
            self,
            "thumb_targets",
            _validate_mapping(
                _with_default_thumb_targets(self.thumb_targets),
                RETARGET_CALIBRATION_POSES,
                ThumbJointTargets,
                label="thumb_targets",
                allow_extra=True,
            ),
        )
        object.__setattr__(
            self,
            "thumb_pose_anchors",
            _validate_mapping(
                _with_default_thumb_pose_anchors(self.thumb_pose_anchors),
                RETARGET_CALIBRATION_POSES,
                ThumbPoseAnchor,
                label="thumb_pose_anchors",
                allow_extra=True,
            ),
        )
        object.__setattr__(
            self,
            "pose_feature_anchors",
            _validate_mapping(
                self.pose_feature_anchors,
                (),
                PoseFeatureAnchor,
                label="pose_feature_anchors",
                allow_extra=True,
            ),
        )

    @classmethod
    def identity(cls, *, handedness: str = "Right") -> HandRetargetCalibration:
        """Return a no-op calibration useful as a default baseline."""

        return cls(
            version=RETARGET_CALIBRATION_SCHEMA_VERSION,
            handedness=handedness,
            poses={
                pose: RetargetPoseStats(samples=0)
                for pose in RETARGET_CALIBRATION_POSES
            },
            finger_curl={
                finger: CurlCalibration(open=0.0, closed=1.0) for finger in FINGER_NAMES
            },
            finger_spread={
                finger: SpreadCalibration(neutral=0.0, spread=1.0)
                for finger in SPREAD_FINGER_NAMES
            },
            pinch={
                finger: PinchCalibration(open=0.0, closed=1.0)
                for finger in PINCH_FINGER_NAMES
            },
            finger_joint_curl=None,
            thumb_abduction=ThumbAbductionCalibration(closed=0.0, spread=1.0),
            robot_pose_targets=default_robot_pose_targets(),
            thumb_targets=default_thumb_targets(),
            thumb_pose_anchors=default_thumb_pose_anchors(),
            pose_feature_anchors={},
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> HandRetargetCalibration:
        finger_joint_curl_data = data.get("finger_joint_curl")
        thumb_abduction_data = data.get("thumb_abduction")
        robot_pose_targets_data = data.get("robot_pose_targets")
        thumb_targets_data = data.get("thumb_targets")
        thumb_pose_anchors_data = data.get("thumb_pose_anchors")
        pose_feature_anchors_data = data.get("pose_feature_anchors")
        return cls(
            version=_required_int(data, "version"),
            handedness=_required_str(data, "handedness"),
            poses=_typed_mapping(
                _required_mapping(data, "poses"),
                RetargetPoseStats,
            ),
            finger_curl=_typed_mapping(
                _required_mapping(data, "finger_curl"),
                CurlCalibration,
            ),
            finger_spread=_typed_mapping(
                _required_mapping(data, "finger_spread"),
                SpreadCalibration,
            ),
            pinch=_typed_mapping(_required_mapping(data, "pinch"), PinchCalibration),
            finger_joint_curl=(
                _typed_finger_joint_curl_mapping(finger_joint_curl_data)
                if isinstance(finger_joint_curl_data, Mapping)
                else None
            ),
            thumb_abduction=(
                ThumbAbductionCalibration.from_dict(thumb_abduction_data)
                if isinstance(thumb_abduction_data, Mapping)
                else ThumbAbductionCalibration(closed=0.0, spread=1.0)
            ),
            robot_pose_targets=(
                _typed_mapping(robot_pose_targets_data, RobotPoseTargets)
                if isinstance(robot_pose_targets_data, Mapping)
                else default_robot_pose_targets()
            ),
            thumb_targets=(
                _typed_mapping(thumb_targets_data, ThumbJointTargets)
                if isinstance(thumb_targets_data, Mapping)
                else default_thumb_targets()
            ),
            thumb_pose_anchors=(
                _typed_mapping(thumb_pose_anchors_data, ThumbPoseAnchor)
                if isinstance(thumb_pose_anchors_data, Mapping)
                else default_thumb_pose_anchors()
            ),
            pose_feature_anchors=(
                _typed_mapping(pose_feature_anchors_data, PoseFeatureAnchor)
                if isinstance(pose_feature_anchors_data, Mapping)
                else {}
            ),
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "version": self.version,
            "handedness": self.handedness,
            "poses": {key: value.to_dict() for key, value in self.poses.items()},
            "finger_curl": {
                key: value.to_dict() for key, value in self.finger_curl.items()
            },
            "finger_joint_curl": {
                finger: {
                    joint: calibration.to_dict()
                    for joint, calibration in joints.items()
                }
                for finger, joints in self.finger_joint_curl.items()
            },
            "finger_spread": {
                key: value.to_dict() for key, value in self.finger_spread.items()
            },
            "pinch": {key: value.to_dict() for key, value in self.pinch.items()},
            "thumb_abduction": self.thumb_abduction.to_dict(),
            "robot_pose_targets": {
                key: value.to_dict() for key, value in self.robot_pose_targets.items()
            },
            "thumb_targets": {
                key: value.to_dict() for key, value in self.thumb_targets.items()
            },
            "thumb_pose_anchors": {
                key: value.to_dict() for key, value in self.thumb_pose_anchors.items()
            },
            "pose_feature_anchors": {
                key: value.to_dict() for key, value in self.pose_feature_anchors.items()
            },
        }
        return data


@dataclass(frozen=True)
class HandFeatureCalibrator:
    """Apply retargeting calibration to raw MediaPipe-derived hand features."""

    calibration: HandRetargetCalibration

    def __call__(self, features: HandFeatures) -> HandFeatures:
        return apply_hand_retarget_calibration(features, self.calibration)


def apply_hand_retarget_calibration(
    features: HandFeatures,
    calibration: HandRetargetCalibration,
) -> HandFeatures:
    """Normalize raw hand features using a captured retargeting calibration."""

    if not features.valid:
        return HandFeatures.empty(
            timestamp_ns=features.timestamp_ns,
            handedness=features.handedness,
        )

    return HandFeatures(
        timestamp_ns=features.timestamp_ns,
        handedness=features.handedness,
        score=features.score,
        finger_curl={
            finger: _calibrate_unipolar(
                features.finger_curl[finger],
                calibration.finger_curl[finger].open,
                calibration.finger_curl[finger].closed,
            )
            for finger in FINGER_NAMES
        },
        finger_joint_curl={
            finger: {
                joint: _calibrate_unipolar(
                    features.finger_joint_curl[finger][joint],
                    calibration.finger_joint_curl[finger][joint].open,
                    calibration.finger_joint_curl[finger][joint].closed,
                )
                for joint in FINGER_CURL_JOINT_NAMES
            }
            for finger in FINGER_NAMES
        },
        finger_spread={
            finger: _calibrate_spread(
                features.finger_spread[finger],
                calibration.finger_spread[finger].neutral,
                calibration.finger_spread[finger].spread,
            )
            for finger in SPREAD_FINGER_NAMES
        },
        pinch={
            finger: _calibrate_unipolar(
                features.pinch[finger],
                calibration.pinch[finger].open,
                calibration.pinch[finger].closed,
            )
            for finger in PINCH_FINGER_NAMES
        },
        palm_size=features.palm_size,
        thumb_abduction=_calibrate_thumb_abduction(
            features.thumb_abduction,
            calibration.thumb_abduction.closed,
            calibration.thumb_abduction.spread,
        ),
        thumb_pose=features.thumb_pose,
        valid=True,
    )


def load_hand_retarget_calibration(path: Path | str) -> HandRetargetCalibration:
    """Load a hand retargeting calibration JSON file."""

    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("retarget calibration JSON root must be an object.")
    return HandRetargetCalibration.from_dict(loaded)


def load_default_hand_retarget_calibration() -> HandRetargetCalibration:
    """Load the packaged default hand retargeting calibration."""

    resource = files("ghand.assets").joinpath(DEFAULT_RETARGET_CALIBRATION_RESOURCE)
    loaded = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("default retarget calibration JSON root must be an object.")
    return HandRetargetCalibration.from_dict(loaded)


def save_hand_retarget_calibration(
    calibration: HandRetargetCalibration,
    path: Path | str,
) -> None:
    """Write a hand retargeting calibration JSON file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(calibration.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_mapping(
    values: Mapping[str, object],
    required_keys: tuple[str, ...],
    value_type: type[Any],
    *,
    label: str,
    allow_extra: bool = False,
) -> dict[str, object]:
    value_by_key = dict(values)
    required = set(required_keys)
    actual = set(value_by_key)
    missing = required - actual
    extra = actual - required
    if missing or (extra and not allow_extra):
        missing_text = ", ".join(sorted(missing)) or "none"
        extra_text = ", ".join(sorted(extra)) or "none"
        raise ValueError(
            f"{label} keys invalid; missing={missing_text}; extra={extra_text}."
        )
    for key, value in value_by_key.items():
        if not isinstance(value, value_type):
            raise ValueError(f"{label}.{key} must be {value_type.__name__}.")
    return value_by_key


def _calibrate_unipolar(value: float, lower_raw: float, upper_raw: float) -> float:
    return _clamp((float(value) - lower_raw) / (upper_raw - lower_raw), 0.0, 1.0)


def _calibrate_spread(value: float, neutral_raw: float, spread_raw: float) -> float:
    spread_range = spread_raw - neutral_raw
    if abs(spread_range) < _MIN_SPREAD_RANGE:
        return 0.0
    calibrated = _clamp((float(value) - neutral_raw) / spread_range, -1.0, 1.0)
    if abs(calibrated) < _SPREAD_DEADBAND:
        return 0.0
    return calibrated


def _calibrate_thumb_abduction(
    value: float, closed_raw: float, spread_raw: float
) -> float:
    spread_range = spread_raw - closed_raw
    if abs(spread_range) < _MIN_THUMB_ABDUCTION_RANGE:
        return 0.0
    return _clamp((float(value) - closed_raw) / spread_range, 0.0, 1.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _lerp(start: float, end: float, amount: float) -> float:
    return float(start) + (float(end) - float(start)) * float(amount)


def _validate_robot_pose_joints(values: Mapping[str, float]) -> dict[str, float]:
    value_by_key = dict(values)
    required = set(ROBOT_TARGET_JOINTS)
    actual = set(value_by_key)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        missing_text = ", ".join(sorted(missing)) or "none"
        extra_text = ", ".join(sorted(extra)) or "none"
        raise ValueError(
            "robot_pose_targets joints invalid; "
            f"missing={missing_text}; extra={extra_text}."
        )
    return {
        joint_name: _validate_float(
            value_by_key[joint_name],
            -1.0,
            1.0,
            f"robot_pose_targets.{joint_name}",
        )
        for joint_name in ROBOT_TARGET_JOINTS
    }


def _validate_thumb_pose_features(values: Mapping[str, float]) -> dict[str, float]:
    value_by_key = dict(values)
    required = set(THUMB_POSE_FEATURE_NAMES)
    actual = set(value_by_key)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        missing_text = ", ".join(sorted(missing)) or "none"
        extra_text = ", ".join(sorted(extra)) or "none"
        raise ValueError(
            "thumb_pose_anchors features invalid; "
            f"missing={missing_text}; extra={extra_text}."
        )
    return {
        feature_name: _validate_float(
            value_by_key[feature_name],
            -1.0,
            1.0,
            f"thumb_pose_anchors.{feature_name}",
        )
        for feature_name in THUMB_POSE_FEATURE_NAMES
    }


def _validate_pose_feature_metric_map(
    values: Mapping[str, float],
    required_keys: tuple[str, ...],
    *,
    lower: float,
    upper: float,
    label: str,
) -> dict[str, float]:
    value_by_key = dict(values)
    required = set(required_keys)
    actual = set(value_by_key)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        missing_text = ", ".join(sorted(missing)) or "none"
        extra_text = ", ".join(sorted(extra)) or "none"
        raise ValueError(
            f"{label} keys invalid; missing={missing_text}; extra={extra_text}."
        )
    return {
        key: _validate_float(value_by_key[key], lower, upper, f"{label}.{key}")
        for key in required_keys
    }


def _validate_pose_feature_finger_joint_curl(
    values: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    value_by_key = dict(values)
    required = set(FINGER_NAMES)
    actual = set(value_by_key)
    if actual != required:
        missing_text = ", ".join(sorted(required - actual)) or "none"
        extra_text = ", ".join(sorted(actual - required)) or "none"
        raise ValueError(
            "pose_feature_anchors.finger_joint_curl keys invalid; "
            f"missing={missing_text}; extra={extra_text}."
        )
    return {
        finger: _validate_pose_feature_metric_map(
            joints,
            FINGER_CURL_JOINT_NAMES,
            lower=0.0,
            upper=1.0,
            label=f"pose_feature_anchors.finger_joint_curl.{finger}",
        )
        for finger, joints in value_by_key.items()
    }


def _raw_finger_joint_curl_from_dict(
    data: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    parsed: dict[str, dict[str, float]] = {}
    for finger, value in data.items():
        if not isinstance(finger, str):
            raise ValueError(
                "pose_feature_anchors.finger_joint_curl mapping keys must be strings."
            )
        if not isinstance(value, Mapping):
            raise ValueError(
                f"pose_feature_anchors.finger_joint_curl.{finger} must be an object."
            )
        parsed[finger] = {
            joint: _required_float(value, joint) for joint in FINGER_CURL_JOINT_NAMES
        }
    return parsed


def _validate_finger_joint_curl_calibration(
    values: Mapping[str, Mapping[str, CurlCalibration]],
) -> dict[str, dict[str, CurlCalibration]]:
    value_by_key = dict(values)
    required = set(FINGER_NAMES)
    actual = set(value_by_key)
    if actual != required:
        missing_text = ", ".join(sorted(required - actual)) or "none"
        extra_text = ", ".join(sorted(actual - required)) or "none"
        raise ValueError(
            "finger_joint_curl keys invalid; "
            f"missing={missing_text}; extra={extra_text}."
        )

    return {
        finger: _validate_mapping(
            joints,
            FINGER_CURL_JOINT_NAMES,
            CurlCalibration,
            label=f"finger_joint_curl.{finger}",
        )
        for finger, joints in value_by_key.items()
    }


def _finger_joint_curl_calibration_from_aggregate(
    finger_curl: Mapping[str, CurlCalibration],
) -> dict[str, dict[str, CurlCalibration]]:
    return {
        finger: {joint: calibration for joint in FINGER_CURL_JOINT_NAMES}
        for finger, calibration in finger_curl.items()
    }


def _typed_finger_joint_curl_mapping(
    data: Mapping[str, object],
) -> dict[str, dict[str, CurlCalibration]]:
    parsed: dict[str, dict[str, CurlCalibration]] = {}
    for finger, value in data.items():
        if not isinstance(finger, str):
            raise ValueError("finger_joint_curl mapping keys must be strings.")
        if not isinstance(value, Mapping):
            raise ValueError(
                f"finger_joint_curl.{finger} calibration entry must be an object."
            )
        parsed[finger] = {
            joint: calibration
            for joint, calibration in _typed_mapping(value, CurlCalibration).items()
        }
    return parsed


def _typed_mapping(
    data: Mapping[str, object], value_type: type[Any]
) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError("retarget calibration mapping keys must be strings.")
        if isinstance(value, value_type):
            parsed[key] = value
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"{key} calibration entry must be an object.")
        parsed[key] = value_type.from_dict(value)
    return parsed


def _required_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = _required(data, key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object.")
    return value


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = _required(data, key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    return value


def _required_int(data: Mapping[str, object], key: str) -> int:
    value = _required(data, key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    return value


def _required_float(data: Mapping[str, object], key: str) -> float:
    value = _required(data, key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number.")
    return float(value)


def _required(data: Mapping[str, object], key: str) -> object:
    if key not in data:
        raise ValueError(f"Missing required calibration field: {key}.")
    return data[key]


def _validate_float(value: float, lower: float, upper: float, label: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite.")
    if not lower <= normalized <= upper:
        raise ValueError(f"{label} must be in range {lower}..{upper}.")
    return normalized
