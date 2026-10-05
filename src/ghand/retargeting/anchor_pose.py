"""Anchor-pose retargeting primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from ghand.retargeting.calibration import (
    HandRetargetCalibration,
    PoseFeatureAnchor,
    RobotPoseTargets,
    apply_hand_retarget_calibration,
    default_robot_pose_targets,
)
from ghand.retargeting.features import (
    FINGER_CURL_JOINT_NAMES,
    PINCH_FINGER_NAMES,
    THUMB_POSE_FEATURE_NAMES,
    HandFeatures,
)
from ghand.hand import JointTarget


ROBOT_FINGER_NAMES: Final[tuple[str, ...]] = ("ring", "middle", "index")
FINGER_JOINT_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "ring": ("ring_mcp_side", "ring_mcp_flex", "ring_pip_flex", "ring_dip_flex"),
    "middle": (
        "middle_mcp_side",
        "middle_mcp_flex",
        "middle_pip_flex",
        "middle_dip_flex",
    ),
    "index": (
        "index_mcp_side",
        "index_mcp_flex",
        "index_pip_flex",
        "index_dip_flex",
    ),
    "thumb": ("thumb_cmc_side", "thumb_cmc_flex", "thumb_mcp_flex", "thumb_ip_flex"),
}
ANCHOR_POSE_CANDIDATES: Final[dict[str, tuple[str, ...]]] = {
    "index": (
        "flat_closed",
        "flat_spread",
        "fist",
        "open_index_flex",
        "index_point",
        "index_pinch",
    ),
    "middle": (
        "flat_closed",
        "flat_spread",
        "fist",
        "open_middle_flex",
        "scissors",
        "middle_pinch",
    ),
    "ring": (
        "flat_closed",
        "flat_spread",
        "fist",
        "open_ring_flex",
        "three_finger_extend",
        "ring_pinch",
    ),
    "thumb": (
        "flat_closed",
        "flat_spread",
        "fist",
        "open_thumb_flex",
        "index_pinch",
        "middle_pinch",
        "ring_pinch",
        "thumbs_up",
    ),
}
DEFAULT_ANCHOR_SOFTMAX_TEMPERATURE: Final[float] = 0.2
_EPSILON: Final[float] = 1e-12


@dataclass(frozen=True)
class AnchorPoseRetargeter:
    """Retarget hand features by blending captured pose anchors.

    The target starts from the canonical flat-closed pose target, then each
    finger group is overridden by a softmax-weighted blend over all captured
    pose anchors. Softmax weighting keeps the output continuous as the hand
    moves between anchors; ``softmax_temperature`` controls sharpness (lower
    values track the nearest anchor more closely).
    """

    side: str = "right"
    pose_feature_anchors: Mapping[str, PoseFeatureAnchor] = field(default_factory=dict)
    robot_pose_targets: Mapping[str, RobotPoseTargets] = field(
        default_factory=lambda: default_anchor_robot_pose_targets(),
    )
    softmax_temperature: float = DEFAULT_ANCHOR_SOFTMAX_TEMPERATURE
    enabled_pose_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        if (
            not math.isfinite(self.softmax_temperature)
            or self.softmax_temperature <= 0.0
        ):
            raise ValueError("softmax_temperature must be positive.")

        robot_pose_targets = default_anchor_robot_pose_targets()
        robot_pose_targets.update(dict(self.robot_pose_targets))
        _validate_pose_feature_anchors(self.pose_feature_anchors)
        _validate_robot_pose_targets(robot_pose_targets)
        enabled_pose_names = _normalize_enabled_pose_names(self.enabled_pose_names)
        pose_feature_anchors = _filter_enabled_pose_feature_anchors(
            self.pose_feature_anchors,
            enabled_pose_names=enabled_pose_names,
        )
        object.__setattr__(self, "robot_pose_targets", robot_pose_targets)
        object.__setattr__(self, "enabled_pose_names", enabled_pose_names)
        object.__setattr__(self, "pose_feature_anchors", pose_feature_anchors)

    @classmethod
    def from_calibration(
        cls,
        calibration: HandRetargetCalibration,
        *,
        side: str = "right",
        softmax_temperature: float = DEFAULT_ANCHOR_SOFTMAX_TEMPERATURE,
        enabled_pose_names: tuple[str, ...] | None = None,
    ) -> AnchorPoseRetargeter:
        """Build an anchor-pose retargeter from a saved calibration JSON."""

        robot_pose_targets = default_anchor_robot_pose_targets()
        robot_pose_targets.update(dict(calibration.robot_pose_targets))
        calibrated_anchors = {
            pose_name: _pose_feature_anchor_from_features(
                apply_hand_retarget_calibration(
                    _hand_features_from_pose_anchor(
                        anchor,
                        handedness=calibration.handedness,
                    ),
                    calibration,
                ),
            )
            for pose_name, anchor in calibration.pose_feature_anchors.items()
        }
        return cls(
            side=side,
            pose_feature_anchors=calibrated_anchors,
            robot_pose_targets=robot_pose_targets,
            softmax_temperature=softmax_temperature,
            enabled_pose_names=enabled_pose_names,
        )

    def __call__(self, features: HandFeatures) -> JointTarget:
        """Retarget one feature vector into a timestamped joint target."""

        if not features.valid:
            return JointTarget.empty(timestamp_ns=features.timestamp_ns)

        joints = self._base_joint_targets()
        prefix = f"{self.side}_hand"

        for group in (*ROBOT_FINGER_NAMES, "thumb"):
            group_targets = self._blend_group_targets(group, features)
            for joint_name, value in group_targets.items():
                joints[f"{prefix}.{joint_name}.pos"] = value

        return JointTarget(timestamp_ns=features.timestamp_ns, joints=joints)

    def _base_joint_targets(self) -> dict[str, float]:
        flat_closed = self.robot_pose_targets["flat_closed"]
        prefix = f"{self.side}_hand"
        return {
            f"{prefix}.{joint_name}.pos": flat_closed.target(joint_name)
            for joint_name in flat_closed.joints
        }

    def _blend_group_targets(
        self,
        group: str,
        features: HandFeatures,
    ) -> dict[str, float]:
        weighted_targets = self._softmax_weighted_targets(group, features)
        if not weighted_targets:
            return {}
        return {
            joint_name: _normalized(
                sum(weight * target.target(joint_name) for weight, target in weighted_targets)
            )
            for joint_name in FINGER_JOINT_GROUPS[group]
        }

    def _softmax_weighted_targets(
        self,
        group: str,
        features: HandFeatures,
    ) -> tuple[tuple[float, RobotPoseTargets], ...]:
        candidates: list[tuple[float, str, RobotPoseTargets]] = []
        for pose_name in ANCHOR_POSE_CANDIDATES[group]:
            anchor = self.pose_feature_anchors.get(pose_name)
            target = self.robot_pose_targets.get(pose_name)
            if anchor is None or target is None:
                continue
            distance = (
                _thumb_pose_distance(features, anchor)
                if group == "thumb"
                else _finger_pose_distance(features, anchor, group)
            )
            candidates.append((distance, pose_name, target))
        if not candidates:
            return ()

        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
        min_distance = candidates[0][0]
        if min_distance <= _EPSILON:
            # Exact anchor replays (e.g. calibrate-check) must reproduce
            # their captured pose target without leakage from other anchors.
            return ((1.0, candidates[0][2]),)

        scores = tuple(
            math.exp(-(distance - min_distance) / self.softmax_temperature)
            for distance, _, _ in candidates
        )
        total = sum(scores)
        return tuple(
            (score / total, target)
            for score, (_, _, target) in zip(scores, candidates, strict=True)
        )


def default_anchor_robot_pose_targets() -> dict[str, RobotPoseTargets]:
    """Return default robot targets for all captured anchor-pose names."""

    targets = default_robot_pose_targets()
    flat_closed = targets["flat_closed"]
    fist = targets["fist"]
    flat_spread = targets["flat_spread"]

    targets["open_thumb_flex"] = _pose_with(
        flat_closed,
        _joint_values(fist, FINGER_JOINT_GROUPS["thumb"]),
    )
    for finger in ROBOT_FINGER_NAMES:
        targets[f"open_{finger}_flex"] = _pose_with(
            flat_closed,
            _joint_values(fist, _finger_flex_joints(finger)),
        )

    targets["index_point"] = _pose_with(
        fist,
        _joint_values(flat_closed, FINGER_JOINT_GROUPS["index"]),
    )
    targets["thumbs_up"] = _pose_with(
        flat_closed,
        {
            **_joint_values(fist, FINGER_JOINT_GROUPS["ring"]),
            **_joint_values(fist, FINGER_JOINT_GROUPS["middle"]),
            **_joint_values(fist, FINGER_JOINT_GROUPS["index"]),
            **_joint_values(flat_spread, FINGER_JOINT_GROUPS["thumb"]),
        },
    )
    return targets


def _finger_pose_distance(
    features: HandFeatures,
    anchor: PoseFeatureAnchor,
    finger: str,
) -> float:
    distance = sum(
        _squared(
            features.finger_joint_curl[finger][joint]
            - anchor.finger_joint_curl[finger][joint]
        )
        for joint in FINGER_CURL_JOINT_NAMES
    )
    distance += _squared(features.finger_spread[finger] - anchor.finger_spread[finger])
    distance += _squared(features.pinch[finger] - anchor.pinch[finger])
    return distance


def _thumb_pose_distance(features: HandFeatures, anchor: PoseFeatureAnchor) -> float:
    distance = sum(
        _squared(
            features.finger_joint_curl["thumb"][joint]
            - anchor.finger_joint_curl["thumb"][joint]
        )
        for joint in FINGER_CURL_JOINT_NAMES
    )
    distance += sum(
        _squared(features.thumb_pose[feature] - anchor.thumb_pose[feature])
        for feature in THUMB_POSE_FEATURE_NAMES
    )
    distance += sum(
        _squared(features.pinch[finger] - anchor.pinch[finger])
        for finger in PINCH_FINGER_NAMES
    )
    return distance


def _hand_features_from_pose_anchor(
    anchor: PoseFeatureAnchor,
    *,
    handedness: str,
) -> HandFeatures:
    return HandFeatures(
        timestamp_ns=0,
        handedness=handedness,
        score=1.0,
        finger_curl=anchor.finger_curl,
        finger_joint_curl=anchor.finger_joint_curl,
        finger_spread=anchor.finger_spread,
        pinch=anchor.pinch,
        palm_size=anchor.palm_size,
        thumb_abduction=anchor.thumb_abduction,
        thumb_pose=anchor.thumb_pose,
        valid=True,
    )


def _pose_feature_anchor_from_features(features: HandFeatures) -> PoseFeatureAnchor:
    return PoseFeatureAnchor(
        finger_curl=features.finger_curl,
        finger_joint_curl=features.finger_joint_curl,
        finger_spread=features.finger_spread,
        pinch=features.pinch,
        thumb_abduction=features.thumb_abduction,
        thumb_pose=features.thumb_pose,
        palm_size=features.palm_size,
    )


def _pose_with(
    base: RobotPoseTargets,
    updates: Mapping[str, float],
) -> RobotPoseTargets:
    joints = dict(base.joints)
    joints.update(updates)
    return RobotPoseTargets(joints=joints)


def _joint_values(
    target: RobotPoseTargets,
    joint_names: tuple[str, ...],
) -> dict[str, float]:
    return {joint_name: target.target(joint_name) for joint_name in joint_names}


def _finger_flex_joints(finger: str) -> tuple[str, ...]:
    return (
        f"{finger}_mcp_flex",
        f"{finger}_pip_flex",
        f"{finger}_dip_flex",
    )


def _remap_flex_joints(
    target: RobotPoseTargets,
    *,
    source: str,
    destination: str,
) -> dict[str, float]:
    return {
        f"{destination}_{joint}_flex": target.target(f"{source}_{joint}_flex")
        for joint in FINGER_CURL_JOINT_NAMES
    }


def _normalize_enabled_pose_names(
    enabled_pose_names: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if enabled_pose_names is None:
        return None
    deduped = tuple(dict.fromkeys(enabled_pose_names))
    if not deduped:
        return None
    return deduped


def _filter_enabled_pose_feature_anchors(
    pose_feature_anchors: Mapping[str, PoseFeatureAnchor],
    *,
    enabled_pose_names: tuple[str, ...] | None,
) -> dict[str, PoseFeatureAnchor]:
    anchors = dict(pose_feature_anchors)
    if enabled_pose_names is None:
        return anchors

    missing = tuple(
        pose_name for pose_name in enabled_pose_names if pose_name not in anchors
    )
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "enabled anchor pose(s) missing from pose_feature_anchors: "
            f"{missing_text}."
        )
    return {pose_name: anchors[pose_name] for pose_name in enabled_pose_names}


def _validate_pose_feature_anchors(
    pose_feature_anchors: Mapping[str, PoseFeatureAnchor],
) -> None:
    for pose_name, anchor in pose_feature_anchors.items():
        if not isinstance(anchor, PoseFeatureAnchor):
            raise ValueError(
                f"pose_feature_anchors.{pose_name} must be PoseFeatureAnchor."
            )


def _validate_robot_pose_targets(
    robot_pose_targets: Mapping[str, RobotPoseTargets],
) -> None:
    for pose_name, target in robot_pose_targets.items():
        if not isinstance(target, RobotPoseTargets):
            raise ValueError(f"robot_pose_targets.{pose_name} must be RobotPoseTargets.")


def _squared(value: float) -> float:
    return float(value) * float(value)


def _normalized(value: float) -> float:
    return min(max(float(value), -1.0), 1.0)
