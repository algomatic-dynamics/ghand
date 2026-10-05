"""Diagnostics for captured hand-retargeting calibration anchors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from ghand.retargeting.anchor_pose import (
    FINGER_JOINT_GROUPS,
    AnchorPoseRetargeter,
    default_anchor_robot_pose_targets,
)
from ghand.retargeting.calibration import (
    HandRetargetCalibration,
    PoseFeatureAnchor,
    RobotPoseTargets,
    apply_hand_retarget_calibration,
)
from ghand.retargeting.features import HandFeatures
from ghand.retargeting.poses import RETARGET_CALIBRATION_CAPTURE_POSES
from ghand.hand import JointTarget


@dataclass(frozen=True)
class RetargetPoseDiagnostic:
    """One pose-row from retargeting calibration diagnostics."""

    pose_name: str
    pose_group: str
    retargeter: str
    samples: int
    affected_joint_count: int
    affected_mean_abs_error: float
    affected_max_abs_error: float
    all_mean_abs_error: float
    all_max_abs_error: float
    worst_joint: str
    worst_joint_abs_error: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def diagnose_hand_retarget_calibration(
    calibration: HandRetargetCalibration,
    *,
    side: str = "right",
    enabled_pose_names: tuple[str, ...] | None = None,
) -> tuple[RetargetPoseDiagnostic, ...]:
    """Compare captured pose anchors against their robot-pose targets."""

    if not calibration.pose_feature_anchors:
        raise ValueError("retarget calibration does not include pose_feature_anchors.")

    robot_pose_targets = default_anchor_robot_pose_targets()
    robot_pose_targets.update(dict(calibration.robot_pose_targets))
    retargeter = AnchorPoseRetargeter.from_calibration(
        calibration,
        side=side,
        enabled_pose_names=enabled_pose_names,
    )

    rows: list[RetargetPoseDiagnostic] = []
    for pose in RETARGET_CALIBRATION_CAPTURE_POSES:
        anchor = calibration.pose_feature_anchors.get(pose.name)
        expected = robot_pose_targets.get(pose.name)
        if anchor is None or expected is None:
            continue

        features = apply_hand_retarget_calibration(
            _hand_features_from_pose_anchor(anchor, handedness=calibration.handedness),
            calibration,
        )
        target = retargeter(features)
        rows.append(
            _pose_diagnostic_row(
                pose_name=pose.name,
                pose_group=pose.group,
                affected_fingers=pose.affected_fingers,
                retargeter="anchor-pose",
                samples=calibration.poses.get(pose.name, calibration.poses["flat_closed"]).samples,
                side=side,
                target=target,
                expected=expected,
            )
        )
    return tuple(rows)


def _pose_diagnostic_row(
    *,
    pose_name: str,
    pose_group: str,
    affected_fingers: tuple[str, ...],
    retargeter: str,
    samples: int,
    side: str,
    target: JointTarget,
    expected: RobotPoseTargets,
) -> RetargetPoseDiagnostic:
    all_errors = _target_errors(side=side, target=target, expected=expected)
    affected_joints = _affected_joint_names(affected_fingers)
    affected_errors = {
        joint_name: all_errors[joint_name]
        for joint_name in affected_joints
        if joint_name in all_errors
    }
    if not affected_errors:
        affected_errors = all_errors

    worst_joint, worst_joint_abs_error = max(
        all_errors.items(),
        key=lambda item: (item[1], item[0]),
    )
    return RetargetPoseDiagnostic(
        pose_name=pose_name,
        pose_group=pose_group,
        retargeter=retargeter,
        samples=samples,
        affected_joint_count=len(affected_errors),
        affected_mean_abs_error=_mean_abs_error(tuple(affected_errors.values())),
        affected_max_abs_error=max(affected_errors.values()),
        all_mean_abs_error=_mean_abs_error(tuple(all_errors.values())),
        all_max_abs_error=max(all_errors.values()),
        worst_joint=worst_joint,
        worst_joint_abs_error=worst_joint_abs_error,
    )


def _target_errors(
    *,
    side: str,
    target: JointTarget,
    expected: RobotPoseTargets,
) -> dict[str, float]:
    prefix = f"{side}_hand"
    return {
        joint_name: abs(
            float(target.joints.get(f"{prefix}.{joint_name}.pos", 0.0))
            - expected.target(joint_name)
        )
        for joint_name in expected.joints
    }


def _affected_joint_names(affected_fingers: tuple[str, ...]) -> tuple[str, ...]:
    joints: list[str] = []
    for finger in affected_fingers:
        group = FINGER_JOINT_GROUPS.get(finger)
        if group is not None:
            joints.extend(group)
    return tuple(dict.fromkeys(joints))


def _mean_abs_error(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


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
