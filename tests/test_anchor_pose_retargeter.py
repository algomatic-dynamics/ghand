import pytest

from ghand.retargeting import (
    AnchorPoseRetargeter,
    CurlCalibration,
    FINGER_CURL_JOINT_NAMES,
    FINGER_NAMES,
    HandFeatures,
    HandRetargetCalibration,
    PINCH_FINGER_NAMES,
    PoseFeatureAnchor,
    RETARGET_CALIBRATION_CAPTURE_POSE_NAMES,
    RETARGET_CALIBRATION_POSES,
    RobotPoseTargets,
    SPREAD_FINGER_NAMES,
    SpreadCalibration,
    PinchCalibration,
    RetargetPoseStats,
    ThumbAbductionCalibration,
    default_anchor_robot_pose_targets,
)
from ghand.hand import JointTarget


def test_anchor_pose_retargeter_returns_empty_target_for_invalid_features() -> None:
    retargeter = AnchorPoseRetargeter(side="right")

    target = retargeter(HandFeatures.empty(timestamp_ns=123))

    assert target == JointTarget.empty(timestamp_ns=123)


def test_anchor_pose_retargeter_without_anchors_uses_flat_closed_baseline() -> None:
    retargeter = AnchorPoseRetargeter(side="right")
    features = _hand_features(index_joint_curl=0.5, index_spread=0.3, index_pinch=0.2)

    target = retargeter(features)

    flat_closed = default_anchor_robot_pose_targets()["flat_closed"]
    assert target.timestamp_ns == features.timestamp_ns
    assert target.joints == {
        f"right_hand.{joint_name}.pos": flat_closed.target(joint_name)
        for joint_name in flat_closed.joints
    }


def test_anchor_pose_retargeter_selects_exact_anchor_for_one_finger_group() -> None:
    robot_pose_targets = default_anchor_robot_pose_targets()
    robot_pose_targets["open_index_flex"] = _target_with(
        robot_pose_targets["open_index_flex"],
        index_mcp_side=0.25,
        index_mcp_flex=0.8,
        index_pip_flex=0.7,
        index_dip_flex=0.6,
    )
    retargeter = AnchorPoseRetargeter(
        side="right",
        pose_feature_anchors={
            "open_index_flex": _pose_anchor(index_joint_curl=1.0),
        },
        robot_pose_targets=robot_pose_targets,
    )

    target = retargeter(_hand_features(index_joint_curl=1.0))

    assert target.joints["right_hand.index_mcp_side.pos"] == pytest.approx(0.25)
    assert target.joints["right_hand.index_mcp_flex.pos"] == pytest.approx(0.8)
    assert target.joints["right_hand.index_pip_flex.pos"] == pytest.approx(0.7)
    assert target.joints["right_hand.index_dip_flex.pos"] == pytest.approx(0.6)
    assert target.joints["right_hand.middle_mcp_flex.pos"] == pytest.approx(0.0)


def test_anchor_pose_retargeter_from_calibration_calibrates_pose_anchors() -> None:
    calibration = HandRetargetCalibration(
        version=1,
        handedness="Right",
        poses={
            pose: RetargetPoseStats(samples=2) for pose in RETARGET_CALIBRATION_POSES
        },
        finger_curl={
            finger: CurlCalibration(open=0.2, closed=0.6) for finger in FINGER_NAMES
        },
        finger_spread={
            finger: SpreadCalibration(neutral=0.0, spread=1.0)
            for finger in SPREAD_FINGER_NAMES
        },
        pinch={
            finger: PinchCalibration(open=0.0, closed=1.0)
            for finger in PINCH_FINGER_NAMES
        },
        thumb_abduction=ThumbAbductionCalibration(closed=0.0, spread=1.0),
        pose_feature_anchors={
            "open_index_flex": _pose_anchor(index_joint_curl=0.6),
        },
    )

    retargeter = AnchorPoseRetargeter.from_calibration(calibration)

    anchor = retargeter.pose_feature_anchors["open_index_flex"]
    assert anchor.finger_joint_curl["index"]["mcp"] == pytest.approx(1.0)
    assert anchor.finger_joint_curl["index"]["pip"] == pytest.approx(1.0)
    assert anchor.finger_joint_curl["index"]["dip"] == pytest.approx(1.0)


def test_anchor_pose_retargeter_rejects_non_positive_softmax_temperature() -> None:
    with pytest.raises(ValueError, match="softmax_temperature"):
        AnchorPoseRetargeter(side="right", softmax_temperature=0.0)


def test_anchor_pose_retargeter_blends_anchors_continuously() -> None:
    retargeter = _two_anchor_retargeter(softmax_temperature=0.1)

    curls = [0.30 + 0.01 * step for step in range(41)]
    values = [
        retargeter(_hand_features(index_joint_curl=curl)).joints[
            "right_hand.index_mcp_flex.pos"
        ]
        for curl in curls
    ]

    # Sweeping between the two anchors must move the target monotonically
    # without discrete jumps, unlike the previous hard top-k selection.
    assert values == sorted(values)
    for previous, current in zip(values[:-1], values[1:], strict=True):
        assert abs(current - previous) < 0.2
    assert values[0] < 0.5 < values[-1]


def test_anchor_pose_retargeter_lower_temperature_tracks_nearest_anchor() -> None:
    features = _hand_features(index_joint_curl=0.7)

    sharp = _two_anchor_retargeter(softmax_temperature=0.01)(features)
    soft = _two_anchor_retargeter(softmax_temperature=1.0)(features)

    sharp_value = sharp.joints["right_hand.index_mcp_flex.pos"]
    soft_value = soft.joints["right_hand.index_mcp_flex.pos"]
    # Nearest anchor (curl=1.0) maps to 1.0; sharper softmax should sit closer.
    assert sharp_value > soft_value
    assert sharp_value == pytest.approx(1.0, abs=0.01)


def _two_anchor_retargeter(*, softmax_temperature: float) -> AnchorPoseRetargeter:
    robot_pose_targets = default_anchor_robot_pose_targets()
    robot_pose_targets["flat_closed"] = _target_with(
        robot_pose_targets["flat_closed"],
        index_mcp_flex=0.0,
    )
    robot_pose_targets["open_index_flex"] = _target_with(
        robot_pose_targets["open_index_flex"],
        index_mcp_flex=1.0,
    )
    return AnchorPoseRetargeter(
        side="right",
        pose_feature_anchors={
            "flat_closed": _pose_anchor(index_joint_curl=0.0),
            "open_index_flex": _pose_anchor(index_joint_curl=1.0),
        },
        robot_pose_targets=robot_pose_targets,
        softmax_temperature=softmax_temperature,
    )


def test_anchor_pose_retargeter_can_limit_enabled_pose_names() -> None:
    retargeter = AnchorPoseRetargeter(
        side="right",
        pose_feature_anchors={
            "flat_closed": _pose_anchor(),
            "open_index_flex": _pose_anchor(index_joint_curl=1.0),
        },
        enabled_pose_names=("open_index_flex",),
    )

    assert tuple(retargeter.pose_feature_anchors) == ("open_index_flex",)


def test_anchor_pose_retargeter_rejects_missing_enabled_pose_name() -> None:
    with pytest.raises(ValueError, match="missing_pose"):
        AnchorPoseRetargeter(
            side="right",
            pose_feature_anchors={"flat_closed": _pose_anchor()},
            enabled_pose_names=("missing_pose",),
        )


def test_default_anchor_robot_pose_targets_cover_capture_pose_registry() -> None:
    targets = default_anchor_robot_pose_targets()

    assert set(RETARGET_CALIBRATION_CAPTURE_POSE_NAMES).issubset(targets)


def _hand_features(
    *,
    index_joint_curl: float = 0.0,
    index_spread: float = 0.0,
    index_pinch: float = 0.0,
) -> HandFeatures:
    finger_joint_curl = {
        finger: {joint: 0.0 for joint in FINGER_CURL_JOINT_NAMES}
        for finger in FINGER_NAMES
    }
    finger_joint_curl["index"] = {
        joint: index_joint_curl for joint in FINGER_CURL_JOINT_NAMES
    }
    return HandFeatures(
        timestamp_ns=123,
        handedness="Right",
        score=0.9,
        finger_curl={
            finger: (index_joint_curl if finger == "index" else 0.0)
            for finger in FINGER_NAMES
        },
        finger_joint_curl=finger_joint_curl,
        finger_spread={
            finger: (index_spread if finger == "index" else 0.0)
            for finger in SPREAD_FINGER_NAMES
        },
        pinch={
            finger: (index_pinch if finger == "index" else 0.0)
            for finger in PINCH_FINGER_NAMES
        },
        palm_size=0.3,
        thumb_abduction=0.0,
    )


def _pose_anchor(
    *,
    index_joint_curl: float = 0.0,
    index_spread: float = 0.0,
    index_pinch: float = 0.0,
) -> PoseFeatureAnchor:
    features = _hand_features(
        index_joint_curl=index_joint_curl,
        index_spread=index_spread,
        index_pinch=index_pinch,
    )
    return PoseFeatureAnchor(
        finger_curl=features.finger_curl,
        finger_joint_curl=features.finger_joint_curl,
        finger_spread=features.finger_spread,
        pinch=features.pinch,
        thumb_abduction=features.thumb_abduction,
        thumb_pose=features.thumb_pose,
        palm_size=features.palm_size,
    )


def _target_with(base: RobotPoseTargets, **updates: float) -> RobotPoseTargets:
    joints = dict(base.joints)
    joints.update(updates)
    return RobotPoseTargets(joints=joints)
