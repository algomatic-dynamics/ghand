import pytest

from ghand.hand import (
    CurrentLimitTarget,
    JointTarget,
    make_hand_joint_names,
    PositionCurrentAction,
    RadianJointTarget,
)
from ghand.retargeting import (
    FINGER_CURL_JOINT_NAMES,
    FINGER_NAMES,
    PINCH_FINGER_NAMES,
    SPREAD_FINGER_NAMES,
    THUMB_POSE_FEATURE_NAMES,
    HandFeatures,
)


def test_hand_features_preserve_validated_metric_maps() -> None:
    features = HandFeatures(
        timestamp_ns=123,
        handedness="Right",
        score=0.95,
        finger_curl={finger: 0.25 for finger in FINGER_NAMES},
        finger_spread={finger: 0.0 for finger in SPREAD_FINGER_NAMES},
        pinch={finger: 0.75 for finger in PINCH_FINGER_NAMES},
        palm_size=0.2,
        valid=True,
    )

    assert features.finger_curl["index"] == 0.25
    assert features.finger_joint_curl["index"] == {
        joint: 0.25 for joint in FINGER_CURL_JOINT_NAMES
    }
    assert features.finger_spread["pinky"] == 0.0
    assert features.pinch["middle"] == 0.75
    assert tuple(features.thumb_pose) == THUMB_POSE_FEATURE_NAMES
    assert features.thumb_pose["index_pinch"] == 0.75
    assert features.to_dict()["handedness"] == "Right"


def test_hand_features_reject_missing_or_out_of_range_metrics() -> None:
    with pytest.raises(ValueError, match="finger_curl keys"):
        HandFeatures(
            timestamp_ns=123,
            handedness="Right",
            score=0.95,
            finger_curl={"index": 0.25},
            finger_spread={finger: 0.0 for finger in SPREAD_FINGER_NAMES},
            pinch={finger: 0.75 for finger in PINCH_FINGER_NAMES},
            palm_size=0.2,
        )

    with pytest.raises(ValueError, match="finger_spread"):
        HandFeatures(
            timestamp_ns=123,
            handedness="Right",
            score=0.95,
            finger_curl={finger: 0.25 for finger in FINGER_NAMES},
            finger_spread={finger: 2.0 for finger in SPREAD_FINGER_NAMES},
            pinch={finger: 0.75 for finger in PINCH_FINGER_NAMES},
            palm_size=0.2,
        )


def test_empty_hand_features_are_invalid_and_zeroed() -> None:
    features = HandFeatures.empty(timestamp_ns=456)

    assert features.valid is False
    assert features.handedness == "Unknown"
    assert all(value == 0.0 for value in features.finger_curl.values())
    assert all(
        value == 0.0
        for joint_curls in features.finger_joint_curl.values()
        for value in joint_curls.values()
    )
    assert all(value == 0.0 for value in features.finger_spread.values())
    assert all(value == 0.0 for value in features.pinch.values())
    assert all(value == 0.0 for value in features.thumb_pose.values())


def test_joint_target_validates_hand_joint_names_and_values() -> None:
    target = JointTarget(
        timestamp_ns=123,
        joints={
            "right_hand.index_mcp_flex.pos": 0.5,
            "right_hand.index_pip_flex.pos": 1.0,
        },
    )

    assert target.joints["right_hand.index_mcp_flex.pos"] == 0.5
    assert target.to_vector(
        (
            "right_hand.index_mcp_flex.pos",
            "right_hand.middle_mcp_flex.pos",
        )
    ) == (0.5, 0.0)


def test_joint_target_rejects_unknown_joint_or_out_of_range_value() -> None:
    with pytest.raises(ValueError, match="Unknown hand joint"):
        JointTarget(timestamp_ns=123, joints={"arm.elbow.pos": 0.0})

    with pytest.raises(ValueError, match="normalized range"):
        JointTarget(timestamp_ns=123, joints={"right_hand.index_mcp_flex.pos": 1.1})


def test_joint_target_empty_and_neutral_constructors() -> None:
    empty = JointTarget.empty(timestamp_ns=123)
    neutral = JointTarget.neutral(timestamp_ns=123, side="right")

    assert empty.joints == {}
    assert tuple(neutral.joints) == make_hand_joint_names("right")
    assert all(value == 0.0 for value in neutral.joints.values())


def test_current_limit_target_validates_hand_joint_names_and_values() -> None:
    target = CurrentLimitTarget(
        timestamp_ns=123,
        currents_ma={
            "right_hand.index_mcp_flex.pos": 80.0,
            "right_hand.index_pip_flex.pos": 60.0,
        },
    )

    assert target.currents_ma["right_hand.index_mcp_flex.pos"] == 80.0
    assert target.to_vector(
        (
            "right_hand.index_mcp_flex.pos",
            "right_hand.middle_mcp_flex.pos",
        )
    ) == (80.0, 0.0)


def test_current_limit_target_rejects_unknown_joint_or_negative_value() -> None:
    with pytest.raises(ValueError, match="Unknown hand joint current limit"):
        CurrentLimitTarget(timestamp_ns=123, currents_ma={"arm.elbow.pos": 0.0})

    with pytest.raises(ValueError, match="current limit"):
        CurrentLimitTarget(
            timestamp_ns=123,
            currents_ma={"right_hand.index_mcp_flex.pos": -1.0},
        )


def test_current_limit_target_empty_and_constant_constructors() -> None:
    empty = CurrentLimitTarget.empty(timestamp_ns=123)
    constant = CurrentLimitTarget.constant(
        timestamp_ns=123,
        side="right",
        current_limit_ma=75.0,
    )

    assert empty.currents_ma == {}
    assert tuple(constant.currents_ma) == make_hand_joint_names("right")
    assert all(value == 75.0 for value in constant.currents_ma.values())


def test_position_current_action_requires_matching_timestamps() -> None:
    position = RadianJointTarget(
        timestamp_ns=123,
        joints={"right_hand.index_mcp_flex.pos": 0.5},
    )
    current_limit = CurrentLimitTarget(
        timestamp_ns=123,
        currents_ma={"right_hand.index_mcp_flex.pos": 80.0},
    )

    action = PositionCurrentAction(position=position, current_limit=current_limit)

    assert action.timestamp_ns == 123
    assert action.position.joints["right_hand.index_mcp_flex.pos"] == 0.5
    assert action.current_limit.currents_ma["right_hand.index_mcp_flex.pos"] == 80.0

    with pytest.raises(ValueError, match="timestamps"):
        PositionCurrentAction(
            position=position,
            current_limit=CurrentLimitTarget.empty(timestamp_ns=124),
        )
