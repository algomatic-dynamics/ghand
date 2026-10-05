import json

import pytest

from ghand.retargeting import (
    DEFAULT_RETARGET_CALIBRATION_RESOURCE,
    FINGER_CURL_JOINT_NAMES,
    FINGER_NAMES,
    PINCH_FINGER_NAMES,
    RETARGET_CALIBRATION_POSES,
    RETARGET_CALIBRATION_CAPTURE_POSE_NAMES,
    RETARGET_CALIBRATION_SCHEMA_VERSION,
    ROBOT_TARGET_JOINTS,
    SPREAD_FINGER_NAMES,
    THUMB_POSE_FEATURE_NAMES,
    CurlCalibration,
    HandFeatureCalibrator,
    HandFeatures,
    HandRetargetCalibration,
    PinchCalibration,
    PoseFeatureAnchor,
    RobotPoseTargets,
    RetargetPoseStats,
    SpreadCalibration,
    ThumbAbductionCalibration,
    ThumbJointTargets,
    ThumbPoseAnchor,
    apply_hand_retarget_calibration,
    default_robot_pose_targets,
    default_thumb_pose_anchors,
    default_thumb_targets,
    load_default_hand_retarget_calibration,
    load_hand_retarget_calibration,
    save_hand_retarget_calibration,
)


def test_hand_retarget_calibration_identity_has_full_schema() -> None:
    calibration = HandRetargetCalibration.identity(handedness="Right")

    assert calibration.version == RETARGET_CALIBRATION_SCHEMA_VERSION
    assert calibration.handedness == "Right"
    assert tuple(calibration.poses) == RETARGET_CALIBRATION_POSES
    assert tuple(calibration.finger_curl) == FINGER_NAMES
    assert tuple(calibration.finger_joint_curl) == FINGER_NAMES
    assert tuple(calibration.finger_joint_curl["index"]) == FINGER_CURL_JOINT_NAMES
    assert tuple(calibration.finger_spread) == SPREAD_FINGER_NAMES
    assert tuple(calibration.pinch) == PINCH_FINGER_NAMES
    assert calibration.finger_curl["index"] == CurlCalibration(open=0.0, closed=1.0)
    assert calibration.finger_joint_curl["index"]["mcp"] == CurlCalibration(
        open=0.0, closed=1.0
    )
    assert calibration.finger_spread["index"] == SpreadCalibration(
        neutral=0.0, spread=1.0
    )
    assert calibration.pinch["index"] == PinchCalibration(open=0.0, closed=1.0)
    assert calibration.thumb_abduction == ThumbAbductionCalibration(
        closed=0.0, spread=1.0
    )
    assert calibration.robot_pose_targets == default_robot_pose_targets()
    assert calibration.thumb_targets == default_thumb_targets()
    assert calibration.thumb_pose_anchors == default_thumb_pose_anchors()
    assert calibration.pose_feature_anchors == {}


def test_default_hand_retarget_calibration_is_packaged_valid_baseline() -> None:
    calibration = load_default_hand_retarget_calibration()

    assert DEFAULT_RETARGET_CALIBRATION_RESOURCE == (
        "calibration/default_retarget_calibration_v1.json"
    )
    assert calibration.version == RETARGET_CALIBRATION_SCHEMA_VERSION
    assert calibration.handedness == "Right"
    assert set(calibration.poses) == set(RETARGET_CALIBRATION_CAPTURE_POSE_NAMES)
    assert all(stats.samples >= 0 for stats in calibration.poses.values())
    assert set(calibration.finger_curl) == set(FINGER_NAMES)
    assert set(calibration.finger_joint_curl) == set(FINGER_NAMES)
    assert set(calibration.finger_spread) == set(SPREAD_FINGER_NAMES)
    assert set(calibration.pinch) == set(PINCH_FINGER_NAMES)
    assert calibration.thumb_abduction.spread != calibration.thumb_abduction.closed
    assert set(RETARGET_CALIBRATION_POSES).issubset(calibration.robot_pose_targets)
    assert set(calibration.robot_pose_targets["flat_closed"].joints) == set(
        ROBOT_TARGET_JOINTS
    )
    assert (
        calibration.robot_pose_targets["flat_closed"]
        == default_robot_pose_targets()["flat_closed"]
    )
    assert calibration.robot_pose_targets["fist"].target(
        "index_mcp_flex"
    ) == pytest.approx(-70.0 / 180.0)
    assert calibration.robot_pose_targets["scissors"].target(
        "middle_mcp_side"
    ) == pytest.approx(-20.0 / 180.0)
    assert set(RETARGET_CALIBRATION_POSES).issubset(calibration.thumb_targets)
    assert set(calibration.thumb_pose_anchors) == set(
        RETARGET_CALIBRATION_CAPTURE_POSE_NAMES
    )
    assert set(calibration.thumb_pose_anchors["flat_closed"].features) == set(
        THUMB_POSE_FEATURE_NAMES
    )
    assert set(calibration.pose_feature_anchors) == set(
        RETARGET_CALIBRATION_CAPTURE_POSE_NAMES
    )


def test_hand_retarget_calibration_round_trips_through_dict_and_json(tmp_path) -> None:
    calibration = HandRetargetCalibration(
        version=RETARGET_CALIBRATION_SCHEMA_VERSION,
        handedness="Right",
        poses={
            pose: RetargetPoseStats(samples=30) for pose in RETARGET_CALIBRATION_POSES
        },
        finger_curl={
            finger: CurlCalibration(open=0.1, closed=0.8) for finger in FINGER_NAMES
        },
        finger_spread={
            finger: SpreadCalibration(neutral=0.1, spread=-0.7)
            for finger in SPREAD_FINGER_NAMES
        },
        pinch={
            finger: PinchCalibration(open=0.0, closed=0.9)
            for finger in PINCH_FINGER_NAMES
        },
    )

    from_dict = HandRetargetCalibration.from_dict(calibration.to_dict())
    path = tmp_path / "right_hand_retarget_calibration.json"
    save_hand_retarget_calibration(calibration, path)
    loaded = load_hand_retarget_calibration(path)

    assert from_dict == calibration
    assert loaded == calibration
    assert json.loads(path.read_text(encoding="utf-8")) == calibration.to_dict()


def test_hand_retarget_calibration_accepts_legacy_json_without_pose_target_fields() -> (
    None
):
    data = HandRetargetCalibration.identity().to_dict()
    del data["finger_joint_curl"]
    del data["thumb_abduction"]
    del data["robot_pose_targets"]
    del data["thumb_targets"]
    del data["thumb_pose_anchors"]

    calibration = HandRetargetCalibration.from_dict(data)

    assert calibration.thumb_abduction == ThumbAbductionCalibration(
        closed=0.0, spread=1.0
    )
    assert calibration.finger_joint_curl == {
        finger: {
            joint: calibration.finger_curl[finger] for joint in FINGER_CURL_JOINT_NAMES
        }
        for finger in FINGER_NAMES
    }
    assert calibration.robot_pose_targets == default_robot_pose_targets()
    assert calibration.thumb_targets == default_thumb_targets()
    assert calibration.thumb_pose_anchors == default_thumb_pose_anchors()


def test_hand_retarget_calibration_round_trips_thumb_pose_anchors() -> None:
    anchors = {
        pose: ThumbPoseAnchor(
            features={
                feature_name: index / 10
                for index, feature_name in enumerate(THUMB_POSE_FEATURE_NAMES)
            }
        )
        for pose in RETARGET_CALIBRATION_POSES
    }
    calibration = HandRetargetCalibration(
        version=RETARGET_CALIBRATION_SCHEMA_VERSION,
        handedness="Right",
        poses={
            pose: RetargetPoseStats(samples=30) for pose in RETARGET_CALIBRATION_POSES
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
        thumb_pose_anchors=anchors,
    )

    loaded = HandRetargetCalibration.from_dict(calibration.to_dict())

    assert loaded.thumb_pose_anchors == anchors


def test_hand_retarget_calibration_round_trips_pose_feature_anchors() -> None:
    anchor = PoseFeatureAnchor(
        finger_curl={finger: 0.25 for finger in FINGER_NAMES},
        finger_joint_curl={
            finger: {joint: 0.25 for joint in FINGER_CURL_JOINT_NAMES}
            for finger in FINGER_NAMES
        },
        finger_spread={finger: 0.0 for finger in SPREAD_FINGER_NAMES},
        pinch={finger: 0.1 for finger in PINCH_FINGER_NAMES},
        thumb_abduction=0.2,
        thumb_pose={feature: 0.0 for feature in THUMB_POSE_FEATURE_NAMES},
        palm_size=0.3,
    )
    calibration = HandRetargetCalibration.identity()
    updated = HandRetargetCalibration.from_dict(
        {
            **calibration.to_dict(),
            "pose_feature_anchors": {"open_index_flex": anchor.to_dict()},
        }
    )
    loaded = HandRetargetCalibration.from_dict(updated.to_dict())

    assert loaded.pose_feature_anchors["open_index_flex"] == anchor


def test_hand_retarget_calibration_backfills_new_pose_for_legacy_json() -> None:
    data = HandRetargetCalibration.identity().to_dict()
    del data["poses"]["scissors"]
    del data["robot_pose_targets"]["scissors"]
    del data["thumb_targets"]["scissors"]
    del data["thumb_pose_anchors"]["scissors"]

    calibration = HandRetargetCalibration.from_dict(data)

    assert calibration.poses["scissors"] == RetargetPoseStats(samples=0)
    assert (
        calibration.robot_pose_targets["scissors"]
        == default_robot_pose_targets()["scissors"]
    )
    assert calibration.thumb_targets["scissors"] == default_thumb_targets()["scissors"]
    assert (
        calibration.thumb_pose_anchors["scissors"]
        == default_thumb_pose_anchors()["scissors"]
    )


def test_hand_retarget_calibration_rejects_unsupported_version() -> None:
    data = HandRetargetCalibration.identity().to_dict()
    data["version"] = 999

    with pytest.raises(ValueError, match="Unsupported retarget calibration version"):
        HandRetargetCalibration.from_dict(data)


def test_hand_retarget_calibration_rejects_missing_feature_keys() -> None:
    data = HandRetargetCalibration.identity().to_dict()
    del data["finger_curl"]["index"]

    with pytest.raises(ValueError, match="finger_curl keys invalid"):
        HandRetargetCalibration.from_dict(data)


def test_hand_retarget_calibration_rejects_degenerate_ranges() -> None:
    with pytest.raises(ValueError, match="curl.closed"):
        CurlCalibration(open=0.5, closed=0.5)

    with pytest.raises(ValueError, match="spread.spread"):
        SpreadCalibration(neutral=0.2, spread=0.2)

    with pytest.raises(ValueError, match="pinch.closed"):
        PinchCalibration(open=0.9, closed=0.1)

    with pytest.raises(ValueError, match="thumb_abduction.spread"):
        ThumbAbductionCalibration(closed=0.2, spread=0.2)

    with pytest.raises(ValueError, match="thumb_targets.thumb_mcp_flex"):
        ThumbJointTargets(
            thumb_cmc_side=0.0,
            thumb_cmc_flex=0.0,
            thumb_mcp_flex=2.0,
            thumb_ip_flex=0.0,
        )

    with pytest.raises(ValueError, match="thumb_pose_anchors"):
        ThumbPoseAnchor(features={"lateral": 2.0})

    invalid_robot_pose = default_robot_pose_targets()["flat_closed"].to_dict()
    invalid_robot_pose["index_mcp_flex"] = 2.0
    with pytest.raises(ValueError, match="robot_pose_targets.index_mcp_flex"):
        RobotPoseTargets(joints=invalid_robot_pose)


def test_apply_hand_retarget_calibration_identity_preserves_valid_features() -> None:
    features = _hand_features(
        finger_curl={finger: index / 10 for index, finger in enumerate(FINGER_NAMES)},
        finger_spread={
            "index": -0.5,
            "middle": 0.0,
            "ring": 0.25,
            "pinky": 0.75,
        },
        pinch={"index": 0.2, "middle": 0.4, "ring": 0.6},
    )

    calibrated = apply_hand_retarget_calibration(
        features,
        HandRetargetCalibration.identity(handedness="Right"),
    )

    assert calibrated == features


def test_apply_hand_retarget_calibration_rescales_and_clamps_raw_features() -> None:
    calibration = HandRetargetCalibration(
        version=RETARGET_CALIBRATION_SCHEMA_VERSION,
        handedness="Right",
        poses={
            pose: RetargetPoseStats(samples=30) for pose in RETARGET_CALIBRATION_POSES
        },
        finger_curl={
            finger: CurlCalibration(open=0.2, closed=0.6) for finger in FINGER_NAMES
        },
        finger_spread={
            finger: SpreadCalibration(neutral=0.2, spread=-0.6)
            for finger in SPREAD_FINGER_NAMES
        },
        pinch={
            finger: PinchCalibration(open=0.1, closed=0.9)
            for finger in PINCH_FINGER_NAMES
        },
    )
    features = _hand_features(
        finger_curl={
            "thumb": 0.1,
            "index": 0.4,
            "middle": 0.6,
            "ring": 0.8,
            "pinky": 0.2,
        },
        finger_spread={
            "index": 0.2,
            "middle": -0.2,
            "ring": -0.6,
            "pinky": 1.0,
        },
        pinch={
            "index": 0.1,
            "middle": 0.5,
            "ring": 1.0,
        },
    )

    calibrated = apply_hand_retarget_calibration(features, calibration)

    assert calibrated.finger_curl == pytest.approx(
        {
            "thumb": 0.0,
            "index": 0.5,
            "middle": 1.0,
            "ring": 1.0,
            "pinky": 0.0,
        }
    )
    assert calibrated.finger_spread == pytest.approx(
        {
            "index": 0.0,
            "middle": 0.5,
            "ring": 1.0,
            "pinky": -1.0,
        }
    )
    assert calibrated.pinch == pytest.approx(
        {
            "index": 0.0,
            "middle": 0.5,
            "ring": 1.0,
        }
    )
    assert calibrated.timestamp_ns == features.timestamp_ns
    assert calibrated.handedness == features.handedness
    assert calibrated.score == features.score
    assert calibrated.palm_size == features.palm_size
    assert calibrated.thumb_abduction == features.thumb_abduction
    assert calibrated.thumb_pose == features.thumb_pose
    assert calibrated.valid is True


def test_apply_hand_retarget_calibration_rescales_thumb_abduction() -> None:
    calibration = HandRetargetCalibration(
        version=RETARGET_CALIBRATION_SCHEMA_VERSION,
        handedness="Right",
        poses={
            pose: RetargetPoseStats(samples=30) for pose in RETARGET_CALIBRATION_POSES
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
        thumb_abduction=ThumbAbductionCalibration(closed=0.2, spread=0.7),
    )

    calibrated = apply_hand_retarget_calibration(
        _hand_features(),
        calibration,
    )

    assert calibrated.thumb_abduction == pytest.approx(0.4)


def test_apply_hand_retarget_calibration_rescales_joint_curl_independently() -> None:
    calibration = HandRetargetCalibration(
        version=RETARGET_CALIBRATION_SCHEMA_VERSION,
        handedness="Right",
        poses={
            pose: RetargetPoseStats(samples=30) for pose in RETARGET_CALIBRATION_POSES
        },
        finger_curl={
            finger: CurlCalibration(open=0.0, closed=1.0) for finger in FINGER_NAMES
        },
        finger_joint_curl={
            finger: {
                joint: CurlCalibration(open=0.2, closed=0.6)
                for joint in FINGER_CURL_JOINT_NAMES
            }
            for finger in FINGER_NAMES
        },
        finger_spread={
            finger: SpreadCalibration(neutral=0.0, spread=1.0)
            for finger in SPREAD_FINGER_NAMES
        },
        pinch={
            finger: PinchCalibration(open=0.0, closed=1.0)
            for finger in PINCH_FINGER_NAMES
        },
    )

    calibrated = apply_hand_retarget_calibration(
        _hand_features(
            finger_joint_curl={
                finger: {"mcp": 0.2, "pip": 0.4, "dip": 0.6} for finger in FINGER_NAMES
            }
        ),
        calibration,
    )

    assert calibrated.finger_joint_curl["index"] == pytest.approx(
        {
            "mcp": 0.0,
            "pip": 0.5,
            "dip": 1.0,
        }
    )


def test_apply_hand_retarget_calibration_ignores_degenerate_spread_range() -> None:
    calibration = HandRetargetCalibration(
        version=RETARGET_CALIBRATION_SCHEMA_VERSION,
        handedness="Right",
        poses={
            pose: RetargetPoseStats(samples=30) for pose in RETARGET_CALIBRATION_POSES
        },
        finger_curl={
            finger: CurlCalibration(open=0.0, closed=1.0) for finger in FINGER_NAMES
        },
        finger_spread={
            finger: SpreadCalibration(neutral=0.2, spread=0.22)
            for finger in SPREAD_FINGER_NAMES
        },
        pinch={
            finger: PinchCalibration(open=0.0, closed=1.0)
            for finger in PINCH_FINGER_NAMES
        },
    )

    calibrated = apply_hand_retarget_calibration(
        _hand_features(finger_spread={finger: 1.0 for finger in SPREAD_FINGER_NAMES}),
        calibration,
    )

    assert calibrated.finger_spread == {finger: 0.0 for finger in SPREAD_FINGER_NAMES}


def test_hand_feature_calibrator_calls_apply_layer() -> None:
    features = _hand_features()
    calibrator = HandFeatureCalibrator(HandRetargetCalibration.identity())

    assert calibrator(features) == apply_hand_retarget_calibration(
        features,
        HandRetargetCalibration.identity(),
    )


def test_apply_hand_retarget_calibration_keeps_invalid_features_zeroed() -> None:
    calibration = HandRetargetCalibration(
        version=RETARGET_CALIBRATION_SCHEMA_VERSION,
        handedness="Right",
        poses={
            pose: RetargetPoseStats(samples=30) for pose in RETARGET_CALIBRATION_POSES
        },
        finger_curl={
            finger: CurlCalibration(open=0.2, closed=0.6) for finger in FINGER_NAMES
        },
        finger_spread={
            finger: SpreadCalibration(neutral=0.2, spread=-0.6)
            for finger in SPREAD_FINGER_NAMES
        },
        pinch={
            finger: PinchCalibration(open=0.1, closed=0.9)
            for finger in PINCH_FINGER_NAMES
        },
    )
    features = HandFeatures.empty(timestamp_ns=123, handedness="Right")

    calibrated = apply_hand_retarget_calibration(features, calibration)

    assert calibrated == features


def _hand_features(
    *,
    finger_curl: dict[str, float] | None = None,
    finger_joint_curl: dict[str, dict[str, float]] | None = None,
    finger_spread: dict[str, float] | None = None,
    pinch: dict[str, float] | None = None,
    thumb_pose: dict[str, float] | None = None,
) -> HandFeatures:
    return HandFeatures(
        timestamp_ns=123,
        handedness="Right",
        score=0.9,
        finger_curl=finger_curl or {finger: 0.5 for finger in FINGER_NAMES},
        finger_joint_curl=finger_joint_curl,
        finger_spread=finger_spread or {finger: 0.0 for finger in SPREAD_FINGER_NAMES},
        pinch=pinch or {finger: 0.5 for finger in PINCH_FINGER_NAMES},
        palm_size=0.3,
        thumb_abduction=0.4,
        thumb_pose=thumb_pose,
    )
