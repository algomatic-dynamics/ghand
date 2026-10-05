import pytest

from ghand.retargeting import (
    FINGER_CURL_JOINT_NAMES,
    FINGER_NAMES,
    HandFeatures,
    HandRetargetCalibration,
    PINCH_FINGER_NAMES,
    PoseFeatureAnchor,
    SPREAD_FINGER_NAMES,
    diagnose_hand_retarget_calibration,
)


def test_diagnose_hand_retarget_calibration_reports_anchor_pose_rows() -> None:
    calibration = _calibration_with_pose_feature_anchors()

    rows = diagnose_hand_retarget_calibration(
        calibration,
        side="right",
    )

    rows_by_pose = {row.pose_name: row for row in rows}
    assert rows_by_pose["flat_closed"].retargeter == "anchor-pose"
    assert rows_by_pose["flat_closed"].all_max_abs_error == pytest.approx(0.0)
    assert rows_by_pose["open_index_flex"].affected_joint_count == 4
    assert rows_by_pose["open_index_flex"].affected_max_abs_error == pytest.approx(0.0)


def test_diagnose_hand_retarget_calibration_rejects_missing_pose_feature_anchors() -> None:
    with pytest.raises(ValueError, match="pose_feature_anchors"):
        diagnose_hand_retarget_calibration(
            HandRetargetCalibration.identity(handedness="Right"),
        )


def test_diagnose_hand_retarget_calibration_rejects_unknown_enabled_anchor() -> None:
    with pytest.raises(ValueError, match="missing_pose"):
        diagnose_hand_retarget_calibration(
            _calibration_with_pose_feature_anchors(),
            enabled_pose_names=("missing_pose",),
        )


def _calibration_with_pose_feature_anchors() -> HandRetargetCalibration:
    calibration = HandRetargetCalibration.identity(handedness="Right")
    data = calibration.to_dict()
    data["poses"]["flat_closed"] = {"samples": 2}
    data["poses"]["open_index_flex"] = {"samples": 2}
    data["pose_feature_anchors"] = {
        "flat_closed": _pose_anchor().to_dict(),
        "open_index_flex": _pose_anchor(index_joint_curl=1.0).to_dict(),
    }
    return HandRetargetCalibration.from_dict(data)


def _pose_anchor(*, index_joint_curl: float = 0.0) -> PoseFeatureAnchor:
    features = _hand_features(index_joint_curl=index_joint_curl)
    return PoseFeatureAnchor(
        finger_curl=features.finger_curl,
        finger_joint_curl=features.finger_joint_curl,
        finger_spread=features.finger_spread,
        pinch=features.pinch,
        thumb_abduction=features.thumb_abduction,
        thumb_pose=features.thumb_pose,
        palm_size=features.palm_size,
    )


def _hand_features(*, index_joint_curl: float = 0.0) -> HandFeatures:
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
        finger_spread={finger: 0.0 for finger in SPREAD_FINGER_NAMES},
        pinch={finger: 0.0 for finger in PINCH_FINGER_NAMES},
        palm_size=0.3,
        thumb_abduction=0.0,
    )
