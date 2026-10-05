from ghand.retargeting import (
    THUMB_POSE_FEATURE_NAMES,
    extract_hand_features,
    features_from_detection,
)
from ghand.teleop.hand_pose import HandDetection, HandFrame, Landmark


def test_extract_hand_features_returns_empty_without_hands() -> None:
    features = extract_hand_features(
        HandFrame(timestamp_ns=123, image_width=640, image_height=480, hands=())
    )

    assert features.valid is False
    assert features.timestamp_ns == 123
    assert features.handedness == "Unknown"


def test_extract_hand_features_selects_preferred_handedness() -> None:
    left = _detection("Left", score=0.99)
    right = _detection("Right", score=0.75)
    frame = HandFrame(
        timestamp_ns=123,
        image_width=640,
        image_height=480,
        hands=(left, right),
    )

    features = extract_hand_features(frame, preferred_handedness="Right")

    assert features.valid is True
    assert features.handedness == "Right"
    assert features.score == 0.75


def test_extract_hand_features_falls_back_to_highest_score() -> None:
    left = _detection("Left", score=0.8)
    right = _detection("Right", score=0.7)
    frame = HandFrame(
        timestamp_ns=123,
        image_width=640,
        image_height=480,
        hands=(left, right),
    )

    features = extract_hand_features(frame, preferred_handedness="Unknown")

    assert features.handedness == "Left"
    assert features.score == 0.8


def test_features_from_detection_reports_more_curl_for_bent_finger() -> None:
    open_features = features_from_detection(_detection("Right"), timestamp_ns=123)
    curled_features = features_from_detection(
        _detection("Right", curled_fingers={"index"}),
        timestamp_ns=123,
    )

    assert open_features.valid is True
    assert open_features.finger_curl["index"] < 0.1
    assert (
        curled_features.finger_curl["index"] > open_features.finger_curl["index"] + 0.25
    )
    assert curled_features.finger_joint_curl["index"]["pip"] > 0.2
    assert curled_features.finger_joint_curl["index"]["dip"] > 0.2


def test_features_from_detection_reports_mcp_flexion_as_curl() -> None:
    open_features = features_from_detection(_detection("Right"), timestamp_ns=123)
    mcp_curled_features = features_from_detection(
        _detection("Right", mcp_curled_fingers={"index"}),
        timestamp_ns=123,
    )

    assert (
        mcp_curled_features.finger_curl["index"]
        > open_features.finger_curl["index"] + 0.25
    )
    assert mcp_curled_features.finger_joint_curl["index"]["mcp"] > 0.7


def test_features_from_detection_reports_pinched_thumb_and_index() -> None:
    open_features = features_from_detection(_detection("Right"), timestamp_ns=123)
    pinched_features = features_from_detection(
        _detection("Right", pinched_index=True),
        timestamp_ns=123,
    )

    assert open_features.pinch["index"] == 0.0
    assert pinched_features.pinch["index"] > 0.8


def test_features_from_detection_reports_thumb_abduction() -> None:
    open_features = features_from_detection(_detection("Right"), timestamp_ns=123)
    pinched_features = features_from_detection(
        _detection("Right", pinched_index=True),
        timestamp_ns=123,
    )

    assert open_features.thumb_abduction > 0.4
    assert pinched_features.thumb_abduction < open_features.thumb_abduction
    assert tuple(open_features.thumb_pose) == THUMB_POSE_FEATURE_NAMES
    assert abs(open_features.thumb_pose["lateral"]) == open_features.thumb_abduction
    assert open_features.thumb_pose["opposition"] == 0.0
    assert pinched_features.thumb_pose["index_pinch"] == pinched_features.pinch["index"]


def test_features_from_detection_reports_signed_spread() -> None:
    features = features_from_detection(
        _detection("Right", splayed=True),
        timestamp_ns=123,
    )

    assert features.finger_spread["index"] < 0.0
    assert features.finger_spread["pinky"] > 0.0


def test_features_from_detection_uses_proximal_finger_axis_for_spread() -> None:
    neutral = features_from_detection(_detection("Right"), timestamp_ns=123)
    points = _landmarks(
        curled_fingers=set(),
        mcp_curled_fingers=set(),
        pinched_index=False,
        splayed=False,
    )
    points[8] = Landmark(0.31, 0.18, 0.0)
    points[20] = Landmark(0.79, 0.24, 0.0)
    tip_only = features_from_detection(
        HandDetection(handedness="Right", score=0.9, landmarks=tuple(points)),
        timestamp_ns=124,
    )

    assert tip_only.finger_spread["index"] == neutral.finger_spread["index"]
    assert tip_only.finger_spread["pinky"] == neutral.finger_spread["pinky"]


def _detection(
    handedness: str,
    *,
    score: float = 0.9,
    curled_fingers: set[str] | None = None,
    mcp_curled_fingers: set[str] | None = None,
    pinched_index: bool = False,
    splayed: bool = False,
) -> HandDetection:
    return HandDetection(
        handedness=handedness,
        score=score,
        landmarks=tuple(
            _landmarks(
                curled_fingers=curled_fingers or set(),
                mcp_curled_fingers=mcp_curled_fingers or set(),
                pinched_index=pinched_index,
                splayed=splayed,
            )
        ),
    )


def _landmarks(
    *,
    curled_fingers: set[str],
    mcp_curled_fingers: set[str],
    pinched_index: bool,
    splayed: bool,
) -> list[Landmark]:
    points = [
        Landmark(0.50, 0.90, 0.0),  # wrist
        Landmark(0.38, 0.74, 0.0),
        Landmark(0.29, 0.63, 0.0),
        Landmark(0.22, 0.52, 0.0),
        Landmark(0.18, 0.42, 0.0),
        Landmark(0.40, 0.62, 0.0),
        Landmark(0.40, 0.46, 0.0),
        Landmark(0.40, 0.32, 0.0),
        Landmark(0.40, 0.18, 0.0),
        Landmark(0.50, 0.60, 0.0),
        Landmark(0.50, 0.43, 0.0),
        Landmark(0.50, 0.29, 0.0),
        Landmark(0.50, 0.16, 0.0),
        Landmark(0.60, 0.62, 0.0),
        Landmark(0.60, 0.46, 0.0),
        Landmark(0.60, 0.32, 0.0),
        Landmark(0.60, 0.18, 0.0),
        Landmark(0.70, 0.66, 0.0),
        Landmark(0.70, 0.51, 0.0),
        Landmark(0.70, 0.37, 0.0),
        Landmark(0.70, 0.24, 0.0),
    ]

    if "index" in curled_fingers:
        points[6] = Landmark(0.40, 0.46, 0.0)
        points[7] = Landmark(0.53, 0.55, 0.0)
        points[8] = Landmark(0.55, 0.67, 0.0)

    if "index" in mcp_curled_fingers:
        points[6] = Landmark(0.40, 0.72, 0.0)
        points[7] = Landmark(0.40, 0.82, 0.0)
        points[8] = Landmark(0.40, 0.92, 0.0)

    if pinched_index:
        points[4] = Landmark(0.405, 0.185, 0.0)

    if splayed:
        points[6] = Landmark(0.31, 0.46, 0.0)
        points[8] = Landmark(0.31, 0.18, 0.0)
        points[18] = Landmark(0.79, 0.51, 0.0)
        points[20] = Landmark(0.79, 0.24, 0.0)

    return points
