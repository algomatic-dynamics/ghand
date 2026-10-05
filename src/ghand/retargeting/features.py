"""Retargeting feature contracts derived from human hand tracking."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Final

from ghand.teleop.hand_pose import HandDetection, HandFrame, Landmark


FINGER_NAMES: Final[tuple[str, ...]] = ("thumb", "index", "middle", "ring", "pinky")
FINGER_CURL_JOINT_NAMES: Final[tuple[str, ...]] = ("mcp", "pip", "dip")
SPREAD_FINGER_NAMES: Final[tuple[str, ...]] = ("index", "middle", "ring", "pinky")
PINCH_FINGER_NAMES: Final[tuple[str, ...]] = ("index", "middle", "ring")
THUMB_POSE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "lateral",
    "opposition",
    "cmc_curl",
    "mcp_curl",
    "ip_curl",
    "index_pinch",
)
MEDIAPIPE_HAND_LANDMARK_COUNT: Final[int] = 21
MEDIAPIPE_FINGER_LANDMARKS: Final[dict[str, tuple[int, int, int, int]]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
WRIST_LANDMARK: Final[int] = 0
MIDDLE_MCP_LANDMARK: Final[int] = 9
THUMB_TIP_LANDMARK: Final[int] = 4
PINCH_DISTANCE_SCALE: Final[float] = 0.45
_EPSILON: Final[float] = 1e-9
_Point = tuple[float, float, float]
_PalmFrame = tuple[_Point, _Point, _Point]


@dataclass(frozen=True)
class HandFeatures:
    """Normalized human-hand features used as retargeting input.

    The values are intentionally device-independent. Hardware mapping happens
    later through joint-name targets and calibration.
    """

    timestamp_ns: int
    handedness: str
    score: float
    finger_curl: Mapping[str, float]
    finger_spread: Mapping[str, float]
    pinch: Mapping[str, float]
    palm_size: float
    thumb_abduction: float = 0.0
    valid: bool = True
    finger_joint_curl: Mapping[str, Mapping[str, float]] | None = None
    thumb_pose: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative.")
        if not self.handedness:
            raise ValueError("handedness must not be empty.")
        object.__setattr__(
            self, "score", _validate_range(self.score, 0.0, 1.0, "score")
        )
        object.__setattr__(
            self,
            "finger_curl",
            _validate_metric_map(
                self.finger_curl,
                FINGER_NAMES,
                lower=0.0,
                upper=1.0,
                label="finger_curl",
            ),
        )
        object.__setattr__(
            self,
            "finger_joint_curl",
            _validate_finger_joint_curl_map(
                self.finger_joint_curl
                if self.finger_joint_curl is not None
                else _finger_joint_curl_from_aggregate(self.finger_curl),
            ),
        )
        object.__setattr__(
            self,
            "finger_spread",
            _validate_metric_map(
                self.finger_spread,
                SPREAD_FINGER_NAMES,
                lower=-1.0,
                upper=1.0,
                label="finger_spread",
            ),
        )
        object.__setattr__(
            self,
            "pinch",
            _validate_metric_map(
                self.pinch,
                PINCH_FINGER_NAMES,
                lower=0.0,
                upper=1.0,
                label="pinch",
            ),
        )
        object.__setattr__(
            self,
            "palm_size",
            _validate_range(self.palm_size, 0.0, math.inf, "palm_size"),
        )
        object.__setattr__(
            self,
            "thumb_abduction",
            _validate_range(self.thumb_abduction, 0.0, 1.0, "thumb_abduction"),
        )
        object.__setattr__(
            self,
            "thumb_pose",
            _validate_metric_map(
                self.thumb_pose
                if self.thumb_pose is not None
                else _thumb_pose_from_legacy(
                    thumb_abduction=self.thumb_abduction,
                    thumb_joint_curl=self.finger_joint_curl["thumb"],
                    index_pinch=self.pinch["index"],
                ),
                THUMB_POSE_FEATURE_NAMES,
                lower=-1.0,
                upper=1.0,
                label="thumb_pose",
            ),
        )

    @classmethod
    def empty(cls, *, timestamp_ns: int, handedness: str = "Unknown") -> HandFeatures:
        """Return an invalid zero feature set for frames without usable hands."""

        return cls(
            timestamp_ns=timestamp_ns,
            handedness=handedness,
            score=0.0,
            finger_curl={finger: 0.0 for finger in FINGER_NAMES},
            finger_joint_curl={
                finger: {joint: 0.0 for joint in FINGER_CURL_JOINT_NAMES}
                for finger in FINGER_NAMES
            },
            finger_spread={finger: 0.0 for finger in SPREAD_FINGER_NAMES},
            pinch={finger: 0.0 for finger in PINCH_FINGER_NAMES},
            palm_size=0.0,
            thumb_abduction=0.0,
            thumb_pose={feature: 0.0 for feature in THUMB_POSE_FEATURE_NAMES},
            valid=False,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def extract_hand_features(
    hand_frame: HandFrame,
    *,
    preferred_handedness: str | None = None,
) -> HandFeatures:
    """Extract normalized features from the best hand in a tracked frame."""

    detection = select_hand_detection(
        hand_frame, preferred_handedness=preferred_handedness
    )
    if detection is None:
        return HandFeatures.empty(timestamp_ns=hand_frame.timestamp_ns)
    return features_from_detection(detection, timestamp_ns=hand_frame.timestamp_ns)


def select_hand_detection(
    hand_frame: HandFrame,
    *,
    preferred_handedness: str | None = None,
) -> HandDetection | None:
    """Select the preferred hand detection, falling back to the highest score."""

    hands = tuple(hand_frame.hands)
    if not hands:
        return None

    if preferred_handedness:
        preferred = preferred_handedness.casefold()
        matches = tuple(
            hand for hand in hands if hand.handedness.casefold() == preferred
        )
        if matches:
            return max(matches, key=lambda hand: hand.score)

    return max(hands, key=lambda hand: hand.score)


def features_from_detection(
    detection: HandDetection,
    *,
    timestamp_ns: int,
) -> HandFeatures:
    """Extract normalized retargeting features from one hand detection."""

    points = _points_from_detection(detection)
    if points is None:
        return HandFeatures.empty(
            timestamp_ns=timestamp_ns, handedness=detection.handedness
        )

    palm_size = _palm_size(points)
    if palm_size <= _EPSILON:
        return HandFeatures.empty(
            timestamp_ns=timestamp_ns, handedness=detection.handedness
        )

    palm_frame = _palm_frame(points)

    finger_joint_curl = {
        finger: _finger_joint_curl(
            points, MEDIAPIPE_FINGER_LANDMARKS[finger], palm_frame
        )
        for finger in FINGER_NAMES
    }

    pinch = {finger: _pinch(points, finger, palm_size) for finger in PINCH_FINGER_NAMES}

    return HandFeatures(
        timestamp_ns=timestamp_ns,
        handedness=detection.handedness,
        score=detection.score,
        finger_curl={
            finger: _aggregate_finger_joint_curl(finger_joint_curl[finger])
            for finger in FINGER_NAMES
        },
        finger_joint_curl=finger_joint_curl,
        finger_spread={
            finger: _finger_spread(points, finger, palm_frame)
            for finger in SPREAD_FINGER_NAMES
        },
        pinch=pinch,
        palm_size=palm_size,
        thumb_abduction=_thumb_abduction(points, palm_frame),
        thumb_pose=_thumb_pose_features(
            points,
            palm_frame,
            finger_joint_curl["thumb"],
            index_pinch=pinch["index"],
        ),
        valid=True,
    )


def _validate_metric_map(
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
    if actual != required:
        missing = ", ".join(sorted(required - actual)) or "none"
        extra = ", ".join(sorted(actual - required)) or "none"
        raise ValueError(
            f"{label} keys must be {required_keys}; missing={missing}; extra={extra}."
        )

    return {
        key: _validate_range(value_by_key[key], lower, upper, f"{label}.{key}")
        for key in required_keys
    }


def _validate_finger_joint_curl_map(
    values: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    value_by_key = dict(values)
    required = set(FINGER_NAMES)
    actual = set(value_by_key)
    if actual != required:
        missing = ", ".join(sorted(required - actual)) or "none"
        extra = ", ".join(sorted(actual - required)) or "none"
        raise ValueError(
            "finger_joint_curl keys must be "
            f"{FINGER_NAMES}; missing={missing}; extra={extra}."
        )

    return {
        finger: _validate_metric_map(
            value_by_key[finger],
            FINGER_CURL_JOINT_NAMES,
            lower=0.0,
            upper=1.0,
            label=f"finger_joint_curl.{finger}",
        )
        for finger in FINGER_NAMES
    }


def _finger_joint_curl_from_aggregate(
    finger_curl: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    return {
        finger: {joint: float(finger_curl[finger]) for joint in FINGER_CURL_JOINT_NAMES}
        for finger in FINGER_NAMES
    }


def _thumb_pose_from_legacy(
    *,
    thumb_abduction: float,
    thumb_joint_curl: Mapping[str, float],
    index_pinch: float,
) -> dict[str, float]:
    return {
        "lateral": float(thumb_abduction),
        "opposition": 0.0,
        "cmc_curl": float(thumb_joint_curl["mcp"]),
        "mcp_curl": float(thumb_joint_curl["pip"]),
        "ip_curl": float(thumb_joint_curl["dip"]),
        "index_pinch": float(index_pinch),
    }


def _validate_range(value: float, lower: float, upper: float, label: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite.")
    if not lower <= normalized <= upper:
        raise ValueError(f"{label} must be in range {lower}..{upper}.")
    return normalized


def _points_from_detection(detection: HandDetection) -> tuple[_Point, ...] | None:
    landmarks = detection.world_landmarks or detection.landmarks
    if len(landmarks) < MEDIAPIPE_HAND_LANDMARK_COUNT:
        return None
    return tuple(
        _point(landmark) for landmark in landmarks[:MEDIAPIPE_HAND_LANDMARK_COUNT]
    )


def _point(landmark: Landmark) -> _Point:
    return (float(landmark.x), float(landmark.y), float(landmark.z))


def _palm_size(points: tuple[_Point, ...]) -> float:
    wrist_to_middle_mcp = _distance(points[WRIST_LANDMARK], points[MIDDLE_MCP_LANDMARK])
    index_to_pinky_mcp = _distance(
        points[MEDIAPIPE_FINGER_LANDMARKS["index"][0]],
        points[MEDIAPIPE_FINGER_LANDMARKS["pinky"][0]],
    )
    return max(wrist_to_middle_mcp, index_to_pinky_mcp)


def _finger_curl(
    points: tuple[_Point, ...],
    landmark_ids: tuple[int, int, int, int],
    palm_frame: _PalmFrame | None,
) -> float:
    return _aggregate_finger_joint_curl(
        _finger_joint_curl(points, landmark_ids, palm_frame)
    )


def _finger_joint_curl(
    points: tuple[_Point, ...],
    landmark_ids: tuple[int, int, int, int],
    palm_frame: _PalmFrame | None,
) -> dict[str, float]:
    base, middle_a, middle_b, tip = (points[index] for index in landmark_ids)
    return {
        "mcp": _mcp_curl(base, middle_a, palm_frame),
        "pip": _joint_curl(base, middle_a, middle_b),
        "dip": _joint_curl(middle_a, middle_b, tip),
    }


def _aggregate_finger_joint_curl(joint_curl: Mapping[str, float]) -> float:
    return _clamp(
        0.35 * float(joint_curl["mcp"])
        + 0.40 * float(joint_curl["pip"])
        + 0.25 * float(joint_curl["dip"]),
        0.0,
        1.0,
    )


def _mcp_curl(base: _Point, pip: _Point, palm_frame: _PalmFrame | None) -> float:
    if palm_frame is None:
        return 0.0

    forward_axis, lateral_axis, _normal_axis = palm_frame
    finger_axis = _normalize(_sub(pip, base))
    if finger_axis is None:
        return 0.0

    flexion_axis = _normalize(_reject(finger_axis, lateral_axis))
    if flexion_axis is None:
        return 0.0

    angle = _angle(forward_axis, flexion_axis)
    if angle is None:
        return 0.0
    return _clamp(angle / (math.pi / 2.0), 0.0, 1.0)


def _joint_curl(previous: _Point, joint: _Point, following: _Point) -> float:
    first = _sub(previous, joint)
    second = _sub(following, joint)
    angle = _angle(first, second)
    if angle is None:
        return 0.0
    return _clamp((math.pi - angle) / math.pi, 0.0, 1.0)


def _finger_spread(
    points: tuple[_Point, ...],
    finger: str,
    palm_frame: _PalmFrame | None,
) -> float:
    if palm_frame is None:
        return 0.0

    mcp_id, pip_id, _dip_id, _tip_id = MEDIAPIPE_FINGER_LANDMARKS[finger]
    _forward_axis, lateral_axis, normal_axis = palm_frame
    finger_axis = _normalize(_sub(points[pip_id], points[mcp_id]))
    if finger_axis is None:
        return 0.0

    in_palm_axis = _normalize(_reject(finger_axis, normal_axis))
    if in_palm_axis is None:
        return 0.0
    return _clamp(_dot(in_palm_axis, lateral_axis), -1.0, 1.0)


def _pinch(points: tuple[_Point, ...], finger: str, palm_size: float) -> float:
    tip_id = MEDIAPIPE_FINGER_LANDMARKS[finger][3]
    distance = _distance(points[THUMB_TIP_LANDMARK], points[tip_id])
    threshold = max(palm_size * PINCH_DISTANCE_SCALE, _EPSILON)
    return 1.0 - _clamp(distance / threshold, 0.0, 1.0)


def _thumb_pose_features(
    points: tuple[_Point, ...],
    palm_frame: _PalmFrame | None,
    thumb_joint_curl: Mapping[str, float],
    *,
    index_pinch: float,
) -> dict[str, float]:
    lateral, opposition = _thumb_direction_features(points, palm_frame)
    return {
        "lateral": lateral,
        "opposition": opposition,
        "cmc_curl": float(thumb_joint_curl["mcp"]),
        "mcp_curl": float(thumb_joint_curl["pip"]),
        "ip_curl": float(thumb_joint_curl["dip"]),
        "index_pinch": float(index_pinch),
    }


def _thumb_abduction(
    points: tuple[_Point, ...], palm_frame: _PalmFrame | None
) -> float:
    lateral, _opposition = _thumb_direction_features(points, palm_frame)
    return abs(lateral)


def _thumb_direction_features(
    points: tuple[_Point, ...],
    palm_frame: _PalmFrame | None,
) -> tuple[float, float]:
    if palm_frame is None:
        return 0.0, 0.0

    _forward_axis, lateral_axis, normal_axis = palm_frame
    thumb_mcp_id = MEDIAPIPE_FINGER_LANDMARKS["thumb"][1]
    thumb_tip_id = MEDIAPIPE_FINGER_LANDMARKS["thumb"][3]
    thumb_axis = _normalize(_sub(points[thumb_tip_id], points[thumb_mcp_id]))
    if thumb_axis is None:
        return 0.0, 0.0

    in_palm_axis = _normalize(_reject(thumb_axis, normal_axis))
    lateral = 0.0 if in_palm_axis is None else _dot(in_palm_axis, lateral_axis)
    opposition = _dot(thumb_axis, normal_axis)
    return _clamp(lateral, -1.0, 1.0), _clamp(opposition, -1.0, 1.0)


def _angle(first: _Point, second: _Point) -> float | None:
    first_norm = _norm(first)
    second_norm = _norm(second)
    if first_norm <= _EPSILON or second_norm <= _EPSILON:
        return None
    cosine = _clamp(_dot(first, second) / (first_norm * second_norm), -1.0, 1.0)
    return math.acos(cosine)


def _palm_frame(points: tuple[_Point, ...]) -> _PalmFrame | None:
    forward_axis = _normalize(_sub(points[MIDDLE_MCP_LANDMARK], points[WRIST_LANDMARK]))
    index_mcp = points[MEDIAPIPE_FINGER_LANDMARKS["index"][0]]
    pinky_mcp = points[MEDIAPIPE_FINGER_LANDMARKS["pinky"][0]]
    lateral_raw = _sub(pinky_mcp, index_mcp)
    if forward_axis is None:
        return None

    lateral_axis = _normalize(_reject(lateral_raw, forward_axis))
    if lateral_axis is None:
        return None

    normal_axis = _normalize(_cross(lateral_axis, forward_axis))
    if normal_axis is None:
        return None

    return (forward_axis, lateral_axis, normal_axis)


def _sub(first: _Point, second: _Point) -> _Point:
    return (first[0] - second[0], first[1] - second[1], first[2] - second[2])


def _scale(vector: _Point, value: float) -> _Point:
    return (vector[0] * value, vector[1] * value, vector[2] * value)


def _reject(vector: _Point, axis: _Point) -> _Point:
    return _sub(vector, _scale(axis, _dot(vector, axis)))


def _dot(first: _Point, second: _Point) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _cross(first: _Point, second: _Point) -> _Point:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _distance(first: _Point, second: _Point) -> float:
    return _norm(_sub(first, second))


def _norm(vector: _Point) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: _Point) -> _Point | None:
    norm = _norm(vector)
    if norm <= _EPSILON:
        return None
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)
