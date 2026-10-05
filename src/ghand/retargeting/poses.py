"""Canonical retargeting calibration pose definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class RetargetCalibrationPoseInstruction:
    """User-facing capture instruction for one calibration pose."""

    name: str
    group: str
    title: str
    instruction: str
    affected_fingers: tuple[str, ...] = ()


RETARGET_CALIBRATION_CAPTURE_POSES: Final[
    tuple[RetargetCalibrationPoseInstruction, ...]
] = (
    RetargetCalibrationPoseInstruction(
        name="flat_closed",
        group="base",
        title="Open hand, fingers together",
        instruction="Extend the fingers and keep them together.",
        affected_fingers=("thumb", "index", "middle", "ring"),
    ),
    RetargetCalibrationPoseInstruction(
        name="flat_spread",
        group="base",
        title="Open hand, fingers spread",
        instruction="Extend the fingers and spread them wide.",
        affected_fingers=("thumb", "index", "middle", "ring"),
    ),
    RetargetCalibrationPoseInstruction(
        name="fist",
        group="base",
        title="Fist",
        instruction="Curl all fingers into a fist.",
        affected_fingers=("thumb", "index", "middle", "ring"),
    ),
    RetargetCalibrationPoseInstruction(
        name="open_thumb_flex",
        group="open_single_flex",
        title="Open hand, thumb flexed",
        instruction="Keep the fingers extended and curl only the thumb.",
        affected_fingers=("thumb",),
    ),
    RetargetCalibrationPoseInstruction(
        name="open_index_flex",
        group="open_single_flex",
        title="Open hand, index flexed",
        instruction="Keep the hand open and curl only the index finger.",
        affected_fingers=("index",),
    ),
    RetargetCalibrationPoseInstruction(
        name="open_middle_flex",
        group="open_single_flex",
        title="Open hand, middle flexed",
        instruction="Keep the hand open and curl only the middle finger.",
        affected_fingers=("middle",),
    ),
    RetargetCalibrationPoseInstruction(
        name="open_ring_flex",
        group="open_single_flex",
        title="Open hand, ring flexed",
        instruction="Keep the hand open and curl only the ring finger.",
        affected_fingers=("ring",),
    ),
    RetargetCalibrationPoseInstruction(
        name="index_point",
        group="fist_extension",
        title="Fist, index extended",
        instruction="Start from a fist and extend only the index finger.",
        affected_fingers=("index",),
    ),
    RetargetCalibrationPoseInstruction(
        name="scissors",
        group="fist_extension",
        title="Scissors / peace sign",
        instruction="Extend the index and middle fingers in a V, and curl the ring and pinky.",
        affected_fingers=("index", "middle", "ring"),
    ),
    RetargetCalibrationPoseInstruction(
        name="three_finger_extend",
        group="fist_extension",
        title="Fist, index/middle/ring extended",
        instruction="Start from a fist and extend the index, middle, and ring fingers.",
        affected_fingers=("index", "middle", "ring"),
    ),
    RetargetCalibrationPoseInstruction(
        name="index_pinch",
        group="pinch",
        title="Thumb-index pinch",
        instruction="Touch the thumb tip to the index fingertip.",
        affected_fingers=("thumb", "index"),
    ),
    RetargetCalibrationPoseInstruction(
        name="middle_pinch",
        group="pinch",
        title="Thumb-middle pinch",
        instruction="Touch the thumb tip to the middle fingertip.",
        affected_fingers=("thumb", "middle"),
    ),
    RetargetCalibrationPoseInstruction(
        name="ring_pinch",
        group="pinch",
        title="Thumb-ring pinch",
        instruction="Touch the thumb tip to the ring fingertip.",
        affected_fingers=("thumb", "ring"),
    ),
    RetargetCalibrationPoseInstruction(
        name="thumbs_up",
        group="thumb_special",
        title="Thumbs up",
        instruction="Curl the index, middle, and ring fingers while keeping only the thumb raised.",
        affected_fingers=("thumb", "index", "middle", "ring"),
    ),
)
RETARGET_CALIBRATION_CAPTURE_POSE_NAMES: Final[tuple[str, ...]] = tuple(
    pose.name for pose in RETARGET_CALIBRATION_CAPTURE_POSES
)


def retarget_pose_groups(
    poses: tuple[
        RetargetCalibrationPoseInstruction, ...
    ] = RETARGET_CALIBRATION_CAPTURE_POSES,
) -> tuple[str, ...]:
    """Return capture pose groups in display order."""

    groups: list[str] = []
    for pose in poses:
        if pose.group not in groups:
            groups.append(pose.group)
    return tuple(groups)
