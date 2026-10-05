"""Stateful operator-hand selection for teleoperation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Final

from ghand.teleop.hand_pose import HandDetection, HandFrame, Landmark

WRIST_LANDMARK: Final[int] = 0
MIDDLE_MCP_LANDMARK: Final[int] = 9
MIN_LANDMARKS_FOR_MATCHING: Final[int] = 10

DEFAULT_INITIAL_ACQUIRE_FRAMES: Final[int] = 1
DEFAULT_REACQUIRE_FRAMES: Final[int] = 3
DEFAULT_SUSPECT_FRAMES: Final[int] = 2
DEFAULT_LOCK_MATCH_DISTANCE: Final[float] = 0.13
DEFAULT_ACQUIRE_MATCH_DISTANCE: Final[float] = 0.16


class OperatorHandTrackingState(str, Enum):
    """Tracking state for one logical operator hand."""

    acquiring = "acquiring"
    locked = "locked"
    suspect = "suspect"
    lost = "lost"


@dataclass(frozen=True)
class OperatorHandSelection:
    """Selected operator hand and tracking diagnostics for one frame."""

    hand_frame: HandFrame
    state: OperatorHandTrackingState
    track_id: int | None
    track_changed: bool
    candidates: int
    selected_index: int | None
    reason: str


@dataclass(frozen=True)
class _HandSnapshot:
    index: int
    hand: HandDetection
    center: tuple[float, float]
    wrist: tuple[float, float]
    palm_size: float


class OperatorHandSelector:
    """Track one logical operator hand across MediaPipe hand detections."""

    def __init__(
        self,
        *,
        target_handedness: str | None = None,
        initial_acquire_frames: int = DEFAULT_INITIAL_ACQUIRE_FRAMES,
        reacquire_frames: int = DEFAULT_REACQUIRE_FRAMES,
        suspect_frames: int = DEFAULT_SUSPECT_FRAMES,
        lock_match_distance: float = DEFAULT_LOCK_MATCH_DISTANCE,
        acquire_match_distance: float = DEFAULT_ACQUIRE_MATCH_DISTANCE,
    ) -> None:
        self.target_handedness = (
            target_handedness.casefold() if target_handedness else None
        )
        self.initial_acquire_frames = _positive_int(
            initial_acquire_frames,
            "initial_acquire_frames",
        )
        self.reacquire_frames = _positive_int(reacquire_frames, "reacquire_frames")
        self.suspect_frames = _positive_int(suspect_frames, "suspect_frames")
        self.lock_match_distance = _positive_finite(
            lock_match_distance,
            "lock_match_distance",
        )
        self.acquire_match_distance = _positive_finite(
            acquire_match_distance,
            "acquire_match_distance",
        )
        self._state = OperatorHandTrackingState.acquiring
        self._locked: _HandSnapshot | None = None
        self._pending: _HandSnapshot | None = None
        self._pending_frames = 0
        self._suspect_frames = 0
        self._next_track_id = 1
        self._track_id: int | None = None
        self._last_output_track_id: int | None = None
        self._ever_locked = False

    def reset(self) -> None:
        """Clear all tracking state."""

        self._state = OperatorHandTrackingState.acquiring
        self._locked = None
        self._pending = None
        self._pending_frames = 0
        self._suspect_frames = 0
        self._track_id = None
        self._last_output_track_id = None
        self._ever_locked = False

    def update(self, hand_frame: HandFrame) -> OperatorHandSelection:
        """Select the current operator hand from a detected hand frame."""

        candidates = _candidate_snapshots(
            hand_frame,
            target_handedness=self.target_handedness,
        )
        if self._state is OperatorHandTrackingState.locked:
            selection = self._update_locked(hand_frame, candidates)
        elif self._state is OperatorHandTrackingState.suspect:
            selection = self._update_suspect(hand_frame, candidates)
        else:
            selection = self._update_acquiring(hand_frame, candidates)

        object.__setattr__(
            selection,
            "track_changed",
            selection.track_id != self._last_output_track_id,
        )
        self._last_output_track_id = selection.track_id
        return selection

    def _update_locked(
        self,
        hand_frame: HandFrame,
        candidates: tuple[_HandSnapshot, ...],
    ) -> OperatorHandSelection:
        match = _best_match(self._locked, candidates)
        if match is not None and match[1] <= self.lock_match_distance:
            snapshot, distance = match
            self._locked = snapshot
            self._pending = None
            self._pending_frames = 0
            self._suspect_frames = 0
            return self._selection(
                hand_frame,
                state=OperatorHandTrackingState.locked,
                snapshot=snapshot,
                candidates=len(candidates),
                reason=f"matched distance={distance:.3f}",
            )

        self._state = OperatorHandTrackingState.suspect
        self._suspect_frames = 1
        return self._empty_selection(
            hand_frame,
            state=OperatorHandTrackingState.suspect,
            candidates=len(candidates),
            reason=_miss_reason(match),
        )

    def _update_suspect(
        self,
        hand_frame: HandFrame,
        candidates: tuple[_HandSnapshot, ...],
    ) -> OperatorHandSelection:
        match = _best_match(self._locked, candidates)
        if match is not None and match[1] <= self.lock_match_distance:
            snapshot, distance = match
            self._state = OperatorHandTrackingState.locked
            self._locked = snapshot
            self._pending = None
            self._pending_frames = 0
            self._suspect_frames = 0
            return self._selection(
                hand_frame,
                state=OperatorHandTrackingState.locked,
                snapshot=snapshot,
                candidates=len(candidates),
                reason=f"recovered distance={distance:.3f}",
            )

        self._suspect_frames += 1
        if self._suspect_frames > self.suspect_frames:
            self._state = OperatorHandTrackingState.lost
            self._locked = None
            self._track_id = None
            self._pending = None
            self._pending_frames = 0
            self._suspect_frames = 0
            return self._empty_selection(
                hand_frame,
                state=OperatorHandTrackingState.lost,
                candidates=len(candidates),
                reason=_miss_reason(match),
            )

        return self._empty_selection(
            hand_frame,
            state=OperatorHandTrackingState.suspect,
            candidates=len(candidates),
            reason=_miss_reason(match),
        )

    def _update_acquiring(
        self,
        hand_frame: HandFrame,
        candidates: tuple[_HandSnapshot, ...],
    ) -> OperatorHandSelection:
        if not candidates:
            self._state = (
                OperatorHandTrackingState.lost
                if self._ever_locked
                else OperatorHandTrackingState.acquiring
            )
            self._pending = None
            self._pending_frames = 0
            return self._empty_selection(
                hand_frame,
                state=self._state,
                candidates=0,
                reason="no matching handedness candidates",
            )

        snapshot = _best_acquire_candidate(candidates)
        if (
            self._pending is not None
            and _match_distance(self._pending, snapshot) <= self.acquire_match_distance
        ):
            self._pending = snapshot
            self._pending_frames += 1
        else:
            self._pending = snapshot
            self._pending_frames = 1

        required_frames = (
            self.reacquire_frames if self._ever_locked else self.initial_acquire_frames
        )
        if self._pending_frames >= required_frames:
            self._state = OperatorHandTrackingState.locked
            self._locked = snapshot
            self._pending = None
            self._pending_frames = 0
            self._suspect_frames = 0
            self._track_id = self._next_track_id
            self._next_track_id += 1
            self._ever_locked = True
            return self._selection(
                hand_frame,
                state=OperatorHandTrackingState.locked,
                snapshot=snapshot,
                candidates=len(candidates),
                reason=f"acquired {required_frames} frame(s)",
            )

        self._state = OperatorHandTrackingState.acquiring
        return self._empty_selection(
            hand_frame,
            state=OperatorHandTrackingState.acquiring,
            candidates=len(candidates),
            reason=f"acquiring {self._pending_frames}/{required_frames}",
        )

    def _selection(
        self,
        hand_frame: HandFrame,
        *,
        state: OperatorHandTrackingState,
        snapshot: _HandSnapshot,
        candidates: int,
        reason: str,
    ) -> OperatorHandSelection:
        selected_frame = replace(hand_frame, hands=(snapshot.hand,))
        return OperatorHandSelection(
            hand_frame=selected_frame,
            state=state,
            track_id=self._track_id,
            track_changed=False,
            candidates=candidates,
            selected_index=snapshot.index,
            reason=reason,
        )

    def _empty_selection(
        self,
        hand_frame: HandFrame,
        *,
        state: OperatorHandTrackingState,
        candidates: int,
        reason: str,
    ) -> OperatorHandSelection:
        return OperatorHandSelection(
            hand_frame=replace(hand_frame, hands=()),
            state=state,
            track_id=None,
            track_changed=False,
            candidates=candidates,
            selected_index=None,
            reason=reason,
        )


def _candidate_snapshots(
    hand_frame: HandFrame,
    *,
    target_handedness: str | None,
) -> tuple[_HandSnapshot, ...]:
    snapshots: list[_HandSnapshot] = []
    for index, hand in enumerate(hand_frame.hands):
        if target_handedness and hand.handedness.casefold() != target_handedness:
            continue
        snapshot = _snapshot(index, hand)
        if snapshot is not None:
            snapshots.append(snapshot)
    return tuple(snapshots)


def _snapshot(index: int, hand: HandDetection) -> _HandSnapshot | None:
    if len(hand.landmarks) < MIN_LANDMARKS_FOR_MATCHING:
        return None
    center = _center(hand.landmarks)
    wrist = _point2(hand.landmarks[WRIST_LANDMARK])
    palm_size = _distance2(
        hand.landmarks[WRIST_LANDMARK],
        hand.landmarks[MIDDLE_MCP_LANDMARK],
    )
    if not math.isfinite(palm_size) or palm_size <= 0.0:
        return None
    return _HandSnapshot(
        index=index,
        hand=hand,
        center=center,
        wrist=wrist,
        palm_size=palm_size,
    )


def _best_acquire_candidate(
    candidates: tuple[_HandSnapshot, ...],
) -> _HandSnapshot:
    return max(candidates, key=lambda snapshot: (snapshot.hand.score, snapshot.palm_size))


def _best_match(
    previous: _HandSnapshot | None,
    candidates: tuple[_HandSnapshot, ...],
) -> tuple[_HandSnapshot, float] | None:
    if previous is None or not candidates:
        return None
    return min(
        ((candidate, _match_distance(previous, candidate)) for candidate in candidates),
        key=lambda item: item[1],
    )


def _match_distance(previous: _HandSnapshot, current: _HandSnapshot) -> float:
    center_distance = _distance_points(previous.center, current.center)
    wrist_distance = _distance_points(previous.wrist, current.wrist)
    landmark_distance = _landmark_rms(previous.hand.landmarks, current.hand.landmarks)
    palm_ratio = abs(math.log(current.palm_size / previous.palm_size))
    handedness_penalty = (
        0.0
        if previous.hand.handedness.casefold() == current.hand.handedness.casefold()
        else 0.20
    )
    return (
        0.30 * center_distance
        + 0.30 * wrist_distance
        + 0.30 * landmark_distance
        + 0.10 * palm_ratio
        + handedness_penalty
    )


def _landmark_rms(
    previous: tuple[Landmark, ...],
    current: tuple[Landmark, ...],
) -> float:
    count = min(len(previous), len(current))
    if count == 0:
        return math.inf
    total = 0.0
    for left, right in zip(previous[:count], current[:count], strict=False):
        dx = left.x - right.x
        dy = left.y - right.y
        total += dx * dx + dy * dy
    return math.sqrt(total / count)


def _center(landmarks: tuple[Landmark, ...]) -> tuple[float, float]:
    count = len(landmarks)
    return (
        sum(landmark.x for landmark in landmarks) / count,
        sum(landmark.y for landmark in landmarks) / count,
    )


def _point2(landmark: Landmark) -> tuple[float, float]:
    return (landmark.x, landmark.y)


def _distance2(left: Landmark, right: Landmark) -> float:
    return _distance_points(_point2(left), _point2(right))


def _distance_points(left: tuple[float, float], right: tuple[float, float]) -> float:
    dx = left[0] - right[0]
    dy = left[1] - right[1]
    return math.sqrt(dx * dx + dy * dy)


def _miss_reason(match: tuple[_HandSnapshot, float] | None) -> str:
    if match is None:
        return "no matching handedness candidates"
    return f"match distance too large ({match[1]:.3f})"


def _positive_int(value: int, label: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{label} must be positive.")
    return normalized


def _positive_finite(value: float, label: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{label} must be positive and finite.")
    return normalized
