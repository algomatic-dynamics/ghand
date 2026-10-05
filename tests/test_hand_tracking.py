import pytest

from ghand.teleop.hand_pose import HandDetection, HandFrame, Landmark
from ghand.teleop.hand_tracking import (
    OperatorHandSelector,
    OperatorHandTrackingState,
)


def test_operator_hand_selector_locks_and_keeps_same_track_for_small_motion() -> None:
    selector = OperatorHandSelector(target_handedness="Right")

    first = selector.update(_frame(_hand(offset_x=0.0), timestamp_ns=1))
    second = selector.update(_frame(_hand(offset_x=0.02), timestamp_ns=2))

    assert first.state is OperatorHandTrackingState.locked
    assert first.track_id == 1
    assert first.track_changed is True
    assert second.state is OperatorHandTrackingState.locked
    assert second.track_id == 1
    assert second.track_changed is False
    assert second.hand_frame.hands[0].landmarks[0].x == pytest.approx(0.52)


def test_operator_hand_selector_prefers_locked_hand_over_higher_score_intruder() -> None:
    selector = OperatorHandSelector(target_handedness="Right")
    selector.update(_frame(_hand(offset_x=0.0, score=0.70), timestamp_ns=1))

    selection = selector.update(
        _frame(
            _hand(offset_x=0.35, score=0.99),
            _hand(offset_x=0.02, score=0.70),
            timestamp_ns=2,
        )
    )

    assert selection.state is OperatorHandTrackingState.locked
    assert selection.track_id == 1
    assert selection.selected_index == 1
    assert selection.hand_frame.hands[0].score == pytest.approx(0.70)


def test_operator_hand_selector_enters_suspect_then_lost_when_hand_disappears() -> None:
    selector = OperatorHandSelector(target_handedness="Right", suspect_frames=1)
    selector.update(_frame(_hand(), timestamp_ns=1))

    suspect = selector.update(_frame(timestamp_ns=2))
    lost = selector.update(_frame(timestamp_ns=3))

    assert suspect.state is OperatorHandTrackingState.suspect
    assert suspect.track_id is None
    assert suspect.hand_frame.hands == ()
    assert lost.state is OperatorHandTrackingState.lost
    assert lost.track_id is None
    assert lost.hand_frame.hands == ()


def test_operator_hand_selector_does_not_immediately_relock_after_lost() -> None:
    selector = OperatorHandSelector(
        target_handedness="Right",
        reacquire_frames=3,
        suspect_frames=1,
    )
    selector.update(_frame(_hand(offset_x=0.0), timestamp_ns=1))
    selector.update(_frame(timestamp_ns=2))
    selector.update(_frame(timestamp_ns=3))

    first_reacquire = selector.update(_frame(_hand(offset_x=0.4), timestamp_ns=4))
    second_reacquire = selector.update(_frame(_hand(offset_x=0.41), timestamp_ns=5))
    locked = selector.update(_frame(_hand(offset_x=0.42), timestamp_ns=6))

    assert first_reacquire.state is OperatorHandTrackingState.acquiring
    assert first_reacquire.track_id is None
    assert second_reacquire.state is OperatorHandTrackingState.acquiring
    assert second_reacquire.track_id is None
    assert locked.state is OperatorHandTrackingState.locked
    assert locked.track_id == 2
    assert locked.track_changed is True


def test_operator_hand_selector_filters_to_target_handedness() -> None:
    selector = OperatorHandSelector(target_handedness="Right", suspect_frames=1)
    selector.update(_frame(_hand(handedness="Right"), timestamp_ns=1))

    suspect = selector.update(_frame(_hand(handedness="Left"), timestamp_ns=2))
    lost = selector.update(_frame(_hand(handedness="Left"), timestamp_ns=3))

    assert suspect.state is OperatorHandTrackingState.suspect
    assert suspect.candidates == 0
    assert lost.state is OperatorHandTrackingState.lost
    assert lost.candidates == 0


def _frame(*hands: HandDetection, timestamp_ns: int) -> HandFrame:
    return HandFrame(
        timestamp_ns=timestamp_ns,
        image_width=640,
        image_height=480,
        hands=hands,
    )


def _hand(
    *,
    handedness: str = "Right",
    score: float = 0.9,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> HandDetection:
    return HandDetection(
        handedness=handedness,
        score=score,
        landmarks=tuple(
            Landmark(point.x + offset_x, point.y + offset_y, point.z)
            for point in _open_hand_landmarks()
        ),
    )


def _open_hand_landmarks() -> tuple[Landmark, ...]:
    return (
        Landmark(0.50, 0.90, 0.0),
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
    )
