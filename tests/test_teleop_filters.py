import pytest

from ghand.teleop.filters import HandFrameOneEuroSmoother, OneEuroFilter
from ghand.teleop.hand_pose import HandDetection, HandFrame, Landmark


def test_one_euro_filter_smooths_toward_new_value() -> None:
    filter_ = OneEuroFilter(min_cutoff=1.0, beta=0.0, derivative_cutoff=1.0)

    assert filter_.filter(0.0, timestamp_ns=0) == 0.0
    second = filter_.filter(1.0, timestamp_ns=33_333_333)
    third = filter_.filter(1.0, timestamp_ns=66_666_666)

    assert 0.0 < second < 1.0
    assert second < third < 1.0


def test_hand_frame_one_euro_smoother_filters_landmarks_without_mutating_raw_frame() -> None:
    smoother = HandFrameOneEuroSmoother(
        min_cutoff=1.0,
        beta=0.0,
        derivative_cutoff=1.0,
    )
    first = _hand_frame(timestamp_ns=0, wrist_x=0.1, world_wrist_x=0.01)
    second = _hand_frame(timestamp_ns=33_333_333, wrist_x=0.9, world_wrist_x=0.09)

    assert smoother.filter(first).hands[0].landmarks[0].x == pytest.approx(0.1)
    filtered = smoother.filter(second)

    assert second.hands[0].landmarks[0].x == pytest.approx(0.9)
    assert 0.1 < filtered.hands[0].landmarks[0].x < 0.9
    assert filtered.hands[0].landmarks[1].x == pytest.approx(
        second.hands[0].landmarks[1].x
    )
    assert filtered.hands[0].world_landmarks is not None
    assert 0.01 < filtered.hands[0].world_landmarks[0].x < 0.09


def test_hand_frame_one_euro_smoother_resets_after_detection_gap() -> None:
    smoother = HandFrameOneEuroSmoother(reset_after_seconds=0.1)
    smoother.filter(_hand_frame(timestamp_ns=0, wrist_x=0.1))
    smoother.filter(HandFrame(timestamp_ns=200_000_000, image_width=3, image_height=2, hands=()))

    filtered = smoother.filter(_hand_frame(timestamp_ns=300_000_000, wrist_x=0.9))

    assert filtered.hands[0].landmarks[0].x == pytest.approx(0.9)


def _hand_frame(
    *,
    timestamp_ns: int,
    wrist_x: float,
    world_wrist_x: float | None = None,
) -> HandFrame:
    world_landmarks = (
        _landmarks(world_wrist_x)
        if world_wrist_x is not None
        else None
    )
    return HandFrame(
        timestamp_ns=timestamp_ns,
        image_width=3,
        image_height=2,
        hands=(
            HandDetection(
                handedness="Right",
                score=0.9,
                landmarks=_landmarks(wrist_x),
                world_landmarks=world_landmarks,
            ),
        ),
    )


def _landmarks(wrist_x: float) -> tuple[Landmark, ...]:
    landmarks = [Landmark(0.5, 0.5, 0.0) for _ in range(21)]
    landmarks[0] = Landmark(wrist_x, 0.5, 0.0)
    return tuple(landmarks)
