"""Temporal filters for hand teleoperation inputs."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Final

from ghand.teleop.hand_pose import HandDetection, HandFrame, Landmark

_NANOSECONDS_PER_SECOND: Final[float] = 1_000_000_000.0
_EPSILON: Final[float] = 1e-12
_FilterKey = tuple[str, str, int, str]


class OneEuroFilter:
    """Scalar One Euro Filter.

    The filter adapts its cutoff frequency from the estimated derivative:
    stable samples are smoothed strongly, while faster movement gets a higher
    cutoff to reduce visible lag.
    """

    def __init__(
        self,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = _positive_finite(min_cutoff, "min_cutoff")
        self.beta = _non_negative_finite(beta, "beta")
        self.derivative_cutoff = _positive_finite(
            derivative_cutoff,
            "derivative_cutoff",
        )
        self._last_timestamp_ns: int | None = None
        self._last_raw_value: float | None = None
        self._last_filtered_value: float | None = None
        self._last_derivative: float = 0.0

    def reset(self) -> None:
        """Drop all filter history."""

        self._last_timestamp_ns = None
        self._last_raw_value = None
        self._last_filtered_value = None
        self._last_derivative = 0.0

    def filter(self, value: float, *, timestamp_ns: int) -> float:
        """Return the filtered scalar sample."""

        sample = _finite(value, "value")
        timestamp = int(timestamp_ns)
        if (
            self._last_timestamp_ns is None
            or self._last_raw_value is None
            or self._last_filtered_value is None
            or timestamp <= self._last_timestamp_ns
        ):
            self._last_timestamp_ns = timestamp
            self._last_raw_value = sample
            self._last_filtered_value = sample
            self._last_derivative = 0.0
            return sample

        dt = (timestamp - self._last_timestamp_ns) / _NANOSECONDS_PER_SECOND
        if dt <= _EPSILON:
            return self._last_filtered_value

        derivative = (sample - self._last_raw_value) / dt
        derivative_alpha = _alpha(self.derivative_cutoff, dt)
        filtered_derivative = _smooth(
            current=derivative,
            previous=self._last_derivative,
            alpha=derivative_alpha,
        )

        cutoff = self.min_cutoff + self.beta * abs(filtered_derivative)
        value_alpha = _alpha(cutoff, dt)
        filtered = _smooth(
            current=sample,
            previous=self._last_filtered_value,
            alpha=value_alpha,
        )

        self._last_timestamp_ns = timestamp
        self._last_raw_value = sample
        self._last_filtered_value = filtered
        self._last_derivative = filtered_derivative
        return filtered


class HandFrameOneEuroSmoother:
    """Apply One Euro filters to the coordinates in a hand frame."""

    def __init__(
        self,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff: float = 1.0,
        reset_after_seconds: float = 0.25,
    ) -> None:
        self.min_cutoff = _positive_finite(min_cutoff, "min_cutoff")
        self.beta = _non_negative_finite(beta, "beta")
        self.derivative_cutoff = _positive_finite(
            derivative_cutoff,
            "derivative_cutoff",
        )
        self.reset_after_seconds = _non_negative_finite(
            reset_after_seconds,
            "reset_after_seconds",
        )
        self._filters: dict[_FilterKey, OneEuroFilter] = {}
        self._last_detection_timestamp_ns: int | None = None

    def reset(self) -> None:
        """Drop all per-landmark filter history."""

        self._filters.clear()
        self._last_detection_timestamp_ns = None

    def filter(self, hand_frame: HandFrame) -> HandFrame:
        """Return a hand frame with smoothed landmark coordinates."""

        if not hand_frame.hands:
            return hand_frame

        timestamp_ns = int(hand_frame.timestamp_ns)
        if self._last_detection_timestamp_ns is not None:
            gap_seconds = (
                timestamp_ns - self._last_detection_timestamp_ns
            ) / _NANOSECONDS_PER_SECOND
            if gap_seconds < 0.0 or gap_seconds > self.reset_after_seconds:
                self._filters.clear()
        self._last_detection_timestamp_ns = timestamp_ns

        smoothed_hands = tuple(
            self._filter_hand(hand, hand_index=index, timestamp_ns=timestamp_ns)
            for index, hand in enumerate(hand_frame.hands)
        )
        return replace(hand_frame, hands=smoothed_hands)

    def _filter_hand(
        self,
        hand: HandDetection,
        *,
        hand_index: int,
        timestamp_ns: int,
    ) -> HandDetection:
        hand_key = f"{hand_index}:{hand.handedness.casefold()}"
        landmarks = self._filter_landmarks(
            landmarks=hand.landmarks,
            hand_key=hand_key,
            space="image",
            timestamp_ns=timestamp_ns,
        )
        world_landmarks = (
            self._filter_landmarks(
                landmarks=hand.world_landmarks,
                hand_key=hand_key,
                space="world",
                timestamp_ns=timestamp_ns,
            )
            if hand.world_landmarks is not None
            else None
        )
        return replace(hand, landmarks=landmarks, world_landmarks=world_landmarks)

    def _filter_landmarks(
        self,
        *,
        landmarks: tuple[Landmark, ...],
        hand_key: str,
        space: str,
        timestamp_ns: int,
    ) -> tuple[Landmark, ...]:
        return tuple(
            Landmark(
                x=self._filter_value(
                    (hand_key, space, landmark_index, "x"),
                    landmark.x,
                    timestamp_ns=timestamp_ns,
                ),
                y=self._filter_value(
                    (hand_key, space, landmark_index, "y"),
                    landmark.y,
                    timestamp_ns=timestamp_ns,
                ),
                z=self._filter_value(
                    (hand_key, space, landmark_index, "z"),
                    landmark.z,
                    timestamp_ns=timestamp_ns,
                ),
            )
            for landmark_index, landmark in enumerate(landmarks)
        )

    def _filter_value(
        self,
        key: _FilterKey,
        value: float,
        *,
        timestamp_ns: int,
    ) -> float:
        scalar_filter = self._filters.get(key)
        if scalar_filter is None:
            scalar_filter = OneEuroFilter(
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                derivative_cutoff=self.derivative_cutoff,
            )
            self._filters[key] = scalar_filter
        return scalar_filter.filter(value, timestamp_ns=timestamp_ns)


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _smooth(*, current: float, previous: float, alpha: float) -> float:
    return alpha * current + (1.0 - alpha) * previous


def _finite(value: float, label: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite.")
    return normalized


def _positive_finite(value: float, label: str) -> float:
    normalized = _finite(value, label)
    if normalized <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return normalized


def _non_negative_finite(value: float, label: str) -> float:
    normalized = _finite(value, label)
    if normalized < 0.0:
        raise ValueError(f"{label} must be non-negative.")
    return normalized
