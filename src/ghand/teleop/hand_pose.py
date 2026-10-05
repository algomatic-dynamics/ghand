"""Device-independent hand-pose models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


@dataclass(frozen=True)
class Landmark:
    """A normalized 3D hand landmark."""

    x: float
    y: float
    z: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Landmark":
        """Create a landmark from a JSON-style dictionary."""

        return cls(x=float(data["x"]), y=float(data["y"]), z=float(data["z"]))


@dataclass(frozen=True)
class HandDetection:
    """Landmarks and classification for one detected hand."""

    handedness: str
    score: float
    landmarks: tuple[Landmark, ...]
    world_landmarks: tuple[Landmark, ...] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HandDetection":
        """Create a hand detection from a JSON-style dictionary."""

        world_landmarks = data.get("world_landmarks")
        return cls(
            handedness=str(data["handedness"]),
            score=float(data["score"]),
            landmarks=tuple(Landmark.from_dict(landmark) for landmark in data["landmarks"]),
            world_landmarks=(
                tuple(Landmark.from_dict(landmark) for landmark in world_landmarks)
                if world_landmarks is not None
                else None
            ),
        )


@dataclass(frozen=True)
class HandFrame:
    """Hand tracking result for one camera frame."""

    timestamp_ns: int
    image_width: int
    image_height: int
    hands: tuple[HandDetection, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HandFrame":
        """Create a hand frame from a JSON-style dictionary."""

        return cls(
            timestamp_ns=int(data["timestamp_ns"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            hands=tuple(HandDetection.from_dict(hand) for hand in data["hands"]),
        )
