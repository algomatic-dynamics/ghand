"""Fixed color, geometry, and target order for the six cube faces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class CubeFace:
    """One color attached to an outward normal in cube-local coordinates."""

    label: str
    color: str
    normal: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


CUBE_FACE_UP_INITIAL_ORIENTATION_COUNT: Final[int] = 24
CUBE_FACE_UP_FINAL_EVALUATION_SEED_OFFSET: Final[int] = 1_000_000
CUBE_FACE_UP_RED_90_ORIENTATION_IDS: Final[tuple[int, ...]] = (8, 9, 10, 11)
CUBE_FACE_UP_RED_90_DIRECTIONS: Final[tuple[str, ...]] = ("+X", "+Y", "-X", "-Y")
# Per-component world XY millimetres / world XYZ rotation-vector degrees.
CUBE_FACE_UP_JITTER_BOUNDS: Final[tuple[float, ...]] = (0.0, 0.5, 1.0, 2.0, 3.0)
CUBE_FACE_UP_JITTER_SELECTION_SAMPLES: Final[int] = 8
CUBE_FACE_UP_JITTER_FINAL_SAMPLES: Final[int] = 25


def cube_face_up_robustness_episode_count(*, final: bool = False) -> int:
    samples = (
        CUBE_FACE_UP_JITTER_FINAL_SAMPLES
        if final
        else CUBE_FACE_UP_JITTER_SELECTION_SAMPLES
    )
    return 4 + 4 * (len(CUBE_FACE_UP_JITTER_BOUNDS) - 1) * samples


def validate_cube_face_up_training_stage(stage: str) -> None:
    if stage not in {"full", "red_90"}:
        raise ValueError("cube_face_up_training_stage must be full or red_90.")


def cube_face_up_red_90_cases() -> tuple[dict[str, int | str], ...]:
    """Blue up, with the red face pointing along world +X, +Y, -X, -Y."""

    return tuple(
        {"initial_orientation_id": orientation, "target_color": "red"}
        for orientation in CUBE_FACE_UP_RED_90_ORIENTATION_IDS
    )


# Face colours of the exported cube scene, independent of any task definition.
CUBE_SCENE_FACES: Final[tuple[CubeFace, ...]] = (
    CubeFace("A", "red", (1.0, 0.0, 0.0), (0.90, 0.16, 0.16, 1.0)),
    CubeFace("B", "blue", (-1.0, 0.0, 0.0), (0.10, 0.42, 0.86, 1.0)),
    CubeFace("C", "green", (0.0, 1.0, 0.0), (0.12, 0.66, 0.28, 1.0)),
    CubeFace("D", "yellow", (0.0, -1.0, 0.0), (0.94, 0.78, 0.12, 1.0)),
    CubeFace("E", "purple", (0.0, 0.0, 1.0), (0.78, 0.22, 0.78, 1.0)),
    CubeFace("F", "orange", (0.0, 0.0, -1.0), (0.98, 0.48, 0.12, 1.0)),
)

# CubeFaceUp v2: the user's physical cube has these three opposite pairs.
# Tuple order is the target-color one-hot order and the initial-pose face order.
# With white up and green facing the viewer, red is on the viewer's right.
CUBE_FACES: Final[tuple[CubeFace, ...]] = (
    CubeFace("A", "red", (1.0, 0.0, 0.0), (0.90, 0.16, 0.16, 1.0)),
    CubeFace("B", "orange", (-1.0, 0.0, 0.0), (0.98, 0.48, 0.12, 1.0)),
    CubeFace("C", "blue", (0.0, 1.0, 0.0), (0.10, 0.42, 0.86, 1.0)),
    CubeFace("D", "green", (0.0, -1.0, 0.0), (0.12, 0.66, 0.28, 1.0)),
    CubeFace("E", "white", (0.0, 0.0, 1.0), (0.94, 0.94, 0.94, 1.0)),
    CubeFace("F", "yellow", (0.0, 0.0, -1.0), (0.94, 0.78, 0.12, 1.0)),
)


def cube_face_for_color(color: str) -> CubeFace:
    """Resolve an exact color name without substituting an unknown target."""

    for face in CUBE_FACES:
        if face.color == color:
            return face
    raise ValueError(
        f"Unknown cube color {color!r}; expected "
        + ", ".join(face.color for face in CUBE_FACES)
        + "."
    )
