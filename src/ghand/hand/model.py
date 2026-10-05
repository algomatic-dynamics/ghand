"""Packaged hand model: URDF location and display orientation."""

from __future__ import annotations

import math
from importlib.resources import files
from pathlib import Path
from typing import Final


DEFAULT_RIGHT_HAND_URDF: Final[str] = "urdf/ghand_right.urdf"

# Rotation that presents the packaged right-hand model palm-up in viewers.
DISPLAY_HAND_ROTATION_EULER_X_RAD: Final[float] = math.pi
DISPLAY_HAND_ROTATION_RPY: Final[tuple[float, float, float]] = (math.pi, 0.0, 0.0)
DISPLAY_HAND_ROTATION_QUAT_WXYZ: Final[tuple[float, float, float, float]] = (
    0.0,
    1.0,
    0.0,
    0.0,
)
DISPLAY_HAND_ROTATION_QUAT_XYZW: Final[tuple[float, float, float, float]] = (
    1.0,
    0.0,
    0.0,
    0.0,
)


def default_right_hand_urdf_path() -> Path:
    """Return the packaged right-hand URDF path."""

    return Path(str(files("ghand.assets") / "models" / DEFAULT_RIGHT_HAND_URDF))
