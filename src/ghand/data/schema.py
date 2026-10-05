"""Shared data schemas used across capture, replay, and control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageSize:
    """Image dimensions in pixels."""

    width: int
    height: int
