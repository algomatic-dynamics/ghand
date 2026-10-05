"""Python SDK for the LEAP V1 KAIZEN robot hand."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["Hand", "__version__"]

__version__ = "0.1.0"

if TYPE_CHECKING:
    from ghand.control.hand_api import Hand


def __getattr__(name: str) -> object:
    if name == "Hand":
        from ghand.control.hand_api import Hand

        return Hand
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
