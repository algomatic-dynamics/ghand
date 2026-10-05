"""Read the measured joint positions once.

Uses the in-memory fake hand unless GHAND_PORT names a serial port.

Run with:
    uv run python examples/01_read_joints.py
    GHAND_PORT=/dev/serial/by-id/<adapter> uv run python examples/01_read_joints.py
"""

from __future__ import annotations

import os

from ghand import Hand


def main() -> None:
    port = os.environ.get("GHAND_PORT")
    hand = Hand.connect(port) if port else Hand.fake()
    with hand:
        print(f"backend: {hand.snapshot().backend}")
        for name, radians in hand.joint_positions().items():
            print(f"{name:18s} {radians:+.3f} rad")


if __name__ == "__main__":
    main()
