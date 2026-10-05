"""Curl every flexion joint to half of its range and open again, twice.

Flexion is negative in the packaged model, so the curled pose sits halfway
toward each joint's lower limit.

Uses the in-memory fake hand unless GHAND_PORT names a serial port. Keep the
hand clear before running this on real hardware.

Run with:
    uv run python examples/03_open_close.py
"""

from __future__ import annotations

import os

from ghand import Hand


def main() -> None:
    port = os.environ.get("GHAND_PORT")
    hand = Hand.connect(port) if port else Hand.fake()
    with hand:
        curled = {
            name: lower / 2.0
            for name, (lower, _) in hand.joint_limits.items()
            if name.endswith("_flex")
        }
        for cycle in range(2):
            hand.move_to(curled, seconds=1.0)
            print(f"cycle {cycle + 1}: closed")
            hand.move_to("home", seconds=1.0)
            print(f"cycle {cycle + 1}: open")


if __name__ == "__main__":
    main()
