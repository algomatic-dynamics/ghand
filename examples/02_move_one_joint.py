"""Flex one joint to a third of its range and return to the packaged zero pose.

Flexion is negative in the packaged model: the lower joint limit is the
fully flexed position and zero is the open hand.

Uses the in-memory fake hand unless GHAND_PORT names a serial port. Keep the
hand clear before running this on real hardware.

Run with:
    uv run python examples/02_move_one_joint.py
"""

from __future__ import annotations

import os

from ghand import Hand

JOINT = "index_mcp_flex"


def main() -> None:
    port = os.environ.get("GHAND_PORT")
    hand = Hand.connect(port) if port else Hand.fake()
    with hand:
        lower, upper = hand.joint_limits[JOINT]
        target = lower / 3.0
        print(f"{JOINT}: range {lower:+.2f}..{upper:+.2f} rad, moving to {target:+.2f} rad")
        hand.move_to({JOINT: target}, seconds=1.0)
        print(f"measured: {hand.joint_positions()[JOINT]:+.3f} rad")
        hand.move_to("home", seconds=1.0)
        print(f"measured after home: {hand.joint_positions()[JOINT]:+.3f} rad")


if __name__ == "__main__":
    main()
