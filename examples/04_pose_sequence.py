"""Make a fist and pinch the thumb against each finger, opening the hand in between.

The hand opens with its fingers spread, makes a fist, and then touches the
thumb to the index, middle, and ring fingertips in turn. It returns to the
open pose after every one of them.

Pose angles are in degrees, and every joint that a pose leaves out is at zero.
Flexion is negative in the packaged model, and a positive side angle swings a
finger toward the thumb.

Uses the in-memory fake hand unless GHAND_PORT names a serial port. Keep the
hand clear before running this on real hardware. When rerun is installed (the
``viewer`` extra), the Rerun viewer opens and shows the hand moving.

Run with:
    uv run python examples/04_pose_sequence.py
"""

from __future__ import annotations

import importlib.util
import math
import os
from collections.abc import Mapping

from ghand import Hand

MOVE_SECONDS = 0.4
HOLD_SECONDS = 0.3
# The defaults (30 Hz commands, 2 rad/s, a 2.9 rad/s servo profile) are too slow for
# the largest move, 90 degrees, to finish within MOVE_SECONDS.
CONTROL_HZ = 60.0
VELOCITY_LIMIT = 6.0
# Servo profile in Dynamixel units: about 5.8 rad/s and 120 rad/s^2. Without any
# profile the servos jump to every command and the motion turns jerky; a gentler
# acceleration makes the thumb lag behind the fingers on the way into the fist.
PROFILE_VELOCITY = 240
PROFILE_ACCELERATION = 320

POSES = {
    "open": {"ring_mcp_side": -15, "index_mcp_side": 15, "thumb_cmc_side": -30},
    "fist": {
        "ring_mcp_flex": -70,
        "ring_pip_flex": -70,
        "ring_dip_flex": -70,
        "middle_mcp_flex": -70,
        "middle_pip_flex": -70,
        "middle_dip_flex": -70,
        "index_mcp_flex": -70,
        "index_pip_flex": -70,
        "index_dip_flex": -70,
        "thumb_cmc_side": -88,
        "thumb_cmc_flex": -90,
        "thumb_mcp_flex": -56,
        "thumb_ip_flex": -62,
    },
    "index_pinch": {
        "index_mcp_flex": -58,
        "index_pip_flex": -55,
        "index_dip_flex": -37,
        "thumb_cmc_side": -65,
        "thumb_cmc_flex": -34,
        "thumb_mcp_flex": 14,
        "thumb_ip_flex": -88,
    },
    "middle_pinch": {
        "middle_mcp_side": 35,
        "middle_mcp_flex": -45,
        "middle_pip_flex": -54,
        "middle_dip_flex": -45,
        "thumb_cmc_side": -65,
        "thumb_cmc_flex": -34,
        "thumb_mcp_flex": -20,
        "thumb_ip_flex": -66,
    },
    "ring_pinch": {
        "ring_mcp_side": 54,
        "ring_mcp_flex": -48,
        "ring_pip_flex": -54,
        "ring_dip_flex": -45,
        "thumb_cmc_side": -65,
        "thumb_cmc_flex": -54,
        "thumb_mcp_flex": -48,
        "thumb_ip_flex": -48,
    },
}
SEQUENCE = (
    "open",
    "fist",
    "open",
    "index_pinch",
    "open",
    "middle_pinch",
    "open",
    "ring_pinch",
    "open",
)


def full_pose(hand: Hand, degrees: Mapping[str, float]) -> dict[str, float]:
    """Return radians for every joint; joints missing from ``degrees`` are zero."""

    pose = dict.fromkeys(hand.joint_names, 0.0)
    pose.update({name: math.radians(value) for name, value in degrees.items()})
    return pose


def main() -> None:
    port = os.environ.get("GHAND_PORT")
    viewer = importlib.util.find_spec("rerun") is not None
    if port:
        hand = Hand.connect(
            port,
            target_hz=CONTROL_HZ,
            target_velocity_limit=VELOCITY_LIMIT,
            profile_acceleration=PROFILE_ACCELERATION,
            profile_velocity=PROFILE_VELOCITY,
            viewer=viewer,
        )
    else:
        hand = Hand.fake(
            target_hz=CONTROL_HZ,
            target_velocity_limit=VELOCITY_LIMIT,
            viewer=viewer,
        )
    with hand:
        poses = {name: full_pose(hand, degrees) for name, degrees in POSES.items()}
        # Open slowly first, because the hand can start in any pose.
        hand.move_to(poses["open"], seconds=2.0)
        for name in SEQUENCE:
            hand.move_to(poses[name], seconds=MOVE_SECONDS)
            print(name)
            # Commanding the same pose again holds it while the runtime keeps stepping.
            hand.move_to(poses[name], seconds=HOLD_SECONDS)


if __name__ == "__main__":
    main()
