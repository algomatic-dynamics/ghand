# Coming from LEAP Hand

LEAP V1 KAIZEN is built to be compatible with the LEAP Hand v1. It uses the
same Dynamixel servos, the same motor IDs per finger, and the same servo zero,
so the LEAP Hand API and code written for it drive LEAP V1 KAIZEN unchanged.
This page is for the other direction: moving LEAP code onto ghand's own Python
API.

## What carries over

- 16 Dynamixel servos on one bus with the LEAP motor IDs: index 0..3,
  middle 4..7, ring 8..11, thumb 12..15.
- The standard assembly puts the servo mid-point at the open pose, so
  Homing Offset(20) stays 0 on every motor. `ghand doctor` checks this.
- One serial adapter, 1,000,000 bps by default.
- The LEAP Hand API itself: `LeapNode`, `set_leap`, `read_pos`, and the rest
  work on LEAP V1 KAIZEN as they do on a LEAP Hand.

## How ghand represents joints

ghand's Python API does not use raw 16-element vectors. It names joints and
measures them in radians from the open pose, with the limits taken from the
packaged URDF.

| Topic | LEAP Hand API | ghand |
| --- | --- | --- |
| Joint identity | vector index = Dynamixel ID | joint name, e.g. `index_mcp_flex` (`ghand.hand.GHAND_HAND_JOINTS`) |
| Vector order | index, middle, ring, thumb | ring, middle, index, thumb |
| Units | `LEAPhand` raw servo radians centred on pi; `LEAPsim` centred on 0 | radians from the open pose; flexion is negative |
| Limits | fixed in `leap_hand_utils` | taken from the packaged URDF and enforced by the runtime |
| Entry point | `LeapNode.set_leap(...)` | `Hand.set_joint_positions(...)` or `Hand.move_to(...)` |

## Converting vectors

The `ghand.hand.leap` helpers translate between the two representations:

```python
from ghand import Hand
from ghand.hand import leap_sim_vector_to_joint_radians, joint_radians_to_leap_sim_vector

leap_sim = [0.0] * 16                     # a LEAPsim vector from your existing code
joints = leap_sim_vector_to_joint_radians(leap_sim)

with Hand.connect() as hand:
    hand.set_joint_positions(joints)
    hand.step()
    back = joint_radians_to_leap_sim_vector(hand.joint_positions())
```

`leap_hand_vector_to_joint_radians` and `joint_radians_to_leap_hand_vector`
handle the pi-centred `LEAPhand` convention. The conversion reorders the
vector by motor ID and applies ghand's joint direction, so a LEAP-trained policy
can be run through `Hand` by converting its action vector before each step.

## Reading values from the servos

LEAP Hand scripts often read raw ticks. In ghand the same information is
available through the runtime snapshot: `hand.snapshot().joints` carries the
target, commanded, and measured angle in radians and the matching ticks for
every joint.
