# ghand

[日本語](./README_ja.md)

**ghand is the Python SDK to control, teleoperate, simulate, and train with dexterous robot hands.**

[![Algomatic Dynamics](./assets/1500x500.jpeg)](https://algomatic-dynamics.com/)

ghand is the Python SDK for the LEAP V1 KAIZEN robot hand by
[Algomatic Dynamics](https://algomatic-dynamics.com/): a four-finger, 16-DoF
hand in the LEAP Hand v1 lineage, built for robot-learning research. The
package covers joint-level control, an operator Web UI, webcam teleoperation
with MediaPipe, MuJoCo simulation, and LeRobot integration.

<table>
<tr>
<th width="50%">Running <a href="./examples/04_pose_sequence.py"><code>examples/04_pose_sequence.py</code></a></th>
<th width="50%">In-hand cube manipulation</th>
</tr>
<tr>
<td>

https://github.com/user-attachments/assets/ecf324b7-3178-4d55-9744-c4e36d94e451

</td>
<td>

https://github.com/user-attachments/assets/88e3a341-87d5-4e12-80e6-18a117da4da2

</td>
</tr>
<tr>
<th>MuJoCo cube scene (<code>--mujoco-scene cube_face_up</code>)</th>
<th>MuJoCo handle scene (<code>--mujoco-scene turn_handle</code>)</th>
</tr>
<tr>
<td>

https://github.com/user-attachments/assets/91aab1bf-f094-4196-9ea4-8a2599b37b7e

</td>
<td>

https://github.com/user-attachments/assets/f75f754b-fc74-462d-89ac-e0b9c76fcc35

</td>
</tr>
<tr>
<th>Operator Web UI (<code>ghand ui</code>)</th>
<th>Rerun viewer (<code>ghand teleop --rerun</code>)</th>
</tr>
<tr>
<td>

https://github.com/user-attachments/assets/0e11f5f0-5e8d-41b9-95f9-ab420718f0f4

</td>
<td>

https://github.com/user-attachments/assets/6a40e5de-6777-4a09-aa5f-7660fe6131df

</td>
</tr>
</table>

## Core features

- **Python API**: one `Hand` class drives the fake hand, the real hand, and
  MuJoCo with the same code.
- **Web UI**: `ghand ui` puts torque, joint targets, and e-stop in the browser,
  next to a 3D model of the hand.
- **Webcam teleoperation**: `ghand teleop` mirrors your hand, tracked by
  MediaPipe, onto LEAP V1 KAIZEN.
- **Rerun visualization**: `--rerun` shows camera, targets, measured state, and
  latency on one timeline.
- **MuJoCo**: the model is generated from the packaged URDF, with cube and
  handle scenes ready to use.
- **LeRobot integration**: `ghand record`, `ghand train`, and `ghand run-policy`
  cover recording, training, and rollout.
- **Hardware checks**: `ghand doctor` inspects the adapter and all 16 motors
  without moving the hand.
- **LEAP Hand compatible**: same servos, motor IDs, and zero, so code written
  for LEAP Hand runs unchanged.

## Install

ghand supports Python 3.11 and 3.12.

```bash
pip install "ghand[ui,teleop,viewer,sim]"   # or: uv add "ghand[ui,teleop,viewer,sim]"
```

The base package depends on `dynamixel-sdk`, `numpy`, `pyserial`, `rich`, and
`typer`. Add the extras you use: `ui` (Web UI), `teleop` (webcam tracking),
`viewer` (Rerun), `sim` (MuJoCo), `lerobot` (datasets, training, rollout), or
`all`.

## Three commands

```bash
ghand ui                 # operator UI on the in-memory fake hand, no robot needed
ghand doctor             # with the hand plugged in: adapter, 16 motors, firmware, voltage, Homing Offset
ghand ui --port auto     # the same UI on the real hand
```

Open <http://localhost:8080>. `--port auto` picks the single USB serial adapter
that is connected; pass the device path when there are several.

## Real hardware

```bash
ls /dev/serial/by-id/                      # the U2D2 adapter, stable across reboots
PORT=/dev/serial/by-id/<adapter>
sudo chmod 666 $PORT                        # or a udev rule for your user group
echo 1 | sudo tee /sys/bus/usb-serial/devices/$(basename $(readlink -f $PORT))/latency_timer  # Linux only: 1 ms USB latency timer, checked by doctor
ghand doctor                                # adapter, permissions, 16 motors, Homing Offset = 0
ghand ui --port auto                        # operator UI with torque control and e-stop
ghand teleop --camera 0 --port auto --rerun # webcam teleoperation
```

```python
from ghand import Hand

with Hand.connect("/dev/serial/by-id/<adapter>") as hand:   # or Hand.connect() with one adapter
    print(hand.joint_positions())            # measured radians, torque is on
    hand.move_to({"index_mcp_flex": -0.6}, seconds=1.0)   # flexion is negative
    hand.move_to("home", seconds=1.0)                        # home pose: open hand, every joint at 0
# torque is released when the block exits
```

Every call goes through the same runtime as the UI and teleoperation, so joint
limits, velocity limits, and emergency stop apply to scripts too. See
[examples/](./examples/) for runnable scripts.

`Hand.connect(..., dry_run=True)` keeps every write off the bus so a script can
be rehearsed against the real adapter. See
[hardware setup](./docs/en/getting-started/hardware-setup.md) for the full check
list before enabling torque.

## Try it without hardware

- `ghand ui` and `Hand.fake()` drive a fake hand in software: no physics,
  commanded joints read back as measured, and nothing talks to a motor.
- `ghand teleop --mujoco` and `Hand.mujoco()` simulate the hand in MuJoCo; the
  model is exported from the packaged URDF on first use, with no download.

## Teleoperate, record, train

```bash
ghand download-hand-model                                    # MediaPipe hand landmarker, once
ghand teleop --camera 0 --port auto --rerun                  # webcam hand → LEAP V1 KAIZEN
ghand record --repo-id <user>/ghand-demo --port auto         # LeRobot dataset
ghand train --repo-id <user>/ghand-demo                      # lerobot-train with LEAP V1 KAIZEN defaults
ghand run-policy --policy-path outputs/train/<run> --backend mujoco   # roll the trained policy out in MuJoCo
```

Installing `ghand[lerobot]` also installs `lerobot_robot_ghand`, so the LeRobot
tools accept `--robot.type=ghand` directly.

## Coming from LEAP Hand

LEAP V1 KAIZEN is built to be compatible with the LEAP Hand v1: the same
servos, motor IDs, and servo zero, so the LEAP Hand API and code written for it
drive LEAP V1 KAIZEN unchanged. ghand's own Python API names joints and
measures them in radians from the open pose; the `ghand.hand.leap` helpers
convert LEAP vectors to and from that representation. See the
[migration notes](./docs/en/guides/coming-from-leap-hand.md).

## Documentation

- [Documentation home](./docs/en/index.md)
- [Quickstart](./docs/en/getting-started/quickstart.md)
- [Hardware setup](./docs/en/getting-started/hardware-setup.md)
- [Teleoperation](./docs/en/guides/teleoperation.md)
- [LeRobot datasets and learning](./docs/en/data-and-learning/lerobot-dataset.md)
- [MuJoCo](./docs/en/simulation/mujoco.md)
- [CLI reference](./docs/en/reference/cli.md) and
  [Python API reference](./docs/en/reference/python-api/control-and-hardware.md)

## License

Apache-2.0 for the code, the packaged URDF, and the meshes. The meshes are
provided for simulation and visualization; the manufacturing CAD of the hand is
not part of this repository. See [LICENSE](./LICENSE), [NOTICE](./NOTICE), and
[CITATION.cff](./CITATION.cff).

The meshes of ROBOTIS components (the DYNAMIXEL XC330 servo and its frame,
horn, and connector parts) were modeled independently by Algomatic Dynamics,
using ROBOTIS's published drawings only as a dimensional reference. They are
not ROBOTIS CAD data. DYNAMIXEL is a trademark of ROBOTIS Co., Ltd.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the development setup and checks.
