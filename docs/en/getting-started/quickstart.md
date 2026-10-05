# Quickstart

This sequence verifies the installation and the operator surface without
commanding a physical robot, then shows the first steps on a real hand.

## 1. Launch the fake operator UI

From a source checkout:

```bash
uv run --extra ui ghand ui
```

From a PyPI installation with the `ui` extra:

```bash
ghand ui
```

Open <http://localhost:8080>. The UI uses an in-memory bus, so joint commands and
telemetry do not leave the process.

## 2. Control the hand from Python

```python
from ghand import Hand

hand = Hand.fake()
with hand:
    lower, upper = hand.joint_limits["index_mcp_flex"]
    hand.move_to({"index_mcp_flex": lower / 2}, seconds=1.0)
    print(hand.joint_positions()["index_mcp_flex"])
    hand.move_to("home", seconds=1.0)
```

Joint positions are radians keyed by short joint names; flexion is negative and
zero is the packaged open pose. `Hand.connect()` targets the real hand and
`Hand.mujoco()` the simulation, with the same methods. `Hand.fake(viewer=True)`
opens the Rerun viewer (`viewer` extra) and logs every snapshot the hand takes,
which also works with the other constructors. The scripts in
`examples/` run on the fake backend by default and on the real hand when
`GHAND_PORT` names its serial port.

## 3. Check the real hand

Plug in the hand and run the hardware check. It reads every motor without moving
anything.

```bash
uv run ghand doctor
```

`doctor` picks the single USB serial adapter it finds; pass `--port` when there
are several. It reports the adapter, port permissions, the Linux USB latency
timer, the 16 motor IDs, firmware versions, Homing Offset, supply voltage,
temperature, and hardware errors. Fix every failed row before enabling torque,
then open the UI on the real hand:

```bash
uv run --extra ui ghand ui --port auto
```

## 4. Check a webcam

```bash
uv run ghand webcam-check --camera 0
```

If the check fails, see [Troubleshooting](../troubleshooting.md).

## 5. Download the hand model and inspect tracking

```bash
uv run ghand download-hand-model
uv run ghand track --camera 0 --display
```

Press `q` in the OpenCV window to exit hand tracking.

## Choose the next backend

- For physical hardware, complete [Hardware setup](hardware-setup.md).
- For webcam-to-hand control, continue to [Teleoperation](../guides/teleoperation.md).
- For a robot-free control backend, continue to [MuJoCo](../simulation/mujoco.md).
