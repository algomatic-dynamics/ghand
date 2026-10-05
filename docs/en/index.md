# ghand documentation

ghand is a Python toolkit for operating, teleoperating, simulating, and collecting
learning data for LEAP V1 KAIZEN, a four-finger, 16-DoF robot hand.

!!! warning "Start without hardware"

    ghand can command physical Dynamixel actuators. Start with the fake or MuJoCo
    backend, keep the mechanism clear, and complete the hardware checks before
    enabling torque on a real hand.

## Choose a workflow

| Goal | Start here |
| --- | --- |
| Try the package without a robot | [Safe quickstart](getting-started/quickstart.md) |
| Control joints from Python | [Quickstart, step 2](getting-started/quickstart.md#2-control-the-hand-from-python) |
| Move over from LEAP Hand | [Coming from LEAP Hand](guides/coming-from-leap-hand.md) |
| Connect a physical hand | [Hardware setup](getting-started/hardware-setup.md) |
| Control the hand from a webcam | [Teleoperation](guides/teleoperation.md) |
| Calibrate or inspect retargeting | [Calibration and diagnostics](guides/calibration-and-diagnostics.md) |
| Record LeRobot data | [LeRobot datasets](data-and-learning/lerobot-dataset.md) |
| Train and evaluate a policy | [Training](data-and-learning/training.md) |
| Run the hand in MuJoCo | [MuJoCo](simulation/mujoco.md) |
| Look up a command | [CLI reference](reference/cli.md) |
| Integrate ghand from Python | [Python API reference](reference/python-api/control-and-hardware.md) |

## System outline

The main runtime path converts a camera or policy input into a hand target, applies
the runtime's limits and pacing, and writes it to real hardware or a simulator.
Recording and visualization observe that same path rather than defining separate
control semantics.

See [Architecture](concepts/architecture.md) for the package boundaries and
[Configuration](reference/configuration.md) for the canonical sources of defaults.

## Local documentation preview

From a source checkout:

```bash
uv run --group docs mkdocs serve
```

Open <http://127.0.0.1:8000>. The strict production-equivalent build is:

```bash
uv run --group docs mkdocs build --strict
```
