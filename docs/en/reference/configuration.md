# Configuration reference

ghand does not use one global user configuration file. Configuration enters
through package metadata, CLI options, calibration JSON, the packaged robot
model, and backend-specific typed objects.

## Canonical sources

| Configuration | Canonical source |
| --- | --- |
| Python compatibility and dependency extras | `pyproject.toml` |
| Command options and defaults | `ghand.cli` and the [CLI reference](cli.md) |
| Hand geometry and joint limits | packaged right-hand URDF |
| Human-hand feature anchors | retargeting calibration JSON |
| Real-hand motor IDs, 2048-tick q=0, directions, and radians/tick mapping | Fixed packaged mapping plus packaged URDF limits; `ghand doctor` verifies Homing Offset(20)=0 |
| Runtime target/telemetry behavior | `HandOperatorRuntime` configuration |
| MuJoCo dynamics and actuation | `MujocoPhysicsProfile` |
| LeRobot feature keys and shapes | `ghand.data` schema builders |

## Read live defaults from Python

Avoid copying a backend profile into a separate config merely to inspect it:

```python
from ghand.sim.mujoco import DEFAULT_MUJOCO_PHYSICS_PROFILE

print(DEFAULT_MUJOCO_PHYSICS_PROFILE.to_dict())
```

Use the public package objects in the [Python API reference](python-api/simulation.md)
for programmatic integration.

## Calibration files

Retargeting calibration JSON captures the relationship between an operator's hand
features and robot pose targets. It does not replace the fixed real-motor mapping
or URDF joint definitions. See
[Calibration and diagnostics](../guides/calibration-and-diagnostics.md).

## Generated simulation assets

MuJoCo cache manifests record the resolved model/profile contract used to produce
each built-in scene. Preserve these artifacts with experiment
metadata when exact reproduction matters.
