# Architecture

ghand keeps input, retargeting, runtime control, backends, recording, and
visualization as separate layers. The CLI composes these layers for operator
workflows.

```text
webcam / policy
       |
       v
teleop tracking and retargeting
       |
       v
joint target adapters and HandOperatorRuntime
       |
       +---------------------+
       |                     |
       v                     v
HandController          simulation backend
       |                (MuJoCo)
       v
DynamixelBus

Runtime snapshots ---> LeRobot recording / Rerun / Web UI
```

## Package boundaries

| Package | Responsibility |
| --- | --- |
| `ghand.hand` | Joint names and order, joint-to-motor calibration, target contracts, the packaged URDF, and LEAP Hand vector conversion |
| `ghand.config` | Shared named workflow and asset-source configuration |
| `ghand.teleop` | Webcam capture, MediaPipe detections, hand tracking, temporal filtering, and hand-frame recording |
| `ghand.retargeting` | Hand features, calibration anchors, diagnostics, and robot joint targets |
| `ghand.control` | Target adapters, joint calibration, runtime pacing, limiting, and snapshots |
| `ghand.hardware` | Dynamixel bus, fake bus, hand controller, telemetry, and motor tests |
| `ghand.data` | LeRobot schemas, runtime-frame conversion, streaming recording, and export |
| `ghand.sim.mujoco` | URDF-to-MJCF export, physics profiles, collision assets, built-in scenes, and the runtime backend |
| `ghand.viewer` | Rerun visualization and URDF pose conversion |
| `ghand.ui` | NiceGUI operator surface and browser model assets |
| `ghand.lerobot_plugin` | LeRobot robot configuration, backend adapter, and policy smoke path |

## Control and observation boundary

`HandOperatorRuntime` owns the state transition from requested targets to values
written to a backend. A runtime snapshot distinguishes the requested/clamped
target from the velocity-limited commanded value and observed motor state.

Dataset capture uses that snapshot so the learning action can represent what the
runtime actually issued. Visualization and telemetry consume the same snapshot,
which makes discrepancies visible without adding a second control path.

## Model and backend boundary

The packaged URDF is the source for hand structure and joint limits. The MuJoCo
exporter derives backend assets from it. Backend-specific physics and
actuation stay in their simulation modules; human-hand feature calibration stays
in `ghand.retargeting`; the joint vocabulary and motor mapping stay in `ghand.hand`.

## Public API boundary

The root `ghand` package currently guarantees only `__version__`. Subpackages
publish their intended surface through `__all__`. The generated
[Python API reference](../reference/python-api/control-and-hardware.md) follows
those package surfaces instead of documenting private helpers from every source
file.
