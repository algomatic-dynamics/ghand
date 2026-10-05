# Calibration and diagnostics

Retargeting calibration records human-hand feature anchors and their robot pose
targets. Keep calibration data separate from Dynamixel zero ticks, directions,
and other motor conversion settings.

## Capture a calibration

Download the MediaPipe model and verify tracking first, then capture the prompted
poses:

```bash
uv run ghand download-hand-model
uv run ghand calibrate \
  --camera 0 \
  --out calibration/operator-right.json \
  --rerun
```

Interactive capture is the default. Follow the pose prompt and only accept a pose
when the intended hand is visible and stable. Use `--timed` for unattended,
time-based capture.

## Inspect captured anchors

```bash
uv run ghand calibrate-check \
  --retarget-calibration calibration/operator-right.json \
  --json-out calibration/operator-right-diagnostics.json
```

The diagnostic output compares captured feature anchors with their robot target
poses. Use it to catch mislabeled or inconsistent captures before moving hardware.

## Use the calibration

Pass the file to teleoperation or dataset recording with
`--retarget-calibration`. Keep the JSON with the experiment metadata so another
operator or robot-model revision does not silently reuse it.

## What calibration does not define

- Dynamixel IDs, zero ticks, direction signs, and ticks-per-radian conversions
- URDF joint axes and limits
- MuJoCo physics parameters
- current limits or runtime safety policy

The standard real-hand mapping is packaged: model q=0 is tick 2048 and
Homing Offset(20) must remain 0. `ghand doctor` verifies
that setting directly, so no separate motor-zero calibration artifact is used.
The other settings belong to the runtime, model, or backend configuration. See
[Hardware setup](../getting-started/hardware-setup.md),
[Architecture](../concepts/architecture.md), and
[Configuration](../reference/configuration.md).
