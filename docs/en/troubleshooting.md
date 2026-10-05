# Troubleshooting

## The serial port cannot be opened

- Confirm the device path after reconnecting the adapter.
- Prefer `/dev/serial/by-id/...` over a changing `/dev/ttyUSB*` number.
- Check permissions and the temporary Ubuntu latency setting in
  [Hardware setup](getting-started/hardware-setup.md).
- Make sure another process does not own the port.
- Confirm the configured baud rate before attempting motion.

Use an encoder-only check before enabling torque:

```bash
uv run ghand motor test --port <serial-port> --id <motor-id> --encoder-only
```

## A webcam cannot be opened

Run the smallest camera probe first:

```bash
uv run ghand webcam-check --camera 0
```

Check OS camera permissions, try the actual camera index, and close applications
that may hold exclusive access. Dataset recording can use separate teleoperation
and observation cameras, so test each index independently.

## The MediaPipe model is missing

```bash
uv run ghand download-hand-model
```

If the default cache is unavailable, download the model in an environment with
network access and pass its path with `--model`.

## An optional dependency is missing

Match the command to its extra:

- `ui` for the operator Web UI
- `sim` for MuJoCo
- `lerobot` for dataset/training/rollout integration

See [Installation](getting-started/installation.md).

## A LeRobot policy rollout rejects the dataset name

Policy-backed recording requires an `eval_`-prefixed dataset name. Do not reuse a
training dataset ID for rollout recording. See
[Policy rollout](data-and-learning/policy-rollout.md#rollout-dataset).

## The documented option differs from the CLI

Treat the live command as authoritative and report the documentation mismatch:

```bash
uv run ghand <command> --help
```

The [CLI reference](reference/cli.md) is generated from the same Typer command
tree during every MkDocs build.
