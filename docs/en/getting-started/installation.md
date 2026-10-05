# Installation

ghand supports Python `>=3.11,<3.13`. Install only the optional feature sets needed
for the workflow you intend to run.

## Install from PyPI

The base package includes the Dynamixel bus, the control runtime, the `Hand`
facade, and the `ghand doctor` and `ghand motor` commands. Its only heavy
dependency is NumPy:

```bash
pip install ghand
```

Add one or more extras for the features you use:

| Extra | Adds |
| --- | --- |
| `teleop` | MediaPipe hand tracking and OpenCV webcam capture |
| `viewer` | Rerun diagnostics and mesh preparation |
| `ui` | NiceGUI operator Web UI |
| `sim` | MuJoCo simulation backend and asset export |
| `lerobot` | LeRobot dataset, training, and rollout integration, plus the `lerobot_robot_ghand` plugin |
| `all` | Every extra above |

For example:

```bash
pip install "ghand[ui,teleop,viewer,sim]"
```

After installation, use the single `ghand` console entrypoint:

```bash
ghand --help
```

## Install from source

The repository uses `uv` for dependency management and command execution.

```bash
git clone https://github.com/algomatic-dynamics/ghand.git
cd ghand
uv sync
uv run ghand --help
```

Sync optional features explicitly:

```bash
uv sync --extra ui --extra teleop --extra viewer --extra sim
uv sync --extra lerobot
uv sync --extra dev --group docs
```

Commands in the rest of this documentation use `uv run` when they are intended
for a source checkout. Remove the `uv run` prefix when using an installed console
script outside the checkout.

## MediaPipe model

The hand landmarker model is downloaded separately on first setup:

```bash
uv run ghand download-hand-model
```

You can also pass a model path explicitly to commands that accept `--model`.

## Next steps

- [Safe quickstart](quickstart.md)
- [Hardware setup](hardware-setup.md)
- [Troubleshooting](../troubleshooting.md)
