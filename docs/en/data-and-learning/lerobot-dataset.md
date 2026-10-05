# LeRobot datasets

`record` records the teleoperation runtime directly into a
`LeRobotDataset`. For real hardware, use one camera for MediaPipe control and a
second camera for the robot observation.

## Install

```bash
uv sync --extra lerobot
```

Combine `lerobot` with `sim` when recording from MuJoCo.

## Record a real-hand dataset

```bash
uv run --extra lerobot ghand record \
  --port /dev/ttyUSB0 \
  --repo-id <user-or-org>/ghand-real \
  --root data/ghand-real \
  --teleop-camera 0 \
  --observation-camera 2 \
  --num-episodes 10 \
  --episode-time-s 10 \
  --reset-time-s 5 \
  --fps 10 \
  --task "Open and close the robot hand" \
  --rerun \
  --push-to-hub
```

- `--teleop-camera` supplies the human hand used by MediaPipe and retargeting.
- `--observation-camera` is stored as `observation.images.front`.
- `--fps` sets both the dataset FPS and the capture-loop pacing target.
- Right Arrow finishes the current episode or reset early.
- Left Arrow discards and re-records the current episode.
- Escape stops the recording session.

`repo_id` identifies the dataset. `root` independently selects its local storage
path; a local path does not change the meaning of the repository identifier.

## Record from MuJoCo

```bash
uv run --extra lerobot --extra sim ghand record \
  --mujoco \
  --mujoco-scene turn_handle \
  --repo-id local/ghand-mujoco \
  --root data/ghand-mujoco \
  --teleop-camera 0 \
  --num-episodes 10 \
  --episode-time-s 10 \
  --reset-time-s 0 \
  --fps 10 \
  --rerun \
  --task "Open and close the turn handle in MuJoCo"
```

MuJoCo observation cameras come from the compiled scene. The runtime can reset the
scene between episodes and re-record attempts.

## Action and state semantics

The position schema stores joint positions and a position action. Dataset capture
uses the runtime's `commanded` position—the velocity-limited value sent to the
backend—rather than only the unclipped retargeting request.

The optional `position-current` schema adds explicit current-limit actions and
current-related observations. It requires the corresponding current-position
runtime control mode; current intent is not encoded into radians.

Use the generated [CLI reference](../reference/cli.md) for the current schema,
camera and backend options, and the [data API](../reference/python-api/data.md)
for exact feature names.

## Continue

- [Training](training.md)
- [Policy rollout](policy-rollout.md)
- [MuJoCo](../simulation/mujoco.md)
