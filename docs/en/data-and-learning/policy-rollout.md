# Policy rollout

`ghand run-policy` wraps `lerobot-record` to evaluate a trained policy against a
MuJoCo, real, or fake LEAP V1 KAIZEN backend while saving the rollout as a
LeRobot dataset.

## MuJoCo rollout

Use the same scene used to record the training data:

```bash
uv run --extra lerobot --extra sim ghand run-policy \
  --policy-path outputs/train/ghand-mujoco-act \
  --mujoco-scene turn_handle \
  --task "turn handle"
```

When `--policy-path` points to a training output root, ghand resolves the latest
checkpoint at `checkpoints/last/pretrained_model`. Camera names and resolutions
come from the policy configuration so rollout observations match training.

## Rollout dataset

By default, rollout data uses a separate identifier and directory shaped like:

```text
repo_id: local/eval_ghand_<timestamp>
root:    data/eval_ghand_<timestamp>
```

LeRobot requires an `eval_` dataset name when a policy is supplied. Do not reuse a
training dataset's `repo_id` and `root` for policy-backed recording. Override both
with `--repo-id` and `--root` when a stable evaluation name is needed.

## Inspect the resolved command

Use `--dry-run` before a long rollout or a real-hardware evaluation:

```bash
uv run --extra lerobot --extra sim ghand run-policy \
  --policy-path outputs/train/ghand-mujoco-act \
  --mujoco-scene turn_handle \
  --dry-run
```

Backend, episode, FPS, dataset, and forwarded LeRobot options are listed in the
[CLI reference](../reference/cli.md).
