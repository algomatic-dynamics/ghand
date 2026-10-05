# Training

`ghand train` is a wrapper around `lerobot-train`. It supplies ACT defaults for
LEAP V1 KAIZEN while forwarding unknown options to LeRobot so advanced settings
remain available.

## Train on a local dataset

```bash
uv run --extra lerobot ghand train \
  --repo-id local/ghand-mujoco \
  --root data/ghand-mujoco
```

The wrapper configures the policy type, training horizon, batch size, ACT chunk
shape, output directory, and device selection. These defaults are defined by the
Typer command and may be overridden explicitly. Use the generated
[CLI reference](../reference/cli.md) or run:

```bash
uv run ghand train --help
```

Unknown LeRobot options are forwarded verbatim. For example, a policy-specific
optimizer setting can be appended to the command without adding a ghand wrapper
option.

## Keep the dataset identity stable

Use the same `repo_id` and `root` pair used for recording. The repository ID is
the dataset identity, while `root` only chooses the local storage location.

Training outputs are policies, not evaluation datasets. Use a separate
`eval_`-prefixed dataset when recording a policy rollout.

## Continue

Evaluate the resulting checkpoint through [Policy rollout](policy-rollout.md).
