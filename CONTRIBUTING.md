# Contributing to ghand

Thank you for helping improve ghand. This page covers the development setup,
the checks every change must pass, and where design decisions are recorded.

## Development setup

```bash
git clone https://github.com/algomatic-dynamics/ghand.git
cd ghand
uv sync --extra dev --extra teleop --extra viewer --extra ui --extra sim
```

The `lerobot` extra is optional and large; add it when you work on dataset
recording or policy rollout.

## Checks

Run these before opening a pull request. Continuous integration runs the same
commands.

```bash
uv run ruff check src tests examples scripts
uv run mypy src/ghand/hand src/ghand/hardware src/ghand/control src/ghand/safety
uv run pytest -q
uv run --group docs mkdocs build --strict
```

Tests never require a camera, a serial adapter, or the network. Use the fake
bus, the MuJoCo backend, dependency injection, and injected clocks instead, and
cover rejection and stop paths as well as the happy path.

## Hardware safety

Changes that touch motor writes, joint limits, torque handling, or the motor
mapping must be exercised on the fake bus first, including failure paths. Keep
the hand clear of people and objects whenever torque can be enabled during a
manual test, and describe the manual test in the pull request.

## Documentation

- `docs/` is the user-facing site; keep command examples runnable.
- When you alter a contract such as the joint order, a schema version, or the
  MuJoCo export cache version, update the matching page under `docs/` (the
  architecture and reference pages) in the same change and call the change out
  in the pull request.

## Pull requests

- Keep each pull request focused on one change and explain the motivation.
- Add or update tests for behaviour changes.
- Do not add command-line options for settings that a good default already
  covers; the project keeps its CLI surface small on purpose.
