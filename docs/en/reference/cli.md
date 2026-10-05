# CLI reference

This page is generated from the Typer application in `ghand.cli`. Command names,
arguments, option types, required values, defaults, and help text therefore track
the code used by the installed `ghand` command.

For task-oriented examples, start from the relevant guide. Use this page when
looking up the complete current option surface.

## Simplified defaults

The following options have been removed. Existing commands should omit them;
removed options are reported as errors.

| Commands | Removed options | Current behavior |
| --- | --- | --- |
| `teleop`, `record`, `ui` | `--fake` | No real port or MuJoCo selection uses the fake backend |
| `track`, `calibrate`, `teleop`, `record` | `--max-hands`, `--min-detection-confidence`, `--min-tracking-confidence` | Tracker defaults: two hands, confidence 0.6 / 0.7 |
| `teleop`, `record` | `--landmark-filter`, `--one-euro-min-cutoff`, `--one-euro-beta` | One Euro filtering at 1 Hz minimum cutoff and beta 20 |
| `teleop`, `record` | `--mujoco-actuator-kp`, `--mujoco-actuator-force-limit` | Shared MuJoCo runtime defaults |
| `teleop` | `--readback-hz`, `--profile-acceleration`, `--profile-velocity`, `--debug-targets` | Readback follows the loop rate and real hardware runs without Dynamixel profile limits; use `--rerun` for target and telemetry inspection |
| `record` | `--target-fps` | Capture-loop pacing follows `--fps` |
| `ui` | `--telemetry-hz`, `--viewer-angle-scale` | Telemetry at 1 Hz and unscaled joint angles |

Detailed tuning remains available through the existing Python APIs. Connection,
camera, velocity, current, recording, and missing-hand options remain available.

## Commands

::: mkdocs-typer2
    :module: ghand.cli
    :pretty: true
    :engine: native
