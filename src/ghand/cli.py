"""Command line entrypoint for ghand."""

from __future__ import annotations

import json
import math
import shlex
import subprocess
import os
import sys
import time
from datetime import datetime
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import median

import typer
from rich.console import Console
from rich.table import Table

from ghand import __version__
from ghand.control import (
    DEFAULT_OPERATOR_TARGET_HZ,
    HandOperatorRuntime,
    OPERATOR_CONTROL_MODE_CURRENT_POSITION,
    OPERATOR_CONTROL_MODE_POSITION,
    OperatorControlMode,
    OperatorRuntimeSnapshot,
    apply_joint_target,
    make_fake_hand_runtime,
)
from ghand.hand import (
    default_right_hand_urdf_path,
    validate_real_hand_homing_offsets,
    GHAND_RIGHT_MOTOR_IDS,
    JointTarget,
    make_default_joint_calibrations,
    RadianJointTarget,
)
from ghand.hardware import (
    AUTO_PORT,
    DEFAULT_BAUDRATE,
    SerialPortError,
    resolve_serial_port,
    usb_serial_latency_timer_ms,
    DEFAULT_MOTOR_TEST_DELTA_TICKS,
    DynamixelBusError,
    HandController,
    MotorEncoderState,
    MotorTestResult,
)
from ghand.sim.mujoco import (
    DEFAULT_MUJOCO_CONTROL_HZ,
    DEFAULT_MUJOCO_RUNTIME_ACTUATOR_FORCE_LIMIT,
    DEFAULT_MUJOCO_RUNTIME_ACTUATOR_KP,
    DEFAULT_MUJOCO_TARGET_VELOCITY_LIMIT,
)
from ghand.data import (
    DEFAULT_CURRENT_CONTACT_THRESHOLD_MA,
    DEFAULT_FPS,
    DEFAULT_FRONT_IMAGE_SHAPE,
    CameraSpec,
    LeRobotDependencyError,
    LeRobotStreamingRecorder,
    OBSERVATION_IMAGES_PREFIX,
    make_hand_lerobot_features,
    make_hand_position_current_lerobot_features,
    make_lerobot_frame_from_runtime_snapshot,
)
from ghand.teleop.filters import HandFrameOneEuroSmoother
from ghand.teleop.hand_tracking import OperatorHandSelection, OperatorHandSelector
from ghand.teleop.mediapipe_hands import (
    HAND_LANDMARKER_MODEL_URL,
    HandModelNotFoundError,
    MediaPipeHandTracker,
    download_hand_landmarker_model,
    draw_hand_frame,
    resolve_hand_landmarker_model,
)
from ghand.retargeting import (
    FINGER_CURL_JOINT_NAMES,
    FINGER_NAMES,
    PINCH_FINGER_NAMES,
    RETARGET_CALIBRATION_POSES,
    RETARGET_CALIBRATION_SCHEMA_VERSION,
    DEFAULT_ANCHOR_SOFTMAX_TEMPERATURE,
    SPREAD_FINGER_NAMES,
    THUMB_POSE_FEATURE_NAMES,
    AnchorPoseRetargeter,
    CurlCalibration,
    HandFeatureCalibrator,
    HandFeatures,
    HandRetargetCalibration,
    PinchCalibration,
    PoseFeatureAnchor,
    RetargetPoseStats,
    SpreadCalibration,
    ThumbAbductionCalibration,
    ThumbPoseAnchor,
    diagnose_hand_retarget_calibration,
    RETARGET_CALIBRATION_CAPTURE_POSES,
    RetargetCalibrationPoseInstruction,
    extract_hand_features,
    load_default_hand_retarget_calibration,
    load_hand_retarget_calibration,
    save_hand_retarget_calibration,
)
from ghand.teleop.webcam import Webcam
from ghand.teleop.webcam import WebcamError, check_webcam
from ghand.viewer import RerunViewer

app = typer.Typer(
    name="ghand",
    help="LEAP V1 KAIZEN hand SDK: control, teleoperation, simulation, and learning tooling.",
    no_args_is_help=True,
)
motor_app = typer.Typer(help="Dynamixel motor checks on the bus.", no_args_is_help=True)
export_app = typer.Typer(
    help="Export simulation assets from the packaged URDF.", no_args_is_help=True
)
app.add_typer(motor_app, name="motor")
app.add_typer(export_app, name="export")
console = Console()
CalibrationKeyReader = Callable[[], str | None]
TELEOP_RERUN_UI_EVERY_FRAMES = 10
TELEOP_CONSOLE_EVERY_FRAMES = 30
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_TELEOP_TARGET_FPS = 15.0
DEFAULT_TELEOP_HOLD_MISSING_SECONDS = 0.5
DEFAULT_MUJOCO_TELEOP_TARGET_FPS = DEFAULT_MUJOCO_CONTROL_HZ
DEFAULT_TELEOP_TARGET_VELOCITY_LIMIT = 1000.0
DEFAULT_MUJOCO_TELEOP_TARGET_VELOCITY_LIMIT = DEFAULT_MUJOCO_TARGET_VELOCITY_LIMIT
# Dynamixel treats a profile value of 0 as "no limit". Teleop runs without a
# velocity/acceleration profile so the hand follows the operator at full motor
# speed, as the LEAP Hand API does; the operator UI keeps the slower defaults.
DEFAULT_TELEOP_PROFILE_ACCELERATION = 0
DEFAULT_TELEOP_PROFILE_VELOCITY = 0
DEFAULT_TELEOP_ONE_EURO_MIN_CUTOFF = 1.0
DEFAULT_TELEOP_ONE_EURO_BETA = 20.0
DEFAULT_TELEOP_ONE_EURO_DERIVATIVE_CUTOFF = 1.0
DEFAULT_TELEOP_LANDMARK_FILTER_RESET_SECONDS = 0.25
DEFAULT_RETARGET_CALIBRATION_MIN_VALID_SAMPLES = 5
DEFAULT_LEROBOT_RECORD_NUM_EPISODES = 50
DEFAULT_LEROBOT_RECORD_EPISODE_TIME_S = 60.0
DEFAULT_LEROBOT_RECORD_RESET_TIME_S = 60.0
DEFAULT_TRAIN_POLICY_TYPE = "act"
DEFAULT_TRAIN_STEPS = 5_000
DEFAULT_TRAIN_BATCH_SIZE = 8
DEFAULT_TRAIN_NUM_WORKERS = 0
DEFAULT_TRAIN_LOG_FREQ = 100
DEFAULT_TRAIN_SAVE_FREQ = 500
DEFAULT_POLICY_CHUNK_SIZE = 20
DEFAULT_POLICY_N_ACTION_STEPS = 10
DEFAULT_RUN_POLICY_FPS = int(DEFAULT_MUJOCO_CONTROL_HZ)
DEFAULT_RUN_POLICY_NUM_EPISODES = 1
DEFAULT_RUN_POLICY_EPISODE_TIME_S = 10.0
LEROBOT_TRAIN_MODULE = "lerobot.scripts.lerobot_train"
LEROBOT_RECORD_MODULE = "lerobot.scripts.lerobot_record"


class SingleHandSideOption(str, Enum):
    """CLI choices for single-hand teleoperation."""

    right = "right"
    left = "left"


class HandednessOption(str, Enum):
    """CLI choices for MediaPipe handedness labels."""

    right = "Right"
    left = "Left"


class OperatorControlModeOption(str, Enum):
    """CLI choices for hand-runtime control mode."""

    position = "position"
    current_position = "current-position"

    @property
    def runtime_value(self) -> OperatorControlMode:
        """Return the internal HandOperatorRuntime control-mode value."""

        if self is OperatorControlModeOption.current_position:
            return OPERATOR_CONTROL_MODE_CURRENT_POSITION
        return OPERATOR_CONTROL_MODE_POSITION


class DatasetSchemaOption(str, Enum):
    """CLI choices for LeRobot dataset joint/action schema."""

    position = "position"
    position_current = "position-current"


class LeRobotPluginBackendOption(str, Enum):
    """CLI choices for the LEAP V1 KAIZEN LeRobot robot plugin backend."""

    fake = "fake"
    mujoco = "mujoco"
    real = "real"


@dataclass(frozen=True)
class TeleopHandResult:
    """Summary of one teleoperation run."""

    processed: int
    frames_with_hands: int
    missing_home_events: int = 0


@dataclass(frozen=True)
class RetargetCalibrationLoadResult:
    """Loaded retarget calibration and a human-readable source label."""

    calibration: HandRetargetCalibration
    calibrator: HandFeatureCalibrator
    source: str


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ghand {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        help="Show the ghand version and exit.",
        is_eager=True,
    ),
) -> None:
    """Run ghand commands."""


# XC330-T288-T on a 12 V supply, +/-10%.
DOCTOR_MIN_INPUT_VOLTAGE = 10.8
DOCTOR_MAX_INPUT_VOLTAGE = 13.2
DOCTOR_MAX_TEMPERATURE_CELSIUS = 60
DOCTOR_MAX_LATENCY_TIMER_MS = 1


class _DoctorAbort(Exception):
    """Stop the remaining hardware checks after a failed prerequisite."""


def _resolve_port_option(port: str | None) -> str | None:
    """Resolve a ``--port`` value, treating ``auto`` as the single adapter found."""

    if port != AUTO_PORT:
        return port
    try:
        return resolve_serial_port(port)
    except SerialPortError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command("doctor")
def doctor(
    port: str | None = typer.Option(
        None,
        "--port",
        help="Dynamixel serial port. Defaults to the single USB serial adapter found.",
    ),
    baudrate: int = typer.Option(
        DEFAULT_BAUDRATE,
        "--baudrate",
        min=1,
        help="Dynamixel bus baudrate.",
    ),
) -> None:
    """Check the adapter, motor IDs, firmware, supply, temperature, and Homing Offset without moving the hand."""

    rows: list[tuple[str, bool, str]] = []
    try:
        device = resolve_serial_port(port)
    except SerialPortError as exc:
        rows.append(("serial adapter", False, str(exc)))
        _print_doctor_rows(rows)
        raise typer.Exit(code=1) from exc
    rows.append(("serial adapter", True, device))
    accessible = os.access(device, os.R_OK | os.W_OK)
    rows.append(
        (
            "port permissions",
            accessible,
            "read/write"
            if accessible
            else "no read/write access; see the hardware setup guide",
        )
    )
    latency = usb_serial_latency_timer_ms(device)
    if latency is not None:
        latency_ok = latency <= DOCTOR_MAX_LATENCY_TIMER_MS
        rows.append(
            (
                "usb latency_timer",
                latency_ok,
                f"{latency} ms"
                + (
                    ""
                    if latency_ok
                    else f"; set it to {DOCTOR_MAX_LATENCY_TIMER_MS} ms for responsive control"
                ),
            )
        )
    ids = GHAND_RIGHT_MOTOR_IDS
    try:
        with HandController.from_port(port=device, baudrate=baudrate) as controller:
            missing: list[str] = []
            for dxl_id in ids:
                try:
                    controller.read_motor_encoder_state(dxl_id)
                except (DynamixelBusError, ValueError) as exc:
                    missing.append(f"ID {dxl_id}: {exc}")
            rows.append(
                (
                    "motor IDs",
                    not missing,
                    f"{len(ids) - len(missing)}/{len(ids)} responded"
                    + ("" if not missing else "; " + "; ".join(missing)),
                )
            )
            if missing:
                raise _DoctorAbort
            versions = sorted(set(controller.read_firmware_versions(ids).values()))
            rows.append(
                (
                    "firmware",
                    len(versions) == 1,
                    "version "
                    + ", ".join(str(version) for version in versions)
                    + ("" if len(versions) == 1 else " (mixed)"),
                )
            )
            try:
                validate_real_hand_homing_offsets(controller.read_homing_offsets(ids))
            except ValueError as exc:
                rows.append(("homing offset", False, str(exc)))
            else:
                rows.append(("homing offset", True, "0 on every motor"))
            telemetry = controller.read_motor_telemetry(ids)
            voltages = [sample.input_voltage for sample in telemetry]
            temperatures = [sample.temperature_celsius for sample in telemetry]
            faults = [sample for sample in telemetry if sample.hardware_error_status]
            voltage_ok = (
                DOCTOR_MIN_INPUT_VOLTAGE <= min(voltages)
                and max(voltages) <= DOCTOR_MAX_INPUT_VOLTAGE
            )
            rows.append(
                (
                    "supply voltage",
                    voltage_ok,
                    f"{min(voltages):.1f}-{max(voltages):.1f} V"
                    + (
                        ""
                        if voltage_ok
                        else f"; expected {DOCTOR_MIN_INPUT_VOLTAGE}-{DOCTOR_MAX_INPUT_VOLTAGE} V"
                    ),
                )
            )
            rows.append(
                (
                    "temperature",
                    max(temperatures) < DOCTOR_MAX_TEMPERATURE_CELSIUS,
                    f"max {max(temperatures)} C",
                )
            )
            rows.append(
                (
                    "hardware errors",
                    not faults,
                    "none"
                    if not faults
                    else ", ".join(
                        f"ID {sample.dxl_id}=0x{sample.hardware_error_status:02x}"
                        for sample in faults
                    ),
                )
            )
    except _DoctorAbort:
        pass
    except (DynamixelBusError, ValueError, OSError) as exc:
        rows.append(("bus", False, str(exc)))
    _print_doctor_rows(rows)
    if not all(ok for _, ok, _ in rows):
        raise typer.Exit(code=1)


def _print_doctor_rows(rows: list[tuple[str, bool, str]]) -> None:
    table = Table(title="ghand doctor")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail")
    for name, ok, detail in rows:
        table.add_row(name, "[green]OK[/green]" if ok else "[red]FAIL[/red]", detail)
    console.print(table)
    failures = sum(1 for _, ok, _ in rows if not ok)
    console.print("All checks passed." if failures == 0 else f"{failures} check(s) failed.")


@app.command("webcam-check")
def webcam_check(
    camera: int = typer.Option(0, "--camera", "-c", help="Camera index to open."),
    camera_width: int = typer.Option(
        DEFAULT_CAMERA_WIDTH,
        "--camera-width",
        min=1,
        help="Requested camera capture width in pixels.",
    ),
    camera_height: int = typer.Option(
        DEFAULT_CAMERA_HEIGHT,
        "--camera-height",
        min=1,
        help="Requested camera capture height in pixels.",
    ),
    camera_fps: float | None = typer.Option(
        None,
        "--camera-fps",
        min=1e-9,
        help="Requested camera capture frame rate.",
    ),
) -> None:
    """Verify that OpenCV can open a webcam and read one frame."""

    try:
        info = check_webcam(
            camera=camera,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        )
    except WebcamError as exc:
        console.print(f"[red]Webcam check failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Webcam Check")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("camera", str(info.camera))
    table.add_row("capture size", f"{info.width} x {info.height}")
    table.add_row("frame size", f"{info.frame_width} x {info.frame_height}")
    table.add_row("channels", str(info.channels))
    table.add_row("fps", f"{info.fps:.2f}")
    table.add_row("backend", info.backend or "unknown")
    console.print(table)
    console.print("[green]Webcam check passed.[/green]")


@app.command("download-hand-model")
def download_hand_model(
    output: Path | None = typer.Option(
        None,
        "--out",
        help="Destination path. Defaults to the ghand cache directory.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Download even if the destination already exists.",
    ),
) -> None:
    """Download the MediaPipe hand landmarker model."""

    console.print(f"Downloading MediaPipe hand model from {HAND_LANDMARKER_MODEL_URL}")
    path = download_hand_landmarker_model(output, overwrite=overwrite)
    console.print(f"[green]Model ready:[/green] {path}")


@app.command("track")
def hand_track(
    camera: int = typer.Option(0, "--camera", "-c", help="Camera index to open."),
    camera_width: int = typer.Option(
        DEFAULT_CAMERA_WIDTH,
        "--camera-width",
        min=1,
        help="Requested camera capture width in pixels.",
    ),
    camera_height: int = typer.Option(
        DEFAULT_CAMERA_HEIGHT,
        "--camera-height",
        min=1,
        help="Requested camera capture height in pixels.",
    ),
    camera_fps: float | None = typer.Option(
        None,
        "--camera-fps",
        min=1e-9,
        help="Requested camera capture frame rate.",
    ),
    frames: int = typer.Option(
        0,
        "--frames",
        "-n",
        min=0,
        help="Number of frames to process. Use 0 to run until interrupted.",
    ),
    display: bool = typer.Option(
        False,
        "--display",
        help="Show an OpenCV overlay window. Press q to quit.",
    ),
    output: Path | None = typer.Option(
        None,
        "--out",
        help="Optional JSONL file for tracked hand frames.",
    ),
    model: Path | None = typer.Option(
        None,
        "--model",
        help="MediaPipe hand_landmarker.task path.",
    ),
) -> None:
    """Track MediaPipe hand landmarks from a webcam."""

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    frames_with_hands = 0
    display_enabled = display
    output_file = output.open("w", encoding="utf-8") if output is not None else None

    try:
        model_path = resolve_hand_landmarker_model(model)
        with Webcam(
            camera=camera,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        ) as webcam, MediaPipeHandTracker(model_path=model_path) as tracker:
            console.print(
                f"Tracking hands from camera {camera} with {model_path}. "
                "Press Ctrl+C to stop"
                + (" or q in the display window" if display else "")
                + "."
            )

            while frames == 0 or processed < frames:
                frame = webcam.read()
                hand_frame = tracker.track(frame)
                processed += 1
                if hand_frame.hands:
                    frames_with_hands += 1

                if output_file is not None:
                    output_file.write(json.dumps(hand_frame.to_dict()) + "\n")

                if processed == 1 or hand_frame.hands:
                    console.print(_hand_frame_summary(processed, hand_frame.hands))

                if display_enabled:
                    display_enabled, stop_requested = _show_hand_frame(
                        "ghand track",
                        frame,
                        hand_frame,
                    )
                    if stop_requested:
                        break
    except KeyboardInterrupt:
        console.print("Stopped.")
    except HandModelNotFoundError as exc:
        console.print(f"[red]Hand tracking failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except WebcamError as exc:
        console.print(f"[red]Hand tracking failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        if output_file is not None:
            output_file.close()
        _close_display_windows(display_enabled)

    console.print(
        f"Processed {processed} frame(s); detected hands in {frames_with_hands} frame(s)."
    )


@app.command("calibrate")
def calibrate_hand_retarget(
    camera: int = typer.Option(0, "--camera", "-c", help="Camera index to open."),
    camera_width: int = typer.Option(
        DEFAULT_CAMERA_WIDTH,
        "--camera-width",
        min=1,
        help="Requested camera capture width.",
    ),
    camera_height: int = typer.Option(
        DEFAULT_CAMERA_HEIGHT,
        "--camera-height",
        min=1,
        help="Requested camera capture height.",
    ),
    output: Path = typer.Option(
        ...,
        "--out",
        help="Destination JSON path for the captured retarget calibration.",
    ),
    handedness: HandednessOption = typer.Option(
        HandednessOption.right,
        "--handedness",
        case_sensitive=False,
        help="MediaPipe handedness label to calibrate.",
    ),
    seconds_per_pose: float = typer.Option(
        2.0,
        "--seconds-per-pose",
        min=0.0,
        help="Seconds to sample each pose. Ignored when --frames-per-pose is set.",
    ),
    frames_per_pose: int = typer.Option(
        0,
        "--frames-per-pose",
        min=0,
        help="Exact frames to sample per pose. Use 0 to sample by time.",
    ),
    ready_delay_seconds: float = typer.Option(
        2.0,
        "--ready-delay-seconds",
        min=0.0,
        help="Seconds to wait before sampling in --timed mode.",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--timed",
        help=(
            "Use terminal keyboard controls for pose capture. "
            "Use --timed for automatic capture."
        ),
    ),
    target_fps: float = typer.Option(
        15.0,
        "--target-fps",
        min=0.0,
        help="Loop pacing target in frames per second. Use 0 for no pacing.",
    ),
    enable_rerun: bool = typer.Option(
        False,
        "--rerun/--no-rerun",
        help=(
            "Stream camera, landmarks, pose instructions, and features to Rerun, "
            "spawning a viewer when needed."
        ),
    ),
    rerun_save: Path | None = typer.Option(
        None,
        "--rerun-save",
        help="Optional .rrd path for saving the Rerun recording.",
    ),
    model: Path | None = typer.Option(
        None,
        "--model",
        help="MediaPipe hand_landmarker.task path.",
    ),
) -> None:
    """Capture hand poses and save a retargeting calibration JSON file."""

    if frames_per_pose == 0 and seconds_per_pose <= 0:
        console.print(
            "[red]Use --frames-per-pose or a positive --seconds-per-pose.[/red]"
        )
        raise typer.Exit(code=1)
    if interactive and not _stdin_supports_interactive_keys():
        console.print(
            "[red]Interactive retarget calibration requires a terminal.[/red] "
            "Use --timed for automatic capture."
        )
        raise typer.Exit(code=1)

    min_valid_samples = DEFAULT_RETARGET_CALIBRATION_MIN_VALID_SAMPLES
    if frames_per_pose > 0:
        min_valid_samples = min(min_valid_samples, frames_per_pose)

    rerun_enabled = enable_rerun or rerun_save is not None
    if interactive and not rerun_enabled:
        console.print(
            "[yellow]Interactive calibration has no visual preview unless "
            "--rerun is enabled.[/yellow]"
        )
    viewer = RerunViewer(
        enabled=rerun_enabled,
        spawn=enable_rerun,
        save_path=rerun_save,
        urdf_path=None,
    )

    try:
        model_path = resolve_hand_landmarker_model(model)
        with (
            Webcam(
                camera=camera,
                width=camera_width,
                height=camera_height,
            ) as webcam,
            MediaPipeHandTracker(model_path=model_path) as tracker,
        ):
            console.print(
                f"Calibrating {handedness.value} hand retargeting from camera {camera} "
                f"with {model_path}."
            )
            viewer.start()
            viewer.log_text("calibrate started")
            calibration = _run_hand_retarget_calibration_loop(
                webcam=webcam,
                tracker=tracker,
                viewer=viewer,
                handedness=handedness.value,
                seconds_per_pose=seconds_per_pose,
                frames_per_pose=frames_per_pose,
                ready_delay_seconds=ready_delay_seconds,
                min_valid_samples=min_valid_samples,
                target_fps=target_fps,
                interactive=interactive,
            )
            viewer.log_text("calibrate stopped")
        save_hand_retarget_calibration(calibration, output)
    except KeyboardInterrupt:
        console.print("Stopped.")
        raise typer.Exit(code=1) from None
    except HandModelNotFoundError as exc:
        console.print(f"[red]Retarget calibration failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except WebcamError as exc:
        console.print(f"[red]Retarget calibration failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]Retarget calibration failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[green]Retarget calibration saved.[/green]")
    console.print(f"path: {output}")
    _print_retarget_calibration_summary(calibration)


@app.command("calibrate-check")
def diagnose_hand_retarget(
    retarget_calibration: Path = typer.Option(
        ...,
        "--retarget-calibration",
        help="Retarget calibration JSON path captured by calibrate.",
    ),
    anchor_pose_names: list[str] | None = typer.Option(
        None,
        "--anchor-pose",
        help=(
            "Limit retargeting to this captured pose name. "
            "Repeat to enable multiple anchors. Defaults to all captured anchors."
        ),
    ),
    hand_side: SingleHandSideOption = typer.Option(
        SingleHandSideOption.right,
        "--hand-side",
        case_sensitive=False,
        help="Robot hand side used for normalized target names.",
    ),
    json_output: Path | None = typer.Option(
        None,
        "--json-out",
        help="Optional JSON path for the diagnostic rows.",
    ),
) -> None:
    """Inspect captured retargeting anchors against robot pose targets."""

    try:
        enabled_pose_names = _resolve_enabled_anchor_pose_names(anchor_pose_names)
        calibration = load_hand_retarget_calibration(retarget_calibration)
        rows = diagnose_hand_retarget_calibration(
            calibration,
            side=hand_side.value,
            enabled_pose_names=enabled_pose_names,
        )
    except ValueError as exc:
        console.print(f"[red]Retarget diagnosis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"[red]Retarget diagnosis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not rows:
        console.print("[yellow]No captured pose anchors found.[/yellow]")
    else:
        table = Table(title="Hand Retarget Diagnostics")
        table.add_column("Pose")
        table.add_column("Group")
        table.add_column("Samples", justify="right")
        table.add_column("Affected mean", justify="right")
        table.add_column("Affected max", justify="right")
        table.add_column("All max", justify="right")
        table.add_column("Worst joint")
        for row in rows:
            table.add_row(
                row.pose_name,
                row.pose_group,
                str(row.samples),
                f"{row.affected_mean_abs_error:.4f}",
                f"{row.affected_max_abs_error:.4f}",
                f"{row.all_max_abs_error:.4f}",
                f"{row.worst_joint} ({row.worst_joint_abs_error:.4f})",
            )
        console.print(table)

    console.print(f"retarget calibration: {retarget_calibration}")
    console.print("retargeter: anchor-pose")
    if enabled_pose_names is not None:
        enabled_pose_text = ", ".join(enabled_pose_names)
        console.print(f"enabled anchor poses: {enabled_pose_text}")
    console.print(f"poses diagnosed: {len(rows)}")

    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps([row.to_dict() for row in rows], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        console.print(f"json_out: {json_output}")


@app.command("teleop")
def teleop_hand(
    port: str | None = typer.Option(
        None,
        "--port",
        help="Dynamixel serial port for the real hardware backend; 'auto' selects the single USB serial adapter found.",
    ),
    mujoco: bool = typer.Option(
        False,
        "--mujoco",
        help="Use a MuJoCo simulation backend instead of a fake or real Dynamixel bus.",
    ),
    mujoco_scene: str | None = typer.Option(
        None,
        "--mujoco-scene",
        help=(
            "MuJoCo scene for --mujoco: a built-in label (cube_face_up, "
            "turn_handle) or a custom scene XML path. Defaults to cube_face_up."
        ),
    ),
    mujoco_viewer: bool = typer.Option(
        False,
        "--mujoco-viewer/--no-mujoco-viewer",
        help="Open MuJoCo's passive viewer for the simulation backend.",
    ),
    camera: int = typer.Option(0, "--camera", "-c", help="Camera index to open."),
    frames: int = typer.Option(
        0,
        "--frames",
        "-n",
        min=0,
        help="Number of frames to process. Use 0 to run until interrupted.",
    ),
    target_fps: float | None = typer.Option(
        None,
        "--target-fps",
        min=0.0,
        help=(
            "Loop pacing target in frames per second. Defaults to 10 with --mujoco "
            "and 15 otherwise. Use 0 for no pacing."
        ),
    ),
    hand_side: SingleHandSideOption = typer.Option(
        SingleHandSideOption.right,
        "--hand-side",
        case_sensitive=False,
        help="Robot hand side to command.",
    ),
    retarget_calibration: Path | None = typer.Option(
        None,
        "--retarget-calibration",
        help=(
            "Retarget calibration JSON path. "
            "Defaults to the packaged baseline calibration."
        ),
    ),
    enable_rerun: bool = typer.Option(
        False,
        "--rerun/--no-rerun",
        help=(
            "Stream camera, landmarks, target, measured state, telemetry, and latency "
            "to Rerun, spawning a viewer when needed."
        ),
    ),
    rerun_save: Path | None = typer.Option(
        None,
        "--rerun-save",
        help="Optional .rrd path for saving the Rerun recording.",
    ),
    control_mode: OperatorControlModeOption = typer.Option(
        OperatorControlModeOption.position,
        "--control-mode",
        case_sensitive=False,
        help="Hand runtime control mode: position or current-position.",
    ),
    default_current_limit_ma: float | None = typer.Option(
        None,
        "--default-current-limit-ma",
        min=0.0,
        help="Default Goal Current(102) limit in mA for current-position mode.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Suppress motor writes for fake or real backends.",
    ),
) -> None:
    """Run webcam hand tracking through retargeting into the hand backend."""

    port = _resolve_port_option(port)
    # Fixed operating defaults; teleop exposes only the choices an operator makes.
    baudrate = DEFAULT_BAUDRATE
    camera_width = DEFAULT_CAMERA_WIDTH
    camera_height = DEFAULT_CAMERA_HEIGHT
    camera_fps: float | None = None
    hold_missing_seconds = DEFAULT_TELEOP_HOLD_MISSING_SECONDS
    home_on_missing = True
    preferred_handedness: str | None = None
    anchor_pose_names: list[str] | None = None
    model: Path | None = None
    target_velocity_limit: float | None = None
    max_current_limit_ma: float | None = None

    if mujoco and port is not None:
        console.print("[red]Choose either --port or --mujoco.[/red]")
        raise typer.Exit(code=1)

    if (port is not None or mujoco) and hand_side is not SingleHandSideOption.right:
        console.print(
            "[red]Real and MuJoCo teleop currently support --hand-side right only.[/red]"
        )
        raise typer.Exit(code=1)

    resolved_target_velocity_limit = _resolve_teleop_target_velocity_limit(
        target_velocity_limit,
        mujoco=mujoco,
    )
    resolved_target_fps = _resolve_teleop_target_fps(target_fps, mujoco=mujoco)
    if resolved_target_velocity_limit <= 0:
        console.print("[red]--target-velocity-limit must be positive.[/red]")
        raise typer.Exit(code=1)
    runtime_control_mode = control_mode.runtime_value
    if mujoco and runtime_control_mode == OPERATOR_CONTROL_MODE_CURRENT_POSITION:
        console.print(
            "[red]--mujoco currently supports --control-mode position only.[/red]"
        )
        raise typer.Exit(code=1)
    if (
        runtime_control_mode == OPERATOR_CONTROL_MODE_CURRENT_POSITION
        and default_current_limit_ma is None
    ):
        console.print(
            "[red]--default-current-limit-ma is required when "
            "--control-mode current-position.[/red]"
        )
        raise typer.Exit(code=1)

    runtime_target_hz = (
        resolved_target_fps if resolved_target_fps > 0 else DEFAULT_OPERATOR_TARGET_HZ
    )
    mujoco_runtime_error_type: type[RuntimeError] | None = None
    if mujoco:
        from ghand.sim.mujoco import MujocoRuntimeError, make_mujoco_hand_runtime

        mujoco_runtime_error_type = MujocoRuntimeError
        try:
            runtime = make_mujoco_hand_runtime(
                scene=mujoco_scene,
                side=hand_side.value,
                dry_run=dry_run,
                target_hz=runtime_target_hz,
                target_velocity_limit=resolved_target_velocity_limit,
                control_mode=runtime_control_mode,
                default_current_limit_ma=default_current_limit_ma,
                max_current_limit_ma=max_current_limit_ma,
                viewer=mujoco_viewer,
            )
        except (MujocoRuntimeError, ValueError) as exc:
            console.print(f"[red]MuJoCo teleop failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        backend_description = runtime.backend.replace(
            "mujoco:", "mujoco backend with ", 1
        )
        console.print("Starting MuJoCo teleop simulation.")
    elif port is None:
        runtime = make_fake_hand_runtime(
            side=hand_side.value,
            dry_run=dry_run,
            target_hz=runtime_target_hz,
            target_velocity_limit=resolved_target_velocity_limit,
            control_mode=runtime_control_mode,
            default_current_limit_ma=default_current_limit_ma,
            max_current_limit_ma=max_current_limit_ma,
        )
        backend_description = "fake backend"
    else:
        assert port is not None
        runtime = HandOperatorRuntime(
            HandController.from_port(port=port, baudrate=baudrate),
            make_default_joint_calibrations(
                side=hand_side.value,
                motor_ids=GHAND_RIGHT_MOTOR_IDS,
            ),
            backend=f"dynamixel:{port}",
            dry_run=dry_run,
            target_hz=runtime_target_hz,
            target_velocity_limit=resolved_target_velocity_limit,
            profile_acceleration=DEFAULT_TELEOP_PROFILE_ACCELERATION,
            profile_velocity=DEFAULT_TELEOP_PROFILE_VELOCITY,
            control_mode=runtime_control_mode,
            default_current_limit_ma=default_current_limit_ma,
            max_current_limit_ma=max_current_limit_ma,
        )
        backend_description = f"dynamixel backend on {port}"
        if runtime.dry_run:
            console.print(
                "Starting real hardware teleop in dry-run mode. "
                "Motor writes are suppressed."
            )
        else:
            console.print("Starting real hardware teleop with motor writes enabled.")
    rerun_enabled = enable_rerun or rerun_save is not None
    viewer = RerunViewer(
        enabled=rerun_enabled,
        spawn=enable_rerun,
        save_path=rerun_save,
        urdf_path=default_right_hand_urdf_path() if rerun_enabled else None,
    )
    preferred = preferred_handedness or hand_side.value.title()
    landmark_smoother = _make_landmark_smoother()

    try:
        enabled_pose_names = _resolve_enabled_anchor_pose_names(anchor_pose_names)
        calibration = _load_retarget_calibration(retarget_calibration)
        retargeter = _make_hand_retargeter(
            hand_side=hand_side.value,
            calibration=calibration.calibration,
            enabled_pose_names=enabled_pose_names,
        )
        model_path = resolve_hand_landmarker_model(model)
        with Webcam(
            camera=camera,
            latest=True,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        ) as webcam, MediaPipeHandTracker(model_path=model_path) as tracker:
            runtime.connect()
            runtime.sync_motion_state_to_measured()
            runtime.set_torque_enabled(True)
            viewer.start()
            viewer.log_text("teleop started")
            viewer.log_text(f"retarget calibration: {calibration.source}")
            viewer.log_text("retargeter: anchor-pose")
            console.print(
                f"Teleop hand {backend_description} from camera {camera} with {model_path}. "
                "Press Ctrl+C to stop."
            )
            console.print(f"retarget calibration: {calibration.source}")
            console.print("retargeter: anchor-pose")
            filter_summary = _landmark_filter_summary()
            viewer.log_text(filter_summary)
            console.print(filter_summary)
            result = _run_teleop_hand_loop(
                webcam=webcam,
                tracker=tracker,
                runtime=runtime,
                retargeter=retargeter,
                feature_calibrator=calibration.calibrator,
                viewer=viewer,
                frames=frames,
                target_fps=resolved_target_fps,
                hold_missing_seconds=hold_missing_seconds,
                home_on_missing=home_on_missing,
                hand_side=hand_side.value,
                preferred_handedness=preferred,
                landmark_smoother=landmark_smoother,
            )
            viewer.log_text("teleop stopped")
    except KeyboardInterrupt:
        console.print("Stopped.")
        result = TeleopHandResult(processed=0, frames_with_hands=0)
    except HandModelNotFoundError as exc:
        console.print(f"[red]Teleop hand failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except WebcamError as exc:
        console.print(f"[red]Teleop hand failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except DynamixelBusError as exc:
        console.print(f"[red]Teleop hand failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        if mujoco_runtime_error_type is None or not isinstance(
            exc,
            mujoco_runtime_error_type,
        ):
            raise
        console.print(f"[red]Teleop hand failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]Teleop hand failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"[red]Teleop hand failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        if runtime.connected:
            runtime.disconnect()

    console.print("[green]Teleop hand completed.[/green]")
    console.print(f"processed: {result.processed}")
    console.print(f"frames with hands: {result.frames_with_hands}")
    console.print(f"missing-hand home events: {result.missing_home_events}")


@app.command("record")
def create_lerobot_dataset(
    repo_id: str = typer.Option(
        ...,
        "--repo-id",
        help="Hugging Face dataset repo ID, for example `user/ghand-real`.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Local dataset root. Defaults to LeRobot's cache location.",
    ),
    task: str = typer.Option(
        "Record LEAP V1 KAIZEN hand teleoperation",
        "--task",
        help="Task string stored on every LeRobot frame.",
    ),
    port: str | None = typer.Option(
        None,
        "--port",
        help="Dynamixel serial port for the real hardware backend; 'auto' selects the single USB serial adapter found.",
    ),
    mujoco: bool = typer.Option(
        False,
        "--mujoco",
        help="Use a MuJoCo simulation backend instead of a fake or real Dynamixel bus.",
    ),
    baudrate: int = typer.Option(
        DEFAULT_BAUDRATE,
        "--baudrate",
        min=1,
        help="Dynamixel bus baudrate for the real hardware backend.",
    ),
    mujoco_scene: str | None = typer.Option(
        None,
        "--mujoco-scene",
        help=(
            "MuJoCo scene for --mujoco: a built-in label (cube_face_up, "
            "turn_handle) or a custom scene XML path. Defaults to cube_face_up."
        ),
    ),
    mujoco_viewer: bool = typer.Option(
        False,
        "--mujoco-viewer/--no-mujoco-viewer",
        help="Open MuJoCo's passive viewer for the simulation backend.",
    ),
    mujoco_auto_reset: bool = typer.Option(
        True,
        "--mujoco-auto-reset/--no-mujoco-auto-reset",
        help="Reset MuJoCo scene state between recorded episodes and re-records.",
    ),
    teleop_camera: int = typer.Option(
        0,
        "--teleop-camera",
        min=0,
        help="Camera index for MediaPipe teleoperation input.",
    ),
    teleop_camera_width: int = typer.Option(
        DEFAULT_CAMERA_WIDTH,
        "--teleop-camera-width",
        min=1,
        help="Requested MediaPipe teleoperation camera capture width.",
    ),
    teleop_camera_height: int = typer.Option(
        DEFAULT_CAMERA_HEIGHT,
        "--teleop-camera-height",
        min=1,
        help="Requested MediaPipe teleoperation camera capture height.",
    ),
    observation_camera: int | None = typer.Option(
        None,
        "--observation-camera",
        min=0,
        help=(
            "Camera index stored as observation.images.front for fake or real backends. "
            "Omit to record no external observation camera. "
            "With --mujoco, compiled MuJoCo model cameras are used instead."
        ),
    ),
    image_width: int = typer.Option(
        DEFAULT_FRONT_IMAGE_SHAPE[1],
        "--image-width",
        min=1,
        help="Stored observation camera image width when image features are enabled.",
    ),
    image_height: int = typer.Option(
        DEFAULT_FRONT_IMAGE_SHAPE[0],
        "--image-height",
        min=1,
        help="Stored observation camera image height when image features are enabled.",
    ),
    num_episodes: int = typer.Option(
        DEFAULT_LEROBOT_RECORD_NUM_EPISODES,
        "--num-episodes",
        min=1,
        help="Number of LeRobot episodes to record.",
    ),
    episode_time_s: float = typer.Option(
        DEFAULT_LEROBOT_RECORD_EPISODE_TIME_S,
        "--episode-time-s",
        min=1e-9,
        help="Maximum recording duration for each episode in seconds.",
    ),
    reset_time_s: float = typer.Option(
        DEFAULT_LEROBOT_RECORD_RESET_TIME_S,
        "--reset-time-s",
        min=0.0,
        help="Non-recorded reset duration between episodes in seconds.",
    ),
    fps: int | None = typer.Option(
        None,
        "--fps",
        min=1,
        help=(
            "Dataset and capture-loop FPS. Defaults to 10 with --mujoco "
            "and 30 otherwise."
        ),
    ),
    hand_side: SingleHandSideOption = typer.Option(
        SingleHandSideOption.right,
        "--hand-side",
        case_sensitive=False,
        help="Robot hand side to command and record.",
    ),
    preferred_handedness: str | None = typer.Option(
        None,
        "--preferred-handedness",
        help="MediaPipe handedness label to prefer. Defaults to the selected hand side.",
    ),
    retarget_calibration: Path | None = typer.Option(
        None,
        "--retarget-calibration",
        help=(
            "Retarget calibration JSON path. "
            "Defaults to the packaged baseline calibration."
        ),
    ),
    anchor_pose_names: list[str] | None = typer.Option(
        None,
        "--anchor-pose",
        help=(
            "Limit retargeting to this captured pose name. "
            "Repeat to enable multiple anchors. Defaults to all captured anchors."
        ),
    ),
    dataset_schema: DatasetSchemaOption = typer.Option(
        DatasetSchemaOption.position,
        "--dataset-schema",
        case_sensitive=False,
        help="LeRobot dataset schema: position or position-current.",
    ),
    include_current_diagnostics: bool = typer.Option(
        False,
        "--include-current-diagnostics/--no-current-diagnostics",
        help="Add current residual and contact flag channels to position-current state.",
    ),
    contact_current_threshold_ma: float = typer.Option(
        DEFAULT_CURRENT_CONTACT_THRESHOLD_MA,
        "--contact-current-threshold-ma",
        min=0.0,
        help="Present-current magnitude above this deadband is treated as contact.",
    ),
    use_videos: bool = typer.Option(
        True,
        "--use-videos/--use-images",
        help="Store camera features as LeRobot videos or raw image frames.",
    ),
    push_to_hub: bool = typer.Option(
        False,
        "--push-to-hub",
        help="Push the completed dataset to Hugging Face Hub.",
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Create or update the Hub dataset as private when pushing.",
    ),
    enable_rerun: bool = typer.Option(
        False,
        "--rerun/--no-rerun",
        help=(
            "Stream camera, landmarks, target, measured state, telemetry, and latency "
            "to Rerun, spawning a viewer when needed."
        ),
    ),
    rerun_save: Path | None = typer.Option(
        None,
        "--rerun-save",
        help="Optional .rrd path for saving the Rerun recording.",
    ),
    model: Path | None = typer.Option(
        None,
        "--model",
        help="MediaPipe hand_landmarker.task path.",
    ),
    target_velocity_limit: float | None = typer.Option(
        None,
        "--target-velocity-limit",
        min=0.0,
        help=(
            "Joint target velocity limit in radians per second for the hand runtime. "
            "Defaults to 20.0 with --mujoco, otherwise 1000.0."
        ),
    ),
    control_mode: OperatorControlModeOption = typer.Option(
        OperatorControlModeOption.position,
        "--control-mode",
        case_sensitive=False,
        help="Hand runtime control mode: position or current-position.",
    ),
    default_current_limit_ma: float | None = typer.Option(
        None,
        "--default-current-limit-ma",
        min=0.0,
        help="Default Goal Current(102) limit in mA for current-position mode.",
    ),
    max_current_limit_ma: float | None = typer.Option(
        None,
        "--max-current-limit-ma",
        min=0.0,
        help="Clamp current-limit commands to this mA value before hardware writes.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Suppress motor writes for fake or real backends.",
    ),
) -> None:
    """Record live hand teleoperation frames into a LeRobotDataset."""

    port = _resolve_port_option(port)

    if mujoco and port is not None:
        console.print("[red]Choose either --port or --mujoco.[/red]")
        raise typer.Exit(code=1)
    if (port is not None or mujoco) and hand_side is not SingleHandSideOption.right:
        console.print(
            "[red]Real and MuJoCo record currently support "
            "--hand-side right only.[/red]"
        )
        raise typer.Exit(code=1)
    resolved_target_velocity_limit = _resolve_teleop_target_velocity_limit(
        target_velocity_limit,
        mujoco=mujoco,
    )
    if resolved_target_velocity_limit <= 0:
        console.print("[red]--target-velocity-limit must be positive.[/red]")
        raise typer.Exit(code=1)
    runtime_control_mode = control_mode.runtime_value
    if mujoco and runtime_control_mode == OPERATOR_CONTROL_MODE_CURRENT_POSITION:
        console.print(
            "[red]--mujoco currently supports --control-mode position only.[/red]"
        )
        raise typer.Exit(code=1)
    if mujoco and dataset_schema is DatasetSchemaOption.position_current:
        console.print(
            "[red]--mujoco currently supports --dataset-schema position only. "
            "MuJoCo current semantics are not implemented yet.[/red]"
        )
        raise typer.Exit(code=1)
    if (
        runtime_control_mode == OPERATOR_CONTROL_MODE_CURRENT_POSITION
        and default_current_limit_ma is None
    ):
        console.print(
            "[red]--default-current-limit-ma is required when "
            "--control-mode current-position.[/red]"
        )
        raise typer.Exit(code=1)
    if (
        dataset_schema is DatasetSchemaOption.position_current
        and default_current_limit_ma is None
    ):
        console.print(
            "[red]--default-current-limit-ma is required when "
            "--dataset-schema position-current.[/red]"
        )
        raise typer.Exit(code=1)
    if (
        include_current_diagnostics
        and dataset_schema is not DatasetSchemaOption.position_current
    ):
        console.print(
            "[red]--include-current-diagnostics requires "
            "--dataset-schema position-current.[/red]"
        )
        raise typer.Exit(code=1)
    if not task:
        console.print("[red]--task must not be empty.[/red]")
        raise typer.Exit(code=1)

    resolved_dataset_fps = _resolve_dataset_fps(fps, mujoco=mujoco)
    resolved_teleop_camera = teleop_camera
    resolved_observation_camera = observation_camera
    resolved_target_fps = float(resolved_dataset_fps)
    image_dtype = "video" if use_videos else "image"
    image_shape = (image_height, image_width, 3)
    push_to_hub_kwargs = {"private": True} if private else None

    mujoco_runtime_error_type: type[RuntimeError] | None = None
    mujoco_camera_names: tuple[str, ...] = ()
    dataset_camera_frame_source: Callable[[], Mapping[str, object]] | None = None
    dataset_image_color_format = "bgr"
    if mujoco:
        from ghand.sim.mujoco import MujocoRuntimeError, make_mujoco_hand_runtime

        mujoco_runtime_error_type = MujocoRuntimeError
        try:
            runtime = make_mujoco_hand_runtime(
                scene=mujoco_scene,
                side=hand_side.value,
                dry_run=dry_run,
                target_hz=resolved_target_fps,
                target_velocity_limit=resolved_target_velocity_limit,
                control_mode=runtime_control_mode,
                default_current_limit_ma=default_current_limit_ma,
                max_current_limit_ma=max_current_limit_ma,
                viewer=mujoco_viewer,
            )
            runtime.connect()
            mujoco_camera_names = _mujoco_runtime_camera_names(runtime)
        except (MujocoRuntimeError, ValueError) as exc:
            console.print(f"[red]MuJoCo dataset capture failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        backend_description = runtime.backend.replace(
            "mujoco:", "mujoco backend with ", 1
        )
        console.print("Starting MuJoCo dataset capture simulation.")
        if mujoco_camera_names:
            try:
                dataset_camera_frame_source = _make_mujoco_camera_frame_source(
                    runtime=runtime,
                    camera_names=mujoco_camera_names,
                    width=image_width,
                    height=image_height,
                )
            except ValueError as exc:
                console.print(f"[red]MuJoCo dataset capture failed:[/red] {exc}")
                raise typer.Exit(code=1) from exc
            dataset_image_color_format = "rgb"
    elif port is None:
        runtime = make_fake_hand_runtime(
            side=hand_side.value,
            dry_run=dry_run,
            target_hz=resolved_target_fps,
            target_velocity_limit=resolved_target_velocity_limit,
            control_mode=runtime_control_mode,
            default_current_limit_ma=default_current_limit_ma,
            max_current_limit_ma=max_current_limit_ma,
        )
        backend_description = "fake backend"
    else:
        assert port is not None
        runtime = HandOperatorRuntime(
            HandController.from_port(port=port, baudrate=baudrate),
            make_default_joint_calibrations(
                side=hand_side.value,
                motor_ids=GHAND_RIGHT_MOTOR_IDS,
            ),
            backend=f"dynamixel:{port}",
            dry_run=dry_run,
            target_hz=resolved_target_fps,
            target_velocity_limit=resolved_target_velocity_limit,
            profile_acceleration=DEFAULT_TELEOP_PROFILE_ACCELERATION,
            profile_velocity=DEFAULT_TELEOP_PROFILE_VELOCITY,
            control_mode=runtime_control_mode,
            default_current_limit_ma=default_current_limit_ma,
            max_current_limit_ma=max_current_limit_ma,
        )
        backend_description = f"dynamixel backend on {port}"
        if runtime.dry_run:
            console.print(
                "Starting real hardware dataset capture in dry-run mode. "
                "Motor writes are suppressed."
            )
        else:
            console.print(
                "Starting real hardware dataset capture with motor writes enabled."
            )

    if mujoco:
        camera_specs = tuple(
            CameraSpec(camera_name, image_shape, image_dtype)
            for camera_name in mujoco_camera_names
        )
    elif resolved_observation_camera is None:
        camera_specs = ()
    else:
        camera_specs = (CameraSpec("front", image_shape, image_dtype),)

    if dataset_schema is DatasetSchemaOption.position_current:
        features = make_hand_position_current_lerobot_features(
            sides=(hand_side.value,),
            cameras=camera_specs,
            image_dtype=image_dtype,
            include_current_diagnostics=include_current_diagnostics,
        )
    else:
        features = make_hand_lerobot_features(
            sides=(hand_side.value,),
            cameras=camera_specs,
            image_dtype=image_dtype,
        )

    rerun_enabled = enable_rerun or rerun_save is not None
    viewer = RerunViewer(
        enabled=rerun_enabled,
        spawn=enable_rerun,
        save_path=rerun_save,
        urdf_path=default_right_hand_urdf_path() if rerun_enabled else None,
    )
    preferred = preferred_handedness or hand_side.value.title()
    landmark_smoother = _make_landmark_smoother()
    recorder: LeRobotStreamingRecorder | None = None
    result = TeleopHandResult(processed=0, frames_with_hands=0)

    try:
        enabled_pose_names = _resolve_enabled_anchor_pose_names(anchor_pose_names)
        calibration = _load_retarget_calibration(retarget_calibration)
        retargeter = _make_hand_retargeter(
            hand_side=hand_side.value,
            calibration=calibration.calibration,
            enabled_pose_names=enabled_pose_names,
        )
        model_path = resolve_hand_landmarker_model(model)
        active_observation_camera = None if mujoco else resolved_observation_camera
        record_teleop_frame_as_observation = (
            active_observation_camera is not None
            and active_observation_camera == resolved_teleop_camera
        )
        camera_description = _dataset_camera_description(
            teleop_camera=resolved_teleop_camera,
            observation_camera=resolved_observation_camera,
            mujoco_camera_names=mujoco_camera_names,
            mujoco=mujoco,
        )
        with (
            Webcam(
                camera=resolved_teleop_camera,
                latest=True,
                width=teleop_camera_width,
                height=teleop_camera_height,
            ) as webcam,
            _open_observation_webcam(
                teleop_camera=resolved_teleop_camera,
                observation_camera=active_observation_camera,
                width=image_width,
                height=image_height,
            ) as observation_webcam,
            MediaPipeHandTracker(model_path=model_path) as tracker,
            LeRobotStreamingRecorder.create(
                repo_id=repo_id,
                root=root,
                fps=resolved_dataset_fps,
                features=features,
                push_to_hub=push_to_hub,
                push_to_hub_kwargs=push_to_hub_kwargs,
                use_videos=use_videos,
            ) as recorder,
        ):
            if not runtime.connected:
                runtime.connect()
            runtime.set_torque_enabled(True)
            viewer.start()
            viewer.log_text("record started")
            viewer.log_text(f"retarget calibration: {calibration.source}")
            viewer.log_text("retargeter: anchor-pose")
            console.print(
                f"Recording LeRobot dataset {repo_id} from {camera_description} "
                f"with {backend_description} and {model_path}."
            )
            console.print(
                "episode controls: Right Arrow saves/continues, "
                "Left Arrow discards and retries, ESC stops."
            )
            console.print(f"dataset schema: {dataset_schema.value}")
            console.print(f"episodes: {num_episodes}")
            console.print(f"episode time: {episode_time_s:g}s")
            console.print(f"reset time: {reset_time_s:g}s")
            console.print(f"retarget calibration: {calibration.source}")
            console.print("retargeter: anchor-pose")
            filter_summary = _landmark_filter_summary()
            viewer.log_text(filter_summary)
            console.print(filter_summary)
            with _lerobot_record_keyboard_events() as record_events:
                recorded_episodes = 0
                while (
                    recorded_episodes < num_episodes
                    and not record_events["stop_recording"]
                ):
                    episode_number = recorded_episodes + 1
                    console.print(f"Recording episode {episode_number}/{num_episodes}.")
                    viewer.log_text(
                        f"recording episode {episode_number}/{num_episodes}"
                    )
                    episode_result = _run_teleop_hand_loop(
                        webcam=webcam,
                        tracker=tracker,
                        runtime=runtime,
                        retargeter=retargeter,
                        feature_calibrator=calibration.calibrator,
                        viewer=viewer,
                        frames=0,
                        control_time_s=episode_time_s,
                        keyboard_events=record_events,
                        target_fps=resolved_target_fps,
                        hold_missing_seconds=0.0,
                        home_on_missing=True,
                        hand_side=hand_side.value,
                        preferred_handedness=preferred,
                        landmark_smoother=landmark_smoother,
                        observation_webcam=observation_webcam,
                        use_teleop_frame_as_observation=record_teleop_frame_as_observation,
                        dataset_camera_frame_source=dataset_camera_frame_source,
                        dataset_image_color_format=dataset_image_color_format,
                        dataset_recorder=recorder,
                        dataset_features=features,
                        dataset_task=task,
                        dataset_default_current_limit_ma=(
                            default_current_limit_ma
                            if dataset_schema is DatasetSchemaOption.position_current
                            else None
                        ),
                        dataset_current_contact_threshold_ma=contact_current_threshold_ma,
                    )
                    result = TeleopHandResult(
                        processed=result.processed + episode_result.processed,
                        frames_with_hands=(
                            result.frames_with_hands + episode_result.frames_with_hands
                        ),
                        missing_home_events=(
                            result.missing_home_events
                            + episode_result.missing_home_events
                        ),
                    )

                    if record_events["rerecord_episode"]:
                        console.print("Re-record episode.")
                        viewer.log_text("re-record episode")
                        record_events["rerecord_episode"] = False
                        record_events["exit_early"] = False
                        recorder.discard_episode()
                        _reset_dataset_runtime_environment(
                            runtime=runtime,
                            enabled=mujoco and mujoco_auto_reset,
                            viewer=viewer,
                        )
                        if reset_time_s > 0 and not record_events["stop_recording"]:
                            _run_lerobot_dataset_reset_loop(
                                webcam=webcam,
                                tracker=tracker,
                                runtime=runtime,
                                retargeter=retargeter,
                                feature_calibrator=calibration.calibrator,
                                viewer=viewer,
                                reset_time_s=reset_time_s,
                                keyboard_events=record_events,
                                target_fps=resolved_target_fps,
                                hand_side=hand_side.value,
                                preferred_handedness=preferred,
                                landmark_smoother=landmark_smoother,
                                observation_webcam=observation_webcam,
                                use_teleop_frame_as_observation=(
                                    record_teleop_frame_as_observation
                                ),
                                dataset_camera_frame_source=dataset_camera_frame_source,
                                dataset_image_color_format=dataset_image_color_format,
                            )
                        continue

                    if recorder.frames_in_episode > 0:
                        recorder.save_episode()
                        recorded_episodes += 1
                    elif not record_events["stop_recording"]:
                        console.print("Skipping empty episode.")
                        recorded_episodes += 1

                    should_prepare_next = (
                        not record_events["stop_recording"]
                        and recorded_episodes < num_episodes
                    )
                    if should_prepare_next:
                        _reset_dataset_runtime_environment(
                            runtime=runtime,
                            enabled=mujoco and mujoco_auto_reset,
                            viewer=viewer,
                        )
                        if reset_time_s > 0:
                            _run_lerobot_dataset_reset_loop(
                                webcam=webcam,
                                tracker=tracker,
                                runtime=runtime,
                                retargeter=retargeter,
                                feature_calibrator=calibration.calibrator,
                                viewer=viewer,
                                reset_time_s=reset_time_s,
                                keyboard_events=record_events,
                                target_fps=resolved_target_fps,
                                hand_side=hand_side.value,
                                preferred_handedness=preferred,
                                landmark_smoother=landmark_smoother,
                                observation_webcam=observation_webcam,
                                use_teleop_frame_as_observation=(
                                    record_teleop_frame_as_observation
                                ),
                                dataset_camera_frame_source=dataset_camera_frame_source,
                                dataset_image_color_format=dataset_image_color_format,
                            )
            viewer.log_text("record stopped")
    except KeyboardInterrupt:
        console.print("Stopped.")
    except LeRobotDependencyError as exc:
        console.print(f"[red]Create LeRobot dataset failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except HandModelNotFoundError as exc:
        console.print(f"[red]Create LeRobot dataset failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except WebcamError as exc:
        console.print(f"[red]Create LeRobot dataset failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except DynamixelBusError as exc:
        console.print(f"[red]Create LeRobot dataset failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        if mujoco_runtime_error_type is None or not isinstance(
            exc,
            mujoco_runtime_error_type,
        ):
            raise
        console.print(f"[red]Create LeRobot dataset failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]Create LeRobot dataset failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"[red]Create LeRobot dataset failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        if runtime.connected:
            runtime.disconnect()

    console.print("[green]LeRobot dataset created.[/green]")
    console.print(f"repo_id: {repo_id}")
    if root is not None:
        console.print(f"root: {root}")
    console.print(f"processed: {result.processed}")
    console.print(f"frames with hands: {result.frames_with_hands}")
    if recorder is not None:
        console.print(f"recorded frames: {recorder.total_frames}")
        console.print(f"saved episodes: {recorder.saved_episodes}")
    if push_to_hub:
        console.print("pushed to hub: yes")


@app.command("policy-smoke")
def lerobot_policy_smoke(
    policy: str | None = typer.Option(
        None,
        "--policy",
        help="Local path or Hub repo ID for a LeRobot policy checkpoint to load.",
    ),
    action_json: str | None = typer.Option(
        None,
        "--action-json",
        help=(
            "JSON action object or vector used instead of loading a policy. "
            "Objects may provide a sparse subset; omitted keys use safe defaults."
        ),
    ),
    backend: LeRobotPluginBackendOption = typer.Option(
        LeRobotPluginBackendOption.fake,
        "--backend",
        case_sensitive=False,
        help="LEAP V1 KAIZEN LeRobot backend used for the one-step smoke.",
    ),
    port: str | None = typer.Option(
        None,
        "--port",
        help="Dynamixel serial port when --backend real; 'auto' selects the single USB serial adapter found.",
    ),
    baudrate: int = typer.Option(
        DEFAULT_BAUDRATE,
        "--baudrate",
        min=1,
        help="Dynamixel baudrate when --backend real.",
    ),
    allow_real_write: bool = typer.Option(
        False,
        "--allow-real-write",
        help="Allow a real backend smoke to configure/write hardware. Otherwise real backend is dry-run.",
    ),
    mujoco_scene: str | None = typer.Option(
        None,
        "--mujoco-scene",
        help=(
            "MuJoCo scene when --backend mujoco: a built-in label "
            "(cube_face_up, turn_handle) or a custom scene XML path."
        ),
    ),
    mujoco_viewer: bool = typer.Option(
        False,
        "--mujoco-viewer",
        help="Open the passive MuJoCo viewer during the smoke.",
    ),
    dataset_schema: DatasetSchemaOption = typer.Option(
        DatasetSchemaOption.position,
        "--dataset-schema",
        case_sensitive=False,
        help="Robot policy schema: position or position-current.",
    ),
    control_mode: OperatorControlModeOption = typer.Option(
        OperatorControlModeOption.position,
        "--control-mode",
        case_sensitive=False,
        help="Runtime control mode used by the LEAP V1 KAIZEN robot plugin.",
    ),
    default_current_limit_ma: float | None = typer.Option(
        None,
        "--default-current-limit-ma",
        min=0.0,
        help="Required for --control-mode current-position and used as default current action.",
    ),
    max_current_limit_ma: float | None = typer.Option(
        None,
        "--max-current-limit-ma",
        min=0.0,
        help="Clamp policy current-limit actions in milliamps.",
    ),
    include_current_diagnostics: bool = typer.Option(
        False,
        "--include-current-diagnostics",
        help="Expose current residual/contact channels in position-current observations.",
    ),
    current_contact_threshold_ma: float = typer.Option(
        DEFAULT_CURRENT_CONTACT_THRESHOLD_MA,
        "--contact-current-threshold-ma",
        min=0.0,
        help="Present-current threshold used for residual/contact observation diagnostics.",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Print machine-readable JSON instead of a compact table.",
    ),
) -> None:
    """Run one LeRobot policy/action step through the LEAP V1 KAIZEN robot plugin."""

    port = _resolve_port_option(port)

    if policy is not None and action_json is not None:
        console.print("[red]Use either --policy or --action-json, not both.[/red]")
        raise typer.Exit(code=1)

    try:
        from ghand.lerobot_plugin import GHandRobotConfig
        from ghand.lerobot_plugin.policy_smoke import (
            parse_action_json,
            run_ghand_policy_smoke,
        )
    except ImportError as exc:
        console.print(
            "[red]LeRobot dependencies are required for this command.[/red] "
            "Run with `uv run --extra lerobot ghand policy-smoke ...`."
        )
        raise typer.Exit(code=1) from exc

    try:
        config = GHandRobotConfig(
            id="policy-smoke",
            backend=backend.value,
            port=port,
            baudrate=baudrate,
            dry_run=(
                backend is LeRobotPluginBackendOption.real and not allow_real_write
            ),
            control_mode=control_mode.value,
            dataset_schema=dataset_schema.value,
            default_current_limit_ma=default_current_limit_ma,
            max_current_limit_ma=max_current_limit_ma,
            mujoco_scene=mujoco_scene,
            mujoco_viewer=mujoco_viewer,
            include_current_diagnostics=include_current_diagnostics,
            current_contact_threshold_ma=current_contact_threshold_ma,
        )
        action_values = parse_action_json(action_json)
        result = run_ghand_policy_smoke(
            config,
            policy_path=policy,
            action_values=action_values,
        )
    except Exception as exc:
        console.print(f"[red]LeRobot policy smoke failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if output_json:
        console.print_json(data=result.to_dict())
        return

    table = Table(title="LeRobot Policy Smoke")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("robot", result.robot_type)
    table.add_row("backend", result.backend)
    table.add_row("schema", result.dataset_schema)
    table.add_row("action source", result.action_source)
    table.add_row("observation features", str(result.observation_feature_count))
    table.add_row("action features", str(result.action_feature_count))
    table.add_row("dataset keys", ", ".join(result.dataset_feature_keys))
    table.add_row("sent action values", str(len(result.sent_action)))
    console.print(table)
    console.print("[green]LeRobot policy smoke passed.[/green]")


@app.command(
    "train",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(
    ctx: typer.Context,
    repo_id: str = typer.Option(
        ...,
        "--repo-id",
        help="LeRobot dataset repo ID, for example `local/ghand-mujoco`.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help=(
            "Dataset root override. Defaults to LeRobot's cache location when omitted."
        ),
    ),
    job_name: str | None = typer.Option(
        None,
        "--job-name",
        help="Training job name. Defaults to `<dataset-name>-<policy-type>`.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Training output directory. Defaults to `outputs/train/<job-name>`.",
    ),
    steps: int = typer.Option(
        DEFAULT_TRAIN_STEPS,
        "--steps",
        min=1,
        help="Total training steps.",
    ),
    batch_size: int = typer.Option(
        DEFAULT_TRAIN_BATCH_SIZE,
        "--batch-size",
        min=1,
        help="Training batch size.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Policy device. Defaults to cuda when available, otherwise cpu.",
    ),
    policy_type: str = typer.Option(
        DEFAULT_TRAIN_POLICY_TYPE,
        "--policy-type",
        help="LeRobot policy type. Chunk options are only applied for act.",
    ),
    chunk_size: int = typer.Option(
        DEFAULT_POLICY_CHUNK_SIZE,
        "--chunk-size",
        min=1,
        help="ACT action chunk size.",
    ),
    n_action_steps: int = typer.Option(
        DEFAULT_POLICY_N_ACTION_STEPS,
        "--n-action-steps",
        min=1,
        help="ACT executed action steps per chunk.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved lerobot-train command without running it.",
    ),
) -> None:
    """Run lerobot-train with recommended LEAP V1 KAIZEN defaults.

    Unknown options are forwarded to lerobot-train verbatim and override the
    generated defaults, e.g. `--policy.optimizer_lr=5e-5`.
    """

    if n_action_steps > chunk_size:
        console.print("[red]--n-action-steps must not exceed --chunk-size.[/red]")
        raise typer.Exit(code=1)

    dataset_name = repo_id.rsplit("/", 1)[-1]
    resolved_root = root
    resolved_job_name = job_name or f"{dataset_name}-{policy_type}"
    resolved_output_dir = output_dir or Path("outputs/train") / resolved_job_name
    resolved_device = device or _default_torch_device()

    args = [
        sys.executable,
        "-m",
        LEROBOT_TRAIN_MODULE,
        f"--dataset.repo_id={repo_id}",
        f"--policy.type={policy_type}",
        f"--policy.device={resolved_device}",
        "--policy.push_to_hub=false",
        f"--output_dir={resolved_output_dir}",
        f"--job_name={resolved_job_name}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--num_workers={DEFAULT_TRAIN_NUM_WORKERS}",
        f"--log_freq={DEFAULT_TRAIN_LOG_FREQ}",
        "--save_checkpoint=true",
        f"--save_freq={DEFAULT_TRAIN_SAVE_FREQ}",
        "--wandb.enable=false",
    ]
    if resolved_root is not None:
        args.append(f"--dataset.root={resolved_root}")
    if policy_type == "act":
        args.append(f"--policy.chunk_size={chunk_size}")
        args.append(f"--policy.n_action_steps={n_action_steps}")
    args = _merge_cli_overrides(args, ctx.args)

    console.print(
        f"dataset: {repo_id}" + ("" if resolved_root is None else f" ({resolved_root})")
    )
    console.print(f"output: {resolved_output_dir}")
    console.print(f"device: {resolved_device}")
    _run_wrapped_lerobot_command(args, dry_run=dry_run)


@app.command(
    "run-policy",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run_policy(
    ctx: typer.Context,
    policy_path: Path = typer.Option(
        ...,
        "--policy-path",
        help=(
            "Trained policy directory. Train output roots resolve to "
            "checkpoints/last/pretrained_model automatically."
        ),
    ),
    backend: LeRobotPluginBackendOption = typer.Option(
        LeRobotPluginBackendOption.mujoco,
        "--backend",
        case_sensitive=False,
        help="LEAP V1 KAIZEN robot backend for the rollout: mujoco, real, or fake.",
    ),
    port: str | None = typer.Option(
        None,
        "--port",
        help="Dynamixel serial port, required with --backend real; 'auto' selects the single USB serial adapter found.",
    ),
    baudrate: int = typer.Option(
        DEFAULT_BAUDRATE,
        "--baudrate",
        min=1,
        help="Dynamixel baudrate with --backend real.",
    ),
    observation_camera: int | None = typer.Option(
        None,
        "--observation-camera",
        min=0,
        help=(
            "Webcam index serving the policy's observation camera with "
            "--backend real or fake. Required when the policy expects images."
        ),
    ),
    mujoco_scene: str | None = typer.Option(
        None,
        "--mujoco-scene",
        help=(
            "MuJoCo scene used for the rollout: a built-in label (cube_face_up, "
            "turn_handle) or a custom scene XML path. Use the same scene the "
            "training data was recorded on. Defaults to cube_face_up."
        ),
    ),
    repo_id: str | None = typer.Option(
        None,
        "--repo-id",
        help="Rollout dataset repo ID. Defaults to `local/eval_ghand_<timestamp>`.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Rollout dataset root. Defaults to `data/<dataset-name>`.",
    ),
    task: str = typer.Option(
        "Evaluate LEAP V1 KAIZEN policy",
        "--task",
        help="Task string stored on every rollout frame.",
    ),
    num_episodes: int = typer.Option(
        DEFAULT_RUN_POLICY_NUM_EPISODES,
        "--num-episodes",
        min=1,
        help="Number of rollout episodes to record.",
    ),
    episode_time_s: float = typer.Option(
        DEFAULT_RUN_POLICY_EPISODE_TIME_S,
        "--episode-time-s",
        min=1e-9,
        help="Maximum rollout duration for each episode in seconds.",
    ),
    fps: int = typer.Option(
        DEFAULT_RUN_POLICY_FPS,
        "--fps",
        min=1,
        help="Rollout FPS for both the robot control loop and the dataset.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Policy device. Defaults to cuda when available, otherwise cpu.",
    ),
    mujoco_viewer: bool = typer.Option(
        True,
        "--mujoco-viewer/--no-mujoco-viewer",
        help="Open MuJoCo's passive viewer during the rollout.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved lerobot-record command without running it.",
    ),
) -> None:
    """Roll out a trained policy with lerobot-record on MuJoCo or real hardware.

    Camera names and resolution are derived from the policy's config so the
    rollout observations match training. Unknown options are forwarded to
    lerobot-record verbatim and override the generated defaults.
    """

    port = _resolve_port_option(port)

    try:
        resolved_policy_path = _resolve_pretrained_policy_path(policy_path)
        camera_names, camera_width, camera_height = _policy_camera_settings(
            resolved_policy_path
        )
        _validate_run_policy_backend_options(
            backend=backend,
            port=port,
            observation_camera=observation_camera,
            mujoco_scene=mujoco_scene,
            camera_names=camera_names,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Run policy failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    resolved_repo_id = (
        repo_id or f"local/eval_ghand_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    dataset_name = resolved_repo_id.rsplit("/", 1)[-1]
    resolved_root = root or Path("data") / dataset_name
    resolved_device = device or _default_torch_device()

    args = [
        sys.executable,
        "-m",
        LEROBOT_RECORD_MODULE,
        "--robot.type=ghand",
        f"--robot.backend={backend.value}",
        f"--robot.target_fps={fps}",
        f"--policy.path={resolved_policy_path}",
        f"--policy.device={resolved_device}",
        f"--dataset.repo_id={resolved_repo_id}",
        f"--dataset.root={resolved_root}",
        f"--dataset.single_task={task}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        "--dataset.reset_time_s=0",
        f"--dataset.fps={fps}",
        "--dataset.push_to_hub=false",
        "--dataset.video=true",
        "--play_sounds=false",
    ]
    if backend is LeRobotPluginBackendOption.mujoco:
        args.append(f"--robot.mujoco_viewer={'true' if mujoco_viewer else 'false'}")
        args.append(
            f"--robot.target_velocity_limit={DEFAULT_MUJOCO_TARGET_VELOCITY_LIMIT}"
        )
        args.append(f"--robot.mujoco_actuator_kp={DEFAULT_MUJOCO_RUNTIME_ACTUATOR_KP}")
        args.append(
            "--robot.mujoco_actuator_force_limit="
            f"{DEFAULT_MUJOCO_RUNTIME_ACTUATOR_FORCE_LIMIT}"
        )
        if mujoco_scene is not None:
            args.append(f"--robot.mujoco_scene={mujoco_scene}")
        if camera_names:
            args.append(f"--robot.mujoco_camera_names={','.join(camera_names)}")
    else:
        if backend is LeRobotPluginBackendOption.real:
            args.append(f"--robot.port={port}")
            args.append(f"--robot.baudrate={baudrate}")
        if observation_camera is not None:
            args.append(f"--robot.observation_camera={observation_camera}")
            if camera_names:
                args.append(f"--robot.observation_camera_name={camera_names[0]}")
    if camera_width is not None and camera_height is not None:
        args.append(f"--robot.camera_width={camera_width}")
        args.append(f"--robot.camera_height={camera_height}")
    args = _merge_cli_overrides(args, ctx.args)

    console.print(f"backend: {backend.value}")
    console.print(f"policy: {resolved_policy_path}")
    console.print(f"rollout dataset: {resolved_repo_id} ({resolved_root})")
    if camera_names:
        cameras_text = ", ".join(camera_names)
        console.print(
            f"cameras from policy config: {cameras_text} ({camera_width}x{camera_height})"
        )
    console.print(f"device: {resolved_device}")
    _run_wrapped_lerobot_command(args, dry_run=dry_run)


def _validate_run_policy_backend_options(
    *,
    backend: LeRobotPluginBackendOption,
    port: str | None,
    observation_camera: int | None,
    mujoco_scene: str | None,
    camera_names: Sequence[str],
) -> None:
    if backend is LeRobotPluginBackendOption.mujoco:
        if observation_camera is not None:
            raise ValueError(
                "--observation-camera is for --backend real/fake; "
                "MuJoCo rollouts render model cameras instead."
            )
        return

    if mujoco_scene is not None:
        raise ValueError("--mujoco-scene requires --backend mujoco.")
    if backend is LeRobotPluginBackendOption.real and not port:
        raise ValueError("--port is required with --backend real.")
    if camera_names and observation_camera is None:
        cameras_text = ", ".join(camera_names)
        raise ValueError(
            f"The policy expects camera image(s) ({cameras_text}); "
            "pass --observation-camera <index> to serve them from a webcam."
        )
    if len(camera_names) > 1:
        cameras_text = ", ".join(camera_names)
        raise ValueError(
            "Only one observation camera is supported with --backend "
            f"real/fake, but the policy expects: {cameras_text}."
        )


def _default_torch_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _merge_cli_overrides(defaults: list[str], overrides: Sequence[str]) -> list[str]:
    """Drop generated `--key=value` defaults that the user overrides explicitly."""

    override_keys = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    merged = [
        arg
        for arg in defaults
        if not arg.startswith("--") or arg.split("=", 1)[0] not in override_keys
    ]
    merged.extend(overrides)
    return merged


def _resolve_pretrained_policy_path(path: Path) -> Path:
    """Accept either a pretrained_model dir or a train output root."""

    if (path / "config.json").is_file():
        return path
    candidate = path / "checkpoints" / "last" / "pretrained_model"
    if (candidate / "config.json").is_file():
        return candidate
    raise ValueError(
        f"No policy config.json found under {path} or {candidate}. "
        "Pass the pretrained_model directory of a checkpoint."
    )


def _policy_camera_settings(
    policy_path: Path,
) -> tuple[tuple[str, ...], int | None, int | None]:
    """Derive MuJoCo camera names and resolution from a policy config."""

    config = json.loads((policy_path / "config.json").read_text(encoding="utf-8"))
    input_features = config.get("input_features") or {}
    names: list[str] = []
    shapes: list[tuple[int, int, int]] = []
    for key, spec in input_features.items():
        if not key.startswith(f"{OBSERVATION_IMAGES_PREFIX}."):
            continue
        names.append(key.removeprefix(f"{OBSERVATION_IMAGES_PREFIX}."))
        shape = spec.get("shape") if isinstance(spec, Mapping) else None
        if isinstance(shape, list) and len(shape) == 3:
            shapes.append(tuple(int(value) for value in shape))

    width: int | None = None
    height: int | None = None
    if shapes:
        if len(set(shapes)) > 1:
            raise ValueError(
                "Policy image features use mixed shapes; pass "
                "--robot.camera_width/--robot.camera_height explicitly."
            )
        channels, shape_height, shape_width = shapes[0]
        if channels != 3:
            raise ValueError(f"Expected channel-first image shape, got {shapes[0]}.")
        width, height = shape_width, shape_height
    return tuple(names), width, height


def _run_wrapped_lerobot_command(args: list[str], *, dry_run: bool) -> None:
    console.print(f"$ {shlex.join(args)}")
    if dry_run:
        return
    exit_code = _run_lerobot_subprocess(args)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def _run_lerobot_subprocess(args: list[str]) -> int:
    return subprocess.call(args)


@motor_app.command("test")
def motor_test(
    port: str = typer.Option(
        ...,
        "--port",
        help="Dynamixel serial port, or 'auto' for the single USB serial adapter found.",
    ),
    dxl_id: int = typer.Option(
        ...,
        "--id",
        min=0,
        max=252,
        help="Dynamixel ID to test.",
    ),
    baudrate: int = typer.Option(
        DEFAULT_BAUDRATE,
        "--baudrate",
        min=1,
        help="Dynamixel bus baudrate.",
    ),
    delta_ticks: int = typer.Option(
        DEFAULT_MOTOR_TEST_DELTA_TICKS,
        "--delta-ticks",
        help="Small relative motion in raw Dynamixel position ticks.",
    ),
    keep_torque: bool = typer.Option(
        False,
        "--keep-torque",
        help="Leave torque enabled after the test. Default is to torque off.",
    ),
    encoder_only: bool = typer.Option(
        False,
        "--encoder-only",
        help="Only read the present encoder position. Do not torque or move the motor.",
    ),
) -> None:
    """Move one specified Dynamixel ID a small amount and return it."""

    port = _resolve_port_option(port)

    if not encoder_only:
        console.print(
            f"Moving Dynamixel ID {dxl_id} on {port} by {delta_ticks} tick(s) and returning."
        )

    try:
        with HandController.from_port(port=port, baudrate=baudrate) as controller:
            if encoder_only:
                encoder_state = controller.read_motor_encoder_state(dxl_id)
            else:
                result = controller.test_motor(
                    dxl_id,
                    delta_ticks=delta_ticks,
                    keep_torque=keep_torque,
                )
    except (DynamixelBusError, ValueError) as exc:
        console.print(f"[red]Motor test failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if encoder_only:
        _print_motor_encoder_state(encoder_state)
    else:
        _print_motor_test_result(result)


@motor_app.command("scan")
def motor_scan(
    port: str = typer.Option(
        ...,
        "--port",
        help="Dynamixel serial port, or 'auto' for the single USB serial adapter found.",
    ),
    baudrate: int = typer.Option(
        DEFAULT_BAUDRATE,
        "--baudrate",
        min=1,
        help="Dynamixel bus baudrate.",
    ),
    all_ids: bool = typer.Option(
        False,
        "--all-ids",
        help="Scan Dynamixel IDs 0..15 instead of the LEAP V1 KAIZEN real-hand motor map.",
    ),
) -> None:
    """Read encoder state for the expected LEAP V1 KAIZEN Dynamixel IDs without moving motors."""

    port = _resolve_port_option(port)

    ids = tuple(range(16)) if all_ids else GHAND_RIGHT_MOTOR_IDS
    rows: list[tuple[int, str, str, str]] = []
    ok_count = 0
    try:
        with HandController.from_port(port=port, baudrate=baudrate) as controller:
            for dxl_id in ids:
                try:
                    state = controller.read_motor_encoder_state(dxl_id)
                except (DynamixelBusError, ValueError) as exc:
                    rows.append((dxl_id, "-", "-", str(exc)))
                else:
                    ok_count += 1
                    rows.append((dxl_id, str(state.model_number), str(state.position), "ok"))
    except (DynamixelBusError, ValueError) as exc:
        console.print(f"[red]Motor scan failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _print_motor_scan_result(rows, ok_count=ok_count, total=len(ids), port=port)
    if ok_count != len(ids):
        raise typer.Exit(code=1)


@export_app.command("mujoco")
def export_mujoco_model(
    output_dir: Path = typer.Option(
        ...,
        "--out",
        help="Directory where ghand_right_hand.xml and all built-in scene XMLs are written.",
    ),
    urdf: Path | None = typer.Option(
        None,
        "--urdf",
        help="Source URDF. Defaults to the packaged right-hand LEAP V1 KAIZEN model.",
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Load the exported XML with MuJoCo before returning.",
    ),
) -> None:
    """Export the packaged hand URDF into MuJoCo MJCF assets."""

    from ghand.sim.mujoco import MujocoExportError, export_mujoco_hand_model

    try:
        result = export_mujoco_hand_model(
            output_dir=output_dir,
            urdf_path=urdf,
            validate=validate,
        )
    except MujocoExportError as exc:
        console.print(f"[red]MuJoCo export failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[green]MuJoCo model exported.[/green]")
    console.print(f"hand XML: {result.hand_xml}")
    if result.scene_xml is not None:
        console.print(f"cube scene XML: {result.scene_xml}")
    if result.turn_handle_scene_xml is not None:
        console.print(f"turn handle scene XML: {result.turn_handle_scene_xml}")
    console.print(f"joints: {len(result.joint_names)}")
    console.print(f"actuators: {len(result.actuator_names)}")


@app.command("ui")
def operator_ui(
    port: str | None = typer.Option(
        None,
        "--port",
        help="Dynamixel serial port for the real hardware backend; 'auto' selects the single USB serial adapter found.",
    ),
    baudrate: int = typer.Option(
        DEFAULT_BAUDRATE,
        "--baudrate",
        min=1,
        help="Dynamixel bus baudrate for the real hardware backend.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Web UI host."),
    ui_port: int = typer.Option(
        8080, "--ui-port", min=1, max=65535, help="Web UI port."
    ),
    urdf: Path | None = typer.Option(
        None,
        "--urdf",
        help="URDF path for the operator UI 3D viewer. Defaults to packaged right-hand model.",
    ),
    model_root: Path | None = typer.Option(
        None,
        "--model-root",
        help="Model root served at /models. Defaults to packaged ghand assets.",
    ),
    enable_3d: bool = typer.Option(
        True,
        "--3d/--no-3d",
        help="Show the read-only URDF hand viewer in the operator UI.",
    ),
    control_mode: OperatorControlModeOption = typer.Option(
        OperatorControlModeOption.position,
        "--control-mode",
        case_sensitive=False,
        help="Hand runtime control mode: position or current-position.",
    ),
    default_current_limit_ma: float | None = typer.Option(
        None,
        "--default-current-limit-ma",
        min=0.0,
        help="Default Goal Current(102) limit in mA for current-position mode.",
    ),
    max_current_limit_ma: float | None = typer.Option(
        None,
        "--max-current-limit-ma",
        min=0.0,
        help="Clamp current-limit commands to this mA value before hardware writes.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Suppress motor writes for fake or real backends.",
    ),
) -> None:
    """Run the operator Web UI for fake or real hand-controller backends."""

    port = _resolve_port_option(port)

    runtime_control_mode = control_mode.runtime_value
    if (
        runtime_control_mode == OPERATOR_CONTROL_MODE_CURRENT_POSITION
        and default_current_limit_ma is None
    ):
        console.print(
            "[red]--default-current-limit-ma is required when "
            "--control-mode current-position.[/red]"
        )
        raise typer.Exit(code=1)

    if port is None:
        controller = HandController.from_fake()
        backend = "fake"
        runtime_dry_run = dry_run
        calibrations = make_default_joint_calibrations()
    else:
        assert port is not None
        controller = HandController.from_port(port=port, baudrate=baudrate)
        backend = f"dynamixel:{port}"
        runtime_dry_run = dry_run
        calibrations = make_default_joint_calibrations(
            motor_ids=GHAND_RIGHT_MOTOR_IDS,
        )
        if runtime_dry_run:
            console.print(
                "Starting real hardware UI in dry-run mode. "
                "Motor writes are suppressed."
            )
        else:
            console.print("Starting real hardware UI with motor writes enabled.")

    runtime = HandOperatorRuntime(
        controller,
        calibrations,
        backend=backend,
        dry_run=runtime_dry_run,
        control_mode=runtime_control_mode,
        default_current_limit_ma=default_current_limit_ma,
        max_current_limit_ma=max_current_limit_ma,
    )

    try:
        _launch_operator_ui(
            runtime,
            host=host,
            port=ui_port,
            enable_3d=enable_3d,
            model_root=model_root,
            urdf_path=urdf,
        )
    except RuntimeError as exc:
        console.print(f"[red]Operator UI failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _dataset_camera_description(
    *,
    teleop_camera: int,
    observation_camera: int | None,
    mujoco_camera_names: Sequence[str],
    mujoco: bool,
) -> str:
    if mujoco:
        if mujoco_camera_names:
            camera_text = ", ".join(mujoco_camera_names)
            return f"teleop camera {teleop_camera} and MuJoCo camera(s) {camera_text}"
        return f"teleop camera {teleop_camera} without observation camera"
    if observation_camera is None:
        return f"teleop camera {teleop_camera} without observation camera"
    return f"teleop camera {teleop_camera} and observation camera {observation_camera}"


def _mujoco_runtime_camera_names(runtime: HandOperatorRuntime) -> tuple[str, ...]:
    camera_names = getattr(runtime.controller.bus, "camera_names", None)
    if not callable(camera_names):
        return ()
    return tuple(str(name) for name in camera_names())


def _make_mujoco_camera_frame_source(
    *,
    runtime: HandOperatorRuntime,
    camera_names: Sequence[str],
    width: int,
    height: int,
) -> Callable[[], Mapping[str, object]]:
    render_camera = getattr(runtime.controller.bus, "render_camera", None)
    if not callable(render_camera):
        raise ValueError("MuJoCo runtime does not support camera rendering.")

    def read_camera_frames() -> Mapping[str, object]:
        return {
            camera_name: render_camera(camera_name, width=width, height=height)
            for camera_name in camera_names
        }

    return read_camera_frames


@contextmanager
def _open_observation_webcam(
    *,
    teleop_camera: int,
    observation_camera: int | None,
    width: int | None = None,
    height: int | None = None,
) -> Iterator[object | None]:
    if observation_camera is None or observation_camera == teleop_camera:
        yield None
        return
    with Webcam(
        camera=observation_camera,
        latest=True,
        width=width,
        height=height,
    ) as webcam:
        yield webcam


def _reset_dataset_runtime_environment(
    *,
    runtime: HandOperatorRuntime,
    enabled: bool,
    viewer: RerunViewer,
) -> bool:
    if not enabled:
        return False
    reset_scene = getattr(runtime.controller.bus, "reset_scene", None)
    if not callable(reset_scene):
        return False
    runtime.reset_motion_state()
    reset_scene()
    viewer.log_text("mujoco scene reset")
    console.print("MuJoCo scene reset.")
    return True


def _run_lerobot_dataset_reset_loop(
    *,
    webcam: object,
    tracker: object,
    runtime: HandOperatorRuntime,
    retargeter: Callable[[HandFeatures], JointTarget],
    feature_calibrator: Callable[[HandFeatures], HandFeatures] | None,
    viewer: RerunViewer,
    reset_time_s: float,
    keyboard_events: dict[str, bool],
    target_fps: float,
    hand_side: str,
    preferred_handedness: str | None,
    landmark_smoother: HandFrameOneEuroSmoother | None,
    observation_webcam: object | None,
    use_teleop_frame_as_observation: bool,
    dataset_camera_frame_source: Callable[[], Mapping[str, object]] | None,
    dataset_image_color_format: str,
) -> TeleopHandResult:
    console.print(
        f"Reset the environment for {reset_time_s:g}s. "
        "Press Right Arrow to continue early."
    )
    viewer.log_text("reset the environment")
    return _run_teleop_hand_loop(
        webcam=webcam,
        tracker=tracker,
        runtime=runtime,
        retargeter=retargeter,
        feature_calibrator=feature_calibrator,
        viewer=viewer,
        frames=0,
        control_time_s=reset_time_s,
        keyboard_events=keyboard_events,
        target_fps=target_fps,
        hold_missing_seconds=0.0,
        home_on_missing=True,
        hand_side=hand_side,
        preferred_handedness=preferred_handedness,
        landmark_smoother=landmark_smoother,
        observation_webcam=observation_webcam,
        use_teleop_frame_as_observation=use_teleop_frame_as_observation,
        dataset_camera_frame_source=dataset_camera_frame_source,
        dataset_image_color_format=dataset_image_color_format,
    )


def _run_teleop_hand_loop(
    *,
    webcam: object,
    tracker: object,
    runtime: HandOperatorRuntime,
    retargeter: Callable[[HandFeatures], JointTarget],
    viewer: RerunViewer,
    frames: int,
    target_fps: float,
    hold_missing_seconds: float,
    home_on_missing: bool,
    hand_side: str,
    preferred_handedness: str | None,
    control_time_s: float | None = None,
    keyboard_events: dict[str, bool] | None = None,
    observation_webcam: object | None = None,
    use_teleop_frame_as_observation: bool = False,
    dataset_camera_frame_source: Callable[[], Mapping[str, object]] | None = None,
    dataset_image_color_format: str = "bgr",
    landmark_smoother: HandFrameOneEuroSmoother | None = None,
    feature_calibrator: Callable[[HandFeatures], HandFeatures] | None = None,
    dataset_recorder: LeRobotStreamingRecorder | None = None,
    dataset_features: Mapping[str, Mapping[str, object]] | None = None,
    dataset_task: str | None = None,
    dataset_default_current_limit_ma: float | None = None,
    dataset_current_contact_threshold_ma: float = DEFAULT_CURRENT_CONTACT_THRESHOLD_MA,
) -> TeleopHandResult:
    processed = 0
    frames_with_hands = 0
    missing_home_events = 0
    last_target_time: float | None = None
    missing_home_sent = False
    last_rerun_hand_count: int | None = None
    last_console_hand_count: int | None = None
    last_tracking_key: tuple[str, int | None, int, int | None, str] | None = None
    operator_selector = OperatorHandSelector(target_handedness=preferred_handedness)
    target_period = 1.0 / target_fps if target_fps > 0 else 0.0
    started = 0.0
    if control_time_s is not None and target_fps <= 0:
        started = time.monotonic()
    frame_limit = frames if frames > 0 else None
    if control_time_s is not None and target_fps > 0:
        duration_frames = max(0, math.ceil(control_time_s * target_fps))
        frame_limit = (
            duration_frames
            if frame_limit is None
            else min(frame_limit, duration_frames)
        )

    while True:
        if keyboard_events is not None and keyboard_events["exit_early"]:
            keyboard_events["exit_early"] = False
            break
        if frame_limit is not None and processed >= frame_limit:
            break
        loop_started = time.monotonic()
        if (
            frame_limit is None
            and control_time_s is not None
            and loop_started - started >= control_time_s
        ):
            break
        frame = webcam.read()

        inference_started = time.monotonic()
        raw_hand_frame = tracker.track(frame)
        inference_seconds = time.monotonic() - inference_started
        selection = operator_selector.update(raw_hand_frame)
        if selection.track_changed and landmark_smoother is not None:
            landmark_smoother.reset()
        hand_frame = selection.hand_frame
        control_hand_frame = (
            landmark_smoother.filter(hand_frame)
            if landmark_smoother is not None
            else hand_frame
        )

        features = extract_hand_features(
            control_hand_frame,
            preferred_handedness=None,
        )
        if feature_calibrator is not None:
            features = feature_calibrator(features)
        target = retargeter(features)
        if target.joints:
            last_target_time = loop_started
            missing_home_sent = False
        elif _should_home_missing_target(
            now=loop_started,
            last_target_time=last_target_time,
            hold_missing_seconds=hold_missing_seconds,
            home_on_missing=home_on_missing,
            missing_home_sent=missing_home_sent,
        ):
            target = JointTarget.neutral(
                timestamp_ns=hand_frame.timestamp_ns, side=hand_side
            )
            missing_home_events += 1
            missing_home_sent = True
            viewer.log_text(
                "hand target missing; homing",
                level="WARN",
                timestamp_ns=hand_frame.timestamp_ns,
            )
        # Capture the observation (state + camera images) BEFORE applying the
        # action so recorded frames pair (o_t, a_t): the policy must see the
        # state the action acted on, matching how `lerobot-record` rolls out.
        observation_snapshot = (
            runtime.observe(now=loop_started) if dataset_recorder is not None else None
        )
        camera_frames: Mapping[str, object]
        if dataset_camera_frame_source is not None:
            camera_frames = dataset_camera_frame_source()
        elif observation_webcam is not None:
            camera_frames = {"front": observation_webcam.read()}
        elif use_teleop_frame_as_observation:
            camera_frames = {"front": frame}
        else:
            camera_frames = {}
        snapshot = apply_joint_target(runtime, target, now=loop_started)
        if camera_frames:
            viewer.log_observation_images(
                camera_frames,
                timestamp_ns=hand_frame.timestamp_ns,
                image_color_format=dataset_image_color_format,
            )
        if dataset_recorder is not None:
            if dataset_features is None or dataset_task is None:
                raise ValueError(
                    "dataset_features and dataset_task are required for recording."
                )
            assert observation_snapshot is not None
            dataset_recorder.add_frame(
                make_lerobot_frame_from_runtime_snapshot(
                    snapshot=observation_snapshot,
                    features=dict(dataset_features),
                    task=dataset_task,
                    camera_frames=camera_frames,
                    hand_frame=hand_frame,
                    action=_commanded_action_from_snapshot(
                        snapshot,
                        timestamp_ns=hand_frame.timestamp_ns,
                    ),
                    current_limit_action=_current_limits_from_snapshot(snapshot),
                    default_current_limit_ma=dataset_default_current_limit_ma,
                    current_contact_threshold_ma=dataset_current_contact_threshold_ma,
                    action_source="commanded",
                    image_color_format=dataset_image_color_format,
                )
            )
        processed += 1
        hand_count = len(hand_frame.hands)
        if hand_count:
            frames_with_hands += 1

        viewer.log_camera(frame, timestamp_ns=raw_hand_frame.timestamp_ns)
        viewer.log_hand_overlay(raw_hand_frame)

        if _should_log_teleop_rerun_ui(
            frame_index=processed,
            hand_count=hand_count,
            previous_hand_count=last_rerun_hand_count,
        ):
            viewer.log_hand_frame(raw_hand_frame)
            viewer.log_hand_features(features)
            viewer.log_joint_targets(target)
            viewer.log_runtime_snapshot(snapshot, timestamp_ns=hand_frame.timestamp_ns)
            viewer.log_latency(
                "mediapipe/inference",
                inference_seconds,
                timestamp_ns=hand_frame.timestamp_ns,
            )
            last_rerun_hand_count = hand_count

        tracking_key = _operator_hand_tracking_log_key(selection)
        if tracking_key != last_tracking_key:
            tracking_summary = _operator_hand_tracking_summary(selection)
            viewer.log_text(
                tracking_summary,
                level="INFO",
                timestamp_ns=raw_hand_frame.timestamp_ns,
            )
            console.print(tracking_summary)
            last_tracking_key = tracking_key

        if _should_print_teleop_summary(
            frame_index=processed,
            hand_count=hand_count,
            previous_hand_count=last_console_hand_count,
        ):
            console.print(_hand_frame_summary(processed, hand_frame.hands))
            last_console_hand_count = hand_count

        elapsed = time.monotonic() - loop_started
        if target_period > 0 and elapsed < target_period:
            time.sleep(target_period - elapsed)

    return TeleopHandResult(
        processed=processed,
        frames_with_hands=frames_with_hands,
        missing_home_events=missing_home_events,
    )


def _commanded_action_from_snapshot(
    snapshot: OperatorRuntimeSnapshot,
    *,
    timestamp_ns: int,
) -> RadianJointTarget:
    """Return the post-apply commanded joint radians as the recorded action."""

    return RadianJointTarget(
        timestamp_ns=timestamp_ns,
        joints={state.joint_name: float(state.commanded) for state in snapshot.joints},
    )


def _current_limits_from_snapshot(
    snapshot: OperatorRuntimeSnapshot,
) -> dict[str, float] | None:
    """Return the post-apply current limits keyed by joint position channel."""

    limits = {
        state.joint_name: float(state.current_limit_ma)
        for state in snapshot.joints
        if state.current_limit_ma is not None
    }
    return limits or None


@contextmanager
def _lerobot_record_keyboard_events() -> Iterator[dict[str, bool]]:
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }
    listener = None
    try:
        try:
            from pynput import keyboard

            def on_press(key: object) -> None:
                if key == keyboard.Key.right:
                    events["exit_early"] = True
                elif key == keyboard.Key.left:
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    events["stop_recording"] = True
                    events["exit_early"] = True

            listener = keyboard.Listener(on_press=on_press)
            listener.start()
        except Exception:
            listener = None
        yield events
    finally:
        if listener is not None:
            listener.stop()


def _should_log_teleop_rerun_ui(
    *,
    frame_index: int,
    hand_count: int,
    previous_hand_count: int | None,
) -> bool:
    return (
        frame_index == 1
        or hand_count != previous_hand_count
        or frame_index % TELEOP_RERUN_UI_EVERY_FRAMES == 0
    )


def _should_print_teleop_summary(
    *,
    frame_index: int,
    hand_count: int,
    previous_hand_count: int | None,
) -> bool:
    return (
        frame_index == 1
        or hand_count != previous_hand_count
        or frame_index % TELEOP_CONSOLE_EVERY_FRAMES == 0
    )


def _resolve_teleop_target_fps(
    target_fps: float | None,
    *,
    mujoco: bool,
) -> float:
    if target_fps is not None:
        return float(target_fps)
    if mujoco:
        return DEFAULT_MUJOCO_TELEOP_TARGET_FPS
    return DEFAULT_TELEOP_TARGET_FPS


def _resolve_dataset_fps(fps: int | None, *, mujoco: bool) -> int:
    if fps is not None:
        return int(fps)
    if mujoco:
        return int(DEFAULT_MUJOCO_CONTROL_HZ)
    return DEFAULT_FPS


def _resolve_teleop_target_velocity_limit(
    target_velocity_limit: float | None,
    *,
    mujoco: bool,
) -> float:
    if target_velocity_limit is not None:
        return float(target_velocity_limit)
    if mujoco:
        return DEFAULT_MUJOCO_TELEOP_TARGET_VELOCITY_LIMIT
    return DEFAULT_TELEOP_TARGET_VELOCITY_LIMIT


def _make_landmark_smoother() -> HandFrameOneEuroSmoother:
    return HandFrameOneEuroSmoother(
        min_cutoff=DEFAULT_TELEOP_ONE_EURO_MIN_CUTOFF,
        beta=DEFAULT_TELEOP_ONE_EURO_BETA,
        derivative_cutoff=DEFAULT_TELEOP_ONE_EURO_DERIVATIVE_CUTOFF,
        reset_after_seconds=DEFAULT_TELEOP_LANDMARK_FILTER_RESET_SECONDS,
    )


def _landmark_filter_summary() -> str:
    return (
        "landmark filter: one-euro "
        f"min_cutoff={DEFAULT_TELEOP_ONE_EURO_MIN_CUTOFF:g}Hz "
        f"beta={DEFAULT_TELEOP_ONE_EURO_BETA:g} "
        f"derivative_cutoff={DEFAULT_TELEOP_ONE_EURO_DERIVATIVE_CUTOFF:g}Hz "
        f"reset_after={DEFAULT_TELEOP_LANDMARK_FILTER_RESET_SECONDS:g}s"
    )


def _operator_hand_tracking_summary(selection: OperatorHandSelection) -> str:
    track = "-" if selection.track_id is None else str(selection.track_id)
    selected = (
        "-" if selection.selected_index is None else str(selection.selected_index)
    )
    return (
        "operator tracking: "
        f"state={selection.state.value} "
        f"track={track} "
        f"candidates={selection.candidates} "
        f"selected={selected} "
        f"reason={selection.reason}"
    )


def _operator_hand_tracking_log_key(
    selection: OperatorHandSelection,
) -> tuple[str, int | None, int, int | None, str]:
    return (
        selection.state.value,
        selection.track_id,
        selection.candidates,
        selection.selected_index,
        _operator_tracking_reason_category(selection.reason),
    )


def _operator_tracking_reason_category(reason: str) -> str:
    if reason.startswith("matched distance="):
        return "matched"
    if reason.startswith("recovered distance="):
        return "recovered"
    if reason.startswith("match distance too large"):
        return "match distance too large"
    return reason


def _load_retarget_calibration(
    path: Path | None,
) -> RetargetCalibrationLoadResult:
    if path is None:
        calibration = load_default_hand_retarget_calibration()
        return RetargetCalibrationLoadResult(
            calibration=calibration,
            calibrator=HandFeatureCalibrator(calibration),
            source="packaged default",
        )

    calibration_path = path.expanduser()
    calibration = load_hand_retarget_calibration(calibration_path)
    return RetargetCalibrationLoadResult(
        calibration=calibration,
        calibrator=HandFeatureCalibrator(calibration),
        source=str(calibration_path),
    )


def _resolve_enabled_anchor_pose_names(
    anchor_pose_names: list[str] | None,
) -> tuple[str, ...] | None:
    return tuple(anchor_pose_names or ()) or None


def _make_hand_retargeter(
    *,
    hand_side: str,
    calibration: HandRetargetCalibration,
    enabled_pose_names: tuple[str, ...] | None = None,
    softmax_temperature: float = DEFAULT_ANCHOR_SOFTMAX_TEMPERATURE,
) -> Callable[[HandFeatures], JointTarget]:
    if not calibration.pose_feature_anchors:
        raise ValueError(
            "anchor-pose retargeting requires pose_feature_anchors; "
            "rerun calibrate with the current 14-pose flow."
        )
    return AnchorPoseRetargeter.from_calibration(
        calibration,
        side=hand_side,
        softmax_temperature=softmax_temperature,
        enabled_pose_names=enabled_pose_names,
    )


def _run_hand_retarget_calibration_loop(
    *,
    webcam: object,
    tracker: object,
    viewer: RerunViewer,
    handedness: str,
    seconds_per_pose: float,
    frames_per_pose: int,
    ready_delay_seconds: float,
    min_valid_samples: int,
    target_fps: float,
    interactive: bool,
    key_reader: CalibrationKeyReader | None = None,
) -> HandRetargetCalibration:
    pose_samples: dict[str, tuple[HandFeatures, ...]] = {}
    target_period = 1.0 / target_fps if target_fps > 0 else 0.0

    pose_count = len(RETARGET_CALIBRATION_CAPTURE_POSES)
    with _calibration_key_reader(key_reader, enabled=interactive) as read_key:
        for index, pose in enumerate(RETARGET_CALIBRATION_CAPTURE_POSES, start=1):
            while True:
                console.print(f"[{index}/{pose_count}] ({pose.group}) {pose.title}")
                console.print(pose.instruction)
                viewer.log_calibration_pose_instruction(
                    pose_name=pose.name,
                    title=pose.title,
                    instruction=pose.instruction,
                    pose_index=index,
                    pose_count=pose_count,
                    status=_calibration_waiting_status()
                    if interactive
                    else "Timed capture",
                )
                if interactive:
                    console.print(
                        "Show the pose in the live preview, then press Space/Enter/C to capture. "
                        "Press Q to abort."
                    )
                    _wait_for_calibration_action(
                        webcam=webcam,
                        tracker=tracker,
                        viewer=viewer,
                        pose=pose,
                        pose_index=index,
                        pose_count=pose_count,
                        preferred_handedness=handedness,
                        target_period=target_period,
                        read_key=read_key,
                        actions={
                            "confirm": "capture",
                            "c": "capture",
                            "q": "abort",
                        },
                        status=_calibration_waiting_status(),
                    )
                elif ready_delay_seconds > 0:
                    time.sleep(ready_delay_seconds)

                samples = _capture_retarget_calibration_pose_samples(
                    webcam=webcam,
                    tracker=tracker,
                    viewer=viewer,
                    pose=pose,
                    pose_index=index,
                    pose_count=pose_count,
                    preferred_handedness=handedness,
                    seconds_per_pose=seconds_per_pose,
                    frames_per_pose=frames_per_pose,
                    target_period=target_period,
                    status="Capturing",
                )
                if len(samples) < min_valid_samples:
                    message = (
                        f"Pose {pose.name!r} captured {len(samples)} valid sample(s); "
                        f"need at least {min_valid_samples}."
                    )
                    if not interactive:
                        raise ValueError(message)
                    console.print(f"[yellow]{message} Retake the pose.[/yellow]")
                    viewer.log_text(message, level="WARN")
                    continue

                console.print(
                    f"captured {len(samples)} valid sample(s) for {pose.name}."
                )
                if not interactive:
                    pose_samples[pose.name] = tuple(samples)
                    break

                captured_status = _calibration_captured_status(
                    valid_samples=len(samples),
                    final_pose=index == pose_count,
                )
                viewer.log_calibration_pose_instruction(
                    pose_name=pose.name,
                    title=pose.title,
                    instruction=pose.instruction,
                    pose_index=index,
                    pose_count=pose_count,
                    valid_samples=len(samples),
                    status=captured_status,
                )
                console.print(captured_status)
                if index == pose_count:
                    pose_samples[pose.name] = tuple(samples)
                    break

                action = _wait_for_calibration_action(
                    webcam=webcam,
                    tracker=tracker,
                    viewer=viewer,
                    pose=pose,
                    pose_index=index,
                    pose_count=pose_count,
                    preferred_handedness=handedness,
                    target_period=target_period,
                    read_key=read_key,
                    actions={
                        "confirm": "next",
                        "n": "next",
                        "r": "retake",
                        "q": "abort",
                    },
                    status=captured_status,
                    valid_samples=len(samples),
                )
                if action == "retake":
                    console.print("Retaking current pose.")
                    continue
                pose_samples[pose.name] = tuple(samples)
                break

    return _build_hand_retarget_calibration_from_pose_samples(
        pose_samples,
        handedness=handedness,
    )


def _wait_for_calibration_action(
    *,
    webcam: object,
    tracker: object,
    viewer: RerunViewer,
    pose: RetargetCalibrationPoseInstruction,
    pose_index: int,
    pose_count: int,
    preferred_handedness: str,
    target_period: float,
    read_key: CalibrationKeyReader,
    actions: Mapping[str, str],
    status: str,
    valid_samples: int = 0,
) -> str:
    while True:
        action = _calibration_key_action(read_key(), actions)
        if action == "abort":
            raise KeyboardInterrupt
        if action is not None:
            return action

        loop_started = time.monotonic() if target_period > 0 else 0.0
        frame = webcam.read()
        hand_frame = tracker.track(frame)
        features = extract_hand_features(
            hand_frame,
            preferred_handedness=preferred_handedness,
        )
        _log_calibration_frame(
            viewer=viewer,
            frame=frame,
            hand_frame=hand_frame,
            features=features,
            pose=pose,
            pose_index=pose_index,
            pose_count=pose_count,
            valid_samples=valid_samples,
            status=status,
        )
        if target_period > 0:
            elapsed = time.monotonic() - loop_started
            if elapsed < target_period:
                time.sleep(target_period - elapsed)


def _calibration_key_action(
    key: str | None,
    actions: Mapping[str, str],
) -> str | None:
    if key is None:
        return None
    return actions.get(_normalize_calibration_key(key))


def _normalize_calibration_key(key: str) -> str:
    if key in {" ", "\r", "\n"}:
        return "confirm"
    return key.lower()


@contextmanager
def _calibration_key_reader(
    key_reader: CalibrationKeyReader | None,
    *,
    enabled: bool,
) -> Iterator[CalibrationKeyReader]:
    if key_reader is not None:
        yield key_reader
        return
    if not enabled:
        yield lambda: None
        return
    with _terminal_key_reader() as read_key:
        yield read_key


@contextmanager
def _terminal_key_reader() -> Iterator[CalibrationKeyReader]:
    import select
    import termios
    import tty

    if not _stdin_supports_interactive_keys():
        raise ValueError("Interactive retarget calibration requires a terminal.")

    fd = sys.stdin.fileno()
    original_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)

        def read_key() -> str | None:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if not readable:
                return None
            return sys.stdin.read(1)

        yield read_key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)


def _stdin_supports_interactive_keys() -> bool:
    return sys.stdin.isatty()


def _calibration_waiting_status() -> str:
    return "Waiting for capture: press Space/Enter/C. Press Q to abort."


def _calibration_captured_status(*, valid_samples: int, final_pose: bool) -> str:
    if final_pose:
        return f"Captured {valid_samples} valid sample(s). Finishing calibration."

    return (
        f"Captured {valid_samples} valid sample(s). "
        "Press Space/Enter/N to advance to the next pose, R to retake, or Q to abort."
    )


def _log_calibration_frame(
    *,
    viewer: RerunViewer,
    frame: object,
    hand_frame: object,
    features: HandFeatures,
    pose: RetargetCalibrationPoseInstruction,
    pose_index: int,
    pose_count: int,
    valid_samples: int,
    status: str,
) -> None:
    viewer.log_camera(frame, timestamp_ns=hand_frame.timestamp_ns)
    viewer.log_hand_frame(hand_frame)
    viewer.log_hand_features(features)
    viewer.log_calibration_pose_instruction(
        pose_name=pose.name,
        title=pose.title,
        instruction=pose.instruction,
        pose_index=pose_index,
        pose_count=pose_count,
        valid_samples=valid_samples,
        timestamp_ns=hand_frame.timestamp_ns,
        status=status,
    )


def _capture_retarget_calibration_pose_samples(
    *,
    webcam: object,
    tracker: object,
    viewer: RerunViewer,
    pose: RetargetCalibrationPoseInstruction,
    pose_index: int,
    pose_count: int,
    preferred_handedness: str,
    seconds_per_pose: float,
    frames_per_pose: int,
    target_period: float,
    status: str,
) -> list[HandFeatures]:
    samples: list[HandFeatures] = []
    processed = 0
    started = time.monotonic() if frames_per_pose == 0 else 0.0

    while True:
        if frames_per_pose > 0 and processed >= frames_per_pose:
            break
        if (
            frames_per_pose == 0
            and processed > 0
            and time.monotonic() - started >= seconds_per_pose
        ):
            break

        loop_started = time.monotonic() if target_period > 0 else 0.0
        frame = webcam.read()
        hand_frame = tracker.track(frame)
        features = extract_hand_features(
            hand_frame,
            preferred_handedness=preferred_handedness,
        )
        if features.valid:
            samples.append(features)

        _log_calibration_frame(
            viewer=viewer,
            frame=frame,
            hand_frame=hand_frame,
            features=features,
            pose=pose,
            pose_index=pose_index,
            pose_count=pose_count,
            valid_samples=len(samples),
            status=status,
        )

        processed += 1
        if processed == 1:
            console.print(_hand_frame_summary(processed, hand_frame.hands))

        if target_period > 0:
            elapsed = time.monotonic() - loop_started
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

    return samples


def _build_hand_retarget_calibration_from_pose_samples(
    pose_samples: Mapping[str, Sequence[HandFeatures]],
    *,
    handedness: str,
) -> HandRetargetCalibration:
    missing_poses = set(RETARGET_CALIBRATION_POSES) - set(pose_samples)
    if missing_poses:
        missing = ", ".join(sorted(missing_poses))
        raise ValueError(f"Missing retarget calibration pose sample(s): {missing}.")

    flat_closed = pose_samples["flat_closed"]
    fist = pose_samples["fist"]
    flat_spread = pose_samples["flat_spread"]
    scissors = pose_samples["scissors"]
    index_pinch = pose_samples["index_pinch"]

    return HandRetargetCalibration(
        version=RETARGET_CALIBRATION_SCHEMA_VERSION,
        handedness=handedness,
        poses={
            pose: RetargetPoseStats(samples=len(samples))
            for pose, samples in pose_samples.items()
        },
        finger_curl={
            finger: CurlCalibration(
                open=_median_feature(flat_closed, "finger_curl", finger),
                closed=_median_feature(fist, "finger_curl", finger),
            )
            for finger in FINGER_NAMES
        },
        finger_joint_curl={
            finger: {
                joint: _curl_calibration_with_fallback(
                    open=_median_nested_feature(
                        flat_closed, "finger_joint_curl", finger, joint
                    ),
                    closed=_median_nested_feature(
                        fist, "finger_joint_curl", finger, joint
                    ),
                )
                for joint in FINGER_CURL_JOINT_NAMES
            }
            for finger in FINGER_NAMES
        },
        finger_spread={
            finger: _spread_calibration_with_fallback(
                neutral=_median_feature(flat_closed, "finger_spread", finger),
                spread=_median_feature(
                    scissors if finger == "middle" else flat_spread,
                    "finger_spread",
                    finger,
                ),
            )
            for finger in SPREAD_FINGER_NAMES
        },
        pinch={
            finger: (
                _pinch_calibration_with_fallback(
                    open=_median_feature(flat_closed, "pinch", finger),
                    closed=_median_feature(index_pinch, "pinch", finger),
                )
                if finger == "index"
                else PinchCalibration(open=0.0, closed=1.0)
            )
            for finger in PINCH_FINGER_NAMES
        },
        thumb_abduction=_thumb_abduction_calibration_with_fallback(
            closed=_median_scalar(flat_closed, "thumb_abduction"),
            spread=_median_scalar(flat_spread, "thumb_abduction"),
        ),
        thumb_pose_anchors={
            pose: ThumbPoseAnchor(features=_median_thumb_pose(samples))
            for pose, samples in pose_samples.items()
        },
        pose_feature_anchors={
            pose: _pose_feature_anchor(samples)
            for pose, samples in pose_samples.items()
        },
    )


def _pose_feature_anchor(samples: Sequence[HandFeatures]) -> PoseFeatureAnchor:
    if not samples:
        raise ValueError("Cannot compute calibration median without samples.")
    return PoseFeatureAnchor(
        finger_curl={
            finger: _median_feature(samples, "finger_curl", finger)
            for finger in FINGER_NAMES
        },
        finger_joint_curl={
            finger: {
                joint: _median_nested_feature(
                    samples, "finger_joint_curl", finger, joint
                )
                for joint in FINGER_CURL_JOINT_NAMES
            }
            for finger in FINGER_NAMES
        },
        finger_spread={
            finger: _median_feature(samples, "finger_spread", finger)
            for finger in SPREAD_FINGER_NAMES
        },
        pinch={
            finger: _median_feature(samples, "pinch", finger)
            for finger in PINCH_FINGER_NAMES
        },
        thumb_abduction=_median_scalar(samples, "thumb_abduction"),
        thumb_pose=_median_thumb_pose(samples),
        palm_size=_median_scalar(samples, "palm_size"),
    )


def _median_feature(
    samples: Sequence[HandFeatures], feature_name: str, finger: str
) -> float:
    if not samples:
        raise ValueError("Cannot compute calibration median without samples.")
    values = [float(getattr(sample, feature_name)[finger]) for sample in samples]
    return float(median(values))


def _median_nested_feature(
    samples: Sequence[HandFeatures],
    feature_name: str,
    finger: str,
    joint: str,
) -> float:
    if not samples:
        raise ValueError("Cannot compute calibration median without samples.")
    values = [float(getattr(sample, feature_name)[finger][joint]) for sample in samples]
    return float(median(values))


def _median_scalar(samples: Sequence[HandFeatures], feature_name: str) -> float:
    if not samples:
        raise ValueError("Cannot compute calibration median without samples.")
    values = [float(getattr(sample, feature_name)) for sample in samples]
    return float(median(values))


def _median_thumb_pose(samples: Sequence[HandFeatures]) -> dict[str, float]:
    if not samples:
        raise ValueError("Cannot compute calibration median without samples.")
    return {
        feature_name: float(
            median(float(sample.thumb_pose[feature_name]) for sample in samples)
        )
        for feature_name in THUMB_POSE_FEATURE_NAMES
    }


def _spread_calibration_with_fallback(
    *, neutral: float, spread: float
) -> SpreadCalibration:
    if abs(spread - neutral) > 1e-9:
        return SpreadCalibration(neutral=neutral, spread=spread)
    fallback_spread = 1.0 if neutral < 1.0 else -1.0
    return SpreadCalibration(neutral=neutral, spread=fallback_spread)


def _curl_calibration_with_fallback(*, open: float, closed: float) -> CurlCalibration:
    if closed - open > 1e-9:
        return CurlCalibration(open=open, closed=closed)
    return CurlCalibration(open=0.0, closed=1.0)


def _pinch_calibration_with_fallback(*, open: float, closed: float) -> PinchCalibration:
    if closed - open > 1e-9:
        return PinchCalibration(open=open, closed=closed)
    return PinchCalibration(open=0.0, closed=1.0)


def _thumb_abduction_calibration_with_fallback(
    *,
    closed: float,
    spread: float,
) -> ThumbAbductionCalibration:
    if abs(spread - closed) > 1e-9:
        return ThumbAbductionCalibration(closed=closed, spread=spread)
    fallback_spread = 1.0 if closed < 1.0 else 0.0
    return ThumbAbductionCalibration(closed=closed, spread=fallback_spread)


def _print_retarget_calibration_summary(calibration: HandRetargetCalibration) -> None:
    table = Table(title="Retarget Calibration")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("version", str(calibration.version))
    table.add_row("handedness", calibration.handedness)
    for pose in RETARGET_CALIBRATION_CAPTURE_POSES:
        if pose.name in calibration.poses:
            table.add_row(
                f"{pose.group}/{pose.name} samples",
                str(calibration.poses[pose.name].samples),
            )
    console.print(table)


def _should_home_missing_target(
    *,
    now: float,
    last_target_time: float | None,
    hold_missing_seconds: float,
    home_on_missing: bool,
    missing_home_sent: bool,
) -> bool:
    if not home_on_missing or missing_home_sent:
        return False
    if last_target_time is None:
        return True
    return now - last_target_time >= hold_missing_seconds


def _show_hand_frame(
    window_name: str,
    frame: object,
    hand_frame: object,
) -> tuple[bool, bool]:
    """Return (display_enabled, stop_requested) for one OpenCV display update."""

    try:
        import cv2

        draw_hand_frame(frame, hand_frame)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return True, True
    except Exception as exc:
        console.print(
            "[yellow]OpenCV display unavailable; continuing without --display:[/yellow] "
            f"{_short_exception_message(exc)}"
        )
        return False, False

    return True, False


def _close_display_windows(display_enabled: bool) -> None:
    if not display_enabled:
        return

    try:
        import cv2

        cv2.destroyAllWindows()
    except Exception as exc:
        console.print(
            "[yellow]OpenCV display cleanup skipped:[/yellow] "
            f"{_short_exception_message(exc)}"
        )


def _short_exception_message(exc: Exception, limit: int = 180) -> str:
    message = (
        str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    )
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _hand_frame_summary(frame_index: int, hands: object) -> str:
    hand_list = list(hands)
    if not hand_list:
        return f"frame={frame_index} hands=0"

    summary = ", ".join(
        f"{hand.handedness}:{hand.score:.2f}:landmarks={len(hand.landmarks)}"
        for hand in hand_list
    )
    return f"frame={frame_index} hands={len(hand_list)} {summary}"


def _print_motor_encoder_state(state: MotorEncoderState) -> None:
    table = Table(title="Motor Encoder")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("id", str(state.dxl_id))
    table.add_row("model", str(state.model_number))
    table.add_row("present position", str(state.position))
    console.print(table)
    console.print("[green]Motor encoder read completed.[/green]")


def _print_motor_scan_result(
    rows: Sequence[tuple[int, str, str, str]],
    *,
    ok_count: int,
    total: int,
    port: str,
) -> None:
    table = Table(title=f"Motor Scan: {port}")
    table.add_column("ID")
    table.add_column("Model")
    table.add_column("Present Position")
    table.add_column("Status")
    for dxl_id, model, position, status in rows:
        style = None if status == "ok" else "red"
        table.add_row(str(dxl_id), model, position, status, style=style)
    console.print(table)
    if ok_count == total:
        console.print(f"[green]Motor scan completed: {ok_count}/{total} OK.[/green]")
    else:
        console.print(f"[red]Motor scan completed: {ok_count}/{total} OK.[/red]")


def _print_motor_test_result(result: MotorTestResult) -> None:
    table = Table(title="Motor Test")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("id", str(result.dxl_id))
    table.add_row("model", str(result.model_number))
    table.add_row("start position", str(result.start_position))
    table.add_row("target position", str(result.target_position))
    table.add_row("returned position", str(result.returned_position))
    table.add_row("final position", str(result.final_position))
    table.add_row("applied delta ticks", str(result.applied_delta_ticks))
    table.add_row("temperature", f"{result.temperature_celsius} C")
    table.add_row("input voltage", f"{result.input_voltage:.1f} V")
    table.add_row("hardware error status", str(result.hardware_error_status))
    table.add_row("torque after", "on" if result.torque_enabled_after else "off")
    console.print(table)
    console.print("[green]Motor test completed.[/green]")


def _launch_operator_ui(
    runtime: HandOperatorRuntime,
    *,
    host: str,
    port: int,
    enable_3d: bool = True,
    model_root: Path | None = None,
    urdf_path: Path | None = None,
    viewer_angle_scale: float = 1.0,
) -> None:
    from ghand.ui import run_operator_ui

    run_operator_ui(
        runtime,
        host=host,
        port=port,
        enable_3d=enable_3d,
        model_root=model_root,
        urdf_path=urdf_path,
        viewer_angle_scale=viewer_angle_scale,
    )


if __name__ == "__main__":
    app()
