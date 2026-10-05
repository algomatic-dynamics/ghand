import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

import ghand.cli as cli
from ghand.cli import app
from ghand.control import make_fake_hand_runtime
from ghand.hand import (
    DEFAULT_JOINT_DIRECTIONS,
    GHAND_RIGHT_MOTOR_IDS,
    JointTarget,
)
from ghand.hardware import DynamixelBusError, FakeBus, MotorEncoderState, MotorTestResult
from ghand.retargeting import (
    CurlCalibration,
    FINGER_NAMES,
    PINCH_FINGER_NAMES,
    RETARGET_CALIBRATION_CAPTURE_POSE_NAMES,
    RETARGET_CALIBRATION_POSES,
    SPREAD_FINGER_NAMES,
    HandFeatures,
    HandRetargetCalibration,
    PinchCalibration,
    PoseFeatureAnchor,
    RetargetPoseStats,
    SpreadCalibration,
    load_hand_retarget_calibration,
    save_hand_retarget_calibration,
)
from ghand.teleop.hand_pose import HandDetection, HandFrame, Landmark


runner = CliRunner()


def test_mujoco_cli_control_defaults_share_physics_profile() -> None:
    assert cli._resolve_teleop_target_fps(None, mujoco=True) == pytest.approx(
        cli.DEFAULT_MUJOCO_CONTROL_HZ
    )
    assert cli._resolve_dataset_fps(None, mujoco=True) == int(
        cli.DEFAULT_MUJOCO_CONTROL_HZ
    )
    assert cli.DEFAULT_RUN_POLICY_FPS == int(cli.DEFAULT_MUJOCO_CONTROL_HZ)


class FakeMotorController:
    instances = []

    def __init__(self, *, port: str, baudrate: int) -> None:
        self.port = port
        self.baudrate = baudrate
        self.encoder_calls = []
        self.test_calls = []
        self.fail_encoder_ids = set()
        FakeMotorController.instances.append(self)

    @classmethod
    def from_port(cls, *, port: str, baudrate: int):
        return cls(port=port, baudrate=baudrate)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        pass

    def read_motor_encoder_state(self, dxl_id: int) -> MotorEncoderState:
        self.encoder_calls.append(dxl_id)
        if dxl_id in self.fail_encoder_ids:
            raise DynamixelBusError(f"Dynamixel ID {dxl_id} ping failed")
        return MotorEncoderState(dxl_id=dxl_id, model_number=1234, position=2048)

    def test_motor(self, dxl_id: int, **kwargs) -> MotorTestResult:
        self.test_calls.append((dxl_id, kwargs))
        return MotorTestResult(
            dxl_id=dxl_id,
            model_number=1234,
            start_position=2048,
            target_position=2112,
            returned_position=2112,
            final_position=2048,
            applied_delta_ticks=64,
            temperature_celsius=31,
            input_voltage=12.2,
            hardware_error_status=0,
            torque_enabled_after=kwargs["keep_torque"],
        )


class FakeWebcam:
    instances = []

    def __init__(
        self,
        camera: int,
        *,
        latest: bool = False,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
    ) -> None:
        self.camera = camera
        self.latest = latest
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = [
            np.zeros((2, 3, 3), dtype=np.uint8),
            np.ones((2, 3, 3), dtype=np.uint8),
        ]
        self.reads = 0
        FakeWebcam.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        pass

    def read(self):
        frame = self.frames[self.reads % len(self.frames)]
        self.reads += 1
        return frame


class FakeHandTracker:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.tracks = 0
        FakeHandTracker.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        pass

    def track(self, frame):
        self.tracks += 1
        return HandFrame(
            timestamp_ns=self.tracks,
            image_width=3,
            image_height=2,
            hands=(
                HandDetection(
                    handedness="Right",
                    score=0.9,
                    landmarks=tuple(_open_hand_landmarks()),
                ),
            ),
        )


class FakeLandmarkSmoother:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls = []
        self.reset_calls = 0
        FakeLandmarkSmoother.instances.append(self)

    def reset(self) -> None:
        self.reset_calls += 1

    def filter(self, hand_frame):
        self.calls.append(hand_frame)
        hand = hand_frame.hands[0]
        landmarks = list(hand.landmarks)
        landmarks[0] = Landmark(0.25, landmarks[0].y, landmarks[0].z)
        return HandFrame(
            timestamp_ns=hand_frame.timestamp_ns,
            image_width=hand_frame.image_width,
            image_height=hand_frame.image_height,
            hands=(
                HandDetection(
                    handedness=hand.handedness,
                    score=hand.score,
                    landmarks=tuple(landmarks),
                    world_landmarks=hand.world_landmarks,
                ),
            ),
        )


class FakeRerunViewer:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls = []
        FakeRerunViewer.instances.append(self)

    def start(self):
        self.calls.append(("start",))
        return self

    def log_camera(self, frame, *, timestamp_ns=None) -> None:
        self.calls.append(("camera", timestamp_ns))

    def log_observation_images(
        self,
        camera_frames,
        *,
        timestamp_ns=None,
        image_color_format="bgr",
    ) -> None:
        self.calls.append(
            (
                "observation_images",
                timestamp_ns,
                image_color_format,
                dict(camera_frames),
            )
        )

    def log_hand_frame(self, hand_frame) -> None:
        self.calls.append(("hand", hand_frame.timestamp_ns))

    def log_hand_overlay(self, hand_frame) -> None:
        self.calls.append(("overlay", hand_frame.timestamp_ns, len(hand_frame.hands)))

    def log_hand_features(self, features) -> None:
        self.calls.append(("features", features.timestamp_ns, features.valid, features))

    def log_calibration_pose_instruction(
        self,
        *,
        pose_name: str,
        title: str,
        instruction: str,
        pose_index: int,
        pose_count: int,
        valid_samples: int = 0,
        timestamp_ns=None,
        status: str | None = None,
    ) -> None:
        self.calls.append(
            (
                "pose",
                pose_name,
                title,
                instruction,
                pose_index,
                pose_count,
                valid_samples,
                timestamp_ns,
                status,
            )
        )

    def log_joint_targets(self, target) -> None:
        self.calls.append(("target", target.timestamp_ns, dict(target.joints)))

    def log_runtime_snapshot(self, snapshot, *, timestamp_ns=None) -> None:
        self.calls.append(
            (
                "snapshot",
                timestamp_ns,
                len(snapshot.joints),
                snapshot.backend,
                snapshot.dry_run,
            )
        )

    def log_latency(self, name, seconds, *, timestamp_ns=None) -> None:
        self.calls.append(("latency", name, timestamp_ns))

    def log_text(self, message, *, level="INFO", timestamp_ns=None) -> None:
        self.calls.append(("text", message, level, timestamp_ns))


class FakeStreamingRecorder:
    create_calls = []
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.frames = []
        self.total_frames = 0
        self.frames_in_episode = 0
        self.saved_episodes = 0
        self.closed = False
        FakeStreamingRecorder.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.create_calls = []
        cls.instances = []

    @classmethod
    def create(cls, **kwargs):
        cls.create_calls.append(kwargs)
        return cls(**kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def add_frame(self, frame) -> None:
        self.frames.append(frame)
        self.total_frames += 1
        self.frames_in_episode += 1

    def save_episode(self) -> None:
        self.saved_episodes += 1
        self.frames_in_episode = 0

    def discard_episode(self) -> None:
        if self.frames_in_episode:
            del self.frames[-self.frames_in_episode :]
            self.total_frames -= self.frames_in_episode
            self.frames_in_episode = 0

    def close(self, *, save_pending_episode: bool = True) -> None:
        if save_pending_episode and self.frames_in_episode:
            self.save_episode()
        self.closed = True


class SequenceWebcam:
    def __init__(self, frame_count: int) -> None:
        self.frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(frame_count)]
        self.index = 0

    def read(self):
        frame = self.frames[self.index]
        self.index += 1
        return frame


class SequenceTracker:
    def __init__(self, hand_frames: list[HandFrame]) -> None:
        self.hand_frames = hand_frames
        self.index = 0

    def track(self, frame):
        hand_frame = self.hand_frames[self.index]
        self.index += 1
        return hand_frame


class SequenceRetargeter:
    def __init__(self, targets: list[JointTarget]) -> None:
        self.targets = targets
        self.index = 0

    def retarget_frame(self, hand_frame, *, preferred_handedness=None):
        target = self.targets[self.index]
        self.index += 1
        return target

    def __call__(self, features):
        target = self.targets[self.index]
        self.index += 1
        return target


class FakeClock:
    def __init__(self, monotonic_values: list[float]) -> None:
        self.monotonic_values = monotonic_values
        self.sleep_calls = []

    def monotonic(self) -> float:
        return self.monotonic_values.pop(0)

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)


def test_calibrate_hand_retarget_cli_captures_pose_calibration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    feature_queue = _extended_calibration_feature_queue()

    def fake_extract_hand_features(hand_frame, *, preferred_handedness=None):
        assert preferred_handedness == "Right"
        return feature_queue.pop(0)

    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "HandFrameOneEuroSmoother", FakeLandmarkSmoother)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(cli, "extract_hand_features", fake_extract_hand_features)
    output = tmp_path / "right_hand_retarget_calibration.json"
    rerun_save = tmp_path / "calibration.rrd"

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--camera",
            "2",
            "--out",
            str(output),
            "--timed",
            "--frames-per-pose",
            "2",
            "--ready-delay-seconds",
            "0",
            "--target-fps",
            "0",
            "--rerun",
            "--rerun-save",
            str(rerun_save),
        ],
    )

    assert result.exit_code == 0
    assert "Retarget calibration saved" in result.output
    assert feature_queue == []
    assert FakeWebcam.instances[0].camera == 2
    assert FakeHandTracker.instances[0].kwargs["model_path"] == Path("/tmp/model.task")
    assert FakeHandTracker.instances[0].tracks == 28
    viewer = FakeRerunViewer.instances[0]
    assert viewer.kwargs["spawn"] is True
    assert viewer.kwargs["save_path"] == rerun_save
    assert ("start",) in viewer.calls
    assert sum(1 for call in viewer.calls if call[0] == "camera") == 28
    assert sum(1 for call in viewer.calls if call[0] == "features") == 28
    pose_calls = [call for call in viewer.calls if call[0] == "pose"]
    assert len(pose_calls) == 42
    assert pose_calls[0][1] == "flat_closed"
    assert pose_calls[-1][1] == "thumbs_up"
    assert pose_calls[-1][6] == 2
    calibration = load_hand_retarget_calibration(output)
    assert calibration.handedness == "Right"
    assert set(calibration.poses) == set(RETARGET_CALIBRATION_CAPTURE_POSE_NAMES)
    assert calibration.poses["flat_closed"].samples == 2
    assert calibration.poses["fist"].samples == 2
    assert calibration.poses["flat_spread"].samples == 2
    assert calibration.poses["scissors"].samples == 2
    assert calibration.poses["index_pinch"].samples == 2
    assert calibration.poses["thumbs_up"].samples == 2
    assert calibration.finger_curl["index"].open == pytest.approx(0.15)
    assert calibration.finger_curl["index"].closed == pytest.approx(0.75)
    assert calibration.finger_joint_curl["index"]["mcp"].open == pytest.approx(0.15)
    assert calibration.finger_joint_curl["index"]["mcp"].closed == pytest.approx(0.75)
    assert calibration.finger_spread["index"].neutral == pytest.approx(0.2)
    assert calibration.finger_spread["index"].spread == pytest.approx(-0.6)
    assert calibration.finger_spread["middle"].spread == pytest.approx(-0.4)
    assert calibration.pinch["index"].open == pytest.approx(0.1)
    assert calibration.pinch["index"].closed == pytest.approx(0.8)
    assert calibration.pinch["middle"].open == 0.0
    assert calibration.pinch["middle"].closed == 1.0
    assert calibration.thumb_abduction.closed == pytest.approx(0.2)
    assert calibration.thumb_abduction.spread == pytest.approx(0.8)
    assert calibration.thumb_pose_anchors is not None
    assert calibration.thumb_pose_anchors["flat_closed"].features[
        "lateral"
    ] == pytest.approx(0.2)
    assert calibration.thumb_pose_anchors["index_pinch"].features[
        "index_pinch"
    ] == pytest.approx(0.8)
    assert set(calibration.pose_feature_anchors) == set(
        RETARGET_CALIBRATION_CAPTURE_POSE_NAMES
    )
    assert calibration.pose_feature_anchors["open_index_flex"].finger_curl[
        "index"
    ] == pytest.approx(0.55)


def test_calibrate_hand_retarget_cli_rejects_missing_sampling_window() -> None:
    result = runner.invoke(
        app,
        [
            "calibrate",
            "--out",
            "/tmp/calibration.json",
            "--seconds-per-pose",
            "0",
            "--frames-per-pose",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "Use --frames-per-pose" in result.output


def test_calibrate_hand_retarget_cli_rejects_interactive_without_tty(
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "calibrate",
            "--out",
            str(tmp_path / "calibration.json"),
            "--frames-per-pose",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "requires a terminal" in result.output
    assert "--timed" in result.output


def test_hand_retarget_calibration_loop_interactive_keys_capture_and_advance(
    monkeypatch,
) -> None:
    feature_queue = _extended_calibration_feature_queue()
    keys = list("cn" * 13 + "c")

    def fake_extract_hand_features(hand_frame, *, preferred_handedness=None):
        assert preferred_handedness == "Right"
        return feature_queue.pop(0)

    def read_key() -> str | None:
        return keys.pop(0)

    monkeypatch.setattr(cli, "extract_hand_features", fake_extract_hand_features)
    viewer = FakeRerunViewer()

    calibration = cli._run_hand_retarget_calibration_loop(
        webcam=SequenceWebcam(frame_count=28),
        tracker=SequenceTracker(
            [_hand_frame(timestamp_ns=index, has_hand=True) for index in range(28)]
        ),
        viewer=viewer,
        handedness="Right",
        seconds_per_pose=0.0,
        frames_per_pose=2,
        ready_delay_seconds=0.0,
        min_valid_samples=2,
        target_fps=0.0,
        interactive=True,
        key_reader=read_key,
    )

    assert keys == []
    assert feature_queue == []
    assert set(calibration.poses) == set(RETARGET_CALIBRATION_CAPTURE_POSE_NAMES)
    assert calibration.poses["flat_closed"].samples == 2
    assert calibration.poses["fist"].samples == 2
    assert calibration.poses["flat_spread"].samples == 2
    assert calibration.poses["scissors"].samples == 2
    assert calibration.poses["index_pinch"].samples == 2
    assert calibration.poses["thumbs_up"].samples == 2
    assert calibration.finger_curl["index"].open == pytest.approx(0.15)
    assert calibration.finger_curl["index"].closed == pytest.approx(0.75)
    assert calibration.thumb_abduction.closed == pytest.approx(0.2)
    assert calibration.thumb_abduction.spread == pytest.approx(0.8)
    assert calibration.thumb_pose_anchors is not None
    assert calibration.thumb_pose_anchors["scissors"].features[
        "lateral"
    ] == pytest.approx(0.5)
    pose_statuses = [call[8] for call in viewer.calls if call[0] == "pose" and call[8]]
    assert any("Waiting for capture" in status for status in pose_statuses)
    assert any("Captured 2 valid sample" in status for status in pose_statuses)
    assert any("Finishing calibration" in status for status in pose_statuses)


def test_hand_retarget_calibration_uses_default_index_pinch_when_pose_is_degenerate() -> (
    None
):
    calibration = cli._build_hand_retarget_calibration_from_pose_samples(
        {
            "flat_closed": _calibration_features(0.1, 0.2, 0.8, thumb_abduction=0.2),
            "fist": _calibration_features(0.7, 0.2, 0.1, thumb_abduction=0.2),
            "flat_spread": _calibration_features(0.1, -0.6, 0.1, thumb_abduction=0.8),
            "scissors": _calibration_features(0.3, -0.4, 0.1, thumb_abduction=0.5),
            "index_pinch": _calibration_features(0.1, 0.2, 0.1, thumb_abduction=0.2),
        },
        handedness="Right",
    )

    assert calibration.pinch["index"] == PinchCalibration(open=0.0, closed=1.0)


def test_calibrate_hand_retarget_help_uses_rerun_not_opencv_display() -> None:
    result = runner.invoke(app, ["calibrate", "--help"])

    assert result.exit_code == 0
    assert "--interactive" in result.output
    assert "--timed" in result.output
    assert "--rerun" in result.output
    assert "--rerun-spawn" not in result.output
    assert "--display" not in result.output


def test_export_mujoco_model_help_omits_per_scene_options() -> None:
    result = runner.invoke(
        app, ["export", "mujoco", "--help"], env={"COLUMNS": "180"}
    )

    assert result.exit_code == 0
    assert "--cube-scene" not in result.output
    assert "--turn-handle-scene" not in result.output
    assert "--validate" in result.output


def test_teleop_hand_help_exposes_operation_inputs() -> None:
    result = runner.invoke(app, ["teleop", "--help"], env={"COLUMNS": "180"})

    assert result.exit_code == 0
    assert "--fake" not in result.output
    assert "--port" in result.output
    assert "--mujoco" in result.output
    assert "--mujoco-scene" in result.output
    assert "--mujoco-viewer" in result.output
    assert "--mujoco-actuator-kp" not in result.output
    assert "--mujoco-actuator-force-limit" not in result.output
    assert "--baudrate" not in result.output
    assert "--dry-run" in result.output
    assert "--enable-writes" not in result.output
    assert "Retarget" in result.output
    assert "calibration JSON" in result.output
    assert "--retargeter" not in result.output
    assert "--anchor-pose" not in result.output
    assert "--rerun" in result.output
    assert "--rerun-spawn" not in result.output
    assert "--target-fps" in result.output
    assert "--camera-width" not in result.output
    assert "--camera-height" not in result.output
    assert "--camera-fps" not in result.output
    assert "--readback-hz" not in result.output
    assert "--profile-acceleration" not in result.output
    assert "--profile-velocity" not in result.output
    assert "--landmark-filter" not in result.output
    assert "--one-euro-min-cutoff" not in result.output
    assert "--one-euro-beta" not in result.output
    assert "--one-euro-derivative-cutoff" not in result.output
    assert "--landmark-filter-reset-seconds" not in result.output
    assert "--display" not in result.output
    assert "Seconds to hold" not in result.output
    assert "Home the hand" not in result.output


def test_teleop_hand_fake_runs_webcam_to_fake_runtime_and_rerun(
    monkeypatch, tmp_path: Path
) -> None:
    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "teleop",
            "--camera",
            "2",
            "--frames",
            "2",
            "--target-fps",
            "0",
            "--rerun",
            "--rerun-save",
            str(tmp_path / "teleop.rrd"),
        ],
    )

    assert result.exit_code == 0
    assert "Teleop hand completed" in result.output
    assert "landmark filter: one-euro" in result.output
    assert "processed: 2" in result.output
    assert "frames with hands: 2" in result.output
    assert "missing-hand home events: 0" in result.output
    assert FakeWebcam.instances[0].camera == 2
    assert FakeWebcam.instances[0].latest is True
    assert FakeWebcam.instances[0].width == 640
    assert FakeWebcam.instances[0].height == 480
    assert FakeWebcam.instances[0].fps is None
    assert FakeHandTracker.instances[0].kwargs["model_path"] == Path("/tmp/model.task")
    assert FakeHandTracker.instances[0].tracks == 2
    viewer = FakeRerunViewer.instances[0]
    assert viewer.kwargs["spawn"] is True
    assert viewer.kwargs["save_path"] == tmp_path / "teleop.rrd"
    assert ("start",) in viewer.calls
    assert sum(1 for call in viewer.calls if call[0] == "camera") == 2
    assert sum(1 for call in viewer.calls if call[0] == "features") == 1
    assert sum(1 for call in viewer.calls if call[0] == "target") == 1
    assert sum(1 for call in viewer.calls if call[0] == "snapshot") == 1


def test_teleop_hand_one_euro_landmark_filter_feeds_retargeting(monkeypatch) -> None:
    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeLandmarkSmoother.instances = []
    FakeRerunViewer.instances = []
    feature_input_wrist_x = []
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "HandFrameOneEuroSmoother", FakeLandmarkSmoother)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    def fake_extract_hand_features(hand_frame, *, preferred_handedness=None):
        feature_input_wrist_x.append(hand_frame.hands[0].landmarks[0].x)
        return _feature_sample(0.4, -0.2, 0.5)

    monkeypatch.setattr(cli, "extract_hand_features", fake_extract_hand_features)

    result = runner.invoke(
        app,
        [
            "teleop",
            "--camera",
            "2",
            "--frames",
            "1",
            "--target-fps",
            "0",
            "--rerun",
        ],
    )

    assert result.exit_code == 0
    assert "landmark filter: one-euro" in result.output
    smoother = FakeLandmarkSmoother.instances[0]
    assert smoother.kwargs == {
        "min_cutoff": 1.0,
        "beta": 20.0,
        "derivative_cutoff": 1.0,
        "reset_after_seconds": 0.25,
    }
    assert smoother.calls[0].hands[0].landmarks[0].x == pytest.approx(0.5)
    assert feature_input_wrist_x == [pytest.approx(0.25)]
    viewer = FakeRerunViewer.instances[0]
    assert (
        "text",
        "landmark filter: one-euro min_cutoff=1Hz beta=20 "
        "derivative_cutoff=1Hz reset_after=0.25s",
        "INFO",
        None,
    ) in viewer.calls


def test_teleop_hand_loop_resets_landmark_filter_when_operator_track_changes(
    monkeypatch,
) -> None:
    class RecordingSmoother:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.filtered_frames = []

        def reset(self) -> None:
            self.reset_calls += 1

        def filter(self, hand_frame):
            self.filtered_frames.append(hand_frame)
            return hand_frame

    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    retargeter = SequenceRetargeter(
        [JointTarget.empty(timestamp_ns=index) for index in range(1, 8)]
    )
    clock = FakeClock([float(index) for index in range(28)])
    smoother = RecordingSmoother()
    monkeypatch.setattr(cli, "time", clock)

    result = cli._run_teleop_hand_loop(
        webcam=SequenceWebcam(frame_count=7),
        tracker=SequenceTracker(
            [
                _translated_hand_frame(timestamp_ns=1, offset_x=0.0),
                _hand_frame(timestamp_ns=2, has_hand=False),
                _hand_frame(timestamp_ns=3, has_hand=False),
                _hand_frame(timestamp_ns=4, has_hand=False),
                _translated_hand_frame(timestamp_ns=5, offset_x=0.40),
                _translated_hand_frame(timestamp_ns=6, offset_x=0.41),
                _translated_hand_frame(timestamp_ns=7, offset_x=0.42),
            ]
        ),
        runtime=runtime,
        retargeter=retargeter,
        viewer=FakeRerunViewer(),
        frames=7,
        target_fps=0.0,
        hold_missing_seconds=0.0,
        home_on_missing=False,
        hand_side="right",
        preferred_handedness="Right",
        landmark_smoother=smoother,
    )

    assert result.processed == 7
    assert smoother.reset_calls == 3
    assert [len(frame.hands) for frame in smoother.filtered_frames] == [
        1,
        0,
        0,
        0,
        0,
        0,
        1,
    ]


def test_teleop_hand_real_backend_uses_port_with_writes_enabled_by_default(
    monkeypatch,
) -> None:
    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    FakeLandmarkSmoother.instances = []
    port_calls = []
    controllers = []

    def fake_from_port(*, port: str, baudrate: int):
        port_calls.append((port, baudrate))
        controller = cli.HandController.from_fake(
            motor_ids=GHAND_RIGHT_MOTOR_IDS
        )
        controllers.append(controller)
        return controller

    monkeypatch.setattr(cli.HandController, "from_port", staticmethod(fake_from_port))
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(cli, "HandFrameOneEuroSmoother", FakeLandmarkSmoother)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "teleop",
            "--port",
            "/dev/ttyUSB0",
            "--camera",
            "2",
            "--frames",
            "1",
            "--target-fps",
            "0",
            "--rerun",
        ],
    )

    assert result.exit_code == 0
    assert "motor writes enabled" in result.output
    assert "dynamixel backend on /dev/ttyUSB0" in result.output
    assert port_calls == [("/dev/ttyUSB0", cli.DEFAULT_BAUDRATE)]
    profile_calls = [call for call in controllers[0].bus.calls if call[0] == "profile"]
    assert len(profile_calls) == 16
    assert all(
        call[2:]
        == (cli.DEFAULT_TELEOP_PROFILE_ACCELERATION, cli.DEFAULT_TELEOP_PROFILE_VELOCITY)
        for call in profile_calls
    )
    assert FakeWebcam.instances[0].latest is True
    assert FakeWebcam.instances[0].width == cli.DEFAULT_CAMERA_WIDTH
    assert FakeWebcam.instances[0].height == cli.DEFAULT_CAMERA_HEIGHT
    smoother = FakeLandmarkSmoother.instances[0]
    assert smoother.kwargs["min_cutoff"] == pytest.approx(1.0)
    viewer = FakeRerunViewer.instances[0]
    snapshot_calls = [call for call in viewer.calls if call[0] == "snapshot"]
    assert snapshot_calls == [("snapshot", 1, 16, "dynamixel:/dev/ttyUSB0", False)]


def test_teleop_hand_real_backend_supports_dry_run(monkeypatch) -> None:
    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    port_calls = []
    controllers = []

    def fake_from_port(*, port: str, baudrate: int):
        port_calls.append((port, baudrate))
        controller = cli.HandController.from_fake(
            motor_ids=GHAND_RIGHT_MOTOR_IDS
        )
        controllers.append(controller)
        return controller

    monkeypatch.setattr(cli.HandController, "from_port", staticmethod(fake_from_port))
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "teleop",
            "--port",
            "/dev/ttyUSB0",
            "--dry-run",
            "--camera",
            "2",
            "--frames",
            "1",
            "--target-fps",
            "0",
            "--rerun",
        ],
    )

    assert result.exit_code == 0
    assert "dry-run mode" in result.output
    assert "suppressed" in result.output
    assert port_calls == [("/dev/ttyUSB0", 1_000_000)]
    viewer = FakeRerunViewer.instances[0]
    snapshot_calls = [call for call in viewer.calls if call[0] == "snapshot"]
    assert snapshot_calls == [("snapshot", 1, 16, "dynamixel:/dev/ttyUSB0", True)]


def test_teleop_hand_mujoco_backend_uses_scene_and_viewer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import ghand.sim.mujoco as mujoco_sim

    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    FakeLandmarkSmoother.instances = []
    make_calls = []
    scene_xml = tmp_path / "cube_face_up_scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")

    def fake_make_mujoco_hand_runtime(**kwargs):
        make_calls.append(kwargs)
        runtime = make_fake_hand_runtime(
            side=kwargs["side"],
            dry_run=kwargs["dry_run"],
            target_hz=kwargs["target_hz"],
            target_velocity_limit=kwargs["target_velocity_limit"],
            control_mode=kwargs["control_mode"],
            default_current_limit_ma=kwargs["default_current_limit_ma"],
            max_current_limit_ma=kwargs["max_current_limit_ma"],
        )
        runtime.backend = f"mujoco:{kwargs['scene']}"
        return runtime

    monkeypatch.setattr(
        mujoco_sim, "make_mujoco_hand_runtime", fake_make_mujoco_hand_runtime
    )
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(cli, "HandFrameOneEuroSmoother", FakeLandmarkSmoother)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "teleop",
            "--mujoco",
            "--mujoco-scene",
            str(scene_xml),
            "--mujoco-viewer",
            "--camera",
            "2",
            "--frames",
            "1",
            "--rerun",
        ],
    )

    assert result.exit_code == 0
    assert "Starting MuJoCo teleop simulation" in result.output
    assert "mujoco backend with" in result.output
    assert make_calls == [
        {
            "scene": str(scene_xml),
            "side": "right",
            "dry_run": False,
            "target_hz": cli.DEFAULT_MUJOCO_CONTROL_HZ,
            "target_velocity_limit": 20.0,
            "control_mode": "position",
            "default_current_limit_ma": None,
            "max_current_limit_ma": None,
            "viewer": True,
        }
    ]
    smoother = FakeLandmarkSmoother.instances[0]
    assert smoother.kwargs["min_cutoff"] == pytest.approx(1.0)
    assert smoother.kwargs["beta"] == pytest.approx(cli.DEFAULT_TELEOP_ONE_EURO_BETA)
    viewer = FakeRerunViewer.instances[0]
    snapshot_calls = [call for call in viewer.calls if call[0] == "snapshot"]
    assert snapshot_calls == [("snapshot", 1, 16, f"mujoco:{scene_xml}", False)]


def test_teleop_hand_cli_applies_retarget_calibration(
    monkeypatch, tmp_path: Path
) -> None:
    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )
    monkeypatch.setattr(
        cli,
        "extract_hand_features",
        lambda _hand_frame, *, preferred_handedness=None: _feature_sample(
            0.4, -0.2, 0.5
        ),
    )
    calibration_path = tmp_path / "right_hand_retarget_calibration.json"
    save_hand_retarget_calibration(
        _scaled_retarget_calibration_with_pose_feature_anchors(),
        calibration_path,
    )

    result = runner.invoke(
        app,
        [
            "teleop",
            "--camera",
            "2",
            "--frames",
            "1",
            "--target-fps",
            "0",
            "--retarget-calibration",
            str(calibration_path),
            "--rerun",
        ],
    )

    assert result.exit_code == 0
    assert "retarget calibration:" in result.output
    viewer = FakeRerunViewer.instances[0]
    assert (
        "text",
        f"retarget calibration: {calibration_path}",
        "INFO",
        None,
    ) in viewer.calls
    feature_calls = [call for call in viewer.calls if call[0] == "features"]
    assert len(feature_calls) == 1
    calibrated = feature_calls[0][3]
    assert calibrated.finger_curl["index"] == pytest.approx(0.5)
    assert calibrated.finger_spread["index"] == pytest.approx(0.5)
    assert calibrated.pinch["index"] == pytest.approx(0.5)
    target_calls = [call for call in viewer.calls if call[0] == "target"]
    assert target_calls[0][2]["right_hand.index_mcp_flex.pos"] == pytest.approx(
        -0.1944444444
    )


def test_teleop_hand_anchor_pose_requires_pose_feature_anchors(tmp_path: Path) -> None:
    calibration_path = tmp_path / "right_hand_retarget_calibration.json"
    save_hand_retarget_calibration(_scaled_retarget_calibration(), calibration_path)

    result = runner.invoke(
        app,
        [
            "teleop",
            "--frames",
            "1",
            "--retarget-calibration",
            str(calibration_path),
        ],
    )

    assert result.exit_code == 1
    assert "anchor-pose retargeting requires pose_feature_anchors" in result.output


def test_diagnose_hand_retarget_cli_reports_rows_and_writes_json(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "right_hand_retarget_calibration.json"
    json_output = tmp_path / "diagnostics.json"
    save_hand_retarget_calibration(
        _retarget_calibration_with_pose_feature_anchors(),
        calibration_path,
    )

    result = runner.invoke(
        app,
        [
            "calibrate-check",
            "--retarget-calibration",
            str(calibration_path),
            "--anchor-pose",
            "flat_closed",
            "--anchor-pose",
            "open_index_flex",
            "--json-out",
            str(json_output),
        ],
    )

    assert result.exit_code == 0
    assert "Hand Retarget Diagnostics" in result.output
    assert "Retargeter" not in result.output
    assert "open_index_flex" in result.output
    assert "enabled anchor poses: flat_closed, open_index_flex" in result.output
    data = json.loads(json_output.read_text(encoding="utf-8"))
    assert {row["pose_name"] for row in data} == {"flat_closed", "open_index_flex"}
    assert data[0]["retargeter"] == "anchor-pose"


def test_diagnose_hand_retarget_help_omits_backend_option() -> None:
    result = runner.invoke(app, ["calibrate-check", "--help"])

    assert result.exit_code == 0
    assert "--retargeter" not in result.output
    assert "both" not in result.output
    assert "--anchor-pose" in result.output


def test_teleop_hand_rejects_removed_fake_option() -> None:
    result = runner.invoke(
        app, ["teleop", "--fake", "--port", "/dev/ttyUSB0", "--frames", "1"]
    )

    assert result.exit_code == 2
    assert "No such option: --fake" in result.output


def test_teleop_hand_rejects_mujoco_with_other_backends() -> None:
    result = runner.invoke(
        app,
        ["teleop", "--mujoco", "--port", "/dev/ttyUSB0", "--frames", "1"],
    )

    assert result.exit_code == 1
    assert "Choose either --port or --mujoco" in result.output


def test_teleop_hand_mujoco_rejects_current_position_mode() -> None:
    result = runner.invoke(
        app,
        [
            "teleop",
            "--mujoco",
            "--control-mode",
            "current-position",
            "--frames",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "--mujoco currently supports --control-mode position only" in result.output


def test_teleop_hand_current_position_mode_requires_default_current_limit() -> None:
    result = runner.invoke(
        app,
        [
            "teleop",
            "--control-mode",
            "current-position",
            "--frames",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "--default-current-limit-ma is required" in result.output


def test_create_lerobot_dataset_help_includes_recording_options() -> None:
    result = runner.invoke(
        app, ["record", "--help"], env={"COLUMNS": "180"}
    )

    assert result.exit_code == 0
    assert "--repo-id" in result.output
    assert "--root" in result.output
    assert "--use-videos" in result.output
    assert "--use-images" in result.output
    assert "--num-episodes" in result.output
    assert "--episode-time-s" in result.output
    assert "--reset-time-s" in result.output
    assert "--frames" not in result.output
    assert "--push-to-hub" in result.output
    assert "--control-mode" in result.output
    assert "--mujoco" in result.output
    assert "--mujoco-scene" in result.output
    assert "--mujoco-viewer" in result.output
    assert "--mujoco-actuator-kp" not in result.output
    assert "--mujoco-actuator-force-limit" not in result.output
    assert "--mujoco-auto-reset" in result.output
    assert "--landmark-filter" not in result.output
    assert "--one-euro-min-cutoff" not in result.output
    assert "--camera" not in result.output
    assert "--teleop-camera" in result.output
    assert "--teleop-camera-width" in result.output
    assert "--teleop-camera-height" in result.output
    assert "--observation-camera" in result.output
    assert "--retargeter" not in result.output
    assert "--anchor-pose" in result.output
    assert "LeRobot dataset" in result.output
    assert "contact flag" in result.output
    assert "Default Goal" in result.output
    assert "--dry-run" in result.output
    assert "--rerun-spawn" not in result.output
    assert "LeRobotDataset" in result.output


def test_create_lerobot_dataset_fake_records_frames(
    monkeypatch, tmp_path: Path
) -> None:
    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    FakeStreamingRecorder.reset()
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(cli, "LeRobotStreamingRecorder", FakeStreamingRecorder)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "record",
            "--repo-id",
            "local/ghand-record",
            "--root",
            str(tmp_path / "dataset"),
            "--task",
            "Open and close",
            "--teleop-camera",
            "2",
            "--num-episodes",
            "1",
            "--episode-time-s",
            "0.4",
            "--reset-time-s",
            "0",
            "--fps",
            "5",
            "--image-width",
            "3",
            "--image-height",
            "2",
            "--use-images",
            "--rerun",
        ],
    )

    assert result.exit_code == 0
    assert "LeRobot dataset created" in result.output
    assert "repo_id: local/ghand-record" in result.output
    assert "processed: 2" in result.output
    assert "recorded frames: 2" in result.output
    assert "saved episodes: 1" in result.output
    assert "without" in result.output
    assert "observation camera" in result.output
    assert [instance.camera for instance in FakeWebcam.instances] == [2]
    assert FakeWebcam.instances[0].latest is True
    assert FakeWebcam.instances[0].width == cli.DEFAULT_CAMERA_WIDTH
    assert FakeWebcam.instances[0].height == cli.DEFAULT_CAMERA_HEIGHT
    assert FakeHandTracker.instances[0].kwargs["model_path"] == Path("/tmp/model.task")
    create_kwargs = FakeStreamingRecorder.create_calls[0]
    assert create_kwargs["repo_id"] == "local/ghand-record"
    assert create_kwargs["root"] == tmp_path / "dataset"
    assert create_kwargs["fps"] == 5
    assert create_kwargs["use_videos"] is False
    assert create_kwargs.get("dataset_create_kwargs") is None
    features = create_kwargs["features"]
    assert "observation.images.front" not in features
    recorder = FakeStreamingRecorder.instances[0]
    assert recorder.closed
    assert len(recorder.frames) == 2
    first_frame = recorder.frames[0]
    assert first_frame["task"] == "Open and close"
    assert "observation.images.front" not in first_frame
    assert first_frame["observation.state"].shape == (16,)
    assert first_frame["action"].shape == (16,)
    assert "aux.teleop.valid" in first_frame
    viewer = FakeRerunViewer.instances[0]
    assert viewer.kwargs["spawn"] is True
    assert ("start",) in viewer.calls
    assert ("text", "record started", "INFO", None) in viewer.calls
    assert not any(call[0] == "observation_images" for call in viewer.calls)


def test_create_lerobot_dataset_records_multiple_episodes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    FakeStreamingRecorder.reset()
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(cli, "LeRobotStreamingRecorder", FakeStreamingRecorder)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "record",
            "--repo-id",
            "local/ghand-record-episodes",
            "--root",
            str(tmp_path / "dataset"),
            "--num-episodes",
            "2",
            "--episode-time-s",
            "0.2",
            "--reset-time-s",
            "0",
            "--fps",
            "5",
            "--image-width",
            "3",
            "--image-height",
            "2",
            "--use-images",
        ],
    )

    assert result.exit_code == 0
    assert "Recording episode 1/2" in result.output
    assert "Recording episode 2/2" in result.output
    assert "recorded frames: 2" in result.output
    assert "saved episodes: 2" in result.output
    recorder = FakeStreamingRecorder.instances[0]
    assert recorder.total_frames == 2
    assert len(recorder.frames) == 2
    assert recorder.saved_episodes == 2


def test_create_lerobot_dataset_can_rerecord_current_episode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    event_box = {}

    @contextmanager
    def fake_keyboard_events():
        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
        }
        event_box["events"] = events
        yield events

    class RerecordOnceTracker(FakeHandTracker):
        def track(self, frame):
            hand_frame = super().track(frame)
            if self.tracks == 1:
                event_box["events"]["rerecord_episode"] = True
                event_box["events"]["exit_early"] = True
            return hand_frame

    FakeWebcam.instances = []
    RerecordOnceTracker.instances = []
    FakeRerunViewer.instances = []
    FakeStreamingRecorder.reset()
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", RerecordOnceTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(cli, "LeRobotStreamingRecorder", FakeStreamingRecorder)
    monkeypatch.setattr(cli, "_lerobot_record_keyboard_events", fake_keyboard_events)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "record",
            "--repo-id",
            "local/ghand-rerecord",
            "--root",
            str(tmp_path / "dataset"),
            "--num-episodes",
            "1",
            "--episode-time-s",
            "0.2",
            "--reset-time-s",
            "0",
            "--fps",
            "5",
            "--image-width",
            "3",
            "--image-height",
            "2",
            "--use-images",
        ],
    )

    assert result.exit_code == 0
    assert "Re-record episode." in result.output
    assert "recorded frames: 1" in result.output
    assert "saved episodes: 1" in result.output
    recorder = FakeStreamingRecorder.instances[0]
    assert recorder.total_frames == 1
    assert len(recorder.frames) == 1
    assert recorder.saved_episodes == 1


def test_create_lerobot_dataset_can_split_teleop_and_observation_cameras(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class CameraValueWebcam:
        instances = []

        def __init__(
            self,
            camera: int,
            *,
            latest: bool = False,
            width: int | None = None,
            height: int | None = None,
        ) -> None:
            self.camera = camera
            self.latest = latest
            self.width = width
            self.height = height
            self.frame = np.full((2, 3, 3), camera, dtype=np.uint8)
            CameraValueWebcam.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def read(self):
            return self.frame

    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    FakeStreamingRecorder.reset()
    monkeypatch.setattr(cli, "Webcam", CameraValueWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(cli, "LeRobotStreamingRecorder", FakeStreamingRecorder)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "record",
            "--repo-id",
            "local/ghand-split-camera-record",
            "--root",
            str(tmp_path / "dataset"),
            "--task",
            "Split cameras",
            "--teleop-camera",
            "0",
            "--observation-camera",
            "2",
            "--num-episodes",
            "1",
            "--episode-time-s",
            "0.2",
            "--reset-time-s",
            "0",
            "--fps",
            "5",
            "--image-width",
            "3",
            "--image-height",
            "2",
            "--use-images",
        ],
    )

    assert result.exit_code == 0
    assert "teleop camera 0" in result.output
    assert "observation camera 2" in result.output
    assert [instance.camera for instance in CameraValueWebcam.instances] == [0, 2]
    assert all(instance.latest for instance in CameraValueWebcam.instances)
    assert [instance.width for instance in CameraValueWebcam.instances] == [
        cli.DEFAULT_CAMERA_WIDTH,
        3,
    ]
    assert [instance.height for instance in CameraValueWebcam.instances] == [
        cli.DEFAULT_CAMERA_HEIGHT,
        2,
    ]
    first_frame = FakeStreamingRecorder.instances[0].frames[0]
    assert first_frame["task"] == "Split cameras"
    assert first_frame["observation.images.front"].shape == (2, 3, 3)
    assert np.all(first_frame["observation.images.front"] == 2)
    viewer = FakeRerunViewer.instances[0]
    observation_calls = [
        call for call in viewer.calls if call[0] == "observation_images"
    ]
    assert len(observation_calls) == 1
    _, timestamp_ns, image_color_format, camera_frames = observation_calls[0]
    assert timestamp_ns == 1
    assert image_color_format == "bgr"
    assert np.all(camera_frames["front"] == 2)


def test_create_lerobot_dataset_mujoco_backend_uses_scene_and_viewer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import ghand.sim.mujoco as mujoco_sim

    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    FakeLandmarkSmoother.instances = []
    FakeStreamingRecorder.reset()
    make_calls = []
    scene_xml = tmp_path / "cube_face_up_scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")

    def fake_make_mujoco_hand_runtime(**kwargs):
        make_calls.append(kwargs)
        runtime = make_fake_hand_runtime(
            side=kwargs["side"],
            dry_run=kwargs["dry_run"],
            target_hz=kwargs["target_hz"],
            target_velocity_limit=kwargs["target_velocity_limit"],
            control_mode=kwargs["control_mode"],
            default_current_limit_ma=kwargs["default_current_limit_ma"],
            max_current_limit_ma=kwargs["max_current_limit_ma"],
        )
        runtime.backend = f"mujoco:{kwargs['scene']}"

        def camera_names():
            return ("overview",)

        def render_camera(camera_name: str, *, width: int, height: int):
            assert camera_name == "overview"
            return np.full((height, width, 3), 7, dtype=np.uint8)

        runtime.controller.bus.camera_names = camera_names
        runtime.controller.bus.render_camera = render_camera
        return runtime

    monkeypatch.setattr(
        mujoco_sim, "make_mujoco_hand_runtime", fake_make_mujoco_hand_runtime
    )
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(cli, "HandFrameOneEuroSmoother", FakeLandmarkSmoother)
    monkeypatch.setattr(cli, "LeRobotStreamingRecorder", FakeStreamingRecorder)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "record",
            "--mujoco",
            "--mujoco-scene",
            str(scene_xml),
            "--mujoco-viewer",
            "--repo-id",
            "local/ghand-mujoco-record",
            "--root",
            str(tmp_path / "dataset"),
            "--task",
            "Open and close in MuJoCo",
            "--teleop-camera",
            "2",
            "--num-episodes",
            "1",
            "--episode-time-s",
            "0.2",
            "--reset-time-s",
            "0",
            "--fps",
            "5",
            "--image-width",
            "3",
            "--image-height",
            "2",
            "--use-images",
            "--rerun",
        ],
    )

    assert result.exit_code == 0
    assert "Starting MuJoCo dataset capture simulation" in result.output
    assert "mujoco" in result.output
    assert "backend with" in result.output
    assert "dataset schema: position" in result.output
    assert "MuJoCo camera(s) overview" in result.output
    assert make_calls == [
        {
            "scene": str(scene_xml),
            "side": "right",
            "dry_run": False,
            "target_hz": 5.0,
            "target_velocity_limit": 20.0,
            "control_mode": "position",
            "default_current_limit_ma": None,
            "max_current_limit_ma": None,
            "viewer": True,
        }
    ]
    assert [instance.camera for instance in FakeWebcam.instances] == [2]
    features = FakeStreamingRecorder.create_calls[0]["features"]
    assert features["observation.images.overview"]["shape"] == (2, 3, 3)
    assert features["observation.images.overview"]["dtype"] == "image"
    assert features["observation.state"]["shape"] == (16,)
    first_frame = FakeStreamingRecorder.instances[0].frames[0]
    assert first_frame["task"] == "Open and close in MuJoCo"
    assert first_frame["observation.images.overview"].shape == (2, 3, 3)
    assert np.all(first_frame["observation.images.overview"] == 7)
    assert first_frame["observation.state"].shape == (16,)
    assert first_frame["action"].shape == (16,)
    smoother = FakeLandmarkSmoother.instances[0]
    assert smoother.kwargs["min_cutoff"] == pytest.approx(1.0)
    assert smoother.kwargs["beta"] == pytest.approx(cli.DEFAULT_TELEOP_ONE_EURO_BETA)
    assert len(smoother.calls) == 1
    assert "landmark filter: one-euro" in result.output
    viewer = FakeRerunViewer.instances[0]
    observation_calls = [
        call for call in viewer.calls if call[0] == "observation_images"
    ]
    assert len(observation_calls) == 1
    _, timestamp_ns, image_color_format, camera_frames = observation_calls[0]
    assert timestamp_ns == 1
    assert image_color_format == "rgb"
    assert np.all(camera_frames["overview"] == 7)
    snapshot_calls = [call for call in viewer.calls if call[0] == "snapshot"]
    assert snapshot_calls == [("snapshot", 1, 16, f"mujoco:{scene_xml}", False)]


def test_create_lerobot_dataset_mujoco_auto_resets_between_episodes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import ghand.sim.mujoco as mujoco_sim

    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    FakeStreamingRecorder.reset()
    reset_snapshots = []
    scene_xml = tmp_path / "cube_face_up_scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")

    def fake_make_mujoco_hand_runtime(**kwargs):
        runtime = make_fake_hand_runtime(
            side=kwargs["side"],
            dry_run=kwargs["dry_run"],
            target_hz=kwargs["target_hz"],
            target_velocity_limit=kwargs["target_velocity_limit"],
            control_mode=kwargs["control_mode"],
            default_current_limit_ma=kwargs["default_current_limit_ma"],
            max_current_limit_ma=kwargs["max_current_limit_ma"],
        )
        runtime.backend = f"mujoco:{kwargs['scene']}"

        def reset_scene():
            reset_snapshots.append(
                tuple(
                    (joint.target, joint.commanded)
                    for joint in runtime.snapshot().joints
                )
            )

        runtime.controller.bus.reset_scene = reset_scene
        return runtime

    monkeypatch.setattr(
        mujoco_sim, "make_mujoco_hand_runtime", fake_make_mujoco_hand_runtime
    )
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(cli, "LeRobotStreamingRecorder", FakeStreamingRecorder)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    @contextmanager
    def fake_keyboard_events():
        yield {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
        }

    monkeypatch.setattr(cli, "_lerobot_record_keyboard_events", fake_keyboard_events)

    result = runner.invoke(
        app,
        [
            "record",
            "--mujoco",
            "--mujoco-scene",
            str(scene_xml),
            "--repo-id",
            "local/ghand-mujoco-reset",
            "--root",
            str(tmp_path / "dataset"),
            "--num-episodes",
            "2",
            "--episode-time-s",
            "0.2",
            "--reset-time-s",
            "0",
            "--fps",
            "5",
            "--image-width",
            "3",
            "--image-height",
            "2",
            "--use-images",
        ],
    )

    assert result.exit_code == 0
    assert "MuJoCo scene reset." in result.output
    assert "saved episodes: 2" in result.output
    assert len(reset_snapshots) == 1
    assert all(
        target == pytest.approx(0.0) for target, _commanded in reset_snapshots[0]
    )
    assert all(
        commanded == pytest.approx(0.0) for _target, commanded in reset_snapshots[0]
    )


def test_create_lerobot_dataset_mujoco_rerecord_resets_before_retry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import ghand.sim.mujoco as mujoco_sim

    event_box = {}

    @contextmanager
    def fake_keyboard_events():
        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
        }
        event_box["events"] = events
        yield events

    class RerecordOnceTracker(FakeHandTracker):
        def track(self, frame):
            hand_frame = super().track(frame)
            if self.tracks == 1:
                event_box["events"]["rerecord_episode"] = True
                event_box["events"]["exit_early"] = True
            return hand_frame

    FakeWebcam.instances = []
    RerecordOnceTracker.instances = []
    FakeRerunViewer.instances = []
    FakeStreamingRecorder.reset()
    reset_calls = []
    scene_xml = tmp_path / "cube_face_up_scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")

    def fake_make_mujoco_hand_runtime(**kwargs):
        runtime = make_fake_hand_runtime(
            side=kwargs["side"],
            dry_run=kwargs["dry_run"],
            target_hz=kwargs["target_hz"],
            target_velocity_limit=kwargs["target_velocity_limit"],
            control_mode=kwargs["control_mode"],
            default_current_limit_ma=kwargs["default_current_limit_ma"],
            max_current_limit_ma=kwargs["max_current_limit_ma"],
        )
        runtime.backend = f"mujoco:{kwargs['scene']}"
        runtime.controller.bus.reset_scene = lambda: reset_calls.append("reset")
        return runtime

    monkeypatch.setattr(
        mujoco_sim, "make_mujoco_hand_runtime", fake_make_mujoco_hand_runtime
    )
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", RerecordOnceTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(cli, "LeRobotStreamingRecorder", FakeStreamingRecorder)
    monkeypatch.setattr(cli, "_lerobot_record_keyboard_events", fake_keyboard_events)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "record",
            "--mujoco",
            "--mujoco-scene",
            str(scene_xml),
            "--repo-id",
            "local/ghand-mujoco-rerecord-reset",
            "--root",
            str(tmp_path / "dataset"),
            "--num-episodes",
            "1",
            "--episode-time-s",
            "0.2",
            "--reset-time-s",
            "0",
            "--fps",
            "5",
            "--image-width",
            "3",
            "--image-height",
            "2",
            "--use-images",
        ],
    )

    assert result.exit_code == 0
    assert "Re-record episode." in result.output
    assert reset_calls == ["reset"]
    assert "saved episodes: 1" in result.output


def test_create_lerobot_dataset_can_record_position_current_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    FakeStreamingRecorder.reset()
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(cli, "LeRobotStreamingRecorder", FakeStreamingRecorder)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "record",
            "--repo-id",
            "local/ghand-current-record",
            "--root",
            str(tmp_path / "dataset"),
            "--task",
            "Grasp object",
            "--teleop-camera",
            "2",
            "--num-episodes",
            "1",
            "--episode-time-s",
            "0.2",
            "--reset-time-s",
            "0",
            "--fps",
            "5",
            "--image-width",
            "3",
            "--image-height",
            "2",
            "--use-images",
            "--dataset-schema",
            "position-current",
            "--default-current-limit-ma",
            "80",
        ],
    )

    assert result.exit_code == 0
    assert "dataset schema: position-current" in result.output
    features = FakeStreamingRecorder.create_calls[0]["features"]
    state_names = features["observation.state"]["names"]
    action_names = features["action"]["names"]
    assert features["observation.state"]["shape"] == (32,)
    assert features["action"]["shape"] == (32,)
    assert "right_hand.index_mcp_flex.present_current_ma" in state_names
    assert "right_hand.index_mcp_flex.current_limit_ma" in action_names
    first_frame = FakeStreamingRecorder.instances[0].frames[0]
    assert first_frame["observation.state"].shape == (32,)
    assert first_frame["action"].shape == (32,)
    current_index = action_names.index("right_hand.index_mcp_flex.current_limit_ma")
    present_current_index = state_names.index(
        "right_hand.index_mcp_flex.present_current_ma"
    )
    assert first_frame["action"][current_index] == pytest.approx(80.0)
    assert first_frame["observation.state"][present_current_index] == pytest.approx(0.0)


def test_create_lerobot_dataset_can_record_current_diagnostics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    FakeStreamingRecorder.reset()
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(cli, "LeRobotStreamingRecorder", FakeStreamingRecorder)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "record",
            "--repo-id",
            "local/ghand-current-diagnostics",
            "--root",
            str(tmp_path / "dataset"),
            "--num-episodes",
            "1",
            "--episode-time-s",
            "0.2",
            "--reset-time-s",
            "0",
            "--fps",
            "5",
            "--image-width",
            "3",
            "--image-height",
            "2",
            "--use-images",
            "--dataset-schema",
            "position-current",
            "--default-current-limit-ma",
            "80",
            "--include-current-diagnostics",
            "--contact-current-threshold-ma",
            "40",
        ],
    )

    assert result.exit_code == 0
    features = FakeStreamingRecorder.create_calls[0]["features"]
    state_names = features["observation.state"]["names"]
    assert features["observation.state"]["shape"] == (64,)
    assert "right_hand.index_mcp_flex.current_residual_ma" in state_names
    assert "right_hand.index_mcp_flex.contact" in state_names
    first_frame = FakeStreamingRecorder.instances[0].frames[0]
    residual_index = state_names.index("right_hand.index_mcp_flex.current_residual_ma")
    contact_index = state_names.index("right_hand.index_mcp_flex.contact")
    assert first_frame["observation.state"].shape == (64,)
    assert first_frame["observation.state"][residual_index] == pytest.approx(0.0)
    assert first_frame["observation.state"][contact_index] == pytest.approx(0.0)


def test_create_lerobot_dataset_real_backend_supports_dry_run(
    monkeypatch, tmp_path: Path
) -> None:
    FakeWebcam.instances = []
    FakeHandTracker.instances = []
    FakeRerunViewer.instances = []
    FakeStreamingRecorder.reset()
    port_calls = []
    controllers = []

    def fake_from_port(*, port: str, baudrate: int):
        port_calls.append((port, baudrate))
        controller = cli.HandController.from_fake(
            motor_ids=GHAND_RIGHT_MOTOR_IDS
        )
        controllers.append(controller)
        return controller

    monkeypatch.setattr(cli.HandController, "from_port", staticmethod(fake_from_port))
    monkeypatch.setattr(cli, "Webcam", FakeWebcam)
    monkeypatch.setattr(cli, "MediaPipeHandTracker", FakeHandTracker)
    monkeypatch.setattr(cli, "RerunViewer", FakeRerunViewer)
    monkeypatch.setattr(cli, "LeRobotStreamingRecorder", FakeStreamingRecorder)
    monkeypatch.setattr(
        cli, "resolve_hand_landmarker_model", lambda model: Path("/tmp/model.task")
    )
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _seconds: None)
    )

    result = runner.invoke(
        app,
        [
            "record",
            "--port",
            "/dev/ttyUSB0",
            "--dry-run",
            "--repo-id",
            "local/ghand-real-record",
            "--root",
            str(tmp_path / "dataset"),
            "--num-episodes",
            "1",
            "--episode-time-s",
            "0.2",
            "--reset-time-s",
            "0",
            "--fps",
            "5",
            "--image-width",
            "3",
            "--image-height",
            "2",
            "--use-images",
        ],
    )

    assert result.exit_code == 0
    assert "dry-run mode" in result.output
    assert "suppressed" in result.output
    assert port_calls == [("/dev/ttyUSB0", 1_000_000)]
    assert FakeStreamingRecorder.instances[0].total_frames == 1


def test_create_lerobot_dataset_rejects_removed_fake_option() -> None:
    result = runner.invoke(
        app,
        [
            "record",
            "--fake",
            "--port",
            "/dev/ttyUSB0",
            "--repo-id",
            "local/bad",
        ],
    )

    assert result.exit_code == 2
    assert "No such option: --fake" in result.output


def test_create_lerobot_dataset_rejects_mujoco_with_other_backends() -> None:
    result = runner.invoke(
        app,
        [
            "record",
            "--mujoco",
            "--port",
            "/dev/ttyUSB0",
            "--repo-id",
            "local/bad",
        ],
    )

    assert result.exit_code == 1
    assert "Choose either --port or --mujoco" in result.output


def test_create_lerobot_dataset_mujoco_rejects_current_position_mode() -> None:
    result = runner.invoke(
        app,
        [
            "record",
            "--mujoco",
            "--repo-id",
            "local/bad",
            "--control-mode",
            "current-position",
            "--default-current-limit-ma",
            "80",
        ],
    )

    assert result.exit_code == 1
    assert "--mujoco currently supports --control-mode position only" in result.output


def test_create_lerobot_dataset_mujoco_rejects_position_current_schema() -> None:
    result = runner.invoke(
        app,
        [
            "record",
            "--mujoco",
            "--repo-id",
            "local/bad",
            "--dataset-schema",
            "position-current",
            "--default-current-limit-ma",
            "80",
        ],
    )

    assert result.exit_code == 1
    assert "--mujoco currently supports --dataset-schema position only" in result.output


def test_create_lerobot_dataset_position_current_schema_requires_default_current_limit() -> (
    None
):
    result = runner.invoke(
        app,
        [
            "record",
            "--repo-id",
            "local/bad",
            "--dataset-schema",
            "position-current",
        ],
    )

    assert result.exit_code == 1
    assert "--default-current-limit-ma is required" in result.output


def test_create_lerobot_dataset_current_diagnostics_requires_position_current_schema() -> (
    None
):
    result = runner.invoke(
        app,
        [
            "record",
            "--repo-id",
            "local/bad",
            "--include-current-diagnostics",
        ],
    )

    assert result.exit_code == 1
    assert "--include-current-diagnostics requires" in result.output


def test_teleop_hand_rerun_ui_logging_is_decimated_but_logs_hand_count_changes() -> (
    None
):
    assert cli._should_log_teleop_rerun_ui(
        frame_index=1,
        hand_count=1,
        previous_hand_count=None,
    )
    assert not cli._should_log_teleop_rerun_ui(
        frame_index=2,
        hand_count=1,
        previous_hand_count=1,
    )
    assert not cli._should_log_teleop_rerun_ui(
        frame_index=3,
        hand_count=1,
        previous_hand_count=1,
    )
    assert cli._should_log_teleop_rerun_ui(
        frame_index=10,
        hand_count=1,
        previous_hand_count=1,
    )
    assert cli._should_log_teleop_rerun_ui(
        frame_index=4,
        hand_count=0,
        previous_hand_count=1,
    )


def test_operator_hand_tracking_log_key_ignores_match_distance_noise() -> None:
    first = SimpleNamespace(
        state=SimpleNamespace(value="locked"),
        track_id=1,
        candidates=1,
        selected_index=0,
        reason="matched distance=0.001",
    )
    second = SimpleNamespace(
        state=SimpleNamespace(value="locked"),
        track_id=1,
        candidates=1,
        selected_index=0,
        reason="matched distance=0.042",
    )

    assert cli._operator_hand_tracking_log_key(
        first
    ) == cli._operator_hand_tracking_log_key(second)


def test_teleop_hand_console_summary_is_sparse_but_logs_hand_count_changes() -> None:
    assert cli._should_print_teleop_summary(
        frame_index=1,
        hand_count=1,
        previous_hand_count=None,
    )
    assert not cli._should_print_teleop_summary(
        frame_index=2,
        hand_count=1,
        previous_hand_count=1,
    )
    assert cli._should_print_teleop_summary(
        frame_index=30,
        hand_count=1,
        previous_hand_count=1,
    )
    assert cli._should_print_teleop_summary(
        frame_index=31,
        hand_count=0,
        previous_hand_count=1,
    )


def test_teleop_hand_loop_homes_after_missing_hand_timeout(monkeypatch) -> None:
    hand_frames = [
        _hand_frame(timestamp_ns=1, has_hand=True),
        _hand_frame(timestamp_ns=2, has_hand=False),
        _hand_frame(timestamp_ns=3, has_hand=False),
    ]
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    retargeter = SequenceRetargeter(
        [
            JointTarget(
                timestamp_ns=1,
                joints={"right_hand.index_mcp_flex.pos": 0.5},
            ),
            JointTarget.empty(timestamp_ns=2),
            JointTarget.empty(timestamp_ns=3),
        ]
    )
    viewer = FakeRerunViewer()
    clock = FakeClock(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.2,
            0.2,
            0.2,
            0.6,
            0.6,
            0.6,
            0.6,
        ]
    )
    monkeypatch.setattr(cli, "time", clock)

    result = cli._run_teleop_hand_loop(
        webcam=SequenceWebcam(frame_count=3),
        tracker=SequenceTracker(hand_frames),
        runtime=runtime,
        retargeter=retargeter,
        viewer=viewer,
        frames=3,
        target_fps=0.0,
        hold_missing_seconds=0.5,
        home_on_missing=True,
        hand_side="right",
        preferred_handedness="Right",
    )

    snapshot = runtime.snapshot()
    assert result.processed == 3
    assert result.frames_with_hands == 1
    assert result.missing_home_events == 1
    assert _snapshot_joint(snapshot, "right_hand.index_mcp_flex.pos").target == 0.0
    assert ("text", "hand target missing; homing", "WARN", 3) in viewer.calls


def test_teleop_hand_loop_records_observation_before_applying_action(
    monkeypatch,
) -> None:
    from ghand.data import make_hand_lerobot_features

    FakeStreamingRecorder.reset()
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    retargeter = SequenceRetargeter(
        [
            JointTarget(
                timestamp_ns=1,
                joints={"right_hand.index_mcp_flex.pos": 0.5},
            ),
            JointTarget(
                timestamp_ns=2,
                joints={"right_hand.index_mcp_flex.pos": 0.5},
            ),
        ]
    )
    clock = FakeClock([0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1])
    monkeypatch.setattr(cli, "time", clock)
    features = make_hand_lerobot_features(sides=("right",), cameras=())
    recorder = FakeStreamingRecorder()

    cli._run_teleop_hand_loop(
        webcam=SequenceWebcam(frame_count=2),
        tracker=SequenceTracker(
            [
                _hand_frame(timestamp_ns=1, has_hand=True),
                _hand_frame(timestamp_ns=2, has_hand=True),
            ]
        ),
        runtime=runtime,
        retargeter=retargeter,
        viewer=FakeRerunViewer(),
        frames=2,
        target_fps=0.0,
        hold_missing_seconds=0.0,
        home_on_missing=False,
        hand_side="right",
        preferred_handedness="Right",
        dataset_recorder=recorder,
        dataset_features=features,
        dataset_task="Alignment test",
    )

    calibration = next(
        calibration
        for calibration in runtime.calibrations
        if calibration.joint_name == "right_hand.index_mcp_flex.pos"
    )
    commanded = calibration.normalized_to_angle_rad(0.5)
    tick_tolerance = 1 / calibration.ticks_per_radian
    state_names = features["observation.state"]["names"]
    action_names = features["action"]["names"]
    joint_state_index = state_names.index("right_hand.index_mcp_flex.pos")
    joint_action_index = action_names.index("right_hand.index_mcp_flex.pos")
    first_frame, second_frame = recorder.frames

    # Frame t stores the state BEFORE action a_t was applied: (o_t, a_t).
    assert first_frame["observation.state"][joint_state_index] == pytest.approx(0.0)
    assert first_frame["action"][joint_action_index] == pytest.approx(commanded)
    # The next frame observes the result of the previous action.
    assert second_frame["observation.state"][joint_state_index] == pytest.approx(
        commanded, abs=tick_tolerance
    )
    assert second_frame["action"][joint_action_index] == pytest.approx(commanded)


def test_teleop_hand_loop_can_hold_missing_hand_indefinitely(monkeypatch) -> None:
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    retargeter = SequenceRetargeter(
        [
            JointTarget(
                timestamp_ns=1,
                joints={"right_hand.index_mcp_flex.pos": 0.5},
            ),
            JointTarget.empty(timestamp_ns=2),
        ]
    )
    clock = FakeClock([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    monkeypatch.setattr(cli, "time", clock)

    result = cli._run_teleop_hand_loop(
        webcam=SequenceWebcam(frame_count=2),
        tracker=SequenceTracker(
            [
                _hand_frame(timestamp_ns=1, has_hand=True),
                _hand_frame(timestamp_ns=2, has_hand=False),
            ]
        ),
        runtime=runtime,
        retargeter=retargeter,
        viewer=FakeRerunViewer(),
        frames=2,
        target_fps=0.0,
        hold_missing_seconds=0.5,
        home_on_missing=False,
        hand_side="right",
        preferred_handedness="Right",
    )

    assert result.missing_home_events == 0
    calibration = next(
        calibration
        for calibration in runtime.calibrations
        if calibration.joint_name == "right_hand.index_mcp_flex.pos"
    )
    assert _snapshot_joint(
        runtime.snapshot(), "right_hand.index_mcp_flex.pos"
    ).target == pytest.approx(calibration.normalized_to_angle_rad(0.5))


def test_train_cli_builds_recommended_lerobot_train_command(monkeypatch) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        cli, "_run_lerobot_subprocess", lambda args: captured.append(args) or 0
    )

    result = runner.invoke(app, ["train", "--repo-id", "local/ghand-mujoco"])

    assert result.exit_code == 0
    (args,) = captured
    assert args[1:3] == ["-m", cli.LEROBOT_TRAIN_MODULE]
    assert "--dataset.repo_id=local/ghand-mujoco" in args
    assert not any(arg.startswith("--dataset.root=") for arg in args)
    assert "--policy.type=act" in args
    assert "--policy.chunk_size=20" in args
    assert "--policy.n_action_steps=10" in args
    assert f"--steps={cli.DEFAULT_TRAIN_STEPS}" in args
    assert "--batch_size=8" in args
    assert "--output_dir=outputs/train/ghand-mujoco-act" in args
    assert "--job_name=ghand-mujoco-act" in args
    assert "--policy.push_to_hub=false" in args
    assert "--wandb.enable=false" in args


def test_train_cli_forwards_overrides_and_drops_conflicting_defaults(
    monkeypatch,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        cli, "_run_lerobot_subprocess", lambda args: captured.append(args) or 0
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--repo-id",
            "local/ghand-mujoco",
            "--root",
            "data/custom-root",
            "--batch_size=16",
            "--policy.optimizer_lr=5e-5",
        ],
    )

    assert result.exit_code == 0
    (args,) = captured
    assert "--dataset.root=data/custom-root" in args
    assert "--batch_size=16" in args
    assert "--batch_size=8" not in args
    assert "--policy.optimizer_lr=5e-5" in args


def test_train_cli_rejects_action_steps_above_chunk_size(monkeypatch) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        cli, "_run_lerobot_subprocess", lambda args: captured.append(args) or 0
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--repo-id",
            "local/ghand-mujoco",
            "--chunk-size",
            "10",
            "--n-action-steps",
            "20",
        ],
    )

    assert result.exit_code == 1
    assert captured == []


def _write_policy_config(pretrained_dir: Path, *, image_shape=(3, 480, 640)) -> None:
    pretrained_dir.mkdir(parents=True)
    config = {
        "type": "act",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [16]},
            "observation.images.overview": {
                "type": "VISUAL",
                "shape": list(image_shape),
            },
        },
    }
    (pretrained_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_run_policy_cli_derives_rollout_settings_from_policy_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        cli, "_run_lerobot_subprocess", lambda args: captured.append(args) or 0
    )
    train_root = tmp_path / "train-output"
    _write_policy_config(
        train_root / "checkpoints" / "last" / "pretrained_model",
        image_shape=(3, 240, 320),
    )

    result = runner.invoke(
        app,
        [
            "run-policy",
            "--policy-path",
            str(train_root),
            "--repo-id",
            "local/eval-test",
            "--mujoco-scene",
            "scenes/turn_handle.xml",
        ],
    )

    assert result.exit_code == 0
    (args,) = captured
    assert args[1:3] == ["-m", cli.LEROBOT_RECORD_MODULE]
    expected_policy = train_root / "checkpoints" / "last" / "pretrained_model"
    assert f"--policy.path={expected_policy}" in args
    assert "--robot.type=ghand" in args
    assert "--robot.backend=mujoco" in args
    assert not any(arg.startswith("--robot.discover_packages_path") for arg in args)
    assert "--robot.mujoco_scene=scenes/turn_handle.xml" in args
    assert "--robot.mujoco_camera_names=overview" in args
    assert "--robot.camera_width=320" in args
    assert "--robot.camera_height=240" in args
    assert "--robot.target_fps=10" in args
    assert "--robot.target_velocity_limit=20.0" in args
    assert "--robot.mujoco_actuator_kp=20.0" in args
    assert "--robot.mujoco_actuator_force_limit=1.0" in args
    assert "--dataset.fps=10" in args
    assert "--dataset.repo_id=local/eval-test" in args
    assert "--dataset.root=data/eval-test" in args
    assert "--dataset.push_to_hub=false" in args


def test_run_policy_cli_builds_real_backend_rollout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        cli, "_run_lerobot_subprocess", lambda args: captured.append(args) or 0
    )
    _write_policy_config(tmp_path / "pretrained_model", image_shape=(3, 240, 320))

    result = runner.invoke(
        app,
        [
            "run-policy",
            "--policy-path",
            str(tmp_path / "pretrained_model"),
            "--backend",
            "real",
            "--port",
            "/dev/ttyUSB0",
            "--observation-camera",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    (args,) = captured
    assert "--robot.backend=real" in args
    assert "--robot.port=/dev/ttyUSB0" in args
    assert "--robot.observation_camera=2" in args
    assert "--robot.observation_camera_name=overview" in args
    assert "--robot.camera_width=320" in args
    assert "--robot.camera_height=240" in args
    assert not any(arg.startswith("--robot.mujoco_") for arg in args)


def test_run_policy_cli_real_backend_requires_port(monkeypatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        cli, "_run_lerobot_subprocess", lambda args: captured.append(args) or 0
    )
    _write_policy_config(tmp_path / "pretrained_model")

    result = runner.invoke(
        app,
        [
            "run-policy",
            "--policy-path",
            str(tmp_path / "pretrained_model"),
            "--backend",
            "real",
            "--observation-camera",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert captured == []


def test_run_policy_cli_real_backend_requires_observation_camera_for_image_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        cli, "_run_lerobot_subprocess", lambda args: captured.append(args) or 0
    )
    _write_policy_config(tmp_path / "pretrained_model")

    result = runner.invoke(
        app,
        [
            "run-policy",
            "--policy-path",
            str(tmp_path / "pretrained_model"),
            "--backend",
            "real",
            "--port",
            "/dev/ttyUSB0",
        ],
    )

    assert result.exit_code == 1
    assert captured == []


def test_run_policy_cli_rejects_observation_camera_with_mujoco(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        cli, "_run_lerobot_subprocess", lambda args: captured.append(args) or 0
    )
    _write_policy_config(tmp_path / "pretrained_model")

    result = runner.invoke(
        app,
        [
            "run-policy",
            "--policy-path",
            str(tmp_path / "pretrained_model"),
            "--observation-camera",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert captured == []


def test_run_policy_cli_fails_when_policy_config_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        cli, "_run_lerobot_subprocess", lambda args: captured.append(args) or 0
    )

    result = runner.invoke(
        app,
        ["run-policy", "--policy-path", str(tmp_path / "missing")],
    )

    assert result.exit_code == 1
    assert captured == []


def test_run_policy_cli_propagates_subprocess_exit_code(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "_run_lerobot_subprocess", lambda args: 3)
    _write_policy_config(tmp_path / "pretrained_model")

    result = runner.invoke(
        app,
        ["run-policy", "--policy-path", str(tmp_path / "pretrained_model")],
    )

    assert result.exit_code == 3


def test_motor_test_cli_encoder_only_reads_without_prompt_or_motion(
    monkeypatch,
) -> None:
    FakeMotorController.instances = []
    monkeypatch.setattr(cli, "HandController", FakeMotorController)

    result = runner.invoke(
        app,
        [
            "motor", "test",
            "--port",
            "/dev/ttyUSB0",
            "--id",
            "3",
            "--encoder-only",
        ],
    )

    assert result.exit_code == 0
    assert "Motor encoder read completed" in result.output
    assert "present position" in result.output
    controller = FakeMotorController.instances[0]
    assert controller.encoder_calls == [3]
    assert controller.test_calls == []


def test_motor_test_cli_calls_controller_without_confirmation(monkeypatch) -> None:
    FakeMotorController.instances = []
    monkeypatch.setattr(cli, "HandController", FakeMotorController)

    result = runner.invoke(
        app,
        [
            "motor", "test",
            "--port",
            "/dev/ttyUSB0",
            "--id",
            "3",
            "--baudrate",
            "4000000",
            "--delta-ticks",
            "64",
            "--keep-torque",
        ],
    )

    assert result.exit_code == 0
    assert "Motor test completed" in result.output
    assert "torque after" in result.output
    controller = FakeMotorController.instances[0]
    assert controller.port == "/dev/ttyUSB0"
    assert controller.baudrate == 4_000_000
    assert controller.test_calls == [
        (
            3,
            {
                "delta_ticks": 64,
                "keep_torque": True,
            },
        )
    ]


def test_motor_test_cli_reports_bus_error(monkeypatch) -> None:
    class FailingMotorController(FakeMotorController):
        def test_motor(self, dxl_id: int, **kwargs) -> MotorTestResult:
            raise DynamixelBusError("no response")

    FailingMotorController.instances = []
    monkeypatch.setattr(cli, "HandController", FailingMotorController)

    result = runner.invoke(
        app,
        [
            "motor", "test",
            "--port",
            "/dev/ttyUSB0",
            "--id",
            "3",
        ],
    )

    assert result.exit_code == 1
    assert "Motor test failed" in result.output
    assert "no response" in result.output


def test_motor_scan_cli_reads_real_motor_map_without_motion(monkeypatch) -> None:
    FakeMotorController.instances = []
    monkeypatch.setattr(cli, "HandController", FakeMotorController)

    result = runner.invoke(
        app,
        [
            "motor", "scan",
            "--port",
            "/dev/cu.usbserial-FTBEQOHA",
            "--baudrate",
            "4000000",
        ],
    )

    assert result.exit_code == 0
    assert "Motor scan completed: 16/16 OK" in result.output
    controller = FakeMotorController.instances[0]
    assert controller.port == "/dev/cu.usbserial-FTBEQOHA"
    assert controller.baudrate == 4_000_000
    assert controller.encoder_calls == list(GHAND_RIGHT_MOTOR_IDS)
    assert controller.test_calls == []


def test_motor_scan_cli_can_scan_zero_to_fifteen(monkeypatch) -> None:
    FakeMotorController.instances = []
    monkeypatch.setattr(cli, "HandController", FakeMotorController)

    result = runner.invoke(
        app,
        [
            "motor", "scan",
            "--port",
            "/dev/cu.usbserial-FTBEQOHA",
            "--all-ids",
        ],
    )

    assert result.exit_code == 0
    assert FakeMotorController.instances[0].encoder_calls == list(range(16))


def test_motor_scan_cli_reports_failed_ids(monkeypatch) -> None:
    class PartiallyFailingMotorController(FakeMotorController):
        def __init__(self, *, port: str, baudrate: int) -> None:
            super().__init__(port=port, baudrate=baudrate)
            self.fail_encoder_ids = {9}

    PartiallyFailingMotorController.instances = []
    monkeypatch.setattr(cli, "HandController", PartiallyFailingMotorController)

    result = runner.invoke(
        app,
        [
            "motor", "scan",
            "--port",
            "/dev/cu.usbserial-FTBEQOHA",
        ],
    )

    assert result.exit_code == 1
    assert "Motor scan completed: 15/16 OK" in result.output
    assert "Dynamixel ID 9 ping failed" in result.output


def test_operator_ui_cli_uses_fake_backend_by_default(monkeypatch) -> None:
    calls = []

    def fake_launch_operator_ui(runtime, *, host: str, port: int, **kwargs) -> None:
        calls.append((runtime, host, port, kwargs))

    monkeypatch.setattr(cli, "_launch_operator_ui", fake_launch_operator_ui)

    result = runner.invoke(app, ["ui", "--ui-port", "9001"])

    assert result.exit_code == 0
    runtime, host, port, kwargs = calls[0]
    assert host == "127.0.0.1"
    assert port == 9001
    assert kwargs["enable_3d"] is True
    assert kwargs["model_root"] is None
    assert kwargs["urdf_path"] is None
    assert runtime.backend == "fake"
    assert runtime.dry_run is False
    assert runtime.telemetry_hz == 1.0
    assert runtime.control_mode == "position"
    assert isinstance(runtime.controller.bus, FakeBus)


def test_operator_ui_cli_can_use_current_position_mode(monkeypatch) -> None:
    calls = []

    def fake_launch_operator_ui(runtime, *, host: str, port: int, **kwargs) -> None:
        calls.append((runtime, host, port, kwargs))

    monkeypatch.setattr(cli, "_launch_operator_ui", fake_launch_operator_ui)

    result = runner.invoke(
        app,
        [
            "ui",
            "--control-mode",
            "current-position",
            "--default-current-limit-ma",
            "90",
            "--max-current-limit-ma",
            "80",
        ],
    )

    assert result.exit_code == 0
    runtime, _host, _port, _kwargs = calls[0]
    assert runtime.control_mode == "current_position"
    assert (
        _snapshot_joint(
            runtime.snapshot(), "right_hand.index_mcp_flex.pos"
        ).current_limit_ma
        == 80.0
    )


def test_operator_ui_current_position_mode_requires_default_current_limit(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        cli, "_launch_operator_ui", lambda *args, **kwargs: calls.append(args)
    )

    result = runner.invoke(app, ["ui", "--control-mode", "current-position"])

    assert result.exit_code == 1
    assert "--default-current-limit-ma is required" in result.output
    assert calls == []


def test_operator_ui_cli_uses_real_backend_with_writes_enabled_by_default(
    monkeypatch,
) -> None:
    calls = []

    def fake_launch_operator_ui(runtime, *, host: str, port: int, **kwargs) -> None:
        calls.append((runtime, host, port, kwargs))

    monkeypatch.setattr(cli, "_launch_operator_ui", fake_launch_operator_ui)

    result = runner.invoke(
        app,
        [
            "ui",
            "--port",
            "/dev/ttyUSB0",
            "--host",
            "0.0.0.0",
            "--ui-port",
            "9002",
        ],
    )

    assert result.exit_code == 0
    assert "motor writes enabled" in result.output
    runtime, host, port, kwargs = calls[0]
    assert host == "0.0.0.0"
    assert port == 9002
    assert kwargs["enable_3d"] is True
    assert runtime.backend == "dynamixel:/dev/ttyUSB0"
    assert runtime.dry_run is False
    assert runtime.dxl_ids == GHAND_RIGHT_MOTOR_IDS
    assert (
        _snapshot_joint(runtime.snapshot(), "right_hand.index_mcp_flex.pos").dxl_id == 1
    )
    assert (
        _snapshot_joint(runtime.snapshot(), "right_hand.index_mcp_flex.pos").pose_dxl_id
        == 9
    )
    assert (
        _calibration(runtime, "right_hand.index_mcp_flex.pos").direction
        == DEFAULT_JOINT_DIRECTIONS[9]
    )
    assert runtime.controller.bus.port == "/dev/ttyUSB0"


def test_operator_ui_cli_real_backend_supports_dry_run(monkeypatch) -> None:
    calls = []

    def fake_launch_operator_ui(runtime, *, host: str, port: int, **kwargs) -> None:
        calls.append((runtime, host, port, kwargs))

    monkeypatch.setattr(cli, "_launch_operator_ui", fake_launch_operator_ui)

    result = runner.invoke(
        app,
        [
            "ui",
            "--port",
            "/dev/ttyUSB0",
            "--dry-run",
            "--ui-port",
            "9002",
        ],
    )

    assert result.exit_code == 0
    assert "dry-run mode" in result.output
    assert "Motor writes are suppressed" in result.output
    runtime, _host, _port, _kwargs = calls[0]
    assert runtime.backend == "dynamixel:/dev/ttyUSB0"
    assert runtime.dry_run is True


def test_operator_ui_cli_rejects_removed_fake_option(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        cli, "_launch_operator_ui", lambda *args, **kwargs: calls.append(args)
    )

    result = runner.invoke(app, ["ui", "--fake", "--port", "/dev/ttyUSB0"])

    assert result.exit_code == 2
    assert "No such option: --fake" in result.output
    assert calls == []


def _hand_frame(*, timestamp_ns: int, has_hand: bool) -> HandFrame:
    return HandFrame(
        timestamp_ns=timestamp_ns,
        image_width=3,
        image_height=2,
        hands=(
            (
                HandDetection(
                    handedness="Right",
                    score=0.9,
                    landmarks=tuple(_open_hand_landmarks()),
                ),
            )
            if has_hand
            else ()
        ),
    )


def _translated_hand_frame(*, timestamp_ns: int, offset_x: float) -> HandFrame:
    return HandFrame(
        timestamp_ns=timestamp_ns,
        image_width=3,
        image_height=2,
        hands=(
            HandDetection(
                handedness="Right",
                score=0.9,
                landmarks=tuple(
                    Landmark(landmark.x + offset_x, landmark.y, landmark.z)
                    for landmark in _open_hand_landmarks()
                ),
            ),
        ),
    )


def _snapshot_joint(snapshot, joint_name: str):
    return next(state for state in snapshot.joints if state.joint_name == joint_name)


def _calibration(runtime, joint_name: str):
    return next(
        calibration
        for calibration in runtime.calibrations
        if calibration.joint_name == joint_name
    )


def _extended_calibration_feature_queue() -> list[HandFeatures]:
    pose_values = (
        (0.1, 0.2, 0.1, 0.2),
        (0.1, -0.6, 0.1, 0.8),
        (0.7, 0.2, 0.1, 0.2),
        (0.2, 0.2, 0.1, 0.7),
        (0.5, 0.2, 0.1, 0.2),
        (0.45, 0.2, 0.1, 0.2),
        (0.4, 0.2, 0.1, 0.2),
        (0.2, 0.15, 0.1, 0.4),
        (0.3, -0.4, 0.1, 0.5),
        (0.15, -0.1, 0.1, 0.3),
        (0.1, 0.2, 0.8, 0.2),
        (0.1, 0.2, 0.1, 0.3),
        (0.1, 0.2, 0.1, 0.4),
        (0.6, 0.2, 0.1, 0.9),
    )
    return [
        sample
        for curl, spread, pinch_index, thumb_abduction in pose_values
        for sample in _calibration_features(
            curl,
            spread,
            pinch_index,
            thumb_abduction=thumb_abduction,
        )
    ]


def _calibration_features(
    curl: float,
    spread: float,
    pinch_index: float,
    *,
    thumb_abduction: float = 0.0,
) -> tuple[HandFeatures, HandFeatures]:
    return (
        _feature_sample(curl, spread, pinch_index, thumb_abduction=thumb_abduction),
        _feature_sample(
            curl + 0.1, spread, pinch_index, thumb_abduction=thumb_abduction
        ),
    )


def _feature_sample(
    curl: float,
    spread: float,
    pinch_index: float,
    *,
    thumb_abduction: float = 0.0,
) -> HandFeatures:
    return HandFeatures(
        timestamp_ns=123,
        handedness="Right",
        score=0.9,
        finger_curl={finger: curl for finger in FINGER_NAMES},
        finger_spread={finger: spread for finger in SPREAD_FINGER_NAMES},
        pinch={
            finger: (pinch_index if finger == "index" else 0.1)
            for finger in PINCH_FINGER_NAMES
        },
        palm_size=0.3,
        thumb_abduction=thumb_abduction,
    )


def _retarget_calibration_with_pose_feature_anchors() -> HandRetargetCalibration:
    calibration = HandRetargetCalibration.identity(handedness="Right")
    data = calibration.to_dict()
    data["poses"]["flat_closed"] = {"samples": 2}
    data["poses"]["open_index_flex"] = {"samples": 2}
    data["pose_feature_anchors"] = {
        "flat_closed": _pose_feature_anchor_from_features(
            _feature_sample(0.0, 0.0, 0.0)
        ).to_dict(),
        "open_index_flex": _pose_feature_anchor_from_features(
            _feature_sample(1.0, 0.0, 0.0)
        ).to_dict(),
    }
    return HandRetargetCalibration.from_dict(data)


def _scaled_retarget_calibration_with_pose_feature_anchors() -> HandRetargetCalibration:
    calibration = _scaled_retarget_calibration()
    data = calibration.to_dict()
    data["pose_feature_anchors"] = {
        "flat_closed": _pose_feature_anchor_from_features(
            _feature_sample(0.2, 0.2, 0.1)
        ).to_dict(),
        "open_index_flex": _pose_feature_anchor_from_features(
            _feature_sample(0.6, 0.2, 0.1)
        ).to_dict(),
    }
    return HandRetargetCalibration.from_dict(data)


def _pose_feature_anchor_from_features(features: HandFeatures) -> PoseFeatureAnchor:
    return PoseFeatureAnchor(
        finger_curl=features.finger_curl,
        finger_joint_curl=features.finger_joint_curl,
        finger_spread=features.finger_spread,
        pinch=features.pinch,
        thumb_abduction=features.thumb_abduction,
        thumb_pose=features.thumb_pose,
        palm_size=features.palm_size,
    )


def _scaled_retarget_calibration() -> HandRetargetCalibration:
    return HandRetargetCalibration(
        version=1,
        handedness="Right",
        poses={
            pose: RetargetPoseStats(samples=2) for pose in RETARGET_CALIBRATION_POSES
        },
        finger_curl={
            finger: CurlCalibration(open=0.2, closed=0.6) for finger in FINGER_NAMES
        },
        finger_spread={
            finger: SpreadCalibration(neutral=0.2, spread=-0.6)
            for finger in SPREAD_FINGER_NAMES
        },
        pinch={
            finger: PinchCalibration(open=0.1, closed=0.9)
            for finger in PINCH_FINGER_NAMES
        },
    )


def _open_hand_landmarks() -> tuple[Landmark, ...]:
    return (
        Landmark(0.50, 0.90, 0.0),
        Landmark(0.38, 0.74, 0.0),
        Landmark(0.29, 0.63, 0.0),
        Landmark(0.22, 0.52, 0.0),
        Landmark(0.18, 0.42, 0.0),
        Landmark(0.40, 0.62, 0.0),
        Landmark(0.40, 0.46, 0.0),
        Landmark(0.40, 0.32, 0.0),
        Landmark(0.40, 0.18, 0.0),
        Landmark(0.50, 0.60, 0.0),
        Landmark(0.50, 0.43, 0.0),
        Landmark(0.50, 0.29, 0.0),
        Landmark(0.50, 0.16, 0.0),
        Landmark(0.60, 0.62, 0.0),
        Landmark(0.60, 0.46, 0.0),
        Landmark(0.60, 0.32, 0.0),
        Landmark(0.60, 0.18, 0.0),
        Landmark(0.70, 0.66, 0.0),
        Landmark(0.70, 0.51, 0.0),
        Landmark(0.70, 0.37, 0.0),
        Landmark(0.70, 0.24, 0.0),
    )


