from pathlib import Path

import pytest


def test_lerobot_plugin_registers_ghand_robot_config(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobotConfig

    config = GHandRobotConfig(id="test", calibration_dir=tmp_path)

    assert config.type == "ghand"
    assert config.backend == "fake"
    assert config.baudrate == 1_000_000
    assert config.target_fps == pytest.approx(30.0)
    assert config.target_velocity_limit == pytest.approx(1000.0)


def test_lerobot_plugin_mujoco_defaults_match_shared_physics_profile(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobotConfig
    from ghand.sim.mujoco import TELEOP_MUJOCO_PHYSICS_PROFILE

    config = GHandRobotConfig(
        id="test",
        calibration_dir=tmp_path,
        backend="mujoco",
    )

    assert config.target_velocity_limit == pytest.approx(
        TELEOP_MUJOCO_PHYSICS_PROFILE.target_velocity_limit_rad_s
    )
    assert config.target_fps == pytest.approx(TELEOP_MUJOCO_PHYSICS_PROFILE.control_hz)
    assert config.mujoco_actuator_kp == pytest.approx(
        TELEOP_MUJOCO_PHYSICS_PROFILE.actuator_kp
    )
    assert config.mujoco_actuator_force_limit == pytest.approx(
        TELEOP_MUJOCO_PHYSICS_PROFILE.actuator_force_limit
    )


def test_lerobot_plugin_make_robot_from_config(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.robots.utils")

    from lerobot.robots.utils import make_robot_from_config

    from ghand.lerobot_plugin import GHandRobot, GHandRobotConfig

    config = GHandRobotConfig(
        id="test",
        calibration_dir=tmp_path,
        mujoco_camera_names="overview,wrist",
    )

    robot = make_robot_from_config(config)

    assert isinstance(robot, GHandRobot)
    assert robot.robot_type == "ghand"
    assert robot.action_features
    assert set(robot.action_features) <= set(robot.observation_features)
    assert robot.observation_features["overview"] == (480, 640, 3)
    assert robot.observation_features["wrist"] == (480, 640, 3)
    assert not robot.is_connected
    robot.connect()
    assert robot.is_connected
    robot.disconnect()
    assert not robot.is_connected


def test_lerobot_plugin_rejects_invalid_backend(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobotConfig

    with pytest.raises(ValueError, match="backend"):
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="invalid",
        )


def test_lerobot_plugin_rejects_mujoco_current_position(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobotConfig

    with pytest.raises(ValueError, match="MuJoCo"):
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="mujoco",
            control_mode="current-position",
        )


def test_lerobot_plugin_fake_runtime_observation_and_action(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobot, GHandRobotConfig

    robot = GHandRobot(
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="fake",
            target_fps=30.0,
            target_velocity_limit=1000.0,
        )
    )
    action = {name: 0.0 for name in robot.action_features}
    first_joint = next(iter(robot.action_features))
    action[first_joint] = 0.25

    robot.connect()
    try:
        sent_action = robot.send_action(action)
        observation = robot.get_observation()
    finally:
        robot.disconnect()

    assert sent_action[first_joint] == pytest.approx(0.25)
    assert observation[first_joint] == pytest.approx(0.25, abs=0.005)
    assert set(robot.action_features) <= set(observation)
    assert not robot.is_connected


def test_lerobot_plugin_get_observation_does_not_command_or_step_backend(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobot, GHandRobotConfig

    robot = GHandRobot(
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="fake",
            target_velocity_limit=1000.0,
        )
    )

    robot.connect()
    try:
        assert robot.runtime is not None
        bus = robot.runtime.controller.bus
        calls_before = len(bus.calls)
        observation = robot.get_observation()
        new_calls = bus.calls[calls_before:]
    finally:
        robot.disconnect()

    assert new_calls, "get_observation() must read back motor state"
    assert all(str(call[0]).startswith(("sync_read", "read")) for call in new_calls)
    assert set(robot.action_features) <= set(observation)


def test_lerobot_plugin_send_action_requires_connected_runtime(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.robots.config")
    pytest.importorskip("lerobot.utils.errors")

    from lerobot.utils.errors import DeviceNotConnectedError

    from ghand.lerobot_plugin import GHandRobot, GHandRobotConfig

    robot = GHandRobot(GHandRobotConfig(id="test", calibration_dir=tmp_path))
    action = {name: 0.0 for name in robot.action_features}

    with pytest.raises(DeviceNotConnectedError):
        robot.send_action(action)


def test_lerobot_plugin_send_action_validates_action_schema(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobot, GHandRobotConfig

    robot = GHandRobot(GHandRobotConfig(id="test", calibration_dir=tmp_path))
    action = {name: 0.0 for name in robot.action_features}
    first_joint = next(iter(robot.action_features))

    robot.connect()
    try:
        missing = dict(action)
        missing.pop(first_joint)
        with pytest.raises(ValueError, match="Missing LEAP V1 KAIZEN action key"):
            robot.send_action(missing)

        extra = dict(action)
        extra["extra.pos"] = 0.0
        with pytest.raises(ValueError, match="Unsupported LEAP V1 KAIZEN action key"):
            robot.send_action(extra)

        nonfinite = dict(action)
        nonfinite[first_joint] = float("nan")
        with pytest.raises(ValueError, match="must be finite"):
            robot.send_action(nonfinite)
    finally:
        robot.disconnect()


def test_lerobot_plugin_rejects_current_position_without_default_current(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobotConfig

    with pytest.raises(ValueError, match="default_current_limit_ma"):
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            control_mode="current-position",
        )


def test_lerobot_plugin_rejects_real_left_side(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobotConfig

    with pytest.raises(ValueError, match="side='right'"):
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="real",
            side="left",
            port="/dev/ttyUSB0",
        )


def test_lerobot_plugin_position_current_features_match_schema(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.data import (
        make_hand_position_current_action_names,
        make_hand_position_current_state_names,
    )
    from ghand.lerobot_plugin import GHandRobot, GHandRobotConfig

    robot = GHandRobot(
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            control_mode="current-position",
            dataset_schema="position-current",
            default_current_limit_ma=50.0,
        )
    )

    assert tuple(robot.observation_features) == make_hand_position_current_state_names(
        ("right",)
    )
    assert tuple(robot.action_features) == make_hand_position_current_action_names(
        ("right",)
    )
    assert len(robot.observation_features) == 32
    assert len(robot.action_features) == 32


def test_lerobot_plugin_position_current_runtime_applies_goal_current(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobot, GHandRobotConfig

    robot = GHandRobot(
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="fake",
            control_mode="current-position",
            dataset_schema="position-current",
            default_current_limit_ma=50.0,
            max_current_limit_ma=100.0,
            target_velocity_limit=1000.0,
        )
    )
    action = {name: 0.0 for name in robot.action_features}
    position_key = "right_hand.index_mcp_flex.pos"
    current_key = "right_hand.index_mcp_flex.current_limit_ma"
    action[position_key] = 0.3
    action[current_key] = 90.0

    robot.connect()
    try:
        sent_action = robot.send_action(action)
        assert robot.runtime is not None
        assert robot.runtime.controller.bus.motors[9].goal_current_ma == pytest.approx(
            90.0
        )
    finally:
        robot.disconnect()

    assert sent_action[position_key] == pytest.approx(0.3)
    assert sent_action[current_key] == pytest.approx(90.0)


def test_lerobot_plugin_position_current_observation_can_emit_diagnostics(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobot, GHandRobotConfig

    robot = GHandRobot(
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="fake",
            control_mode="current-position",
            dataset_schema="position-current",
            default_current_limit_ma=50.0,
            include_current_diagnostics=True,
            current_contact_threshold_ma=50.0,
            target_velocity_limit=1000.0,
        )
    )

    robot.connect()
    try:
        assert robot.runtime is not None
        robot.runtime.controller.bus.motors[9].present_current_ma = 80.0
        observation = robot.get_observation()
    finally:
        robot.disconnect()

    assert observation["right_hand.index_mcp_flex.present_current_ma"] == pytest.approx(
        80.0
    )
    assert observation[
        "right_hand.index_mcp_flex.current_residual_ma"
    ] == pytest.approx(30.0)
    assert observation["right_hand.index_mcp_flex.contact"] == pytest.approx(1.0)


def test_lerobot_plugin_rejects_position_current_without_current_control(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobotConfig

    with pytest.raises(ValueError, match="requires control_mode"):
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            dataset_schema="position-current",
        )


def test_lerobot_plugin_rejects_mujoco_position_current_schema(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobotConfig

    with pytest.raises(ValueError, match="MuJoCo"):
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="mujoco",
            control_mode="current-position",
            dataset_schema="position-current",
            default_current_limit_ma=50.0,
        )


def test_lerobot_policy_smoke_default_action_uses_current_default(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobotConfig
    from ghand.lerobot_plugin.policy_smoke import run_ghand_policy_smoke

    result = run_ghand_policy_smoke(
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="fake",
            control_mode="current-position",
            dataset_schema="position-current",
            default_current_limit_ma=55.0,
            target_velocity_limit=1000.0,
        )
    )

    assert result.action_source == "default"
    assert result.action_feature_count == 32
    assert result.sent_action[
        "right_hand.index_mcp_flex.current_limit_ma"
    ] == pytest.approx(55.0)


def test_lerobot_policy_smoke_sparse_action_json_reaches_runtime(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobotConfig
    from ghand.lerobot_plugin.policy_smoke import run_ghand_policy_smoke

    result = run_ghand_policy_smoke(
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="fake",
            target_velocity_limit=1000.0,
        ),
        action_values={"right_hand.index_mcp_flex.pos": 0.2},
    )

    assert result.action_source == "json"
    assert result.sent_action["right_hand.index_mcp_flex.pos"] == pytest.approx(0.2)
    assert "observation.state" in result.dataset_feature_keys
    assert "action" in result.dataset_feature_keys


def test_lerobot_policy_smoke_cli_json(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.robots.config")

    from typer.testing import CliRunner

    from ghand.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "policy-smoke",
            "--dataset-schema",
            "position-current",
            "--control-mode",
            "current-position",
            "--default-current-limit-ma",
            "50",
            "--action-json",
            '{"right_hand.index_mcp_flex.pos": 0.25, "right_hand.index_mcp_flex.current_limit_ma": 90.0}',
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = __import__("json").loads(result.output)
    assert payload["dataset_schema"] == "position-current"
    assert payload["action_source"] == "json"
    assert payload["action_feature_count"] == 32
    assert payload["sent_action"]["right_hand.index_mcp_flex.pos"] == pytest.approx(
        0.25, abs=0.02
    )
    assert payload["sent_action"][
        "right_hand.index_mcp_flex.current_limit_ma"
    ] == pytest.approx(90.0)


def test_lerobot_plugin_rejects_observation_camera_with_mujoco_backend(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobotConfig

    with pytest.raises(ValueError, match="observation_camera"):
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="mujoco",
            observation_camera=0,
        )


class _FakeObservationWebcam:
    """Webcam stand-in returning a fixed BGR frame."""

    instances: list["_FakeObservationWebcam"] = []

    def __init__(
        self,
        camera: int = 0,
        *,
        latest: bool = False,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        import numpy as np

        self.camera = camera
        self.latest = latest
        self.width = width
        self.height = height
        self.opened = False
        self.closed = False
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[..., 0] = 255  # blue channel in BGR
        self.frame = frame
        _FakeObservationWebcam.instances.append(self)

    def open(self) -> None:
        self.opened = True

    def read(self):
        return self.frame

    def close(self) -> None:
        self.closed = True


def test_lerobot_plugin_fake_backend_serves_webcam_observation_camera(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip("lerobot.robots.config")

    from ghand.lerobot_plugin import GHandRobot, GHandRobotConfig

    _FakeObservationWebcam.instances = []
    monkeypatch.setattr("ghand.lerobot_plugin.robot.Webcam", _FakeObservationWebcam)
    robot = GHandRobot(
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="fake",
            observation_camera=2,
        )
    )

    assert robot.observation_features["front"] == (480, 640, 3)

    robot.connect()
    try:
        observation = robot.get_observation()
    finally:
        robot.disconnect()

    (webcam,) = _FakeObservationWebcam.instances
    assert webcam.camera == 2
    assert webcam.opened
    assert webcam.width == 640
    assert webcam.height == 480
    assert webcam.closed
    frame = observation["front"]
    assert frame.shape == (480, 640, 3)
    # The BGR webcam frame must arrive RGB: blue ends up in the last channel.
    assert frame[0, 0, 2] == 255
    assert frame[0, 0, 0] == 0


def test_lerobot_plugin_webcam_observation_resizes_to_camera_shape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip("lerobot.robots.config")
    pytest.importorskip("cv2")

    from ghand.lerobot_plugin import GHandRobot, GHandRobotConfig

    _FakeObservationWebcam.instances = []
    monkeypatch.setattr("ghand.lerobot_plugin.robot.Webcam", _FakeObservationWebcam)
    robot = GHandRobot(
        GHandRobotConfig(
            id="test",
            calibration_dir=tmp_path,
            backend="fake",
            observation_camera=0,
            camera_width=320,
            camera_height=240,
        )
    )

    robot.connect()
    try:
        observation = robot.get_observation()
    finally:
        robot.disconnect()

    (webcam,) = _FakeObservationWebcam.instances
    assert webcam.width == 320
    assert webcam.height == 240
    assert observation["front"].shape == (240, 320, 3)
