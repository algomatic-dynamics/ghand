import numpy as np
import pytest

from ghand.control import apply_radian_joint_target, make_fake_hand_runtime
from ghand.control.joint_target_adapter import apply_position_current_action
from ghand.data import (
    ACTION_KEY,
    AUX_TELEOP_HANDEDNESS_KEY,
    AUX_TELEOP_LANDMARKS_KEY,
    AUX_TELEOP_SCORE_KEY,
    AUX_TELEOP_VALID_KEY,
    AUX_TELEOP_WORLD_LANDMARKS_KEY,
    CameraSpec,
    FRONT_IMAGE_KEY,
    HANDEDNESS_RIGHT,
    OBSERVATION_STATE_KEY,
    make_hand_lerobot_features,
    make_hand_position_current_lerobot_features,
    make_lerobot_frame_from_runtime_snapshot,
)
from ghand.hand import (
    CurrentLimitTarget,
    PositionCurrentAction,
    RadianJointTarget,
)
from ghand.teleop.hand_pose import HandDetection, HandFrame, Landmark


def test_lerobot_frame_uses_measured_state_target_action_camera_and_aux() -> None:
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    target = RadianJointTarget(
        timestamp_ns=123_000_000,
        joints={"right_hand.index_mcp_flex.pos": 0.5},
    )
    snapshot = apply_radian_joint_target(runtime, target, now=0.0)
    features = make_hand_lerobot_features(front_image_shape=(2, 2, 3), image_dtype="image")
    frame_bgr = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    hand_frame = HandFrame(
        timestamp_ns=123_000_000,
        image_width=2,
        image_height=2,
        hands=(
            HandDetection(
                handedness="Right",
                score=0.75,
                landmarks=_landmarks(0.0),
                world_landmarks=_landmarks(1.0),
            ),
        ),
    )

    frame = make_lerobot_frame_from_runtime_snapshot(
        snapshot=snapshot,
        features=features,
        task="record right hand",
        camera_frames={"front": frame_bgr},
        hand_frame=hand_frame,
    )

    joint_names = features[OBSERVATION_STATE_KEY]["names"]
    assert isinstance(joint_names, list)
    joint_index = joint_names.index("right_hand.index_mcp_flex.pos")
    joint_state = _joint(snapshot, "right_hand.index_mcp_flex.pos")
    assert frame["task"] == "record right hand"
    assert frame["timestamp"] == pytest.approx(0.123)
    assert frame[OBSERVATION_STATE_KEY].dtype == np.float32
    assert frame[ACTION_KEY].dtype == np.float32
    assert frame[OBSERVATION_STATE_KEY].shape == (16,)
    assert frame[ACTION_KEY].shape == (16,)
    assert frame[OBSERVATION_STATE_KEY][joint_index] == pytest.approx(joint_state.measured)
    assert frame[ACTION_KEY][joint_index] == pytest.approx(joint_state.target)
    assert frame[FRONT_IMAGE_KEY].dtype == np.uint8
    assert frame[FRONT_IMAGE_KEY][0, 0].tolist() == [3, 2, 1]

    right_slot = 1
    assert frame[AUX_TELEOP_VALID_KEY].tolist() == [0, 1]
    assert frame[AUX_TELEOP_HANDEDNESS_KEY].tolist() == [0, HANDEDNESS_RIGHT]
    assert frame[AUX_TELEOP_SCORE_KEY][right_slot] == pytest.approx(0.75)
    assert frame[AUX_TELEOP_LANDMARKS_KEY][right_slot, 0].tolist() == pytest.approx(
        [0.0, 0.1, 0.2]
    )
    assert frame[AUX_TELEOP_WORLD_LANDMARKS_KEY][right_slot, 0].tolist() == pytest.approx(
        [1.0, 1.1, 1.2]
    )


def test_lerobot_frame_can_use_commanded_action_source() -> None:
    runtime = make_fake_hand_runtime(target_hz=10.0, target_velocity_limit=1.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    target = RadianJointTarget(
        timestamp_ns=1,
        joints={"right_hand.index_mcp_flex.pos": 0.5},
    )
    snapshot = apply_radian_joint_target(runtime, target, now=0.0)
    features = make_hand_lerobot_features(
        front_image_shape=(1, 1, 3),
        image_dtype="image",
        include_aux=False,
    )

    frame = make_lerobot_frame_from_runtime_snapshot(
        snapshot=snapshot,
        features=features,
        task="commanded action",
        camera_frames={"front": np.zeros((1, 1, 3), dtype=np.uint8)},
        action_source="commanded",
    )

    joint_names = features[ACTION_KEY]["names"]
    assert isinstance(joint_names, list)
    joint_index = joint_names.index("right_hand.index_mcp_flex.pos")
    joint_state = _joint(snapshot, "right_hand.index_mcp_flex.pos")
    assert joint_state.target == pytest.approx(0.5)
    assert joint_state.commanded == pytest.approx(0.1)
    assert frame[ACTION_KEY][joint_index] == pytest.approx(joint_state.commanded)


def test_lerobot_frame_accepts_explicit_radian_action_override() -> None:
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    snapshot = apply_radian_joint_target(
        runtime,
        RadianJointTarget(
            timestamp_ns=1,
            joints={
                "right_hand.index_mcp_flex.pos": 0.5,
                "right_hand.index_pip_flex.pos": 0.25,
            },
        ),
        now=0.0,
    )
    features = make_hand_lerobot_features(
        front_image_shape=(1, 1, 3),
        image_dtype="image",
        include_aux=False,
    )

    frame = make_lerobot_frame_from_runtime_snapshot(
        snapshot=snapshot,
        features=features,
        task="override action",
        camera_frames={"front": np.zeros((1, 1, 3), dtype=np.uint8)},
        action=RadianJointTarget(
            timestamp_ns=2,
            joints={"right_hand.index_mcp_flex.pos": -0.125},
        ),
    )

    joint_names = features[ACTION_KEY]["names"]
    assert isinstance(joint_names, list)
    mcp_index = joint_names.index("right_hand.index_mcp_flex.pos")
    pip_index = joint_names.index("right_hand.index_pip_flex.pos")
    assert frame[ACTION_KEY][mcp_index] == pytest.approx(-0.125)
    assert frame[ACTION_KEY][pip_index] == pytest.approx(
        _joint(snapshot, "right_hand.index_pip_flex.pos").target
    )
    assert AUX_TELEOP_VALID_KEY not in frame


def test_lerobot_frame_supports_position_current_action_schema() -> None:
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.controller.bus.motors[9].present_current_ma = 12.0
    runtime.controller.bus.motors[10].present_current_ma = -3.0
    snapshot = apply_radian_joint_target(
        runtime,
        RadianJointTarget(
            timestamp_ns=1,
            joints={
                "right_hand.index_mcp_flex.pos": 0.5,
                "right_hand.index_pip_flex.pos": 0.25,
            },
        ),
        now=0.0,
    )
    features = make_hand_position_current_lerobot_features(
        front_image_shape=(1, 1, 3),
        image_dtype="image",
        include_aux=False,
    )

    frame = make_lerobot_frame_from_runtime_snapshot(
        snapshot=snapshot,
        features=features,
        task="position current action",
        camera_frames={"front": np.zeros((1, 1, 3), dtype=np.uint8)},
        current_limit_action={"right_hand.index_mcp_flex.pos": 90.0},
        default_current_limit_ma=50.0,
    )

    state_names = features[OBSERVATION_STATE_KEY]["names"]
    action_names = features[ACTION_KEY]["names"]
    assert isinstance(state_names, list)
    assert isinstance(action_names, list)
    mcp_position_index = state_names.index("right_hand.index_mcp_flex.pos")
    mcp_current_index = state_names.index("right_hand.index_mcp_flex.present_current_ma")
    pip_current_index = state_names.index("right_hand.index_pip_flex.present_current_ma")
    mcp_action_position_index = action_names.index("right_hand.index_mcp_flex.pos")
    mcp_action_current_index = action_names.index(
        "right_hand.index_mcp_flex.current_limit_ma"
    )
    pip_action_current_index = action_names.index(
        "right_hand.index_pip_flex.current_limit_ma"
    )

    assert frame[OBSERVATION_STATE_KEY].shape == (32,)
    assert frame[ACTION_KEY].shape == (32,)
    assert frame[OBSERVATION_STATE_KEY][mcp_position_index] == pytest.approx(
        _joint(snapshot, "right_hand.index_mcp_flex.pos").measured
    )
    assert frame[OBSERVATION_STATE_KEY][mcp_current_index] == pytest.approx(12.0)
    assert frame[OBSERVATION_STATE_KEY][pip_current_index] == pytest.approx(-3.0)
    assert frame[ACTION_KEY][mcp_action_position_index] == pytest.approx(
        _joint(snapshot, "right_hand.index_mcp_flex.pos").target
    )
    assert frame[ACTION_KEY][mcp_action_current_index] == pytest.approx(90.0)
    assert frame[ACTION_KEY][pip_action_current_index] == pytest.approx(50.0)


def test_lerobot_frame_requires_current_limit_values_for_current_action_schema() -> None:
    runtime = make_fake_hand_runtime()
    runtime.connect()
    snapshot = runtime.step(now=0.0)
    features = make_hand_position_current_lerobot_features(
        front_image_shape=(1, 1, 3),
        image_dtype="image",
        include_aux=False,
    )

    with pytest.raises(ValueError, match="current_limit_ma"):
        make_lerobot_frame_from_runtime_snapshot(
            snapshot=snapshot,
            features=features,
            task="missing current limit",
            camera_frames={"front": np.zeros((1, 1, 3), dtype=np.uint8)},
        )


def test_lerobot_frame_can_use_runtime_current_limit_action_snapshot() -> None:
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    snapshot = apply_position_current_action(
        runtime,
        PositionCurrentAction(
            position=RadianJointTarget(
                timestamp_ns=1,
                joints={"right_hand.index_mcp_flex.pos": 0.5},
            ),
            current_limit=CurrentLimitTarget(
                timestamp_ns=1,
                currents_ma={"right_hand.index_mcp_flex.pos": 90.0},
            ),
        ),
        now=0.0,
    )
    features = make_hand_position_current_lerobot_features(
        front_image_shape=(1, 1, 3),
        image_dtype="image",
        include_aux=False,
    )

    frame = make_lerobot_frame_from_runtime_snapshot(
        snapshot=snapshot,
        features=features,
        task="snapshot current action",
        camera_frames={"front": np.zeros((1, 1, 3), dtype=np.uint8)},
        default_current_limit_ma=50.0,
    )

    action_names = features[ACTION_KEY]["names"]
    assert isinstance(action_names, list)
    mcp_action_current_index = action_names.index(
        "right_hand.index_mcp_flex.current_limit_ma"
    )
    pip_action_current_index = action_names.index(
        "right_hand.index_pip_flex.current_limit_ma"
    )
    assert frame[ACTION_KEY][mcp_action_current_index] == pytest.approx(90.0)
    assert frame[ACTION_KEY][pip_action_current_index] == pytest.approx(50.0)


def test_lerobot_frame_can_emit_current_residual_and_contact_flags() -> None:
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.controller.bus.motors[9].present_current_ma = 80.0
    runtime.controller.bus.motors[10].present_current_ma = -20.0
    snapshot = runtime.step(now=0.0)
    features = make_hand_position_current_lerobot_features(
        front_image_shape=(1, 1, 3),
        image_dtype="image",
        include_aux=False,
        include_current_diagnostics=True,
    )

    frame = make_lerobot_frame_from_runtime_snapshot(
        snapshot=snapshot,
        features=features,
        task="current diagnostics",
        camera_frames={"front": np.zeros((1, 1, 3), dtype=np.uint8)},
        default_current_limit_ma=90.0,
        current_contact_threshold_ma=50.0,
    )

    state_names = features[OBSERVATION_STATE_KEY]["names"]
    mcp_residual_index = state_names.index(
        "right_hand.index_mcp_flex.current_residual_ma"
    )
    pip_residual_index = state_names.index(
        "right_hand.index_pip_flex.current_residual_ma"
    )
    mcp_contact_index = state_names.index("right_hand.index_mcp_flex.contact")
    pip_contact_index = state_names.index("right_hand.index_pip_flex.contact")
    assert frame[OBSERVATION_STATE_KEY].shape == (64,)
    assert frame[OBSERVATION_STATE_KEY][mcp_residual_index] == pytest.approx(30.0)
    assert frame[OBSERVATION_STATE_KEY][pip_residual_index] == pytest.approx(0.0)
    assert frame[OBSERVATION_STATE_KEY][mcp_contact_index] == pytest.approx(1.0)
    assert frame[OBSERVATION_STATE_KEY][pip_contact_index] == pytest.approx(0.0)


def test_lerobot_frame_supports_multiple_dataset_cameras_without_color_conversion() -> None:
    runtime = make_fake_hand_runtime()
    features = make_hand_lerobot_features(
        cameras=(
            CameraSpec("front", (1, 1, 3), "image"),
            CameraSpec("external", (1, 2, 3), "image"),
        ),
        include_aux=False,
    )
    front_rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
    external_rgb = np.array([[[4, 5, 6], [7, 8, 9]]], dtype=np.uint8)

    frame = make_lerobot_frame_from_runtime_snapshot(
        snapshot=runtime.snapshot(),
        features=features,
        task="multi camera",
        camera_frames={
            FRONT_IMAGE_KEY: front_rgb,
            "external": external_rgb,
        },
        image_color_format="rgb",
    )

    assert frame[FRONT_IMAGE_KEY].tolist() == front_rgb.tolist()
    assert frame["observation.images.external"].tolist() == external_rgb.tolist()


def test_lerobot_frame_rejects_missing_joint_or_camera() -> None:
    runtime = make_fake_hand_runtime()
    features = make_hand_lerobot_features(front_image_shape=(1, 1, 3), image_dtype="image")

    with pytest.raises(ValueError, match="Missing camera frame"):
        make_lerobot_frame_from_runtime_snapshot(
            snapshot=runtime.snapshot(),
            features=features,
            task="missing camera",
        )

    broken_features = make_hand_lerobot_features(
        front_image_shape=(1, 1, 3),
        image_dtype="image",
        include_aux=False,
    )
    assert isinstance(broken_features[OBSERVATION_STATE_KEY]["names"], list)
    broken_features[OBSERVATION_STATE_KEY]["names"].append("right_arm.elbow.pos")
    with pytest.raises(ValueError, match="right_arm.elbow.pos"):
        make_lerobot_frame_from_runtime_snapshot(
            snapshot=runtime.snapshot(),
            features=broken_features,
            task="missing joint",
            camera_frames={"front": np.zeros((1, 1, 3), dtype=np.uint8)},
        )

    with pytest.raises(ValueError, match="feature shape"):
        make_lerobot_frame_from_runtime_snapshot(
            snapshot=runtime.snapshot(),
            features=features,
            task="wrong camera shape",
            camera_frames={"front": np.zeros((2, 1, 3), dtype=np.uint8)},
        )


def _landmarks(offset: float) -> tuple[Landmark, ...]:
    return tuple(
        Landmark(offset + index, offset + index + 0.1, offset + index + 0.2)
        for index in range(21)
    )


def _joint(snapshot, joint_name):
    return next(state for state in snapshot.joints if state.joint_name == joint_name)
