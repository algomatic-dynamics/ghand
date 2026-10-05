import math

import pytest

from ghand.control import (
    NormalizedToRadianJointTargetAdapter,
    PositionCurrentActionVectorAdapter,
    apply_joint_target,
    apply_position_current_action,
    apply_radian_joint_target,
    make_fake_hand_runtime,
)
from ghand.hand import (
    CurrentLimitTarget,
    JointCalibration,
    JointTarget,
    PositionCurrentAction,
    RadianJointTarget,
)
from ghand.data import ACTION_KEY, make_hand_position_current_lerobot_features


def test_apply_joint_target_writes_through_fake_runtime() -> None:
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)

    target = JointTarget(
        timestamp_ns=123,
        joints={
            "right_hand.index_mcp_flex.pos": 0.5,
            "right_hand.index_pip_flex.pos": 0.25,
        },
    )
    snapshot = apply_joint_target(runtime, target, now=0.0)

    index_mcp = _joint(snapshot, "right_hand.index_mcp_flex.pos")
    index_pip = _joint(snapshot, "right_hand.index_pip_flex.pos")
    index_mcp_calibration = _calibration(runtime, "right_hand.index_mcp_flex.pos")
    index_pip_calibration = _calibration(runtime, "right_hand.index_pip_flex.pos")
    expected_index_mcp_rad = index_mcp_calibration.normalized_to_angle_rad(0.5)
    expected_index_pip_rad = index_pip_calibration.normalized_to_angle_rad(0.25)
    assert expected_index_mcp_rad == pytest.approx(math.pi / 2)
    assert index_mcp.target == pytest.approx(expected_index_mcp_rad)
    assert index_mcp.commanded == pytest.approx(expected_index_mcp_rad)
    assert index_mcp.measured == pytest.approx(
        expected_index_mcp_rad,
        abs=1 / index_mcp_calibration.ticks_per_radian,
    )
    expected_index_mcp_tick = index_mcp_calibration.angle_rad_to_tick(expected_index_mcp_rad)
    expected_index_pip_tick = index_pip_calibration.angle_rad_to_tick(expected_index_pip_rad)
    assert index_mcp.target_tick == expected_index_mcp_tick
    assert index_mcp.measured_tick == expected_index_mcp_tick
    assert index_pip.target == pytest.approx(expected_index_pip_rad)
    assert index_pip.measured == pytest.approx(
        expected_index_pip_rad,
        abs=1 / index_pip_calibration.ticks_per_radian,
    )
    assert ("goal", 9, expected_index_mcp_tick) in runtime.controller.bus.calls
    assert ("goal", 10, expected_index_pip_tick) in runtime.controller.bus.calls


def test_apply_joint_target_preserves_previous_targets_for_empty_update() -> None:
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)

    apply_joint_target(
        runtime,
        JointTarget(
            timestamp_ns=123,
            joints={"right_hand.index_mcp_flex.pos": 0.5},
        ),
        now=0.0,
    )
    snapshot = apply_joint_target(runtime, JointTarget.empty(timestamp_ns=124), now=0.1)

    calibration = _calibration(runtime, "right_hand.index_mcp_flex.pos")
    assert _joint(snapshot, "right_hand.index_mcp_flex.pos").target == pytest.approx(
        calibration.normalized_to_angle_rad(0.5)
    )


def test_apply_joint_target_rejects_targets_outside_runtime_joint_set() -> None:
    runtime = make_fake_hand_runtime(side="right")
    target = JointTarget(
        timestamp_ns=123,
        joints={"left_hand.index_mcp_flex.pos": 0.5},
    )

    with pytest.raises(ValueError, match="not configured in runtime"):
        apply_joint_target(runtime, target, now=0.0)


def test_make_fake_hand_runtime_can_use_left_hand_joint_names() -> None:
    runtime = make_fake_hand_runtime(side="left", target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    target = JointTarget(
        timestamp_ns=123,
        joints={"left_hand.index_mcp_flex.pos": 0.5},
    )

    snapshot = apply_joint_target(runtime, target, now=0.0)

    calibration = _calibration(runtime, "left_hand.index_mcp_flex.pos")
    assert _joint(snapshot, "left_hand.index_mcp_flex.pos").measured == pytest.approx(
        calibration.normalized_to_angle_rad(0.5),
        abs=1 / calibration.ticks_per_radian,
    )
    expected_tick = calibration.angle_rad_to_tick(calibration.normalized_to_angle_rad(0.5))
    assert ("goal", 9, expected_tick) in runtime.controller.bus.calls


def test_normalized_to_radian_adapter_maps_each_side_of_zero() -> None:
    calibration = JointCalibration(
        joint_name="right_hand.index_mcp_flex.pos",
        dxl_id=9,
        lower_rad=-0.5,
        upper_rad=1.25,
    )
    adapter = NormalizedToRadianJointTargetAdapter((calibration,))

    target = adapter(
        JointTarget(
            timestamp_ns=123,
            joints={
                "right_hand.index_mcp_flex.pos": -0.4,
            },
        )
    )

    assert isinstance(target, RadianJointTarget)
    assert target.timestamp_ns == 123
    assert target.joints["right_hand.index_mcp_flex.pos"] == pytest.approx(-0.2)


def test_apply_radian_joint_target_bypasses_normalized_conversion() -> None:
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    target = RadianJointTarget(
        timestamp_ns=123,
        joints={"right_hand.index_mcp_flex.pos": 0.5},
    )

    snapshot = apply_radian_joint_target(runtime, target, now=0.0)

    assert _joint(snapshot, "right_hand.index_mcp_flex.pos").target == 0.5


def test_position_current_action_vector_adapter_parses_lerobot_action_names() -> None:
    features = make_hand_position_current_lerobot_features(include_aux=False)
    action_names = features[ACTION_KEY]["names"]
    assert isinstance(action_names, list)
    values = [0.0] * len(action_names)
    values[action_names.index("right_hand.index_mcp_flex.pos")] = 0.5
    values[action_names.index("right_hand.index_mcp_flex.current_limit_ma")] = 90.0

    adapter = PositionCurrentActionVectorAdapter.from_features(features)
    action = adapter(values, timestamp_ns=123)

    assert action.timestamp_ns == 123
    assert action.position.joints["right_hand.index_mcp_flex.pos"] == pytest.approx(0.5)
    assert action.current_limit.currents_ma["right_hand.index_mcp_flex.pos"] == pytest.approx(
        90.0
    )
    assert action.current_limit.currents_ma["right_hand.middle_mcp_flex.pos"] == 0.0


def test_position_current_action_vector_adapter_rejects_wrong_shape_or_schema() -> None:
    features = make_hand_position_current_lerobot_features(include_aux=False)
    adapter = PositionCurrentActionVectorAdapter.from_features(features)

    with pytest.raises(ValueError, match="length"):
        adapter((0.0,), timestamp_ns=123)

    with pytest.raises(ValueError, match="current limit"):
        adapter(
            [0.0] * 16 + [-1.0] * 16,
            timestamp_ns=123,
        )

    with pytest.raises(ValueError, match="current-limit channels"):
        PositionCurrentActionVectorAdapter(
            ("right_hand.index_mcp_flex.pos",),
        )


def test_apply_position_current_action_stores_current_limits_without_hardware_current_write() -> None:
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    action = PositionCurrentAction(
        position=RadianJointTarget(
            timestamp_ns=123,
            joints={"right_hand.index_mcp_flex.pos": 0.5},
        ),
        current_limit=CurrentLimitTarget(
            timestamp_ns=123,
            currents_ma={"right_hand.index_mcp_flex.pos": 90.0},
        ),
    )

    snapshot = apply_position_current_action(runtime, action, now=0.0)

    index_mcp = _joint(snapshot, "right_hand.index_mcp_flex.pos")
    assert index_mcp.target == pytest.approx(0.5)
    assert index_mcp.current_limit_ma == pytest.approx(90.0)
    assert not any(call[0] == "goal_current" for call in runtime.controller.bus.calls)


def test_apply_position_current_action_writes_current_in_current_position_mode() -> None:
    runtime = make_fake_hand_runtime(
        target_velocity_limit=1000.0,
        control_mode="current_position",
        default_current_limit_ma=50.0,
    )
    runtime.connect()
    runtime.set_torque_enabled(True)
    action = PositionCurrentAction(
        position=RadianJointTarget(
            timestamp_ns=123,
            joints={"right_hand.index_mcp_flex.pos": 0.5},
        ),
        current_limit=CurrentLimitTarget(
            timestamp_ns=123,
            currents_ma={"right_hand.index_mcp_flex.pos": 90.0},
        ),
    )

    snapshot = apply_position_current_action(runtime, action, now=0.0)

    assert _joint(snapshot, "right_hand.index_mcp_flex.pos").current_limit_ma == 90.0
    assert runtime.controller.bus.motors[9].goal_current_ma == 90.0
    assert any(call[0] == "sync_goal_current" for call in runtime.controller.bus.calls)


def test_apply_position_current_action_rejects_current_limit_for_other_hand() -> None:
    runtime = make_fake_hand_runtime(side="right")
    action = PositionCurrentAction(
        position=RadianJointTarget.empty(timestamp_ns=123),
        current_limit=CurrentLimitTarget(
            timestamp_ns=123,
            currents_ma={"left_hand.index_mcp_flex.pos": 90.0},
        ),
    )

    with pytest.raises(ValueError, match="Current-limit target"):
        apply_position_current_action(runtime, action, now=0.0)


def _joint(snapshot, joint_name):
    return next(state for state in snapshot.joints if state.joint_name == joint_name)


def _calibration(runtime, joint_name):
    return next(
        calibration
        for calibration in runtime.calibrations
        if calibration.joint_name == joint_name
    )
