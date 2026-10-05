import pytest

from ghand.control import (
    HandOperatorRuntime,
    OPERATOR_CONTROL_MODE_CURRENT_POSITION,
)
from ghand.hand import (
    GHAND_RIGHT_MOTOR_IDS,
    DEFAULT_JOINT_DIRECTIONS,
    DEFAULT_JOINT_TICKS_PER_RADIAN,
    make_default_joint_calibrations,
)
from ghand.hardware import HandController, OPERATING_MODE_CURRENT_BASED_POSITION


def test_default_joint_calibrations_map_right_hand_joints_to_motor_ids() -> None:
    calibrations = make_default_joint_calibrations()

    assert len(calibrations) == 16
    assert calibrations[0].joint_name == "right_hand.ring_mcp_side.pos"
    assert calibrations[0].dxl_id == 0
    assert calibrations[0].pose_dxl_id == 0
    assert calibrations[0].direction == DEFAULT_JOINT_DIRECTIONS[0]
    assert calibrations[0].ticks_per_radian == DEFAULT_JOINT_TICKS_PER_RADIAN[0]
    assert calibrations[-1].joint_name == "right_hand.thumb_ip_flex.pos"
    assert calibrations[-1].dxl_id == 15
    assert calibrations[-1].pose_dxl_id == 15
    target_tick = calibrations[0].angle_rad_to_tick(0.5)
    assert calibrations[0].tick_to_angle_rad(target_tick) == pytest.approx(
        0.5,
        abs=1 / calibrations[0].ticks_per_radian,
    )


def test_default_joint_calibrations_can_separate_real_motor_ids_from_urdf_ids() -> None:
    calibrations = make_default_joint_calibrations(
        motor_ids=GHAND_RIGHT_MOTOR_IDS,
    )

    ring_mcp_side = calibrations[0]
    index_mcp_flex = calibrations[9]
    assert ring_mcp_side.joint_name == "right_hand.ring_mcp_side.pos"
    assert ring_mcp_side.dxl_id == 8
    assert ring_mcp_side.pose_dxl_id == 0
    assert ring_mcp_side.direction == DEFAULT_JOINT_DIRECTIONS[0]
    assert ring_mcp_side.ticks_per_radian == DEFAULT_JOINT_TICKS_PER_RADIAN[0]
    assert index_mcp_flex.joint_name == "right_hand.index_mcp_flex.pos"
    assert index_mcp_flex.dxl_id == 1
    assert index_mcp_flex.pose_dxl_id == 9
    assert index_mcp_flex.direction == DEFAULT_JOINT_DIRECTIONS[9]
    assert index_mcp_flex.ticks_per_radian == DEFAULT_JOINT_TICKS_PER_RADIAN[9]


def test_default_joint_calibrations_apply_per_joint_motor_directions() -> None:
    directions = (1,) * 9 + (-1,) + (1,) * 6
    calibrations = make_default_joint_calibrations(directions=directions)
    index_mcp_flex = calibrations[9]

    assert index_mcp_flex.joint_name == "right_hand.index_mcp_flex.pos"
    assert index_mcp_flex.direction == -1
    target_tick = index_mcp_flex.angle_rad_to_tick(0.5)
    assert target_tick < index_mcp_flex.zero_tick
    assert index_mcp_flex.tick_to_angle_rad(target_tick) == pytest.approx(
        0.5,
        abs=1 / index_mcp_flex.ticks_per_radian,
    )


def test_default_joint_calibrations_apply_per_joint_ticks_per_radian() -> None:
    directions = (1,) * 16
    ticks_per_radian = (512.0,) * 9 + (1024.0,) + (512.0,) * 6
    calibrations = make_default_joint_calibrations(
        directions=directions,
        ticks_per_radian=ticks_per_radian,
    )
    index_mcp_flex = calibrations[9]

    assert index_mcp_flex.joint_name == "right_hand.index_mcp_flex.pos"
    assert index_mcp_flex.ticks_per_radian == 1024.0
    assert index_mcp_flex.angle_rad_to_tick(0.5) == 2560
    assert index_mcp_flex.tick_to_angle_rad(2560) == 0.5


def test_operator_runtime_writes_through_fake_hand_controller() -> None:
    controller = HandController.from_fake()
    calibrations = make_default_joint_calibrations()
    runtime = HandOperatorRuntime(
        controller,
        calibrations,
        backend="fake",
        dry_run=False,
        target_velocity_limit=1000.0,
    )

    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.set_target("right_hand.index_mcp_flex.pos", 0.5)
    snapshot = runtime.step(now=0.0)

    state = _joint(snapshot, "right_hand.index_mcp_flex.pos")
    index_mcp_flex = calibrations[9]
    expected_tick = index_mcp_flex.angle_rad_to_tick(0.5)
    assert state.target == 0.5
    assert state.commanded == 0.5
    assert state.measured == pytest.approx(0.5, abs=1 / index_mcp_flex.ticks_per_radian)
    assert state.target_tick == expected_tick
    assert state.measured_tick == expected_tick
    assert ("goal", 9, expected_tick) in controller.bus.calls


def test_operator_runtime_uses_configured_profile_on_connect() -> None:
    controller = HandController.from_fake()
    runtime = HandOperatorRuntime(
        controller,
        make_default_joint_calibrations(),
        backend="fake",
        dry_run=False,
        profile_acceleration=40,
        profile_velocity=80,
    )

    runtime.connect()

    profile_calls = [call for call in controller.bus.calls if call[0] == "profile"]
    assert len(profile_calls) == 16
    assert ("profile", 9, 40, 80) in profile_calls
    assert controller.bus.motors[9].profile_acceleration == 40
    assert controller.bus.motors[9].profile_velocity == 80


def test_operator_runtime_observe_reads_back_without_writing_targets() -> None:
    controller = HandController.from_fake()
    calibrations = make_default_joint_calibrations()
    runtime = HandOperatorRuntime(
        controller,
        calibrations,
        backend="fake",
        dry_run=False,
        target_velocity_limit=1000.0,
    )

    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.set_target("right_hand.index_mcp_flex.pos", 0.5)
    calls_before = len(controller.bus.calls)
    snapshot = runtime.observe(now=0.0)

    new_calls = controller.bus.calls[calls_before:]
    assert new_calls, "observe() must read back motor state"
    assert all(str(call[0]).startswith(("sync_read", "read")) for call in new_calls)
    state = _joint(snapshot, "right_hand.index_mcp_flex.pos")
    assert state.target == 0.5
    assert state.commanded == 0.0
    assert state.measured == pytest.approx(0.0)

    index_mcp_flex = calibrations[9]
    controller.bus.motors[9].position = index_mcp_flex.angle_rad_to_tick(0.25)
    snapshot = runtime.observe(now=2.0)
    assert _joint(snapshot, "right_hand.index_mcp_flex.pos").measured == pytest.approx(
        0.25, abs=1 / index_mcp_flex.ticks_per_radian
    )


def test_operator_runtime_can_sync_motion_state_to_measured_before_torque() -> None:
    calibrations = make_default_joint_calibrations()
    index_mcp_flex = calibrations[9]
    start_tick = index_mcp_flex.angle_rad_to_tick(0.25)
    controller = HandController.from_fake(positions={9: start_tick})
    runtime = HandOperatorRuntime(
        controller,
        calibrations,
        backend="fake",
        dry_run=False,
        target_velocity_limit=1000.0,
    )

    runtime.connect()
    snapshot = runtime.sync_motion_state_to_measured(now=0.0)

    state = _joint(snapshot, "right_hand.index_mcp_flex.pos")
    assert state.target == pytest.approx(0.25, abs=1 / index_mcp_flex.ticks_per_radian)
    assert state.commanded == pytest.approx(
        0.25, abs=1 / index_mcp_flex.ticks_per_radian
    )
    assert state.measured == pytest.approx(
        0.25, abs=1 / index_mcp_flex.ticks_per_radian
    )
    assert state.measured_tick == start_tick
    assert not any(call[0] == "goal" for call in controller.bus.calls)

    runtime.set_torque_enabled(True)
    snapshot = runtime.step(now=0.1)

    state = _joint(snapshot, "right_hand.index_mcp_flex.pos")
    assert state.commanded == pytest.approx(
        0.25, abs=1 / index_mcp_flex.ticks_per_radian
    )
    assert ("goal", 9, start_tick) in controller.bus.calls


def test_operator_runtime_can_write_real_id_map_while_preserving_urdf_ids() -> None:
    controller = HandController.from_fake(motor_ids=GHAND_RIGHT_MOTOR_IDS)
    calibrations = make_default_joint_calibrations(
        motor_ids=GHAND_RIGHT_MOTOR_IDS,
    )
    runtime = HandOperatorRuntime(
        controller,
        calibrations,
        backend="fake-real-map",
        dry_run=False,
        target_velocity_limit=1000.0,
    )

    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.set_target("right_hand.index_mcp_flex.pos", 0.5)
    snapshot = runtime.step(now=0.0)

    state = _joint(snapshot, "right_hand.index_mcp_flex.pos")
    assert state.dxl_id == 1
    assert state.pose_dxl_id == 9
    index_mcp_flex = calibrations[9]
    assert state.measured == pytest.approx(0.5, abs=1 / index_mcp_flex.ticks_per_radian)
    assert ("goal", 1, index_mcp_flex.angle_rad_to_tick(0.5)) in controller.bus.calls


def test_operator_runtime_dry_run_suppresses_motor_writes() -> None:
    controller = HandController.from_fake()
    runtime = HandOperatorRuntime(
        controller,
        make_default_joint_calibrations(),
        backend="fake",
        dry_run=True,
        target_velocity_limit=1000.0,
    )

    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.set_target("right_hand.index_mcp_flex.pos", 0.5)
    snapshot = runtime.step(now=0.0)

    state = _joint(snapshot, "right_hand.index_mcp_flex.pos")
    assert state.target == 0.5
    assert state.commanded == 0.0
    assert state.measured == 0.0
    assert not any(call[0] == "goal" for call in controller.bus.calls)
    assert not any(call[0] == "torque" for call in controller.bus.calls)


def test_operator_runtime_reads_health_telemetry_at_lower_rate() -> None:
    controller = HandController.from_fake()
    controller.bus.motors[9].present_current_ma = 12.0
    runtime = HandOperatorRuntime(
        controller,
        make_default_joint_calibrations(),
        backend="fake",
        dry_run=True,
        telemetry_hz=1.0,
    )

    runtime.connect()
    controller.bus.calls.clear()

    first = runtime.step(now=0.0)
    second = runtime.step(now=0.1)
    runtime.step(now=0.9)

    assert _count_calls(controller.bus.calls, "sync_read_position") == 3
    assert _count_calls(controller.bus.calls, "sync_read_current") == 1
    assert _count_calls(controller.bus.calls, "sync_read_temperature") == 1
    assert _count_calls(controller.bus.calls, "sync_read_voltage") == 1
    assert _count_calls(controller.bus.calls, "sync_read_error") == 1
    assert _joint(first, "right_hand.index_mcp_flex.pos").present_current_ma == 12.0
    assert _joint(first, "right_hand.index_mcp_flex.pos").temperature_celsius == 31
    assert _joint(second, "right_hand.index_mcp_flex.pos").temperature_celsius == 31

    runtime.step(now=1.0)

    assert _count_calls(controller.bus.calls, "sync_read_position") == 4
    assert _count_calls(controller.bus.calls, "sync_read_current") == 2
    assert _count_calls(controller.bus.calls, "sync_read_temperature") == 2
    assert _count_calls(controller.bus.calls, "sync_read_voltage") == 2
    assert _count_calls(controller.bus.calls, "sync_read_error") == 2


def test_operator_runtime_combines_active_safety_readback() -> None:
    controller = HandController.from_fake()
    calibrations = make_default_joint_calibrations()
    calibration = calibrations[9]
    motor = controller.bus.motors[calibration.dxl_id]
    motor.position = calibration.angle_rad_to_tick(0.25)
    motor.present_current_ma = 12.0
    motor.temperature_celsius = 34
    motor.input_voltage = 11.9
    motor.hardware_error_status = 2
    motor.torque_enabled = True
    runtime = HandOperatorRuntime(
        controller,
        calibrations,
        backend="fake",
        dry_run=True,
        combined_safety_readback=True,
    )

    runtime.connect()
    controller.bus.calls.clear()
    snapshot = runtime.observe(now=0.0)

    assert controller.bus.calls == [("sync_read_safety", runtime.dxl_ids)]
    state = _joint(snapshot, calibration.joint_name)
    assert state.measured == pytest.approx(0.25, abs=1 / calibration.ticks_per_radian)
    assert state.present_current_ma == 12.0
    assert state.temperature_celsius == 34
    assert state.input_voltage == 11.9
    assert state.hardware_error_status == 2
    assert state.torque_enabled is True


def test_operator_runtime_can_write_targets_without_a_second_readback() -> None:
    controller = HandController.from_fake()
    calibrations = make_default_joint_calibrations()
    calibration = calibrations[9]
    runtime = HandOperatorRuntime(
        controller,
        calibrations,
        backend="fake",
        dry_run=False,
        target_velocity_limit=1000.0,
        combined_safety_readback=True,
    )

    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.observe(now=0.0)
    controller.bus.calls.clear()
    runtime.set_target(calibration.joint_name, 0.25)
    snapshot = runtime.write_targets(now=0.0)

    assert any(call[0] == "sync_goal" for call in controller.bus.calls)
    assert not any(call[0] == "sync_read_safety" for call in controller.bus.calls)
    state = _joint(snapshot, calibration.joint_name)
    assert state.commanded == pytest.approx(0.25, abs=1 / calibration.ticks_per_radian)
    assert state.measured == 0.0


def test_operator_runtime_can_throttle_position_readback_without_throttling_writes() -> (
    None
):
    controller = HandController.from_fake()
    runtime = HandOperatorRuntime(
        controller,
        make_default_joint_calibrations(),
        backend="fake",
        dry_run=False,
        target_velocity_limit=1000.0,
        readback_hz=5.0,
        telemetry_hz=0.0,
    )

    runtime.connect()
    runtime.set_torque_enabled(True)
    controller.bus.calls.clear()

    runtime.set_target("right_hand.index_mcp_flex.pos", 0.5)
    runtime.step(now=0.0)
    runtime.step(now=0.1)
    runtime.step(now=0.19)
    runtime.step(now=0.2)

    assert _count_calls(controller.bus.calls, "sync_goal") == 4
    assert _count_calls(controller.bus.calls, "sync_read_position") == 2


def test_operator_runtime_stores_current_limits_without_writing_hardware_current() -> (
    None
):
    controller = HandController.from_fake()
    runtime = HandOperatorRuntime(
        controller,
        make_default_joint_calibrations(),
        backend="fake",
        dry_run=False,
    )

    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.set_current_limit("right_hand.index_mcp_flex.pos", 90.0)
    snapshot = runtime.step(now=0.0)

    assert _joint(snapshot, "right_hand.index_mcp_flex.pos").current_limit_ma == 90.0
    assert not any(call[0] == "goal_current" for call in controller.bus.calls)

    with pytest.raises(ValueError, match="non-negative"):
        runtime.set_current_limit("right_hand.index_mcp_flex.pos", -1.0)


def test_operator_runtime_writes_current_limits_in_current_position_mode() -> None:
    controller = HandController.from_fake()
    runtime = HandOperatorRuntime(
        controller,
        make_default_joint_calibrations(),
        backend="fake",
        dry_run=False,
        target_velocity_limit=1000.0,
        control_mode=OPERATOR_CONTROL_MODE_CURRENT_POSITION,
        default_current_limit_ma=120.0,
        max_current_limit_ma=100.0,
    )

    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.set_current_limit("right_hand.index_mcp_flex.pos", 90.0)
    runtime.set_current_limit("right_hand.index_pip_flex.pos", 120.0)
    runtime.set_target("right_hand.index_mcp_flex.pos", 0.5)
    snapshot = runtime.step(now=0.0)

    assert (
        controller.bus.motors[9].operating_mode == OPERATING_MODE_CURRENT_BASED_POSITION
    )
    assert _joint(snapshot, "right_hand.index_mcp_flex.pos").current_limit_ma == 90.0
    assert _joint(snapshot, "right_hand.index_pip_flex.pos").current_limit_ma == 100.0
    assert (
        "sync_goal_current",
        {dxl_id: 100.0 for dxl_id in range(16)},
    ) in controller.bus.calls
    assert any(
        call[0] == "sync_goal_current" and call[1][9] == 90.0 and call[1][10] == 100.0
        for call in controller.bus.calls
    )
    assert controller.bus.motors[9].goal_current_ma == 90.0
    assert controller.bus.motors[10].goal_current_ma == 100.0
    assert any(call[0] == "sync_goal" for call in controller.bus.calls)

    goal_current_write_count = _count_calls(controller.bus.calls, "sync_goal_current")
    runtime.step(now=0.1)
    assert (
        _count_calls(controller.bus.calls, "sync_goal_current")
        == goal_current_write_count
    )


def test_operator_runtime_emergency_stop_latches_and_disables_torque() -> None:
    controller = HandController.from_fake()
    runtime = HandOperatorRuntime(
        controller,
        make_default_joint_calibrations(),
        backend="fake",
        dry_run=False,
    )

    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.emergency_stop_now()
    runtime.set_target("right_hand.index_mcp_flex.pos", 0.5)
    snapshot = runtime.step(now=0.0)

    state = _joint(snapshot, "right_hand.index_mcp_flex.pos")
    assert snapshot.emergency_stop is True
    assert snapshot.last_error is not None
    assert state.target == 0.0
    assert state.torque_enabled is False
    assert ("torque", 9, False) in controller.bus.calls
    assert not any(call == ("goal", 9, 2304) for call in controller.bus.calls)


def test_operator_runtime_disconnect_closes_even_if_torque_off_fails() -> None:
    controller = HandController.from_fake()
    runtime = HandOperatorRuntime(
        controller,
        make_default_joint_calibrations(),
        backend="fake",
        dry_run=False,
    )
    original_set_torque_enabled = controller.bus.set_torque_enabled

    runtime.connect()
    runtime.set_torque_enabled(True)

    def failing_set_torque_enabled(dxl_id: int, enabled: bool) -> None:
        if not enabled:
            raise RuntimeError("port is in use")
        original_set_torque_enabled(dxl_id, enabled)

    controller.bus.set_torque_enabled = failing_set_torque_enabled

    runtime.disconnect()

    assert controller.is_connected is False
    assert runtime.last_error == "port is in use"


def _joint(snapshot, joint_name):
    return next(state for state in snapshot.joints if state.joint_name == joint_name)


def _count_calls(calls, name):
    return sum(1 for call in calls if call[0] == name)


def test_operator_start_and_resume_use_measured_pose_instead_of_stale_targets() -> None:
    calibrations = make_default_joint_calibrations()
    controller = HandController.from_fake(positions={9: 2200})
    runtime = HandOperatorRuntime(
        controller, calibrations, backend="fake", dry_run=False
    )
    runtime.set_target(calibrations[9].joint_name, 1.0)
    snapshot = runtime.start()
    assert snapshot.connected
    assert all(joint.torque_enabled for joint in snapshot.joints)
    assert controller.bus.motors[9].goal_position == 2200
    assert controller.bus.motors[9].profile_acceleration == 80
    assert controller.bus.motors[9].profile_velocity == 120
    runtime.emergency_stop_now()
    controller.bus.motors[9].position = 2300
    snapshot = runtime.start()
    assert not snapshot.emergency_stop
    assert controller.bus.motors[9].goal_position == 2300
    assert sum(call[0] == "connect" for call in controller.bus.calls) == 1
    runtime.disconnect()


def test_operator_start_read_failure_does_not_enable_torque(monkeypatch) -> None:
    controller = HandController.from_fake()
    runtime = HandOperatorRuntime(
        controller, make_default_joint_calibrations(), backend="fake", dry_run=False
    )

    def fail_read(ids):
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr(controller, "read_motor_positions", fail_read)
    with pytest.raises(RuntimeError, match="encoder unavailable"):
        runtime.start()
    assert not runtime.connected
    assert not any(call[0] == "torque" and call[2] for call in controller.bus.calls)


def test_operator_runtime_accepts_zero_profile_as_no_limit() -> None:
    controller = HandController.from_fake()
    runtime = HandOperatorRuntime(
        controller,
        make_default_joint_calibrations(),
        backend="fake",
        dry_run=False,
        profile_acceleration=0,
        profile_velocity=0,
    )

    runtime.connect()

    assert ("profile", 9, 0, 0) in controller.bus.calls
    with pytest.raises(ValueError):
        HandOperatorRuntime(
            controller,
            make_default_joint_calibrations(),
            backend="fake",
            dry_run=False,
            profile_velocity=-1,
        )
