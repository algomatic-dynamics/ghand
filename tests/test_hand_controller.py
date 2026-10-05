import pytest

from ghand.hardware import (
    FakeBus,
    HandController,
    MotorEncoderState,
    MotorSnapshot,
    MotorTelemetry,
    MotorTestResult,
    OPERATING_MODE_CURRENT_BASED_POSITION,
    OPERATING_MODE_POSITION,
)


def test_hand_controller_connect_delegates_to_bus() -> None:
    bus = FakeBus()
    controller = HandController(bus)

    controller.connect()
    assert controller.is_connected is True

    controller.disconnect()
    assert controller.is_connected is False
    assert bus.calls == [("connect",), ("disconnect",)]


def test_hand_controller_can_be_created_with_fake_bus() -> None:
    controller = HandController.from_fake(motor_ids=(2,), positions={2: 1024})

    assert isinstance(controller.bus, FakeBus)
    assert controller.bus.read_present_position(2) == 1024


def test_read_motor_encoder_state_does_not_write_to_bus() -> None:
    bus = FakeBus(motor_ids=(3,), positions={3: 2050})
    controller = HandController(bus)

    state = controller.read_motor_encoder_state(3)

    assert state == MotorEncoderState(dxl_id=3, model_number=1234, position=2050)
    assert bus.calls == [("ping", 3), ("read_position", 3)]


def test_hand_controller_bulk_state_and_target_helpers() -> None:
    bus = FakeBus(motor_ids=(2, 3), positions={2: 2048, 3: 2048})
    controller = HandController(bus)

    controller.configure_position_control(
        (2, 3),
        profile_acceleration=10,
        profile_velocity=20,
    )
    controller.set_torque_enabled_many((2, 3), True)
    controller.write_goal_positions({2: 2100, 3: 1980})
    snapshots = controller.read_motor_snapshots((2, 3))

    assert snapshots == (
        MotorSnapshot(
            dxl_id=2,
            position=2100,
            present_current_ma=0.0,
            temperature_celsius=31,
            input_voltage=12.0,
            hardware_error_status=0,
        ),
        MotorSnapshot(
            dxl_id=3,
            position=1980,
            present_current_ma=0.0,
            temperature_celsius=31,
            input_voltage=12.0,
            hardware_error_status=0,
        ),
    )
    assert ("mode", 2, OPERATING_MODE_POSITION) in bus.calls
    assert ("profile", 3, 10, 20) in bus.calls
    assert ("sync_goal", {2: 2100, 3: 1980}) in bus.calls
    assert ("goal", 2, 2100) in bus.calls
    assert ("sync_read_position", (2, 3)) in bus.calls
    assert ("sync_read_current", (2, 3)) in bus.calls
    assert ("sync_read_temperature", (2, 3)) in bus.calls
    assert ("sync_read_voltage", (2, 3)) in bus.calls
    assert ("sync_read_error", (2, 3)) in bus.calls


def test_hand_controller_can_configure_and_write_current_based_position_mode() -> None:
    bus = FakeBus(motor_ids=(2, 3))
    controller = HandController(bus)

    controller.configure_current_based_position_control(
        (2, 3),
        profile_acceleration=10,
        profile_velocity=20,
    )
    controller.write_goal_currents_ma({2: 90.0, 3: 75.0})

    assert ("mode", 2, OPERATING_MODE_CURRENT_BASED_POSITION) in bus.calls
    assert ("mode", 3, OPERATING_MODE_CURRENT_BASED_POSITION) in bus.calls
    assert ("profile", 2, 10, 20) in bus.calls
    assert ("sync_goal_current", {2: 90.0, 3: 75.0}) in bus.calls
    assert bus.motors[2].goal_current_ma == 90.0
    assert bus.motors[3].goal_current_ma == 75.0


def test_hand_controller_can_read_positions_without_health_telemetry() -> None:
    bus = FakeBus(motor_ids=(2, 3), positions={2: 2100, 3: 1980})
    controller = HandController(bus)

    positions = controller.read_motor_positions((2, 3))

    assert positions == {2: 2100, 3: 1980}
    assert bus.calls == [("sync_read_position", (2, 3))]


def test_hand_controller_can_read_signed_homing_offsets() -> None:
    bus = FakeBus(motor_ids=(2, 3), homing_offsets={2: -12, 3: 0})
    controller = HandController(bus)

    offsets = controller.read_homing_offsets((2, 3))

    assert offsets == {2: -12, 3: 0}
    assert bus.calls == [("sync_read_homing_offset", (2, 3))]


def test_hand_controller_can_read_firmware_versions() -> None:
    bus = FakeBus(motor_ids=(2, 3), firmware_version=38)
    bus.motors[3].firmware_version = 40
    controller = HandController(bus)

    versions = controller.read_firmware_versions((2, 3))

    assert versions == {2: 38, 3: 40}
    assert bus.calls == [("sync_read_firmware", (2, 3))]


def test_hand_controller_can_read_current_without_positions() -> None:
    bus = FakeBus(motor_ids=(2, 3))
    bus.motors[2].present_current_ma = 12.0
    bus.motors[3].present_current_ma = -13.0
    controller = HandController(bus)

    currents = controller.read_motor_currents_ma((2, 3))

    assert currents == {2: 12.0, 3: -13.0}
    assert bus.calls == [("sync_read_current", (2, 3))]


def test_hand_controller_can_read_motion_and_torque_state() -> None:
    bus = FakeBus(motor_ids=(2, 3), positions={2: 2100, 3: 1980})
    bus.motors[2].present_current_ma = 12.0
    bus.motors[3].torque_enabled = True
    controller = HandController(bus)

    samples = controller.read_motion_samples((3, 2))
    torque_enabled = controller.read_torque_enabled_many((2, 3))

    assert tuple(sample.dxl_id for sample in samples) == (3, 2)
    assert samples[0].present_position_tick == 1980
    assert samples[1].present_current_ma == 12.0
    assert torque_enabled == {2: False, 3: True}
    assert bus.calls == [
        ("sync_read_motion", (3, 2)),
        ("sync_read_torque", (2, 3)),
    ]


def test_hand_controller_can_read_combined_safety_state() -> None:
    bus = FakeBus(motor_ids=(2, 3), positions={2: 2100, 3: 1980})
    bus.motors[2].present_current_ma = 12.0
    bus.motors[3].torque_enabled = True
    bus.motors[3].hardware_error_status = 2
    bus.motors[3].bus_watchdog = 10
    controller = HandController(bus)

    samples = controller.read_safety_samples((3, 2))

    assert tuple(sample.dxl_id for sample in samples) == (3, 2)
    assert samples[0].torque_enabled is True
    assert samples[0].present_position_tick == 1980
    assert samples[0].hardware_error_status == 2
    assert samples[0].bus_watchdog == 10
    assert samples[1].present_current_ma == 12.0
    assert bus.calls == [("sync_read_safety", (3, 2))]


def test_hand_controller_can_write_and_read_bus_watchdog() -> None:
    bus = FakeBus(motor_ids=(2, 3))
    controller = HandController(bus)

    controller.write_bus_watchdogs((2, 3), 10)
    values = controller.read_bus_watchdogs((2, 3))

    assert values == {2: 10, 3: 10}
    assert bus.calls == [
        ("watchdog", 2, 10),
        ("watchdog", 3, 10),
        ("sync_read_watchdog", (2, 3)),
    ]


def test_hand_controller_can_read_health_telemetry_without_positions() -> None:
    bus = FakeBus(motor_ids=(2, 3), positions={2: 2100, 3: 1980})
    bus.motors[2].present_current_ma = 12.0
    bus.motors[3].present_current_ma = -13.0
    bus.motors[2].temperature_celsius = 30
    bus.motors[3].temperature_celsius = 31
    bus.motors[2].input_voltage = 12.0
    bus.motors[3].input_voltage = 12.1
    bus.motors[3].hardware_error_status = 2
    controller = HandController(bus)

    telemetry = controller.read_motor_telemetry((2, 3))

    assert telemetry == (
        MotorTelemetry(
            dxl_id=2,
            present_current_ma=12.0,
            temperature_celsius=30,
            input_voltage=12.0,
            hardware_error_status=0,
        ),
        MotorTelemetry(
            dxl_id=3,
            present_current_ma=-13.0,
            temperature_celsius=31,
            input_voltage=12.1,
            hardware_error_status=2,
        ),
    )
    assert bus.calls == [
        ("sync_read_current", (2, 3)),
        ("sync_read_temperature", (2, 3)),
        ("sync_read_voltage", (2, 3)),
        ("sync_read_error", (2, 3)),
    ]


def test_hand_controller_rejects_out_of_range_bulk_goal_position() -> None:
    controller = HandController(FakeBus(motor_ids=(1,)))

    with pytest.raises(ValueError, match="position-control range"):
        controller.write_goal_positions({1: 4096})


def test_hand_controller_rejects_invalid_goal_current() -> None:
    controller = HandController(FakeBus(motor_ids=(1,)))

    with pytest.raises(ValueError, match="non-negative"):
        controller.write_goal_currents_ma({1: -1.0})

    with pytest.raises(ValueError, match="finite"):
        controller.write_goal_currents_ma({1: float("nan")})


def test_test_motor_runs_safe_position_mode_sequence() -> None:
    bus = FakeBus(positions={3: 2048})
    bus.motors[3].temperature_celsius = 31
    bus.motors[3].input_voltage = 12.2
    sleep_calls = []
    controller = HandController(bus, sleep=sleep_calls.append)

    result = controller.test_motor(
        3,
        delta_ticks=64,
        profile_acceleration=20,
        profile_velocity=40,
        move_seconds=0.1,
        return_seconds=0.2,
    )

    assert result == MotorTestResult(
        dxl_id=3,
        model_number=1234,
        start_position=2048,
        target_position=2112,
        returned_position=2112,
        final_position=2048,
        applied_delta_ticks=64,
        temperature_celsius=31,
        input_voltage=12.2,
        hardware_error_status=0,
        torque_enabled_after=False,
    )
    assert sleep_calls == [0.1, 0.2]
    assert bus.calls == [
        ("ping", 3),
        ("torque", 3, False),
        ("mode", 3, OPERATING_MODE_POSITION),
        ("profile", 3, 20, 40),
        ("read_position", 3),
        ("goal", 3, 2048),
        ("torque", 3, True),
        ("goal", 3, 2112),
        ("read_position", 3),
        ("goal", 3, 2048),
        ("read_position", 3),
        ("torque", 3, False),
        ("read_temperature", 3),
        ("read_voltage", 3),
        ("read_error", 3),
    ]
    assert bus.motors[3].position == 2048
    assert bus.motors[3].goal_position == 2048
    assert bus.motors[3].torque_enabled is False


def test_test_motor_flips_delta_at_position_boundary() -> None:
    bus = FakeBus(positions={4: 4070})
    controller = HandController(bus, sleep=lambda _: None)

    result = controller.test_motor(4, delta_ticks=64)

    assert result.start_position == 4070
    assert result.target_position == 4006
    assert result.applied_delta_ticks == -64
    assert ("goal", 4, 4006) in bus.calls


def test_test_motor_can_keep_torque_enabled() -> None:
    bus = FakeBus(positions={1: 2048})
    controller = HandController(bus, sleep=lambda _: None)

    result = controller.test_motor(1, keep_torque=True)

    assert result.torque_enabled_after is True
    assert bus.calls.count(("torque", 1, False)) == 1
    assert ("torque", 1, True) in bus.calls
    assert bus.motors[1].torque_enabled is True


def test_test_motor_validates_small_safe_motion_arguments() -> None:
    controller = HandController(FakeBus())

    invalid_kwargs = (
        {"dxl_id": -1},
        {"dxl_id": 253},
        {"delta_ticks": 0},
        {"delta_ticks": 513},
        {"max_delta_ticks": 0},
        {"profile_acceleration": -1},
        {"profile_velocity": 0},
        {"move_seconds": -0.1},
        {"return_seconds": -0.1},
    )

    for kwargs in invalid_kwargs:
        dxl_id = kwargs.pop("dxl_id", 1)
        with pytest.raises(ValueError):
            controller.test_motor(dxl_id, **kwargs)
