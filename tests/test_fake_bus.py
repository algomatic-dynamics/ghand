import pytest

from ghand.hardware import (
    FakeBus,
    FakeMotorState,
    OPERATING_MODE_CURRENT_BASED_POSITION,
)
from ghand.hardware.dynamixel_bus import DynamixelBusError, DynamixelPingResult


def test_fake_bus_default_motors_support_ad_ghand_id_range() -> None:
    bus = FakeBus()

    assert tuple(bus.motors) == tuple(range(16))
    assert bus.ping(15) == DynamixelPingResult(dxl_id=15, model_number=1234)


def test_fake_bus_goal_write_moves_only_when_torque_is_enabled() -> None:
    bus = FakeBus(positions={3: 2048})

    bus.write_goal_position(3, 2112)
    assert bus.motors[3].goal_position == 2112
    assert bus.read_present_position(3) == 2048

    bus.set_torque_enabled(3, True)
    assert bus.read_present_position(3) == 2112

    bus.write_goal_position(3, 2200)
    assert bus.read_present_position(3) == 2200


def test_fake_bus_records_profile_mode_and_sensor_values() -> None:
    bus = FakeBus(motor_ids=(7,), positions={7: 1024})
    bus.motors[7].present_current_ma = -12.0
    bus.motors[7].temperature_celsius = 35
    bus.motors[7].input_voltage = 11.8
    bus.motors[7].hardware_error_status = 2

    bus.write_operating_mode(7, OPERATING_MODE_CURRENT_BASED_POSITION)
    bus.write_profile(7, acceleration=12, velocity=34)
    bus.write_goal_current_ma(7, 90.0)

    motor = bus.motors[7]
    assert motor.operating_mode == OPERATING_MODE_CURRENT_BASED_POSITION
    assert motor.profile_acceleration == 12
    assert motor.profile_velocity == 34
    assert motor.goal_current_ma == 90.0
    assert bus.read_present_current_ma(7) == -12.0
    assert bus.read_present_temperature(7) == 35
    assert bus.read_present_input_voltage(7) == 11.8
    assert bus.read_hardware_error_status(7) == 2


def test_fake_bus_supports_sync_goal_write_and_sync_reads() -> None:
    bus = FakeBus(motor_ids=(2, 3), positions={2: 2048, 3: 1024})
    bus.motors[2].temperature_celsius = 30
    bus.motors[3].temperature_celsius = 31
    bus.motors[2].present_current_ma = 12.0
    bus.motors[3].present_current_ma = -13.0
    bus.motors[2].input_voltage = 12.0
    bus.motors[3].input_voltage = 12.1
    bus.motors[3].hardware_error_status = 2

    bus.set_torque_enabled(2, True)
    bus.set_torque_enabled(3, True)
    bus.sync_write_goal_positions({2: 2100, 3: 1980})
    bus.sync_write_goal_currents_ma({2: 90.0, 3: 75.0})

    assert bus.sync_read_present_positions((2, 3)) == {2: 2100, 3: 1980}
    assert bus.motors[2].goal_current_ma == 90.0
    assert bus.motors[3].goal_current_ma == 75.0
    assert bus.sync_read_present_currents_ma((2, 3)) == {2: 12.0, 3: -13.0}
    assert bus.sync_read_present_temperatures((2, 3)) == {2: 30, 3: 31}
    assert bus.sync_read_present_input_voltages((2, 3)) == {2: 12.0, 3: 12.1}
    assert bus.sync_read_hardware_error_statuses((2, 3)) == {2: 0, 3: 2}
    assert ("sync_goal", {2: 2100, 3: 1980}) in bus.calls
    assert ("sync_goal_current", {2: 90.0, 3: 75.0}) in bus.calls
    assert ("sync_read_position", (2, 3)) in bus.calls


def test_fake_bus_rejects_unknown_or_invalid_ids() -> None:
    bus = FakeBus(motor_ids=(1,))

    with pytest.raises(DynamixelBusError, match="no motor with ID 2"):
        bus.ping(2)

    with pytest.raises(ValueError):
        bus.ping(253)


def test_fake_motor_state_normalizes_positions() -> None:
    state = FakeMotorState(position=4097, goal_position=-1)

    assert state.position == 1
    assert state.goal_position == 4095
