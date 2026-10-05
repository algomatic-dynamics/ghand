"""In-memory Dynamixel-like bus used before physical hardware is available."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from ghand.hardware.dynamixel_bus import (
    CURRENT_MILLIAMPS_PER_UNIT,
    PRESENT_PWM_PERCENT_PER_UNIT,
    PRESENT_VELOCITY_RPM_PER_UNIT,
    DynamixelMotionSample,
    DynamixelSafetySample,
    DynamixelBusError,
    DynamixelPingResult,
    OPERATING_MODE_POSITION,
    BUS_WATCHDOG_MAX_VALUE,
    BUS_WATCHDOG_MIN_VALUE,
    normalize_position_tick,
)


DEFAULT_FAKE_MOTOR_IDS: Final[tuple[int, ...]] = tuple(range(16))
DEFAULT_FAKE_MODEL_NUMBER: Final[int] = 1234
DEFAULT_FAKE_POSITION: Final[int] = 2048
DEFAULT_FAKE_GOAL_CURRENT_MA: Final[float] = 0.0
DEFAULT_FAKE_PRESENT_CURRENT_MA: Final[float] = 0.0
DEFAULT_FAKE_TEMPERATURE_CELSIUS: Final[int] = 31
DEFAULT_FAKE_INPUT_VOLTAGE: Final[float] = 12.0
DEFAULT_FAKE_HOMING_OFFSET: Final[int] = 0
DEFAULT_FAKE_FIRMWARE_VERSION: Final[int] = 38


@dataclass
class FakeMotorState:
    """Mutable fake state for one Dynamixel motor."""

    position: int = DEFAULT_FAKE_POSITION
    goal_position: int = DEFAULT_FAKE_POSITION
    goal_current_ma: float = DEFAULT_FAKE_GOAL_CURRENT_MA
    torque_enabled: bool = False
    operating_mode: int = OPERATING_MODE_POSITION
    profile_acceleration: int = 0
    profile_velocity: int = 0
    model_number: int = DEFAULT_FAKE_MODEL_NUMBER
    present_current_ma: float = DEFAULT_FAKE_PRESENT_CURRENT_MA
    temperature_celsius: int = DEFAULT_FAKE_TEMPERATURE_CELSIUS
    input_voltage: float = DEFAULT_FAKE_INPUT_VOLTAGE
    hardware_error_status: int = 0
    homing_offset: int = DEFAULT_FAKE_HOMING_OFFSET
    realtime_tick_ms: int = 0
    moving: bool = False
    moving_status: int = 0
    present_pwm_raw: int = 0
    present_velocity_raw: int = 0
    bus_watchdog: int = 0
    firmware_version: int = DEFAULT_FAKE_FIRMWARE_VERSION

    def __post_init__(self) -> None:
        self.position = normalize_position_tick(self.position)
        self.goal_position = normalize_position_tick(self.goal_position)


class FakeBus:
    """DynamixelBus-compatible fake bus for no-hardware development and tests."""

    def __init__(
        self,
        *,
        motor_ids: Iterable[int] = DEFAULT_FAKE_MOTOR_IDS,
        positions: Mapping[int, int] | None = None,
        homing_offsets: Mapping[int, int] | None = None,
        model_number: int = DEFAULT_FAKE_MODEL_NUMBER,
        firmware_version: int = DEFAULT_FAKE_FIRMWARE_VERSION,
    ) -> None:
        self.is_connected = False
        self.calls: list[tuple[object, ...]] = []
        resolved_positions = dict(positions or {})
        resolved_homing_offsets = dict(homing_offsets or {})
        ids = set(motor_ids) | set(resolved_positions) | set(resolved_homing_offsets)
        self.motors = {
            dxl_id: FakeMotorState(
                position=resolved_positions.get(dxl_id, DEFAULT_FAKE_POSITION),
                goal_position=resolved_positions.get(dxl_id, DEFAULT_FAKE_POSITION),
                model_number=model_number,
                homing_offset=resolved_homing_offsets.get(
                    dxl_id, DEFAULT_FAKE_HOMING_OFFSET
                ),
                firmware_version=firmware_version,
            )
            for dxl_id in sorted(ids)
        }
        for dxl_id in self.motors:
            _validate_dxl_id(dxl_id)

    def connect(self) -> None:
        self.calls.append(("connect",))
        self.is_connected = True

    def disconnect(self) -> None:
        self.calls.append(("disconnect",))
        self.is_connected = False

    def ping(self, dxl_id: int) -> DynamixelPingResult:
        self.calls.append(("ping", dxl_id))
        motor = self._motor(dxl_id)
        return DynamixelPingResult(dxl_id=dxl_id, model_number=motor.model_number)

    def set_torque_enabled(self, dxl_id: int, enabled: bool) -> None:
        self.calls.append(("torque", dxl_id, enabled))
        motor = self._motor(dxl_id)
        motor.torque_enabled = enabled
        if enabled:
            motor.position = motor.goal_position

    def write_operating_mode(self, dxl_id: int, mode: int) -> None:
        self.calls.append(("mode", dxl_id, mode))
        self._motor(dxl_id).operating_mode = mode

    def write_profile(self, dxl_id: int, *, acceleration: int, velocity: int) -> None:
        self.calls.append(("profile", dxl_id, acceleration, velocity))
        motor = self._motor(dxl_id)
        motor.profile_acceleration = acceleration
        motor.profile_velocity = velocity

    def write_bus_watchdog(self, dxl_id: int, value: int) -> None:
        if not BUS_WATCHDOG_MIN_VALUE <= value <= BUS_WATCHDOG_MAX_VALUE:
            raise ValueError(
                "bus watchdog must be in the Dynamixel range "
                f"{BUS_WATCHDOG_MIN_VALUE}..{BUS_WATCHDOG_MAX_VALUE}."
            )
        self.calls.append(("watchdog", dxl_id, value))
        self._motor(dxl_id).bus_watchdog = value

    def write_goal_position(self, dxl_id: int, position: int) -> None:
        normalized_position = normalize_position_tick(position)
        self.calls.append(("goal", dxl_id, normalized_position))
        motor = self._motor(dxl_id)
        motor.goal_position = normalized_position
        if motor.torque_enabled:
            motor.position = normalized_position

    def write_goal_current_ma(self, dxl_id: int, current_ma: float) -> None:
        current = float(current_ma)
        self.calls.append(("goal_current", dxl_id, current))
        self._motor(dxl_id).goal_current_ma = current

    def sync_write_goal_positions(self, positions: Mapping[int, int]) -> None:
        if not positions:
            return
        normalized_positions = {
            int(dxl_id): normalize_position_tick(position)
            for dxl_id, position in positions.items()
        }
        self.calls.append(("sync_goal", normalized_positions))
        for dxl_id, position in normalized_positions.items():
            self.write_goal_position(dxl_id, position)

    def sync_write_goal_currents_ma(self, currents_ma: Mapping[int, float]) -> None:
        if not currents_ma:
            return
        currents = {
            int(dxl_id): float(current_ma) for dxl_id, current_ma in currents_ma.items()
        }
        self.calls.append(("sync_goal_current", currents))
        for dxl_id, current_ma in currents.items():
            self.write_goal_current_ma(dxl_id, current_ma)

    def read_present_position(self, dxl_id: int) -> int:
        self.calls.append(("read_position", dxl_id))
        return self._motor(dxl_id).position

    def read_firmware_version(self, dxl_id: int) -> int:
        self.calls.append(("read_firmware", dxl_id))
        return self._motor(dxl_id).firmware_version

    def read_present_current_ma(self, dxl_id: int) -> float:
        self.calls.append(("read_current", dxl_id))
        return self._motor(dxl_id).present_current_ma

    def read_present_temperature(self, dxl_id: int) -> int:
        self.calls.append(("read_temperature", dxl_id))
        return self._motor(dxl_id).temperature_celsius

    def read_present_input_voltage(self, dxl_id: int) -> float:
        self.calls.append(("read_voltage", dxl_id))
        return self._motor(dxl_id).input_voltage

    def read_hardware_error_status(self, dxl_id: int) -> int:
        self.calls.append(("read_error", dxl_id))
        return self._motor(dxl_id).hardware_error_status

    def sync_read_present_positions(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_position", ids))
        return {dxl_id: self._motor(dxl_id).position for dxl_id in ids}

    def sync_read_homing_offsets(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_homing_offset", ids))
        return {dxl_id: self._motor(dxl_id).homing_offset for dxl_id in ids}

    def sync_read_firmware_versions(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_firmware", ids))
        return {dxl_id: self._motor(dxl_id).firmware_version for dxl_id in ids}

    def sync_read_present_currents_ma(self, dxl_ids: Iterable[int]) -> dict[int, float]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_current", ids))
        return {dxl_id: self._motor(dxl_id).present_current_ma for dxl_id in ids}

    def sync_read_present_temperatures(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_temperature", ids))
        return {dxl_id: self._motor(dxl_id).temperature_celsius for dxl_id in ids}

    def sync_read_present_input_voltages(
        self, dxl_ids: Iterable[int]
    ) -> dict[int, float]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_voltage", ids))
        return {dxl_id: self._motor(dxl_id).input_voltage for dxl_id in ids}

    def sync_read_hardware_error_statuses(
        self, dxl_ids: Iterable[int]
    ) -> dict[int, int]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_error", ids))
        return {dxl_id: self._motor(dxl_id).hardware_error_status for dxl_id in ids}

    def sync_read_torque_enabled(self, dxl_ids: Iterable[int]) -> dict[int, bool]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_torque", ids))
        return {dxl_id: self._motor(dxl_id).torque_enabled for dxl_id in ids}

    def sync_read_bus_watchdogs(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_watchdog", ids))
        return {dxl_id: self._motor(dxl_id).bus_watchdog for dxl_id in ids}

    def sync_read_motion_samples(
        self,
        dxl_ids: Iterable[int],
    ) -> dict[int, DynamixelMotionSample]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_motion", ids))
        samples: dict[int, DynamixelMotionSample] = {}
        for dxl_id in ids:
            motor = self._motor(dxl_id)
            current_raw = int(
                round(motor.present_current_ma / CURRENT_MILLIAMPS_PER_UNIT)
            )
            samples[dxl_id] = DynamixelMotionSample(
                dxl_id=dxl_id,
                realtime_tick_ms=motor.realtime_tick_ms,
                moving=motor.moving,
                moving_status=motor.moving_status,
                present_pwm_raw=motor.present_pwm_raw,
                present_pwm_percent=(
                    motor.present_pwm_raw * PRESENT_PWM_PERCENT_PER_UNIT
                ),
                present_current_raw=current_raw,
                present_current_ma=motor.present_current_ma,
                present_velocity_raw=motor.present_velocity_raw,
                present_velocity_rad_s=(
                    motor.present_velocity_raw
                    * PRESENT_VELOCITY_RPM_PER_UNIT
                    * 2.0
                    * math.pi
                    / 60.0
                ),
                present_position_tick=motor.position,
            )
        return samples

    def sync_read_safety_samples(
        self,
        dxl_ids: Iterable[int],
    ) -> dict[int, DynamixelSafetySample]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_safety", ids))
        samples: dict[int, DynamixelSafetySample] = {}
        for dxl_id in ids:
            motor = self._motor(dxl_id)
            samples[dxl_id] = DynamixelSafetySample(
                dxl_id=dxl_id,
                torque_enabled=motor.torque_enabled,
                hardware_error_status=motor.hardware_error_status,
                bus_watchdog=motor.bus_watchdog,
                present_current_raw=int(
                    round(motor.present_current_ma / CURRENT_MILLIAMPS_PER_UNIT)
                ),
                present_current_ma=motor.present_current_ma,
                present_position_tick=motor.position,
                input_voltage=motor.input_voltage,
                temperature_celsius=motor.temperature_celsius,
            )
        return samples

    def _motor(self, dxl_id: int) -> FakeMotorState:
        _validate_dxl_id(dxl_id)
        try:
            return self.motors[dxl_id]
        except KeyError as exc:
            raise DynamixelBusError(f"FakeBus has no motor with ID {dxl_id}.") from exc


def _validate_dxl_id(dxl_id: int) -> None:
    if not 0 <= dxl_id <= 252:
        raise ValueError("dxl_id must be in the Dynamixel ID range 0..252.")
