"""Robot-hand controller facade."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from ghand.hardware.dynamixel_bus import (
    DEFAULT_BAUDRATE,
    OPERATING_MODE_CURRENT_BASED_POSITION,
    OPERATING_MODE_POSITION,
    POSITION_CONTROL_MAX_TICK,
    POSITION_CONTROL_MIN_TICK,
    DynamixelMotionSample,
    DynamixelSafetySample,
    DynamixelBus,
    DynamixelPingResult,
    normalize_position_tick,
)


DEFAULT_MOTOR_TEST_DELTA_TICKS = 64
DEFAULT_MOTOR_TEST_MAX_DELTA_TICKS = 512
DEFAULT_MOTOR_TEST_PROFILE_ACCELERATION = 20
DEFAULT_MOTOR_TEST_PROFILE_VELOCITY = 40
DEFAULT_MOTOR_TEST_MOVE_SECONDS = 0.5
DEFAULT_MOTOR_TEST_RETURN_SECONDS = 0.5


class MotorBus(Protocol):
    """Subset of bus operations required by HandController."""

    is_connected: bool

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def ping(self, dxl_id: int) -> DynamixelPingResult: ...

    def set_torque_enabled(self, dxl_id: int, enabled: bool) -> None: ...

    def write_operating_mode(self, dxl_id: int, mode: int) -> None: ...

    def write_profile(
        self, dxl_id: int, *, acceleration: int, velocity: int
    ) -> None: ...

    def write_goal_position(self, dxl_id: int, position: int) -> None: ...

    def sync_write_goal_positions(self, positions: Mapping[int, int]) -> None: ...

    def write_goal_current_ma(self, dxl_id: int, current_ma: float) -> None: ...

    def sync_write_goal_currents_ma(self, currents_ma: Mapping[int, float]) -> None: ...

    def read_present_position(self, dxl_id: int) -> int: ...

    def read_present_current_ma(self, dxl_id: int) -> float: ...

    def read_present_temperature(self, dxl_id: int) -> int: ...

    def read_present_input_voltage(self, dxl_id: int) -> float: ...

    def read_hardware_error_status(self, dxl_id: int) -> int: ...

    def sync_read_present_positions(self, dxl_ids: Iterable[int]) -> dict[int, int]: ...

    def sync_read_homing_offsets(self, dxl_ids: Iterable[int]) -> dict[int, int]: ...

    def sync_read_present_currents_ma(
        self, dxl_ids: Iterable[int]
    ) -> dict[int, float]: ...

    def sync_read_present_temperatures(
        self, dxl_ids: Iterable[int]
    ) -> dict[int, int]: ...

    def sync_read_present_input_voltages(
        self, dxl_ids: Iterable[int]
    ) -> dict[int, float]: ...

    def sync_read_hardware_error_statuses(
        self, dxl_ids: Iterable[int]
    ) -> dict[int, int]: ...

    def sync_read_torque_enabled(self, dxl_ids: Iterable[int]) -> dict[int, bool]: ...

    def sync_read_motion_samples(
        self,
        dxl_ids: Iterable[int],
    ) -> dict[int, DynamixelMotionSample]: ...


@dataclass(frozen=True)
class MotorTelemetry:
    """Low-rate health telemetry for one Dynamixel motor."""

    dxl_id: int
    present_current_ma: float
    temperature_celsius: int
    input_voltage: float
    hardware_error_status: int


@dataclass(frozen=True)
class MotorSnapshot:
    """Current low-level state for one Dynamixel motor."""

    dxl_id: int
    position: int
    present_current_ma: float
    temperature_celsius: int
    input_voltage: float
    hardware_error_status: int


@dataclass(frozen=True)
class MotorEncoderState:
    """Current encoder state for one Dynamixel motor."""

    dxl_id: int
    model_number: int
    position: int


@dataclass(frozen=True)
class MotorTestResult:
    """Summary of a single-motor movement test."""

    dxl_id: int
    model_number: int
    start_position: int
    target_position: int
    returned_position: int
    final_position: int
    applied_delta_ticks: int
    temperature_celsius: int
    input_voltage: float
    hardware_error_status: int
    torque_enabled_after: bool


class HandController:
    """Controller facade for direct Dynamixel hand operations."""

    def __init__(
        self, bus: MotorBus, *, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        self.bus = bus
        self._sleep = sleep

    @classmethod
    def from_port(
        cls, *, port: str, baudrate: int = DEFAULT_BAUDRATE
    ) -> HandController:
        """Create a controller backed by a real Dynamixel serial bus."""

        return cls(DynamixelBus(port=port, baudrate=baudrate))

    @classmethod
    def from_fake(
        cls,
        *,
        motor_ids: Iterable[int] = tuple(range(16)),
        positions: Mapping[int, int] | None = None,
    ) -> HandController:
        """Create a controller backed by an in-memory fake Dynamixel bus."""

        from ghand.hardware.fake_bus import FakeBus

        return cls(FakeBus(motor_ids=motor_ids, positions=positions))

    @property
    def is_connected(self) -> bool:
        """Return whether the underlying bus is connected."""

        return self.bus.is_connected

    def __enter__(self) -> HandController:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.disconnect()

    def connect(self) -> None:
        """Connect the underlying bus."""

        self.bus.connect()

    def disconnect(self) -> None:
        """Disconnect the underlying bus."""

        self.bus.disconnect()

    def configure_position_control(
        self,
        dxl_ids: Iterable[int],
        *,
        profile_acceleration: int = DEFAULT_MOTOR_TEST_PROFILE_ACCELERATION,
        profile_velocity: int = DEFAULT_MOTOR_TEST_PROFILE_VELOCITY,
        configure_position_mode: bool = True,
    ) -> None:
        """Configure motors for position control.

        Profile values of 0 remove the Dynamixel acceleration or velocity limit.
        """

        if profile_acceleration < 0:
            raise ValueError("profile_acceleration must be non-negative.")
        if profile_velocity < 0:  # 0 means no Dynamixel velocity limit.
            raise ValueError("profile_velocity must be non-negative.")

        for dxl_id in dxl_ids:
            _validate_motor_id(dxl_id)
            self.bus.set_torque_enabled(dxl_id, False)
            if configure_position_mode:
                self.bus.write_operating_mode(dxl_id, OPERATING_MODE_POSITION)
            self.bus.write_profile(
                dxl_id,
                acceleration=profile_acceleration,
                velocity=profile_velocity,
            )

    def configure_current_based_position_control(
        self,
        dxl_ids: Iterable[int],
        *,
        profile_acceleration: int = DEFAULT_MOTOR_TEST_PROFILE_ACCELERATION,
        profile_velocity: int = DEFAULT_MOTOR_TEST_PROFILE_VELOCITY,
        configure_operating_mode: bool = True,
    ) -> None:
        """Configure motors for current-based position-control writes."""

        if profile_acceleration < 0:
            raise ValueError("profile_acceleration must be non-negative.")
        if profile_velocity < 0:  # 0 means no Dynamixel velocity limit.
            raise ValueError("profile_velocity must be non-negative.")

        for dxl_id in dxl_ids:
            _validate_motor_id(dxl_id)
            self.bus.set_torque_enabled(dxl_id, False)
            if configure_operating_mode:
                self.bus.write_operating_mode(
                    dxl_id, OPERATING_MODE_CURRENT_BASED_POSITION
                )
            self.bus.write_profile(
                dxl_id,
                acceleration=profile_acceleration,
                velocity=profile_velocity,
            )

    def set_torque_enabled_many(self, dxl_ids: Iterable[int], enabled: bool) -> None:
        """Enable or disable torque for a set of motors."""

        for dxl_id in dxl_ids:
            _validate_motor_id(dxl_id)
            self.bus.set_torque_enabled(dxl_id, enabled)

    def read_torque_enabled_many(self, dxl_ids: Iterable[int]) -> dict[int, bool]:
        """Read back Torque Enable(64) for multiple motors."""

        ids = _validated_motor_ids(dxl_ids)
        if not ids:
            return {}
        return self.bus.sync_read_torque_enabled(ids)

    def write_bus_watchdogs(self, dxl_ids: Iterable[int], value: int) -> None:
        """Set Bus Watchdog(98) for multiple motors."""

        ids = _validated_motor_ids(dxl_ids)
        write_watchdog = getattr(self.bus, "write_bus_watchdog", None)
        if not callable(write_watchdog):
            raise RuntimeError("The selected motor bus has no Bus Watchdog support.")
        for dxl_id in ids:
            write_watchdog(dxl_id, value)

    def read_bus_watchdogs(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        """Read Bus Watchdog(98) for multiple motors."""

        ids = _validated_motor_ids(dxl_ids)
        if not ids:
            return {}
        read_watchdogs = getattr(self.bus, "sync_read_bus_watchdogs", None)
        if not callable(read_watchdogs):
            raise RuntimeError("The selected motor bus has no Bus Watchdog support.")
        return read_watchdogs(ids)

    def read_motion_samples(
        self,
        dxl_ids: Iterable[int],
    ) -> tuple[DynamixelMotionSample, ...]:
        """Read time-aligned high-rate motion telemetry for multiple motors."""

        ids = _validated_motor_ids(dxl_ids)
        if not ids:
            return ()
        samples = self.bus.sync_read_motion_samples(ids)
        return tuple(samples[dxl_id] for dxl_id in ids)

    def read_safety_samples(
        self,
        dxl_ids: Iterable[int],
    ) -> tuple[DynamixelSafetySample, ...]:
        """Read the active-control safety state in one bus transaction."""

        ids = _validated_motor_ids(dxl_ids)
        if not ids:
            return ()
        read_samples = getattr(self.bus, "sync_read_safety_samples", None)
        if not callable(read_samples):
            raise RuntimeError("The selected motor bus has no combined safety readback.")
        samples = read_samples(ids)
        return tuple(samples[dxl_id] for dxl_id in ids)

    def write_goal_positions(self, positions: Mapping[int, int]) -> None:
        """Write absolute position-control targets to multiple motors."""

        validated_positions: dict[int, int] = {}
        for dxl_id, position in positions.items():
            _validate_motor_id(dxl_id)
            _validate_position_control_tick(position)
            validated_positions[int(dxl_id)] = int(position)
        if not validated_positions:
            return
        self.bus.sync_write_goal_positions(validated_positions)

    def write_goal_currents_ma(self, currents_ma: Mapping[int, float]) -> None:
        """Write non-negative Goal Current(102) limits to multiple motors."""

        validated_currents: dict[int, float] = {}
        for dxl_id, current_ma in currents_ma.items():
            _validate_motor_id(dxl_id)
            validated_currents[int(dxl_id)] = _validate_goal_current_ma(current_ma)
        if not validated_currents:
            return
        self.bus.sync_write_goal_currents_ma(validated_currents)

    def read_motor_encoder_state(self, dxl_id: int) -> MotorEncoderState:
        """Read only the current encoder position for one motor."""

        _validate_motor_id(dxl_id)
        ping_result = self.bus.ping(dxl_id)
        return MotorEncoderState(
            dxl_id=dxl_id,
            model_number=ping_result.model_number,
            position=normalize_position_tick(self.bus.read_present_position(dxl_id)),
        )

    def read_motor_snapshot(self, dxl_id: int) -> MotorSnapshot:
        """Read position and basic health telemetry for one motor."""

        _validate_motor_id(dxl_id)
        return MotorSnapshot(
            dxl_id=dxl_id,
            position=normalize_position_tick(self.bus.read_present_position(dxl_id)),
            present_current_ma=self.bus.read_present_current_ma(dxl_id),
            temperature_celsius=self.bus.read_present_temperature(dxl_id),
            input_voltage=self.bus.read_present_input_voltage(dxl_id),
            hardware_error_status=self.bus.read_hardware_error_status(dxl_id),
        )

    def read_motor_positions(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        """Read current encoder positions for multiple motors."""

        ids = _validated_motor_ids(dxl_ids)
        if not ids:
            return {}
        positions = self.bus.sync_read_present_positions(ids)
        return {dxl_id: normalize_position_tick(positions[dxl_id]) for dxl_id in ids}

    def read_homing_offsets(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        """Read signed Homing Offset(20) for multiple motors."""

        ids = _validated_motor_ids(dxl_ids)
        if not ids:
            return {}
        return self.bus.sync_read_homing_offsets(ids)

    def read_firmware_versions(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        """Read firmware versions for multiple motors."""

        ids = _validated_motor_ids(dxl_ids)
        if not ids:
            return {}
        read_versions = getattr(self.bus, "sync_read_firmware_versions", None)
        if not callable(read_versions):
            raise RuntimeError(
                "The selected motor bus has no firmware-version readback."
            )
        return read_versions(ids)

    def read_motor_currents_ma(self, dxl_ids: Iterable[int]) -> dict[int, float]:
        """Read present current for multiple motors in milliamps."""

        ids = _validated_motor_ids(dxl_ids)
        if not ids:
            return {}
        return self.bus.sync_read_present_currents_ma(ids)

    def read_motor_telemetry(
        self, dxl_ids: Iterable[int]
    ) -> tuple[MotorTelemetry, ...]:
        """Read low-rate health telemetry for multiple motors."""

        ids = _validated_motor_ids(dxl_ids)
        if not ids:
            return ()

        currents = self.bus.sync_read_present_currents_ma(ids)
        temperatures = self.bus.sync_read_present_temperatures(ids)
        input_voltages = self.bus.sync_read_present_input_voltages(ids)
        hardware_errors = self.bus.sync_read_hardware_error_statuses(ids)
        return tuple(
            MotorTelemetry(
                dxl_id=dxl_id,
                present_current_ma=currents[dxl_id],
                temperature_celsius=temperatures[dxl_id],
                input_voltage=input_voltages[dxl_id],
                hardware_error_status=hardware_errors[dxl_id],
            )
            for dxl_id in ids
        )

    def read_motor_snapshots(self, dxl_ids: Iterable[int]) -> tuple[MotorSnapshot, ...]:
        """Read position and basic health telemetry for multiple motors."""

        ids = _validated_motor_ids(dxl_ids)
        if not ids:
            return ()

        positions = self.read_motor_positions(ids)
        telemetry_by_id = {
            telemetry.dxl_id: telemetry for telemetry in self.read_motor_telemetry(ids)
        }
        return tuple(
            MotorSnapshot(
                dxl_id=dxl_id,
                position=positions[dxl_id],
                present_current_ma=telemetry_by_id[dxl_id].present_current_ma,
                temperature_celsius=telemetry_by_id[dxl_id].temperature_celsius,
                input_voltage=telemetry_by_id[dxl_id].input_voltage,
                hardware_error_status=telemetry_by_id[dxl_id].hardware_error_status,
            )
            for dxl_id in ids
        )

    def test_motor(
        self,
        dxl_id: int,
        *,
        delta_ticks: int = DEFAULT_MOTOR_TEST_DELTA_TICKS,
        max_delta_ticks: int = DEFAULT_MOTOR_TEST_MAX_DELTA_TICKS,
        profile_acceleration: int = DEFAULT_MOTOR_TEST_PROFILE_ACCELERATION,
        profile_velocity: int = DEFAULT_MOTOR_TEST_PROFILE_VELOCITY,
        move_seconds: float = DEFAULT_MOTOR_TEST_MOVE_SECONDS,
        return_seconds: float = DEFAULT_MOTOR_TEST_RETURN_SECONDS,
        configure_position_mode: bool = True,
        keep_torque: bool = False,
    ) -> MotorTestResult:
        """Move one motor by a small tick delta and return it to the start tick."""

        _validate_motor_id(dxl_id)
        _validate_motor_test_args(
            delta_ticks=delta_ticks,
            max_delta_ticks=max_delta_ticks,
            profile_acceleration=profile_acceleration,
            profile_velocity=profile_velocity,
            move_seconds=move_seconds,
            return_seconds=return_seconds,
        )

        ping_result = self.bus.ping(dxl_id)

        self.bus.set_torque_enabled(dxl_id, False)
        if configure_position_mode:
            self.bus.write_operating_mode(dxl_id, OPERATING_MODE_POSITION)
        self.bus.write_profile(
            dxl_id,
            acceleration=profile_acceleration,
            velocity=profile_velocity,
        )

        start_position = normalize_position_tick(self.bus.read_present_position(dxl_id))
        target_position = _resolve_target_position(start_position, delta_ticks)
        applied_delta_ticks = target_position - start_position

        self.bus.write_goal_position(dxl_id, start_position)
        self.bus.set_torque_enabled(dxl_id, True)
        try:
            self.bus.write_goal_position(dxl_id, target_position)
            self._sleep(move_seconds)
            returned_position = normalize_position_tick(
                self.bus.read_present_position(dxl_id)
            )

            self.bus.write_goal_position(dxl_id, start_position)
            self._sleep(return_seconds)
            final_position = normalize_position_tick(
                self.bus.read_present_position(dxl_id)
            )
        finally:
            if not keep_torque:
                self.bus.set_torque_enabled(dxl_id, False)

        return MotorTestResult(
            dxl_id=dxl_id,
            model_number=ping_result.model_number,
            start_position=start_position,
            target_position=target_position,
            returned_position=returned_position,
            final_position=final_position,
            applied_delta_ticks=applied_delta_ticks,
            temperature_celsius=self.bus.read_present_temperature(dxl_id),
            input_voltage=self.bus.read_present_input_voltage(dxl_id),
            hardware_error_status=self.bus.read_hardware_error_status(dxl_id),
            torque_enabled_after=keep_torque,
        )


def _validate_motor_id(dxl_id: int) -> None:
    if not 0 <= dxl_id <= 252:
        raise ValueError("dxl_id must be in the Dynamixel ID range 0..252.")


def _validated_motor_ids(dxl_ids: Iterable[int]) -> tuple[int, ...]:
    ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
    for dxl_id in ids:
        _validate_motor_id(dxl_id)
    return ids


def _validate_position_control_tick(position: int) -> None:
    if not POSITION_CONTROL_MIN_TICK <= int(position) <= POSITION_CONTROL_MAX_TICK:
        raise ValueError(
            "position must be in the Dynamixel position-control range "
            f"{POSITION_CONTROL_MIN_TICK}..{POSITION_CONTROL_MAX_TICK}."
        )


def _validate_goal_current_ma(current_ma: float) -> float:
    current = float(current_ma)
    if not math.isfinite(current):
        raise ValueError("current_ma must be finite.")
    if current < 0.0:
        raise ValueError("current_ma must be non-negative.")
    return current


def _validate_motor_test_args(
    *,
    delta_ticks: int,
    max_delta_ticks: int,
    profile_acceleration: int,
    profile_velocity: int,
    move_seconds: float,
    return_seconds: float,
) -> None:
    if delta_ticks == 0:
        raise ValueError("delta_ticks must not be zero.")
    if max_delta_ticks <= 0:
        raise ValueError("max_delta_ticks must be positive.")
    if abs(delta_ticks) > max_delta_ticks:
        raise ValueError(f"abs(delta_ticks) must be <= {max_delta_ticks}.")
    if profile_acceleration < 0:
        raise ValueError("profile_acceleration must be non-negative.")
    if profile_velocity <= 0:
        raise ValueError("profile_velocity must be positive.")
    if move_seconds < 0 or return_seconds < 0:
        raise ValueError("move_seconds and return_seconds must be non-negative.")


def _resolve_target_position(start_position: int, delta_ticks: int) -> int:
    target_position = start_position + delta_ticks
    if POSITION_CONTROL_MIN_TICK <= target_position <= POSITION_CONTROL_MAX_TICK:
        return target_position

    fallback_target = start_position - delta_ticks
    if POSITION_CONTROL_MIN_TICK <= fallback_target <= POSITION_CONTROL_MAX_TICK:
        return fallback_target

    raise ValueError(
        "delta_ticks cannot produce an in-range position-control target "
        f"from start_position={start_position}."
    )
