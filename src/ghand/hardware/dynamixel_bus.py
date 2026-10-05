"""Dynamixel SDK backed low-level bus helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

try:
    from dynamixel_sdk import (  # type: ignore[import-untyped]
        COMM_SUCCESS,
        GroupSyncRead,
        GroupSyncWrite,
        PacketHandler,
        PortHandler,
    )
except ImportError as exc:  # pragma: no cover - exercised outside the project venv.
    COMM_SUCCESS = 0
    GroupSyncRead = None  # type: ignore[assignment]
    GroupSyncWrite = None  # type: ignore[assignment]
    PacketHandler = None  # type: ignore[assignment]
    PortHandler = None  # type: ignore[assignment]
    _DYNAMIXEL_SDK_IMPORT_ERROR: ImportError | None = exc
else:
    _DYNAMIXEL_SDK_IMPORT_ERROR = None


PROTOCOL_VERSION: Final[float] = 2.0
DEFAULT_BAUDRATE: Final[int] = 1_000_000

ADDR_FIRMWARE_VERSION: Final[int] = 6
ADDR_OPERATING_MODE: Final[int] = 11
ADDR_HOMING_OFFSET: Final[int] = 20
ADDR_CURRENT_LIMIT: Final[int] = 38
ADDR_TORQUE_ENABLE: Final[int] = 64
ADDR_HARDWARE_ERROR_STATUS: Final[int] = 70
ADDR_BUS_WATCHDOG: Final[int] = 98
ADDR_GOAL_CURRENT: Final[int] = 102
ADDR_PROFILE_ACCELERATION: Final[int] = 108
ADDR_PROFILE_VELOCITY: Final[int] = 112
ADDR_GOAL_POSITION: Final[int] = 116
ADDR_REALTIME_TICK: Final[int] = 120
ADDR_MOVING: Final[int] = 122
ADDR_MOVING_STATUS: Final[int] = 123
ADDR_PRESENT_PWM: Final[int] = 124
ADDR_PRESENT_CURRENT: Final[int] = 126
ADDR_PRESENT_VELOCITY: Final[int] = 128
ADDR_PRESENT_POSITION: Final[int] = 132
ADDR_PRESENT_INPUT_VOLTAGE: Final[int] = 144
ADDR_PRESENT_TEMPERATURE: Final[int] = 146

CURRENT_MILLIAMPS_PER_UNIT: Final[float] = 1.0
PRESENT_PWM_PERCENT_PER_UNIT: Final[float] = 0.113
PRESENT_VELOCITY_RPM_PER_UNIT: Final[float] = 0.229
MOTION_STATE_BLOCK_START: Final[int] = ADDR_REALTIME_TICK
MOTION_STATE_BLOCK_LENGTH: Final[int] = (
    ADDR_PRESENT_POSITION + 4 - MOTION_STATE_BLOCK_START
)
SAFETY_STATE_BLOCK_START: Final[int] = ADDR_TORQUE_ENABLE
SAFETY_STATE_BLOCK_LENGTH: Final[int] = (
    ADDR_PRESENT_TEMPERATURE + 1 - SAFETY_STATE_BLOCK_START
)
OPERATING_MODE_CURRENT_BASED_POSITION: Final[int] = 5
OPERATING_MODE_POSITION: Final[int] = 3
BUS_WATCHDOG_MIN_VALUE: Final[int] = 0
BUS_WATCHDOG_MAX_VALUE: Final[int] = 127
BUS_WATCHDOG_ERROR_VALUE: Final[int] = -1
BUS_WATCHDOG_ERROR_RAW_VALUE: Final[int] = 0xFF
BUS_WATCHDOG_UNIT_MS: Final[int] = 20
POSITION_TICKS_PER_REVOLUTION: Final[int] = 4096
POSITION_CONTROL_MIN_TICK: Final[int] = 0
POSITION_CONTROL_MAX_TICK: Final[int] = POSITION_TICKS_PER_REVOLUTION - 1


class DynamixelBusError(RuntimeError):
    """Raised when a Dynamixel bus operation fails."""


@dataclass(frozen=True)
class DynamixelPingResult:
    """Result returned by a successful Dynamixel ping."""

    dxl_id: int
    model_number: int


@dataclass(frozen=True)
class DynamixelMotionSample:
    """One time-aligned XC330 motion-state block from a group sync read."""

    dxl_id: int
    realtime_tick_ms: int
    moving: bool
    moving_status: int
    present_pwm_raw: int
    present_pwm_percent: float
    present_current_raw: int
    present_current_ma: float
    present_velocity_raw: int
    present_velocity_rad_s: float
    present_position_tick: int


@dataclass(frozen=True)
class DynamixelSafetySample:
    """One time-aligned active-control safety block from a group sync read."""

    dxl_id: int
    torque_enabled: bool
    hardware_error_status: int
    bus_watchdog: int
    present_current_raw: int
    present_current_ma: float
    present_position_tick: int
    input_voltage: float
    temperature_celsius: int


class DynamixelBus:
    """Thin wrapper around ROBOTIS Dynamixel SDK packet operations."""

    def __init__(
        self,
        *,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        protocol_version: float = PROTOCOL_VERSION,
    ) -> None:
        if PortHandler is None or PacketHandler is None:
            raise DynamixelBusError(
                "dynamixel_sdk is required for the real Dynamixel bus. "
                "Install ghand hardware dependencies in this environment before "
                "using a serial-port backend."
            ) from _DYNAMIXEL_SDK_IMPORT_ERROR
        self.port = port
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        self.is_connected = False
        self._port_handler = PortHandler(port)
        self._packet_handler = PacketHandler(protocol_version)

    def __enter__(self) -> DynamixelBus:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.disconnect()

    def connect(self) -> None:
        """Open the serial port and apply the configured baudrate."""

        if self.is_connected:
            return
        if not self._port_handler.openPort():
            raise DynamixelBusError(f"Failed to open Dynamixel port: {self.port}")
        if not self._port_handler.setBaudRate(self.baudrate):
            self._port_handler.closePort()
            raise DynamixelBusError(
                f"Failed to set Dynamixel baudrate {self.baudrate} on {self.port}"
            )
        self.is_connected = True

    def disconnect(self) -> None:
        """Close the serial port."""

        if not self.is_connected:
            return
        self._port_handler.closePort()
        self.is_connected = False

    def ping(self, dxl_id: int) -> DynamixelPingResult:
        """Ping one motor and return its model number."""

        model_number, comm_result, dxl_error = self._packet_handler.ping(
            self._port_handler,
            dxl_id,
        )
        self._raise_on_packet_error("ping", dxl_id, comm_result, dxl_error)
        return DynamixelPingResult(dxl_id=dxl_id, model_number=model_number)

    def read1(self, dxl_id: int, address: int, *, label: str | None = None) -> int:
        """Read a one-byte value."""

        value, comm_result, dxl_error = self._packet_handler.read1ByteTxRx(
            self._port_handler,
            dxl_id,
            address,
        )
        self._raise_on_packet_error(
            label or f"read1@{address}", dxl_id, comm_result, dxl_error
        )
        return int(value)

    def read2(self, dxl_id: int, address: int, *, label: str | None = None) -> int:
        """Read a two-byte value."""

        value, comm_result, dxl_error = self._packet_handler.read2ByteTxRx(
            self._port_handler,
            dxl_id,
            address,
        )
        self._raise_on_packet_error(
            label or f"read2@{address}", dxl_id, comm_result, dxl_error
        )
        return int(value)

    def read4(self, dxl_id: int, address: int, *, label: str | None = None) -> int:
        """Read a four-byte value."""

        value, comm_result, dxl_error = self._packet_handler.read4ByteTxRx(
            self._port_handler,
            dxl_id,
            address,
        )
        self._raise_on_packet_error(
            label or f"read4@{address}", dxl_id, comm_result, dxl_error
        )
        return int(value)

    def write1(
        self, dxl_id: int, address: int, value: int, *, label: str | None = None
    ) -> None:
        """Write a one-byte value."""

        comm_result, dxl_error = self._packet_handler.write1ByteTxRx(
            self._port_handler,
            dxl_id,
            address,
            int(value),
        )
        self._raise_on_packet_error(
            label or f"write1@{address}", dxl_id, comm_result, dxl_error
        )

    def write2(
        self, dxl_id: int, address: int, value: int, *, label: str | None = None
    ) -> None:
        """Write a two-byte value."""

        comm_result, dxl_error = self._packet_handler.write2ByteTxRx(
            self._port_handler,
            dxl_id,
            address,
            int(value),
        )
        self._raise_on_packet_error(
            label or f"write2@{address}", dxl_id, comm_result, dxl_error
        )

    def write4(
        self, dxl_id: int, address: int, value: int, *, label: str | None = None
    ) -> None:
        """Write a four-byte value."""

        comm_result, dxl_error = self._packet_handler.write4ByteTxRx(
            self._port_handler,
            dxl_id,
            address,
            to_unsigned32(value),
        )
        self._raise_on_packet_error(
            label or f"write4@{address}", dxl_id, comm_result, dxl_error
        )

    def set_torque_enabled(self, dxl_id: int, enabled: bool) -> None:
        """Write Torque Enable(64)."""

        self.write1(dxl_id, ADDR_TORQUE_ENABLE, int(enabled), label="torque enable")

    def write_operating_mode(self, dxl_id: int, mode: int) -> None:
        """Write Operating Mode(11). Torque must already be off."""

        self.write1(dxl_id, ADDR_OPERATING_MODE, mode, label="operating mode")

    def write_profile(self, dxl_id: int, *, acceleration: int, velocity: int) -> None:
        """Write Profile Acceleration(108) and Profile Velocity(112)."""

        self.write4(
            dxl_id,
            ADDR_PROFILE_ACCELERATION,
            acceleration,
            label="profile acceleration",
        )
        self.write4(dxl_id, ADDR_PROFILE_VELOCITY, velocity, label="profile velocity")

    def write_bus_watchdog(self, dxl_id: int, value: int) -> None:
        """Write Bus Watchdog(98), in 20 ms units; zero disables it."""

        if not BUS_WATCHDOG_MIN_VALUE <= value <= BUS_WATCHDOG_MAX_VALUE:
            raise ValueError(
                "bus watchdog must be in the Dynamixel range "
                f"{BUS_WATCHDOG_MIN_VALUE}..{BUS_WATCHDOG_MAX_VALUE}."
            )
        self.write1(dxl_id, ADDR_BUS_WATCHDOG, value, label="bus watchdog")

    def write_goal_position(self, dxl_id: int, position: int) -> None:
        """Write Goal Position(116)."""

        self.write4(dxl_id, ADDR_GOAL_POSITION, position, label="goal position")

    def write_goal_current_ma(self, dxl_id: int, current_ma: float) -> None:
        """Write signed Goal Current(102), in milliamps for XC330-class motors."""

        unit_value = int(round(float(current_ma) / CURRENT_MILLIAMPS_PER_UNIT))
        self.write2(
            dxl_id, ADDR_GOAL_CURRENT, to_unsigned16(unit_value), label="goal current"
        )

    def sync_write_goal_positions(self, positions: Mapping[int, int]) -> None:
        """Write Goal Position(116) to multiple motors with one sync-write packet."""

        if not positions:
            return

        writer = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            ADDR_GOAL_POSITION,
            4,
        )
        try:
            for dxl_id, position in positions.items():
                param = _to_little_endian_bytes(position, 4)
                if not writer.addParam(int(dxl_id), param):
                    raise DynamixelBusError(
                        f"Dynamixel ID {dxl_id} goal position could not be added "
                        "to group sync write."
                    )
            self._raise_on_group_comm_error(
                "goal position sync write", writer.txPacket()
            )
        finally:
            writer.clearParam()

    def sync_write_goal_currents_ma(self, currents_ma: Mapping[int, float]) -> None:
        """Write signed Goal Current(102) to multiple motors with one sync-write packet."""

        if not currents_ma:
            return

        writer = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            ADDR_GOAL_CURRENT,
            2,
        )
        try:
            for dxl_id, current_ma in currents_ma.items():
                unit_value = int(round(float(current_ma) / CURRENT_MILLIAMPS_PER_UNIT))
                param = _to_little_endian_bytes(unit_value, 2)
                if not writer.addParam(int(dxl_id), param):
                    raise DynamixelBusError(
                        f"Dynamixel ID {dxl_id} goal current could not be added "
                        "to group sync write."
                    )
            self._raise_on_group_comm_error(
                "goal current sync write", writer.txPacket()
            )
        finally:
            writer.clearParam()

    def read_present_position(self, dxl_id: int) -> int:
        """Read signed Present Position(132)."""

        value = self.read4(dxl_id, ADDR_PRESENT_POSITION, label="present position")
        return to_signed32(value)

    def read_homing_offset(self, dxl_id: int) -> int:
        """Read signed Homing Offset(20)."""

        value = self.read4(dxl_id, ADDR_HOMING_OFFSET, label="homing offset")
        return to_signed32(value)

    def read_firmware_version(self, dxl_id: int) -> int:
        """Read Firmware Version(6)."""

        return self.read1(dxl_id, ADDR_FIRMWARE_VERSION, label="firmware version")

    def read_present_current_ma(self, dxl_id: int) -> float:
        """Read signed Present Current(126), in milliamps for XC330-class motors."""

        value = self.read2(dxl_id, ADDR_PRESENT_CURRENT, label="present current")
        return to_signed16(value) * CURRENT_MILLIAMPS_PER_UNIT

    def read_present_temperature(self, dxl_id: int) -> int:
        """Read Present Temperature(146), in degrees Celsius."""

        return self.read1(dxl_id, ADDR_PRESENT_TEMPERATURE, label="present temperature")

    def read_present_input_voltage(self, dxl_id: int) -> float:
        """Read Present Input Voltage(144), in volts."""

        value = self.read2(
            dxl_id, ADDR_PRESENT_INPUT_VOLTAGE, label="present input voltage"
        )
        return value / 10.0

    def read_hardware_error_status(self, dxl_id: int) -> int:
        """Read Hardware Error Status(70)."""

        return self.read1(
            dxl_id, ADDR_HARDWARE_ERROR_STATUS, label="hardware error status"
        )

    def read_bus_watchdog(self, dxl_id: int) -> int:
        """Read Bus Watchdog(98), mapping its raw error value to ``-1``."""

        value = self.read1(dxl_id, ADDR_BUS_WATCHDOG, label="bus watchdog")
        return (
            BUS_WATCHDOG_ERROR_VALUE if value == BUS_WATCHDOG_ERROR_RAW_VALUE else value
        )

    def sync_read_present_positions(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        """Read signed Present Position(132) from multiple motors."""

        values = self._sync_read_values(
            dxl_ids,
            address=ADDR_PRESENT_POSITION,
            length=4,
            label="present position",
        )
        return {dxl_id: to_signed32(value) for dxl_id, value in values.items()}

    def sync_read_homing_offsets(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        """Read signed Homing Offset(20) from multiple motors."""

        values = self._sync_read_values(
            dxl_ids,
            address=ADDR_HOMING_OFFSET,
            length=4,
            label="homing offset",
        )
        return {dxl_id: to_signed32(value) for dxl_id, value in values.items()}

    def sync_read_firmware_versions(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        """Read Firmware Version(6) from multiple motors."""

        return self._sync_read_values(
            dxl_ids,
            address=ADDR_FIRMWARE_VERSION,
            length=1,
            label="firmware version",
        )

    def sync_read_present_currents_ma(self, dxl_ids: Iterable[int]) -> dict[int, float]:
        """Read signed Present Current(126), in milliamps, from multiple motors."""

        values = self._sync_read_values(
            dxl_ids,
            address=ADDR_PRESENT_CURRENT,
            length=2,
            label="present current",
        )
        return {
            dxl_id: to_signed16(value) * CURRENT_MILLIAMPS_PER_UNIT
            for dxl_id, value in values.items()
        }

    def sync_read_present_temperatures(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        """Read Present Temperature(146) from multiple motors."""

        return self._sync_read_values(
            dxl_ids,
            address=ADDR_PRESENT_TEMPERATURE,
            length=1,
            label="present temperature",
        )

    def sync_read_present_input_voltages(
        self, dxl_ids: Iterable[int]
    ) -> dict[int, float]:
        """Read Present Input Voltage(144), in volts, from multiple motors."""

        values = self._sync_read_values(
            dxl_ids,
            address=ADDR_PRESENT_INPUT_VOLTAGE,
            length=2,
            label="present input voltage",
        )
        return {dxl_id: value / 10.0 for dxl_id, value in values.items()}

    def sync_read_hardware_error_statuses(
        self, dxl_ids: Iterable[int]
    ) -> dict[int, int]:
        """Read Hardware Error Status(70) from multiple motors."""

        return self._sync_read_values(
            dxl_ids,
            address=ADDR_HARDWARE_ERROR_STATUS,
            length=1,
            label="hardware error status",
        )

    def sync_read_torque_enabled(self, dxl_ids: Iterable[int]) -> dict[int, bool]:
        """Read Torque Enable(64) from multiple motors."""

        values = self._sync_read_values(
            dxl_ids,
            address=ADDR_TORQUE_ENABLE,
            length=1,
            label="torque enable",
        )
        return {dxl_id: bool(value) for dxl_id, value in values.items()}

    def sync_read_bus_watchdogs(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        """Read Bus Watchdog(98) from multiple motors."""

        values = self._sync_read_values(
            dxl_ids,
            address=ADDR_BUS_WATCHDOG,
            length=1,
            label="bus watchdog",
        )
        return {
            dxl_id: (
                BUS_WATCHDOG_ERROR_VALUE
                if value == BUS_WATCHDOG_ERROR_RAW_VALUE
                else value
            )
            for dxl_id, value in values.items()
        }

    def sync_read_motion_samples(
        self,
        dxl_ids: Iterable[int],
    ) -> dict[int, DynamixelMotionSample]:
        """Read tick, PWM, current, velocity, and position in one packet cycle."""

        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        if not ids:
            return {}

        reader = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            MOTION_STATE_BLOCK_START,
            MOTION_STATE_BLOCK_LENGTH,
        )
        fields = (
            (ADDR_REALTIME_TICK, 2, "realtime tick"),
            (ADDR_MOVING, 1, "moving"),
            (ADDR_MOVING_STATUS, 1, "moving status"),
            (ADDR_PRESENT_PWM, 2, "present PWM"),
            (ADDR_PRESENT_CURRENT, 2, "present current"),
            (ADDR_PRESENT_VELOCITY, 4, "present velocity"),
            (ADDR_PRESENT_POSITION, 4, "present position"),
        )
        try:
            for dxl_id in ids:
                if not reader.addParam(dxl_id):
                    raise DynamixelBusError(
                        f"Dynamixel ID {dxl_id} motion state could not be added "
                        "to group sync read."
                    )
            self._raise_on_group_comm_error(
                "motion state sync read", reader.txRxPacket()
            )

            samples: dict[int, DynamixelMotionSample] = {}
            for dxl_id in ids:
                for address, length, label in fields:
                    if not reader.isAvailable(dxl_id, address, length):
                        raise DynamixelBusError(
                            f"Dynamixel ID {dxl_id} {label} was not available "
                            "after motion state group sync read."
                        )
                pwm_raw = to_signed16(reader.getData(dxl_id, ADDR_PRESENT_PWM, 2))
                current_raw = to_signed16(
                    reader.getData(dxl_id, ADDR_PRESENT_CURRENT, 2)
                )
                velocity_raw = to_signed32(
                    reader.getData(dxl_id, ADDR_PRESENT_VELOCITY, 4)
                )
                samples[dxl_id] = DynamixelMotionSample(
                    dxl_id=dxl_id,
                    realtime_tick_ms=int(reader.getData(dxl_id, ADDR_REALTIME_TICK, 2)),
                    moving=bool(reader.getData(dxl_id, ADDR_MOVING, 1)),
                    moving_status=int(reader.getData(dxl_id, ADDR_MOVING_STATUS, 1)),
                    present_pwm_raw=pwm_raw,
                    present_pwm_percent=pwm_raw * PRESENT_PWM_PERCENT_PER_UNIT,
                    present_current_raw=current_raw,
                    present_current_ma=current_raw * CURRENT_MILLIAMPS_PER_UNIT,
                    present_velocity_raw=velocity_raw,
                    present_velocity_rad_s=(
                        velocity_raw
                        * PRESENT_VELOCITY_RPM_PER_UNIT
                        * 2.0
                        * math.pi
                        / 60.0
                    ),
                    present_position_tick=to_signed32(
                        reader.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
                    ),
                )
            return samples
        finally:
            reader.clearParam()

    def sync_read_safety_samples(
        self,
        dxl_ids: Iterable[int],
    ) -> dict[int, DynamixelSafetySample]:
        """Read all active-control safety values in one packet cycle."""

        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        if not ids:
            return {}

        reader = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            SAFETY_STATE_BLOCK_START,
            SAFETY_STATE_BLOCK_LENGTH,
        )
        fields = (
            (ADDR_TORQUE_ENABLE, 1, "torque enable"),
            (ADDR_HARDWARE_ERROR_STATUS, 1, "hardware error status"),
            (ADDR_BUS_WATCHDOG, 1, "bus watchdog"),
            (ADDR_PRESENT_CURRENT, 2, "present current"),
            (ADDR_PRESENT_POSITION, 4, "present position"),
            (ADDR_PRESENT_INPUT_VOLTAGE, 2, "present input voltage"),
            (ADDR_PRESENT_TEMPERATURE, 1, "present temperature"),
        )
        try:
            for dxl_id in ids:
                if not reader.addParam(dxl_id):
                    raise DynamixelBusError(
                        f"Dynamixel ID {dxl_id} safety state could not be added "
                        "to group sync read."
                    )
            self._raise_on_group_comm_error(
                "safety state sync read", reader.txRxPacket()
            )

            samples: dict[int, DynamixelSafetySample] = {}
            for dxl_id in ids:
                for address, length, label in fields:
                    if not reader.isAvailable(dxl_id, address, length):
                        raise DynamixelBusError(
                            f"Dynamixel ID {dxl_id} {label} was not available "
                            "after safety state group sync read."
                        )
                watchdog_raw = int(reader.getData(dxl_id, ADDR_BUS_WATCHDOG, 1))
                current_raw = to_signed16(
                    reader.getData(dxl_id, ADDR_PRESENT_CURRENT, 2)
                )
                samples[dxl_id] = DynamixelSafetySample(
                    dxl_id=dxl_id,
                    torque_enabled=bool(reader.getData(dxl_id, ADDR_TORQUE_ENABLE, 1)),
                    hardware_error_status=int(
                        reader.getData(dxl_id, ADDR_HARDWARE_ERROR_STATUS, 1)
                    ),
                    bus_watchdog=(
                        BUS_WATCHDOG_ERROR_VALUE
                        if watchdog_raw == BUS_WATCHDOG_ERROR_RAW_VALUE
                        else watchdog_raw
                    ),
                    present_current_raw=current_raw,
                    present_current_ma=current_raw * CURRENT_MILLIAMPS_PER_UNIT,
                    present_position_tick=to_signed32(
                        reader.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
                    ),
                    input_voltage=(
                        reader.getData(dxl_id, ADDR_PRESENT_INPUT_VOLTAGE, 2) / 10.0
                    ),
                    temperature_celsius=int(
                        reader.getData(dxl_id, ADDR_PRESENT_TEMPERATURE, 1)
                    ),
                )
            return samples
        finally:
            reader.clearParam()

    def _sync_read_values(
        self,
        dxl_ids: Iterable[int],
        *,
        address: int,
        length: int,
        label: str,
    ) -> dict[int, int]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        if not ids:
            return {}

        reader = GroupSyncRead(
            self._port_handler, self._packet_handler, address, length
        )
        try:
            for dxl_id in ids:
                if not reader.addParam(dxl_id):
                    raise DynamixelBusError(
                        f"Dynamixel ID {dxl_id} {label} could not be added "
                        "to group sync read."
                    )
            self._raise_on_group_comm_error(f"{label} sync read", reader.txRxPacket())

            values: dict[int, int] = {}
            for dxl_id in ids:
                if not reader.isAvailable(dxl_id, address, length):
                    raise DynamixelBusError(
                        f"Dynamixel ID {dxl_id} {label} was not available "
                        "after group sync read."
                    )
                values[dxl_id] = int(reader.getData(dxl_id, address, length))
            return values
        finally:
            reader.clearParam()

    def _raise_on_packet_error(
        self,
        operation: str,
        dxl_id: int,
        comm_result: int,
        dxl_error: int,
    ) -> None:
        if comm_result != COMM_SUCCESS:
            message = self._packet_handler.getTxRxResult(comm_result)
            raise DynamixelBusError(
                f"Dynamixel ID {dxl_id} {operation} failed: {message}"
            )
        if dxl_error:
            message = self._packet_handler.getRxPacketError(dxl_error)
            raise DynamixelBusError(
                f"Dynamixel ID {dxl_id} {operation} returned error: {message}"
            )

    def _raise_on_group_comm_error(self, operation: str, comm_result: int) -> None:
        if comm_result != COMM_SUCCESS:
            message = self._packet_handler.getTxRxResult(comm_result)
            raise DynamixelBusError(f"Dynamixel group {operation} failed: {message}")


def to_signed32(value: int) -> int:
    """Interpret an unsigned 32-bit integer as signed."""

    value = int(value)
    if value >= 2**31:
        return value - 2**32
    return value


def to_signed16(value: int) -> int:
    """Interpret an unsigned 16-bit integer as signed."""

    value = int(value) & 0xFFFF
    if value >= 2**15:
        return value - 2**16
    return value


def to_unsigned32(value: int) -> int:
    """Convert an integer to the unsigned 32-bit representation used by SDK writes."""

    return int(value) & 0xFFFFFFFF


def to_unsigned16(value: int) -> int:
    """Convert an integer to the unsigned 16-bit representation used by SDK writes."""

    return int(value) & 0xFFFF


def _to_little_endian_bytes(value: int, length: int) -> list[int]:
    mask = (1 << (8 * length)) - 1
    unsigned = int(value) & mask
    return [(unsigned >> (8 * index)) & 0xFF for index in range(length)]


def normalize_position_tick(position: int) -> int:
    """Normalize a raw position to the 0..4095 position-control range."""

    return int(position) % POSITION_TICKS_PER_REVOLUTION
