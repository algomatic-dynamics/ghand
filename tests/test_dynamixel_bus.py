import math

import pytest

import ghand.hardware.dynamixel_bus as dxl


class FakePortHandler:
    instances = []

    def __init__(self, port: str) -> None:
        self.port = port
        self.opened = False
        self.closed = False
        self.baudrate = None
        self.open_result = True
        self.baudrate_result = True
        FakePortHandler.instances.append(self)

    def openPort(self) -> bool:
        self.opened = True
        return self.open_result

    def setBaudRate(self, baudrate: int) -> bool:
        self.baudrate = baudrate
        return self.baudrate_result

    def closePort(self) -> None:
        self.closed = True


class FakePacketHandler:
    instances = []

    def __init__(self, protocol_version: float) -> None:
        self.protocol_version = protocol_version
        self.read_values = {}
        self.errors = {}
        self.calls = []
        self.add_param_failures = set()
        self.unavailable = set()
        FakePacketHandler.instances.append(self)

    def ping(self, port, dxl_id):
        self.calls.append(("ping", dxl_id))
        return 1234, *self.errors.get("ping", (dxl.COMM_SUCCESS, 0))

    def read1ByteTxRx(self, port, dxl_id, address):
        self.calls.append(("read1", dxl_id, address))
        return self.read_values.get(("read1", address), 0), *self.errors.get(
            "read1", (dxl.COMM_SUCCESS, 0)
        )

    def read2ByteTxRx(self, port, dxl_id, address):
        self.calls.append(("read2", dxl_id, address))
        return self.read_values.get(("read2", address), 0), *self.errors.get(
            "read2", (dxl.COMM_SUCCESS, 0)
        )

    def read4ByteTxRx(self, port, dxl_id, address):
        self.calls.append(("read4", dxl_id, address))
        return self.read_values.get(("read4", address), 0), *self.errors.get(
            "read4", (dxl.COMM_SUCCESS, 0)
        )

    def write1ByteTxRx(self, port, dxl_id, address, value):
        self.calls.append(("write1", dxl_id, address, value))
        return self.errors.get("write1", (dxl.COMM_SUCCESS, 0))

    def write2ByteTxRx(self, port, dxl_id, address, value):
        self.calls.append(("write2", dxl_id, address, value))
        return self.errors.get("write2", (dxl.COMM_SUCCESS, 0))

    def write4ByteTxRx(self, port, dxl_id, address, value):
        self.calls.append(("write4", dxl_id, address, value))
        return self.errors.get("write4", (dxl.COMM_SUCCESS, 0))

    def getTxRxResult(self, comm_result):
        return f"comm error {comm_result}"

    def getRxPacketError(self, dxl_error):
        return f"packet error {dxl_error}"


class FakeGroupSyncWrite:
    instances = []

    def __init__(self, port, packet, address, length) -> None:
        self.port = port
        self.packet = packet
        self.address = address
        self.length = length
        self.params = []
        FakeGroupSyncWrite.instances.append(self)

    def addParam(self, dxl_id, data):
        if ("sync_write", self.address, dxl_id) in self.packet.add_param_failures:
            return False
        self.params.append((dxl_id, tuple(data)))
        return True

    def txPacket(self):
        self.packet.calls.append(
            ("sync_write", self.address, self.length, tuple(self.params))
        )
        return self.packet.errors.get("sync_write", (dxl.COMM_SUCCESS, 0))[0]

    def clearParam(self):
        self.params = []


class FakeGroupSyncRead:
    instances = []

    def __init__(self, port, packet, address, length) -> None:
        self.port = port
        self.packet = packet
        self.address = address
        self.length = length
        self.ids = []
        FakeGroupSyncRead.instances.append(self)

    def addParam(self, dxl_id):
        if ("sync_read", self.address, dxl_id) in self.packet.add_param_failures:
            return False
        self.ids.append(dxl_id)
        return True

    def txRxPacket(self):
        self.packet.calls.append(
            ("sync_read", self.address, self.length, tuple(self.ids))
        )
        return self.packet.errors.get("sync_read", (dxl.COMM_SUCCESS, 0))[0]

    def isAvailable(self, dxl_id, address, length):
        return (address, length, dxl_id) not in self.packet.unavailable

    def getData(self, dxl_id, address, length):
        return self.packet.read_values.get(
            ("sync", address, dxl_id),
            self.packet.read_values.get((f"read{length}", address), 0),
        )

    def clearParam(self):
        self.ids = []


@pytest.fixture
def fake_sdk(monkeypatch):
    FakePortHandler.instances = []
    FakePacketHandler.instances = []
    FakeGroupSyncWrite.instances = []
    FakeGroupSyncRead.instances = []
    monkeypatch.setattr(dxl, "PortHandler", FakePortHandler)
    monkeypatch.setattr(dxl, "PacketHandler", FakePacketHandler)
    monkeypatch.setattr(dxl, "GroupSyncWrite", FakeGroupSyncWrite)
    monkeypatch.setattr(dxl, "GroupSyncRead", FakeGroupSyncRead)

    bus = dxl.DynamixelBus(port="/dev/ttyUSB0", baudrate=4_000_000)
    return bus, FakePortHandler.instances[-1], FakePacketHandler.instances[-1]


def test_default_baudrate_is_one_megabit() -> None:
    assert dxl.DEFAULT_BAUDRATE == 1_000_000


def test_connect_opens_port_and_sets_baudrate(fake_sdk) -> None:
    bus, port, _packet = fake_sdk

    bus.connect()

    assert bus.is_connected is True
    assert port.opened is True
    assert port.baudrate == 4_000_000

    bus.disconnect()

    assert bus.is_connected is False
    assert port.closed is True


def test_ping_returns_model_number(fake_sdk) -> None:
    bus, _port, packet = fake_sdk

    result = bus.ping(3)

    assert result == dxl.DynamixelPingResult(dxl_id=3, model_number=1234)
    assert packet.calls == [("ping", 3)]


def test_helpers_use_expected_control_table_addresses(fake_sdk) -> None:
    bus, _port, packet = fake_sdk
    packet.read_values[("read4", dxl.ADDR_HOMING_OFFSET)] = 0xFFFF_FFF0
    packet.read_values[("read1", dxl.ADDR_FIRMWARE_VERSION)] = 38
    packet.read_values[("read4", dxl.ADDR_PRESENT_POSITION)] = 0xFFFF_FFF0
    packet.read_values[("read2", dxl.ADDR_PRESENT_CURRENT)] = 0xFFFF_FFF0
    packet.read_values[("read2", dxl.ADDR_PRESENT_INPUT_VOLTAGE)] = 121
    packet.read_values[("read1", dxl.ADDR_PRESENT_TEMPERATURE)] = 32
    packet.read_values[("read1", dxl.ADDR_HARDWARE_ERROR_STATUS)] = 0
    packet.read_values[("read1", dxl.ADDR_BUS_WATCHDOG)] = 0xFF

    bus.set_torque_enabled(7, True)
    bus.write_operating_mode(7, dxl.OPERATING_MODE_POSITION)
    bus.write_profile(7, acceleration=20, velocity=40)
    bus.write_goal_position(7, -16)
    bus.write_goal_current_ma(7, -16.0)
    bus.write_bus_watchdog(7, 10)

    assert bus.read_homing_offset(7) == -16
    assert bus.read_firmware_version(7) == 38
    assert bus.read_present_position(7) == -16
    assert bus.read_present_current_ma(7) == -16.0
    assert bus.read_present_input_voltage(7) == 12.1
    assert bus.read_present_temperature(7) == 32
    assert bus.read_hardware_error_status(7) == 0
    assert bus.read_bus_watchdog(7) == -1
    assert packet.calls == [
        ("write1", 7, dxl.ADDR_TORQUE_ENABLE, 1),
        ("write1", 7, dxl.ADDR_OPERATING_MODE, dxl.OPERATING_MODE_POSITION),
        ("write4", 7, dxl.ADDR_PROFILE_ACCELERATION, 20),
        ("write4", 7, dxl.ADDR_PROFILE_VELOCITY, 40),
        ("write4", 7, dxl.ADDR_GOAL_POSITION, 0xFFFF_FFF0),
        ("write2", 7, dxl.ADDR_GOAL_CURRENT, 0xFFF0),
        ("write1", 7, dxl.ADDR_BUS_WATCHDOG, 10),
        ("read4", 7, dxl.ADDR_HOMING_OFFSET),
        ("read1", 7, dxl.ADDR_FIRMWARE_VERSION),
        ("read4", 7, dxl.ADDR_PRESENT_POSITION),
        ("read2", 7, dxl.ADDR_PRESENT_CURRENT),
        ("read2", 7, dxl.ADDR_PRESENT_INPUT_VOLTAGE),
        ("read1", 7, dxl.ADDR_PRESENT_TEMPERATURE),
        ("read1", 7, dxl.ADDR_HARDWARE_ERROR_STATUS),
        ("read1", 7, dxl.ADDR_BUS_WATCHDOG),
    ]


def test_bus_watchdog_rejects_out_of_range_values(fake_sdk) -> None:
    bus, _port, packet = fake_sdk

    with pytest.raises(ValueError, match="0..127"):
        bus.write_bus_watchdog(7, 128)

    assert packet.calls == []


def test_packet_errors_raise_bus_error(fake_sdk) -> None:
    bus, _port, packet = fake_sdk
    packet.errors["read1"] = (dxl.COMM_SUCCESS, 2)

    with pytest.raises(dxl.DynamixelBusError, match="packet error 2"):
        bus.read1(1, dxl.ADDR_TORQUE_ENABLE)


def test_sync_goal_write_uses_group_sync_write(fake_sdk) -> None:
    bus, _port, packet = fake_sdk

    bus.sync_write_goal_positions({2: 2100, 3: -16})

    assert packet.calls == [
        (
            "sync_write",
            dxl.ADDR_GOAL_POSITION,
            4,
            (
                (2, (0x34, 0x08, 0x00, 0x00)),
                (3, (0xF0, 0xFF, 0xFF, 0xFF)),
            ),
        )
    ]


def test_sync_goal_current_write_uses_group_sync_write(fake_sdk) -> None:
    bus, _port, packet = fake_sdk

    bus.sync_write_goal_currents_ma({2: 90.0, 3: -16.0})

    assert packet.calls == [
        (
            "sync_write",
            dxl.ADDR_GOAL_CURRENT,
            2,
            (
                (2, (0x5A, 0x00)),
                (3, (0xF0, 0xFF)),
            ),
        )
    ]


def test_sync_reads_use_group_sync_read(fake_sdk) -> None:
    bus, _port, packet = fake_sdk
    packet.read_values[("sync", dxl.ADDR_HOMING_OFFSET, 2)] = 0xFFFF_FFF0
    packet.read_values[("sync", dxl.ADDR_HOMING_OFFSET, 3)] = 0
    packet.read_values[("sync", dxl.ADDR_FIRMWARE_VERSION, 2)] = 38
    packet.read_values[("sync", dxl.ADDR_FIRMWARE_VERSION, 3)] = 39
    packet.read_values[("sync", dxl.ADDR_PRESENT_POSITION, 2)] = 0xFFFF_FFF0
    packet.read_values[("sync", dxl.ADDR_PRESENT_POSITION, 3)] = 42
    packet.read_values[("sync", dxl.ADDR_PRESENT_CURRENT, 2)] = 0xFFFF_FFF0
    packet.read_values[("sync", dxl.ADDR_PRESENT_CURRENT, 3)] = 42
    packet.read_values[("sync", dxl.ADDR_PRESENT_TEMPERATURE, 2)] = 31
    packet.read_values[("sync", dxl.ADDR_PRESENT_TEMPERATURE, 3)] = 32
    packet.read_values[("sync", dxl.ADDR_PRESENT_INPUT_VOLTAGE, 2)] = 120
    packet.read_values[("sync", dxl.ADDR_PRESENT_INPUT_VOLTAGE, 3)] = 121
    packet.read_values[("sync", dxl.ADDR_HARDWARE_ERROR_STATUS, 2)] = 0
    packet.read_values[("sync", dxl.ADDR_HARDWARE_ERROR_STATUS, 3)] = 2
    packet.read_values[("sync", dxl.ADDR_TORQUE_ENABLE, 2)] = 0
    packet.read_values[("sync", dxl.ADDR_TORQUE_ENABLE, 3)] = 1
    packet.read_values[("sync", dxl.ADDR_BUS_WATCHDOG, 2)] = 10
    packet.read_values[("sync", dxl.ADDR_BUS_WATCHDOG, 3)] = 0xFF

    assert bus.sync_read_homing_offsets((2, 3)) == {2: -16, 3: 0}
    assert bus.sync_read_firmware_versions((2, 3)) == {2: 38, 3: 39}
    assert bus.sync_read_present_positions((2, 3)) == {2: -16, 3: 42}
    assert bus.sync_read_present_currents_ma((2, 3)) == {2: -16.0, 3: 42.0}
    assert bus.sync_read_present_temperatures((2, 3)) == {2: 31, 3: 32}
    assert bus.sync_read_present_input_voltages((2, 3)) == {2: 12.0, 3: 12.1}
    assert bus.sync_read_hardware_error_statuses((2, 3)) == {2: 0, 3: 2}
    assert bus.sync_read_torque_enabled((2, 3)) == {2: False, 3: True}
    assert bus.sync_read_bus_watchdogs((2, 3)) == {2: 10, 3: -1}
    assert packet.calls == [
        ("sync_read", dxl.ADDR_HOMING_OFFSET, 4, (2, 3)),
        ("sync_read", dxl.ADDR_FIRMWARE_VERSION, 1, (2, 3)),
        ("sync_read", dxl.ADDR_PRESENT_POSITION, 4, (2, 3)),
        ("sync_read", dxl.ADDR_PRESENT_CURRENT, 2, (2, 3)),
        ("sync_read", dxl.ADDR_PRESENT_TEMPERATURE, 1, (2, 3)),
        ("sync_read", dxl.ADDR_PRESENT_INPUT_VOLTAGE, 2, (2, 3)),
        ("sync_read", dxl.ADDR_HARDWARE_ERROR_STATUS, 1, (2, 3)),
        ("sync_read", dxl.ADDR_TORQUE_ENABLE, 1, (2, 3)),
        ("sync_read", dxl.ADDR_BUS_WATCHDOG, 1, (2, 3)),
    ]


def test_sync_motion_read_uses_one_contiguous_group_read(fake_sdk) -> None:
    bus, _port, packet = fake_sdk
    packet.read_values[("sync", dxl.ADDR_REALTIME_TICK, 2)] = 123
    packet.read_values[("sync", dxl.ADDR_MOVING, 2)] = 1
    packet.read_values[("sync", dxl.ADDR_MOVING_STATUS, 2)] = 9
    packet.read_values[("sync", dxl.ADDR_PRESENT_PWM, 2)] = 0xFFFF
    packet.read_values[("sync", dxl.ADDR_PRESENT_CURRENT, 2)] = 42
    packet.read_values[("sync", dxl.ADDR_PRESENT_VELOCITY, 2)] = 10
    packet.read_values[("sync", dxl.ADDR_PRESENT_POSITION, 2)] = 2048

    samples = bus.sync_read_motion_samples((2,))

    sample = samples[2]
    assert sample.realtime_tick_ms == 123
    assert sample.moving is True
    assert sample.moving_status == 9
    assert sample.present_pwm_raw == -1
    assert sample.present_pwm_percent == pytest.approx(-0.113)
    assert sample.present_current_raw == 42
    assert sample.present_current_ma == 42.0
    assert sample.present_velocity_raw == 10
    assert sample.present_velocity_rad_s == pytest.approx(
        10 * 0.229 * 2.0 * math.pi / 60.0
    )
    assert sample.present_position_tick == 2048
    assert packet.calls == [
        (
            "sync_read",
            dxl.MOTION_STATE_BLOCK_START,
            dxl.MOTION_STATE_BLOCK_LENGTH,
            (2,),
        )
    ]


def test_sync_safety_read_uses_one_contiguous_group_read(fake_sdk) -> None:
    bus, _port, packet = fake_sdk
    packet.read_values[("sync", dxl.ADDR_TORQUE_ENABLE, 2)] = 1
    packet.read_values[("sync", dxl.ADDR_HARDWARE_ERROR_STATUS, 2)] = 4
    packet.read_values[("sync", dxl.ADDR_BUS_WATCHDOG, 2)] = 0xFF
    packet.read_values[("sync", dxl.ADDR_PRESENT_CURRENT, 2)] = 0xFFFF
    packet.read_values[("sync", dxl.ADDR_PRESENT_POSITION, 2)] = 0xFFFF_FFF0
    packet.read_values[("sync", dxl.ADDR_PRESENT_INPUT_VOLTAGE, 2)] = 121
    packet.read_values[("sync", dxl.ADDR_PRESENT_TEMPERATURE, 2)] = 33

    samples = bus.sync_read_safety_samples((2,))

    sample = samples[2]
    assert sample.torque_enabled is True
    assert sample.hardware_error_status == 4
    assert sample.bus_watchdog == -1
    assert sample.present_current_raw == -1
    assert sample.present_current_ma == -1.0
    assert sample.present_position_tick == -16
    assert sample.input_voltage == 12.1
    assert sample.temperature_celsius == 33
    assert packet.calls == [
        (
            "sync_read",
            dxl.SAFETY_STATE_BLOCK_START,
            dxl.SAFETY_STATE_BLOCK_LENGTH,
            (2,),
        )
    ]


def test_sync_group_errors_raise_bus_error(fake_sdk) -> None:
    bus, _port, packet = fake_sdk
    packet.errors["sync_read"] = (-1000, 0)

    with pytest.raises(
        dxl.DynamixelBusError, match="group present position sync read failed"
    ):
        bus.sync_read_present_positions((2,))


def test_sync_read_unavailable_value_raises_bus_error(fake_sdk) -> None:
    bus, _port, packet = fake_sdk
    packet.unavailable.add((dxl.ADDR_PRESENT_POSITION, 4, 2))

    with pytest.raises(dxl.DynamixelBusError, match="was not available"):
        bus.sync_read_present_positions((2,))


def test_integer_conversion_helpers() -> None:
    assert dxl.to_signed32(0xFFFF_FFFF) == -1
    assert dxl.to_signed32(42) == 42
    assert dxl.to_signed16(0xFFFF) == -1
    assert dxl.to_unsigned16(-1) == 0xFFFF
    assert dxl.to_unsigned32(-1) == 0xFFFF_FFFF
    assert dxl.normalize_position_tick(4097) == 1
