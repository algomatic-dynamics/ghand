from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import ghand.cli as cli
from ghand.cli import app
from ghand.hand import GHAND_RIGHT_MOTOR_IDS
from ghand.hardware import DynamixelBusError, MotorEncoderState, MotorTelemetry

runner = CliRunner()


class FakeDoctorController:
    missing_ids: set[int] = set()
    homing_offsets: dict[int, int] = {}
    temperature = 35
    voltage = 12.0
    hardware_error = 0
    instances: list["FakeDoctorController"] = []

    def __init__(self, port: str, baudrate: int) -> None:
        self.port = port
        self.baudrate = baudrate
        self.writes: list[object] = []
        type(self).instances.append(self)

    @classmethod
    def from_port(cls, *, port: str, baudrate: int) -> "FakeDoctorController":
        return cls(port, baudrate)

    def __enter__(self) -> "FakeDoctorController":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read_motor_encoder_state(self, dxl_id: int) -> MotorEncoderState:
        if dxl_id in self.missing_ids:
            raise DynamixelBusError(f"no response from {dxl_id}")
        return MotorEncoderState(dxl_id=dxl_id, model_number=1200, position=2048)

    def read_firmware_versions(self, dxl_ids) -> dict[int, int]:
        return {dxl_id: 46 for dxl_id in dxl_ids}

    def read_homing_offsets(self, dxl_ids) -> dict[int, int]:
        return {dxl_id: self.homing_offsets.get(dxl_id, 0) for dxl_id in dxl_ids}

    def read_motor_telemetry(self, dxl_ids) -> tuple[MotorTelemetry, ...]:
        return tuple(
            MotorTelemetry(
                dxl_id=dxl_id,
                present_current_ma=0.0,
                temperature_celsius=self.temperature,
                input_voltage=self.voltage,
                hardware_error_status=self.hardware_error,
            )
            for dxl_id in dxl_ids
        )

    def write_goal_positions(self, positions) -> None:  # pragma: no cover - must not be called
        raise AssertionError("doctor must not move motors")


@pytest.fixture
def device(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "usb-FTDI-U2D2-if00-port0"
    path.write_text("")
    monkeypatch.setattr("ghand.hardware.serial_ports.list_serial_ports", lambda: (str(path),))
    monkeypatch.setattr(cli, "HandController", FakeDoctorController)
    FakeDoctorController.instances = []
    FakeDoctorController.missing_ids = set()
    FakeDoctorController.homing_offsets = {}
    FakeDoctorController.temperature = 35
    FakeDoctorController.voltage = 12.0
    FakeDoctorController.hardware_error = 0
    return path


def test_doctor_passes_on_a_healthy_hand_without_a_port(device: Path) -> None:
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "All checks passed." in result.output
    assert "16/16 responded" in result.output
    assert "0 on every motor" in result.output
    controller = FakeDoctorController.instances[0]
    assert controller.port == str(device)
    assert controller.writes == []


def test_doctor_reports_missing_motors_and_stops(device: Path) -> None:
    FakeDoctorController.missing_ids = {GHAND_RIGHT_MOTOR_IDS[0]}

    result = runner.invoke(app, ["doctor", "--port", str(device)])

    assert result.exit_code == 1
    assert "15/16 responded" in result.output
    assert "homing offset" not in result.output


def test_doctor_fails_on_nonzero_homing_offset_and_hot_motor(device: Path) -> None:
    FakeDoctorController.homing_offsets = {GHAND_RIGHT_MOTOR_IDS[3]: 120}
    FakeDoctorController.temperature = 65

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Homing Offset(20) must be 0" in result.output
    assert "max 65 C" in result.output
    assert "2 check(s) failed." in result.output


def test_doctor_fails_on_a_5v_supply(device: Path) -> None:
    FakeDoctorController.voltage = 5.1

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "5.1-5.1 V; expected 10.8-13.2 V" in result.output
    assert "1 check(s) failed." in result.output


def test_doctor_without_any_adapter_fails_cleanly(monkeypatch) -> None:
    monkeypatch.setattr("ghand.hardware.serial_ports.list_serial_ports", lambda: ())

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "No USB serial adapter found" in result.output


def test_port_auto_is_resolved_for_motor_scan(device: Path, monkeypatch) -> None:
    class ScanController(FakeDoctorController):
        pass

    monkeypatch.setattr(cli, "HandController", ScanController)
    result = runner.invoke(app, ["motor", "scan", "--port", "auto"])

    assert result.exit_code == 0, result.output
    assert ScanController.instances[-1].port == str(device)
