from __future__ import annotations

import pytest

from ghand.hardware import SerialPortError, resolve_serial_port
from ghand.hardware import serial_ports


def test_explicit_port_is_returned_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(serial_ports, "list_serial_ports", lambda: ())
    assert resolve_serial_port("/dev/ttyUSB3") == "/dev/ttyUSB3"


@pytest.mark.parametrize("port", [None, "auto"])
def test_single_adapter_is_selected(monkeypatch, port) -> None:
    monkeypatch.setattr(serial_ports, "list_serial_ports", lambda: ("/dev/serial/by-id/usb-a",))
    assert resolve_serial_port(port) == "/dev/serial/by-id/usb-a"


def test_no_adapter_is_an_error(monkeypatch) -> None:
    monkeypatch.setattr(serial_ports, "list_serial_ports", lambda: ())
    with pytest.raises(SerialPortError, match="No USB serial adapter"):
        resolve_serial_port(None)


def test_several_adapters_list_the_choices(monkeypatch) -> None:
    monkeypatch.setattr(
        serial_ports, "list_serial_ports", lambda: ("/dev/serial/by-id/usb-a", "/dev/serial/by-id/usb-b")
    )
    with pytest.raises(SerialPortError, match="usb-a, /dev/serial/by-id/usb-b"):
        resolve_serial_port("auto")
