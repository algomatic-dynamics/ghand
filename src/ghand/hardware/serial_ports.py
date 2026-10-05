"""Discovery of the USB serial adapter that connects the hand."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final


AUTO_PORT: Final[str] = "auto"


class SerialPortError(RuntimeError):
    """Raised when no single USB serial adapter can be selected automatically."""


def list_serial_ports() -> tuple[str, ...]:
    """Return candidate USB serial device paths, stable names first."""

    candidates: list[str] = []
    by_id = Path("/dev/serial/by-id")
    if by_id.is_dir():
        candidates.extend(sorted(str(path) for path in by_id.iterdir()))
    if sys.platform == "darwin":
        candidates.extend(sorted(str(path) for path in Path("/dev").glob("cu.usb*")))
    if not candidates:
        from serial.tools import list_ports  # type: ignore[import-untyped]

        candidates.extend(
            sorted(port.device for port in list_ports.comports() if port.vid is not None)
        )
    return tuple(dict.fromkeys(candidates))


def resolve_serial_port(port: str | None) -> str:
    """Return a device path; ``None`` or ``"auto"`` selects the single adapter found."""

    if port is not None and port != AUTO_PORT:
        return port
    candidates = list_serial_ports()
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SerialPortError(
            "No USB serial adapter found. Connect the hand and pass --port if needed."
        )
    raise SerialPortError(
        "Several USB serial adapters found; choose one with --port: "
        + ", ".join(candidates)
    )


def usb_serial_latency_timer_ms(device: str) -> int | None:
    """Return the Linux usb-serial latency timer for ``device`` when readable."""

    try:
        tty_name = Path(device).resolve().name
        value = Path(f"/sys/bus/usb-serial/devices/{tty_name}/latency_timer").read_text()
    except OSError:
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None
