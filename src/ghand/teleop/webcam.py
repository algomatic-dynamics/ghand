"""Webcam capture utilities."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol


CAP_PROP_BUFFERSIZE = 38
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5


class Capture(Protocol):
    """Subset of the OpenCV VideoCapture API used by ghand."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Any]: ...

    def get(self, prop_id: int) -> float: ...

    def release(self) -> None: ...


@dataclass(frozen=True)
class WebcamInfo:
    """Result of a webcam availability check."""

    camera: int
    width: int
    height: int
    fps: float
    frame_width: int
    frame_height: int
    channels: int
    backend: str | None = None


class WebcamError(RuntimeError):
    """Raised when a webcam cannot be opened or read."""


@dataclass(frozen=True)
class WebcamFrame:
    """BGR pixels timestamped at host receipt, not at sensor exposure."""

    bgr: Any
    sequence: int
    received_monotonic_s: float
    read_started_monotonic_s: float
    timestamp_source: str = "host_receipt_monotonic"


class Webcam:
    """OpenCV-backed webcam source."""

    def __init__(
        self,
        camera: int = 0,
        *,
        latest: bool = False,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
    ) -> None:
        self.camera = camera
        self.latest = latest
        self.width = _validate_capture_dimension(width, "width")
        self.height = _validate_capture_dimension(height, "height")
        self.fps = _validate_capture_fps(fps)
        self._capture: Capture | None = None
        self._latest_condition = threading.Condition()
        self._latest_frame: Any | None = None
        self._latest_packet: WebcamFrame | None = None
        self._latest_error: WebcamError | None = None
        self._latest_sequence = 0
        self._last_read_sequence = 0
        self._latest_stop = threading.Event()
        self._latest_thread: threading.Thread | None = None

    def open(self) -> None:
        """Open the configured camera."""

        if self._capture is not None:
            return

        self._capture = _open_cv_capture(self.camera)
        if not self._capture.isOpened():
            self.close()
            raise WebcamError(f"Could not open camera {self.camera}.")
        _set_capture_dimensions(self._capture, width=self.width, height=self.height)
        if self.fps is not None:
            _set_capture_property(self._capture, CAP_PROP_FPS, self.fps)
        if self.latest:
            _set_capture_property(self._capture, CAP_PROP_BUFFERSIZE, 1)
            self._start_latest_reader()

    def read(self) -> Any:
        """Read one frame from the camera."""

        return self.read_frame().bgr

    def read_frame(self, *, timeout_s: float | None = None) -> WebcamFrame:
        """Read a numbered frame; bound waiting when using the latest reader."""

        if timeout_s is not None and (timeout_s <= 0 or not math.isfinite(timeout_s)):
            raise ValueError("timeout_s must be finite and positive.")
        if self._capture is None:
            self.open()

        assert self._capture is not None
        if self.latest:
            return self._read_latest_frame(timeout_s=timeout_s)

        started = time.monotonic()
        ok, frame = self._capture.read()
        received = time.monotonic()
        if not ok or frame is None:
            raise WebcamError(f"Could not read a frame from camera {self.camera}.")
        self._latest_sequence += 1
        return WebcamFrame(frame, self._latest_sequence, received, started)

    def close(self) -> None:
        """Release the camera if it is open."""

        self._stop_latest_reader()
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Webcam":
        self.open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _start_latest_reader(self) -> None:
        assert self._capture is not None
        self._latest_stop.clear()
        with self._latest_condition:
            self._latest_frame = None
            self._latest_packet = None
            self._latest_error = None
            self._latest_sequence = 0
            self._last_read_sequence = 0

        self._latest_thread = threading.Thread(
            target=self._latest_reader_loop,
            args=(self._capture,),
            name=f"ghand-webcam-{self.camera}-latest",
            daemon=True,
        )
        self._latest_thread.start()

    def _stop_latest_reader(self) -> None:
        self._latest_stop.set()
        with self._latest_condition:
            self._latest_condition.notify_all()

        thread = self._latest_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._latest_thread = None

    def _latest_reader_loop(self, capture: Capture) -> None:
        while not self._latest_stop.is_set():
            started = time.monotonic()
            ok, frame = capture.read()
            received = time.monotonic()
            if not ok or frame is None:
                if not self._latest_stop.is_set():
                    with self._latest_condition:
                        self._latest_error = WebcamError(
                            f"Could not read a frame from camera {self.camera}."
                        )
                        self._latest_condition.notify_all()
                return

            with self._latest_condition:
                self._latest_frame = frame
                self._latest_sequence += 1
                self._latest_packet = WebcamFrame(
                    frame, self._latest_sequence, received, started
                )
                self._latest_condition.notify_all()

    def _read_latest_frame(self, *, timeout_s: float | None = None) -> WebcamFrame:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._latest_condition:
            while (
                self._latest_error is None
                and not self._latest_stop.is_set()
                and self._latest_sequence <= self._last_read_sequence
            ):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise WebcamError(f"Timed out waiting for camera {self.camera}.")
                self._latest_condition.wait(
                    timeout=0.5 if remaining is None else min(0.5, remaining)
                )

            if self._latest_error is not None:
                raise self._latest_error
            if self._latest_stop.is_set():
                raise WebcamError(f"Could not read a frame from camera {self.camera}.")
            if self._latest_packet is None:
                raise WebcamError(f"Could not read a frame from camera {self.camera}.")

            self._last_read_sequence = self._latest_sequence
            return self._latest_packet


def check_webcam(
    camera: int = 0,
    capture: Capture | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
) -> WebcamInfo:
    """Open a webcam and read a single frame to verify basic capture works."""

    owns_capture = capture is None
    capture = capture or _open_cv_capture(camera)
    requested_width = _validate_capture_dimension(width, "width")
    requested_height = _validate_capture_dimension(height, "height")
    requested_fps = _validate_capture_fps(fps)

    try:
        if not capture.isOpened():
            raise WebcamError(f"Could not open camera {camera}.")
        _set_capture_dimensions(
            capture,
            width=requested_width,
            height=requested_height,
        )
        if requested_fps is not None:
            _set_capture_property(capture, CAP_PROP_FPS, requested_fps)

        ok, frame = capture.read()
        if not ok or frame is None:
            raise WebcamError(f"Could not read a frame from camera {camera}.")

        frame_height, frame_width, channels = _frame_shape(frame)
        return WebcamInfo(
            camera=camera,
            width=int(capture.get(CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(CAP_PROP_FPS)),
            frame_width=frame_width,
            frame_height=frame_height,
            channels=channels,
            backend=_backend_name(capture),
        )
    finally:
        if owns_capture:
            capture.release()


def _open_cv_capture(camera: int) -> Capture:
    import cv2

    return cv2.VideoCapture(camera)


def _set_capture_property(capture: Capture, prop_id: int, value: float) -> None:
    setter = getattr(capture, "set", None)
    if setter is None:
        return

    try:
        setter(prop_id, value)
    except Exception:
        return


def _set_capture_dimensions(
    capture: Capture,
    *,
    width: int | None,
    height: int | None,
) -> None:
    if width is not None:
        _set_capture_property(capture, CAP_PROP_FRAME_WIDTH, float(width))
    if height is not None:
        _set_capture_property(capture, CAP_PROP_FRAME_HEIGHT, float(height))


def _validate_capture_dimension(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{label} must be positive.")
    return resolved


def _validate_capture_fps(value: float | None) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if resolved <= 0:
        raise ValueError("fps must be positive.")
    return resolved


def _frame_shape(frame: Any) -> tuple[int, int, int]:
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        raise WebcamError("Camera returned a frame without an image-like shape.")

    height = int(shape[0])
    width = int(shape[1])
    channels = int(shape[2]) if len(shape) >= 3 else 1
    return height, width, channels


def _backend_name(capture: Capture) -> str | None:
    get_backend_name = getattr(capture, "getBackendName", None)
    if get_backend_name is None:
        return None

    try:
        return str(get_backend_name())
    except Exception:
        return None
