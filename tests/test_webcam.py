from __future__ import annotations

import threading
import time

import pytest

import ghand.teleop.webcam as webcam_module
from ghand.teleop.webcam import (
    CAP_PROP_BUFFERSIZE,
    CAP_PROP_FPS,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    Webcam,
    WebcamError,
    check_webcam,
)


class FakeFrame:
    def __init__(self, shape: tuple[int, ...], *, label: str = "") -> None:
        self.shape = shape
        self.label = label


class FakeCapture:
    def __init__(self, opened: bool = True, readable: bool = True) -> None:
        self.opened = opened
        self.readable = readable
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, FakeFrame | None]:
        if not self.readable:
            return False, None
        return True, FakeFrame((480, 640, 3))

    def get(self, prop_id: int) -> float:
        values = {
            CAP_PROP_FRAME_WIDTH: 640.0,
            CAP_PROP_FRAME_HEIGHT: 480.0,
            CAP_PROP_FPS: 30.0,
        }
        return values[prop_id]

    def set(self, prop_id: int, value: float) -> bool:
        self.set_calls.append((prop_id, value))
        return True

    def release(self) -> None:
        self.released = True

    def getBackendName(self) -> str:
        return "FAKE"


def test_check_webcam_returns_capture_info() -> None:
    capture = FakeCapture()

    info = check_webcam(camera=2, capture=capture)

    assert info.camera == 2
    assert info.width == 640
    assert info.height == 480
    assert info.fps == 30.0
    assert info.frame_width == 640
    assert info.frame_height == 480
    assert info.channels == 3
    assert info.backend == "FAKE"
    assert not capture.released


def test_check_webcam_applies_requested_capture_settings() -> None:
    capture = FakeCapture()

    check_webcam(camera=2, capture=capture, width=640, height=480, fps=30.0)

    assert capture.set_calls == [
        (CAP_PROP_FRAME_WIDTH, 640),
        (CAP_PROP_FRAME_HEIGHT, 480),
        (CAP_PROP_FPS, 30.0),
    ]


def test_check_webcam_raises_when_camera_is_not_opened() -> None:
    with pytest.raises(WebcamError, match="Could not open camera 0"):
        check_webcam(capture=FakeCapture(opened=False))


def test_check_webcam_raises_when_frame_cannot_be_read() -> None:
    with pytest.raises(WebcamError, match="Could not read a frame from camera 0"):
        check_webcam(capture=FakeCapture(readable=False))


class SequencedCapture(FakeCapture):
    def __init__(self, frames: list[FakeFrame]) -> None:
        super().__init__()
        self.frames = frames
        self.index = 0
        self.read_all = threading.Event()
        self.release_requested = threading.Event()

    def read(self) -> tuple[bool, FakeFrame | None]:
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            if self.index == len(self.frames):
                self.read_all.set()
            return True, frame

        self.release_requested.wait(timeout=0.005)
        return True, self.frames[-1]

    def release(self) -> None:
        self.release_requested.set()
        super().release()


def test_webcam_latest_mode_returns_newest_frame_and_sets_low_buffer(
    monkeypatch,
) -> None:
    capture = SequencedCapture(
        [
            FakeFrame((480, 640, 3), label="old"),
            FakeFrame((480, 640, 3), label="newest"),
        ]
    )
    monkeypatch.setattr(webcam_module, "_open_cv_capture", lambda _camera: capture)

    with Webcam(camera=2, latest=True, width=320, height=240) as webcam:
        assert capture.read_all.wait(timeout=1.0)
        frame = webcam.read()

    assert frame.label == "newest"
    assert (CAP_PROP_FRAME_WIDTH, 320.0) in capture.set_calls
    assert (CAP_PROP_FRAME_HEIGHT, 240.0) in capture.set_calls
    assert (CAP_PROP_BUFFERSIZE, 1) in capture.set_calls
    assert capture.released


def test_webcam_applies_requested_capture_settings(monkeypatch) -> None:
    capture = SequencedCapture([FakeFrame((480, 640, 3), label="frame")])
    monkeypatch.setattr(webcam_module, "_open_cv_capture", lambda _camera: capture)

    with Webcam(camera=2, latest=True, width=640, height=480, fps=30.0):
        assert capture.read_all.wait(timeout=1.0)

    assert capture.set_calls[:3] == [
        (CAP_PROP_FRAME_WIDTH, 640),
        (CAP_PROP_FRAME_HEIGHT, 480),
        (CAP_PROP_FPS, 30.0),
    ]


def test_webcam_packets_preserve_host_receipt_time_and_sequence(monkeypatch) -> None:
    capture = FakeCapture()
    monkeypatch.setattr(webcam_module, "_open_cv_capture", lambda _: capture)
    with Webcam() as camera:
        before = time.monotonic()
        first = camera.read_frame()
        second = camera.read_frame()
        assert first.sequence == 1 and second.sequence == 2
        assert before <= first.read_started_monotonic_s <= first.received_monotonic_s
        assert first.received_monotonic_s <= second.read_started_monotonic_s
        assert second.timestamp_source == "host_receipt_monotonic"


def test_latest_camera_missing_frame_has_bounded_wait(monkeypatch) -> None:
    release = threading.Event()

    class StalledCapture(FakeCapture):
        def read(self):
            release.wait(timeout=2)
            return False, None

    monkeypatch.setattr(webcam_module, "_open_cv_capture", lambda _: StalledCapture())
    camera = Webcam(latest=True)
    try:
        start = time.monotonic()
        with pytest.raises(WebcamError, match="Timed out"):
            camera.read_frame(timeout_s=0.02)
        assert time.monotonic() - start < 0.5
    finally:
        release.set()
        camera.close()
