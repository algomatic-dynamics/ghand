"""Hand-pose recording utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from ghand.teleop.hand_pose import HandFrame


class HandDataRecorder:
    """Minimal JSONL recorder for webcam hand-pose data."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._file = None

    def __enter__(self) -> "HandDataRecorder":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def write(self, frame: HandFrame) -> None:
        """Write one hand frame as one JSONL record."""

        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", encoding="utf-8")
        self._file.write(json.dumps(frame.to_dict()) + "\n")

    def close(self) -> None:
        """Close the underlying file if open."""

        if self._file is not None:
            self._file.close()
            self._file = None


def read_hand_frames_jsonl(path: Path | str) -> Iterator[HandFrame]:
    """Yield hand frames from a JSONL file produced by ghand hand tracking."""

    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                yield HandFrame.from_dict(data)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid HandFrame JSONL record at line {line_number}.") from exc


def write_hand_frames_jsonl(path: Path | str, frames: Iterable[HandFrame]) -> None:
    """Write hand frames to a JSONL file."""

    with HandDataRecorder(path) as recorder:
        for frame in frames:
            recorder.write(frame)
