import pytest

from ghand.teleop.hand_pose import HandDetection, HandFrame, Landmark
from ghand.teleop.recorder import read_hand_frames_jsonl, write_hand_frames_jsonl


def test_hand_frame_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "hands.jsonl"
    frames = (
        HandFrame(
            timestamp_ns=123,
            image_width=640,
            image_height=480,
            hands=(
                HandDetection(
                    handedness="Right",
                    score=0.9,
                    landmarks=tuple(Landmark(0.1, 0.2, 0.3) for _ in range(21)),
                ),
            ),
        ),
        HandFrame(timestamp_ns=124, image_width=640, image_height=480, hands=()),
    )

    write_hand_frames_jsonl(path, frames)

    assert tuple(read_hand_frames_jsonl(path)) == frames


def test_read_hand_frames_jsonl_reports_invalid_line(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"timestamp_ns": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        tuple(read_hand_frames_jsonl(path))
