from pathlib import Path

import numpy as np
import pytest

from ghand.data import (
    ACTION_KEY,
    FRONT_IMAGE_KEY,
    OBSERVATION_STATE_KEY,
    LeRobotStreamingRecorder,
    make_hand_lerobot_features,
)


class FakeLeRobotDataset:
    create_calls: list[dict] = []
    instances: list["FakeLeRobotDataset"] = []

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.saved_episodes = 0
        self.push_calls: list[dict] = []
        self.finalized = False

    @classmethod
    def reset(cls) -> None:
        cls.create_calls = []
        cls.instances = []

    @classmethod
    def create(cls, **kwargs):
        cls.create_calls.append(kwargs)
        dataset = cls()
        cls.instances.append(dataset)
        return dataset

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved_episodes += 1

    def clear_episode_buffer(self) -> None:
        self.frames.clear()

    def push_to_hub(self, **kwargs) -> None:
        self.push_calls.append(kwargs)

    def finalize(self) -> None:
        self.finalized = True


def test_streaming_recorder_creates_dataset_and_saves_episodes(tmp_path: Path) -> None:
    FakeLeRobotDataset.reset()
    features = make_hand_lerobot_features(front_image_shape=(2, 2, 3), image_dtype="image")
    recorder = LeRobotStreamingRecorder.create(
        repo_id="local/stream",
        root=tmp_path / "dataset",
        fps=15,
        features=features,
        dataset_cls=FakeLeRobotDataset,
        use_videos=False,
        dataset_create_kwargs={"image_writer_threads": 0},
    )
    first = _frame(value=1)
    second = _frame(value=2)

    recorder.add_frame(first)
    recorder.add_frame(second)
    recorder.save_episode()
    recorder.close()

    assert FakeLeRobotDataset.create_calls == [
        {
            "repo_id": "local/stream",
            "fps": 15,
            "features": features,
            "root": tmp_path / "dataset",
            "robot_type": "ghand_hand",
            "use_videos": False,
            "image_writer_threads": 0,
        }
    ]
    dataset = FakeLeRobotDataset.instances[0]
    assert dataset.frames == [
        {key: value for key, value in first.items() if key != "timestamp"},
        {key: value for key, value in second.items() if key != "timestamp"},
    ]
    assert dataset.saved_episodes == 1
    assert dataset.finalized
    assert recorder.total_frames == 2
    assert recorder.saved_episodes == 1
    assert recorder.frames_in_episode == 0
    assert recorder.closed


def test_streaming_recorder_can_discard_current_episode(tmp_path: Path) -> None:
    FakeLeRobotDataset.reset()
    features = make_hand_lerobot_features(front_image_shape=(2, 2, 3), image_dtype="image")
    recorder = LeRobotStreamingRecorder.create(
        repo_id="local/discard-current",
        root=tmp_path / "dataset",
        fps=30,
        features=features,
        dataset_cls=FakeLeRobotDataset,
    )

    recorder.add_frame(_frame(value=1))
    recorder.add_frame(_frame(value=2))
    recorder.discard_episode()

    dataset = FakeLeRobotDataset.instances[0]
    assert dataset.frames == []
    assert dataset.saved_episodes == 0
    assert recorder.total_frames == 0
    assert recorder.frames_in_episode == 0
    recorder.close()


def test_streaming_recorder_close_saves_pending_episode_and_pushes(tmp_path: Path) -> None:
    FakeLeRobotDataset.reset()
    features = make_hand_lerobot_features(front_image_shape=(2, 2, 3), image_dtype="video")

    with LeRobotStreamingRecorder.create(
        repo_id="local/stream-video",
        root=tmp_path / "dataset",
        fps=30,
        features=features,
        dataset_cls=FakeLeRobotDataset,
        push_to_hub=True,
        push_to_hub_kwargs={"private": True},
    ) as recorder:
        recorder.add_frame(_frame(value=1))

    dataset = FakeLeRobotDataset.instances[0]
    assert FakeLeRobotDataset.create_calls[0]["use_videos"] is True
    assert dataset.saved_episodes == 1
    assert dataset.push_calls == [{"private": True}]
    assert dataset.finalized
    assert recorder.closed


def test_streaming_recorder_can_discard_pending_episode_on_close(tmp_path: Path) -> None:
    FakeLeRobotDataset.reset()
    features = make_hand_lerobot_features(front_image_shape=(2, 2, 3), image_dtype="image")
    recorder = LeRobotStreamingRecorder.create(
        repo_id="local/discard",
        root=tmp_path / "dataset",
        fps=30,
        features=features,
        dataset_cls=FakeLeRobotDataset,
    )

    recorder.add_frame(_frame(value=1))
    recorder.close(save_pending_episode=False)

    assert FakeLeRobotDataset.instances[0].saved_episodes == 0
    assert recorder.frames_in_episode == 1
    assert recorder.closed


def test_streaming_recorder_validates_lifecycle_and_args(tmp_path: Path) -> None:
    features = make_hand_lerobot_features(front_image_shape=(2, 2, 3), image_dtype="image")
    with pytest.raises(ValueError, match="repo_id"):
        LeRobotStreamingRecorder.create(
            repo_id="",
            root=tmp_path / "dataset",
            fps=30,
            features=features,
            dataset_cls=FakeLeRobotDataset,
        )
    with pytest.raises(ValueError, match="fps"):
        LeRobotStreamingRecorder.create(
            repo_id="local/bad-fps",
            root=tmp_path / "dataset",
            fps=0,
            features=features,
            dataset_cls=FakeLeRobotDataset,
        )

    recorder = LeRobotStreamingRecorder.create(
        repo_id="local/lifecycle",
        root=tmp_path / "dataset",
        fps=30,
        features=features,
        dataset_cls=FakeLeRobotDataset,
    )
    with pytest.raises(ValueError, match="empty episode"):
        recorder.save_episode()
    recorder.close()
    with pytest.raises(RuntimeError, match="closed"):
        recorder.add_frame(_frame(value=1))


def _frame(*, value: int) -> dict[str, object]:
    return {
        "timestamp": value / 30,
        "task": "stream test",
        OBSERVATION_STATE_KEY: np.full((16,), value, dtype=np.float32),
        ACTION_KEY: np.full((16,), value + 0.5, dtype=np.float32),
        FRONT_IMAGE_KEY: np.full((2, 2, 3), value, dtype=np.uint8),
    }
