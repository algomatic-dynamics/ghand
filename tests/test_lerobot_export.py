from pathlib import Path

import numpy as np
import pytest

from ghand.data.lerobot_export import (
    LeRobotDependencyError,
    write_lerobot_episodes,
)
from ghand.data.lerobot_schema import (
    ACTION_KEY,
    FRONT_IMAGE_KEY,
    OBSERVATION_STATE_KEY,
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

    def push_to_hub(self, **kwargs) -> None:
        self.push_calls.append(kwargs)

    def finalize(self) -> None:
        self.finalized = True


def _make_lerobot_episode(
    *,
    frames_per_episode: int,
    state_size: int = 16,
    image_shape: tuple[int, int, int] = (32, 48, 3),
    episode_index: int = 0,
) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for frame_index in range(frames_per_episode):
        value = episode_index * 10 + frame_index
        frames.append(
            {
                "timestamp": frame_index / 30,
                "task": "test task",
                OBSERVATION_STATE_KEY: np.full((state_size,), value, dtype=np.float32),
                ACTION_KEY: np.full((state_size,), value + 0.5, dtype=np.float32),
                FRONT_IMAGE_KEY: np.full(image_shape, value, dtype=np.uint8),
            }
        )
    return frames


def test_write_lerobot_episodes_uses_dataset_create_and_saves_each_episode(tmp_path: Path) -> None:
    FakeLeRobotDataset.reset()
    features = make_hand_lerobot_features(front_image_shape=(32, 48, 3), image_dtype="image")
    episodes = [
        _make_lerobot_episode(frames_per_episode=2),
        _make_lerobot_episode(frames_per_episode=3, episode_index=1),
    ]

    dataset = write_lerobot_episodes(
        repo_id="local/ghand-frames",
        root=tmp_path / "dataset",
        fps=15,
        features=features,
        episodes=episodes,
        robot_type="ghand_hand",
        dataset_cls=FakeLeRobotDataset,
        use_videos=False,
        dataset_create_kwargs={"image_writer_threads": 0},
    )

    assert dataset is FakeLeRobotDataset.instances[0]
    assert FakeLeRobotDataset.create_calls == [
        {
            "repo_id": "local/ghand-frames",
            "fps": 15,
            "features": features,
            "root": tmp_path / "dataset",
            "robot_type": "ghand_hand",
            "use_videos": False,
            "image_writer_threads": 0,
        }
    ]
    assert len(dataset.frames) == 5
    assert dataset.saved_episodes == 2
    assert dataset.finalized
    assert dataset.frames[0]["task"] == episodes[0][0]["task"]
    assert "timestamp" not in dataset.frames[0]
    assert dataset.frames[0][OBSERVATION_STATE_KEY].shape == (16,)
    assert dataset.frames[0][ACTION_KEY].shape == (16,)


def test_write_lerobot_episodes_inferrs_use_videos_from_features(tmp_path: Path) -> None:
    FakeLeRobotDataset.reset()
    features = make_hand_lerobot_features(front_image_shape=(32, 48, 3), image_dtype="video")
    episodes = [_make_lerobot_episode(frames_per_episode=1)]

    write_lerobot_episodes(
        repo_id="local/video-frames",
        root=tmp_path / "dataset",
        fps=30,
        features=features,
        episodes=episodes,
        dataset_cls=FakeLeRobotDataset,
    )

    assert FakeLeRobotDataset.create_calls[0]["use_videos"] is True


def test_write_lerobot_episodes_can_push_to_hub(tmp_path: Path) -> None:
    FakeLeRobotDataset.reset()
    features = make_hand_lerobot_features(front_image_shape=(32, 48, 3), image_dtype="image")
    episodes = [_make_lerobot_episode(frames_per_episode=1)]

    dataset = write_lerobot_episodes(
        repo_id="local/push-frames",
        root=tmp_path / "dataset",
        fps=30,
        features=features,
        episodes=episodes,
        dataset_cls=FakeLeRobotDataset,
        push_to_hub=True,
        push_to_hub_kwargs={"private": True, "tags": ["test"]},
    )

    assert dataset.push_calls == [{"private": True, "tags": ["test"]}]


def test_write_lerobot_episodes_validates_arguments(tmp_path: Path) -> None:
    features = make_hand_lerobot_features(front_image_shape=(32, 48, 3), image_dtype="image")
    episode = _make_lerobot_episode(frames_per_episode=1)

    invalid_kwargs = (
        {"repo_id": ""},
        {"fps": 0},
        {"episodes": []},
        {"episodes": [[]]},
    )
    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            write_lerobot_episodes(
                repo_id=kwargs.get("repo_id", "local/invalid"),
                root=tmp_path / "dataset",
                fps=kwargs.get("fps", 30),
                features=features,
                episodes=kwargs.get("episodes", [episode]),
                dataset_cls=FakeLeRobotDataset,
            )


def test_write_lerobot_episodes_raises_clear_error_when_lerobot_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("lerobot"):
            raise ImportError("missing lerobot")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    features = make_hand_lerobot_features(front_image_shape=(32, 48, 3), image_dtype="image")
    episode = _make_lerobot_episode(frames_per_episode=1)

    with pytest.raises(LeRobotDependencyError, match="optional 'lerobot' dependency"):
        write_lerobot_episodes(
            repo_id="local/missing-lerobot",
            root=tmp_path / "dataset",
            fps=30,
            features=features,
            episodes=[episode],
        )
