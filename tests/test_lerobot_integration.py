from pathlib import Path

import numpy as np
import pytest

from ghand.data import CameraSpec, LeRobotStreamingRecorder, write_lerobot_episodes
from ghand.data.lerobot_schema import ACTION_KEY, FRONT_IMAGE_KEY, OBSERVATION_STATE_KEY, make_hand_lerobot_features


def test_real_lerobot_dataset_roundtrip_with_bimanual_multicamera(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = "local/ghand-roundtrip-frames"
    root = tmp_path / "dataset"
    features = make_hand_lerobot_features(
        sides=("left", "right"),
        cameras=(
            CameraSpec("front", (32, 48, 3), "image"),
            CameraSpec("external", (16, 24, 3), "image"),
        ),
        image_dtype="image",
        include_aux=False,
    )
    episodes = [
        _make_episode(frames_per_episode=2, state_size=32, episode_index=0),
        _make_episode(frames_per_episode=2, state_size=32, episode_index=1),
    ]

    write_lerobot_episodes(
        repo_id=repo_id,
        root=root,
        fps=5,
        features=features,
        episodes=episodes,
        use_videos=False,
        dataset_create_kwargs={"image_writer_threads": 0},
    )

    loaded = LeRobotDataset(repo_id=repo_id, root=root)

    assert len(loaded) == 4
    assert loaded.features[OBSERVATION_STATE_KEY]["shape"] == (32,)
    assert loaded.features[ACTION_KEY]["shape"] == (32,)
    assert loaded.features[FRONT_IMAGE_KEY]["shape"] == (32, 48, 3)
    assert loaded.features["observation.images.external"]["shape"] == (16, 24, 3)
    assert loaded.meta.total_episodes == 2


def test_real_lerobot_streaming_recorder_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = "local/ghand-stream-roundtrip"
    root = tmp_path / "stream_dataset"
    features = make_hand_lerobot_features(
        cameras=(
            CameraSpec("front", (32, 48, 3), "image"),
            CameraSpec("external", (16, 24, 3), "image"),
        ),
        image_dtype="image",
        include_aux=False,
    )
    frames = _make_episode(frames_per_episode=2, state_size=16, episode_index=0)

    with LeRobotStreamingRecorder.create(
        repo_id=repo_id,
        root=root,
        fps=5,
        features=features,
        use_videos=False,
        dataset_create_kwargs={"image_writer_threads": 0},
    ) as recorder:
        recorder.add_frame(frames[0])
        recorder.save_episode()
        recorder.add_frame(frames[1])

    loaded = LeRobotDataset(repo_id=repo_id, root=root)

    assert len(loaded) == 2
    assert loaded.features[OBSERVATION_STATE_KEY]["shape"] == (16,)
    assert loaded.meta.total_episodes == 2


def _make_episode(
    *,
    frames_per_episode: int,
    state_size: int,
    episode_index: int,
) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for frame_index in range(frames_per_episode):
        value = episode_index * 10 + frame_index
        frames.append(
            {
                "timestamp": frame_index / 5,
                "task": "roundtrip test",
                OBSERVATION_STATE_KEY: np.full((state_size,), value, dtype=np.float32),
                ACTION_KEY: np.full((state_size,), value + 0.5, dtype=np.float32),
                FRONT_IMAGE_KEY: np.full((32, 48, 3), value, dtype=np.uint8),
                "observation.images.external": np.full((16, 24, 3), value, dtype=np.uint8),
            }
        )
    return frames
