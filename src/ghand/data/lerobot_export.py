"""Export ghand frame data as a LeRobotDataset."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ghand.data.lerobot_schema import FeatureDict


class LeRobotDependencyError(ImportError):
    """Raised when LeRobot export is requested without the optional dependency."""


FrameDict = dict[str, object]


def write_lerobot_episodes(
    *,
    repo_id: str,
    root: str | Path | None,
    fps: int,
    features: FeatureDict,
    episodes: list[list[FrameDict]],
    robot_type: str = "ghand_hand",
    push_to_hub: bool = False,
    push_to_hub_kwargs: dict[str, Any] | None = None,
    dataset_cls: type | None = None,
    use_videos: bool | None = None,
    dataset_create_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Write prebuilt frame episodes to a local LeRobotDataset."""

    _validate_export_args(repo_id=repo_id, fps=fps, episodes=episodes)
    resolved_dataset_cls = dataset_cls or _load_lerobot_dataset_class()
    resolved_features = deepcopy(features)
    resolved_use_videos = _features_use_videos(resolved_features) if use_videos is None else use_videos
    create_kwargs = dict(dataset_create_kwargs or {})

    dataset = resolved_dataset_cls.create(
        repo_id=repo_id,
        fps=fps,
        features=resolved_features,
        root=root,
        robot_type=robot_type,
        use_videos=resolved_use_videos,
        **create_kwargs,
    )

    for episode in episodes:
        if not episode:
            raise ValueError("Each episode must contain at least one frame.")
        for frame in episode:
            dataset.add_frame(prepare_lerobot_frame(frame))
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(**(push_to_hub_kwargs or {}))

    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()

    return dataset


def _load_lerobot_dataset_class() -> type:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise LeRobotDependencyError(
            "LeRobot export requires the optional 'lerobot' dependency. "
            "Install it with `uv run --extra lerobot ...` or `pip install ghand[lerobot]`."
        ) from exc
    return LeRobotDataset


def _features_use_videos(features: FeatureDict) -> bool:
    return any(feature.get("dtype") == "video" for feature in features.values())


def _validate_export_args(*, repo_id: str, fps: int, episodes: list[list[FrameDict]]) -> None:
    if not repo_id:
        raise ValueError("repo_id must not be empty.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if not episodes:
        raise ValueError("At least one episode must be provided.")


def prepare_lerobot_frame(frame: FrameDict) -> dict[str, object]:
    """Return a LeRobotDataset.add_frame-compatible copy of a frame."""

    lerobot_frame = dict(frame)
    # LeRobotDataset.add_frame in 0.4.x derives fixed-rate timestamps from
    # frame_index / fps. Passing an explicit timestamp is currently rejected by
    # validation even though the method can pop it later.
    lerobot_frame.pop("timestamp", None)
    return lerobot_frame
