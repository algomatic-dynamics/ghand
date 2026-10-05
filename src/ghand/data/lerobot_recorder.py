"""Streaming LeRobotDataset recorder for live ghand data capture."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ghand.data.lerobot_export import (
    FrameDict,
    _features_use_videos,
    _load_lerobot_dataset_class,
    prepare_lerobot_frame,
)
from ghand.data.lerobot_schema import FeatureDict


class LeRobotStreamingRecorder:
    """Write frames into a LeRobotDataset one episode at a time."""

    def __init__(
        self,
        *,
        dataset: Any,
        push_to_hub: bool = False,
        push_to_hub_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.push_to_hub = push_to_hub
        self.push_to_hub_kwargs = dict(push_to_hub_kwargs or {})
        self.frames_in_episode = 0
        self.total_frames = 0
        self.saved_episodes = 0
        self.closed = False

    @classmethod
    def create(
        cls,
        *,
        repo_id: str,
        root: str | Path | None,
        fps: int,
        features: FeatureDict,
        robot_type: str = "ghand_hand",
        push_to_hub: bool = False,
        push_to_hub_kwargs: dict[str, Any] | None = None,
        dataset_cls: type | None = None,
        use_videos: bool | None = None,
        dataset_create_kwargs: dict[str, Any] | None = None,
    ) -> LeRobotStreamingRecorder:
        """Create a local LeRobotDataset and wrap it for streaming writes."""

        if not repo_id:
            raise ValueError("repo_id must not be empty.")
        if fps <= 0:
            raise ValueError("fps must be positive.")

        resolved_dataset_cls = dataset_cls or _load_lerobot_dataset_class()
        resolved_features = deepcopy(features)
        resolved_use_videos = (
            _features_use_videos(resolved_features) if use_videos is None else use_videos
        )
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
        return cls(
            dataset=dataset,
            push_to_hub=push_to_hub,
            push_to_hub_kwargs=push_to_hub_kwargs,
        )

    def add_frame(self, frame: FrameDict) -> None:
        """Append one frame to the currently open episode."""

        self._raise_if_closed()
        self.dataset.add_frame(prepare_lerobot_frame(frame))
        self.frames_in_episode += 1
        self.total_frames += 1

    def save_episode(self) -> None:
        """Finalize the current episode and start a new one."""

        self._raise_if_closed()
        if self.frames_in_episode <= 0:
            raise ValueError("Cannot save an empty episode.")
        self.dataset.save_episode()
        self.saved_episodes += 1
        self.frames_in_episode = 0

    def discard_episode(self) -> None:
        """Discard frames accumulated for the currently open episode."""

        self._raise_if_closed()
        if self.frames_in_episode <= 0:
            return
        clear_episode_buffer = getattr(self.dataset, "clear_episode_buffer", None)
        if not callable(clear_episode_buffer):
            raise RuntimeError("LeRobotDataset does not support clearing episode buffers.")
        clear_episode_buffer()
        self.total_frames -= self.frames_in_episode
        self.frames_in_episode = 0

    def close(self, *, save_pending_episode: bool = True) -> None:
        """Save pending frames, optionally push to Hub, and finalize the dataset."""

        if self.closed:
            return
        try:
            if save_pending_episode and self.frames_in_episode > 0:
                self.save_episode()
            if self.push_to_hub:
                self.dataset.push_to_hub(**self.push_to_hub_kwargs)
            finalize = getattr(self.dataset, "finalize", None)
            if callable(finalize):
                finalize()
        finally:
            self.closed = True

    def __enter__(self) -> LeRobotStreamingRecorder:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("LeRobotStreamingRecorder is closed.")
