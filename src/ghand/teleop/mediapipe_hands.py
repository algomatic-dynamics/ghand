"""MediaPipe hand tracking integration."""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from ghand.teleop.hand_pose import HAND_CONNECTIONS, HandDetection, HandFrame, Landmark

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_FILENAME = "hand_landmarker.task"
HAND_LANDMARKER_MODEL_ENV = "GHAND_HAND_LANDMARKER_MODEL"


class HandModelNotFoundError(FileNotFoundError):
    """Raised when the MediaPipe hand landmarker model cannot be found."""


def _preprocess_for_hands(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    rgb = cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2RGB)
    return rgb


class MediaPipeHandTracker:
    """MediaPipe-based hand landmark capture."""

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.7,
    ) -> None:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        resolved_model_path = resolve_hand_landmarker_model(model_path)
        base_options = python.BaseOptions(model_asset_path=str(resolved_model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def track(self, frame_bgr: Any, timestamp_ns: int | None = None) -> HandFrame:
        """Track hands in one OpenCV BGR frame."""

        import mediapipe as mp

        timestamp_ns = timestamp_ns or time.time_ns()
        image_height, image_width = _frame_size(frame_bgr)
        frame_rgb = _preprocess_for_hands(frame_bgr)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect_for_video(
            image,
            timestamp_ms=timestamp_ns // 1_000_000,
        )
        hands: list[HandDetection] = []
        landmarks_list = result.hand_landmarks or []
        world_landmarks_list = result.hand_world_landmarks or []
        handedness_list = result.handedness or []

        for index, landmarks in enumerate(landmarks_list):
            handedness, score = _handedness_at(handedness_list, index)
            world_landmarks = (
                _to_landmarks(world_landmarks_list[index])
                if index < len(world_landmarks_list)
                else None
            )
            hands.append(
                HandDetection(
                    handedness=handedness,
                    score=score,
                    landmarks=_to_landmarks(landmarks),
                    world_landmarks=world_landmarks,
                )
            )

        return HandFrame(
            timestamp_ns=timestamp_ns,
            image_width=image_width,
            image_height=image_height,
            hands=tuple(hands),
        )

    def close(self) -> None:
        """Release tracker resources."""

        self._landmarker.close()

    def __enter__(self) -> "MediaPipeHandTracker":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def draw_hand_frame(frame_bgr: Any, hand_frame: HandFrame) -> Any:
    """Draw tracked landmarks on an OpenCV BGR frame."""

    import cv2

    for hand in hand_frame.hands:
        points = [
            (
                int(landmark.x * hand_frame.image_width),
                int(landmark.y * hand_frame.image_height),
            )
            for landmark in hand.landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame_bgr, points[start], points[end], (0, 180, 255), 2)
        for point in points:
            cv2.circle(frame_bgr, point, 3, (0, 255, 0), -1)

        label = f"{hand.handedness} {hand.score:.2f}"
        if points:
            cv2.putText(
                frame_bgr,
                label,
                points[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return frame_bgr


def default_hand_landmarker_model_path() -> Path:
    """Return the default cache path for the MediaPipe hand model."""

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "ghand" / HAND_LANDMARKER_MODEL_FILENAME


def resolve_hand_landmarker_model(model_path: Path | str | None = None) -> Path:
    """Resolve the hand landmarker model path from an option, env var, or cache."""

    candidates: list[Path] = []
    if model_path is not None:
        candidates.append(Path(model_path).expanduser())

    env_model_path = os.environ.get(HAND_LANDMARKER_MODEL_ENV)
    if env_model_path:
        candidates.append(Path(env_model_path).expanduser())

    candidates.append(Path("assets/models") / HAND_LANDMARKER_MODEL_FILENAME)
    candidates.append(default_hand_landmarker_model_path())

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise HandModelNotFoundError(
        "MediaPipe hand landmarker model was not found. "
        "Run `uv run ghand download-hand-model`, pass `--model PATH`, or set "
        f"{HAND_LANDMARKER_MODEL_ENV}. Searched: {searched}"
    )


def download_hand_landmarker_model(
    destination: Path | str | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Download the MediaPipe hand landmarker model to a local cache path."""

    destination_path = (
        Path(destination).expanduser()
        if destination is not None
        else default_hand_landmarker_model_path()
    )
    if destination_path.exists() and not overwrite:
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, temporary_path)
    temporary_path.replace(destination_path)
    return destination_path


def _frame_size(frame: Any) -> tuple[int, int]:
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("Expected an image-like frame with height and width.")
    return int(shape[0]), int(shape[1])


def _to_landmarks(landmark_list: Any) -> tuple[Landmark, ...]:
    landmarks = getattr(landmark_list, "landmark", landmark_list)
    return tuple(
        Landmark(x=float(landmark.x), y=float(landmark.y), z=float(landmark.z))
        for landmark in landmarks
    )


def _handedness_at(handedness_list: list[Any], index: int) -> tuple[str, float]:
    if index >= len(handedness_list):
        return "Unknown", 0.0

    handedness = handedness_list[index]
    classifications = getattr(
        handedness,
        "classification",
        handedness if isinstance(handedness, (list, tuple)) else [],
    )
    if not classifications:
        return "Unknown", 0.0

    classification = classifications[0]
    label = getattr(
        classification,
        "category_name",
        getattr(classification, "label", "Unknown"),
    )
    return str(label), float(classification.score)
