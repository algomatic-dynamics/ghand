from __future__ import annotations

from types import SimpleNamespace

import pytest

from ghand.teleop.mediapipe_hands import (
    HAND_LANDMARKER_MODEL_ENV,
    HandModelNotFoundError,
    _handedness_at,
    _to_landmarks,
    resolve_hand_landmarker_model,
)


def test_to_landmarks_converts_mediapipe_like_list() -> None:
    landmark_list = SimpleNamespace(
        landmark=[
            SimpleNamespace(x=0.1, y=0.2, z=0.3),
            SimpleNamespace(x=0.4, y=0.5, z=0.6),
        ]
    )

    landmarks = _to_landmarks(landmark_list)

    assert len(landmarks) == 2
    assert landmarks[0].x == 0.1
    assert landmarks[1].z == 0.6


def test_to_landmarks_converts_tasks_api_list() -> None:
    landmarks = _to_landmarks(
        [
            SimpleNamespace(x=0.1, y=0.2, z=0.3),
            SimpleNamespace(x=0.4, y=0.5, z=0.6),
        ]
    )

    assert len(landmarks) == 2
    assert landmarks[0].x == 0.1
    assert landmarks[1].z == 0.6


def test_handedness_at_converts_mediapipe_like_classification() -> None:
    handedness_list = [
        SimpleNamespace(
            classification=[SimpleNamespace(label="Left", score=0.75)],
        )
    ]

    assert _handedness_at(handedness_list, 0) == ("Left", 0.75)


def test_handedness_at_converts_tasks_api_category() -> None:
    handedness_list = [
        [SimpleNamespace(category_name="Right", score=0.95)],
    ]

    assert _handedness_at(handedness_list, 0) == ("Right", 0.95)


def test_handedness_at_defaults_when_missing() -> None:
    assert _handedness_at([], 0) == ("Unknown", 0.0)


def test_resolve_hand_landmarker_model_uses_explicit_path(tmp_path) -> None:
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"model")

    assert resolve_hand_landmarker_model(model) == model


def test_resolve_hand_landmarker_model_uses_env_path(tmp_path, monkeypatch) -> None:
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"model")
    monkeypatch.setenv(HAND_LANDMARKER_MODEL_ENV, str(model))

    assert resolve_hand_landmarker_model() == model


def test_resolve_hand_landmarker_model_raises_when_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv(HAND_LANDMARKER_MODEL_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with pytest.raises(HandModelNotFoundError):
        resolve_hand_landmarker_model("/missing/hand_landmarker.task")
