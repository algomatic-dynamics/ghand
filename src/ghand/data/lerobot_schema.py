"""LeRobot feature schema for ghand datasets."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from ghand.hand.joints import (
    HandSide,
    LEFT_HAND_SIDE,
    RIGHT_HAND_SIDE,
    DEFAULT_HAND_SIDES,
    JOINT_PRESENT_CURRENT_KEY,
    JOINT_CURRENT_LIMIT_KEY,
    JOINT_CURRENT_RESIDUAL_KEY,
    JOINT_CONTACT_KEY,
    JointComponent,
    validate_name_part,
    validate_hand_side,
    make_hand_joint_component,
    compose_joint_position_names,
)


DEFAULT_FPS: Final[int] = 30
TELEOP_HAND_SLOTS: Final[int] = 2
HAND_LANDMARK_COUNT: Final[int] = 21
LANDMARK_DIMS: Final[int] = 3
TELEOP_HAND_SLOT_NAMES: Final[tuple[str, str]] = (LEFT_HAND_SIDE, RIGHT_HAND_SIDE)

OBSERVATION_STATE_KEY: Final[str] = "observation.state"
ACTION_KEY: Final[str] = "action"
OBSERVATION_IMAGES_PREFIX: Final[str] = "observation.images"
DEFAULT_CAMERA_NAME: Final[str] = "front"
DEFAULT_IMAGE_DTYPE: Final[str] = "video"
FRONT_IMAGE_KEY: Final[str] = "observation.images.front"
AUX_TELEOP_LANDMARKS_KEY: Final[str] = "aux.teleop.landmarks"
AUX_TELEOP_WORLD_LANDMARKS_KEY: Final[str] = "aux.teleop.world_landmarks"
AUX_TELEOP_HANDEDNESS_KEY: Final[str] = "aux.teleop.handedness"
AUX_TELEOP_SCORE_KEY: Final[str] = "aux.teleop.score"
AUX_TELEOP_VALID_KEY: Final[str] = "aux.teleop.valid"

HANDEDNESS_UNKNOWN: Final[int] = 0
HANDEDNESS_LEFT: Final[int] = 1
HANDEDNESS_RIGHT: Final[int] = 2
HANDEDNESS_LABEL_TO_ID: Final[dict[str, int]] = {
    "Unknown": HANDEDNESS_UNKNOWN,
    "Left": HANDEDNESS_LEFT,
    "Right": HANDEDNESS_RIGHT,
}

DEFAULT_FRONT_IMAGE_SHAPE: Final[tuple[int, int, int]] = (480, 640, 3)

FeatureSpec = dict[str, object]
FeatureDict = dict[str, FeatureSpec]


@dataclass(frozen=True)
class CameraSpec:
    """Image or video feature for a LeRobot observation camera."""

    name: str
    shape: tuple[int, int, int] = DEFAULT_FRONT_IMAGE_SHAPE
    dtype: str = DEFAULT_IMAGE_DTYPE

    def __post_init__(self) -> None:
        validate_name_part(self.name, "camera name")
        _validate_camera_shape(self.shape)
        if self.dtype not in {"image", "video"}:
            raise ValueError("camera dtype must be 'image' or 'video'.")

    def feature_key(self) -> str:
        """Return the LeRobot feature key for this camera."""

        return camera_feature_key(self.name)


def _validate_camera_shape(shape: tuple[int, int, int]) -> None:
    if len(shape) != 3:
        raise ValueError("camera shape must be (height, width, channels).")
    if min(shape) <= 0:
        raise ValueError("camera shape dimensions must be positive.")
    if shape[2] != 3:
        raise ValueError("camera shape must have 3 channels.")


DEFAULT_CAMERA: Final[CameraSpec] = CameraSpec(DEFAULT_CAMERA_NAME, DEFAULT_FRONT_IMAGE_SHAPE, DEFAULT_IMAGE_DTYPE)
DEFAULT_CAMERAS: Final[tuple[CameraSpec, ...]] = (DEFAULT_CAMERA,)


def camera_feature_key(name: str) -> str:
    """Return the LeRobot observation image key for a camera name."""

    validate_name_part(name, "camera name")
    return f"{OBSERVATION_IMAGES_PREFIX}.{name}"


def _resolve_cameras(
    *,
    cameras: Sequence[CameraSpec] | None,
    front_image_shape: tuple[int, int, int],
    image_dtype: str,
) -> tuple[CameraSpec, ...]:
    if cameras is None:
        return (CameraSpec(DEFAULT_CAMERA_NAME, front_image_shape, image_dtype),)
    return tuple(cameras)


def make_camera_features(
    cameras: Sequence[CameraSpec] | None = None,
    *,
    front_image_shape: tuple[int, int, int] = DEFAULT_FRONT_IMAGE_SHAPE,
    image_dtype: str = DEFAULT_IMAGE_DTYPE,
) -> FeatureDict:
    """Return LeRobot image/video features for camera specs."""

    resolved_cameras = _resolve_cameras(
        cameras=cameras,
        front_image_shape=front_image_shape,
        image_dtype=image_dtype,
    )
    features: FeatureDict = {}
    for camera in resolved_cameras:
        key = camera.feature_key()
        if key in features:
            raise ValueError(f"Duplicate camera feature key: {key!r}.")
        features[key] = {
            "dtype": camera.dtype,
            "shape": camera.shape,
            "names": ["height", "width", "channels"],
        }
    return features


def make_hand_joint_position_names(sides: Sequence[HandSide | str] = DEFAULT_HAND_SIDES) -> tuple[str, ...]:
    """Return concatenated hand joint channel names in the requested side order."""

    if not sides:
        raise ValueError("At least one hand side must be provided.")

    seen: set[HandSide] = set()
    components: list[JointComponent] = []
    for side in sides:
        normalized_side = validate_hand_side(side)
        if normalized_side in seen:
            raise ValueError(f"Duplicate hand side: {normalized_side!r}.")
        seen.add(normalized_side)
        components.append(make_hand_joint_component(normalized_side))
    return compose_joint_position_names(components)


def make_hand_joint_present_current_names(
    sides: Sequence[HandSide | str] = DEFAULT_HAND_SIDES,
) -> tuple[str, ...]:
    """Return concatenated present-current channel names in the requested side order."""

    return _make_hand_joint_names_for_key(sides, JOINT_PRESENT_CURRENT_KEY)


def make_hand_joint_current_limit_action_names(
    sides: Sequence[HandSide | str] = DEFAULT_HAND_SIDES,
) -> tuple[str, ...]:
    """Return concatenated current-limit action channel names in the requested side order."""

    return _make_hand_joint_names_for_key(sides, JOINT_CURRENT_LIMIT_KEY)


def make_hand_joint_current_residual_names(
    sides: Sequence[HandSide | str] = DEFAULT_HAND_SIDES,
) -> tuple[str, ...]:
    """Return concatenated current-residual observation channel names."""

    return _make_hand_joint_names_for_key(sides, JOINT_CURRENT_RESIDUAL_KEY)


def make_hand_joint_contact_names(
    sides: Sequence[HandSide | str] = DEFAULT_HAND_SIDES,
) -> tuple[str, ...]:
    """Return concatenated contact-flag observation channel names."""

    return _make_hand_joint_names_for_key(sides, JOINT_CONTACT_KEY)


def make_hand_position_current_state_names(
    sides: Sequence[HandSide | str] = DEFAULT_HAND_SIDES,
    *,
    include_current_diagnostics: bool = False,
) -> tuple[str, ...]:
    """Return state names for measured joint radians plus present current."""

    names = (
        make_hand_joint_position_names(sides)
        + make_hand_joint_present_current_names(sides)
    )
    if include_current_diagnostics:
        names += (
            make_hand_joint_current_residual_names(sides)
            + make_hand_joint_contact_names(sides)
        )
    return names


def make_hand_position_current_action_names(
    sides: Sequence[HandSide | str] = DEFAULT_HAND_SIDES,
) -> tuple[str, ...]:
    """Return action names for target joint radians plus current limits."""

    return (
        make_hand_joint_position_names(sides)
        + make_hand_joint_current_limit_action_names(sides)
    )


def _make_hand_joint_names_for_key(
    sides: Sequence[HandSide | str],
    position_key: str,
) -> tuple[str, ...]:
    if not sides:
        raise ValueError("At least one hand side must be provided.")

    seen: set[HandSide] = set()
    components: list[JointComponent] = []
    for side in sides:
        normalized_side = validate_hand_side(side)
        if normalized_side in seen:
            raise ValueError(f"Duplicate hand side: {normalized_side!r}.")
        seen.add(normalized_side)
        components.append(make_hand_joint_component(normalized_side, position_key=position_key))
    return compose_joint_position_names(components)


def make_joint_lerobot_features(
    *,
    components: Sequence[JointComponent],
    cameras: Sequence[CameraSpec] | None = None,
    front_image_shape: tuple[int, int, int] = DEFAULT_FRONT_IMAGE_SHAPE,
    image_dtype: str = DEFAULT_IMAGE_DTYPE,
    include_aux: bool = True,
) -> FeatureDict:
    """Return a fresh LeRobot feature dictionary for composed joint components."""

    joint_names = list(compose_joint_position_names(components))
    features: FeatureDict = {
        OBSERVATION_STATE_KEY: {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": list(joint_names),
        },
    }
    features.update(
        make_camera_features(
            cameras,
            front_image_shape=front_image_shape,
            image_dtype=image_dtype,
        )
    )

    if include_aux:
        features.update(make_aux_teleop_features())

    return features


def make_hand_lerobot_features(
    *,
    sides: Sequence[HandSide | str] = DEFAULT_HAND_SIDES,
    cameras: Sequence[CameraSpec] | None = None,
    front_image_shape: tuple[int, int, int] = DEFAULT_FRONT_IMAGE_SHAPE,
    image_dtype: str = DEFAULT_IMAGE_DTYPE,
    include_aux: bool = True,
) -> FeatureDict:
    """Return a fresh LeRobot feature dictionary for LEAP V1 KAIZEN hand datasets."""

    components = [make_hand_joint_component(side) for side in sides]
    return make_joint_lerobot_features(
        components=components,
        cameras=cameras,
        front_image_shape=front_image_shape,
        image_dtype=image_dtype,
        include_aux=include_aux,
    )


def make_hand_position_current_lerobot_features(
    *,
    sides: Sequence[HandSide | str] = DEFAULT_HAND_SIDES,
    cameras: Sequence[CameraSpec] | None = None,
    front_image_shape: tuple[int, int, int] = DEFAULT_FRONT_IMAGE_SHAPE,
    image_dtype: str = DEFAULT_IMAGE_DTYPE,
    include_aux: bool = True,
    include_current_diagnostics: bool = False,
) -> FeatureDict:
    """Return LeRobot features for position plus current-limited grasp actions."""

    state_names = list(
        make_hand_position_current_state_names(
            sides,
            include_current_diagnostics=include_current_diagnostics,
        )
    )
    action_names = list(make_hand_position_current_action_names(sides))
    features: FeatureDict = {
        OBSERVATION_STATE_KEY: {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
    }
    features.update(
        make_camera_features(
            cameras,
            front_image_shape=front_image_shape,
            image_dtype=image_dtype,
        )
    )
    if include_aux:
        features.update(make_aux_teleop_features())
    return features


def make_aux_teleop_features() -> FeatureDict:
    """Return MediaPipe teleop auxiliary features."""

    return {
        AUX_TELEOP_LANDMARKS_KEY: {
            "dtype": "float32",
            "shape": (TELEOP_HAND_SLOTS, HAND_LANDMARK_COUNT, LANDMARK_DIMS),
            "names": None,
        },
        AUX_TELEOP_WORLD_LANDMARKS_KEY: {
            "dtype": "float32",
            "shape": (TELEOP_HAND_SLOTS, HAND_LANDMARK_COUNT, LANDMARK_DIMS),
            "names": None,
        },
        AUX_TELEOP_HANDEDNESS_KEY: {
            "dtype": "int64",
            "shape": (TELEOP_HAND_SLOTS,),
            "names": list(TELEOP_HAND_SLOT_NAMES),
        },
        AUX_TELEOP_SCORE_KEY: {
            "dtype": "float32",
            "shape": (TELEOP_HAND_SLOTS,),
            "names": list(TELEOP_HAND_SLOT_NAMES),
        },
        AUX_TELEOP_VALID_KEY: {
            "dtype": "int64",
            "shape": (TELEOP_HAND_SLOTS,),
            "names": list(TELEOP_HAND_SLOT_NAMES),
        },
    }


def make_lerobot_features(
    *,
    sides: Sequence[HandSide | str] = DEFAULT_HAND_SIDES,
    cameras: Sequence[CameraSpec] | None = None,
    front_image_shape: tuple[int, int, int] = DEFAULT_FRONT_IMAGE_SHAPE,
    image_dtype: str = DEFAULT_IMAGE_DTYPE,
    include_aux: bool = True,
) -> FeatureDict:
    """Return a fresh default LeRobot feature dictionary for LEAP V1 KAIZEN hand datasets."""

    return make_hand_lerobot_features(
        sides=sides,
        cameras=cameras,
        front_image_shape=front_image_shape,
        image_dtype=image_dtype,
        include_aux=include_aux,
    )


GHAND_LEROBOT_FEATURES: Final[FeatureDict] = make_hand_lerobot_features()


def copy_lerobot_features() -> FeatureDict:
    """Return a deep copy of the default ghand LeRobot features."""

    return deepcopy(GHAND_LEROBOT_FEATURES)
