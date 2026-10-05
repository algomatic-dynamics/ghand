"""Export the packaged LEAP V1 KAIZEN URDF into MuJoCo assets for RL experiments."""

from __future__ import annotations

import importlib
import math
import os
import shutil
import tempfile
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations, product
from pathlib import Path
from typing import Final
from xml.etree import ElementTree as ET

import numpy as np

from ghand.hand import (
    default_right_hand_urdf_path,
    DISPLAY_HAND_ROTATION_QUAT_WXYZ,
)
from ghand.sim.mujoco.cube_faces import CUBE_SCENE_FACES as CUBE_FACES
from ghand.sim.mujoco.collision_assets import (
    MujocoCollisionAssetError,
    MujocoCollisionAssetManifest,
    default_mujoco_collision_manifest_path,
    load_mujoco_collision_asset_manifest,
    sha256_file,
)
from ghand.sim.mujoco.profile import (
    DEFAULT_MUJOCO_PHYSICS_PROFILE,
    HAND_SELF_COLLISION_DISABLED,
    HAND_SELF_COLLISION_SAFE_CROSS_LINK_V1,
    MujocoPhysicsProfile,
    mujoco_joint_ranges_from_urdf,
)

DEFAULT_HAND_XML_NAME = "ghand_right_hand.xml"
DEFAULT_CUBE_SCENE_XML_NAME = "cube_face_up_scene.xml"
DEFAULT_TURN_HANDLE_SCENE_XML_NAME = "turn_handle_scene.xml"
DEFAULT_POLICY_CAMERA_NAME = "policy_front"
DEFAULT_MODEL_NAME = "ghand_right_hand"
DEFAULT_SCENE_MODEL_NAME = "ghand_cube_face_up"
DEFAULT_TURN_HANDLE_SCENE_MODEL_NAME = "ghand_turn_handle"
DEFAULT_JOINT_DAMPING = DEFAULT_MUJOCO_PHYSICS_PROFILE.joint_damping
DEFAULT_JOINT_ARMATURE = DEFAULT_MUJOCO_PHYSICS_PROFILE.joint_armature
DEFAULT_ACTUATOR_KP = DEFAULT_MUJOCO_PHYSICS_PROFILE.actuator_kp
DEFAULT_ACTUATOR_FORCE_LIMIT = DEFAULT_MUJOCO_PHYSICS_PROFILE.actuator_force_limit
DEFAULT_FINGER_COLLISION_RADIUS = (
    DEFAULT_MUJOCO_PHYSICS_PROFILE.finger_collision_radius_m
)
DEFAULT_FINGER_COLLISION_HALF_LENGTH = (
    DEFAULT_MUJOCO_PHYSICS_PROFILE.finger_collision_half_length_m
)
DEFAULT_PALM_COLLISION_POS = DEFAULT_MUJOCO_PHYSICS_PROFILE.palm_collision_pos_m
DEFAULT_PALM_COLLISION_SIZE = DEFAULT_MUJOCO_PHYSICS_PROFILE.palm_collision_size_m
DEFAULT_CUBE_HALF_SIZE = DEFAULT_MUJOCO_PHYSICS_PROFILE.cube_half_size_m
DEFAULT_CUBE_MASS_KG = DEFAULT_MUJOCO_PHYSICS_PROFILE.cube_mass_kg
DEFAULT_FLOOR_Z = DEFAULT_MUJOCO_PHYSICS_PROFILE.floor_z_m
DEFAULT_TURN_HANDLE_CENTER_POS = DEFAULT_MUJOCO_PHYSICS_PROFILE.turn_handle_center_pos_m
DEFAULT_TURN_HANDLE_BAR_HALF_LENGTH = (
    DEFAULT_MUJOCO_PHYSICS_PROFILE.turn_handle_bar_half_length_m
)
DEFAULT_TURN_HANDLE_BAR_RADIUS = DEFAULT_MUJOCO_PHYSICS_PROFILE.turn_handle_bar_radius_m
DEFAULT_TURN_HANDLE_KNOB_RADIUS = (
    DEFAULT_MUJOCO_PHYSICS_PROFILE.turn_handle_knob_radius_m
)
DEFAULT_TURN_HANDLE_HUB_RADIUS = DEFAULT_MUJOCO_PHYSICS_PROFILE.turn_handle_hub_radius_m
DEFAULT_TURN_HANDLE_HUB_HALF_LENGTH = (
    DEFAULT_MUJOCO_PHYSICS_PROFILE.turn_handle_hub_half_length_m
)
DEFAULT_TURN_HANDLE_MASS_KG = DEFAULT_MUJOCO_PHYSICS_PROFILE.turn_handle_mass_kg
DEFAULT_TURN_HANDLE_DAMPING = DEFAULT_MUJOCO_PHYSICS_PROFILE.turn_handle_damping
DEFAULT_TURN_HANDLE_ARMATURE = DEFAULT_MUJOCO_PHYSICS_PROFILE.turn_handle_armature
DEFAULT_TURN_HANDLE_FRICTIONLOSS = (
    DEFAULT_MUJOCO_PHYSICS_PROFILE.turn_handle_frictionloss
)
DEFAULT_TURN_HANDLE_JOINT_NAME = "turn_handle_hinge"
DEFAULT_TURN_HANDLE_OBSERVATION_CAMERA_Z = 0.46
DEFAULT_TURN_HANDLE_OBSERVATION_CAMERA_X_OFFSET = -0.02
DEFAULT_TURN_HANDLE_OBSERVATION_CAMERA_XYAXES = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
DEFAULT_TURN_HANDLE_OBSERVATION_CAMERA_FOVY_DEG = 55.0
DEFAULT_SCENE_HAND_MOUNT_NAME = "ghand_hand_mount"
DEFAULT_SCENE_HAND_QUAT = DISPLAY_HAND_ROTATION_QUAT_WXYZ
DEFAULT_MJWARP_CONTACT_SENSOR_MAXMATCH = 128
HAND_COLLISION_CONTYPE = str(DEFAULT_MUJOCO_PHYSICS_PROFILE.hand_collision_contype)
HAND_COLLISION_CONAFFINITY = str(
    DEFAULT_MUJOCO_PHYSICS_PROFILE.hand_collision_conaffinity
)
CUBE_COLLISION_CONTYPE = str(DEFAULT_MUJOCO_PHYSICS_PROFILE.cube_collision_contype)
CUBE_COLLISION_CONAFFINITY = str(
    DEFAULT_MUJOCO_PHYSICS_PROFILE.cube_collision_conaffinity
)
FLOOR_COLLISION_CONTYPE = str(DEFAULT_MUJOCO_PHYSICS_PROFILE.floor_collision_contype)
FLOOR_COLLISION_CONAFFINITY = str(
    DEFAULT_MUJOCO_PHYSICS_PROFILE.floor_collision_conaffinity
)
TURN_HANDLE_COLLISION_CONTYPE = str(
    DEFAULT_MUJOCO_PHYSICS_PROFILE.turn_handle_collision_contype
)
TURN_HANDLE_COLLISION_CONAFFINITY = str(
    DEFAULT_MUJOCO_PHYSICS_PROFILE.turn_handle_collision_conaffinity
)

# The source visual meshes make useful hand-object convex hulls, but a few of
# those hulls overlap neighboring links in the valid cube pre-grasp.  Give each
# hand geom a private collision bit and enable only cross-link pairs that do
# not overlap in the pre-grasp.  The common hand bit remains present
# so cube/floor/handle contact keeps the existing physics contract.
_MIDDLE_SELF_COLLISION_GEOM_NAMES: Final[tuple[str, ...]] = (
    "hnx330_n102_3_collision",
    "hnx330_n102_1_collision",
    "hnx330_n102_9_collision",
    "hnx330_n102_2_collision",
)
_RING_SELF_COLLISION_GEOM_NAMES: Final[tuple[str, ...]] = (
    "hnx330_n102_10_collision",
    "hnx330_n102_13_collision",
    "hnx330_n102_14_collision",
    "hnx330_n102_5_collision",
)
SAFE_HAND_SELF_COLLISION_GEOM_GROUPS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("palm", ("palm_collision",)),
    (
        "index",
        (
            "hnx330_n102_12_collision",
            "hnx330_n102_collision",
            "hnx330_n102_11_collision",
            "hnx330_n102_15_collision",
        ),
    ),
    ("middle", _MIDDLE_SELF_COLLISION_GEOM_NAMES),
    ("ring", _RING_SELF_COLLISION_GEOM_NAMES),
    (
        "thumb",
        (
            "hnx330_n102_8_collision",
            "hnx330_n102_7_collision",
            "xc330_t288_t_7_collision",
            "xc330_t288_t_3_collision",
        ),
    ),
)
SAFE_HAND_SELF_COLLISION_GEOM_NAMES: Final[tuple[str, ...]] = tuple(
    geom_name
    for _, geom_names in SAFE_HAND_SELF_COLLISION_GEOM_GROUPS
    for geom_name in geom_names
)
_SAFE_PALM_SELF_COLLISION_PARTNERS: Final[frozenset[str]] = frozenset(
    {
        "hnx330_n102_collision",
        "hnx330_n102_11_collision",
        "hnx330_n102_15_collision",
        "hnx330_n102_1_collision",
        "hnx330_n102_9_collision",
        "hnx330_n102_2_collision",
        "hnx330_n102_14_collision",
        "hnx330_n102_5_collision",
        "xc330_t288_t_7_collision",
        "xc330_t288_t_3_collision",
    }
)


def _ordered_geom_pair(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first <= second else (second, first)


_UNSAFE_PREGRASP_SELF_COLLISION_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        _ordered_geom_pair(
            "hnx330_n102_9_collision",
            "hnx330_n102_13_collision",
        ),
        _ordered_geom_pair(
            "hnx330_n102_2_collision",
            "hnx330_n102_14_collision",
        ),
        _ordered_geom_pair(
            "hnx330_n102_2_collision",
            "hnx330_n102_5_collision",
        ),
    }
)

# These convex hulls are tangent or naturally closer than the standard margin
# in validated task motions.  All enabled middle-ring pairs are also kept at
# actual contact, without a dedicated clearance margin.  Keep every pair active
# and hard, but do not create artificial proximity forces before penetration.
_ZERO_MARGIN_SELF_COLLISION_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        _ordered_geom_pair(
            "hnx330_n102_collision",
            "hnx330_n102_3_collision",
        ),
        *(
            _ordered_geom_pair(middle, ring)
            for middle in _MIDDLE_SELF_COLLISION_GEOM_NAMES
            for ring in _RING_SELF_COLLISION_GEOM_NAMES
        ),
    }
)


def _safe_hand_self_collision_pairs() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = [
        ("palm_collision", partner)
        for partner in SAFE_HAND_SELF_COLLISION_GEOM_NAMES
        if partner in _SAFE_PALM_SELF_COLLISION_PARTNERS
    ]
    finger_groups = SAFE_HAND_SELF_COLLISION_GEOM_GROUPS[1:]
    for (_, first_geoms), (_, second_geoms) in combinations(finger_groups, 2):
        for first in first_geoms:
            for second in second_geoms:
                if (
                    _ordered_geom_pair(first, second)
                    in _UNSAFE_PREGRASP_SELF_COLLISION_PAIRS
                ):
                    continue
                pairs.append((first, second))
    return tuple(pairs)


SAFE_HAND_SELF_COLLISION_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    _safe_hand_self_collision_pairs()
)
HAND_SELF_COLLISION_PAIR_NAMES: Final[tuple[str, ...]] = tuple(
    f"hand_self_collision_pair_{index:03d}"
    for index in range(len(SAFE_HAND_SELF_COLLISION_PAIRS))
)
HAND_SELF_COLLISION_CONTACT_SENSOR_NAMES: Final[tuple[str, ...]] = tuple(
    f"hand_self_collision_{index:03d}"
    for index in range(len(SAFE_HAND_SELF_COLLISION_PAIRS))
)
CUBE_HAND_CONTACT_SENSOR_NAMES: Final[tuple[str, ...]] = tuple(
    f"cube_hand_contact_{index:02d}"
    for index in range(len(SAFE_HAND_SELF_COLLISION_GEOM_NAMES))
)
CUBE_FLOOR_CONTACT_SENSOR_NAME: Final[str] = "cube_floor_contact"
CUBE_POSITION_SENSOR_NAME: Final[str] = "cube_position_world"
CUBE_QUATERNION_SENSOR_NAME: Final[str] = "cube_quaternion_wxyz"
CUBE_ANGULAR_VELOCITY_SENSOR_NAME: Final[str] = "cube_angular_velocity_world"
_HAND_SELF_COLLISION_UNIQUE_BIT_OFFSET: Final[int] = 4


class MujocoExportError(RuntimeError):
    """Raised when a LEAP V1 KAIZEN MuJoCo model cannot be exported."""


@dataclass(frozen=True)
class MujocoHandExportResult:
    """Paths and channel metadata produced by a MuJoCo hand export."""

    hand_xml: Path
    scene_xml: Path | None
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    turn_handle_scene_xml: Path | None = None


@dataclass(frozen=True)
class CollisionGeomSpec:
    """One body-local MuJoCo convex mesh collision geom."""

    geom_name: str
    mesh_name: str
    mesh_file: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class VisualGeomSpec:
    """Body-local MuJoCo visual mesh geom converted from source URDF visual meshes."""

    geom_name: str
    mesh_name: str
    mesh_file: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]


def export_mujoco_hand_model(
    output_dir: Path | str,
    *,
    urdf_path: Path | str | None = None,
    hand_xml_name: str = DEFAULT_HAND_XML_NAME,
    scene_xml_name: str = DEFAULT_CUBE_SCENE_XML_NAME,
    turn_handle_scene_xml_name: str = DEFAULT_TURN_HANDLE_SCENE_XML_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    physics_profile: MujocoPhysicsProfile = DEFAULT_MUJOCO_PHYSICS_PROFILE,
    validate: bool = False,
) -> MujocoHandExportResult:
    """Export the LEAP V1 KAIZEN right-hand URDF to a lightweight MuJoCo MJCF model.

    MuJoCo's URDF compiler is used to preserve the URDF kinematic tree. The
    exported MJCF then adds converted visual mesh geoms, versioned convex-part
    collision geoms derived from the original PLY meshes, hinge limits, and
    position actuators.
    """

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    source_urdf = (
        Path(urdf_path) if urdf_path is not None else default_right_hand_urdf_path()
    )
    if not source_urdf.is_file():
        raise MujocoExportError(f"URDF not found: {source_urdf}")

    hand_xml = output_root / hand_xml_name
    try:
        joint_ranges = mujoco_joint_ranges_from_urdf(source_urdf)
    except (OSError, ValueError, ET.ParseError) as exc:
        raise MujocoExportError(
            f"Failed to load LEAP V1 KAIZEN joint limits from URDF: {source_urdf}: {exc}"
        ) from exc
    mesh_dir = output_root / "meshes"
    collision_manifest_path = (
        default_mujoco_collision_manifest_path()
        if source_urdf.resolve() == default_right_hand_urdf_path().resolve()
        else source_urdf.with_suffix(".mujoco-collision.json")
    )
    mesh_collision_specs = _mesh_collision_specs_from_urdf(
        source_urdf,
        mesh_dir=mesh_dir,
        collision_manifest_path=collision_manifest_path,
    )
    visual_geom_specs = _visual_geom_specs_from_urdf(
        source_urdf,
        mesh_dir=mesh_dir,
    )
    tree = _compile_urdf_to_mjcf_tree(source_urdf)
    joint_names = prepare_compiled_mjcf_tree(
        tree,
        model_name=model_name,
        joint_ranges=joint_ranges,
        mesh_collision_specs=mesh_collision_specs,
        visual_geom_specs=visual_geom_specs,
        physics_profile=physics_profile,
    )
    _write_tree(tree, hand_xml)

    scene_xml = output_root / scene_xml_name
    write_cube_face_up_scene(
        hand_xml=hand_xml,
        scene_xml=scene_xml,
        physics_profile=physics_profile,
    )

    turn_handle_scene_xml = output_root / turn_handle_scene_xml_name
    write_turn_handle_scene(
        hand_xml=hand_xml,
        scene_xml=turn_handle_scene_xml,
        physics_profile=physics_profile,
    )

    if validate:
        _validate_mujoco_xml(hand_xml)
        _validate_mujoco_xml(scene_xml)
        _validate_mujoco_xml(turn_handle_scene_xml)

    return MujocoHandExportResult(
        hand_xml=hand_xml,
        scene_xml=scene_xml,
        joint_names=joint_names,
        actuator_names=tuple(f"{joint_name}_position" for joint_name in joint_names),
        turn_handle_scene_xml=turn_handle_scene_xml,
    )


def prepare_compiled_mjcf_tree(
    tree: ET.ElementTree,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    joint_ranges: Mapping[str, tuple[float, float]] | None = None,
    mesh_collision_specs: Mapping[str, Sequence[CollisionGeomSpec]] | None = None,
    visual_geom_specs: Mapping[str, Sequence[VisualGeomSpec]] | None = None,
    physics_profile: MujocoPhysicsProfile = DEFAULT_MUJOCO_PHYSICS_PROFILE,
) -> tuple[str, ...]:
    """Add RL-friendly collision geoms and actuators to a compiled MJCF tree."""

    root = tree.getroot()
    if _tag(root) != "mujoco":
        raise MujocoExportError("Expected a compiled MJCF tree rooted at <mujoco>.")
    root.set("model", model_name)
    _ensure_compiler(root)
    _ensure_option(root, physics_profile)
    _add_mesh_assets(root, mesh_collision_specs or {}, visual_geom_specs or {})
    worldbody = _required_child(root, "worldbody")
    joint_names = _configure_joints(worldbody, joint_ranges or {}, physics_profile)
    _add_visual_geometries(worldbody, visual_geom_specs or {})
    _add_collision_geometries(worldbody, mesh_collision_specs or {}, physics_profile)
    _configure_hand_self_collision_masks(worldbody, physics_profile)
    _add_hand_self_collision_contact_pairs(root, worldbody, physics_profile)
    _replace_actuators(root, joint_names, joint_ranges or {}, physics_profile)
    return joint_names


def write_cube_face_up_scene(
    *,
    hand_xml: Path | str,
    scene_xml: Path | str,
    model_name: str = DEFAULT_SCENE_MODEL_NAME,
    cube_half_size: float | None = None,
    cube_mass_kg: float | None = None,
    physics_profile: MujocoPhysicsProfile = DEFAULT_MUJOCO_PHYSICS_PROFILE,
) -> Path:
    """Write a MuJoCo scene containing the exported hand and a labeled cube."""

    hand_path = Path(hand_xml)
    scene_path = Path(scene_xml)
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    half = _validate_positive(
        physics_profile.cube_half_size_m if cube_half_size is None else cube_half_size,
        "cube_half_size",
    )
    mass = _validate_positive(
        physics_profile.cube_mass_kg if cube_mass_kg is None else cube_mass_kg,
        "cube_mass_kg",
    )

    root = ET.Element("mujoco", {"model": model_name})
    hand_tree = ET.parse(hand_path)
    hand_root = hand_tree.getroot()
    _copy_hand_compiler(root, hand_root)
    ET.SubElement(
        root,
        "option",
        {
            "timestep": _fmt(physics_profile.physics_timestep_s),
            "integrator": physics_profile.integrator,
            "cone": physics_profile.cone,
            "impratio": _fmt(physics_profile.solver_impratio),
        },
    )
    asset = ET.SubElement(root, "asset")
    _copy_hand_assets(
        asset,
        hand_root=hand_root,
        hand_xml=hand_path,
        scene_xml=scene_path,
    )
    _add_material(asset, "floor_mat", "0.18 0.18 0.18 1")
    for name, rgba in _cube_face_materials().items():
        _add_material(asset, name, rgba)

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody, "light", {"name": "key", "pos": "0 -0.4 0.5", "dir": "0 0.6 -1"}
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": "0.18 -0.30 0.16",
            "xyaxes": "0.86 0.51 0 -0.20 0.33 0.92",
        },
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": DEFAULT_POLICY_CAMERA_NAME,
            "pos": "0.18 -0.30 0.16",
            "xyaxes": "0.86 0.51 0 -0.20 0.33 0.92",
            "fovy": "45",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": f"0 0 {_fmt(physics_profile.floor_z_m)}",
            "size": "0.35 0.35 0.02",
            "material": "floor_mat",
            "friction": _vec(physics_profile.floor_friction),
            "contype": str(physics_profile.floor_collision_contype),
            "conaffinity": str(physics_profile.floor_collision_conaffinity),
        },
    )
    _add_mounted_hand(worldbody, hand_root)
    cube = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "letter_cube",
            "pos": _vec(_default_cube_start_pos(half, physics_profile)),
            # Domain randomization changes the compiled COM/inertia at reset.
            # Disable MuJoCo's same-frame simple-body optimization so
            # ``mj_setConst`` can recompute derived quantities safely.
            "simple": "false",
        },
    )
    ET.SubElement(cube, "freejoint", {"name": "cube_freejoint"})
    ET.SubElement(
        cube,
        "geom",
        {
            "name": "cube_collision",
            "type": "box",
            "size": _vec((half, half, half)),
            "mass": _fmt(mass),
            "rgba": "0 0 0 0",
            "friction": _vec(physics_profile.cube_friction),
            "contype": str(physics_profile.cube_collision_contype),
            "conaffinity": str(physics_profile.cube_collision_conaffinity),
        },
    )
    _add_cube_face_visuals(cube, half)
    _add_cube_face_sites(cube, half)
    _copy_hand_runtime_sections(root, hand_root)
    _add_cube_sensors(root)

    tree = ET.ElementTree(root)
    _write_tree(tree, scene_path)
    return scene_path


def write_turn_handle_scene(
    *,
    hand_xml: Path | str,
    scene_xml: Path | str,
    model_name: str = DEFAULT_TURN_HANDLE_SCENE_MODEL_NAME,
    handle_center_pos: Sequence[float] | None = None,
    bar_half_length: float | None = None,
    bar_radius: float | None = None,
    knob_radius: float | None = None,
    hub_radius: float | None = None,
    hub_half_length: float | None = None,
    handle_mass_kg: float | None = None,
    physics_profile: MujocoPhysicsProfile = DEFAULT_MUJOCO_PHYSICS_PROFILE,
) -> Path:
    """Write a MuJoCo scene containing the hand and a passive turn handle."""

    hand_path = Path(hand_xml)
    scene_path = Path(scene_xml)
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    center = _validate_vector3(
        physics_profile.turn_handle_center_pos_m
        if handle_center_pos is None
        else handle_center_pos,
        "handle_center_pos",
    )
    half_length = _validate_positive(
        physics_profile.turn_handle_bar_half_length_m
        if bar_half_length is None
        else bar_half_length,
        "bar_half_length",
    )
    bar_r = _validate_positive(
        physics_profile.turn_handle_bar_radius_m if bar_radius is None else bar_radius,
        "bar_radius",
    )
    knob_r = _validate_positive(
        physics_profile.turn_handle_knob_radius_m
        if knob_radius is None
        else knob_radius,
        "knob_radius",
    )
    hub_r = _validate_positive(
        physics_profile.turn_handle_hub_radius_m if hub_radius is None else hub_radius,
        "hub_radius",
    )
    hub_half = _validate_positive(
        physics_profile.turn_handle_hub_half_length_m
        if hub_half_length is None
        else hub_half_length,
        "hub_half_length",
    )
    mass = _validate_positive(
        physics_profile.turn_handle_mass_kg
        if handle_mass_kg is None
        else handle_mass_kg,
        "handle_mass_kg",
    )

    root = ET.Element("mujoco", {"model": model_name})
    hand_tree = ET.parse(hand_path)
    hand_root = hand_tree.getroot()
    _copy_hand_compiler(root, hand_root)
    ET.SubElement(
        root,
        "option",
        {
            "timestep": _fmt(physics_profile.physics_timestep_s),
            "integrator": physics_profile.integrator,
            "cone": physics_profile.cone,
            "impratio": _fmt(physics_profile.solver_impratio),
        },
    )
    asset = ET.SubElement(root, "asset")
    _copy_hand_assets(
        asset,
        hand_root=hand_root,
        hand_xml=hand_path,
        scene_xml=scene_path,
    )
    _add_material(asset, "floor_mat", "0.18 0.18 0.18 1")
    _add_material(asset, "turn_handle_mat", "0.92 0.92 0.88 1")
    _add_material(asset, "turn_handle_hub_mat", "0.50 0.50 0.50 1")
    _add_material(asset, "turn_handle_target_mat", "0.10 0.70 0.28 0.75")

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody, "light", {"name": "key", "pos": "0 -0.4 0.5", "dir": "0 0.6 -1"}
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": _vec(
                (
                    center[0] + DEFAULT_TURN_HANDLE_OBSERVATION_CAMERA_X_OFFSET,
                    center[1],
                    DEFAULT_TURN_HANDLE_OBSERVATION_CAMERA_Z,
                )
            ),
            "xyaxes": _vec(DEFAULT_TURN_HANDLE_OBSERVATION_CAMERA_XYAXES),
            "fovy": _fmt(DEFAULT_TURN_HANDLE_OBSERVATION_CAMERA_FOVY_DEG),
        },
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": DEFAULT_POLICY_CAMERA_NAME,
            "pos": _vec(
                (
                    center[0] + DEFAULT_TURN_HANDLE_OBSERVATION_CAMERA_X_OFFSET,
                    center[1],
                    DEFAULT_TURN_HANDLE_OBSERVATION_CAMERA_Z,
                )
            ),
            "xyaxes": _vec(DEFAULT_TURN_HANDLE_OBSERVATION_CAMERA_XYAXES),
            "fovy": _fmt(DEFAULT_TURN_HANDLE_OBSERVATION_CAMERA_FOVY_DEG),
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": f"0 0 {_fmt(physics_profile.floor_z_m)}",
            "size": "0.35 0.35 0.02",
            "material": "floor_mat",
            "friction": _vec(physics_profile.floor_friction),
            "contype": str(physics_profile.floor_collision_contype),
            "conaffinity": str(physics_profile.floor_collision_conaffinity),
        },
    )
    _add_mounted_hand(worldbody, hand_root)
    _add_turn_handle(
        worldbody,
        center,
        half_length,
        bar_r,
        knob_r,
        hub_r,
        hub_half,
        mass,
        physics_profile,
    )
    _copy_hand_runtime_sections(root, hand_root)
    _add_turn_handle_sensors(root)

    tree = ET.ElementTree(root)
    _write_tree(tree, scene_path)
    return scene_path


def _copy_hand_compiler(root: ET.Element, hand_root: ET.Element) -> None:
    compiler = hand_root.find("compiler")
    if compiler is not None:
        root.append(deepcopy(compiler))


def _copy_hand_assets(
    asset: ET.Element,
    *,
    hand_root: ET.Element,
    hand_xml: Path,
    scene_xml: Path,
) -> None:
    hand_asset = hand_root.find("asset")
    if hand_asset is None:
        return
    for child in hand_asset:
        copied = deepcopy(child)
        file_attr = copied.get("file")
        if file_attr is not None:
            file_path = Path(file_attr)
            if not file_path.is_absolute():
                file_path = (hand_xml.parent / file_path).resolve()
            copied.set("file", os.path.relpath(file_path, scene_xml.parent))
        asset.append(copied)


def _add_mounted_hand(worldbody: ET.Element, hand_root: ET.Element) -> None:
    mount = ET.SubElement(
        worldbody,
        "body",
        {
            "name": DEFAULT_SCENE_HAND_MOUNT_NAME,
            "quat": _vec(DEFAULT_SCENE_HAND_QUAT),
        },
    )
    hand_worldbody = hand_root.find("worldbody")
    if hand_worldbody is None:
        return
    for child in hand_worldbody:
        mount.append(deepcopy(child))


def _copy_hand_runtime_sections(root: ET.Element, hand_root: ET.Element) -> None:
    skipped = {"compiler", "option", "asset", "worldbody"}
    for child in hand_root:
        if _tag(child) in skipped:
            continue
        root.append(deepcopy(child))


def _add_turn_handle(
    worldbody: ET.Element,
    center: tuple[float, float, float],
    half_length: float,
    bar_radius: float,
    knob_radius: float,
    hub_radius: float,
    hub_half_length: float,
    mass: float,
    physics_profile: MujocoPhysicsProfile,
) -> None:
    support_half_height = max((center[2] - physics_profile.floor_z_m) * 0.5, 0.001)
    support_center_z = physics_profile.floor_z_m + support_half_height
    ET.SubElement(
        worldbody,
        "body",
        {
            "name": "turn_handle_support",
            "pos": _vec((center[0], center[1], support_center_z)),
        },
    )

    handle = ET.SubElement(
        worldbody, "body", {"name": "turn_handle", "pos": _vec(center)}
    )
    ET.SubElement(
        handle,
        "joint",
        {
            "name": DEFAULT_TURN_HANDLE_JOINT_NAME,
            "type": "hinge",
            "axis": "0 0 1",
            "limited": "false",
            "damping": _fmt(physics_profile.turn_handle_damping),
            "armature": _fmt(physics_profile.turn_handle_armature),
            "frictionloss": _fmt(physics_profile.turn_handle_frictionloss),
        },
    )
    _add_turn_handle_geom(
        handle,
        name="turn_handle_bar",
        geom_type="capsule",
        mass=mass * 0.5,
        material="turn_handle_mat",
        physics_profile=physics_profile,
        extra={
            "fromto": f"{_fmt(-half_length)} 0 0 {_fmt(half_length)} 0 0",
            "size": _fmt(bar_radius),
        },
    )
    for side, x_pos in (("negative", -half_length), ("positive", half_length)):
        _add_turn_handle_geom(
            handle,
            name=f"turn_handle_{side}_knob",
            geom_type="sphere",
            mass=mass * 0.2,
            material="turn_handle_mat",
            physics_profile=physics_profile,
            extra={"pos": _vec((x_pos, 0.0, 0.0)), "size": _fmt(knob_radius)},
        )
    _add_turn_handle_geom(
        handle,
        name="turn_handle_hub",
        geom_type="cylinder",
        mass=mass * 0.1,
        material="turn_handle_hub_mat",
        physics_profile=physics_profile,
        extra={"size": f"{_fmt(hub_radius)} {_fmt(hub_half_length)}"},
    )
    ET.SubElement(
        handle,
        "site",
        {
            "name": "turn_handle_axis",
            "pos": "0 0 0",
            "type": "sphere",
            "size": "0.003",
            "rgba": "0.2 0.2 0.2 0.6",
        },
    )
    for side, x_pos in (("negative", -half_length), ("positive", half_length)):
        ET.SubElement(
            handle,
            "site",
            {
                "name": f"turn_handle_{side}_tip",
                "pos": _vec((x_pos, 0.0, 0.0)),
                "type": "sphere",
                "size": "0.004",
                "rgba": "1 1 1 0.45",
            },
        )

    target = ET.SubElement(
        worldbody, "body", {"name": "turn_handle_target", "pos": _vec(center)}
    )
    ET.SubElement(
        target,
        "site",
        {
            "name": "turn_handle_target_tip",
            "pos": _vec((0.0, half_length, 0.0)),
            "type": "sphere",
            "size": "0.006",
            "rgba": "0.10 0.70 0.28 0.75",
        },
    )


def _add_turn_handle_geom(
    parent: ET.Element,
    *,
    name: str,
    geom_type: str,
    mass: float,
    material: str,
    physics_profile: MujocoPhysicsProfile,
    extra: Mapping[str, str],
) -> None:
    attrib = {
        "name": name,
        "type": geom_type,
        "mass": _fmt(mass),
        "material": material,
        "friction": _vec(physics_profile.turn_handle_friction),
        "condim": "4",
        "contype": str(physics_profile.turn_handle_collision_contype),
        "conaffinity": str(physics_profile.turn_handle_collision_conaffinity),
    }
    attrib.update(extra)
    ET.SubElement(parent, "geom", attrib)


def _add_turn_handle_sensors(root: ET.Element) -> None:
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    ET.SubElement(
        sensor,
        "jointpos",
        {"name": "turn_handle_angle", "joint": DEFAULT_TURN_HANDLE_JOINT_NAME},
    )
    ET.SubElement(
        sensor,
        "jointvel",
        {"name": "turn_handle_velocity", "joint": DEFAULT_TURN_HANDLE_JOINT_NAME},
    )


def _add_cube_sensors(root: ET.Element) -> None:
    """Add fixed-size state/contact channels supported by MJWarp.

    MJWarp stores raw contacts in a batch-global variable-length buffer.  These
    sensors keep the task interface fixed per world, while the CPU environment
    can continue to use MuJoCo's raw contact records as its canonical path.
    """

    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    maxmatch = custom.find("./numeric[@name='contact_sensor_maxmatch']")
    if maxmatch is None:
        maxmatch = ET.SubElement(
            custom,
            "numeric",
            {"name": "contact_sensor_maxmatch"},
        )
    maxmatch.set("data", str(DEFAULT_MJWARP_CONTACT_SENSOR_MAXMATCH))

    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    for tag, name in (
        ("framepos", CUBE_POSITION_SENSOR_NAME),
        ("framequat", CUBE_QUATERNION_SENSOR_NAME),
        ("frameangvel", CUBE_ANGULAR_VELOCITY_SENSOR_NAME),
    ):
        ET.SubElement(
            sensor,
            tag,
            {
                "name": name,
                "objtype": "body",
                "objname": "letter_cube",
            },
        )
    for sensor_name, (first_geom, second_geom) in zip(
        HAND_SELF_COLLISION_CONTACT_SENSOR_NAMES,
        SAFE_HAND_SELF_COLLISION_PAIRS,
        strict=True,
    ):
        ET.SubElement(
            sensor,
            "contact",
            {
                "name": sensor_name,
                "body1": _logical_collision_body_name(first_geom),
                "body2": _logical_collision_body_name(second_geom),
                "num": "1",
                "reduce": "mindist",
                "data": "found dist",
            },
        )
    for sensor_name, hand_geom in zip(
        CUBE_HAND_CONTACT_SENSOR_NAMES,
        SAFE_HAND_SELF_COLLISION_GEOM_NAMES,
        strict=True,
    ):
        ET.SubElement(
            sensor,
            "contact",
            {
                "name": sensor_name,
                "body1": "letter_cube",
                "body2": _logical_collision_body_name(hand_geom),
                "num": "1",
                "reduce": "mindist",
                "data": "found",
            },
        )
    ET.SubElement(
        sensor,
        "contact",
        {
            "name": CUBE_FLOOR_CONTACT_SENSOR_NAME,
            "geom1": "cube_collision",
            "geom2": "floor",
            "num": "1",
            "reduce": "mindist",
            "data": "found",
        },
    )


def _compile_urdf_to_mjcf_tree(urdf_path: Path) -> ET.ElementTree:
    try:
        mujoco = importlib.import_module("mujoco")
    except ImportError as exc:
        raise MujocoExportError(
            "MuJoCo is required for URDF export. Install the optional dependencies "
            "or run with `uv run --extra sim ghand export mujoco ...`."
        ) from exc

    try:
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
        with tempfile.TemporaryDirectory(prefix="ghand_mujoco_export_") as tmp:
            compiled_path = Path(tmp) / "compiled.xml"
            mujoco.mj_saveLastXML(str(compiled_path), model)
            return ET.parse(compiled_path)
    except Exception as exc:
        raise MujocoExportError(
            f"Failed to compile URDF with MuJoCo: {urdf_path}: {exc}"
        ) from exc


def _validate_mujoco_xml(path: Path) -> None:
    try:
        mujoco = importlib.import_module("mujoco")
        mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        raise MujocoExportError(
            f"Exported MuJoCo XML did not load: {path}: {exc}"
        ) from exc


def _ensure_compiler(root: ET.Element) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")


def _ensure_option(root: ET.Element, physics_profile: MujocoPhysicsProfile) -> None:
    option = root.find("option")
    if option is None:
        compiler = root.find("compiler")
        insert_at = list(root).index(compiler) + 1 if compiler is not None else 0
        option = ET.Element("option")
        root.insert(insert_at, option)
    option.set("timestep", _fmt(physics_profile.physics_timestep_s))
    option.set("integrator", physics_profile.integrator)
    option.set("cone", physics_profile.cone)
    option.set("impratio", _fmt(physics_profile.solver_impratio))


def _add_mesh_assets(
    root: ET.Element,
    mesh_collision_specs: Mapping[str, Sequence[CollisionGeomSpec]],
    visual_geom_specs: Mapping[str, Sequence[VisualGeomSpec]],
) -> None:
    mesh_specs: list[CollisionGeomSpec | VisualGeomSpec] = []
    for collision_specs in mesh_collision_specs.values():
        mesh_specs.extend(collision_specs)
    for visual_specs in visual_geom_specs.values():
        mesh_specs.extend(visual_specs)
    if not mesh_specs:
        return
    asset = root.find("asset")
    if asset is None:
        worldbody = root.find("worldbody")
        insert_at = list(root).index(worldbody) if worldbody is not None else len(root)
        asset = ET.Element("asset")
        root.insert(insert_at, asset)
    existing_mesh_names = {
        mesh.get("name")
        for mesh in asset.findall("mesh")
        if mesh.get("name") is not None
    }
    for spec in mesh_specs:
        if spec.mesh_name in existing_mesh_names:
            continue
        attributes = {
            "name": spec.mesh_name,
            "file": spec.mesh_file,
        }
        if isinstance(spec, CollisionGeomSpec):
            attributes["scale"] = _vec(spec.scale)
        ET.SubElement(asset, "mesh", attributes)
        existing_mesh_names.add(spec.mesh_name)


def _configure_joints(
    worldbody: ET.Element,
    joint_ranges: Mapping[str, tuple[float, float]],
    physics_profile: MujocoPhysicsProfile,
) -> tuple[str, ...]:
    joints = sorted(
        (
            joint
            for joint in worldbody.iter("joint")
            if _joint_sort_key(joint.get("name")) is not None
        ),
        key=lambda joint: _joint_sort_key(joint.get("name")) or 0,
    )
    joint_names: list[str] = []
    for joint in joints:
        name = joint.get("name")
        if name is None:
            continue
        lower, upper = joint_ranges.get(name, (-math.pi, math.pi))
        joint.set("limited", "true")
        joint.set("range", f"{_fmt(lower)} {_fmt(upper)}")
        joint.set("damping", _fmt(physics_profile.joint_damping))
        joint.set("armature", _fmt(physics_profile.joint_armature))
        joint.set("frictionloss", _fmt(physics_profile.joint_frictionloss))
        joint.set("solreflimit", _vec(physics_profile.joint_limit_solref))
        joint.set("solimplimit", _vec(physics_profile.joint_limit_solimp))
        joint_names.append(name)
    if not joint_names:
        raise MujocoExportError("Compiled MJCF did not contain any id<N> hand joints.")
    return tuple(joint_names)


def _add_visual_geometries(
    worldbody: ET.Element,
    visual_geom_specs: Mapping[str, Sequence[VisualGeomSpec]],
) -> None:
    for spec in visual_geom_specs.get("palm", ()):
        _add_visual_geometry(worldbody, spec)
    for body in worldbody.iter("body"):
        body_name = body.get("name")
        if body_name is None:
            continue
        for spec in visual_geom_specs.get(body_name, ()):
            _add_visual_geometry(body, spec)


def _add_visual_geometry(parent: ET.Element, spec: VisualGeomSpec) -> None:
    if parent.find(f"./geom[@name='{spec.geom_name}']") is not None:
        return
    ET.SubElement(
        parent,
        "geom",
        {
            "name": spec.geom_name,
            "type": "mesh",
            "mesh": spec.mesh_name,
            "pos": _vec(spec.pos),
            "quat": _vec(spec.quat),
            "rgba": "0.62 0.62 0.66 1",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "mass": "0",
        },
    )


def _add_collision_geometries(
    worldbody: ET.Element,
    mesh_collision_specs: Mapping[str, Sequence[CollisionGeomSpec]],
    physics_profile: MujocoPhysicsProfile,
) -> None:
    if worldbody.find("./geom[@name='palm_collision']") is None:
        palm_specs = mesh_collision_specs.get("palm", ())
        if not palm_specs:
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": "palm_collision",
                    "type": "box",
                    "pos": _vec(physics_profile.palm_collision_pos_m),
                    "size": _vec(physics_profile.palm_collision_size_m),
                    "rgba": "0.22 0.22 0.24 0.45",
                    "friction": _vec(physics_profile.palm_friction),
                    "condim": "4",
                    "contype": str(physics_profile.hand_collision_contype),
                    "conaffinity": str(physics_profile.hand_collision_conaffinity),
                },
            )
        else:
            for spec in palm_specs:
                _add_collision_geometry(
                    worldbody,
                    spec,
                    rgba="0.22 0.22 0.24 0.45",
                    friction=physics_profile.palm_friction,
                    physics_profile=physics_profile,
                )
    for body in worldbody.iter("body"):
        joint = body.find("joint")
        if joint is None or _joint_sort_key(joint.get("name")) is None:
            continue
        body_name = body.get("name") or joint.get("name") or "body"
        geom_name = f"{body_name}_collision"
        if body.find(f"./geom[@name='{geom_name}']") is not None:
            continue
        mesh_specs = mesh_collision_specs.get(body_name, ())
        if mesh_specs:
            for spec in mesh_specs:
                _add_collision_geometry(
                    body,
                    spec,
                    rgba="0.72 0.72 0.76 0.55",
                    friction=physics_profile.finger_friction,
                    physics_profile=physics_profile,
                )
            continue
        ET.SubElement(
            body,
            "geom",
            {
                "name": geom_name,
                "type": "capsule",
                "pos": f"0 0 {_fmt(physics_profile.finger_collision_half_length_m)}",
                "size": (
                    f"{_fmt(physics_profile.finger_collision_radius_m)} "
                    f"{_fmt(physics_profile.finger_collision_half_length_m)}"
                ),
                "rgba": "0.72 0.72 0.76 0.55",
                "friction": _vec(physics_profile.finger_friction),
                "condim": "4",
                "contype": str(physics_profile.hand_collision_contype),
                "conaffinity": str(physics_profile.hand_collision_conaffinity),
            },
        )


def _add_collision_geometry(
    parent: ET.Element,
    spec: CollisionGeomSpec,
    *,
    rgba: str,
    friction: Sequence[float],
    physics_profile: MujocoPhysicsProfile,
) -> None:
    ET.SubElement(
        parent,
        "geom",
        {
            "name": spec.geom_name,
            "type": "mesh",
            "mesh": spec.mesh_name,
            "pos": _vec(spec.pos),
            "quat": _vec(spec.quat),
            "rgba": rgba,
            "friction": _vec(friction),
            "condim": "4",
            "contype": str(physics_profile.hand_collision_contype),
            "conaffinity": str(physics_profile.hand_collision_conaffinity),
        },
    )


def _configure_hand_self_collision_masks(
    worldbody: ET.Element,
    physics_profile: MujocoPhysicsProfile,
) -> None:
    """Enable broad-phase-filtered self collision for validated link pairs."""

    mode = physics_profile.hand_self_collision_mode
    if mode == HAND_SELF_COLLISION_DISABLED:
        return
    if mode != HAND_SELF_COLLISION_SAFE_CROSS_LINK_V1:
        raise MujocoExportError(f"Unsupported hand self-collision mode: {mode!r}.")

    collision_geoms = _collision_geoms_by_logical_name(worldbody)
    missing = tuple(
        name
        for name in SAFE_HAND_SELF_COLLISION_GEOM_NAMES
        if not collision_geoms[name]
    )
    if missing:
        raise MujocoExportError(
            "Safe hand self-collision contract is missing collision geoms: "
            + ", ".join(missing)
        )

    unique_bit_by_geom = {
        name: 1 << (_HAND_SELF_COLLISION_UNIQUE_BIT_OFFSET + index)
        for index, name in enumerate(SAFE_HAND_SELF_COLLISION_GEOM_NAMES)
    }
    reserved_bits = sum(unique_bit_by_geom.values())
    existing_object_masks = (
        physics_profile.hand_collision_contype
        | physics_profile.hand_collision_conaffinity
        | physics_profile.cube_collision_contype
        | physics_profile.cube_collision_conaffinity
        | physics_profile.floor_collision_contype
        | physics_profile.floor_collision_conaffinity
        | physics_profile.turn_handle_collision_contype
        | physics_profile.turn_handle_collision_conaffinity
    )
    if reserved_bits & existing_object_masks:
        raise MujocoExportError(
            "Safe hand self-collision bits overlap an existing collision mask."
        )

    partner_bits_by_geom = {name: 0 for name in SAFE_HAND_SELF_COLLISION_GEOM_NAMES}
    for first, second in SAFE_HAND_SELF_COLLISION_PAIRS:
        partner_bits_by_geom[first] |= unique_bit_by_geom[second]
        partner_bits_by_geom[second] |= unique_bit_by_geom[first]

    for name in SAFE_HAND_SELF_COLLISION_GEOM_NAMES:
        for geom in collision_geoms[name]:
            geom.set(
                "contype",
                str(physics_profile.hand_collision_contype | unique_bit_by_geom[name]),
            )
            geom.set(
                "conaffinity",
                str(
                    physics_profile.hand_collision_conaffinity
                    | partner_bits_by_geom[name]
                ),
            )


def _add_hand_self_collision_contact_pairs(
    root: ET.Element,
    worldbody: ET.Element,
    physics_profile: MujocoPhysicsProfile,
) -> None:
    """Persist pair-specific hard contact parameters in the generated MJCF."""

    if physics_profile.hand_self_collision_mode == HAND_SELF_COLLISION_DISABLED:
        return

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    sliding, torsional, rolling = physics_profile.finger_friction
    pair_friction = (sliding, sliding, torsional, rolling, rolling)
    collision_geoms = _collision_geoms_by_logical_name(worldbody)
    for name, (first, second) in zip(
        HAND_SELF_COLLISION_PAIR_NAMES,
        SAFE_HAND_SELF_COLLISION_PAIRS,
        strict=True,
    ):
        margin = (
            0.0
            if _ordered_geom_pair(first, second) in _ZERO_MARGIN_SELF_COLLISION_PAIRS
            else physics_profile.hand_self_collision_margin_m
        )
        first_parts = collision_geoms[first]
        second_parts = collision_geoms[second]
        for first_index, second_index in product(
            range(len(first_parts)), range(len(second_parts))
        ):
            pair_name = (
                name
                if len(first_parts) == 1 and len(second_parts) == 1
                else f"{name}_{first_index:03d}_{second_index:03d}"
            )
            ET.SubElement(
                contact,
                "pair",
                {
                    "name": pair_name,
                    "geom1": first_parts[first_index].get("name", ""),
                    "geom2": second_parts[second_index].get("name", ""),
                    "condim": "4",
                    "friction": _vec(pair_friction),
                    "solref": _vec(physics_profile.hand_self_collision_solref),
                    "solimp": _vec(physics_profile.hand_self_collision_solimp),
                    "margin": _fmt(margin),
                },
            )


def _collision_geoms_by_logical_name(
    worldbody: ET.Element,
) -> dict[str, tuple[ET.Element, ...]]:
    geoms_by_name = {
        name: geom
        for geom in worldbody.iter("geom")
        if (name := geom.get("name")) is not None
    }
    return {
        logical_name: tuple(
            geom
            for name, geom in geoms_by_name.items()
            if name == logical_name or name.startswith(f"{logical_name}_p")
        )
        for logical_name in SAFE_HAND_SELF_COLLISION_GEOM_NAMES
    }


def _logical_collision_body_name(geom_name: str) -> str:
    if geom_name == "palm_collision":
        return DEFAULT_SCENE_HAND_MOUNT_NAME
    suffix = "_collision"
    if not geom_name.endswith(suffix):
        raise MujocoExportError(f"Invalid logical hand collision geom: {geom_name!r}.")
    return geom_name[: -len(suffix)]


def _mesh_collision_specs_from_urdf(
    urdf_path: Path,
    *,
    mesh_dir: Path,
    collision_manifest_path: Path,
) -> dict[str, tuple[CollisionGeomSpec, ...]]:
    try:
        manifest = load_mujoco_collision_asset_manifest(collision_manifest_path)
    except MujocoCollisionAssetError as exc:
        raise MujocoExportError(str(exc)) from exc
    source_urdf_sha256 = sha256_file(urdf_path)
    if source_urdf_sha256 != manifest.source_urdf_sha256:
        raise MujocoExportError(
            "MuJoCo collision asset manifest does not match the source URDF: "
            f"{source_urdf_sha256} != {manifest.source_urdf_sha256}. "
            "Regenerate the collision assets before exporting."
        )

    root = ET.parse(urdf_path).getroot()
    links = _urdf_links_by_name(root)
    children = _urdf_children_by_parent(root)
    dynamic_roots = _dynamic_child_links(root)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_asset_cache: dict[tuple[str, tuple[float, float, float]], tuple[str, str]] = {}
    source_hash_cache: dict[Path, str] = {}
    used_source_sha256: set[str] = set()
    specs: dict[str, tuple[CollisionGeomSpec, ...]] = {}
    palm_specs = _fixed_assembly_collision_geom_specs(
        urdf_path=urdf_path,
        links=links,
        children=children,
        root_link="root",
        body_key="palm",
        mesh_dir=mesh_dir,
        manifest=manifest,
        mesh_asset_cache=mesh_asset_cache,
        source_hash_cache=source_hash_cache,
        used_source_sha256=used_source_sha256,
    )
    if palm_specs:
        specs["palm"] = palm_specs
    for link_name in dynamic_roots:
        link_specs = _fixed_assembly_collision_geom_specs(
            urdf_path=urdf_path,
            links=links,
            children=children,
            root_link=link_name,
            body_key=link_name,
            mesh_dir=mesh_dir,
            manifest=manifest,
            mesh_asset_cache=mesh_asset_cache,
            source_hash_cache=source_hash_cache,
            used_source_sha256=used_source_sha256,
        )
        if link_specs:
            specs[link_name] = link_specs

    unused_source_sha256 = set(manifest.sources_by_sha256) - used_source_sha256
    if unused_source_sha256:
        unused_files = sorted(
            manifest.sources_by_sha256[digest].source_file
            for digest in unused_source_sha256
        )
        raise MujocoExportError(
            "MuJoCo collision asset manifest contains source meshes not used by "
            f"the matching URDF: {', '.join(unused_files)}"
        )
    return specs


def _visual_geom_specs_from_urdf(
    urdf_path: Path,
    *,
    mesh_dir: Path,
) -> dict[str, tuple[VisualGeomSpec, ...]]:
    root = ET.parse(urdf_path).getroot()
    links = _urdf_links_by_name(root)
    children = _urdf_children_by_parent(root)
    dynamic_roots = _dynamic_child_links(root)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    specs: dict[str, tuple[VisualGeomSpec, ...]] = {}
    mesh_asset_cache: dict[
        tuple[Path, tuple[float, float, float]], tuple[str, str] | None
    ] = {}
    palm_specs = _fixed_assembly_visual_geom_specs(
        urdf_path=urdf_path,
        links=links,
        children=children,
        root_link="root",
        body_key="palm",
        mesh_dir=mesh_dir,
        mesh_asset_cache=mesh_asset_cache,
    )
    if palm_specs:
        specs["palm"] = palm_specs
    for link_name in dynamic_roots:
        link_specs = _fixed_assembly_visual_geom_specs(
            urdf_path=urdf_path,
            links=links,
            children=children,
            root_link=link_name,
            body_key=link_name,
            mesh_dir=mesh_dir,
            mesh_asset_cache=mesh_asset_cache,
        )
        if link_specs:
            specs[link_name] = link_specs
    return specs


def _urdf_links_by_name(root: ET.Element) -> dict[str, ET.Element]:
    return {
        name: link
        for link in root.findall("link")
        if (name := link.get("name")) is not None
    }


def _urdf_children_by_parent(
    root: ET.Element,
) -> dict[str, list[tuple[ET.Element, str, str]]]:
    children: dict[str, list[tuple[ET.Element, str, str]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.get("link")
        child_link = child.get("link")
        joint_type = joint.get("type", "")
        if parent_link is None or child_link is None:
            continue
        children.setdefault(parent_link, []).append((joint, joint_type, child_link))
    return children


def _dynamic_child_links(root: ET.Element) -> tuple[str, ...]:
    links: list[str] = []
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        child = joint.find("child")
        if child is not None and (link := child.get("link")) is not None:
            links.append(link)
    return tuple(links)


def _fixed_assembly_collision_geom_specs(
    *,
    urdf_path: Path,
    links: Mapping[str, ET.Element],
    children: Mapping[str, Sequence[tuple[ET.Element, str, str]]],
    root_link: str,
    body_key: str,
    mesh_dir: Path,
    manifest: MujocoCollisionAssetManifest,
    mesh_asset_cache: dict[tuple[str, tuple[float, float, float]], tuple[str, str]],
    source_hash_cache: dict[Path, str],
    used_source_sha256: set[str],
) -> tuple[CollisionGeomSpec, ...]:
    specs: list[CollisionGeomSpec] = []

    def visit(link_name: str, transform: np.ndarray) -> None:
        link = links.get(link_name)
        if link is not None:
            specs.extend(
                _link_collision_geom_specs(
                    urdf_path=urdf_path,
                    link=link,
                    link_transform=transform,
                    mesh_dir=mesh_dir,
                    manifest=manifest,
                    mesh_asset_cache=mesh_asset_cache,
                    source_hash_cache=source_hash_cache,
                    used_source_sha256=used_source_sha256,
                )
            )
        for joint, joint_type, child_link in children.get(link_name, ()):
            if joint_type != "fixed":
                continue
            visit(child_link, transform @ _origin_transform(joint.find("origin")))

    visit(root_link, np.eye(4))
    logical_name = "palm_collision" if body_key == "palm" else f"{body_key}_collision"
    representative_index = next(
        (
            index
            for index, spec in enumerate(specs)
            if spec.mesh_name.startswith("tip_soft")
        ),
        0,
    )
    ordered_specs = (
        specs[representative_index : representative_index + 1]
        + specs[:representative_index]
        + specs[representative_index + 1 :]
    )
    return tuple(
        replace(
            spec,
            geom_name=(logical_name if index == 0 else f"{logical_name}_p{index:03d}"),
        )
        for index, spec in enumerate(ordered_specs)
    )


def _link_collision_geom_specs(
    *,
    urdf_path: Path,
    link: ET.Element,
    link_transform: np.ndarray,
    mesh_dir: Path,
    manifest: MujocoCollisionAssetManifest,
    mesh_asset_cache: dict[tuple[str, tuple[float, float, float]], tuple[str, str]],
    source_hash_cache: dict[Path, str],
    used_source_sha256: set[str],
) -> tuple[CollisionGeomSpec, ...]:
    specs: list[CollisionGeomSpec] = []
    for visual in link.findall("visual"):
        mesh_element = visual.find("geometry/mesh")
        if mesh_element is None:
            continue
        filename = mesh_element.get("filename")
        if filename is None:
            raise MujocoExportError("URDF visual mesh is missing its filename.")
        source_path = (urdf_path.parent / filename).resolve()
        if not source_path.is_file():
            raise MujocoExportError(f"URDF visual mesh not found: {source_path}")
        source_sha256 = source_hash_cache.get(source_path)
        if source_sha256 is None:
            source_sha256 = sha256_file(source_path)
            source_hash_cache[source_path] = source_sha256
        source_asset = manifest.sources_by_sha256.get(source_sha256)
        if source_asset is None:
            raise MujocoExportError(
                "MuJoCo collision asset manifest has no entry for URDF visual "
                f"mesh {source_path} ({source_sha256})."
            )
        used_source_sha256.add(source_sha256)
        scale = _parse_vector(
            mesh_element.get("scale"),
            default=(1.0, 1.0, 1.0),
        )
        if any(value <= 0.0 for value in scale):
            raise MujocoExportError(
                f"Collision mesh scale must be positive for {source_path}: {scale}"
            )
        transform = link_transform @ _origin_transform(visual.find("origin"))
        for part in source_asset.parts:
            cache_key = (part.sha256, scale)
            mesh_asset = mesh_asset_cache.get(cache_key)
            if mesh_asset is None:
                scale_name = "_".join(_safe_mesh_name(_fmt(value)) for value in scale)
                mesh_name = (
                    f"{_safe_mesh_name(part.file.stem)}_{part.sha256[:12]}_"
                    f"{scale_name}_collision_mesh"
                )
                destination = mesh_dir / f"{mesh_name}.obj"
                shutil.copyfile(part.file, destination)
                mesh_asset = (mesh_name, f"meshes/{destination.name}")
                mesh_asset_cache[cache_key] = mesh_asset
            mesh_name, mesh_file = mesh_asset
            specs.append(
                CollisionGeomSpec(
                    geom_name="",
                    mesh_name=mesh_name,
                    mesh_file=mesh_file,
                    pos=(
                        float(transform[0, 3]),
                        float(transform[1, 3]),
                        float(transform[2, 3]),
                    ),
                    quat=_matrix_to_quat(transform[:3, :3]),
                    scale=scale,
                )
            )
    return tuple(specs)


def _fixed_assembly_visual_geom_specs(
    *,
    urdf_path: Path,
    links: Mapping[str, ET.Element],
    children: Mapping[str, Sequence[tuple[ET.Element, str, str]]],
    root_link: str,
    body_key: str,
    mesh_dir: Path,
    mesh_asset_cache: dict[
        tuple[Path, tuple[float, float, float]], tuple[str, str] | None
    ],
) -> tuple[VisualGeomSpec, ...]:
    specs: list[VisualGeomSpec] = []
    body_name = _safe_mesh_name(body_key)

    def visit(link_name: str, transform: np.ndarray) -> None:
        link = links.get(link_name)
        if link is not None:
            link_name_safe = _safe_mesh_name(link_name)
            specs.extend(
                _link_visual_geom_specs(
                    urdf_path=urdf_path,
                    link=link,
                    link_transform=transform,
                    mesh_dir=mesh_dir,
                    name_prefix=f"{body_name}_{link_name_safe}",
                    start_index=len(specs),
                    mesh_asset_cache=mesh_asset_cache,
                )
            )
        for joint, joint_type, child_link in children.get(link_name, ()):
            if joint_type != "fixed":
                continue
            visit(child_link, transform @ _origin_transform(joint.find("origin")))

    visit(root_link, np.eye(4))
    return tuple(specs)


def _link_visual_geom_specs(
    *,
    urdf_path: Path,
    link: ET.Element,
    link_transform: np.ndarray,
    mesh_dir: Path,
    name_prefix: str,
    start_index: int,
    mesh_asset_cache: dict[
        tuple[Path, tuple[float, float, float]], tuple[str, str] | None
    ],
) -> tuple[VisualGeomSpec, ...]:
    specs: list[VisualGeomSpec] = []
    for visual in link.findall("visual"):
        mesh_element = visual.find("geometry/mesh")
        if mesh_element is None:
            continue
        filename = mesh_element.get("filename")
        if filename is None:
            continue
        mesh_path = (urdf_path.parent / filename).resolve()
        if not mesh_path.is_file():
            continue
        scale = _parse_vector(mesh_element.get("scale"), default=(1.0, 1.0, 1.0))
        transform = link_transform @ _origin_transform(visual.find("origin"))
        mesh_asset_key = (mesh_path, scale)
        if mesh_asset_key not in mesh_asset_cache:
            mesh_name = (
                f"{_safe_mesh_name(mesh_path.stem)}_{len(mesh_asset_cache)}_visual_mesh"
            )
            mesh_asset_cache[mesh_asset_key] = _export_visual_mesh(
                source_mesh_path=mesh_path,
                scale=scale,
                mesh_dir=mesh_dir,
                mesh_name=mesh_name,
            )
        mesh_asset = mesh_asset_cache[mesh_asset_key]
        if mesh_asset is None:
            continue
        mesh_name, mesh_file = mesh_asset
        specs.append(
            VisualGeomSpec(
                geom_name=f"{name_prefix}_{start_index + len(specs)}_visual",
                mesh_name=mesh_name,
                mesh_file=mesh_file,
                pos=(
                    float(transform[0, 3]),
                    float(transform[1, 3]),
                    float(transform[2, 3]),
                ),
                quat=_matrix_to_quat(transform[:3, :3]),
            )
        )
    return tuple(specs)


def _export_visual_mesh(
    *,
    source_mesh_path: Path,
    scale: Sequence[float],
    mesh_dir: Path,
    mesh_name: str,
) -> tuple[str, str] | None:
    try:
        import trimesh

        mesh = trimesh.load_mesh(source_mesh_path, process=False)
    except Exception:
        return None
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        return None
    scaled_vertices = vertices * np.asarray(scale, dtype=float)
    try:
        visual_mesh = trimesh.Trimesh(
            vertices=scaled_vertices,
            faces=faces,
            process=False,
        )
        mesh_path = mesh_dir / f"{mesh_name}.stl"
        visual_mesh.export(mesh_path, file_type="stl")
    except Exception as exc:
        raise MujocoExportError(
            f"Failed to build visual mesh {mesh_name}: {exc}"
        ) from exc
    return mesh_name, f"meshes/{mesh_path.name}"


def _replace_actuators(
    root: ET.Element,
    joint_names: Sequence[str],
    joint_ranges: Mapping[str, tuple[float, float]],
    physics_profile: MujocoPhysicsProfile,
) -> None:
    for actuator in list(root.findall("actuator")):
        root.remove(actuator)
    actuators = ET.SubElement(root, "actuator")
    for joint_name in joint_names:
        lower, upper = joint_ranges.get(joint_name, (-math.pi, math.pi))
        ET.SubElement(
            actuators,
            "position",
            {
                "name": f"{joint_name}_position",
                "joint": joint_name,
                "kp": _fmt(physics_profile.actuator_kp),
                "kv": _fmt(physics_profile.actuator_velocity_feedback_gain),
                "ctrlrange": f"{_fmt(lower)} {_fmt(upper)}",
                "forcerange": (
                    f"{_fmt(-physics_profile.actuator_force_limit)} "
                    f"{_fmt(physics_profile.actuator_force_limit)}"
                ),
            },
        )


def _add_material(asset: ET.Element, name: str, rgba: str) -> None:
    ET.SubElement(asset, "material", {"name": name, "rgba": rgba})


def _cube_face_materials() -> dict[str, str]:
    return {
        f"cube_face_{face.label}_mat": (
            " ".join(f"{value:.2f}" for value in face.rgba[:3]) + f" {face.rgba[3]:g}"
        )
        for face in CUBE_FACES
    }


def _add_cube_face_visuals(cube: ET.Element, half: float) -> None:
    thickness = max(half * 0.03, 0.0005)
    for face in CUBE_FACES:
        pos = tuple(value * (half - thickness) for value in face.normal)
        size = tuple(thickness if value else half for value in face.normal)
        ET.SubElement(
            cube,
            "geom",
            {
                "name": f"cube_face_{face.label}",
                "type": "box",
                "pos": _vec(pos),
                "size": _vec(size),
                "material": f"cube_face_{face.label}_mat",
                "contype": "0",
                "conaffinity": "0",
                "mass": "0",
            },
        )


def _add_cube_face_sites(cube: ET.Element, half: float) -> None:
    for face in CUBE_FACES:
        ET.SubElement(
            cube,
            "site",
            {
                "name": f"cube_face_{face.label}_normal",
                "pos": _vec(tuple(value * half for value in face.normal)),
                "user": _vec(face.normal),
                "type": "sphere",
                "size": "0.0015",
                "rgba": "1 1 1 0.4",
            },
        )


def _default_cube_start_pos(
    cube_half_size: float,
    physics_profile: MujocoPhysicsProfile,
) -> tuple[float, float, float]:
    palm_pos = _rotate_vector_by_quat(
        DEFAULT_SCENE_HAND_QUAT,
        physics_profile.palm_collision_pos_m,
    )
    return (
        palm_pos[0],
        palm_pos[1],
        palm_pos[2] + physics_profile.palm_collision_size_m[2] + cube_half_size + 0.002,
    )


def _origin_transform(origin: ET.Element | None) -> np.ndarray:
    if origin is None:
        return np.eye(4)
    xyz = _parse_vector(origin.get("xyz"), default=(0.0, 0.0, 0.0))
    rpy = _parse_vector(origin.get("rpy"), default=(0.0, 0.0, 0.0))
    transform = np.eye(4)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def _parse_vector(
    value: str | None, *, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    if value is None:
        return default
    parts = value.split()
    if len(parts) != 3:
        return default
    try:
        parsed = tuple(float(part) for part in parts)
    except ValueError:
        return default
    if not all(math.isfinite(part) for part in parsed):
        return default
    return parsed  # type: ignore[return-value]


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, cr, -sr),
            (0.0, sr, cr),
        )
    )
    rotation_y = np.asarray(
        (
            (cp, 0.0, sp),
            (0.0, 1.0, 0.0),
            (-sp, 0.0, cp),
        )
    )
    rotation_z = np.asarray(
        (
            (cy, -sy, 0.0),
            (sy, cy, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    return rotation_z @ rotation_y @ rotation_x


def _matrix_to_quat(matrix: np.ndarray) -> tuple[float, float, float, float]:
    rotation = np.asarray(matrix, dtype=float)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            (
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ),
            dtype=float,
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = (
                math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            )
            quat = np.asarray(
                (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ),
                dtype=float,
            )
        elif axis == 1:
            scale = (
                math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            )
            quat = np.asarray(
                (
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ),
                dtype=float,
            )
        else:
            scale = (
                math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            )
            quat = np.asarray(
                (
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ),
                dtype=float,
            )
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    return tuple(float(value) for value in quat)  # type: ignore[return-value]


def _rotate_vector_by_quat(
    quat: Sequence[float],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    rotation = _quat_to_matrix(quat)
    rotated = rotation @ np.asarray(vector, dtype=float)
    return tuple(float(value) for value in rotated)  # type: ignore[return-value]


def _quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0.0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _required_child(root: ET.Element, name: str) -> ET.Element:
    child = root.find(name)
    if child is None:
        raise MujocoExportError(f"Expected <{name}> in compiled MJCF.")
    return child


def _joint_sort_key(name: str | None) -> int | None:
    if name is None or not name.startswith("id"):
        return None
    suffix = name[2:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _safe_mesh_name(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isascii() and character.isalnum() else "_"
        for character in value
    )
    compacted = "_".join(part for part in normalized.split("_") if part)
    return compacted or "mesh"


def _validate_positive(value: float, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} must be positive.")
    return parsed


def _validate_vector3(
    values: Sequence[float], label: str
) -> tuple[float, float, float]:
    parsed = tuple(float(value) for value in values)
    if len(parsed) != 3 or not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{label} must contain three finite values.")
    return parsed  # type: ignore[return-value]


def _write_tree(tree: ET.ElementTree, path: Path) -> None:
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=False)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")


def _vec(values: Sequence[float]) -> str:
    return " ".join(_fmt(value) for value in values)


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]
