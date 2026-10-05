from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from ghand.hand import (
    default_right_hand_urdf_path,
    DISPLAY_HAND_ROTATION_QUAT_WXYZ,
)
from ghand.sim.mujoco.collision_assets import (
    load_mujoco_collision_asset_manifest,
)
from ghand.sim.mujoco import (
    DEFAULT_MUJOCO_PHYSICS_PROFILE,
    HAND_SELF_COLLISION_DISABLED,
    MujocoExportError,
    export_mujoco_hand_model,
    mujoco_joint_ranges_from_urdf,
    write_cube_face_up_scene,
    write_turn_handle_scene,
)
from ghand.sim.mujoco.exporter import (
    CUBE_ANGULAR_VELOCITY_SENSOR_NAME,
    CUBE_COLLISION_CONAFFINITY,
    CUBE_COLLISION_CONTYPE,
    CUBE_FLOOR_CONTACT_SENSOR_NAME,
    CUBE_HAND_CONTACT_SENSOR_NAMES,
    CUBE_POSITION_SENSOR_NAME,
    CUBE_QUATERNION_SENSOR_NAME,
    DEFAULT_CUBE_HALF_SIZE,
    DEFAULT_POLICY_CAMERA_NAME,
    FLOOR_COLLISION_CONAFFINITY,
    FLOOR_COLLISION_CONTYPE,
    HAND_COLLISION_CONAFFINITY,
    HAND_COLLISION_CONTYPE,
    HAND_SELF_COLLISION_CONTACT_SENSOR_NAMES,
    HAND_SELF_COLLISION_PAIR_NAMES,
    SAFE_HAND_SELF_COLLISION_GEOM_NAMES,
    SAFE_HAND_SELF_COLLISION_PAIRS,
    TURN_HANDLE_COLLISION_CONAFFINITY,
    TURN_HANDLE_COLLISION_CONTYPE,
    DEFAULT_TURN_HANDLE_JOINT_NAME,
    prepare_compiled_mjcf_tree,
)


def test_prepare_compiled_mjcf_tree_adds_limits_collision_and_actuators() -> None:
    tree = ET.ElementTree(
        ET.fromstring(
            """
<mujoco model="compiled">
  <compiler angle="radian" />
  <worldbody>
    <body name="finger_base">
      <joint name="id1" pos="0 0 0" axis="0 -1 0" />
      <body name="finger_mid">
        <joint name="id0" pos="0 0 0" axis="0 -1 0" />
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
        )
    )

    physics_profile = replace(
        DEFAULT_MUJOCO_PHYSICS_PROFILE,
        physics_timestep_s=0.004,
        joint_damping=0.05,
        joint_armature=0.002,
        joint_frictionloss=0.0248,
        joint_limit_solref=(0.008, 1.0),
        joint_limit_solimp=(0.95, 0.999, 0.001, 0.5, 2.0),
        actuator_kp=4.0,
        actuator_velocity_feedback_gain=0.1096,
        actuator_force_limit=0.4,
        solver_impratio=10.0,
        hand_self_collision_mode=HAND_SELF_COLLISION_DISABLED,
    )
    joint_names = prepare_compiled_mjcf_tree(
        tree,
        model_name="test_hand",
        joint_ranges={"id0": (-0.5, 1.0), "id1": (-1.0, 0.75)},
        physics_profile=physics_profile,
    )

    root = tree.getroot()
    assert root.attrib["model"] == "test_hand"
    assert joint_names == ("id0", "id1")
    assert root.find("option").attrib["integrator"] == "implicitfast"
    assert root.find("option").attrib["timestep"] == "0.004"
    assert root.find("option").attrib["impratio"] == "10"
    assert root.find(".//joint[@name='id0']").attrib["range"] == "-0.5 1"
    assert root.find(".//joint[@name='id0']").attrib["damping"] == "0.05"
    assert root.find(".//joint[@name='id0']").attrib["armature"] == "0.002"
    assert root.find(".//joint[@name='id0']").attrib["frictionloss"] == "0.0248"
    assert root.find(".//joint[@name='id0']").attrib["solreflimit"] == "0.008 1"
    assert (
        root.find(".//joint[@name='id0']").attrib["solimplimit"]
        == "0.95 0.999 0.001 0.5 2"
    )
    palm_collision = root.find(".//geom[@name='palm_collision']")
    finger_base_collision = root.find(".//geom[@name='finger_base_collision']")
    assert palm_collision is not None
    assert palm_collision.attrib["contype"] == HAND_COLLISION_CONTYPE
    assert palm_collision.attrib["conaffinity"] == HAND_COLLISION_CONAFFINITY
    assert finger_base_collision.attrib["type"] == "capsule"
    assert finger_base_collision.attrib["contype"] == HAND_COLLISION_CONTYPE
    assert finger_base_collision.attrib["conaffinity"] == HAND_COLLISION_CONAFFINITY
    assert root.find(".//position[@name='id0_position']").attrib["joint"] == "id0"
    assert root.find(".//position[@name='id0_position']").attrib["kp"] == "4"
    assert root.find(".//position[@name='id0_position']").attrib["kv"] == "0.1096"
    assert (
        root.find(".//position[@name='id0_position']").attrib["forcerange"]
        == "-0.4 0.4"
    )
    assert (
        root.find(".//position[@name='id1_position']").attrib["ctrlrange"] == "-1 0.75"
    )


def test_write_cube_face_up_scene_references_hand_and_face_sites(
    tmp_path: Path,
) -> None:
    hand_xml = tmp_path / "ghand_right_hand.xml"
    hand_xml.write_text('<mujoco model="hand"><worldbody /></mujoco>', encoding="utf-8")
    scene_xml = tmp_path / "cube_face_up_scene.xml"

    written = write_cube_face_up_scene(hand_xml=hand_xml, scene_xml=scene_xml)

    assert written == scene_xml
    root = ET.parse(scene_xml).getroot()
    hand_mount = root.find(".//body[@name='ghand_hand_mount']")
    assert hand_mount is not None
    assert hand_mount.attrib["quat"] == " ".join(
        f"{value:g}" for value in DISPLAY_HAND_ROTATION_QUAT_WXYZ
    )
    assert root.find(".//body[@name='letter_cube']") is not None
    overview_camera = root.find(".//camera[@name='overview']")
    policy_camera = root.find(f".//camera[@name='{DEFAULT_POLICY_CAMERA_NAME}']")
    assert overview_camera is not None
    assert policy_camera is not None
    assert policy_camera.attrib["pos"] == overview_camera.attrib["pos"]
    assert policy_camera.attrib["xyaxes"] == overview_camera.attrib["xyaxes"]
    assert policy_camera.attrib["fovy"] == "45"
    expected_cube_z = 0.054 + DEFAULT_CUBE_HALF_SIZE
    assert (
        root.find(".//body[@name='letter_cube']").attrib["pos"]
        == f"0.02 -0.008 {expected_cube_z:g}"
    )
    floor = root.find(".//geom[@name='floor']")
    cube_collision = root.find(".//geom[@name='cube_collision']")
    assert floor.attrib["contype"] == FLOOR_COLLISION_CONTYPE
    assert floor.attrib["conaffinity"] == FLOOR_COLLISION_CONAFFINITY
    assert cube_collision.attrib["type"] == "box"
    assert cube_collision.attrib["contype"] == CUBE_COLLISION_CONTYPE
    assert cube_collision.attrib["conaffinity"] == CUBE_COLLISION_CONAFFINITY
    assert root.find(".//pair[@name='cube_palm_soft_contact']") is None
    assert root.find(".//geom[@name='cube_face_A']").attrib["contype"] == "0"
    assert root.find(".//site[@name='cube_face_E_normal']").attrib["user"] == "0 0 1"
    sensor = root.find("./sensor")
    assert sensor is not None
    assert sensor.find(f"./framepos[@name='{CUBE_POSITION_SENSOR_NAME}']") is not None
    assert (
        sensor.find(f"./framequat[@name='{CUBE_QUATERNION_SENSOR_NAME}']") is not None
    )
    assert (
        sensor.find(f"./frameangvel[@name='{CUBE_ANGULAR_VELOCITY_SENSOR_NAME}']")
        is not None
    )
    self_collision_sensors = [
        sensor.find(f"./contact[@name='{name}']")
        for name in HAND_SELF_COLLISION_CONTACT_SENSOR_NAMES
    ]
    assert len(self_collision_sensors) == len(SAFE_HAND_SELF_COLLISION_PAIRS) == 103
    assert all(channel is not None for channel in self_collision_sensors)
    assert all(
        channel.attrib["data"] == "found dist" for channel in self_collision_sensors
    )
    assert all(
        "body1" in channel.attrib and "body2" in channel.attrib
        for channel in self_collision_sensors
    )
    contact_sensor_maxmatch = root.find(
        "./custom/numeric[@name='contact_sensor_maxmatch']"
    )
    assert contact_sensor_maxmatch is not None
    assert contact_sensor_maxmatch.attrib["data"] == "128"
    assert all(
        sensor.find(f"./contact[@name='{name}']") is not None
        for name in CUBE_HAND_CONTACT_SENSOR_NAMES
    )
    assert (
        sensor.find(f"./contact[@name='{CUBE_FLOOR_CONTACT_SENSOR_NAME}']") is not None
    )


def test_write_turn_handle_scene_references_hand_and_task_metadata(
    tmp_path: Path,
) -> None:
    hand_xml = tmp_path / "ghand_right_hand.xml"
    hand_xml.write_text('<mujoco model="hand"><worldbody /></mujoco>', encoding="utf-8")
    scene_xml = tmp_path / "turn_handle_scene.xml"

    written = write_turn_handle_scene(hand_xml=hand_xml, scene_xml=scene_xml)

    assert written == scene_xml
    root = ET.parse(scene_xml).getroot()
    assert root.attrib["model"] == "ghand_turn_handle"
    assert root.find(".//body[@name='ghand_hand_mount']") is not None
    handle = root.find(".//body[@name='turn_handle']")
    assert handle is not None
    joint = handle.find("joint")
    assert joint is not None
    assert joint.attrib["name"] == DEFAULT_TURN_HANDLE_JOINT_NAME
    assert joint.attrib["type"] == "hinge"
    assert joint.attrib["axis"] == "0 0 1"
    assert joint.attrib["limited"] == "false"
    bar = root.find(".//geom[@name='turn_handle_bar']")
    assert bar is not None
    assert bar.attrib["type"] == "capsule"
    assert bar.attrib["fromto"] == "-0.12 0 0 0.12 0 0"
    assert bar.attrib["contype"] == TURN_HANDLE_COLLISION_CONTYPE
    assert bar.attrib["conaffinity"] == TURN_HANDLE_COLLISION_CONAFFINITY
    assert root.find(".//body[@name='turn_handle_support']") is not None
    assert root.find(".//site[@name='turn_handle_axis']") is not None
    target = root.find(".//site[@name='turn_handle_target_tip']")
    assert target is not None
    assert target.attrib["pos"] == "0 0.12 0"
    assert root.find("./sensor/jointpos[@name='turn_handle_angle']").attrib[
        "joint"
    ] == (DEFAULT_TURN_HANDLE_JOINT_NAME)
    assert root.find("./sensor/jointvel[@name='turn_handle_velocity']").attrib[
        "joint"
    ] == (DEFAULT_TURN_HANDLE_JOINT_NAME)


def test_cube_face_colors_match_their_outward_normals(tmp_path: Path) -> None:
    hand_xml = tmp_path / "hand.xml"
    hand_xml.write_text("<mujoco><worldbody /></mujoco>", encoding="utf-8")
    scene_xml = tmp_path / "cube.xml"
    write_cube_face_up_scene(hand_xml=hand_xml, scene_xml=scene_xml)
    root = ET.parse(scene_xml).getroot()
    expected = (
        ("A", (1, 0, 0), "0.90 0.16 0.16 1"),
        ("B", (-1, 0, 0), "0.10 0.42 0.86 1"),
        ("C", (0, 1, 0), "0.12 0.66 0.28 1"),
        ("D", (0, -1, 0), "0.94 0.78 0.12 1"),
        ("E", (0, 0, 1), "0.78 0.22 0.78 1"),
        ("F", (0, 0, -1), "0.98 0.48 0.12 1"),
    )
    for label, normal, rgba in expected:
        material = root.find(f"./asset/material[@name='cube_face_{label}_mat']")
        face = root.find(f".//geom[@name='cube_face_{label}']")
        site = root.find(f".//site[@name='cube_face_{label}_normal']")
        assert material is not None and face is not None and site is not None
        assert material.attrib["rgba"] == rgba
        assert face.attrib["material"] == material.attrib["name"]
        assert face.attrib["contype"] == face.attrib["conaffinity"] == "0"
        assert face.attrib["mass"] == "0"
        assert tuple(map(float, site.attrib["user"].split())) == normal
        assert tuple(map(float, site.attrib["pos"].split())) == pytest.approx(
            tuple(value * DEFAULT_CUBE_HALF_SIZE for value in normal)
        )
        for value, position, size in zip(
            normal,
            map(float, face.attrib["pos"].split()),
            map(float, face.attrib["size"].split()),
            strict=True,
        ):
            if value:
                assert position * value > 0
                assert position * value + size == pytest.approx(DEFAULT_CUBE_HALF_SIZE)
            else:
                assert position == 0


def test_write_turn_handle_scene_uses_top_down_observation_camera(
    tmp_path: Path,
) -> None:
    hand_xml = tmp_path / "ghand_right_hand.xml"
    hand_xml.write_text('<mujoco model="hand"><worldbody /></mujoco>', encoding="utf-8")
    scene_xml = tmp_path / "turn_handle_scene.xml"

    write_turn_handle_scene(hand_xml=hand_xml, scene_xml=scene_xml)

    camera = ET.parse(scene_xml).getroot().find(".//camera[@name='overview']")
    policy_camera = (
        ET.parse(scene_xml)
        .getroot()
        .find(f".//camera[@name='{DEFAULT_POLICY_CAMERA_NAME}']")
    )
    assert camera is not None
    assert policy_camera is not None
    assert camera.attrib["pos"] == "0.06 0.02 0.46"
    assert camera.attrib["xyaxes"] == "1 0 0 0 1 0"
    assert camera.attrib["fovy"] == "55"
    assert policy_camera.attrib == {
        **camera.attrib,
        "name": DEFAULT_POLICY_CAMERA_NAME,
    }


def test_export_packaged_urdf_loads_with_mujoco(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")

    result = export_mujoco_hand_model(tmp_path, validate=True)

    assert result.hand_xml.is_file()
    assert result.scene_xml is not None and result.scene_xml.is_file()
    assert (
        result.turn_handle_scene_xml is not None
        and result.turn_handle_scene_xml.is_file()
    )
    assert result.joint_names == tuple(f"id{i}" for i in range(16))
    assert result.actuator_names == tuple(f"id{i}_position" for i in range(16))

    hand_root = ET.parse(result.hand_xml).getroot()
    expected_joint_ranges = mujoco_joint_ranges_from_urdf(
        default_right_hand_urdf_path()
    )
    for joint_name, (lower, upper) in expected_joint_ranges.items():
        joint = hand_root.find(f".//joint[@name='{joint_name}']")
        actuator = hand_root.find(f".//position[@name='{joint_name}_position']")
        assert joint is not None
        assert actuator is not None
        assert tuple(
            float(value) for value in joint.attrib["range"].split()
        ) == pytest.approx((lower, upper))
        assert tuple(
            float(value) for value in actuator.attrib["ctrlrange"].split()
        ) == pytest.approx((lower, upper))
    collision_manifest = load_mujoco_collision_asset_manifest()
    expected_collision_asset_count = sum(
        len(source.parts) for source in collision_manifest.sources_by_sha256.values()
    )
    collision_mesh_files = sorted((tmp_path / "meshes").glob("*_collision_mesh.obj"))
    visual_mesh_files = sorted((tmp_path / "meshes").glob("*_visual_mesh.stl"))
    collision_mesh_assets = [
        mesh
        for mesh in hand_root.findall("./asset/mesh")
        if (mesh.get("name") or "").endswith("_collision_mesh")
    ]
    visual_mesh_assets = [
        mesh
        for mesh in hand_root.findall("./asset/mesh")
        if (mesh.get("name") or "").endswith("_visual_mesh")
    ]
    mesh_collision_geoms = [
        geom
        for geom in hand_root.findall(".//geom")
        if geom.get("type") == "mesh"
        and _logical_hand_collision_name(geom.get("name") or "") is not None
    ]
    mesh_visual_geoms = [
        geom
        for geom in hand_root.findall(".//geom")
        if geom.get("type") == "mesh" and (geom.get("name") or "").endswith("_visual")
    ]
    assert len(collision_mesh_files) == expected_collision_asset_count == 43
    assert len(visual_mesh_files) >= 17
    assert len(collision_mesh_assets) == len(collision_mesh_files)
    assert len(visual_mesh_assets) == len(visual_mesh_files)
    assert all(
        (tmp_path / mesh.attrib["file"]).is_file() for mesh in collision_mesh_assets
    )
    assert all(
        (tmp_path / mesh.attrib["file"]).is_file() for mesh in visual_mesh_assets
    )
    assert hand_root.find(".//geom[@name='palm_collision']").attrib["type"] == "mesh"
    visual_mesh_names = {mesh.attrib["name"] for mesh in visual_mesh_assets}
    assert len(mesh_visual_geoms) >= len(visual_mesh_assets)
    assert all(geom.attrib["mesh"] in visual_mesh_names for geom in mesh_visual_geoms)
    assert all(geom.attrib["contype"] == "0" for geom in mesh_visual_geoms)
    assert all(geom.attrib["conaffinity"] == "0" for geom in mesh_visual_geoms)
    assert len(mesh_collision_geoms) == 171
    collision_geoms = [
        geom
        for geom in hand_root.findall(".//geom")
        if _logical_hand_collision_name(geom.get("name") or "") is not None
    ]
    assert all(
        int(geom.attrib["contype"]) & int(HAND_COLLISION_CONTYPE)
        for geom in collision_geoms
    )
    collision_geoms_by_logical_name = {
        logical_name: tuple(
            geom
            for geom in collision_geoms
            if _logical_hand_collision_name(geom.attrib["name"]) == logical_name
        )
        for logical_name in SAFE_HAND_SELF_COLLISION_GEOM_NAMES
    }
    for logical_name, physical_geoms in collision_geoms_by_logical_name.items():
        assert any(geom.attrib["name"] == logical_name for geom in physical_geoms)
        assert len({geom.attrib["contype"] for geom in physical_geoms}) == 1
        assert len({geom.attrib["conaffinity"] for geom in physical_geoms}) == 1
    expected_physical_pair_count = sum(
        len(collision_geoms_by_logical_name[first])
        * len(collision_geoms_by_logical_name[second])
        for first, second in SAFE_HAND_SELF_COLLISION_PAIRS
    )
    assert expected_physical_pair_count == 9512
    logical_pair_index = {
        logical_pair: index
        for index, logical_pair in enumerate(SAFE_HAND_SELF_COLLISION_PAIRS)
    }
    observed_logical_pair_counts: Counter[tuple[str, str]] = Counter()
    self_collision_pairs = hand_root.findall("./contact/pair")
    assert len(self_collision_pairs) == expected_physical_pair_count
    middle_geoms = {
        "hnx330_n102_3_collision",
        "hnx330_n102_1_collision",
        "hnx330_n102_9_collision",
        "hnx330_n102_2_collision",
    }
    ring_geoms = {
        "hnx330_n102_10_collision",
        "hnx330_n102_13_collision",
        "hnx330_n102_14_collision",
        "hnx330_n102_5_collision",
    }
    for pair in self_collision_pairs:
        expected_geoms = (
            _logical_hand_collision_name(pair.attrib["geom1"]),
            _logical_hand_collision_name(pair.attrib["geom2"]),
        )
        assert None not in expected_geoms
        assert expected_geoms in logical_pair_index
        observed_logical_pair_counts[expected_geoms] += 1
        base_name = HAND_SELF_COLLISION_PAIR_NAMES[logical_pair_index[expected_geoms]]
        assert pair.attrib["name"].startswith(f"{base_name}_")
        assert pair.attrib["condim"] == "4"
        assert tuple(float(value) for value in pair.attrib["friction"].split()) == (
            1.25,
            1.25,
            0.03,
            0.001,
            0.001,
        )
        assert tuple(float(value) for value in pair.attrib["solref"].split()) == (
            0.005,
            1.0,
        )
        assert tuple(float(value) for value in pair.attrib["solimp"].split()) == (
            0.99,
            0.999,
            0.0005,
            0.5,
            2.0,
        )
        expected_margin = (
            0.0
            if (
                frozenset(expected_geoms)
                == frozenset({"hnx330_n102_collision", "hnx330_n102_3_collision"})
                or expected_geoms[0] in middle_geoms
                and expected_geoms[1] in ring_geoms
                or expected_geoms[1] in middle_geoms
                and expected_geoms[0] in ring_geoms
            )
            else 0.0015
        )
        assert float(pair.attrib["margin"]) == pytest.approx(expected_margin)
    assert observed_logical_pair_counts == Counter(
        {
            logical_pair: (
                len(collision_geoms_by_logical_name[logical_pair[0]])
                * len(collision_geoms_by_logical_name[logical_pair[1]])
            )
            for logical_pair in SAFE_HAND_SELF_COLLISION_PAIRS
        }
    )
    scene_root = ET.parse(result.scene_xml).getroot()
    assert (
        scene_root.find(".//geom[@name='floor']").attrib["contype"]
        == FLOOR_COLLISION_CONTYPE
    )
    assert (
        scene_root.find(".//geom[@name='floor']").attrib["conaffinity"]
        == FLOOR_COLLISION_CONAFFINITY
    )
    assert (
        scene_root.find(".//geom[@name='cube_collision']").attrib["contype"]
        == CUBE_COLLISION_CONTYPE
    )
    assert (
        scene_root.find(".//geom[@name='cube_collision']").attrib["conaffinity"]
        == CUBE_COLLISION_CONAFFINITY
    )
    assert len(scene_root.findall("./contact/pair")) == expected_physical_pair_count

    hand_model = mujoco.MjModel.from_xml_path(str(result.hand_xml))
    scene_model = mujoco.MjModel.from_xml_path(str(result.scene_xml))
    scene_data = mujoco.MjData(scene_model)
    mujoco.mj_forward(scene_model, scene_data)
    handle_model = mujoco.MjModel.from_xml_path(str(result.turn_handle_scene_xml))
    handle_data = mujoco.MjData(handle_model)
    mujoco.mj_forward(handle_model, handle_data)

    assert hand_model.njnt == 16
    assert hand_model.nu == 16
    assert hand_model.npair == expected_physical_pair_count
    assert scene_model.npair == expected_physical_pair_count
    assert handle_model.npair == expected_physical_pair_count
    assert _maximum_hard_target_self_penetration_m(mujoco, hand_model) < 0.002
    assert hand_model.nmesh >= len(collision_mesh_assets)
    assert hand_model.ngeom >= 17
    assert scene_model.njnt > hand_model.njnt
    assert scene_model.ngeom > hand_model.ngeom
    assert mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, "letter_cube") >= 0
    assert _initial_hand_hand_contact_count(mujoco, scene_model, scene_data) == 0
    assert (
        mujoco.mj_name2id(
            handle_model, mujoco.mjtObj.mjOBJ_JOINT, DEFAULT_TURN_HANDLE_JOINT_NAME
        )
        >= 0
    )
    assert mujoco.mj_name2id(handle_model, mujoco.mjtObj.mjOBJ_BODY, "turn_handle") >= 0
    assert _initial_hand_hand_contact_count(mujoco, handle_model, handle_data) == 0


def test_export_custom_urdf_requires_matching_collision_manifest(
    tmp_path: Path,
) -> None:
    custom_urdf = tmp_path / "custom_hand.urdf"
    custom_urdf.write_bytes(default_right_hand_urdf_path().read_bytes())

    with pytest.raises(
        MujocoExportError,
        match=r"collision asset manifest not found: .*custom_hand\.mujoco-collision\.json",
    ):
        export_mujoco_hand_model(tmp_path / "export", urdf_path=custom_urdf)


def _initial_hand_hand_contact_count(mujoco, model, data) -> int:
    count = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        first = _geom_name(mujoco, model, contact.geom1)
        second = _geom_name(mujoco, model, contact.geom2)
        if _is_hand_collision_geom(first) and _is_hand_collision_geom(second):
            count += 1
    return count


def _maximum_hard_target_self_penetration_m(mujoco, model) -> float:
    data = mujoco.MjData(model)
    hard_target = (
        1.1342034339904785,
        -1.0160917043685913,
        -1.745469570159912,
        0.31416594982147217,
        -0.4030826687812805,
        -1.7454692125320435,
        -1.7454721927642822,
        -1.7453426122665405,
        -0.9602378010749817,
        -1.745362639427185,
        0.31389763951301575,
        -1.7452818155288696,
        -1.8330267667770386,
        -2.102452516555786,
        -1.8853938579559326,
        -1.8850163221359253,
    )
    data.ctrl[:] = hard_target
    maximum_penetration_m = 0.0
    for _ in range(500):
        mujoco.mj_step(model, data)
        step_penetration_m = max(
            (max(-float(data.contact[index].dist), 0.0) for index in range(data.ncon)),
            default=0.0,
        )
        maximum_penetration_m = max(
            maximum_penetration_m,
            step_penetration_m,
        )
    return maximum_penetration_m


def _geom_name(mujoco, model, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""


def _is_hand_collision_geom(name: str) -> bool:
    return _logical_hand_collision_name(name) is not None


def _logical_hand_collision_name(name: str) -> str | None:
    for logical_name in SAFE_HAND_SELF_COLLISION_GEOM_NAMES:
        if name == logical_name or name.startswith(f"{logical_name}_p"):
            return logical_name
    return None
