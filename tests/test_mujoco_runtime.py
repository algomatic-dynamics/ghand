from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ghand.hand import make_default_joint_calibrations
from ghand.sim.mujoco import runtime as mujoco_runtime
from ghand.sim.mujoco.runtime import (
    DEFAULT_MUJOCO_PRESENT_CURRENT_MA,
    DEFAULT_MUJOCO_RUNTIME_ACTUATOR_FORCE_LIMIT,
    DEFAULT_MUJOCO_RUNTIME_ACTUATOR_KP,
    MujocoMotorBus,
    MujocoRuntimeError,
    make_mujoco_hand_runtime,
    resolve_mujoco_scene_path,
)
from ghand.sim.mujoco.profile import (
    DEFAULT_MUJOCO_PHYSICS_PROFILE,
    TELEOP_MUJOCO_PHYSICS_PROFILE,
)


FAST_TEST_MUJOCO_PROFILE = replace(
    DEFAULT_MUJOCO_PHYSICS_PROFILE,
    control_hz=100.0,
)


def test_mujoco_motor_bus_maps_dynamixel_ticks_to_mjcf_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_mujoco = _fake_mujoco_module()
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")
    calibrations = make_default_joint_calibrations()
    bus = MujocoMotorBus(
        model_path=scene_xml,
        calibrations=calibrations,
        step_dt=0.01,
        physics_profile=FAST_TEST_MUJOCO_PROFILE,
    )
    index_mcp_flex = calibrations[9]
    target_tick = index_mcp_flex.angle_rad_to_tick(0.5)

    bus.connect()
    bus.set_torque_enabled(index_mcp_flex.dxl_id, True)
    bus.sync_write_goal_positions({index_mcp_flex.dxl_id: target_tick})

    assert fake_mujoco.step_calls == 0
    assert fake_mujoco.last_data.ctrl[9] == pytest.approx(
        index_mcp_flex.tick_to_angle_rad(target_tick)
    )
    assert bus.read_present_position(index_mcp_flex.dxl_id) == index_mcp_flex.zero_tick

    bus.step()

    assert fake_mujoco.step_calls == 5
    assert fake_mujoco.last_data.ctrl[9] == pytest.approx(
        index_mcp_flex.tick_to_angle_rad(target_tick)
    )
    assert bus.read_present_position(index_mcp_flex.dxl_id) == target_tick
    assert bus.read_present_current_ma(index_mcp_flex.dxl_id) == 0.0


def test_make_mujoco_hand_runtime_uses_existing_operator_runtime_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_mujoco = _fake_mujoco_module()
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")
    runtime = make_mujoco_hand_runtime(
        scene=scene_xml,
        target_hz=20.0,
        target_velocity_limit=1000.0,
        step_dt=0.05,
    )

    assert runtime.controller.bus.step_dt == 0.05
    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.set_target("right_hand.index_mcp_flex.pos", 0.5)
    snapshot = runtime.step(now=0.0)

    index_state = next(
        state
        for state in snapshot.joints
        if state.joint_name == "right_hand.index_mcp_flex.pos"
    )
    assert snapshot.backend == f"mujoco:{scene_xml}"
    index_calibration = runtime.calibrations[9]
    assert index_state.measured == pytest.approx(
        index_calibration.upper_rad,
        abs=1 / index_calibration.ticks_per_radian,
    )
    assert index_state.present_current_ma == DEFAULT_MUJOCO_PRESENT_CURRENT_MA
    assert fake_mujoco.last_model.dof_damping == pytest.approx(
        [TELEOP_MUJOCO_PHYSICS_PROFILE.joint_damping] * 16
    )
    assert fake_mujoco.last_model.dof_armature == pytest.approx(
        [TELEOP_MUJOCO_PHYSICS_PROFILE.joint_armature] * 16
    )
    assert fake_mujoco.last_model.dof_frictionloss == pytest.approx(
        [TELEOP_MUJOCO_PHYSICS_PROFILE.joint_frictionloss] * 16
    )
    assert fake_mujoco.last_model.opt.impratio == pytest.approx(
        TELEOP_MUJOCO_PHYSICS_PROFILE.solver_impratio
    )
    np.testing.assert_allclose(
        fake_mujoco.last_model.jnt_solref,
        np.tile(TELEOP_MUJOCO_PHYSICS_PROFILE.joint_limit_solref, (16, 1)),
    )
    np.testing.assert_allclose(
        fake_mujoco.last_model.jnt_solimp,
        np.tile(TELEOP_MUJOCO_PHYSICS_PROFILE.joint_limit_solimp, (16, 1)),
    )


def test_mujoco_hand_runtime_observe_does_not_step_physics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_mujoco = _fake_mujoco_module()
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")
    runtime = make_mujoco_hand_runtime(
        scene=scene_xml,
        target_hz=20.0,
        target_velocity_limit=1000.0,
        step_dt=0.05,
    )

    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.set_target("right_hand.index_mcp_flex.pos", 0.5)
    runtime.step(now=0.0)
    steps_after_one_tick = fake_mujoco.step_calls

    snapshot = runtime.observe(now=0.05)

    assert fake_mujoco.step_calls == steps_after_one_tick
    index_state = next(
        state
        for state in snapshot.joints
        if state.joint_name == "right_hand.index_mcp_flex.pos"
    )
    index_calibration = runtime.calibrations[9]
    assert index_state.measured == pytest.approx(
        index_calibration.upper_rad,
        abs=1 / index_calibration.ticks_per_radian,
    )


def test_mujoco_motor_bus_reset_scene_restores_initial_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_mujoco = _fake_mujoco_module()
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")
    calibrations = make_default_joint_calibrations()
    bus = MujocoMotorBus(
        model_path=scene_xml,
        calibrations=calibrations,
        step_dt=0.01,
        physics_profile=FAST_TEST_MUJOCO_PROFILE,
    )
    index_mcp_flex = calibrations[9]
    target_tick = index_mcp_flex.angle_rad_to_tick(0.5)

    bus.connect()
    bus.set_torque_enabled(index_mcp_flex.dxl_id, True)
    bus.sync_write_goal_positions({index_mcp_flex.dxl_id: target_tick})
    bus.write_goal_current_ma(index_mcp_flex.dxl_id, 120.0)
    bus.step()
    fake_mujoco.last_data.qvel[9] = 3.0
    fake_mujoco.last_data.time = 1.5

    bus.reset_scene()

    assert fake_mujoco.reset_calls == 1
    assert fake_mujoco.last_data.time == 0.0
    assert fake_mujoco.last_data.qvel[9] == 0.0
    assert fake_mujoco.last_data.ctrl[9] == pytest.approx(0.0)
    assert fake_mujoco.last_data.qpos[9] == pytest.approx(0.0)
    assert bus.read_present_position(index_mcp_flex.dxl_id) == index_mcp_flex.zero_tick
    bus.step()
    assert fake_mujoco.last_data.ctrl[9] == pytest.approx(0.0)


def test_mujoco_motor_bus_renders_model_cameras(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_mujoco = _fake_mujoco_module()
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")
    bus = MujocoMotorBus(
        model_path=scene_xml,
        calibrations=make_default_joint_calibrations(),
        step_dt=0.01,
        physics_profile=FAST_TEST_MUJOCO_PROFILE,
    )

    bus.connect()
    frame = bus.render_camera("overview", width=3, height=2)

    assert bus.camera_names() == ("overview",)
    assert frame.shape == (2, 3, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame == 9)
    assert fake_mujoco.renderer_calls == [("overview", 2, 3)]


def test_mujoco_runtime_steps_physics_once_per_runtime_tick(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_mujoco = _fake_mujoco_module()
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")
    runtime = make_mujoco_hand_runtime(
        scene=scene_xml,
        target_hz=20.0,
        target_velocity_limit=1000.0,
        step_dt=0.05,
    )

    runtime.connect()
    runtime.set_torque_enabled(True)
    runtime.set_target("right_hand.index_mcp_flex.pos", 0.5)

    runtime.step(now=0.0)
    assert fake_mujoco.step_calls == 25

    runtime.step(now=0.05)
    assert fake_mujoco.step_calls == 50


def test_make_mujoco_hand_runtime_defaults_step_dt_from_target_hz(
    tmp_path: Path,
) -> None:
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")

    runtime = make_mujoco_hand_runtime(scene=scene_xml)

    assert runtime.controller.bus.step_dt == pytest.approx(0.1)
    assert runtime.target_hz == pytest.approx(TELEOP_MUJOCO_PHYSICS_PROFILE.control_hz)
    assert runtime.controller.bus.actuator_kp == pytest.approx(
        DEFAULT_MUJOCO_RUNTIME_ACTUATOR_KP
    )
    assert runtime.controller.bus.actuator_force_limit == pytest.approx(
        DEFAULT_MUJOCO_RUNTIME_ACTUATOR_FORCE_LIMIT
    )
    assert runtime.target_velocity_limit == pytest.approx(
        TELEOP_MUJOCO_PHYSICS_PROFILE.target_velocity_limit_rad_s
    )
    assert runtime.controller.bus.physics_profile == TELEOP_MUJOCO_PHYSICS_PROFILE


def test_make_mujoco_hand_runtime_rejects_control_step_mismatch(tmp_path: Path) -> None:
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")

    with pytest.raises(ValueError, match="step_dt must equal 1 / target_hz"):
        make_mujoco_hand_runtime(
            scene=scene_xml,
            target_hz=20.0,
            step_dt=0.01,
        )


def test_make_mujoco_hand_runtime_rejects_current_position_mode(tmp_path: Path) -> None:
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")

    with pytest.raises(ValueError, match="control-mode position"):
        make_mujoco_hand_runtime(
            scene=scene_xml,
            control_mode="current_position",
        )


def test_resolve_mujoco_scene_label_exports_into_fingerprint_cache_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    urdf = _write_mujoco_cache_test_urdf(tmp_path)
    export_calls = _install_fake_mujoco_export(monkeypatch)
    monkeypatch.setattr(mujoco_runtime, "default_right_hand_urdf_path", lambda: urdf)
    monkeypatch.setenv("GHAND_MUJOCO_CACHE_DIR", str(tmp_path / "cache"))

    cube = resolve_mujoco_scene_path("cube_face_up")
    default = resolve_mujoco_scene_path(None)
    turn_handle = resolve_mujoco_scene_path("turn_handle")

    manifest_path = cube.parent / mujoco_runtime.MUJOCO_CACHE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert cube.is_file()
    assert default == cube
    assert turn_handle.parent == cube.parent
    assert cube.parent.parent == tmp_path / "cache"
    assert cube.parent.name.startswith(f"{mujoco_runtime.MUJOCO_EXPORT_CACHE_VERSION}-")
    assert export_calls == [(cube.parent, urdf)]
    assert manifest["fingerprint"] == cube.parent.name
    assert manifest["source_urdf"] == str(urdf)
    assert manifest["outputs"]["cube_face_up_scene_xml"] == "cube_face_up_scene.xml"
    assert manifest["physics_profile"] == DEFAULT_MUJOCO_PHYSICS_PROFILE.to_dict()
    assert manifest["collision_asset_version"] == "v1"
    assert manifest["collision_asset_sha256_by_path"]


def test_resolve_mujoco_scene_regenerates_when_manifest_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    urdf = _write_mujoco_cache_test_urdf(tmp_path)
    export_calls = _install_fake_mujoco_export(monkeypatch)
    monkeypatch.setattr(mujoco_runtime, "default_right_hand_urdf_path", lambda: urdf)
    monkeypatch.setenv("GHAND_MUJOCO_CACHE_DIR", str(tmp_path / "cache"))

    cube = resolve_mujoco_scene_path("cube_face_up")
    (cube.parent / mujoco_runtime.MUJOCO_CACHE_MANIFEST_NAME).unlink()

    assert resolve_mujoco_scene_path("cube_face_up") == cube
    assert export_calls == [(cube.parent, urdf), (cube.parent, urdf)]


def test_resolve_mujoco_scene_regenerates_when_manifest_fingerprint_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    urdf = _write_mujoco_cache_test_urdf(tmp_path)
    export_calls = _install_fake_mujoco_export(monkeypatch)
    monkeypatch.setattr(mujoco_runtime, "default_right_hand_urdf_path", lambda: urdf)
    monkeypatch.setenv("GHAND_MUJOCO_CACHE_DIR", str(tmp_path / "cache"))

    cube = resolve_mujoco_scene_path("cube_face_up")
    manifest_path = cube.parent / mujoco_runtime.MUJOCO_CACHE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fingerprint"] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert resolve_mujoco_scene_path("cube_face_up") == cube
    assert export_calls == [(cube.parent, urdf), (cube.parent, urdf)]


def test_resolve_mujoco_scene_regenerates_when_output_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    urdf = _write_mujoco_cache_test_urdf(tmp_path)
    export_calls = _install_fake_mujoco_export(monkeypatch)
    monkeypatch.setattr(mujoco_runtime, "default_right_hand_urdf_path", lambda: urdf)
    monkeypatch.setenv("GHAND_MUJOCO_CACHE_DIR", str(tmp_path / "cache"))

    cube = resolve_mujoco_scene_path("cube_face_up")
    turn_handle_scene = cube.parent / "turn_handle_scene.xml"
    turn_handle_scene.unlink()

    assert resolve_mujoco_scene_path("turn_handle") == turn_handle_scene
    assert export_calls == [(cube.parent, urdf), (cube.parent, urdf)]


def test_resolve_mujoco_scene_uses_new_cache_when_urdf_mesh_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    urdf = _write_mujoco_cache_test_urdf(tmp_path)
    export_calls = _install_fake_mujoco_export(monkeypatch)
    monkeypatch.setattr(mujoco_runtime, "default_right_hand_urdf_path", lambda: urdf)
    monkeypatch.setenv("GHAND_MUJOCO_CACHE_DIR", str(tmp_path / "cache"))

    first = resolve_mujoco_scene_path("cube_face_up")
    (tmp_path / "meshes" / "finger.stl").write_bytes(b"changed mesh")
    second = resolve_mujoco_scene_path("cube_face_up")

    assert second.parent != first.parent
    assert export_calls == [(first.parent, urdf), (second.parent, urdf)]


def test_resolve_mujoco_scene_uses_new_cache_when_collision_asset_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    urdf = _write_mujoco_cache_test_urdf(tmp_path)
    collision_part = tmp_path / "collision_part.obj"
    collision_part.write_bytes(b"collision asset v1")
    collision_assets = SimpleNamespace(
        asset_version="test_collision_assets_v1",
        files=(collision_part,),
    )
    export_calls = _install_fake_mujoco_export(monkeypatch)
    monkeypatch.setattr(mujoco_runtime, "default_right_hand_urdf_path", lambda: urdf)
    monkeypatch.setattr(
        mujoco_runtime,
        "load_mujoco_collision_asset_manifest",
        lambda: collision_assets,
    )
    monkeypatch.setenv("GHAND_MUJOCO_CACHE_DIR", str(tmp_path / "cache"))

    first = resolve_mujoco_scene_path("cube_face_up")
    collision_part.write_bytes(b"collision asset v2")
    second = resolve_mujoco_scene_path("cube_face_up")

    assert second.parent != first.parent
    assert export_calls == [(first.parent, urdf), (second.parent, urdf)]


def test_resolve_mujoco_scene_uses_new_cache_when_physics_profile_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    urdf = _write_mujoco_cache_test_urdf(tmp_path)
    export_calls = _install_fake_mujoco_export(monkeypatch)
    monkeypatch.setattr(mujoco_runtime, "default_right_hand_urdf_path", lambda: urdf)
    monkeypatch.setenv("GHAND_MUJOCO_CACHE_DIR", str(tmp_path / "cache"))

    first = resolve_mujoco_scene_path("cube_face_up")
    changed_profile = replace(DEFAULT_MUJOCO_PHYSICS_PROFILE, actuator_kp=7.0)
    second = resolve_mujoco_scene_path(
        "cube_face_up",
        physics_profile=changed_profile,
    )

    assert second.parent != first.parent
    manifest = json.loads(
        (second.parent / mujoco_runtime.MUJOCO_CACHE_MANIFEST_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["physics_profile"]["actuator_kp"] == 7.0
    assert export_calls == [(first.parent, urdf), (second.parent, urdf)]


def test_resolve_mujoco_scene_rejects_unknown_label() -> None:
    with pytest.raises(MujocoRuntimeError, match="cube_face_up, turn_handle"):
        resolve_mujoco_scene_path("no_such_scene")


def test_resolve_mujoco_scene_accepts_existing_xml_path(tmp_path: Path) -> None:
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<mujoco />", encoding="utf-8")

    assert resolve_mujoco_scene_path(scene_xml) == scene_xml


def _write_mujoco_cache_test_urdf(tmp_path: Path) -> Path:
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "finger.stl").write_bytes(b"solid mesh")
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
<robot name="cache_test_hand">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="meshes/finger.stl" />
      </geometry>
    </visual>
  </link>
</robot>
""".strip(),
        encoding="utf-8",
    )
    return urdf


def _install_fake_mujoco_export(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Path, Path | None]]:
    export_calls: list[tuple[Path, Path | None]] = []

    def fake_export(
        output_dir: Path | str,
        *,
        urdf_path: Path | str | None = None,
        **_: object,
    ) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name in (
            "ghand_right_hand.xml",
            "cube_face_up_scene.xml",
            "turn_handle_scene.xml",
        ):
            (output / name).write_text("<mujoco />", encoding="utf-8")
        export_calls.append(
            (output, Path(urdf_path) if urdf_path is not None else None)
        )

    monkeypatch.setattr(mujoco_runtime, "export_mujoco_hand_model", fake_export)
    return export_calls


def _fake_mujoco_module() -> SimpleNamespace:
    module = SimpleNamespace()
    module.mjtObj = SimpleNamespace(
        mjOBJ_JOINT="joint",
        mjOBJ_ACTUATOR="actuator",
        mjOBJ_CAMERA="camera",
    )
    module.step_calls = 0
    module.forward_calls = 0
    module.reset_calls = 0
    module.renderer_calls = []
    module.last_model = None
    module.last_data = None

    class MjModel:
        def __init__(self, path: str) -> None:
            self.path = path
            self.opt = SimpleNamespace(timestep=0.002, impratio=0.0)
            self.joint_name_to_id = {f"id{i}": i for i in range(16)}
            self.actuator_name_to_id = {f"id{i}_position": i for i in range(16)}
            self.camera_name_to_id = {"overview": 0}
            self.camera_names = ("overview",)
            self.ncam = 1
            self.jnt_qposadr = list(range(16))
            self.jnt_dofadr = list(range(16))
            self.jnt_solref = np.zeros((16, 2), dtype=np.float64)
            self.jnt_solimp = np.zeros((16, 5), dtype=np.float64)
            self.dof_damping = [0.0] * 16
            self.dof_armature = [0.0] * 16
            self.dof_frictionloss = [0.0] * 16

        @classmethod
        def from_xml_path(cls, path: str) -> MjModel:
            model = cls(path)
            module.last_model = model
            return model

    def mj_data(model: MjModel) -> SimpleNamespace:
        data = SimpleNamespace(
            model=model,
            qpos=[0.0] * 16,
            qvel=[0.0] * 16,
            ctrl=[0.0] * 16,
            time=0.0,
        )
        module.last_data = data
        return data

    def mj_name2id(model: MjModel, obj_type: str, name: str) -> int:
        if obj_type == module.mjtObj.mjOBJ_JOINT:
            return model.joint_name_to_id.get(name, -1)
        if obj_type == module.mjtObj.mjOBJ_ACTUATOR:
            return model.actuator_name_to_id.get(name, -1)
        if obj_type == module.mjtObj.mjOBJ_CAMERA:
            return model.camera_name_to_id.get(name, -1)
        return -1

    def mj_id2name(model: MjModel, obj_type: str, obj_id: int) -> str | None:
        if obj_type == module.mjtObj.mjOBJ_CAMERA:
            try:
                return model.camera_names[obj_id]
            except IndexError:
                return None
        return None

    def mj_forward(_model: MjModel, _data: SimpleNamespace) -> None:
        module.forward_calls += 1

    def mj_reset_data(_model: MjModel, data: SimpleNamespace) -> None:
        module.reset_calls += 1
        data.qpos[:] = [0.0] * len(data.qpos)
        data.qvel[:] = [0.0] * len(data.qvel)
        data.ctrl[:] = [0.0] * len(data.ctrl)
        data.time = 0.0

    def mj_step(_model: MjModel, data: SimpleNamespace) -> None:
        module.step_calls += 1
        data.time += _model.opt.timestep
        for index, ctrl in enumerate(data.ctrl):
            data.qpos[index] = ctrl

    class Renderer:
        def __init__(self, _model: MjModel, *, height: int, width: int) -> None:
            self.height = height
            self.width = width
            self.camera = None

        def update_scene(self, _data: SimpleNamespace, *, camera: str) -> None:
            self.camera = camera
            module.renderer_calls.append((camera, self.height, self.width))

        def render(self) -> np.ndarray:
            return np.full((self.height, self.width, 3), 9, dtype=np.uint8)

    module.MjModel = MjModel
    module.MjData = mj_data
    module.Renderer = Renderer
    module.mj_name2id = mj_name2id
    module.mj_id2name = mj_id2name
    module.mj_forward = mj_forward
    module.mj_resetData = mj_reset_data
    module.mj_step = mj_step
    return module
