"""MuJoCo-backed hand runtime adapters for teleoperation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

import numpy as np

from ghand.control import (
    HandOperatorRuntime,
    OperatorControlMode,
)
from ghand.hand import (
    default_right_hand_urdf_path,
    JointCalibration,
)
from ghand.hardware import (
    DEFAULT_FAKE_MODEL_NUMBER,
    DynamixelBusError,
    DynamixelPingResult,
    HandController,
    OPERATING_MODE_POSITION,
    normalize_position_tick,
)
from ghand.sim.mujoco.collision_assets import (
    MujocoCollisionAssetError,
    load_mujoco_collision_asset_manifest,
)
from ghand.sim.mujoco.exporter import (
    DEFAULT_CUBE_SCENE_XML_NAME,
    DEFAULT_HAND_XML_NAME,
    DEFAULT_TURN_HANDLE_SCENE_XML_NAME,
    export_mujoco_hand_model,
)
from ghand.sim.mujoco.profile import (
    DEFAULT_MUJOCO_PHYSICS_PROFILE,
    TELEOP_MUJOCO_PHYSICS_PROFILE,
    MujocoPhysicsProfile,
    make_mujoco_joint_calibrations,
)


DEFAULT_MUJOCO_CONTROL_HZ: Final[float] = TELEOP_MUJOCO_PHYSICS_PROFILE.control_hz
DEFAULT_MUJOCO_STEP_DT: Final[float] = 1.0 / DEFAULT_MUJOCO_CONTROL_HZ
# Keep the runtime actuator overrides aligned with the exporter values the
# generated scene XMLs were produced (and tuned) with.
DEFAULT_MUJOCO_RUNTIME_ACTUATOR_KP: Final[float] = (
    TELEOP_MUJOCO_PHYSICS_PROFILE.actuator_kp
)
DEFAULT_MUJOCO_RUNTIME_ACTUATOR_FORCE_LIMIT: Final[float] = (
    TELEOP_MUJOCO_PHYSICS_PROFILE.actuator_force_limit
)
DEFAULT_MUJOCO_TARGET_VELOCITY_LIMIT: Final[float] = (
    TELEOP_MUJOCO_PHYSICS_PROFILE.target_velocity_limit_rad_s
)
DEFAULT_MUJOCO_PRESENT_CURRENT_MA: Final[float] = 0.0
MUJOCO_SCENE_LABELS: Final[tuple[str, ...]] = ("cube_face_up", "turn_handle")
DEFAULT_MUJOCO_SCENE_LABEL: Final[str] = "cube_face_up"
_SCENE_XML_NAME_BY_LABEL: Final[dict[str, str]] = {
    "cube_face_up": DEFAULT_CUBE_SCENE_XML_NAME,
    "turn_handle": DEFAULT_TURN_HANDLE_SCENE_XML_NAME,
}
DEFAULT_MUJOCO_TEMPERATURE_CELSIUS: Final[int] = 0
DEFAULT_MUJOCO_INPUT_VOLTAGE: Final[float] = 0.0
MUJOCO_EXPORT_CACHE_VERSION: Final[str] = "v17"
MUJOCO_CACHE_MANIFEST_NAME: Final[str] = "manifest.json"
_MUJOCO_CACHE_OUTPUTS: Final[dict[str, str]] = {
    "hand_xml": DEFAULT_HAND_XML_NAME,
    "cube_face_up_scene_xml": DEFAULT_CUBE_SCENE_XML_NAME,
    "turn_handle_scene_xml": DEFAULT_TURN_HANDLE_SCENE_XML_NAME,
}


@dataclass(frozen=True)
class _MujocoSceneCacheKey:
    fingerprint: str
    ghand_version: str
    mujoco_version: str
    source_urdf: Path
    source_urdf_sha256: str
    mesh_sha256_by_path: dict[str, str]
    collision_asset_version: str
    collision_asset_sha256_by_path: dict[str, str]
    physics_profile: dict[str, Any]


class MujocoRuntimeError(RuntimeError):
    """Raised when the MuJoCo hand backend cannot be initialized."""


class MujocoMotorBus:
    """Dynamixel-like bus backed by a MuJoCo hand model.

    The public methods intentionally match :class:`ghand.hardware.MotorBus`.
    This lets ``HandOperatorRuntime`` drive either real Dynamixels or a MuJoCo
    model without changing the teleoperation loop.
    """

    def __init__(
        self,
        *,
        model_path: Path | str,
        calibrations: Sequence[JointCalibration],
        step_dt: float | None = None,
        physics_profile: MujocoPhysicsProfile = TELEOP_MUJOCO_PHYSICS_PROFILE,
        actuator_kp: float | None = None,
        actuator_force_limit: float | None = None,
        model_number: int = DEFAULT_FAKE_MODEL_NUMBER,
        viewer: bool = False,
        resource_owner: object | None = None,
    ) -> None:
        if not calibrations:
            raise ValueError("At least one joint calibration is required.")
        dxl_ids = [calibration.dxl_id for calibration in calibrations]
        if len(set(dxl_ids)) != len(dxl_ids):
            raise ValueError("Joint calibration motor IDs must be unique.")

        self.model_path = Path(model_path)
        self.calibrations = tuple(calibrations)
        self.physics_profile = physics_profile.with_runtime_overrides(
            actuator_kp=actuator_kp,
            actuator_force_limit=actuator_force_limit,
        )
        profile_step_dt = 1.0 / self.physics_profile.control_hz
        if step_dt is not None and not math.isclose(
            float(step_dt),
            profile_step_dt,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("step_dt must equal 1 / physics_profile.control_hz.")
        self.step_dt = profile_step_dt
        self.actuator_kp = self.physics_profile.actuator_kp
        self.actuator_force_limit = self.physics_profile.actuator_force_limit
        self.model_number = int(model_number)
        self.viewer = bool(viewer)
        self.is_connected = False
        self.calls: list[tuple[object, ...]] = []

        self._resource_owner = resource_owner
        self._mujoco: Any | None = None
        self._model: Any | None = None
        self._data: Any | None = None
        self._passive_viewer: Any | None = None
        self._renderer_by_size: dict[tuple[int, int], Any] = {}
        self._steps_per_runtime_tick = 1
        self._joint_id_by_dxl: dict[int, int] = {}
        self._joint_qposadr_by_dxl: dict[int, int] = {}
        self._joint_dofadr_by_dxl: dict[int, int] = {}
        self._actuator_id_by_dxl: dict[int, int] = {}
        self._calibration_by_dxl = {
            calibration.dxl_id: calibration for calibration in self.calibrations
        }
        self._goal_position_by_dxl = {
            calibration.dxl_id: calibration.zero_tick for calibration in self.calibrations
        }
        self._goal_current_ma_by_dxl = {
            calibration.dxl_id: DEFAULT_MUJOCO_PRESENT_CURRENT_MA
            for calibration in self.calibrations
        }
        self._torque_enabled_by_dxl = {
            calibration.dxl_id: False for calibration in self.calibrations
        }
        self._operating_mode_by_dxl = {
            calibration.dxl_id: OPERATING_MODE_POSITION
            for calibration in self.calibrations
        }
        self._profile_by_dxl = {
            calibration.dxl_id: (0, 0) for calibration in self.calibrations
        }

    @property
    def model(self) -> Any:
        """Return the loaded MuJoCo model."""

        if self._model is None:
            raise MujocoRuntimeError("MuJoCo model is not connected.")
        return self._model

    @property
    def data(self) -> Any:
        """Return the loaded MuJoCo data."""

        if self._data is None:
            raise MujocoRuntimeError("MuJoCo data is not connected.")
        return self._data

    def connect(self) -> None:
        """Load the MuJoCo model and initialize hand joint controls."""

        self.calls.append(("connect",))
        if self.is_connected:
            return
        if not self.model_path.is_file():
            raise MujocoRuntimeError(f"MuJoCo model XML not found: {self.model_path}")

        mujoco = _import_mujoco()
        try:
            model = mujoco.MjModel.from_xml_path(str(self.model_path))
            data = mujoco.MjData(model)
        except Exception as exc:
            raise MujocoRuntimeError(
                f"Failed to load MuJoCo model XML: {self.model_path}: {exc}"
            ) from exc

        self._mujoco = mujoco
        self._model = model
        self._data = data
        self._steps_per_runtime_tick = _steps_for_dt(model, self.step_dt)
        self._resolve_channels()
        self._apply_actuator_overrides()
        self._reset_hand_state()
        self._passive_viewer = self._launch_viewer() if self.viewer else None
        self.is_connected = True

    def disconnect(self) -> None:
        """Close the optional viewer and detach the MuJoCo model."""

        self.calls.append(("disconnect",))
        if self.is_connected:
            for dxl_id in self._torque_enabled_by_dxl:
                self._torque_enabled_by_dxl[dxl_id] = False
            self._apply_controls()
        if self._passive_viewer is not None:
            close = getattr(self._passive_viewer, "close", None)
            if close is not None:
                close()
        for renderer in self._renderer_by_size.values():
            close = getattr(renderer, "close", None)
            if close is not None:
                close()
        self._renderer_by_size.clear()
        self._passive_viewer = None
        self._model = None
        self._data = None
        self._mujoco = None
        self.is_connected = False

    def ping(self, dxl_id: int) -> DynamixelPingResult:
        self.calls.append(("ping", dxl_id))
        self._calibration(dxl_id)
        return DynamixelPingResult(dxl_id=dxl_id, model_number=self.model_number)

    def set_torque_enabled(self, dxl_id: int, enabled: bool) -> None:
        self.calls.append(("torque", dxl_id, enabled))
        self._calibration(dxl_id)
        self._torque_enabled_by_dxl[int(dxl_id)] = bool(enabled)
        if self.is_connected:
            self._apply_controls()

    def write_operating_mode(self, dxl_id: int, mode: int) -> None:
        self.calls.append(("mode", dxl_id, mode))
        self._calibration(dxl_id)
        self._operating_mode_by_dxl[int(dxl_id)] = int(mode)

    def write_profile(self, dxl_id: int, *, acceleration: int, velocity: int) -> None:
        self.calls.append(("profile", dxl_id, acceleration, velocity))
        self._calibration(dxl_id)
        self._profile_by_dxl[int(dxl_id)] = (int(acceleration), int(velocity))

    def write_goal_position(self, dxl_id: int, position: int) -> None:
        self.calls.append(("goal", dxl_id, normalize_position_tick(position)))
        self._write_goal_positions({int(dxl_id): int(position)})

    def sync_write_goal_positions(self, positions: Mapping[int, int]) -> None:
        if not positions:
            return
        normalized_positions = {
            int(dxl_id): normalize_position_tick(position)
            for dxl_id, position in positions.items()
        }
        self.calls.append(("sync_goal", normalized_positions))
        self._write_goal_positions(normalized_positions)

    def step(self) -> None:
        """Advance one runtime tick of MuJoCo physics."""

        self.calls.append(("step",))
        if not self.is_connected:
            return
        self._apply_controls()
        self._step()

    def reset_scene(self) -> None:
        """Reset MuJoCo scene data to the compiled initial state."""

        self.calls.append(("reset_scene",))
        if not self.is_connected:
            return
        mujoco = self._required_mujoco()
        reset_data = getattr(mujoco, "mj_resetData", None)
        if reset_data is None:
            raise MujocoRuntimeError("MuJoCo mj_resetData is not available.")
        reset_data(self.model, self.data)
        for calibration in self.calibrations:
            dxl_id = calibration.dxl_id
            self._goal_position_by_dxl[dxl_id] = calibration.zero_tick
            self._goal_current_ma_by_dxl[dxl_id] = DEFAULT_MUJOCO_PRESENT_CURRENT_MA
        self._reset_hand_state()
        self._sync_viewer()

    def camera_names(self) -> tuple[str, ...]:
        """Return names of cameras compiled into the connected MuJoCo model."""

        model = self.model
        mujoco = self._required_mujoco()
        camera_count = int(getattr(model, "ncam", 0))
        if camera_count <= 0:
            return ()
        camera_obj = getattr(mujoco.mjtObj, "mjOBJ_CAMERA", None)
        id_to_name = getattr(mujoco, "mj_id2name", None)
        names: list[str] = []
        for camera_id in range(camera_count):
            name = None
            if id_to_name is not None and camera_obj is not None:
                name = id_to_name(model, camera_obj, camera_id)
            if not name:
                model_names = getattr(model, "camera_names", None)
                if model_names is not None:
                    name = model_names[camera_id]
            names.append(str(name or f"camera_{camera_id}"))
        return tuple(names)

    def render_camera(self, camera_name: str, *, width: int, height: int) -> np.ndarray:
        """Render one RGB frame from a named MuJoCo camera."""

        if width <= 0 or height <= 0:
            raise ValueError("render width and height must be positive.")
        renderer = self._renderer_for_size(width=width, height=height)
        renderer.update_scene(self.data, camera=camera_name)
        frame = np.asarray(renderer.render())
        expected_shape = (height, width, 3)
        if frame.shape != expected_shape:
            raise MujocoRuntimeError(
                f"MuJoCo camera {camera_name!r} rendered shape {frame.shape}, "
                f"expected {expected_shape}."
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return np.ascontiguousarray(frame)

    def write_goal_current_ma(self, dxl_id: int, current_ma: float) -> None:
        current = float(current_ma)
        self.calls.append(("goal_current", dxl_id, current))
        self._calibration(dxl_id)
        self._goal_current_ma_by_dxl[int(dxl_id)] = current

    def sync_write_goal_currents_ma(self, currents_ma: Mapping[int, float]) -> None:
        if not currents_ma:
            return
        currents = {int(dxl_id): float(current_ma) for dxl_id, current_ma in currents_ma.items()}
        self.calls.append(("sync_goal_current", currents))
        for dxl_id, current_ma in currents.items():
            self._calibration(dxl_id)
            self._goal_current_ma_by_dxl[dxl_id] = current_ma

    def read_present_position(self, dxl_id: int) -> int:
        self.calls.append(("read_position", dxl_id))
        return self._read_position_tick(dxl_id)

    def read_present_current_ma(self, dxl_id: int) -> float:
        self.calls.append(("read_current", dxl_id))
        self._calibration(dxl_id)
        return DEFAULT_MUJOCO_PRESENT_CURRENT_MA

    def read_present_temperature(self, dxl_id: int) -> int:
        self.calls.append(("read_temperature", dxl_id))
        self._calibration(dxl_id)
        return DEFAULT_MUJOCO_TEMPERATURE_CELSIUS

    def read_present_input_voltage(self, dxl_id: int) -> float:
        self.calls.append(("read_voltage", dxl_id))
        self._calibration(dxl_id)
        return DEFAULT_MUJOCO_INPUT_VOLTAGE

    def read_hardware_error_status(self, dxl_id: int) -> int:
        self.calls.append(("read_error", dxl_id))
        self._calibration(dxl_id)
        return 0

    def sync_read_present_positions(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_position", ids))
        return {dxl_id: self._read_position_tick(dxl_id) for dxl_id in ids}

    def sync_read_present_currents_ma(self, dxl_ids: Iterable[int]) -> dict[int, float]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_current", ids))
        return {dxl_id: DEFAULT_MUJOCO_PRESENT_CURRENT_MA for dxl_id in ids}

    def sync_read_present_temperatures(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_temperature", ids))
        return {dxl_id: DEFAULT_MUJOCO_TEMPERATURE_CELSIUS for dxl_id in ids}

    def sync_read_present_input_voltages(self, dxl_ids: Iterable[int]) -> dict[int, float]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_voltage", ids))
        return {dxl_id: DEFAULT_MUJOCO_INPUT_VOLTAGE for dxl_id in ids}

    def sync_read_hardware_error_statuses(self, dxl_ids: Iterable[int]) -> dict[int, int]:
        ids = tuple(int(dxl_id) for dxl_id in dxl_ids)
        self.calls.append(("sync_read_error", ids))
        return {dxl_id: 0 for dxl_id in ids}

    def _write_goal_positions(self, positions: Mapping[int, int]) -> None:
        for dxl_id, position in positions.items():
            calibration = self._calibration(dxl_id)
            self._goal_position_by_dxl[calibration.dxl_id] = normalize_position_tick(position)
        if self.is_connected:
            self._apply_controls()

    def _resolve_channels(self) -> None:
        mujoco = self._required_mujoco()
        model = self.model
        joint_obj = mujoco.mjtObj.mjOBJ_JOINT
        actuator_obj = mujoco.mjtObj.mjOBJ_ACTUATOR
        for calibration in self.calibrations:
            joint_name = f"id{calibration.pose_dxl_id}"
            actuator_name = f"{joint_name}_position"
            joint_id = int(mujoco.mj_name2id(model, joint_obj, joint_name))
            actuator_id = int(mujoco.mj_name2id(model, actuator_obj, actuator_name))
            if joint_id < 0:
                raise MujocoRuntimeError(
                    f"MuJoCo model is missing hand joint {joint_name!r}."
                )
            if actuator_id < 0:
                raise MujocoRuntimeError(
                    f"MuJoCo model is missing position actuator {actuator_name!r}."
                )
            self._joint_id_by_dxl[calibration.dxl_id] = joint_id
            self._joint_qposadr_by_dxl[calibration.dxl_id] = int(model.jnt_qposadr[joint_id])
            if hasattr(model, "jnt_dofadr"):
                self._joint_dofadr_by_dxl[calibration.dxl_id] = int(
                    model.jnt_dofadr[joint_id]
                )
            self._actuator_id_by_dxl[calibration.dxl_id] = actuator_id

    def _reset_hand_state(self) -> None:
        for calibration in self.calibrations:
            dxl_id = calibration.dxl_id
            angle = calibration.tick_to_angle_rad(self._goal_position_by_dxl[dxl_id])
            self.data.qpos[self._joint_qposadr_by_dxl[dxl_id]] = angle
            self.data.ctrl[self._actuator_id_by_dxl[dxl_id]] = angle
        self._required_mujoco().mj_forward(self.model, self.data)

    def _apply_actuator_overrides(self) -> None:
        model = self.model
        if hasattr(model, "opt"):
            model.opt.impratio = self.physics_profile.solver_impratio
        joint_ids = tuple(self._joint_id_by_dxl.values())
        if hasattr(model, "jnt_solref"):
            for joint_id in joint_ids:
                model.jnt_solref[joint_id, :] = self.physics_profile.joint_limit_solref
        if hasattr(model, "jnt_solimp"):
            for joint_id in joint_ids:
                model.jnt_solimp[joint_id, :] = self.physics_profile.joint_limit_solimp
        dof_ids = tuple(self._joint_dofadr_by_dxl.values())
        if hasattr(model, "dof_damping"):
            for dof_id in dof_ids:
                model.dof_damping[dof_id] = self.physics_profile.joint_damping
        if hasattr(model, "dof_armature"):
            for dof_id in dof_ids:
                model.dof_armature[dof_id] = self.physics_profile.joint_armature
        if hasattr(model, "dof_frictionloss"):
            for dof_id in dof_ids:
                model.dof_frictionloss[dof_id] = self.physics_profile.joint_frictionloss
        actuator_ids = tuple(self._actuator_id_by_dxl.values())
        if self.actuator_force_limit is not None and hasattr(model, "actuator_forcerange"):
            for actuator_id in actuator_ids:
                model.actuator_forcerange[actuator_id, 0] = -self.actuator_force_limit
                model.actuator_forcerange[actuator_id, 1] = self.actuator_force_limit
            if hasattr(model, "actuator_forcelimited"):
                for actuator_id in actuator_ids:
                    model.actuator_forcelimited[actuator_id] = True
        if self.actuator_kp is None:
            return
        if hasattr(model, "actuator_gainprm"):
            for actuator_id in actuator_ids:
                model.actuator_gainprm[actuator_id, 0] = self.actuator_kp
        if hasattr(model, "actuator_biasprm"):
            for actuator_id in actuator_ids:
                model.actuator_biasprm[actuator_id, 1] = -self.actuator_kp
                model.actuator_biasprm[actuator_id, 2] = (
                    -self.physics_profile.actuator_velocity_feedback_gain
                )

    def _apply_controls(self) -> None:
        if not self.is_connected and self._data is None:
            return
        for calibration in self.calibrations:
            dxl_id = calibration.dxl_id
            if self._torque_enabled_by_dxl[dxl_id]:
                target = calibration.tick_to_angle_rad(self._goal_position_by_dxl[dxl_id])
            else:
                target = float(self.data.qpos[self._joint_qposadr_by_dxl[dxl_id]])
            self.data.ctrl[self._actuator_id_by_dxl[dxl_id]] = target

    def _step(self) -> None:
        mujoco = self._required_mujoco()
        for _ in range(self._steps_per_runtime_tick):
            mujoco.mj_step(self.model, self.data)
        self._sync_viewer()

    def _sync_viewer(self) -> None:
        if self._passive_viewer is None:
            return
        sync = getattr(self._passive_viewer, "sync", None)
        if sync is not None:
            sync()

    def _launch_viewer(self) -> object:
        try:
            viewer_module = importlib.import_module("mujoco.viewer")
            return viewer_module.launch_passive(self.model, self.data)
        except Exception as exc:
            raise MujocoRuntimeError(f"Failed to launch MuJoCo viewer: {exc}") from exc

    def _renderer_for_size(self, *, width: int, height: int) -> object:
        key = (height, width)
        renderer = self._renderer_by_size.get(key)
        if renderer is not None:
            return renderer
        renderer_cls = getattr(self._required_mujoco(), "Renderer", None)
        if renderer_cls is None:
            raise MujocoRuntimeError("MuJoCo Renderer is not available.")
        try:
            renderer = renderer_cls(self.model, height=height, width=width)
        except Exception as exc:
            raise MujocoRuntimeError(f"Failed to create MuJoCo renderer: {exc}") from exc
        self._renderer_by_size[key] = renderer
        return renderer

    def _read_position_tick(self, dxl_id: int) -> int:
        calibration = self._calibration(dxl_id)
        if not self.is_connected:
            return normalize_position_tick(self._goal_position_by_dxl[calibration.dxl_id])
        angle = float(self.data.qpos[self._joint_qposadr_by_dxl[calibration.dxl_id]])
        return calibration.angle_rad_to_tick(angle)

    def _calibration(self, dxl_id: int) -> JointCalibration:
        try:
            return self._calibration_by_dxl[int(dxl_id)]
        except KeyError as exc:
            raise DynamixelBusError(f"MuJoCo bus has no motor with ID {dxl_id}.") from exc

    def _required_mujoco(self) -> Any:
        if self._mujoco is None:
            raise MujocoRuntimeError("MuJoCo model is not connected.")
        return self._mujoco


def make_mujoco_hand_runtime(
    *,
    scene: Path | str | None = None,
    side: str = "right",
    dry_run: bool = False,
    target_hz: float = DEFAULT_MUJOCO_CONTROL_HZ,
    readback_hz: float | None = None,
    target_velocity_limit: float = None,
    control_mode: OperatorControlMode = "position",
    default_current_limit_ma: float | None = None,
    max_current_limit_ma: float | None = None,
    step_dt: float | None = None,
    actuator_kp: float | None = None,
    actuator_force_limit: float | None = None,
    physics_profile: MujocoPhysicsProfile = TELEOP_MUJOCO_PHYSICS_PROFILE,
    viewer: bool = False,
) -> HandOperatorRuntime:
    """Create a ``HandOperatorRuntime`` backed by a MuJoCo scene.

    ``scene`` accepts a built-in scene label (``cube_face_up`` or
    ``turn_handle``) or a path to a custom scene XML. Built-in scenes are
    exported from the packaged URDF into a per-version cache directory on
    first use.
    """

    if side != "right":
        raise ValueError("MuJoCo teleop currently supports side='right' only.")
    if control_mode != "position":
        raise ValueError("MuJoCo teleop currently supports --control-mode position only.")
    if target_hz <= 0:
        raise ValueError("target_hz must be positive.")

    effective_profile = physics_profile.with_runtime_overrides(
        control_hz=target_hz,
        target_velocity_limit_rad_s=target_velocity_limit,
        actuator_kp=actuator_kp,
        actuator_force_limit=actuator_force_limit,
    )
    source_urdf = default_right_hand_urdf_path()
    resolved_scene = resolve_mujoco_scene_path(
        scene,
        physics_profile=effective_profile,
    )
    try:
        calibrations = make_mujoco_joint_calibrations(
            urdf_path=source_urdf,
            side=side,
        )
    except (OSError, ValueError, ET.ParseError) as exc:
        raise MujocoRuntimeError(
            f"Failed to load MuJoCo runtime joint limits from {source_urdf}: {exc}"
        ) from exc
    resolved_step_dt = 1.0 / effective_profile.control_hz
    if step_dt is not None and not math.isclose(
        float(step_dt),
        resolved_step_dt,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "step_dt must equal 1 / target_hz so control and simulation time stay aligned."
        )
    bus = MujocoMotorBus(
        model_path=resolved_scene,
        calibrations=calibrations,
        step_dt=resolved_step_dt,
        physics_profile=effective_profile,
        viewer=viewer,
    )
    return HandOperatorRuntime(
        HandController(bus),
        calibrations,
        backend=f"mujoco:{resolved_scene}",
        dry_run=dry_run,
        target_hz=target_hz,
        readback_hz=readback_hz,
        target_velocity_limit=effective_profile.target_velocity_limit_rad_s,
        control_mode=control_mode,
        default_current_limit_ma=default_current_limit_ma,
        max_current_limit_ma=max_current_limit_ma,
    )


def resolve_mujoco_scene_path(
    scene: Path | str | None = None,
    *,
    physics_profile: MujocoPhysicsProfile = DEFAULT_MUJOCO_PHYSICS_PROFILE,
) -> Path:
    """Resolve a built-in scene label or custom scene XML path.

    ``scene`` may be one of :data:`MUJOCO_SCENE_LABELS` (generated into the
    cache directory on first use), a path to an existing scene XML, or
    ``None`` for the default ``cube_face_up`` scene.
    """

    if scene is None:
        scene = DEFAULT_MUJOCO_SCENE_LABEL
    label = str(scene)
    if label in _SCENE_XML_NAME_BY_LABEL:
        return mujoco_scene_path_for_label(label, physics_profile=physics_profile)
    path = Path(scene)
    if not path.is_file():
        labels = ", ".join(MUJOCO_SCENE_LABELS)
        raise MujocoRuntimeError(
            f"Unknown MuJoCo scene {label!r}: expected a built-in scene label "
            f"({labels}) or a path to an existing scene XML."
        )
    return path


def mujoco_scene_path_for_label(
    label: str,
    *,
    cache_dir: Path | str | None = None,
    physics_profile: MujocoPhysicsProfile = DEFAULT_MUJOCO_PHYSICS_PROFILE,
) -> Path:
    """Return the cached scene XML for a built-in label, exporting on first use.

    Built-in scenes are stored under a content-addressed cache directory. The
    fingerprint covers the packaged URDF, local meshes referenced by the URDF,
    the versioned convex collision assets, the full physics profile, the
    exporter cache schema, and the installed ghand/MuJoCo versions.
    """

    try:
        scene_xml_name = _SCENE_XML_NAME_BY_LABEL[label]
    except KeyError as exc:
        labels = ", ".join(MUJOCO_SCENE_LABELS)
        raise MujocoRuntimeError(
            f"Unknown MuJoCo scene label {label!r}. Available labels: {labels}."
        ) from exc

    cache_root = (
        Path(cache_dir) if cache_dir is not None else default_mujoco_scene_cache_dir()
    )
    cache_key = _mujoco_scene_cache_key(physics_profile=physics_profile)
    export_dir = cache_root / cache_key.fingerprint
    scene_path = export_dir / scene_xml_name
    if not _mujoco_scene_cache_is_current(export_dir, cache_key):
        export_mujoco_hand_model(
            export_dir,
            urdf_path=cache_key.source_urdf,
            physics_profile=physics_profile,
        )
        _write_mujoco_scene_cache_manifest(export_dir, cache_key)
    if not scene_path.is_file():
        raise MujocoRuntimeError(f"MuJoCo export did not produce {scene_path}.")
    return scene_path


def default_mujoco_scene_cache_dir() -> Path:
    """Return the root cache directory for generated MuJoCo assets.

    Respects ``GHAND_MUJOCO_CACHE_DIR`` and falls back to
    ``$XDG_CACHE_HOME/ghand/mujoco`` (``~/.cache`` by default). Built-in scene
    exports are written below this root using a content fingerprint.
    """

    override = os.environ.get("GHAND_MUJOCO_CACHE_DIR")
    if override:
        return Path(override)
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return base / "ghand" / "mujoco"


def _mujoco_scene_cache_key(
    *,
    physics_profile: MujocoPhysicsProfile = DEFAULT_MUJOCO_PHYSICS_PROFILE,
) -> _MujocoSceneCacheKey:
    source_urdf = default_right_hand_urdf_path()
    if not source_urdf.is_file():
        raise MujocoRuntimeError(f"URDF not found: {source_urdf}")

    source_urdf_sha256 = _sha256_file(source_urdf)
    mesh_sha256_by_path: dict[str, str] = {}
    for mesh_path in _local_urdf_mesh_paths(source_urdf):
        mesh_sha256_by_path[str(mesh_path)] = (
            _sha256_file(mesh_path) if mesh_path.is_file() else "<missing>"
        )
    try:
        collision_assets = load_mujoco_collision_asset_manifest()
    except MujocoCollisionAssetError as exc:
        raise MujocoRuntimeError(str(exc)) from exc
    collision_asset_sha256_by_path = {
        str(path): _sha256_file(path) for path in collision_assets.files
    }

    ghand_version = _ghand_version()
    mujoco_version = _mujoco_version()
    physics_profile_payload = physics_profile.to_dict()
    payload = {
        "cache_version": MUJOCO_EXPORT_CACHE_VERSION,
        "ghand_version": ghand_version,
        "mujoco_version": mujoco_version,
        "outputs": _MUJOCO_CACHE_OUTPUTS,
        "physics_profile": physics_profile_payload,
        "source_urdf_sha256": source_urdf_sha256,
        "mesh_sha256_by_path": mesh_sha256_by_path,
        "collision_asset_version": collision_assets.asset_version,
        "collision_asset_sha256_by_path": collision_asset_sha256_by_path,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return _MujocoSceneCacheKey(
        fingerprint=f"{MUJOCO_EXPORT_CACHE_VERSION}-{digest[:16]}",
        ghand_version=ghand_version,
        mujoco_version=mujoco_version,
        source_urdf=source_urdf,
        source_urdf_sha256=source_urdf_sha256,
        mesh_sha256_by_path=mesh_sha256_by_path,
        collision_asset_version=collision_assets.asset_version,
        collision_asset_sha256_by_path=collision_asset_sha256_by_path,
        physics_profile=physics_profile_payload,
    )


def _mujoco_scene_cache_is_current(
    export_dir: Path,
    cache_key: _MujocoSceneCacheKey,
) -> bool:
    manifest_path = export_dir / MUJOCO_CACHE_MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    expected_fields: Mapping[str, object] = {
        "cache_version": MUJOCO_EXPORT_CACHE_VERSION,
        "fingerprint": cache_key.fingerprint,
        "ghand_version": cache_key.ghand_version,
        "mujoco_version": cache_key.mujoco_version,
        "source_urdf": str(cache_key.source_urdf),
        "source_urdf_sha256": cache_key.source_urdf_sha256,
        "mesh_sha256_by_path": cache_key.mesh_sha256_by_path,
        "collision_asset_version": cache_key.collision_asset_version,
        "collision_asset_sha256_by_path": cache_key.collision_asset_sha256_by_path,
        "outputs": _MUJOCO_CACHE_OUTPUTS,
        "physics_profile": cache_key.physics_profile,
    }
    for key, expected in expected_fields.items():
        if manifest.get(key) != expected:
            return False
    return _mujoco_cache_outputs_exist(export_dir)


def _write_mujoco_scene_cache_manifest(
    export_dir: Path,
    cache_key: _MujocoSceneCacheKey,
) -> None:
    missing_outputs = [
        name
        for name in _MUJOCO_CACHE_OUTPUTS.values()
        if not (export_dir / name).is_file()
    ]
    if missing_outputs:
        missing = ", ".join(str(export_dir / name) for name in missing_outputs)
        raise MujocoRuntimeError(f"MuJoCo export did not produce required files: {missing}")

    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cache_version": MUJOCO_EXPORT_CACHE_VERSION,
        "fingerprint": cache_key.fingerprint,
        "generated_at": datetime.now(UTC).isoformat(),
        "ghand_version": cache_key.ghand_version,
        "mujoco_version": cache_key.mujoco_version,
        "source_urdf": str(cache_key.source_urdf),
        "source_urdf_sha256": cache_key.source_urdf_sha256,
        "mesh_sha256_by_path": cache_key.mesh_sha256_by_path,
        "collision_asset_version": cache_key.collision_asset_version,
        "collision_asset_sha256_by_path": cache_key.collision_asset_sha256_by_path,
        "outputs": _MUJOCO_CACHE_OUTPUTS,
        "physics_profile": cache_key.physics_profile,
    }
    (export_dir / MUJOCO_CACHE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mujoco_cache_outputs_exist(export_dir: Path) -> bool:
    return all((export_dir / name).is_file() for name in _MUJOCO_CACHE_OUTPUTS.values())


def _local_urdf_mesh_paths(urdf_path: Path) -> tuple[Path, ...]:
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise MujocoRuntimeError(
            f"Failed to parse URDF for MuJoCo cache: {urdf_path}: {exc}"
        ) from exc

    mesh_paths: set[Path] = set()
    for mesh_element in root.iter("mesh"):
        filename = mesh_element.get("filename")
        if filename is None:
            continue
        mesh_path = _local_urdf_mesh_path(urdf_path, filename)
        if mesh_path is not None:
            mesh_paths.add(mesh_path)
    return tuple(sorted(mesh_paths, key=lambda path: str(path)))


def _local_urdf_mesh_path(urdf_path: Path, filename: str) -> Path | None:
    parsed = urlparse(filename)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    return (urdf_path.parent / unquote(parsed.path)).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ghand_version() -> str:
    try:
        return importlib.metadata.version("ghand")
    except importlib.metadata.PackageNotFoundError:
        return "unversioned"


def _mujoco_version() -> str:
    try:
        return importlib.metadata.version("mujoco")
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled"


def _import_mujoco() -> Any:
    try:
        return importlib.import_module("mujoco")
    except ImportError as exc:
        raise MujocoRuntimeError(
            "MuJoCo is required for the simulation backend. Install the optional "
            "dependencies or run with `uv run --extra sim ghand teleop --mujoco ...`."
        ) from exc


def _steps_for_dt(model: Any, step_dt: float) -> int:
    timestep = float(getattr(model.opt, "timestep", 0.0))
    if timestep <= 0:
        return 1
    return max(1, int(round(step_dt / timestep)))
