"""LeRobot Robot runtime adapter for LEAP V1 KAIZEN."""

from __future__ import annotations

import math
import time
from typing import cast

import numpy as np
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError

from ghand.control import (
    HandOperatorRuntime,
    OPERATOR_CONTROL_MODE_CURRENT_POSITION,
    OPERATOR_CONTROL_MODE_POSITION,
    OperatorControlMode,
    OperatorRuntimeSnapshot,
    apply_position_current_action,
    apply_radian_joint_target,
    make_fake_hand_runtime,
)
from ghand.hand import (
    CurrentLimitTarget,
    GHAND_RIGHT_MOTOR_IDS,
    JOINT_CONTACT_KEY,
    JOINT_CURRENT_LIMIT_KEY,
    JOINT_CURRENT_RESIDUAL_KEY,
    JOINT_POSITION_KEY,
    JOINT_PRESENT_CURRENT_KEY,
    make_default_joint_calibrations,
    PositionCurrentAction,
    RadianJointTarget,
)
from ghand.data.lerobot_schema import (
    make_hand_joint_position_names,
    make_hand_position_current_action_names,
    make_hand_position_current_state_names,
)
from ghand.hardware import HandController
from ghand.lerobot_plugin.config_ghand import GHandRobotConfig
from ghand.teleop.webcam import Webcam


class GHandRobot(Robot):
    """LeRobot-compatible LEAP V1 KAIZEN robot facade backed by ``HandOperatorRuntime``."""

    config_class = GHandRobotConfig
    name = "ghand"

    def __init__(self, config: GHandRobotConfig) -> None:
        super().__init__(config)
        self.config = config
        self._runtime: HandOperatorRuntime | None = None
        self._observation_webcam: Webcam | None = None
        self._joint_names = make_hand_joint_position_names((config.side,))
        self._state_names = self._make_state_names()
        self._action_names = self._make_action_names()

    @property
    def observation_features(self) -> dict[str, object]:
        features: dict[str, object] = {name: float for name in self._state_names}
        for camera_name in self._camera_names:
            features[camera_name] = (
                self.config.camera_height,
                self.config.camera_width,
                3,
            )
        return features

    @property
    def action_features(self) -> dict[str, type[float]]:
        return {name: float for name in self._action_names}

    @property
    def cameras(self) -> dict[str, object]:
        """Return camera placeholders for LeRobot recorder image writer sizing."""

        return {name: object() for name in self._camera_names}

    @property
    def is_connected(self) -> bool:
        return self._runtime is not None and self._runtime.connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def runtime(self) -> HandOperatorRuntime | None:
        """Return the underlying runtime for diagnostics and tests."""

        return self._runtime

    def connect(self, calibrate: bool = True) -> None:
        if self._runtime is None:
            self._runtime = self._make_runtime()
        if not self._runtime.connected:
            self._runtime.connect()
        if (
            self.config.observation_camera is not None
            and self._observation_webcam is None
        ):
            webcam = Webcam(
                camera=self.config.observation_camera,
                latest=True,
                width=self.config.camera_width,
                height=self.config.camera_height,
            )
            webcam.open()
            self._observation_webcam = webcam
        self._runtime.set_torque_enabled(True)

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> RobotObservation:
        """Return the current observation without advancing the backend.

        Physics stepping happens only in :meth:`send_action`; otherwise one
        ``lerobot-record`` loop iteration (observe + act) would advance MuJoCo
        twice per frame and diverge from the recorded training dynamics.
        """

        runtime = self._require_runtime()
        snapshot = runtime.observe()
        observation = self._observation_from_snapshot(snapshot)
        for camera_name in self._camera_names:
            observation[camera_name] = self._capture_camera(runtime, camera_name)
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        runtime = self._require_runtime()
        self._validate_action_keys(action)
        timestamp_ns = time.time_ns()
        if self.config.dataset_schema == "position-current":
            structured_action = self._position_current_action(
                action, timestamp_ns=timestamp_ns
            )
            snapshot = apply_position_current_action(runtime, structured_action)
            return self._sent_position_current_action(snapshot)

        target = RadianJointTarget(
            timestamp_ns=timestamp_ns,
            joints=self._position_targets(action),
        )
        snapshot = apply_radian_joint_target(runtime, target)
        return self._sent_position_action(snapshot)

    def reset(self) -> None:
        """Reset runtime motion state and MuJoCo scene state when available."""

        runtime = self._require_runtime()
        runtime.reset_motion_state()
        reset_scene = getattr(runtime.controller.bus, "reset_scene", None)
        if callable(reset_scene) and self.config.mujoco_auto_reset:
            reset_scene()

    def disconnect(self) -> None:
        if self._observation_webcam is not None:
            self._observation_webcam.close()
            self._observation_webcam = None
        if self._runtime is None:
            return
        if self._runtime.connected:
            self._runtime.disconnect()

    @property
    def _camera_names(self) -> tuple[str, ...]:
        if self.config.observation_camera is not None:
            return (self.config.observation_camera_name.strip(),)
        raw_names = self.config.mujoco_camera_names
        if raw_names is None or raw_names.strip() == "":
            return ()
        return tuple(name.strip() for name in raw_names.split(",") if name.strip())

    def _make_state_names(self) -> tuple[str, ...]:
        if self.config.dataset_schema == "position-current":
            return make_hand_position_current_state_names(
                (self.config.side,),
                include_current_diagnostics=self.config.include_current_diagnostics,
            )
        return self._joint_names

    def _make_action_names(self) -> tuple[str, ...]:
        if self.config.dataset_schema == "position-current":
            return make_hand_position_current_action_names((self.config.side,))
        return self._joint_names

    def _make_runtime(self) -> HandOperatorRuntime:
        runtime_control_mode = self._runtime_control_mode()
        target_fps = self.config.target_fps
        if target_fps is None:  # Resolved by GHandRobotConfig.__post_init__.
            raise ValueError("target_fps was not resolved.")
        target_velocity_limit = self.config.target_velocity_limit
        if target_velocity_limit is None:  # Resolved by GHandRobotConfig.__post_init__.
            raise ValueError("target_velocity_limit was not resolved.")
        if self.config.backend == "fake":
            return make_fake_hand_runtime(
                side=self.config.side,
                dry_run=self.config.dry_run,
                target_hz=target_fps,
                target_velocity_limit=target_velocity_limit,
                control_mode=runtime_control_mode,
                default_current_limit_ma=self.config.default_current_limit_ma,
                max_current_limit_ma=self.config.max_current_limit_ma,
            )
        if self.config.backend == "mujoco":
            from ghand.sim.mujoco import make_mujoco_hand_runtime

            return make_mujoco_hand_runtime(
                scene=self.config.mujoco_scene,
                side=self.config.side,
                dry_run=self.config.dry_run,
                target_hz=target_fps,
                target_velocity_limit=target_velocity_limit,
                control_mode=runtime_control_mode,
                default_current_limit_ma=self.config.default_current_limit_ma,
                max_current_limit_ma=self.config.max_current_limit_ma,
                actuator_kp=self.config.mujoco_actuator_kp,
                actuator_force_limit=self.config.mujoco_actuator_force_limit,
                viewer=self.config.mujoco_viewer,
            )
        if self.config.port is None:
            raise ValueError("port is required when backend='real'.")
        return HandOperatorRuntime(
            HandController.from_port(
                port=self.config.port, baudrate=self.config.baudrate
            ),
            make_default_joint_calibrations(
                side=self.config.side,
                motor_ids=GHAND_RIGHT_MOTOR_IDS,
            ),
            backend=f"dynamixel:{self.config.port}",
            dry_run=self.config.dry_run,
            target_hz=target_fps,
            target_velocity_limit=target_velocity_limit,
            control_mode=runtime_control_mode,
            default_current_limit_ma=self.config.default_current_limit_ma,
            max_current_limit_ma=self.config.max_current_limit_ma,
        )

    def _runtime_control_mode(self) -> OperatorControlMode:
        if self.config.control_mode == "current-position":
            return OPERATOR_CONTROL_MODE_CURRENT_POSITION
        return OPERATOR_CONTROL_MODE_POSITION

    def _require_runtime(self) -> HandOperatorRuntime:
        if self._runtime is None or not self._runtime.connected:
            raise DeviceNotConnectedError(
                "GHandRobot is not connected. Run `.connect()` first."
            )
        return self._runtime

    def _validate_action_keys(self, action: RobotAction) -> None:
        action_keys = set(action)
        expected_keys = set(self._action_names)
        missing = tuple(name for name in self._action_names if name not in action_keys)
        extras = tuple(sorted(action_keys - expected_keys))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Missing LEAP V1 KAIZEN action key(s): {missing_text}.")
        if extras:
            extras_text = ", ".join(extras)
            raise ValueError(f"Unsupported LEAP V1 KAIZEN action key(s): {extras_text}.")

    def _position_targets(self, action: RobotAction) -> dict[str, float]:
        targets: dict[str, float] = {}
        for joint_name in self._joint_names:
            targets[joint_name] = _finite_float(
                action[joint_name], f"{joint_name} action value"
            )
        return targets

    def _position_current_action(
        self,
        action: RobotAction,
        *,
        timestamp_ns: int,
    ) -> PositionCurrentAction:
        positions: dict[str, float] = {}
        current_limits: dict[str, float] = {}
        for joint_name in self._joint_names:
            positions[joint_name] = _finite_float(
                action[joint_name], f"{joint_name} action value"
            )
            current_key = _replace_joint_value_key(joint_name, JOINT_CURRENT_LIMIT_KEY)
            current_limits[joint_name] = _finite_float(
                action[current_key],
                f"{current_key} action value",
            )
        return PositionCurrentAction(
            position=RadianJointTarget(timestamp_ns=timestamp_ns, joints=positions),
            current_limit=CurrentLimitTarget(
                timestamp_ns=timestamp_ns, currents_ma=current_limits
            ),
        )

    def _observation_from_snapshot(
        self, snapshot: OperatorRuntimeSnapshot
    ) -> RobotObservation:
        joints_by_name = {state.joint_name: state for state in snapshot.joints}
        observation: RobotObservation = {}
        for joint_name in self._joint_names:
            state = joints_by_name[joint_name]
            observation[joint_name] = float(state.measured)
            if self.config.dataset_schema != "position-current":
                continue
            present_current_ma = (
                0.0
                if state.present_current_ma is None
                else float(state.present_current_ma)
            )
            observation[
                _replace_joint_value_key(joint_name, JOINT_PRESENT_CURRENT_KEY)
            ] = present_current_ma
            if self.config.include_current_diagnostics:
                residual_ma = max(
                    abs(present_current_ma) - self.config.current_contact_threshold_ma,
                    0.0,
                )
                observation[
                    _replace_joint_value_key(joint_name, JOINT_CURRENT_RESIDUAL_KEY)
                ] = residual_ma
                observation[_replace_joint_value_key(joint_name, JOINT_CONTACT_KEY)] = (
                    1.0 if residual_ma > 0.0 else 0.0
                )
        return observation

    def _sent_position_action(self, snapshot: OperatorRuntimeSnapshot) -> RobotAction:
        return {
            state.joint_name: float(state.commanded)
            for state in snapshot.joints
            if state.joint_name in self._joint_names
        }

    def _sent_position_current_action(
        self, snapshot: OperatorRuntimeSnapshot
    ) -> RobotAction:
        sent: RobotAction = {}
        for state in snapshot.joints:
            if state.joint_name not in self._joint_names:
                continue
            sent[state.joint_name] = float(state.commanded)
            current_key = _replace_joint_value_key(
                state.joint_name, JOINT_CURRENT_LIMIT_KEY
            )
            sent[current_key] = (
                0.0 if state.current_limit_ma is None else float(state.current_limit_ma)
            )
        return sent

    def _capture_camera(self, runtime: HandOperatorRuntime, camera_name: str) -> object:
        if self._observation_webcam is not None:
            return self._webcam_frame(camera_name)
        return self._render_camera(runtime, camera_name)

    def _render_camera(self, runtime: HandOperatorRuntime, camera_name: str) -> object:
        render_camera = getattr(runtime.controller.bus, "render_camera", None)
        if not callable(render_camera):
            raise RuntimeError(
                f"LEAP V1 KAIZEN backend {self.config.backend!r} does not provide camera {camera_name!r}."
            )
        return render_camera(
            camera_name,
            width=self.config.camera_width,
            height=self.config.camera_height,
        )

    def _webcam_frame(self, camera_name: str) -> np.ndarray:
        assert self._observation_webcam is not None
        image = np.asarray(self._observation_webcam.read())
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(
                f"Observation camera {camera_name!r} returned a non-BGR frame "
                f"with shape {getattr(image, 'shape', None)}."
            )
        expected = (self.config.camera_height, self.config.camera_width)
        if tuple(image.shape[:2]) != expected:
            import cv2

            image = cv2.resize(
                image,
                (self.config.camera_width, self.config.camera_height),
                interpolation=cv2.INTER_AREA,
            )
        # OpenCV captures BGR; recorded datasets and MuJoCo renders are RGB.
        return np.ascontiguousarray(image[..., ::-1])


def _finite_float(value: object, label: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be finite.")
    return resolved


def _replace_joint_value_key(name: str, value_key: str) -> str:
    prefix, separator, key = name.rpartition(".")
    if separator == "" or key != JOINT_POSITION_KEY:
        raise ValueError(
            f"Expected a joint position channel ending with '.{JOINT_POSITION_KEY}': {name}."
        )
    return f"{prefix}.{value_key}"


def ensure_ghand_robot(robot: Robot) -> GHandRobot:
    """Return a typed GHandRobot or raise a clear error."""

    if not isinstance(robot, GHandRobot):
        raise TypeError(f"Expected GHandRobot, got {type(robot).__name__}.")
    return cast(GHandRobot, robot)
