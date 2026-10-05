"""LeRobot RobotConfig registration for LEAP V1 KAIZEN."""

from __future__ import annotations

import math
from dataclasses import dataclass
from lerobot.robots.config import RobotConfig

from ghand.hardware import DEFAULT_BAUDRATE
from ghand.sim.mujoco.profile import TELEOP_MUJOCO_PHYSICS_PROFILE


GHandRobotBackend = str
GHandHandSide = str
GHandControlMode = str
GHandDatasetSchema = str
DEFAULT_LEROBOT_TARGET_FPS = 30.0
DEFAULT_LEROBOT_TARGET_VELOCITY_LIMIT = 1000.0


@RobotConfig.register_subclass("ghand")
@dataclass(kw_only=True)
class GHandRobotConfig(RobotConfig):
    """Configuration parsed by ``lerobot-record --robot.type=ghand``."""

    backend: GHandRobotBackend = "fake"
    side: GHandHandSide = "right"
    port: str | None = None
    baudrate: int = DEFAULT_BAUDRATE
    dry_run: bool = False
    target_fps: float | None = None
    target_velocity_limit: float | None = None
    control_mode: GHandControlMode = "position"
    dataset_schema: GHandDatasetSchema = "position"
    default_current_limit_ma: float | None = None
    max_current_limit_ma: float | None = None
    mujoco_scene: str | None = None
    mujoco_viewer: bool = False
    mujoco_actuator_kp: float = TELEOP_MUJOCO_PHYSICS_PROFILE.actuator_kp
    mujoco_actuator_force_limit: float = (
        TELEOP_MUJOCO_PHYSICS_PROFILE.actuator_force_limit
    )
    mujoco_auto_reset: bool = True
    camera_width: int = 640
    camera_height: int = 480
    mujoco_camera_names: str | None = None
    observation_camera: int | None = None
    observation_camera_name: str = "front"
    include_current_diagnostics: bool = False
    current_contact_threshold_ma: float = 50.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.target_fps is None:
            self.target_fps = (
                TELEOP_MUJOCO_PHYSICS_PROFILE.control_hz
                if self.backend == "mujoco"
                else DEFAULT_LEROBOT_TARGET_FPS
            )
        if self.target_velocity_limit is None:
            self.target_velocity_limit = (
                TELEOP_MUJOCO_PHYSICS_PROFILE.target_velocity_limit_rad_s
                if self.backend == "mujoco"
                else DEFAULT_LEROBOT_TARGET_VELOCITY_LIMIT
            )
        if self.backend not in {"fake", "mujoco", "real"}:
            raise ValueError("backend must be one of: fake, mujoco, real.")
        if self.side not in {"right", "left"}:
            raise ValueError("side must be 'right' or 'left'.")
        if self.control_mode not in {"position", "current-position"}:
            raise ValueError("control_mode must be 'position' or 'current-position'.")
        if self.dataset_schema not in {"position", "position-current"}:
            raise ValueError("dataset_schema must be 'position' or 'position-current'.")
        if self.backend == "real" and not self.port:
            raise ValueError("port is required when backend='real'.")
        if self.backend in {"real", "mujoco"} and self.side != "right":
            raise ValueError("Real and MuJoCo LEAP V1 KAIZEN robots currently support side='right' only.")
        if self.backend == "mujoco" and (
            self.control_mode != "position" or self.dataset_schema != "position"
        ):
            raise ValueError("MuJoCo LEAP V1 KAIZEN robot currently supports position-only control only.")
        if self.dataset_schema == "position-current" and self.control_mode != "current-position":
            raise ValueError("dataset_schema='position-current' requires control_mode='current-position'.")
        if self.include_current_diagnostics and self.dataset_schema != "position-current":
            raise ValueError("include_current_diagnostics requires dataset_schema='position-current'.")
        if self.control_mode == "current-position" and self.default_current_limit_ma is None:
            raise ValueError("default_current_limit_ma is required when control_mode='current-position'.")
        if self.baudrate <= 0:
            raise ValueError("baudrate must be positive.")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive.")
        if self.target_velocity_limit <= 0:
            raise ValueError("target_velocity_limit must be positive.")
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("camera_width and camera_height must be positive.")
        if self.observation_camera is not None and self.observation_camera < 0:
            raise ValueError("observation_camera must be non-negative.")
        if self.observation_camera is not None and self.backend == "mujoco":
            raise ValueError(
                "observation_camera is for fake/real backends; "
                "use mujoco_camera_names with backend='mujoco'."
            )
        if not self.observation_camera_name.strip():
            raise ValueError("observation_camera_name must not be empty.")
        if self.mujoco_actuator_kp <= 0:
            raise ValueError("mujoco_actuator_kp must be positive.")
        if self.mujoco_actuator_force_limit <= 0:
            raise ValueError("mujoco_actuator_force_limit must be positive.")
        if not math.isfinite(float(self.current_contact_threshold_ma)):
            raise ValueError("current_contact_threshold_ma must be finite.")
        if self.current_contact_threshold_ma < 0:
            raise ValueError("current_contact_threshold_ma must be non-negative.")
