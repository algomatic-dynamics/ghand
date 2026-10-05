"""Rerun viewer integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ghand.control import OperatorRuntimeSnapshot
from ghand.hand import (
    DISPLAY_HAND_ROTATION_RPY,
    JointTarget,
    UrdfPoseModel,
)
from ghand.retargeting import (
    THUMB_POSE_FEATURE_NAMES,
    HandFeatures,
)
from ghand.teleop.hand_pose import HAND_CONNECTIONS, HandFrame
from ghand.viewer.urdf_meshes import UrdfMeshConversionError, prepare_rerun_urdf

DEFAULT_RERUN_APPLICATION_ID = "ghand"
DEFAULT_RERUN_PREFIX = "ghand"
TIMESTAMP_NS_TIMELINE = "timestamp_ns"
TIME_SECONDS_TIMELINE = "time_seconds"
DEFAULT_URDF_ENTITY_PATH = "robot_right_hand"
DEFAULT_URDF_TRANSFORM_ENTITY_PATH = "joint_transforms"
DEFAULT_URDF_FRAME_PREFIX = "ghand/right_hand/"
HAND_LANDMARK_COLOR: tuple[int, int, int, int] = (0, 255, 0, 255)
HAND_SKELETON_COLOR: tuple[int, int, int, int] = (255, 180, 0, 255)
CURL_ORDER: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
SPREAD_ORDER: tuple[str, ...] = ("index", "middle", "ring", "pinky")
PINCH_ORDER: tuple[str, ...] = ("index", "middle", "ring")
THUMB_POSE_ORDER: tuple[str, ...] = THUMB_POSE_FEATURE_NAMES
FEATURE_CHART_COLOR: tuple[int, int, int, int] = (70, 180, 120, 255)
TARGET_CHART_COLOR: tuple[int, int, int, int] = (90, 140, 255, 255)
MEASURED_CHART_COLOR: tuple[int, int, int, int] = (255, 180, 80, 255)
ERROR_CHART_COLOR: tuple[int, int, int, int] = (240, 90, 90, 255)
ROBOT_HAND_INITIAL_EYE_POSITION: tuple[float, float, float] = (0.20, -0.34, 0.16)
ROBOT_HAND_INITIAL_LOOK_TARGET: tuple[float, float, float] = (0.055, -0.005, 0.0)
ROBOT_HAND_INITIAL_EYE_UP: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class RerunViewer:
    """Small Rerun logging surface for teleoperation and dataset debugging."""

    application_id: str = DEFAULT_RERUN_APPLICATION_ID
    prefix: str = DEFAULT_RERUN_PREFIX
    spawn: bool = False
    save_path: Path | str | None = None
    urdf_path: Path | str | None = None
    urdf_entity_path: str = DEFAULT_URDF_ENTITY_PATH
    urdf_transform_entity_path: str = DEFAULT_URDF_TRANSFORM_ENTITY_PATH
    urdf_frame_prefix: str = DEFAULT_URDF_FRAME_PREFIX
    urdf_joint_angle_scale: float = 1.0
    enabled: bool = True
    rerun: Any | None = field(default=None, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    _urdf_logged: bool = field(default=False, init=False, repr=False)
    _urdf_tree: Any | None = field(default=None, init=False, repr=False)
    _mapping_label_key: tuple[tuple[int, str], ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _logged_hand_overlay_count: int = field(default=0, init=False, repr=False)
    _urdf_pose_model: UrdfPoseModel | None = field(default=None, init=False, repr=False)

    def start(self) -> RerunViewer:
        """Initialize Rerun logging once."""

        if not self.enabled or self._started:
            return self

        rr = self._rr()
        rr.init(self.application_id, spawn=self.spawn)
        if self.save_path is not None:
            rr.save(Path(self.save_path))
        self._started = True
        self._send_blueprint()
        self._log_urdf_static()
        return self

    def log_camera(self, frame_bgr: Any, *, timestamp_ns: int | None = None) -> None:
        """Log an OpenCV-style BGR camera image.

        Hand landmarks are logged separately by :meth:`log_hand_overlay` so the
        viewer can keep them in sync with the image without baking them in.
        """

        if not self.enabled:
            return
        self._set_time(timestamp_ns)
        self._rr().log(
            self._path("camera", "image"), self._rr().Image(_bgr_to_rgb(frame_bgr))
        )

    def log_observation_images(
        self,
        camera_frames: Mapping[str, Any],
        *,
        timestamp_ns: int | None = None,
        image_color_format: str = "bgr",
    ) -> None:
        """Log dataset/policy observation images by camera name."""

        if not self.enabled or not camera_frames:
            return
        if image_color_format not in {"bgr", "rgb"}:
            raise ValueError("image_color_format must be 'bgr' or 'rgb'.")
        self._set_time(timestamp_ns)
        rr = self._rr()
        for raw_name, frame in camera_frames.items():
            camera_name = str(raw_name).removeprefix("observation.images.")
            image = _bgr_to_rgb(frame) if image_color_format == "bgr" else np.asarray(frame)
            rr.log(
                self._path("observation", "images", camera_name),
                rr.Image(np.ascontiguousarray(image)),
            )


    def log_hand_frame(self, hand_frame: HandFrame) -> None:
        """Log the 2D overlays plus the per-hand count, score, and handedness."""

        if not self.enabled:
            return
        self.log_hand_overlay(hand_frame)
        rr = self._rr()
        rr.log(self._path("hand", "count"), rr.Scalars(float(len(hand_frame.hands))))
        for index, hand in enumerate(hand_frame.hands):
            hand_path = ("hand", str(index))
            rr.log(self._path(*hand_path, "score"), rr.Scalars(float(hand.score)))
            rr.log(
                self._path(*hand_path, "handedness"),
                rr.TextLog(hand.handedness, level="INFO"),
            )

    def log_hand_overlay(self, hand_frame: HandFrame) -> None:
        """Log the 2D landmark and skeleton overlays for the camera image.

        Call this for every camera frame so the overlay never lags behind the
        image; hands that disappear are cleared immediately.
        """

        if not self.enabled:
            return
        self._set_time(hand_frame.timestamp_ns)
        rr = self._rr()
        hand_count = len(hand_frame.hands)
        self._clear_stale_hand_overlays(rr, hand_count)
        for index, hand in enumerate(hand_frame.hands):
            points = [
                (
                    float(landmark.x * hand_frame.image_width),
                    float(landmark.y * hand_frame.image_height),
                )
                for landmark in hand.landmarks
            ]
            labels = [str(point_index) for point_index in range(len(points))]
            strips = [
                [points[start], points[end]]
                for start, end in HAND_CONNECTIONS
                if start < len(points) and end < len(points)
            ]
            camera_overlay_path = ("camera", "overlay", "hand", str(index))
            rr.log(
                self._path(*camera_overlay_path, "landmarks2d"),
                rr.Points2D(
                    points,
                    radii=4.0,
                    labels=labels,
                    colors=[HAND_LANDMARK_COLOR] * len(points),
                    draw_order=10.0,
                ),
            )
            rr.log(
                self._path(*camera_overlay_path, "skeleton2d"),
                rr.LineStrips2D(
                    strips,
                    radii=2.0,
                    colors=[HAND_SKELETON_COLOR] * len(strips),
                    draw_order=9.0,
                ),
            )
        self._logged_hand_overlay_count = hand_count

    def _clear_stale_hand_overlays(self, rr: Any, hand_count: int) -> None:
        previous_count = self._logged_hand_overlay_count
        if previous_count <= hand_count:
            return

        if hand_count == 0:
            rr.log(self._path("camera", "overlay", "hand"), rr.Clear(recursive=True))
            rr.log(self._path("hand"), rr.Clear(recursive=True))
            return

        for index in range(hand_count, previous_count):
            rr.log(
                self._path("camera", "overlay", "hand", str(index)),
                rr.Clear(recursive=True),
            )
            rr.log(self._path("hand", str(index)), rr.Clear(recursive=True))

    def log_hand_features(self, features: HandFeatures) -> None:
        """Log retargeting input features for mapping calibration."""

        if not self.enabled:
            return
        self._set_time(features.timestamp_ns)
        self._log_scalar(("features", "valid"), float(features.valid))
        self._log_scalar(("features", "score"), features.score)
        self._log_scalar(("features", "palm_size"), features.palm_size)
        self._log_scalar(("features", "thumb_abduction"), features.thumb_abduction)
        for feature_name, value in features.thumb_pose.items():
            self._log_scalar(("features", "thumb_pose", feature_name), value)
        self._rr().log(
            self._path("features", "handedness"),
            self._rr().TextLog(features.handedness, level="INFO"),
        )
        for finger, value in features.finger_curl.items():
            self._log_scalar(("features", "finger_curl", finger), value)
        for finger, value in features.finger_spread.items():
            self._log_scalar(("features", "finger_spread", finger), value)
        for finger, value in features.pinch.items():
            self._log_scalar(("features", "pinch", finger), value)
        rr = self._rr()
        rr.log(
            self._path("mapping", "features", "curl"),
            rr.BarChart(
                [float(features.finger_curl[finger]) for finger in CURL_ORDER],
                color=FEATURE_CHART_COLOR,
            ),
        )
        rr.log(
            self._path("mapping", "features", "spread"),
            rr.BarChart(
                [float(features.finger_spread[finger]) for finger in SPREAD_ORDER],
                color=FEATURE_CHART_COLOR,
            ),
        )
        rr.log(
            self._path("mapping", "features", "pinch"),
            rr.BarChart(
                [float(features.pinch[finger]) for finger in PINCH_ORDER],
                color=FEATURE_CHART_COLOR,
            ),
        )
        rr.log(
            self._path("mapping", "features", "thumb_pose"),
            rr.BarChart(
                [float(features.thumb_pose[feature]) for feature in THUMB_POSE_ORDER],
                color=FEATURE_CHART_COLOR,
            ),
        )

    def log_calibration_pose_instruction(
        self,
        *,
        pose_name: str,
        title: str,
        instruction: str,
        pose_index: int,
        pose_count: int,
        valid_samples: int = 0,
        timestamp_ns: int | None = None,
        status: str | None = None,
    ) -> None:
        """Log the active hand-retarget calibration pose guide."""

        if not self.enabled:
            return
        self._set_time(timestamp_ns)
        lines = [
            f"## {pose_index}/{pose_count} {title}",
            "",
            instruction,
            "",
            f"- pose: `{pose_name}`",
            f"- valid samples: {valid_samples}",
        ]
        if status:
            lines.extend(("", status))
        markdown = "\n".join(lines)
        self._rr().log(
            self._path("calibration", "pose_guide"),
            self._rr().TextDocument(markdown, media_type="text/markdown"),
        )

    def log_joint_targets(self, target: JointTarget) -> None:
        """Log desired normalized retargeting output before radian conversion."""

        if not self.enabled:
            return
        self._set_time(target.timestamp_ns)
        for joint_name, value in target.joints.items():
            self._log_scalar(
                ("retargeting", "normalized_target", *_joint_path_parts(joint_name)),
                value,
            )

    def log_runtime_snapshot(
        self,
        snapshot: OperatorRuntimeSnapshot,
        *,
        timestamp_ns: int | None = None,
    ) -> None:
        """Log radian target, measured, commanded, and telemetry runtime values."""

        if not self.enabled:
            return
        self._set_time(timestamp_ns)
        self._log_scalar(("runtime", "connected"), float(snapshot.connected))
        self._log_scalar(("runtime", "dry_run"), float(snapshot.dry_run))
        self._log_scalar(("runtime", "emergency_stop"), float(snapshot.emergency_stop))
        for joint in snapshot.joints:
            joint_path = _joint_path_parts(joint.joint_name)
            motor_path = ("motors", f"id{joint.dxl_id}")
            self._log_scalar(("joints", "target", *joint_path), joint.target)
            self._log_scalar(("joints", "commanded", *joint_path), joint.commanded)
            self._log_scalar(("joints", "measured", *joint_path), joint.measured)
            self._log_scalar((*motor_path, "target_tick"), float(joint.target_tick))
            self._log_scalar((*motor_path, "position_tick"), float(joint.measured_tick))
            self._log_scalar(
                (*motor_path, "torque_enabled"), float(joint.torque_enabled)
            )
            if joint.current_limit_ma is not None:
                self._log_scalar(
                    (*motor_path, "current_limit_ma"),
                    joint.current_limit_ma,
                )
            if joint.present_current_ma is not None:
                self._log_scalar(
                    (*motor_path, "present_current_ma"),
                    joint.present_current_ma,
                )
            if joint.temperature_celsius is not None:
                self._log_scalar(
                    (*motor_path, "temperature_celsius"),
                    float(joint.temperature_celsius),
                )
            if joint.input_voltage is not None:
                self._log_scalar((*motor_path, "input_voltage"), joint.input_voltage)
            if joint.hardware_error_status is not None:
                self._log_scalar(
                    (*motor_path, "hardware_error_status"),
                    float(joint.hardware_error_status),
                )
        self._log_mapping_joint_bars(snapshot)
        self._log_urdf_pose(
            _snapshot_joint_angles(
                snapshot,
                self.urdf_joint_angle_scale,
                self._pose_model(),
            )
        )
        if snapshot.last_error:
            self.log_text(snapshot.last_error, level="ERROR", timestamp_ns=timestamp_ns)

    def log_latency(
        self,
        name: str,
        seconds: float,
        *,
        timestamp_ns: int | None = None,
    ) -> None:
        """Log a latency scalar in seconds."""

        if not self.enabled:
            return
        self._set_time(timestamp_ns)
        self._log_scalar(("latency", *_latency_path_parts(name)), seconds)

    def log_text(
        self,
        message: str,
        *,
        level: str = "INFO",
        timestamp_ns: int | None = None,
    ) -> None:
        """Log a text message into the debug timeline."""

        if not self.enabled:
            return
        self._set_time(timestamp_ns)
        self._rr().log(self._path("logs"), self._rr().TextLog(message, level=level))

    def _set_time(self, timestamp_ns: int | None) -> None:
        self.start()
        if not self.enabled or timestamp_ns is None:
            return
        rr = self._rr()
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence(TIMESTAMP_NS_TIMELINE, int(timestamp_ns))
            rr.set_time_seconds(
                TIME_SECONDS_TIMELINE, int(timestamp_ns) / 1_000_000_000
            )
        else:
            rr.set_time(TIMESTAMP_NS_TIMELINE, sequence=int(timestamp_ns))
            rr.set_time(
                TIME_SECONDS_TIMELINE, duration=int(timestamp_ns) / 1_000_000_000
            )

    def _log_scalar(self, path_parts: tuple[str, ...], value: float) -> None:
        self.start()
        if not self.enabled:
            return
        self._rr().log(self._path(*path_parts), self._rr().Scalars(float(value)))

    def _log_urdf_static(self) -> None:
        if self.urdf_path is None or self._urdf_logged:
            return

        rr = self._rr()
        urdf_module = getattr(rr, "urdf", None)
        urdf_tree_type = getattr(urdf_module, "UrdfTree", None)
        if urdf_tree_type is None:
            self._urdf_logged = True
            rr.log(
                self._path("logs"),
                rr.TextLog(
                    "URDF logging requires rerun-sdk >= 0.32 with rerun.urdf support.",
                    level="WARN",
                ),
            )
            return

        source_urdf_path = Path(self.urdf_path)
        prepared_urdf_path: Path | None = None
        try:
            urdf_path = prepare_rerun_urdf(
                source_urdf_path,
                root_rotation_rpy=DISPLAY_HAND_ROTATION_RPY,
            )
            prepared_urdf_path = urdf_path
            self._urdf_tree = urdf_tree_type.from_file_path(
                urdf_path,
                entity_path_prefix=self._urdf_entity_prefix(),
                frame_prefix=self.urdf_frame_prefix,
                static_transform_entity_path=_join_entity_path(
                    self._urdf_entity_prefix(),
                    "tf_static",
                ),
            )
            self._urdf_tree.log_urdf_to_recording()
            self._urdf_logged = True
        except UrdfMeshConversionError as exc:
            self._urdf_logged = True
            rr.log(
                self._path("logs"),
                rr.TextLog(str(exc), level="WARN"),
            )
        except Exception as exc:
            self._urdf_logged = True
            rr.log(
                self._path("logs"),
                rr.TextLog(
                    "URDF mesh logging failed: "
                    f"{prepared_urdf_path or source_urdf_path}: {exc}",
                    level="WARN",
                ),
            )

    def _log_urdf_pose(self, joint_angles: dict[str, float]) -> None:
        if not self._urdf_logged:
            self.start()
        if not self.enabled or self.urdf_path is None:
            return

        rr = self._rr()
        if self._urdf_tree is None:
            return

        transform_path = _join_entity_path(
            self._urdf_entity_prefix(),
            _clean_path_part(self.urdf_transform_entity_path),
        )
        for joint in self._urdf_tree.joints():
            if joint.name not in joint_angles or joint.joint_type == "fixed":
                continue
            rr.log(
                transform_path,
                joint.compute_transform(joint_angles[joint.name], clamp=True),
            )

    def _send_blueprint(self) -> None:
        rr = self._rr()
        if not hasattr(rr, "send_blueprint"):
            return

        try:
            import rerun.blueprint as rrb

            camera_view = rrb.Spatial2DView(
                name="Teleop Camera",
                origin=f"/{self._path('camera')}",
                contents="$origin/**",
            )
            observation_view = rrb.Spatial2DView(
                name="Observation Images",
                origin=f"/{self._path('observation', 'images')}",
                contents="$origin/**",
            )
            calibration_view = rrb.TextDocumentView(
                name="Calibration Pose",
                origin=f"/{self._path('calibration', 'pose_guide')}",
            )
            mapping_view = rrb.Vertical(
                rrb.Grid(
                    rrb.BarChartView(
                        name="Curl",
                        origin=f"/{self._path('mapping', 'features', 'curl')}",
                    ),
                    rrb.BarChartView(
                        name="Spread",
                        origin=f"/{self._path('mapping', 'features', 'spread')}",
                    ),
                    rrb.BarChartView(
                        name="Pinch",
                        origin=f"/{self._path('mapping', 'features', 'pinch')}",
                    ),
                    rrb.BarChartView(
                        name="Thumb",
                        origin=f"/{self._path('mapping', 'features', 'thumb_pose')}",
                    ),
                    grid_columns=4,
                ),
                rrb.Grid(
                    rrb.BarChartView(
                        name="Target",
                        origin=f"/{self._path('mapping', 'joints', 'target')}",
                    ),
                    rrb.BarChartView(
                        name="Measured",
                        origin=f"/{self._path('mapping', 'joints', 'measured')}",
                    ),
                    rrb.BarChartView(
                        name="Error",
                        origin=f"/{self._path('mapping', 'joints', 'error')}",
                    ),
                    grid_columns=3,
                ),
                rrb.TextDocumentView(
                    name="Mapping Labels",
                    origin=f"/{self._path('mapping', 'labels')}",
                ),
                row_shares=[0.35, 0.4, 0.25],
                name="Mapping",
            )
            if self.urdf_path is None:
                blueprint = rrb.Blueprint(
                    rrb.Vertical(
                        camera_view,
                        observation_view,
                        calibration_view,
                        mapping_view,
                        row_shares=[0.35, 0.35, 0.10, 0.20],
                    ),
                    auto_views=False,
                )
            else:
                urdf_prefix = self._urdf_entity_prefix()
                blueprint = rrb.Blueprint(
                    rrb.Horizontal(
                        rrb.Vertical(
                            camera_view,
                            observation_view,
                            calibration_view,
                            mapping_view,
                            row_shares=[0.35, 0.35, 0.10, 0.20],
                        ),
                        rrb.Spatial3DView(
                            name="Robot Hand",
                            origin="/",
                            contents=f"/{urdf_prefix}/**",
                            eye_controls=_robot_hand_eye_controls(rrb),
                        ),
                        column_shares=[0.55, 0.45],
                    ),
                    auto_views=False,
                )
            rr.send_blueprint(blueprint)
        except Exception:
            return

    def _path(self, *parts: str) -> str:
        return "/".join(
            (self.prefix.strip("/"), *(_clean_path_part(part) for part in parts))
        )

    def _relative_path(self, path: str) -> str:
        return self._path(*_name_path_parts(path))

    def _urdf_entity_prefix(self) -> str:
        return _clean_path_part(f"{self.prefix}_{self.urdf_entity_path}")

    def _log_mapping_joint_bars(self, snapshot: OperatorRuntimeSnapshot) -> None:
        joints = snapshot.joints
        if not joints:
            return

        self._log_mapping_labels(joints)
        rr = self._rr()
        targets = [float(joint.target) for joint in joints]
        measured = [float(joint.measured) for joint in joints]
        errors = [
            target - actual for target, actual in zip(targets, measured, strict=True)
        ]
        rr.log(
            self._path("mapping", "joints", "target"),
            rr.BarChart(targets, color=TARGET_CHART_COLOR),
        )
        rr.log(
            self._path("mapping", "joints", "measured"),
            rr.BarChart(measured, color=MEASURED_CHART_COLOR),
        )
        rr.log(
            self._path("mapping", "joints", "error"),
            rr.BarChart(errors, color=ERROR_CHART_COLOR),
        )

    def _log_mapping_labels(self, joints: tuple[Any, ...]) -> None:
        label_key = tuple(
            (int(joint.dxl_id), str(joint.joint_name)) for joint in joints
        )
        if self._mapping_label_key == label_key:
            return

        self._mapping_label_key = label_key
        self._rr().log(
            self._path("mapping", "labels"),
            self._rr().TextDocument(
                _mapping_labels_markdown(joints), media_type="text/markdown"
            ),
            static=True,
        )

    def _rr(self) -> Any:
        if self.rerun is None:
            import rerun as rr

            self.rerun = rr
        return self.rerun

    def _pose_model(self) -> UrdfPoseModel:
        if self._urdf_pose_model is None:
            self._urdf_pose_model = (
                UrdfPoseModel.from_urdf_file(self.urdf_path)
                if self.urdf_path is not None
                else UrdfPoseModel.empty()
            )
        return self._urdf_pose_model


def _bgr_to_rgb(frame_bgr: Any) -> np.ndarray:
    frame = np.asarray(frame_bgr)
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        return frame[..., :3][..., ::-1]
    return frame


def _snapshot_joint_angles(
    snapshot: OperatorRuntimeSnapshot,
    joint_angle_scale: float,
    pose_model: UrdfPoseModel,
) -> dict[str, float]:
    _ = pose_model
    return {
        f"id{joint.pose_dxl_id}": joint.measured * joint_angle_scale
        for joint in snapshot.joints
    }


def _robot_hand_eye_controls(rrb: Any) -> Any:
    return rrb.EyeControls3D(
        kind="orbital",
        position=ROBOT_HAND_INITIAL_EYE_POSITION,
        look_target=ROBOT_HAND_INITIAL_LOOK_TARGET,
        eye_up=ROBOT_HAND_INITIAL_EYE_UP,
    )


def _mapping_labels_markdown(joints: tuple[Any, ...]) -> str:
    feature_lines = (
        f"- curl: {_indexed_labels(CURL_ORDER)}",
        f"- spread: {_indexed_labels(SPREAD_ORDER)}",
        f"- pinch: {_indexed_labels(PINCH_ORDER)}",
        f"- thumb: {_indexed_labels(THUMB_POSE_ORDER)}",
        "- note: human pinky features are logged for debugging; robot targets use "
        "URDF fingers ring/middle/index/thumb.",
    )
    joint_lines = [
        f"| {index} | {joint.dxl_id} | {joint.pose_dxl_id} | "
        f"`{_short_joint_name(joint.joint_name)}` |"
        for index, joint in enumerate(joints)
    ]
    return "\n".join(
        (
            "### Feature Bars",
            *feature_lines,
            "",
            "### Joint Bars",
            "| index | dxl id | urdf id | joint |",
            "|---:|---:|---:|---|",
            *joint_lines,
        )
    )


def _indexed_labels(labels: tuple[str, ...]) -> str:
    return ", ".join(f"{index}={label}" for index, label in enumerate(labels))


def _short_joint_name(joint_name: str) -> str:
    parts = joint_name.split(".")
    if len(parts) >= 3:
        return parts[-2]
    return joint_name


def _join_entity_path(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def _joint_path_parts(joint_name: str) -> tuple[str, ...]:
    return tuple(_clean_path_part(part) for part in joint_name.split("."))


def _name_path_parts(name: str) -> tuple[str, ...]:
    return tuple(_clean_path_part(part) for part in name.split("/") if part)


def _latency_path_parts(name: str) -> tuple[str, ...]:
    parts = list(_name_path_parts(name))
    if not parts:
        return ("seconds",)
    parts[-1] = f"{parts[-1]}_seconds"
    return tuple(parts)


def _clean_path_part(part: str) -> str:
    return str(part).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
