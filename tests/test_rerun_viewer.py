from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from ghand.control import JointRuntimeState, OperatorRuntimeSnapshot, apply_joint_target, make_fake_hand_runtime
from ghand.hand import (
    DISPLAY_HAND_ROTATION_RPY,
    JointTarget,
    UrdfJointLimit,
    UrdfPoseModel,
)
from ghand.retargeting import HandFeatures
from ghand.teleop.hand_pose import HandDetection, HandFrame, Landmark
from ghand.viewer import RerunViewer
from ghand.viewer.rerun_viewer import (
    ROBOT_HAND_INITIAL_EYE_POSITION,
    ROBOT_HAND_INITIAL_EYE_UP,
    ROBOT_HAND_INITIAL_LOOK_TARGET,
    _robot_hand_eye_controls,
    _snapshot_joint_angles,
)


def test_rerun_viewer_start_initializes_and_saves_recording(tmp_path: Path) -> None:
    rerun = FakeRerun()
    save_path = tmp_path / "debug.rrd"
    viewer = RerunViewer(
        application_id="test-app",
        spawn=True,
        save_path=save_path,
        rerun=rerun,
    )

    viewer.start()
    viewer.start()

    assert rerun.calls.count(("init", "test-app", True)) == 1
    assert rerun.calls.count(("save", save_path)) == 1
    assert any(call[0] == "send_blueprint" for call in rerun.calls)


def test_rerun_blueprint_includes_observation_image_view(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecordingBlueprintModule:
        def Blueprint(self, *children: object, **kwargs: object) -> dict[str, object]:
            return {"kind": "Blueprint", "children": children, "kwargs": kwargs}

        def Vertical(self, *children: object, **kwargs: object) -> dict[str, object]:
            return {"kind": "Vertical", "children": children, "kwargs": kwargs}

        def Horizontal(self, *children: object, **kwargs: object) -> dict[str, object]:
            return {"kind": "Horizontal", "children": children, "kwargs": kwargs}

        def Grid(self, *children: object, **kwargs: object) -> dict[str, object]:
            return {"kind": "Grid", "children": children, "kwargs": kwargs}

        def Spatial2DView(self, **kwargs: object) -> dict[str, object]:
            return {"kind": "Spatial2DView", "kwargs": kwargs}

        def Spatial3DView(self, **kwargs: object) -> dict[str, object]:
            return {"kind": "Spatial3DView", "kwargs": kwargs}

        def TextDocumentView(self, **kwargs: object) -> dict[str, object]:
            return {"kind": "TextDocumentView", "kwargs": kwargs}

        def BarChartView(self, **kwargs: object) -> dict[str, object]:
            return {"kind": "BarChartView", "kwargs": kwargs}

        def EyeControls3D(self, **kwargs: object) -> dict[str, object]:
            return dict(kwargs)

    rerun = FakeRerun()
    blueprint_module = RecordingBlueprintModule()
    monkeypatch.setitem(sys.modules, "rerun.blueprint", blueprint_module)
    if "rerun" in sys.modules:
        monkeypatch.setattr(sys.modules["rerun"], "blueprint", blueprint_module)

    RerunViewer(rerun=rerun).start()

    blueprint = next(call[1] for call in rerun.calls if call[0] == "send_blueprint")
    blueprint_text = repr(blueprint)
    assert "Observation Images" in blueprint_text
    assert "/ghand/observation/images" in blueprint_text


def test_robot_hand_eye_controls_use_palm_side_four_finger_view() -> None:
    controls = _robot_hand_eye_controls(FakeBlueprintModule())

    assert controls == {
        "kind": "orbital",
        "position": ROBOT_HAND_INITIAL_EYE_POSITION,
        "look_target": ROBOT_HAND_INITIAL_LOOK_TARGET,
        "eye_up": ROBOT_HAND_INITIAL_EYE_UP,
    }


def test_log_camera_converts_bgr_frame_to_rgb_image() -> None:
    rerun = FakeRerun()
    viewer = RerunViewer(rerun=rerun)
    frame_bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

    viewer.log_camera(frame_bgr, timestamp_ns=123)

    assert ("set_time_sequence", "timestamp_ns", 123) in rerun.calls
    image_logs = [
        call
        for call in rerun.calls
        if call[0] == "log" and call[1] == "ghand/camera/image"
    ]
    assert len(image_logs) == 1
    image = image_logs[0][2]
    assert image.kind == "Image"
    assert image.data.tolist() == [[[3, 2, 1], [6, 5, 4]]]


def test_log_observation_images_uses_dataset_paths_and_color_format() -> None:
    rerun = FakeRerun()
    viewer = RerunViewer(rerun=rerun)
    frame_bgr = np.array([[[1, 2, 3]]], dtype=np.uint8)
    frame_rgb = np.array([[[4, 5, 6]]], dtype=np.uint8)

    viewer.log_observation_images({"front": frame_bgr}, timestamp_ns=123)
    viewer.log_observation_images(
        {"observation.images.overview": frame_rgb},
        timestamp_ns=124,
        image_color_format="rgb",
    )

    assert ("set_time_sequence", "timestamp_ns", 123) in rerun.calls
    assert ("set_time_sequence", "timestamp_ns", 124) in rerun.calls
    front = _entity_at(rerun, "ghand/observation/images/front")
    overview = _entity_at(rerun, "ghand/observation/images/overview")
    assert front.kind == "Image"
    assert front.data.tolist() == [[[3, 2, 1]]]
    assert overview.kind == "Image"
    assert overview.data.tolist() == [[[4, 5, 6]]]


def test_log_hand_frame_writes_landmarks_skeleton_and_counts() -> None:
    rerun = FakeRerun()
    viewer = RerunViewer(rerun=rerun)
    frame = HandFrame(
        timestamp_ns=123,
        image_width=100,
        image_height=200,
        hands=(
            HandDetection(
                handedness="Right",
                score=0.9,
                landmarks=tuple(Landmark(i / 20, i / 40, 0.0) for i in range(21)),
            ),
        ),
    )

    viewer.log_hand_frame(frame)

    assert _scalar_at(rerun, "ghand/hand/count") == 1.0
    assert _scalar_at(rerun, "ghand/hand/0/score") == 0.9
    overlay_points = _entity_at(rerun, "ghand/camera/overlay/hand/0/landmarks2d")
    overlay_skeleton = _entity_at(rerun, "ghand/camera/overlay/hand/0/skeleton2d")
    assert not any(
        call[1] == "ghand/hand/0/landmarks2d" for call in rerun.calls if call[0] == "log"
    )
    assert overlay_points.kind == "Points2D"
    assert overlay_points.positions[1] == (5.0, 5.0)
    assert overlay_skeleton.kind == "LineStrips2D"
    assert overlay_skeleton.strips[0] == [(0.0, 0.0), (5.0, 5.0)]
    assert overlay_points.draw_order == 10.0
    assert overlay_skeleton.draw_order == 9.0


def test_log_hand_frame_clears_overlay_when_hand_disappears() -> None:
    rerun = FakeRerun()
    viewer = RerunViewer(rerun=rerun)
    detected = HandFrame(
        timestamp_ns=123,
        image_width=100,
        image_height=200,
        hands=(
            HandDetection(
                handedness="Right",
                score=0.9,
                landmarks=tuple(Landmark(i / 20, i / 40, 0.0) for i in range(21)),
            ),
        ),
    )
    missing = HandFrame(timestamp_ns=124, image_width=100, image_height=200, hands=())

    viewer.log_hand_frame(detected)
    viewer.log_hand_frame(missing)

    assert _scalar_at(rerun, "ghand/hand/count") == 0.0
    clear_logs = [
        call
        for call in rerun.calls
        if call[0] == "log"
        and isinstance(call[2], FakeEntity)
        and call[2].kind == "Clear"
    ]
    assert any(
        call[1] == "ghand/camera/overlay/hand"
        and call[2].kind == "Clear"
        and call[2].recursive is True
        for call in clear_logs
    )
    assert any(
        call[1] == "ghand/hand" and call[2].kind == "Clear" and call[2].recursive is True
        for call in clear_logs
    )


def test_log_joint_targets_and_runtime_snapshot_write_scalar_series() -> None:
    rerun = FakeRerun()
    viewer = RerunViewer(rerun=rerun)
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    target = JointTarget(
        timestamp_ns=123,
        joints={"right_hand.index_mcp_flex.pos": 0.5},
    )

    snapshot = apply_joint_target(runtime, target, now=0.0)
    viewer.log_joint_targets(target)
    viewer.log_runtime_snapshot(snapshot, timestamp_ns=124)
    state = next(
        state
        for state in snapshot.joints
        if state.joint_name == "right_hand.index_mcp_flex.pos"
    )

    assert (
        _scalar_at(rerun, "ghand/retargeting/normalized_target/right_hand/index_mcp_flex/pos")
        == 0.5
    )
    assert _scalar_at(rerun, "ghand/joints/target/right_hand/index_mcp_flex/pos") == pytest.approx(
        state.target
    )
    assert (
        _scalar_at(rerun, "ghand/joints/measured/right_hand/index_mcp_flex/pos")
        == pytest.approx(state.measured)
    )
    assert (
        _scalar_at(rerun, "ghand/joints/commanded/right_hand/index_mcp_flex/pos")
        == pytest.approx(state.commanded)
    )
    assert _scalar_at(rerun, "ghand/motors/id9/position_tick") == float(state.target_tick)
    assert _scalar_at(rerun, "ghand/motors/id9/present_current_ma") == 0.0
    assert _scalar_at(rerun, "ghand/motors/id9/input_voltage") == 12.0
    assert _scalar_at(rerun, "ghand/runtime/connected") == 1.0
    target_bars = _entity_at(rerun, "ghand/mapping/joints/target")
    measured_bars = _entity_at(rerun, "ghand/mapping/joints/measured")
    error_bars = _entity_at(rerun, "ghand/mapping/joints/error")
    labels = _entity_at(rerun, "ghand/mapping/labels")
    assert target_bars.kind == "BarChart"
    assert target_bars.values[9] == pytest.approx(state.target)
    assert measured_bars.kind == "BarChart"
    assert measured_bars.values[9] == pytest.approx(state.measured)
    assert error_bars.kind == "BarChart"
    assert error_bars.values[9] == pytest.approx(state.commanded - state.measured)
    assert labels.kind == "TextDocument"
    assert "9 | 9 | 9 | `index_mcp_flex`" in labels.text


def test_snapshot_joint_angles_use_runtime_radians_and_urdf_id_key() -> None:
    snapshot = OperatorRuntimeSnapshot(
        backend="test",
        connected=True,
        dry_run=False,
        emergency_stop=False,
        last_error=None,
        joints=(
            JointRuntimeState(
                joint_name="right_hand.index_mcp_flex.pos",
                dxl_id=1,
                urdf_dxl_id=9,
                target=0.0,
                measured=0.75,
                commanded=0.0,
                target_tick=2048,
                measured_tick=2048,
                torque_enabled=False,
            ),
        ),
    )
    pose_model = UrdfPoseModel(
        limits={"id9": UrdfJointLimit(joint_name="id9", lower=0.0, upper=2.0)}
    )

    angles = _snapshot_joint_angles(snapshot, 1.0, pose_model)

    assert angles == {"id9": pytest.approx(0.75)}


def test_log_hand_overlay_tracks_each_frame_and_clears_lost_hands() -> None:
    rerun = FakeRerun()
    viewer = RerunViewer(rerun=rerun)
    hand = HandDetection(
        handedness="Right",
        score=0.9,
        landmarks=tuple(Landmark(0.5, i / 20, 0.0) for i in range(21)),
    )

    viewer.log_hand_overlay(
        HandFrame(timestamp_ns=1, image_width=80, image_height=80, hands=(hand,))
    )
    viewer.log_hand_overlay(
        HandFrame(timestamp_ns=2, image_width=80, image_height=80, hands=(hand,))
    )
    viewer.log_hand_overlay(
        HandFrame(timestamp_ns=3, image_width=80, image_height=80, hands=())
    )

    overlay_logs = [
        call for call in rerun.calls if call[1] == "ghand/camera/overlay/hand/0/landmarks2d"
    ]
    assert len(overlay_logs) == 2  # one overlay per frame, nothing skipped
    assert not any(call[1] == "ghand/hand/count" for call in rerun.calls)
    assert any(
        call[1] == "ghand/camera/overlay/hand" and call[2].kind == "Clear"
        for call in rerun.calls
    )


def test_log_hand_features_writes_retargeting_feature_series() -> None:
    rerun = FakeRerun()
    viewer = RerunViewer(rerun=rerun)
    features = HandFeatures(
        timestamp_ns=123,
        handedness="Right",
        score=0.9,
        finger_curl={
            "thumb": 0.2,
            "index": 0.6,
            "middle": 0.4,
            "ring": 0.3,
            "pinky": 0.8,
        },
        finger_spread={
            "index": -0.5,
            "middle": 0.0,
            "ring": 0.25,
            "pinky": 0.5,
        },
        pinch={
            "index": 0.75,
            "middle": 0.25,
            "ring": 0.0,
        },
        palm_size=0.3,
        thumb_abduction=0.7,
    )

    viewer.log_hand_features(features)

    assert _scalar_at(rerun, "ghand/features/valid") == 1.0
    assert _scalar_at(rerun, "ghand/features/score") == 0.9
    assert _scalar_at(rerun, "ghand/features/palm_size") == 0.3
    assert _scalar_at(rerun, "ghand/features/thumb_abduction") == 0.7
    assert _scalar_at(rerun, "ghand/features/thumb_pose/lateral") == 0.7
    assert _scalar_at(rerun, "ghand/features/thumb_pose/index_pinch") == 0.75
    assert _scalar_at(rerun, "ghand/features/finger_curl/index") == 0.6
    assert _scalar_at(rerun, "ghand/features/finger_spread/index") == -0.5
    assert _scalar_at(rerun, "ghand/features/pinch/index") == 0.75
    curl = _entity_at(rerun, "ghand/mapping/features/curl")
    spread = _entity_at(rerun, "ghand/mapping/features/spread")
    pinch = _entity_at(rerun, "ghand/mapping/features/pinch")
    thumb_pose = _entity_at(rerun, "ghand/mapping/features/thumb_pose")
    assert curl.kind == "BarChart"
    assert curl.values == [0.2, 0.6, 0.4, 0.3, 0.8]
    assert spread.kind == "BarChart"
    assert spread.values == [-0.5, 0.0, 0.25, 0.5]
    assert pinch.kind == "BarChart"
    assert pinch.values == [0.75, 0.25, 0.0]
    assert thumb_pose.kind == "BarChart"
    assert thumb_pose.values == [0.7, 0.0, 0.2, 0.2, 0.2, 0.75]
    handedness = _entity_at(rerun, "ghand/features/handedness")
    assert handedness.kind == "TextLog"
    assert handedness.text == "Right"


def test_log_calibration_pose_instruction_writes_markdown_guide() -> None:
    rerun = FakeRerun()
    viewer = RerunViewer(rerun=rerun)

    viewer.log_calibration_pose_instruction(
        pose_name="flat_closed",
        title="Flat hand, fingers together",
        instruction="Extend the fingers and keep them together.",
        pose_index=1,
        pose_count=4,
        valid_samples=7,
        timestamp_ns=123,
        status="Waiting for capture: press Space.",
    )

    guide = _entity_at(rerun, "ghand/calibration/pose_guide")
    assert guide.kind == "TextDocument"
    assert guide.media_type == "text/markdown"
    assert "1/4 Flat hand, fingers together" in guide.text
    assert "pose: `flat_closed`" in guide.text
    assert "valid samples: 7" in guide.text
    assert "Waiting for capture" in guide.text


def test_rerun_viewer_logs_native_urdf_and_fk_transforms(tmp_path: Path) -> None:
    urdf_path = tmp_path / "hand.urdf"
    urdf_path.write_text(
        "<robot name='mini_hand'><link name='base' /></robot>",
        encoding="utf-8",
    )
    rerun = FakeRerun()
    viewer = RerunViewer(rerun=rerun, urdf_path=urdf_path)
    runtime = make_fake_hand_runtime(target_velocity_limit=1000.0)
    runtime.connect()
    runtime.set_torque_enabled(True)
    target = JointTarget(
        timestamp_ns=123,
        joints={"right_hand.index_mcp_flex.pos": 0.5},
    )

    snapshot = apply_joint_target(runtime, target, now=0.0)
    viewer.log_runtime_snapshot(snapshot, timestamp_ns=123)

    transform_logs = [
        call
        for call in rerun.calls
        if call[0] == "log" and call[2].kind == "ComputedTransform"
    ]
    assert viewer._urdf_tree is not None
    assert viewer._urdf_tree.logged is True
    assert viewer._urdf_tree.path != urdf_path
    urdf_root = ElementTree.parse(viewer._urdf_tree.path).getroot()
    display_joint = urdf_root.find("./joint[@name='ghand_display_root_to_base']")
    assert display_joint is not None
    assert display_joint.find("origin").attrib["rpy"] == " ".join(
        f"{value:.8g}" for value in DISPLAY_HAND_ROTATION_RPY
    )
    assert display_joint.find("parent").attrib["link"] == "ghand_display_root"
    assert display_joint.find("child").attrib["link"] == "base"
    assert viewer._urdf_tree.kwargs["entity_path_prefix"] == "ghand_robot_right_hand"
    assert viewer._urdf_tree.kwargs["frame_prefix"] == "ghand/right_hand/"
    assert (
        viewer._urdf_tree.kwargs["static_transform_entity_path"]
        == "ghand_robot_right_hand/tf_static"
    )
    assert any(call[0] == "send_blueprint" for call in rerun.calls)
    assert len(transform_logs) == 1
    assert transform_logs[0][1] == "ghand_robot_right_hand/joint_transforms"
    state = next(
        state
        for state in snapshot.joints
        if state.joint_name == "right_hand.index_mcp_flex.pos"
    )
    assert transform_logs[0][2].angle == pytest.approx(state.measured)
    assert transform_logs[0][2].clamp is True


def test_log_latency_and_text() -> None:
    rerun = FakeRerun()
    viewer = RerunViewer(rerun=rerun)

    viewer.log_latency("mediapipe/inference", 0.012, timestamp_ns=123)
    viewer.log_text("tracking started", level="INFO", timestamp_ns=123)

    assert _scalar_at(rerun, "ghand/latency/mediapipe/inference_seconds") == 0.012
    text = _entity_at(rerun, "ghand/logs")
    assert text.kind == "TextLog"
    assert text.text == "tracking started"
    assert text.level == "INFO"


def test_disabled_viewer_is_noop() -> None:
    rerun = FakeRerun()
    viewer = RerunViewer(enabled=False, rerun=rerun)

    viewer.start()
    viewer.log_latency("x", 0.1)

    assert rerun.calls == []


def _entity_at(rerun: FakeRerun, path: str) -> FakeEntity:
    matches = [call[2] for call in rerun.calls if call[0] == "log" and call[1] == path]
    assert matches
    return matches[-1]


def _scalar_at(rerun: FakeRerun, path: str) -> float:
    entity = _entity_at(rerun, path)
    assert entity.kind == "Scalars"
    return float(entity.value)


class FakeEntity:
    def __init__(self, kind: str, **kwargs: object) -> None:
        self.kind = kind
        self.__dict__.update(kwargs)


class FakeBlueprintModule:
    def EyeControls3D(self, **kwargs: object) -> dict[str, object]:
        return dict(kwargs)


class FakeRerun:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.urdf = FakeUrdfModule()

    def init(self, application_id: str, *, spawn: bool = False) -> None:
        self.calls.append(("init", application_id, spawn))

    def save(self, path: Path) -> None:
        self.calls.append(("save", path))

    def set_time_sequence(self, timeline: str, sequence: int) -> None:
        self.calls.append(("set_time_sequence", timeline, sequence))

    def set_time_seconds(self, timeline: str, seconds: float) -> None:
        self.calls.append(("set_time_seconds", timeline, seconds))

    def log(
        self, entity_path: str, entity: FakeEntity, *, static: bool = False
    ) -> None:
        self.calls.append(("log", entity_path, entity, static))

    def send_blueprint(self, blueprint: object) -> None:
        self.calls.append(("send_blueprint", blueprint))

    def Image(self, image: object) -> FakeEntity:
        return FakeEntity("Image", data=image)

    def Points2D(
        self,
        positions: list[tuple[float, float]],
        *,
        radii: float | None = None,
        colors: list[tuple[int, int, int, int]] | None = None,
        labels: list[str] | None = None,
        draw_order: float | None = None,
    ) -> FakeEntity:
        return FakeEntity(
            "Points2D",
            positions=positions,
            radii=radii,
            colors=colors,
            labels=labels,
            draw_order=draw_order,
        )

    def LineStrips2D(
        self,
        strips: list[list[tuple[float, float]]],
        *,
        radii: float | None = None,
        colors: list[tuple[int, int, int, int]] | None = None,
        draw_order: float | None = None,
    ) -> FakeEntity:
        return FakeEntity(
            "LineStrips2D",
            strips=strips,
            radii=radii,
            colors=colors,
            draw_order=draw_order,
        )

    def Scalars(self, value: float) -> FakeEntity:
        return FakeEntity("Scalars", value=value)

    def TextLog(self, text: str, *, level: str = "INFO") -> FakeEntity:
        return FakeEntity("TextLog", text=text, level=level)

    def Clear(self, *, recursive: bool) -> FakeEntity:
        return FakeEntity("Clear", recursive=recursive)

    def TextDocument(self, text: str, *, media_type: str | None = None) -> FakeEntity:
        return FakeEntity("TextDocument", text=text, media_type=media_type)

    def BarChart(
        self,
        values: list[float],
        *,
        color: tuple[int, int, int, int] | None = None,
    ) -> FakeEntity:
        return FakeEntity("BarChart", values=values, color=color)


class FakeUrdfModule:
    UrdfTree = None


class FakeUrdfTree:
    def __init__(self, path: Path, kwargs: dict[str, object]) -> None:
        self.path = path
        self.kwargs = kwargs
        self.logged = False
        self._joints = (
            FakeUrdfJoint("id9", "continuous"),
            FakeUrdfJoint("fixed_joint", "fixed"),
        )

    @classmethod
    def from_file_path(cls, path: Path, **kwargs: object) -> "FakeUrdfTree":
        return cls(Path(path), kwargs)

    def log_urdf_to_recording(self) -> None:
        self.logged = True

    def joints(self) -> tuple["FakeUrdfJoint", ...]:
        return self._joints


class FakeUrdfJoint:
    def __init__(self, name: str, joint_type: str) -> None:
        self.name = name
        self.joint_type = joint_type

    def compute_transform(self, angle: float, *, clamp: bool = True) -> FakeEntity:
        return FakeEntity(
            "ComputedTransform", joint_name=self.name, angle=angle, clamp=clamp
        )


FakeUrdfModule.UrdfTree = FakeUrdfTree
