import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghand.control import JointRuntimeState
from ghand.hand import (
    DISPLAY_HAND_ROTATION_EULER_X_RAD,
    UrdfJointLimit,
    UrdfPoseModel,
)
from ghand.ui.operator import (
    _format_joint_degrees,
    _viewer_joint_angle,
)


def test_operator_ui_viewer_joint_angle_uses_target_before_connection() -> None:
    state = _joint_state(target=0.5, measured=0.0)
    snapshot = SimpleNamespace(connected=False, dry_run=False)

    angle = _viewer_joint_angle(
        snapshot,
        state,
        viewer_angle_scale=2.0,
        pose_model=UrdfPoseModel.empty(),
    )

    assert angle == pytest.approx(1.0)


def test_operator_ui_viewer_joint_angle_uses_measured_when_connected() -> None:
    state = _joint_state(target=0.2, measured=0.5)
    snapshot = SimpleNamespace(connected=True, dry_run=False)

    angle = _viewer_joint_angle(
        snapshot,
        state,
        viewer_angle_scale=2.0,
        pose_model=UrdfPoseModel.empty(),
    )

    assert angle == pytest.approx(1.0)


def test_operator_ui_viewer_joint_angle_uses_target_in_dry_run() -> None:
    state = _joint_state(target=0.5, measured=0.0)
    snapshot = SimpleNamespace(connected=True, dry_run=True)

    angle = _viewer_joint_angle(
        snapshot,
        state,
        viewer_angle_scale=2.0,
        pose_model=UrdfPoseModel.empty(),
    )

    assert angle == pytest.approx(1.0)


def test_operator_ui_viewer_joint_angle_uses_runtime_radians_not_urdf_normalization() -> None:
    state = _joint_state(target=0.0, measured=1.25, dxl_id=1, urdf_dxl_id=9)
    snapshot = SimpleNamespace(connected=True, dry_run=False)
    pose_model = UrdfPoseModel(
        limits={"id9": UrdfJointLimit(joint_name="id9", lower=0.0, upper=2.0)}
    )

    angle = _viewer_joint_angle(
        snapshot,
        state,
        viewer_angle_scale=1.0,
        pose_model=pose_model,
    )

    assert angle == pytest.approx(1.25)


def test_operator_ui_displays_targets_in_degrees() -> None:
    degrees = math.degrees(0.5)

    assert degrees == pytest.approx(28.647889)
    assert _format_joint_degrees(degrees) == "+28.6 deg"
    assert _format_joint_degrees(-0.0) == "+0.0 deg"


def test_operator_ui_converts_display_degrees_back_to_radians() -> None:
    value = math.radians(90.0)

    assert value == pytest.approx(math.pi / 2.0)


def test_operator_ui_degree_slider_range_is_sorted() -> None:
    assert UrdfPoseModel.empty().degree_range(dxl_id=9) == pytest.approx((-180.0, 180.0))


def test_operator_ui_static_viewer_applies_shared_root_orientation() -> None:
    viewer_js = Path("src/ghand/ui/static/urdf_viewer.js").read_text(encoding="utf-8")

    assert DISPLAY_HAND_ROTATION_EULER_X_RAD == pytest.approx(math.pi)
    assert "rootRotationXRad" in viewer_js
    assert "this.robot.rotation.x = this.rootRotationXRad;" in viewer_js
    assert "radius * -2.0" in viewer_js


def _joint_state(
    *,
    target: float,
    measured: float,
    dxl_id: int = 9,
    urdf_dxl_id: int | None = None,
) -> JointRuntimeState:
    return JointRuntimeState(
        joint_name="right_hand.index_mcp_flex.pos",
        dxl_id=dxl_id,
        target=target,
        measured=measured,
        commanded=measured,
        target_tick=2048,
        measured_tick=2048,
        torque_enabled=False,
        urdf_dxl_id=urdf_dxl_id,
    )
