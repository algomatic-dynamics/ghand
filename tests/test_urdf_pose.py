import math
from pathlib import Path

import pytest

from ghand.hand import (
    UrdfJointLimit,
    UrdfPoseModel,
    load_urdf_joint_limits,
)


def test_urdf_pose_model_defaults_missing_joint_limits_to_full_rotation() -> None:
    model = UrdfPoseModel.empty()

    assert model.degree_range(dxl_id=9) == pytest.approx((-180.0, 180.0))
    assert model.target_to_angle(dxl_id=9, value=-1.0) == pytest.approx(-math.pi)
    assert model.target_to_angle(dxl_id=9, value=0.0) == pytest.approx(0.0)
    assert model.target_to_angle(dxl_id=9, value=1.0) == pytest.approx(math.pi)
    assert model.degrees_to_target(dxl_id=9, degrees=90.0) == pytest.approx(0.5)


def test_urdf_pose_model_reads_explicit_urdf_joint_limits(tmp_path: Path) -> None:
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="hand">
          <joint name="id9" type="revolute">
            <parent link="a" />
            <child link="b" />
            <limit lower="-0.5" upper="1.0" effort="1" velocity="1" />
          </joint>
        </robot>
        """,
        encoding="utf-8",
    )

    model = UrdfPoseModel.from_urdf_file(urdf)

    assert model.degree_range(dxl_id=9) == pytest.approx((math.degrees(-0.5), math.degrees(1.0)))
    assert model.target_to_angle(dxl_id=9, value=-1.0) == pytest.approx(-0.5)
    assert model.target_to_angle(dxl_id=9, value=0.0) == pytest.approx(0.25)
    assert model.target_to_angle(dxl_id=9, value=1.0) == pytest.approx(1.0)
    assert model.angle_to_target(dxl_id=9, angle=0.25) == pytest.approx(0.0)


def test_urdf_pose_model_assigns_default_limits_when_urdf_limit_is_missing(tmp_path: Path) -> None:
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="hand">
          <joint name="id9" type="continuous">
            <parent link="a" />
            <child link="b" />
          </joint>
          <joint name="fixed_joint" type="fixed">
            <parent link="b" />
            <child link="c" />
          </joint>
        </robot>
        """,
        encoding="utf-8",
    )

    limits = load_urdf_joint_limits(urdf)

    assert limits["id9"] == UrdfJointLimit(joint_name="id9")
    assert "fixed_joint" not in limits
