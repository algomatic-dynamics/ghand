from __future__ import annotations

import math
import re
import subprocess
import sys

import pytest

from ghand.hand import (
    DEFAULT_RIGHT_HAND_URDF,
    GHAND_HAND_JOINTS,
    GHAND_RIGHT_MOTOR_IDS,
    LEAP_VECTOR_LENGTH,
    default_right_hand_urdf_path,
    joint_radians_to_leap_hand_vector,
    joint_radians_to_leap_sim_vector,
    leap_hand_vector_to_joint_radians,
    leap_sim_vector_to_joint_radians,
    make_hand_joint_names,
    make_real_hand_joint_calibrations,
)


def test_base_packages_import_without_optional_dependencies() -> None:
    script = (
        "import sys\n"
        "import ghand.hand, ghand.hardware, ghand.control\n"
        "loaded = [m for m in ('cv2', 'mediapipe', 'scipy', 'rerun', 'trimesh', "
        "'mujoco', 'nicegui', 'lerobot') if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == ""


def test_packaged_model_has_no_revision_in_its_file_name() -> None:
    assert "rev" not in DEFAULT_RIGHT_HAND_URDF
    path = default_right_hand_urdf_path()
    assert path.is_file()
    assert re.search(r"rev\d", path.read_text()) is None


def test_real_hand_mapping_matches_the_joint_vocabulary() -> None:
    calibrations = make_real_hand_joint_calibrations()
    assert tuple(c.joint_name for c in calibrations) == make_hand_joint_names("right")
    assert tuple(c.dxl_id for c in calibrations) == GHAND_RIGHT_MOTOR_IDS
    assert len(GHAND_HAND_JOINTS) == LEAP_VECTOR_LENGTH


def test_leap_sim_vector_maps_by_motor_id_with_packaged_direction() -> None:
    vector = [0.01 * index for index in range(LEAP_VECTOR_LENGTH)]
    joints = leap_sim_vector_to_joint_radians(vector)
    names = make_hand_joint_names("right")

    assert set(joints) == set(names)
    # LEAP index finger (IDs 0..3) is the third ghand finger group; ring (IDs 8..11) is first.
    assert joints["right_hand.ring_mcp_side.pos"] == pytest.approx(-vector[8])
    assert joints["right_hand.index_mcp_side.pos"] == pytest.approx(-vector[0])
    assert joints["right_hand.thumb_ip_flex.pos"] == pytest.approx(-vector[15])
    assert joint_radians_to_leap_sim_vector(joints) == pytest.approx(vector)


def test_leap_hand_vector_is_pi_centred() -> None:
    neutral = [math.pi] * LEAP_VECTOR_LENGTH
    joints = leap_hand_vector_to_joint_radians(neutral)
    assert all(value == pytest.approx(0.0) for value in joints.values())
    assert joint_radians_to_leap_hand_vector({}) == pytest.approx(neutral)


@pytest.mark.parametrize("bad", [[0.0] * 15, [0.0] * 17, [float("nan")] + [0.0] * 15])
def test_leap_vectors_are_validated(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        leap_sim_vector_to_joint_radians(bad)
