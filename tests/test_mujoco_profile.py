from __future__ import annotations

import json
import math

import pytest

from ghand.sim.mujoco.profile import (
    DEFAULT_MUJOCO_PHYSICS_PROFILE,
    HAND_SELF_COLLISION_SAFE_CROSS_LINK_V1,
    TELEOP_MUJOCO_PHYSICS_PROFILE,
    MUJOCO_VALIDATED_PROFILE_SCHEMA_VERSION,
    VALIDATED_MUJOCO_PARAMETER_NAMES,
    MujocoPhysicsProfile,
    MujocoPhysicsProfileArtifactError,
    load_validated_mujoco_physics_profile,
    make_mujoco_joint_calibrations,
    mujoco_joint_ranges_from_urdf,
    validated_mujoco_parameters_sha256,
)
from ghand.hand import default_right_hand_urdf_path


def test_default_mujoco_physics_profile_is_json_serializable() -> None:
    payload = DEFAULT_MUJOCO_PHYSICS_PROFILE.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["schema_version"] == "v4"
    assert payload["physics_timestep_s"] == pytest.approx(0.002)
    assert payload["control_hz"] == pytest.approx(10.0)
    assert payload["target_velocity_limit_rad_s"] == pytest.approx(20.0)
    assert payload["actuator_kp"] == pytest.approx(6.0)
    assert payload["actuator_velocity_feedback_gain"] == pytest.approx(0.0)
    assert payload["joint_frictionloss"] == pytest.approx(0.0)
    assert payload["joint_limit_solref"] == pytest.approx((0.02, 1.0))
    assert payload["joint_limit_solimp"] == pytest.approx((0.9, 0.95, 0.001, 0.5, 2.0))
    assert payload["solver_impratio"] == pytest.approx(1.0)
    assert payload["actuator_force_limit"] == pytest.approx(0.8)
    assert payload["hand_self_collision_mode"] == HAND_SELF_COLLISION_SAFE_CROSS_LINK_V1
    assert payload["hand_self_collision_solref"] == pytest.approx([0.005, 1.0])
    assert payload["hand_self_collision_solimp"] == pytest.approx(
        [0.99, 0.999, 0.0005, 0.5, 2.0]
    )
    assert payload["hand_self_collision_margin_m"] == pytest.approx(0.0015)


def test_teleop_profile_is_responsive_without_changing_default_dynamics() -> None:
    profile = TELEOP_MUJOCO_PHYSICS_PROFILE

    assert profile.name == "ghand-teleop-responsive"
    assert profile.control_hz == pytest.approx(10.0)
    assert profile.target_velocity_limit_rad_s == pytest.approx(20.0)
    assert profile.actuator_kp == pytest.approx(20.0)
    assert profile.actuator_force_limit == pytest.approx(1.0)
    assert profile.joint_damping == pytest.approx(0.06)
    assert profile.joint_armature == pytest.approx(0.0001)
    assert DEFAULT_MUJOCO_PHYSICS_PROFILE.actuator_kp == pytest.approx(6.0)
    assert DEFAULT_MUJOCO_PHYSICS_PROFILE.actuator_force_limit == pytest.approx(0.8)
    assert DEFAULT_MUJOCO_PHYSICS_PROFILE.joint_damping == pytest.approx(0.025)
    assert DEFAULT_MUJOCO_PHYSICS_PROFILE.joint_armature == pytest.approx(0.001)


def test_mujoco_physics_profile_runtime_overrides_are_validated() -> None:
    changed = DEFAULT_MUJOCO_PHYSICS_PROFILE.with_runtime_overrides(
        target_velocity_limit_rad_s=12.0,
        actuator_kp=4.0,
        actuator_force_limit=0.5,
    )

    assert changed.target_velocity_limit_rad_s == pytest.approx(12.0)
    assert changed.actuator_kp == pytest.approx(4.0)
    assert changed.actuator_force_limit == pytest.approx(0.5)
    assert DEFAULT_MUJOCO_PHYSICS_PROFILE.actuator_kp == pytest.approx(6.0)

    with pytest.raises(ValueError, match="actuator_kp"):
        MujocoPhysicsProfile(actuator_kp=0.0)
    with pytest.raises(ValueError, match="actuator_velocity_feedback_gain"):
        MujocoPhysicsProfile(actuator_velocity_feedback_gain=-0.1)
    with pytest.raises(ValueError, match="joint_frictionloss"):
        MujocoPhysicsProfile(joint_frictionloss=-0.1)
    with pytest.raises(ValueError, match="solver_impratio"):
        MujocoPhysicsProfile(solver_impratio=0.0)
    with pytest.raises(ValueError, match="joint_limit_solref"):
        MujocoPhysicsProfile(joint_limit_solref=(0.001, 1.0))
    with pytest.raises(ValueError, match="integer number of physics timesteps"):
        MujocoPhysicsProfile(control_hz=15.0)
    with pytest.raises(ValueError, match="hand_self_collision_mode"):
        MujocoPhysicsProfile(hand_self_collision_mode="all")
    with pytest.raises(ValueError, match="time constant"):
        MujocoPhysicsProfile(hand_self_collision_solref=(0.003, 1.0))
    with pytest.raises(ValueError, match="exactly 5 values"):
        MujocoPhysicsProfile(hand_self_collision_solimp=(0.99, 0.999, 0.0005, 0.5))


def test_mujoco_joint_calibrations_match_packaged_urdf() -> None:
    urdf = default_right_hand_urdf_path()
    ranges = mujoco_joint_ranges_from_urdf(urdf)
    calibrations = make_mujoco_joint_calibrations(urdf_path=urdf)

    assert tuple(ranges) == tuple(f"id{index}" for index in range(16))
    assert len(calibrations) == 16
    for calibration in calibrations:
        assert (
            calibration.lower_rad,
            calibration.upper_rad,
        ) == pytest.approx(ranges[f"id{calibration.pose_dxl_id}"])

    assert ranges["id9"] == pytest.approx((-1.74533, 0.314159))


def test_mujoco_joint_calibrations_preserve_canonical_teleop_angles() -> None:
    calibrations = make_mujoco_joint_calibrations(
        urdf_path=default_right_hand_urdf_path()
    )
    index_mcp_flex = next(
        calibration
        for calibration in calibrations
        if calibration.joint_name == "right_hand.index_mcp_flex.pos"
    )

    assert index_mcp_flex.normalized_to_angle_rad(-70.0 / 180.0) == pytest.approx(
        math.radians(-70.0)
    )
    assert index_mcp_flex.normalized_to_angle_rad(0.5) == pytest.approx(
        index_mcp_flex.upper_rad
    )


def test_validated_profile_loader_overlays_only_identified_parameters(
    tmp_path,
) -> None:
    parameters = {
        "actuator_kp": 4.0,
        "joint_damping": 0.05,
    }
    artifact = {
        "schema_version": MUJOCO_VALIDATED_PROFILE_SCHEMA_VERSION,
        "artifact_id": "ghand-mujoco-validated-test",
        "status": "validated",
        "source_validation": {"path": "validation.json", "sha256": "a" * 64},
        "validated_parameter_names": list(VALIDATED_MUJOCO_PARAMETER_NAMES),
        "parameters": parameters,
        "parameters_sha256": validated_mujoco_parameters_sha256(parameters),
        "application": {
            "mode": "explicit_load_only",
            "default_profile_modified": False,
        },
    }
    path = tmp_path / "validated_profile.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    loaded = load_validated_mujoco_physics_profile(
        path,
        base_profile=DEFAULT_MUJOCO_PHYSICS_PROFILE,
    )

    assert loaded.actuator_kp == pytest.approx(4.0)
    assert loaded.joint_damping == pytest.approx(0.05)
    assert loaded.joint_armature == pytest.approx(
        DEFAULT_MUJOCO_PHYSICS_PROFILE.joint_armature
    )
    assert loaded.control_hz == DEFAULT_MUJOCO_PHYSICS_PROFILE.control_hz
    assert loaded.cube_half_size_m == (
        DEFAULT_MUJOCO_PHYSICS_PROFILE.cube_half_size_m
    )
    assert loaded.actuator_force_limit == (
        DEFAULT_MUJOCO_PHYSICS_PROFILE.actuator_force_limit
    )
    assert loaded.hand_self_collision_mode == (
        DEFAULT_MUJOCO_PHYSICS_PROFILE.hand_self_collision_mode
    )

    artifact["schema_version"] = "ghand-mujoco-validated-profile-v2"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(MujocoPhysicsProfileArtifactError, match="Unsupported"):
        load_validated_mujoco_physics_profile(path)

    artifact["schema_version"] = MUJOCO_VALIDATED_PROFILE_SCHEMA_VERSION
    artifact["status"] = "candidate"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(
        MujocoPhysicsProfileArtifactError, match="not passed validation"
    ):
        load_validated_mujoco_physics_profile(path)

    artifact["status"] = "validated"
    artifact["parameters"]["joint_damping"] = -0.01
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(
        MujocoPhysicsProfileArtifactError,
        match="finite and non-negative",
    ):
        load_validated_mujoco_physics_profile(path)


def test_loader_accepts_existing_explicitly_approved_profile(tmp_path) -> None:
    parameters = {"actuator_kp": 4.0, "joint_damping": 0.05}
    artifact = {
        "schema_version": "ghand-mujoco-validated-profile-v3",
        "artifact_id": "ghand-mujoco-validated-legacy",
        "status": "approved",
        "approval": {
            "approved_by": "existing-user",
            "approved_at_utc": "2026-08-28T00:00:00+00:00",
        },
        "source_validation": {"path": "validation.json", "sha256": "a" * 64},
        "validated_parameter_names": list(VALIDATED_MUJOCO_PARAMETER_NAMES),
        "parameters": parameters,
        "parameters_sha256": validated_mujoco_parameters_sha256(parameters),
    }
    path = tmp_path / "existing-validated-profile.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    loaded = load_validated_mujoco_physics_profile(path)

    assert loaded.actuator_kp == pytest.approx(4.0)
    assert loaded.joint_damping == pytest.approx(0.05)
