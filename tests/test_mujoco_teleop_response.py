from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ghand.control import apply_joint_target, make_fake_hand_runtime
from ghand.hand import (
    default_right_hand_urdf_path,
    JointTarget,
)
from ghand.retargeting.anchor_pose import default_anchor_robot_pose_targets
from ghand.sim.mujoco import (
    HAND_SELF_COLLISION_DISABLED,
    TELEOP_MUJOCO_PHYSICS_PROFILE,
    export_mujoco_hand_model,
    make_mujoco_hand_runtime,
    make_mujoco_joint_calibrations,
)


def test_teleop_profile_tracks_canonical_pose_steps_and_ramps(
    tmp_path: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    profile = replace(
        TELEOP_MUJOCO_PHYSICS_PROFILE,
        hand_self_collision_mode=HAND_SELF_COLLISION_DISABLED,
    )
    exported = export_mujoco_hand_model(tmp_path, physics_profile=profile)
    model = mujoco.MjModel.from_xml_path(str(exported.hand_xml))
    poses, qpos_ids, dof_ids, actuator_ids = _teleop_test_channels(mujoco, model)

    assert model.npair == 0

    step_metrics = _canonical_step_metrics(
        mujoco,
        model,
        poses=poses,
        qpos_ids=qpos_ids,
        dof_ids=dof_ids,
        actuator_ids=actuator_ids,
    )
    ramp_metrics = _canonical_ramp_metrics(
        mujoco,
        model,
        poses=poses,
        qpos_ids=qpos_ids,
        dof_ids=dof_ids,
        actuator_ids=actuator_ids,
    )

    assert step_metrics["error_100_ms_rad"] < 0.08
    assert step_metrics["error_200_ms_rad"] < 0.007
    assert step_metrics["error_500_ms_rad"] < 0.005
    assert step_metrics["overshoot_rad"] < 0.10
    assert step_metrics["peak_velocity_rad_s"] < 22.0
    assert ramp_metrics["tracking_error_rad"] < 0.015
    assert ramp_metrics["hold_error_rad"] < 0.005


def test_normalized_teleop_target_matches_fake_and_mujoco_radians(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    profile = replace(
        TELEOP_MUJOCO_PHYSICS_PROFILE,
        hand_self_collision_mode=HAND_SELF_COLLISION_DISABLED,
    )
    exported = export_mujoco_hand_model(tmp_path, physics_profile=profile)
    fake_runtime = make_fake_hand_runtime(
        target_hz=profile.control_hz,
        target_velocity_limit=profile.target_velocity_limit_rad_s,
    )
    mujoco_runtime = make_mujoco_hand_runtime(
        scene=exported.hand_xml,
        physics_profile=profile,
    )
    joint_name = "right_hand.index_mcp_flex.pos"
    target = JointTarget(
        timestamp_ns=1,
        joints={joint_name: -70.0 / 180.0},
    )

    measured = {}
    try:
        for label, runtime in (("fake", fake_runtime), ("mujoco", mujoco_runtime)):
            runtime.connect()
            runtime.sync_motion_state_to_measured(now=0.0)
            runtime.set_torque_enabled(True)
            apply_joint_target(runtime, target, now=0.1)
            snapshot = apply_joint_target(runtime, target, now=0.2)
            state = next(
                state for state in snapshot.joints if state.joint_name == joint_name
            )
            assert state.target == pytest.approx(np.deg2rad(-70.0))
            assert state.commanded == pytest.approx(np.deg2rad(-70.0))
            measured[label] = state.measured
    finally:
        fake_runtime.disconnect()
        mujoco_runtime.disconnect()

    assert measured["mujoco"] == pytest.approx(measured["fake"], abs=0.002)


def _teleop_test_channels(mujoco, model):
    calibrations = make_mujoco_joint_calibrations(
        urdf_path=default_right_hand_urdf_path()
    )
    poses = {
        pose_name: np.asarray(
            [
                calibration.normalized_to_angle_rad(
                    pose.joints[calibration.joint_name.split(".")[1]]
                )
                for calibration in calibrations
            ],
            dtype=np.float64,
        )
        for pose_name, pose in default_anchor_robot_pose_targets().items()
    }
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"id{calibration.pose_dxl_id}",
            )
            for calibration in calibrations
        ],
        dtype=np.int32,
    )
    actuator_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"id{calibration.pose_dxl_id}_position",
            )
            for calibration in calibrations
        ],
        dtype=np.int32,
    )
    return (
        poses,
        np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32),
        np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32),
        actuator_ids,
    )


def _canonical_pose_scenarios(poses):
    pose_names = tuple(name for name in poses if name != "flat_closed")
    return tuple(("flat_closed", name) for name in pose_names) + tuple(
        (name, "flat_closed") for name in pose_names
    )


def _canonical_step_metrics(
    mujoco,
    model,
    *,
    poses,
    qpos_ids,
    dof_ids,
    actuator_ids,
):
    maximum_error_by_step = {49: 0.0, 99: 0.0, 249: 0.0}
    maximum_overshoot = 0.0
    peak_velocity = 0.0
    data = mujoco.MjData(model)
    for start_name, destination_name in _canonical_pose_scenarios(poses):
        start = poses[start_name]
        destination = poses[destination_name]
        delta = destination - start
        moving = np.abs(delta) > 1e-6
        direction = np.sign(delta[moving])
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_ids] = start
        data.ctrl[actuator_ids] = start
        mujoco.mj_forward(model, data)
        data.ctrl[actuator_ids] = destination
        for step in range(250):
            mujoco.mj_step(model, data)
            position = np.asarray(data.qpos[qpos_ids], dtype=np.float64)
            peak_velocity = max(
                peak_velocity,
                float(np.max(np.abs(data.qvel[dof_ids]))),
            )
            signed_beyond = (position[moving] - destination[moving]) * direction
            maximum_overshoot = max(
                maximum_overshoot,
                float(np.max(np.maximum(signed_beyond, 0.0))),
            )
            if step in maximum_error_by_step:
                maximum_error_by_step[step] = max(
                    maximum_error_by_step[step],
                    float(np.max(np.abs(destination - position))),
                )

    return {
        "error_100_ms_rad": maximum_error_by_step[49],
        "error_200_ms_rad": maximum_error_by_step[99],
        "error_500_ms_rad": maximum_error_by_step[249],
        "overshoot_rad": maximum_overshoot,
        "peak_velocity_rad_s": peak_velocity,
    }


def _canonical_ramp_metrics(
    mujoco,
    model,
    *,
    poses,
    qpos_ids,
    dof_ids,
    actuator_ids,
):
    del dof_ids
    ramp_frames = 5
    hold_frames = 5
    maximum_tracking_error = 0.0
    maximum_hold_error = 0.0
    data = mujoco.MjData(model)
    for start_name, destination_name in _canonical_pose_scenarios(poses):
        start = poses[start_name]
        destination = poses[destination_name]
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_ids] = start
        data.ctrl[actuator_ids] = start
        mujoco.mj_forward(model, data)
        for frame in range(ramp_frames + hold_frames):
            amount = min((frame + 1) / ramp_frames, 1.0)
            target = start + amount * (destination - start)
            data.ctrl[actuator_ids] = target
            for _ in range(TELEOP_MUJOCO_PHYSICS_PROFILE.physics_steps_per_control):
                mujoco.mj_step(model, data)
            error = float(np.max(np.abs(target - data.qpos[qpos_ids])))
            if frame < ramp_frames:
                maximum_tracking_error = max(maximum_tracking_error, error)
            else:
                maximum_hold_error = max(maximum_hold_error, error)

    return {
        "tracking_error_rad": maximum_tracking_error,
        "hold_error_rad": maximum_hold_error,
    }
