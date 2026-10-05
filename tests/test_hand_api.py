from __future__ import annotations

import math

import pytest

from ghand import Hand
from ghand.control import (
    DEFAULT_OPERATOR_PROFILE_ACCELERATION,
    DEFAULT_OPERATOR_PROFILE_VELOCITY,
    HOME_POSE,
)
from ghand.hand import GHAND_RIGHT_MOTOR_IDS, make_hand_joint_names
from ghand.hardware import HandController


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def make_hand(**kwargs) -> tuple[Hand, FakeClock]:
    clock = FakeClock()
    hand = Hand.fake(sleep=clock.sleep, clock=clock, target_velocity_limit=100.0, **kwargs)
    return hand, clock


def test_short_names_follow_the_packaged_joint_order() -> None:
    hand, _ = make_hand()
    assert hand.joint_names[:4] == (
        "ring_mcp_side",
        "ring_mcp_flex",
        "ring_pip_flex",
        "ring_dip_flex",
    )
    assert len(hand.joint_names) == 16
    assert set(hand.joint_limits) == set(hand.joint_names)
    # URDF limits, not the +/-pi placeholder of the default calibration.
    assert all(upper - lower < 2 * math.pi for lower, upper in hand.joint_limits.values())


def test_targets_are_clamped_to_the_urdf_range() -> None:
    hand, _ = make_hand()
    with hand:
        _, upper = hand.joint_limits["index_mcp_flex"]
        hand.set_joint_positions({"index_mcp_flex": upper + 1.0})
        snapshot = hand.step()
    joint = next(j for j in snapshot.joints if j.joint_name.endswith("index_mcp_flex.pos"))
    assert joint.target == pytest.approx(upper)


def test_context_manager_enables_and_releases_torque() -> None:
    hand, _ = make_hand()
    with hand as active:
        assert active is hand
        assert hand.snapshot().connected
        assert all(joint.torque_enabled for joint in hand.snapshot().joints)
    assert not hand.snapshot().connected
    assert not any(joint.torque_enabled for joint in hand.snapshot().joints)


def test_set_and_step_accept_short_and_full_joint_names() -> None:
    hand, _ = make_hand()
    with hand:
        hand.set_joint_positions({"index_mcp_flex": -0.5, "right_hand.index_pip_flex.pos": -0.25})
        snapshot = hand.step()
    by_name = {joint.joint_name: joint for joint in snapshot.joints}
    assert by_name["right_hand.index_mcp_flex.pos"].target == pytest.approx(-0.5)
    assert by_name["right_hand.index_pip_flex.pos"].target == pytest.approx(-0.25)


def test_unknown_joint_names_are_rejected() -> None:
    hand, _ = make_hand()
    with hand, pytest.raises(KeyError, match="pinky_mcp_flex"):
        hand.set_joint_positions({"pinky_mcp_flex": 0.1})


def test_move_to_interpolates_and_home_returns_to_zero() -> None:
    hand, clock = make_hand(target_hz=10.0)
    with hand:
        lower, _ = hand.joint_limits["index_mcp_flex"]
        snapshot = hand.move_to({"index_mcp_flex": lower / 2.0}, seconds=1.0)
        assert clock.now == pytest.approx(0.9)
        assert hand.joint_positions()["index_mcp_flex"] == pytest.approx(lower / 2.0, abs=1e-3)
        assert snapshot.joints[0].commanded == pytest.approx(0.0)
        hand.move_to(HOME_POSE, seconds=0.5)
        assert all(
            math.isclose(value, 0.0, abs_tol=1e-3) for value in hand.joint_positions().values()
        )


def test_move_to_keeps_the_control_period_however_long_a_step_takes(monkeypatch) -> None:
    hand, clock = make_hand(target_hz=10.0)
    step_times: list[float] = []
    runtime_step = hand.runtime.step

    def slow_step(*, now=None):
        step_times.append(clock.now)
        clock.now += 0.03  # bus traffic and viewer logging
        return runtime_step(now=now)

    monkeypatch.setattr(hand.runtime, "step", slow_step)
    with hand:
        hand.move_to({"index_mcp_flex": -0.2}, seconds=0.5)
        hand.move_to(HOME_POSE, seconds=0.3)
    # consecutive moves continue on the same 0.1 s grid
    assert step_times == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])


def test_move_to_rejects_unknown_poses_and_negative_durations() -> None:
    hand, _ = make_hand()
    with hand:
        with pytest.raises(ValueError, match="Unknown pose"):
            hand.move_to("fist")
        with pytest.raises(ValueError, match="non-negative"):
            hand.move_to(HOME_POSE, seconds=-1.0)


def test_connect_resolves_the_single_adapter_without_opening_it(monkeypatch) -> None:
    monkeypatch.setattr(
        "ghand.hardware.serial_ports.list_serial_ports",
        lambda: ("/dev/serial/by-id/usb-FTDI-U2D2-if00-port0",),
    )
    hand = Hand.connect()
    assert hand.runtime.backend == "dynamixel:/dev/serial/by-id/usb-FTDI-U2D2-if00-port0"
    assert not hand.snapshot().connected
    assert hand.runtime.joint_names == make_hand_joint_names("right")


def test_connect_keeps_the_default_servo_profile() -> None:
    hand = Hand.connect("/dev/serial/by-id/test-adapter")
    assert hand.runtime.profile_acceleration == DEFAULT_OPERATOR_PROFILE_ACCELERATION
    assert hand.runtime.profile_velocity == DEFAULT_OPERATOR_PROFILE_VELOCITY


def test_connect_writes_the_requested_servo_profile() -> None:
    hand = Hand.connect(
        "/dev/serial/by-id/test-adapter",
        profile_acceleration=0,
        profile_velocity=0,
    )
    # Swap the unopened serial controller for the fake bus to see what start writes.
    hand.runtime.controller = HandController.from_fake(motor_ids=GHAND_RIGHT_MOTOR_IDS)
    with hand:
        pass
    profile_calls = [
        call for call in hand.runtime.controller.bus.calls if call[0] == "profile"
    ]
    assert len(profile_calls) == 16
    assert all(call[2:] == (0, 0) for call in profile_calls)


def test_connect_rejects_a_negative_servo_profile() -> None:
    with pytest.raises(ValueError, match="profile_velocity"):
        Hand.connect("/dev/serial/by-id/test-adapter", profile_velocity=-1)
    with pytest.raises(ValueError, match="profile_acceleration"):
        Hand.connect("/dev/serial/by-id/test-adapter", profile_acceleration=-1)


class FakeViewer:
    def __init__(self) -> None:
        self.started = 0
        self.snapshots: list[object] = []

    def start(self) -> "FakeViewer":
        self.started += 1
        return self

    def log_runtime_snapshot(self, snapshot, *, timestamp_ns=None) -> None:
        self.snapshots.append(snapshot)


def test_viewer_receives_every_runtime_snapshot() -> None:
    hand, _ = make_hand()
    viewer = FakeViewer()
    hand.viewer = viewer
    with hand:
        hand.set_joint_positions({"index_mcp_flex": -0.2})
        hand.step()
        hand.joint_positions()
    assert viewer.started == 1
    # start, step, and the observe inside joint_positions each log one snapshot
    assert len(viewer.snapshots) == 3
    assert viewer.snapshots[-1].joints[0].joint_name == "right_hand.ring_mcp_side.pos"


def test_viewer_flag_requires_the_viewer_extra(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ghand.viewer":
            raise ImportError("rerun missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="viewer"):
        Hand.fake(viewer=True)
