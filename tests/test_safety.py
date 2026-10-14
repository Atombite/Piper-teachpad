# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import json
import time
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from piper_control.config import ArmConfig, ArmSide
from piper_control.controller import ArmController, StateConflict
from piper_control.drivers.piper import PiperArmDriver
from piper_control.drivers.simulated import SimulatedArmDriver
from piper_control.mailbox import LatestValue
from piper_control.types import ArmFeedback, GripperDiagnostics, ServiceState
from piper_control.vectors import GripperCommand, ServoCommand


ZERO6 = (0.0,) * 6


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _arm(side: ArmSide) -> dict[str, object]:
    return {
        "server": {"host": "127.0.0.1", "port": 8765 if side is ArmSide.RIGHT else 9876},
        "control_hz": 10,
        "feedback_timeout_s": 0.5,
        "shutdown_home_timeout_s": 1.0,
        "name": f"{side.value}_arm",
        "can_port": "can0" if side is ArmSide.RIGHT else "can1",
        "auto_initialize": False,
        "receive_latency_s": 0.0,
        "calibration_joints_pos_deg": [0, 0, 0, 0, 0, 0],
        "reset_joints_pos_deg": [0, 0, 0, 0, 0, 0],
        "min_joints_deg": [-150, 0, -170, -100, -70, -120],
        "max_joints_deg": [150, 180, 0, 100, 70, 120],
        "max_joints_speed_deg_s": [30, 30, 30, 30, 30, 30],
        "default_joints_speed_deg_s": [10, 10, 10, 10, 10, 10],
        "gripper": {
            "enabled": True,
            "min_position_m": 0.0,
            "max_position_m": 0.07,
            "max_speed_m_s": 0.05,
            "default_speed_m_s": 0.02,
            "effort_nm": 1.0,
        },
    }


def _config(side: ArmSide = ArmSide.RIGHT) -> ArmConfig:
    return ArmConfig.model_validate(_arm(side))


def _controller(
    *, side: ArmSide = ArmSide.RIGHT, auto_timestamp: bool = False
) -> tuple[ArmController, ArmConfig, dict[str, SimulatedArmDriver]]:
    config = _config(side)
    driver = SimulatedArmDriver(
        ArmFeedback(joints_deg=(5, 0, 0, 0, 0, 0), received_at_s=1.0),
        auto_timestamp=auto_timestamp,
    )
    drivers = {side.value: driver}
    controller = ArmController(
        side, config, driver, sleep=lambda _seconds: None
    )
    controller.tick(dt=0.1)
    return controller, config, drivers


class _RecoverySdk:
    """Small mutable SDK double used through the production adapter."""

    def __init__(self) -> None:
        self.joints = [5000, 0, 0, 0, 0, 0]
        self.emergency_modes: list[int] = []
        self.arm_status = 0
        self.enabled = True
        self.feedback_hz = 100.0
        self.feedback_timestamp = 1.0
        self.low_speed_timestamp = 1.0
        self.feedback_step_s = 0.1
        self.freeze_feedback_timestamp = False
        self.enable_calls = 0
        self.fail_reenable = False
        self.block_reenable = False
        self.reenable_entered = Event()
        self.release_reenable = Event()
        self.recovering = False

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        if name in {"DisableArm", "DisablePiper", "ResetPiper"}:
            raise AssertionError(f"forbidden SDK member accessed: {name}")
        raise AttributeError(name)

    def ConnectPort(self) -> None: pass
    def EnablePiper(self) -> bool:
        self.enable_calls += 1
        if self.fail_reenable and self.recovering:
            raise RuntimeError("re-enable failed")
        if self.block_reenable and self.recovering:
            self.reenable_entered.set()
            assert self.release_reenable.wait(1.0)
        self.enabled = True
        self.feedback_hz = 100.0
        if self.recovering and not self.freeze_feedback_timestamp:
            self.feedback_timestamp += 1.0
        return True
    def MotionCtrl_2(self, *_args) -> None: pass  # type: ignore[no-untyped-def]
    def JointCtrl(self, *joints) -> None:  # type: ignore[no-untyped-def]
        self.joints[:] = joints
    def GripperCtrl(self, *_args) -> None: pass  # type: ignore[no-untyped-def]
    def EmergencyStop(self, mode: int) -> None:
        self.emergency_modes.append(mode)
        self.arm_status = 1 if mode == 0x01 else 0
        if mode == 0x01:
            self.recovering = False
            self.enabled = False
            self.feedback_hz = 0.0
        elif mode == 0x02:
            self.recovering = True
    def get_connect_status(self) -> bool: return True
    def GetArmEnableStatus(self): return [self.enabled] * 6  # type: ignore[no-untyped-def]
    def GetArmLowSpdInfoMsgs(self):  # type: ignore[no-untyped-def]
        self.low_speed_timestamp += 0.1
        motors = {
            f"motor_{index}": SimpleNamespace(
                foc_status=SimpleNamespace(driver_enable_status=self.enabled)
            )
            for index in range(1, 7)
        }
        return SimpleNamespace(time_stamp=self.low_speed_timestamp, **motors)
    def GetArmJointMsgs(self):  # type: ignore[no-untyped-def]
        if not self.freeze_feedback_timestamp:
            self.feedback_timestamp += self.feedback_step_s
        state = SimpleNamespace(
            **{f"joint_{index}": value for index, value in enumerate(self.joints, 1)}
        )
        return SimpleNamespace(
            time_stamp=self.feedback_timestamp,
            Hz=self.feedback_hz,
            joint_state=state,
        )
    def GetArmGripperMsgs(self):  # type: ignore[no-untyped-def]
        state = SimpleNamespace(grippers_angle=0)
        return SimpleNamespace(
            time_stamp=self.feedback_timestamp,
            Hz=self.feedback_hz,
            gripper_state=state,
        )
    def GetArmEndPoseMsgs(self):  # type: ignore[no-untyped-def]
        pose = SimpleNamespace(
            X_axis=0, Y_axis=0, Z_axis=0,
            RX_axis=0, RY_axis=0, RZ_axis=0,
        )
        return SimpleNamespace(
            time_stamp=self.feedback_timestamp,
            Hz=self.feedback_hz,
            end_pose=pose,
        )
    def GetArmStatus(self):  # type: ignore[no-untyped-def]
        status = SimpleNamespace(err_code=0, arm_status=self.arm_status)
        return SimpleNamespace(
            time_stamp=self.feedback_timestamp,
            Hz=self.feedback_hz,
            arm_status=status,
        )


def _production_controller(side: ArmSide = ArmSide.RIGHT):  # type: ignore[no-untyped-def]
    config = _config(side)
    sdk = _RecoverySdk()
    sdks = {side.value: sdk}
    driver = PiperArmDriver(config, sdk=sdk)
    controller = ArmController(
        side, config, driver, sleep=lambda _seconds: None
    )
    controller.tick(dt=0.1)
    return controller, sdks


def test_emergency_on_left_controller_does_not_touch_right_driver() -> None:
    left, _, left_drivers = _controller(side=ArmSide.LEFT)
    right, _, right_drivers = _controller()
    left_drivers["left"].set_fault(7)

    left.tick(dt=0.1)
    left.tick(dt=0.1)

    snapshot = left.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert snapshot.fault_reason == "left arm fault code 7"
    assert left_drivers["left"].emergency_stop_calls == 1
    assert right_drivers["right"].emergency_stop_calls == 0
    assert right.snapshot().state is not ServiceState.EMERGENCY_STOP
    assert left_drivers["left"].disable_calls == left_drivers["left"].reset_calls == 0


def test_publish_runtime_fault_immediately_exposes_terminal_snapshot() -> None:
    left, _, left_drivers = _controller(side=ArmSide.LEFT)
    right, _, right_drivers = _controller()

    left.publish_runtime_fault("control worker failed: unexpected SDK crash")

    snapshot = left.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert snapshot.emergency_stopped is True
    assert snapshot.fault_reason == "control worker failed: unexpected SDK crash"
    assert snapshot.hold is True
    assert snapshot.moving is False
    assert right.snapshot().state is not ServiceState.EMERGENCY_STOP
    assert right_drivers["right"].emergency_stop_calls == 0
    deadline = time.monotonic() + 0.5
    while left_drivers["left"].emergency_stop_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert left_drivers["left"].emergency_stop_calls == 1


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"stale": True, "received_at_s": 2.0}, "feedback stale"),
        ({"enabled": False, "received_at_s": 2.0}, "arm disabled"),
        (
            {
                "joints_deg": (float("nan"), 0, 0, 0, 0, 0),
                "received_at_s": 2.0,
            },
            "feedback is non-finite",
        ),
    ],
)
def test_unhealthy_feedback_stops_only_its_controller(
    changes: dict[str, object], reason: str
) -> None:
    controller, _, drivers = _controller(side=ArmSide.LEFT)
    current = drivers["left"].read_feedback()
    drivers["left"].set_feedback(replace(current, **changes))

    controller.tick(dt=0.1)

    assert reason in (controller.snapshot().fault_reason or "")
    assert drivers["left"].emergency_stop_calls == 1


def test_fault_state_is_published_before_blocking_hardware_emergency_stop() -> None:
    entered_stop = Event()
    release_stop = Event()

    class BlockingEmergencyDriver(SimulatedArmDriver):
        def emergency_stop(self) -> None:
            entered_stop.set()
            assert release_stop.wait(1.0)
            super().emergency_stop()

    config = _config(ArmSide.LEFT)
    driver = BlockingEmergencyDriver(ArmFeedback(received_at_s=1.0))
    controller = ArmController(ArmSide.LEFT, config, driver)
    controller.tick(dt=0.1)
    current = driver.read_feedback()
    driver.set_feedback(replace(current, enabled=False, received_at_s=2.0))

    fault_tick = Thread(target=controller.tick, kwargs={"dt": 0.1})
    fault_tick.start()
    try:
        assert entered_stop.wait(1.0)
        snapshot = controller.snapshot()
        assert snapshot.state is ServiceState.EMERGENCY_STOP
        assert snapshot.emergency_stopped is True
        assert snapshot.hold is True
        assert snapshot.moving is False
        assert snapshot.fault_reason == "left arm disabled"
        assert fault_tick.is_alive()
    finally:
        release_stop.set()
        fault_tick.join(1.0)
    assert not fault_tick.is_alive()


@pytest.mark.parametrize("joint_index", range(6))
@pytest.mark.parametrize("bound", ["minimum", "maximum"])
def test_out_of_range_joint_feedback_stops_only_its_controller(
    joint_index: int, bound: str
) -> None:
    left, config, left_drivers = _controller(side=ArmSide.LEFT)
    right, _, right_drivers = _controller()
    current = left_drivers["left"].read_feedback()
    joints = list(current.joints_deg)
    if bound == "minimum":
        joints[joint_index] = config.min_joints_deg[joint_index] - 2.0001
    else:
        joints[joint_index] = config.max_joints_deg[joint_index] + 2.0001
    left_drivers["left"].set_feedback(
        replace(current, joints_deg=tuple(joints), received_at_s=2.0)
    )

    left.tick(dt=0.1)

    snapshot = left.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert f"joint {joint_index + 1} feedback" in (snapshot.fault_reason or "")
    assert left_drivers["left"].emergency_stop_calls == 1
    assert right.snapshot().state is not ServiceState.EMERGENCY_STOP
    assert right_drivers["right"].emergency_stop_calls == 0


@pytest.mark.parametrize(
    ("bound", "offset"),
    [("minimum", -2.0), ("maximum", 2.0)],
)
def test_joint_feedback_boundary_allows_two_degree_tolerance(
    bound: str, offset: float,
) -> None:
    controller, config, drivers = _controller()
    current = drivers["right"].read_feedback()
    joints = list(current.joints_deg)
    boundary = (
        config.min_joints_deg[0]
        if bound == "minimum"
        else config.max_joints_deg[0]
    )
    joints[0] = boundary + offset
    drivers["right"].set_feedback(
        replace(current, joints_deg=tuple(joints), received_at_s=2.0)
    )

    controller.tick(dt=0.1)

    assert controller.snapshot().state is not ServiceState.EMERGENCY_STOP


def test_out_of_range_enabled_gripper_feedback_stops_controller() -> None:
    left, config, left_drivers = _controller(side=ArmSide.LEFT)
    right, _, right_drivers = _controller()
    current = left_drivers["left"].read_feedback()
    left_drivers["left"].set_feedback(
        replace(
            current,
            gripper_position=config.gripper.max_position_m + 0.0000011,
            received_at_s=2.0,
        )
    )

    left.tick(dt=0.1)

    assert left.snapshot().state is ServiceState.EMERGENCY_STOP
    assert "gripper feedback" in (left.snapshot().fault_reason or "")
    assert left_drivers["left"].emergency_stop_calls == 1
    assert right.snapshot().state is not ServiceState.EMERGENCY_STOP
    assert right_drivers["right"].emergency_stop_calls == 0


def test_gripper_feedback_boundary_allows_one_micrometre_tolerance() -> None:
    controller, config, drivers = _controller()
    current = drivers["right"].read_feedback()
    drivers["right"].set_feedback(
        replace(
            current,
            gripper_position=config.gripper.max_position_m + 0.000001,
            received_at_s=2.0,
        )
    )

    controller.tick(dt=0.1)

    assert controller.snapshot().state is not ServiceState.EMERGENCY_STOP


def test_disabled_gripper_feedback_range_is_not_enforced() -> None:
    arm = _arm(ArmSide.RIGHT)
    arm["gripper"] = {**arm["gripper"], "enabled": False}  # type: ignore[arg-type]
    config = ArmConfig.model_validate(arm)
    driver = SimulatedArmDriver(
        ArmFeedback(gripper_position=1.0, received_at_s=1.0)
    )
    controller = ArmController(ArmSide.RIGHT, config, driver)

    controller.tick(dt=0.1)

    assert controller.snapshot().state is not ServiceState.EMERGENCY_STOP


def test_disabled_piper_arm_is_not_misreported_as_disabled_gripper() -> None:
    controller, _, drivers = _controller()
    current = drivers["right"].read_feedback()
    drivers["right"].set_feedback(
        replace(
            current,
            enabled=False,
            gripper=GripperDiagnostics(
                backend="piper",
                enabled=False,
                communication_ok=True,
            ),
            received_at_s=2.0,
        )
    )

    controller.tick(dt=0.1)

    assert controller.snapshot().fault_reason == "right arm disabled"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"tcp_pose": (float("nan"), 0, 0, 0, 0, 0)}, "non-finite"),
        ({"received_at_s": float("inf")}, "non-finite"),
        ({"robot_timestamp_s": float("nan")}, "non-finite"),
    ],
)
def test_nonfinite_metadata_preserves_complete_safe_snapshot_and_hold_sample(
    changes: dict[str, object], reason: str
) -> None:
    controller, _, drivers = _controller()
    valid = drivers["right"].read_feedback()
    drivers["right"].set_feedback(replace(valid, **changes))

    controller.tick(dt=0.1)
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert reason in (snapshot.fault_reason or "")
    assert snapshot.actual_tcp_pose == valid.tcp_pose
    assert snapshot.robot_receive_timestamp == valid.received_at_s
    assert snapshot.robot_timestamp == (
        valid.robot_timestamp_s
        if valid.robot_timestamp_s is not None
        else valid.received_at_s
    )
    json.dumps(snapshot.to_json(), allow_nan=False)
    assert drivers["right"].joint_commands[-1] == valid.joints_deg


def test_nonfinite_simulation_timestamp_preserves_last_safe_simulation_time() -> None:
    controller, _, drivers = _controller()
    valid = replace(
        drivers["right"].read_feedback(),
        received_at_s=2.0,
        robot_timestamp_s=0.75,
    )
    drivers["right"].set_feedback(valid)
    controller.tick(dt=0.1)
    drivers["right"].set_feedback(
        replace(valid, received_at_s=3.0, robot_timestamp_s=float("nan"))
    )

    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert snapshot.robot_timestamp == 0.75
    assert snapshot.arm.robot_timestamp == 0.75


def test_manual_emergency_is_idempotent_and_preserves_first_cause() -> None:
    controller, _, drivers = _controller()

    controller.emergency_stop("operator button")
    controller.emergency_stop("later reason")
    controller.tick(dt=0.1)

    assert controller.snapshot().fault_reason == "operator button"
    assert drivers["right"].emergency_stop_calls == 1


def test_starting_emergency_connects_only_and_retries_after_connect() -> None:
    right_connected = Event()
    release_right_connect = Event()

    class BlockingConnectDriver(SimulatedArmDriver):
        def connect(self) -> None:
            super().connect()
            right_connected.set()
            assert release_right_connect.wait(1.0)

    config = _config()
    right = BlockingConnectDriver(ArmFeedback(received_at_s=1.0))
    controller = ArmController(ArmSide.RIGHT, config, right)
    tick_thread = Thread(target=controller.tick, kwargs={"dt": 0.1})
    tick_thread.start()
    assert right_connected.wait(1.0)

    stop_thread = Thread(target=controller.emergency_stop, args=("operator",))
    stop_thread.start()
    release_right_connect.set()
    tick_thread.join(1.0)
    stop_thread.join(1.0)

    assert not tick_thread.is_alive() and not stop_thread.is_alive()
    # The stop sent while connect() was still in flight is deliberately retried
    # after connect returns, because connect may reset the device-side state.
    assert right.emergency_stop_calls == 2
    assert right.enable_calls == 0
    assert not right.joint_commands


def test_tick_retries_failed_emergency_delivery() -> None:
    class FailFirstStopDriver(SimulatedArmDriver):
        def __init__(self, feedback: ArmFeedback) -> None:
            super().__init__(feedback)
            self.stop_attempts = 0

        def emergency_stop(self) -> None:
            self.stop_attempts += 1
            if self.stop_attempts == 1:
                raise RuntimeError("transient stop failure")
            super().emergency_stop()

    config = _config(ArmSide.LEFT)
    left = FailFirstStopDriver(ArmFeedback(received_at_s=1.0))
    controller = ArmController(ArmSide.LEFT, config, left)
    controller.tick(dt=0.1)

    controller.emergency_stop("operator")
    assert left.emergency_stop_calls == 0
    controller.tick(dt=0.1)

    assert left.emergency_stop_calls == 1
    assert left.stop_attempts == 2


def test_emergency_immediately_rejects_commands_before_state_publication() -> None:
    controller, config, _ = _controller()
    controller.request_homing()
    for _ in range(8):
        controller.tick(dt=0.1)
    assert controller.snapshot().state is ServiceState.RUNNING

    controller.emergency_stop("operator")

    assert controller.snapshot().state is ServiceState.RUNNING
    with pytest.raises(StateConflict, match="emergency"):
        controller.submit_servo(ServoCommand(ZERO6, (10.0,) * 6))
    with pytest.raises(StateConflict, match="emergency"):
        controller.submit_gripper(GripperCommand(0.0, 0.02))


def test_emergency_immediately_rejects_homing_before_state_publication() -> None:
    controller, _, _ = _controller()
    assert controller.snapshot().state is ServiceState.RUNNING

    controller.emergency_stop("operator")

    with pytest.raises(StateConflict, match="emergency"):
        controller.request_homing()


def test_emergency_clears_command_that_won_concurrent_mailbox_race() -> None:
    put_entered = Event()
    release_put = Event()

    class BlockingMailbox(LatestValue):
        def put(self, value):  # type: ignore[no-untyped-def]
            put_entered.set()
            assert release_put.wait(1.0)
            super().put(value)

    controller, config, _ = _controller()
    controller.request_homing()
    for _ in range(8):
        controller.tick(dt=0.1)
    mailbox = BlockingMailbox()
    controller._servo_mailbox = mailbox  # type: ignore[attr-defined]
    command = ServoCommand((20.0, 0, 0, 0, 0, 0), config.default_joints_speed_deg_s)
    submit = Thread(target=controller.submit_servo, args=(command,))
    submit.start()
    assert put_entered.wait(1.0)

    stopped = Event()
    stop = Thread(
        target=lambda: (controller.emergency_stop("operator"), stopped.set())
    )
    stop.start()
    release_put.set()
    submit.join(1.0)
    stop.join(1.0)

    assert stopped.is_set()
    assert mailbox.take() is None


def test_manual_emergency_does_not_wait_for_controller_state_lock() -> None:
    controller, _, drivers = _controller()
    state_locked = Event()
    release_state = Event()
    emergency_returned = Event()

    def hold_state_lock() -> None:
        with controller._state_lock:  # type: ignore[attr-defined]
            state_locked.set()
            assert release_state.wait(1.0)

    locker = Thread(target=hold_state_lock)
    locker.start()
    assert state_locked.wait(1.0)

    stopper = Thread(
        target=lambda: (
            controller.emergency_stop("operator button"),
            emergency_returned.set(),
        )
    )
    stopper.start()
    returned_while_state_locked = emergency_returned.wait(0.2)
    release_state.set()
    locker.join(1.0)
    stopper.join(1.0)

    assert returned_while_state_locked
    assert drivers["right"].emergency_stop_calls == 1


def test_emergency_after_joint_sdk_call_prevents_later_tick_sends() -> None:
    joint_call_entered = Event()
    release_joint_call = Event()
    emergency_completed = Event()

    class CoordinatedDriver(SimulatedArmDriver):
        block_next_joint = False

        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            with self._lock:  # type: ignore[attr-defined]
                super().command_joints(joints_deg)
                if self.block_next_joint:
                    self.block_next_joint = False
                    joint_call_entered.set()
                    assert release_joint_call.wait(1.0)
            if joint_call_entered.is_set():
                assert emergency_completed.wait(1.0)

        def emergency_stop(self) -> None:
            super().emergency_stop()
            emergency_completed.set()

    config = _config()
    right = CoordinatedDriver(ArmFeedback(received_at_s=1.0))
    controller = ArmController(ArmSide.RIGHT, config, right)
    controller.tick(dt=0.1)
    controller.request_homing()
    for _ in range(3):
        controller.tick(dt=0.1)
    right.block_next_joint = True
    right_grippers_before = len(right.gripper_commands)

    tick_thread = Thread(target=controller.tick, kwargs={"dt": 0.1})
    tick_thread.start()
    assert joint_call_entered.wait(1.0)
    stop_thread = Thread(target=controller.emergency_stop, args=("operator button",))
    stop_thread.start()
    release_joint_call.set()
    tick_thread.join(1.0)
    stop_thread.join(1.0)

    assert not tick_thread.is_alive()
    assert len(right.gripper_commands) == right_grippers_before


def test_production_driver_recovery_rearms_and_returns_to_running() -> None:
    controller, sdks = _production_controller()

    controller.emergency_stop("operator")
    controller.tick(dt=0.1)
    controller.request_recovery()
    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.RUNNING
    assert sdks["right"].emergency_modes == [0x01, 0x02]
    assert sdks["right"].enable_calls == 6


@pytest.mark.parametrize("joint_2_mdeg", [-1424, -2000])
def test_production_recovery_allows_two_degree_joint_boundary_overshoot(
    joint_2_mdeg: int,
) -> None:
    controller, sdks = _production_controller()
    sdks["right"].joints[1] = joint_2_mdeg

    controller.emergency_stop("operator")
    controller.tick(dt=0.1)
    controller.request_recovery()
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.RUNNING
    assert snapshot.actual_q[1] == pytest.approx(joint_2_mdeg / 1000.0)
    assert sdks["right"].emergency_modes == [0x01, 0x02]


def test_running_projects_tolerated_feedback_back_inside_command_limits() -> None:
    controller, sdks = _production_controller()
    controller.request_homing()
    for _ in range(8):
        controller.tick(dt=0.1)
    assert controller.snapshot().state is ServiceState.RUNNING

    sdks["right"].joints[1] = -1000
    controller.tick(dt=0.01)

    assert controller.snapshot().state is ServiceState.RUNNING
    assert sdks["right"].joints[1] == 0
    assert sdks["right"].emergency_modes == []


def test_shutdown_projects_tolerated_feedback_back_inside_command_limits() -> None:
    controller, sdks = _production_controller()
    controller.request_homing()
    for _ in range(8):
        controller.tick(dt=0.1)
    assert controller.snapshot().state is ServiceState.RUNNING

    sdks["right"].joints[1] = -1000
    controller.request_shutdown()
    controller.tick(dt=0.01)

    assert controller.snapshot().state is ServiceState.SHUTTING_DOWN
    assert sdks["right"].joints[1] == 0
    assert sdks["right"].emergency_modes == []


def test_production_recovery_rejects_joint_boundary_overshoot_beyond_tolerance() -> None:
    controller, sdks = _production_controller()
    sdks["right"].joints[1] = -2001

    controller.emergency_stop("operator")
    controller.tick(dt=0.1)
    controller.request_recovery()
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert "joint 2 feedback -2.001" in (snapshot.fault_reason or "")
    assert sdks["right"].emergency_modes == [0x01, 0x01]


def test_production_recovery_rejects_feedback_outside_configured_limits() -> None:
    controller, sdks = _production_controller(ArmSide.LEFT)
    sdks["left"].joints = [-3037, -2848, -64, 577, -676, -9077]

    controller.emergency_stop("operator")
    controller.tick(dt=0.1)
    controller.request_recovery()
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert "joint 2 feedback" in (snapshot.fault_reason or "")
    assert sdks["left"].joints == [-3037, -2848, -64, 577, -676, -9077]


def test_production_recovery_rejects_large_out_of_range_feedback_before_hold() -> None:
    controller, sdks = _production_controller()
    sdks["right"].joints[1] = -10001

    controller.emergency_stop("operator")
    controller.tick(dt=0.1)
    controller.request_recovery()
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert "recovery rejected" in (snapshot.fault_reason or "")
    assert "joint 2" in (snapshot.fault_reason or "")
    assert sdks["right"].joints[1] == -10001


def test_production_recovery_requires_feedback_to_advance_after_reenable() -> None:
    controller, sdks = _production_controller()
    sdks["right"].freeze_feedback_timestamp = True

    controller.emergency_stop("operator")
    controller.tick(dt=0.1)
    controller.request_recovery()
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert "feedback did not advance" in (snapshot.fault_reason or "")


def test_production_driver_detects_external_emergency_stop() -> None:
    controller, sdks = _production_controller()
    sdks["right"].arm_status = 1

    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert controller.snapshot().fault_reason == "right arm fault code 1"
    assert sdks["right"].emergency_modes == [0x01]


def test_production_rearm_failure_restops_controller_arm() -> None:
    controller, sdks = _production_controller(ArmSide.LEFT)
    sdks["left"].fail_reenable = True

    controller.emergency_stop("operator")
    controller.tick(dt=0.1)
    controller.request_recovery()
    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert sdks["left"].emergency_modes == [0x01, 0x02, 0x01]


def test_new_emergency_during_production_rearm_restops_controller_arm() -> None:
    controller, sdks = _production_controller()
    sdks["right"].block_reenable = True
    controller.emergency_stop("operator")
    controller.tick(dt=0.1)
    controller.request_recovery()

    recovery = Thread(target=controller.tick, kwargs={"dt": 0.1})
    recovery.start()
    assert sdks["right"].reenable_entered.wait(1.0)
    new_stop = Thread(target=controller.emergency_stop, args=("new stop",))
    new_stop.start()
    sdks["right"].release_reenable.set()
    recovery.join(1.0)
    new_stop.join(1.0)

    assert not recovery.is_alive() and not new_stop.is_alive()
    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert sdks["right"].emergency_modes[-1] == 0x01


def test_recovery_discards_old_commands_and_targets_current_feedback() -> None:
    config = _config()
    clock = ManualClock()
    clock.now = 1.0
    driver = SimulatedArmDriver(
        ArmFeedback(joints_deg=(5, 0, 0, 0, 0, 0), received_at_s=1.0),
        auto_timestamp=True,
        clock=clock,
    )
    drivers = {"right": driver}
    controller = ArmController(
        ArmSide.RIGHT,
        config,
        driver,
        clock=clock,
        sleep=clock.advance,
    )
    controller.tick(dt=0.1)
    controller.request_homing()
    for _ in range(8):
        controller.tick(dt=0.1)
    controller.submit_servo(
        ServoCommand((20.0, 0, 0, 0, 0, 0), config.default_joints_speed_deg_s)
    )
    controller.emergency_stop("operator button")
    controller.tick(dt=0.1)
    drivers["right"].set_feedback(
        ArmFeedback(
            joints_deg=(3, 0, 0, 0, 0, 0),
            connected=True,
            enabled=True,
            received_at_s=2.0,
        )
    )
    clock.now = 2.0

    controller.request_recovery()
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.RUNNING
    assert snapshot.target_q == snapshot.actual_q
    assert drivers["right"].recover_emergency_calls == 1
    controller.submit_servo(
        ServoCommand((10.0, 0, 0, 0, 0, 0), config.default_joints_speed_deg_s)
    )


def test_recovery_requires_three_consecutive_healthy_advancing_frames() -> None:
    recovery_feedback = [
        ArmFeedback(connected=True, enabled=True, received_at_s=2.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=3.0, stale=True),
        ArmFeedback(connected=True, enabled=True, received_at_s=4.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=5.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=6.0),
    ]

    class JitteringRecoveryDriver(SimulatedArmDriver):
        def __init__(self) -> None:
            super().__init__(ArmFeedback(received_at_s=1.0))
            self.recovery_reads = 0
            self._released = False
            self._recovering = False
            self._release_timestamp = 1.0

        def recover_emergency(self) -> None:
            super().recover_emergency()
            self._released = True

        def enable(self) -> None:
            super().enable()
            if self.enable_calls > 1:
                self._recovering = True

        def read_feedback(self) -> ArmFeedback:
            if self._recovering and recovery_feedback:
                self.recovery_reads += 1
                feedback = recovery_feedback.pop(0)
                self.set_feedback(feedback)
                return feedback
            if self._released and not self._recovering:
                self._release_timestamp += 0.2
                feedback = replace(
                    super().read_feedback(),
                    received_at_s=self._release_timestamp,
                )
                self.set_feedback(feedback)
                return feedback
            return super().read_feedback()

    config = _config(ArmSide.LEFT)
    driver = JitteringRecoveryDriver()
    controller = ArmController(
        ArmSide.LEFT, config, driver, sleep=lambda _seconds: None
    )
    controller.tick(dt=0.1)
    controller.emergency_stop("operator")
    controller.tick(dt=0.1)

    controller.request_recovery()
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.RUNNING
    assert driver.recovery_reads == 5
    assert recovery_feedback == []
    assert snapshot.robot_receive_timestamp == 6.0


def test_recovery_requires_configured_continuous_feedback_windows() -> None:
    config_data = _arm(ArmSide.RIGHT)
    config_data["control_hz"] = 100
    config = ArmConfig.model_validate(config_data)
    clock = ManualClock()
    clock.now = 1.0
    driver = SimulatedArmDriver(
        ArmFeedback(received_at_s=1.0),
        auto_timestamp=True,
        clock=clock,
    )
    controller = ArmController(
        ArmSide.RIGHT,
        config,
        driver,
        clock=clock,
        sleep=clock.advance,
    )
    controller.tick(dt=0.01)
    controller.emergency_stop("operator")
    controller.tick(dt=0.01)

    controller.request_recovery()
    controller.tick(dt=0.01)

    assert controller.snapshot().state is ServiceState.RUNNING
    assert clock.now >= 0.8


def test_recovery_waits_for_release_feedback_before_reenable() -> None:
    config = _config(ArmSide.RIGHT)
    clock = ManualClock()
    clock.now = 1.0

    class TimedRecoveryDriver(SimulatedArmDriver):
        def __init__(self) -> None:
            super().__init__(
                ArmFeedback(received_at_s=1.0),
                auto_timestamp=True,
                clock=clock,
            )
            self.release_at: float | None = None
            self.reenable_at: float | None = None

        def recover_emergency(self) -> None:
            super().recover_emergency()
            self.release_at = clock()

        def enable(self) -> None:
            super().enable()
            if self.enable_calls > 1:
                self.reenable_at = clock()

    driver = TimedRecoveryDriver()
    controller = ArmController(
        ArmSide.RIGHT,
        config,
        driver,
        clock=clock,
        sleep=clock.advance,
    )
    controller.tick(dt=0.1)
    controller.emergency_stop("operator")
    controller.tick(dt=0.1)

    controller.request_recovery()
    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.RUNNING
    assert driver.release_at is not None
    assert driver.reenable_at is not None
    assert driver.reenable_at - driver.release_at >= 0.3


def test_recovery_waits_through_transient_fault_after_emergency_release() -> None:
    release_feedback = [
        ArmFeedback(connected=True, fault_code=63, received_at_s=2.0),
        ArmFeedback(connected=True, received_at_s=3.0),
        ArmFeedback(connected=True, received_at_s=4.0),
        ArmFeedback(connected=True, received_at_s=5.0),
    ]
    enabled_feedback = [
        ArmFeedback(connected=True, enabled=True, received_at_s=6.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=7.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=8.0),
    ]

    class TransientReleaseFaultDriver(SimulatedArmDriver):
        def __init__(self) -> None:
            super().__init__(ArmFeedback(received_at_s=1.0))
            self._phase = "stopped"

        def recover_emergency(self) -> None:
            super().recover_emergency()
            self._phase = "released"

        def enable(self) -> None:
            super().enable()
            if self.enable_calls > 1:
                self._phase = "enabled"

        def read_feedback(self) -> ArmFeedback:
            queue = (
                release_feedback
                if self._phase == "released"
                else enabled_feedback
                if self._phase == "enabled"
                else []
            )
            if queue:
                feedback = queue.pop(0)
                self.set_feedback(feedback)
                return feedback
            return super().read_feedback()

    config = _config(ArmSide.RIGHT)
    driver = TransientReleaseFaultDriver()
    controller = ArmController(
        ArmSide.RIGHT,
        config,
        driver,
        sleep=lambda _seconds: None,
    )
    controller.tick(dt=0.1)
    controller.emergency_stop("operator")
    controller.tick(dt=0.1)

    controller.request_recovery()
    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.RUNNING
    assert release_feedback == []
    assert enabled_feedback == []
    assert driver.enable_calls == 2


def test_recovery_observes_enabled_feedback_for_half_second() -> None:
    config = _config(ArmSide.RIGHT)
    clock = ManualClock()
    clock.now = 1.0

    class TimedRecoveryDriver(SimulatedArmDriver):
        def __init__(self) -> None:
            super().__init__(
                ArmFeedback(received_at_s=1.0),
                auto_timestamp=True,
                clock=clock,
            )
            self.reenable_at: float | None = None

        def enable(self) -> None:
            super().enable()
            if self.enable_calls > 1:
                self.reenable_at = clock()

    driver = TimedRecoveryDriver()
    controller = ArmController(
        ArmSide.RIGHT,
        config,
        driver,
        clock=clock,
        sleep=clock.advance,
    )
    controller.tick(dt=0.1)
    controller.emergency_stop("operator")
    controller.tick(dt=0.1)

    controller.request_recovery()
    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.RUNNING
    assert driver.reenable_at is not None
    assert clock() - driver.reenable_at >= 0.5


def test_recovery_elapsed_time_does_not_trigger_control_overrun() -> None:
    config = _config(ArmSide.RIGHT)
    clock = ManualClock()
    clock.now = 1.0
    driver = SimulatedArmDriver(
        ArmFeedback(received_at_s=1.0),
        auto_timestamp=True,
        clock=clock,
    )
    controller = ArmController(
        ArmSide.RIGHT,
        config,
        driver,
        clock=clock,
        sleep=clock.advance,
    )
    controller.tick(dt=0.1)
    controller.emergency_stop("operator")
    controller.tick(dt=0.1)
    controller.request_recovery()
    controller.tick(dt=0.1)
    assert controller.snapshot().state is ServiceState.RUNNING

    controller.tick(dt=3.0)

    assert controller.snapshot().state is ServiceState.RUNNING
    assert driver.emergency_stop_calls == 1


def test_recovery_allows_two_seconds_for_enabled_feedback_to_settle() -> None:
    config = _config(ArmSide.LEFT)
    clock = ManualClock()
    clock.now = 1.0

    class SlowlyEnabledDriver(SimulatedArmDriver):
        def __init__(self) -> None:
            super().__init__(
                ArmFeedback(received_at_s=1.0),
                auto_timestamp=True,
                clock=clock,
            )
            self.reenable_at: float | None = None

        def enable(self) -> None:
            super().enable()
            if self.enable_calls > 1:
                self.reenable_at = clock()

        def read_feedback(self) -> ArmFeedback:
            feedback = super().read_feedback()
            if self.reenable_at is None:
                return feedback
            return replace(
                feedback,
                enabled=clock() - self.reenable_at >= 1.2,
            )

    driver = SlowlyEnabledDriver()
    controller = ArmController(
        ArmSide.LEFT,
        config,
        driver,
        clock=clock,
        sleep=clock.advance,
    )
    controller.tick(dt=0.1)
    controller.emergency_stop("operator")
    controller.tick(dt=0.1)

    controller.request_recovery()
    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.RUNNING
    assert driver.reenable_at is not None
    assert clock() - driver.reenable_at >= 1.7


def test_recovery_rejects_healthy_feedback_returned_after_deadline() -> None:
    config_data = _arm(ArmSide.RIGHT)
    config_data["control_hz"] = 100
    config = ArmConfig.model_validate(config_data)
    clock = ManualClock()

    class LateRecoveryDriver(SimulatedArmDriver):
        def __init__(self) -> None:
            super().__init__(ArmFeedback(received_at_s=1.0))
            self._released = False
            self._post_enable = False
            self._timestamp = 1.0
            self._delayed = False

        def recover_emergency(self) -> None:
            super().recover_emergency()
            self._released = True

        def enable(self) -> None:
            super().enable()
            if self.enable_calls > 1:
                self._post_enable = True

        def read_feedback(self) -> ArmFeedback:
            if self._released and not self._post_enable:
                self._timestamp += 1.0
                feedback = replace(
                    super().read_feedback(),
                    received_at_s=self._timestamp,
                )
                self.set_feedback(feedback)
                return feedback
            if not self._post_enable:
                return super().read_feedback()
            if not self._delayed:
                self._delayed = True
                clock.advance(config.feedback_timeout_s + 0.01)
            self._timestamp += 1.0
            feedback = ArmFeedback(
                connected=True,
                enabled=True,
                received_at_s=self._timestamp,
            )
            self.set_feedback(feedback)
            return feedback

    driver = LateRecoveryDriver()
    controller = ArmController(
        ArmSide.RIGHT,
        config,
        driver,
        clock=clock,
        sleep=clock.advance,
    )
    controller.tick(dt=0.01)
    controller.emergency_stop("operator")
    controller.tick(dt=0.01)

    controller.request_recovery()
    controller.tick(dt=0.01)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert "deadline" in (snapshot.fault_reason or "")


def test_recovery_waits_through_transient_input_fault_before_rearming() -> None:
    recovery_feedback = [
        ArmFeedback(connected=True, enabled=True, fault_code=63, received_at_s=2.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=3.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=4.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=5.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=6.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=7.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=8.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=9.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=10.0),
        ArmFeedback(connected=True, enabled=True, received_at_s=11.0),
    ]

    class TransientFaultRecoveryDriver(SimulatedArmDriver):
        def __init__(self) -> None:
            super().__init__(ArmFeedback(received_at_s=1.0))
            self.inject_recovery_feedback = False
            self.recovery_reads = 0

        def read_feedback(self) -> ArmFeedback:
            if self.inject_recovery_feedback and recovery_feedback:
                self.recovery_reads += 1
                feedback = recovery_feedback.pop(0)
                self.set_feedback(feedback)
                return feedback
            return super().read_feedback()

    config = _config(ArmSide.RIGHT)
    driver = TransientFaultRecoveryDriver()
    controller = ArmController(
        ArmSide.RIGHT, config, driver, sleep=lambda _seconds: None
    )
    controller.tick(dt=0.1)
    controller.emergency_stop("operator")
    controller.tick(dt=0.1)
    driver.inject_recovery_feedback = True

    controller.request_recovery()
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.RUNNING
    assert snapshot.robot_receive_timestamp == 11.0
    assert driver.recovery_reads == 10
    assert driver.recover_emergency_calls == 1


def test_recovery_with_persistently_stale_feedback_restops_and_requires_new_request() -> None:
    config = _config()
    clock = ManualClock()
    clock.now = 1.0
    driver = SimulatedArmDriver(
        ArmFeedback(joints_deg=(5, 0, 0, 0, 0, 0), received_at_s=1.0),
        auto_timestamp=True,
        clock=clock,
    )
    controller = ArmController(
        ArmSide.RIGHT,
        config,
        driver,
        clock=clock,
        sleep=clock.advance,
    )
    controller.tick(dt=0.1)
    controller.emergency_stop("operator button")
    controller.tick(dt=0.1)
    unhealthy = driver.read_feedback()
    driver.set_feedback(replace(unhealthy, stale=True, received_at_s=2.0))
    clock.now = 2.0

    controller.request_recovery()
    controller.tick(dt=0.1)
    driver.set_feedback(replace(unhealthy, stale=False, received_at_s=3.0))
    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert driver.recover_emergency_calls == 1
    assert driver.emergency_stop_calls == 2
    assert "post-recovery health check failed" in (
        controller.snapshot().fault_reason or ""
    )


def test_recovery_rejects_out_of_range_feedback_and_restops_controller() -> None:
    controller, config, drivers = _controller(auto_timestamp=True)
    controller.emergency_stop("operator button")
    controller.tick(dt=0.1)
    current = drivers["right"].read_feedback()
    joints = list(current.joints_deg)
    joints[4] = config.max_joints_deg[4] + 2.001
    drivers["right"].set_feedback(
        replace(current, joints_deg=tuple(joints), received_at_s=2.0)
    )

    controller.request_recovery()
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert "recovery rejected" in (snapshot.fault_reason or "")
    assert "joint 5 feedback" in (snapshot.fault_reason or "")
    assert drivers["right"].recover_emergency_calls == 0


def test_recovery_rejects_out_of_range_gripper_feedback() -> None:
    controller, config, drivers = _controller(auto_timestamp=True)
    controller.emergency_stop("operator button")
    controller.tick(dt=0.1)
    current = drivers["right"].read_feedback()
    drivers["right"].set_feedback(
        replace(
            current,
            gripper_position=config.gripper.min_position_m - 0.001,
            received_at_s=2.0,
        )
    )

    controller.request_recovery()
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert "recovery rejected" in (snapshot.fault_reason or "")
    assert "gripper feedback" in (snapshot.fault_reason or "")
    assert drivers["right"].recover_emergency_calls == 0


def test_recovery_failure_restops_controller_arm() -> None:
    class FailingRecoverDriver(SimulatedArmDriver):
        def recover_emergency(self) -> None:
            super().recover_emergency()
            raise RuntimeError("recover failed")

    config = _config(ArmSide.LEFT)
    left = FailingRecoverDriver(ArmFeedback(received_at_s=1.0))
    controller = ArmController(ArmSide.LEFT, config, left)
    controller.tick(dt=0.1)
    controller.emergency_stop("operator")
    controller.tick(dt=0.1)

    controller.request_recovery()
    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert left.emergency_stop_calls == 2
    with pytest.raises(RuntimeError, match="emergency"):
        left.command_joints((1, 0, 0, 0, 0, 0))


def test_new_emergency_during_recovery_aborts_and_restops_controller_arm() -> None:
    recover_entered = Event()
    release_recover = Event()

    class BlockingRecoverDriver(SimulatedArmDriver):
        def recover_emergency(self) -> None:
            super().recover_emergency()
            recover_entered.set()
            assert release_recover.wait(1.0)

    config = _config()
    right = BlockingRecoverDriver(ArmFeedback(received_at_s=1.0))
    controller = ArmController(ArmSide.RIGHT, config, right)
    controller.tick(dt=0.1)
    controller.emergency_stop("first cause")
    controller.tick(dt=0.1)
    controller.request_recovery()
    recovery_tick = Thread(target=controller.tick, kwargs={"dt": 0.1})
    recovery_tick.start()
    assert recover_entered.wait(1.0)

    new_stop = Thread(target=controller.emergency_stop, args=("new request",))
    new_stop.start()
    release_recover.set()
    recovery_tick.join(1.0)
    new_stop.join(1.0)

    assert not recovery_tick.is_alive() and not new_stop.is_alive()
    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert controller.snapshot().fault_reason == "first cause"
    assert right.emergency_stop_calls == 2


def test_emergency_after_recovery_request_but_before_transaction_invalidates_it() -> None:
    config = _config()
    right = SimulatedArmDriver(ArmFeedback(received_at_s=1.0))
    controller = ArmController(ArmSide.RIGHT, config, right)
    controller.tick(dt=0.1)
    controller.emergency_stop("first cause")
    controller.tick(dt=0.1)
    controller._emergency_operation_lock.acquire()  # type: ignore[attr-defined]
    controller.request_recovery()

    recovery_tick = Thread(target=controller.tick, kwargs={"dt": 0.1})
    recovery_tick.start()
    new_generation_latched = Event()
    original_latch = controller._latch_emergency  # type: ignore[attr-defined]

    def observed_latch(reason: str) -> int:
        generation = original_latch(reason)
        new_generation_latched.set()
        return generation

    controller._latch_emergency = observed_latch  # type: ignore[attr-defined]
    new_stop = Thread(target=controller.emergency_stop, args=("new request",))
    new_stop.start()
    assert new_generation_latched.wait(1.0)
    controller._emergency_operation_lock.release()  # type: ignore[attr-defined]
    recovery_tick.join(1.0)
    new_stop.join(1.0)

    assert not recovery_tick.is_alive() and not new_stop.is_alive()
    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert right.recover_emergency_calls == 0
    assert right.emergency_stop_calls == 2


def test_disconnect_holds_current_feedback_and_clears_pending_command() -> None:
    controller, config, drivers = _controller()
    controller.request_homing()
    for _ in range(8):
        controller.tick(dt=0.1)
    controller.submit_servo(
        ServoCommand((20.0, 0, 0, 0, 0, 0), config.default_joints_speed_deg_s)
    )
    current = drivers["right"].read_feedback().joints_deg

    controller.client_disconnected()
    controller.tick(dt=0.1)

    assert drivers["right"].hold_calls >= 1
    assert controller.snapshot().target_q == current
    assert controller.snapshot().target_qd == ZERO6


def test_disconnect_hold_persists_across_feedback_drift_until_new_command() -> None:
    controller, config, drivers = _controller()
    driver = drivers["right"]
    controller.request_homing()
    for _ in range(8):
        controller.tick(dt=0.1)
    controller.submit_servo(
        ServoCommand((20.0, 0, 0, 0, 0, 0), config.default_joints_speed_deg_s)
    )
    controller.tick(dt=0.1)

    controller.client_disconnected()
    held = controller.snapshot()
    drifted = replace(
        driver.read_feedback(),
        joints_deg=(0.1, 0, 0, 0, 0, 0),
        received_at_s=2.0,
    )
    driver.set_feedback(drifted)
    controller.tick(dt=0.1)

    after_drift = controller.snapshot()
    assert held.hold is True
    assert after_drift.hold is True
    assert after_drift.target_q == drifted.joints_deg
    assert driver.hold_calls >= 2

    controller.submit_servo(
        ServoCommand((2.0, 0, 0, 0, 0, 0), config.default_joints_speed_deg_s)
    )
    controller.tick(dt=0.1)
    assert controller.snapshot().target_q == (2.0, 0, 0, 0, 0, 0)
    assert controller.snapshot().hold is False
