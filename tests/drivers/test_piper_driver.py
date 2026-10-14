# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError
import threading
import time
from types import SimpleNamespace

import pytest

from piper_control.config import ArmConfig, GripperConfig, ServerConfig
from piper_control.drivers.base import ArmDriver
from piper_control.drivers.piper import PiperArmDriver
from piper_control.types import ArmFeedback, RawJointFeedback


def arm_config(
    calibration: tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0),
    *,
    gripper_enabled: bool = True,
) -> ArmConfig:
    return ArmConfig(
        server=ServerConfig(host="127.0.0.1", port=8765),
        control_hz=100,
        feedback_timeout_s=0.5,
        shutdown_home_timeout_s=3,
        name="right_arm",
        can_port="can-test",
        auto_initialize=True,
        calibration_joints_pos_deg=calibration,
        reset_joints_pos_deg=(0, 0, 0, 0, 0, 0),
        min_joints_deg=(-140, 0, -160, -90, -60, -110),
        max_joints_deg=(140, 160, 0, 90, 60, 110),
        max_joints_speed_deg_s=(30, 30, 30, 30, 30, 30),
        default_joints_speed_deg_s=(10, 10, 10, 10, 10, 10),
        gripper=GripperConfig(
            enabled=gripper_enabled,
            min_position_m=0,
            max_position_m=0.07,
            max_speed_m_s=0.05,
            default_speed_m_s=0.02,
            effort_nm=1,
        ),
    )


class FakeSdk:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.joints_mdeg = (11000, 22000, 33000, 44000, 55000, 66000)
        self.gripper_um = 35000
        self.enable_results = [True]
        self.connected = True
        self.enabled = True
        self.fault_code = 0
        self.low_speed_timestamp = 1.0
        self.enable_called = threading.Event()
        self.connect_error: BaseException | None = None

    def __getattr__(self, name: str) -> object:
        if name in {"DisableArm", "DisablePiper", "ResetPiper"}:
            raise AssertionError(f"forbidden SDK member accessed: {name}")
        raise AttributeError(name)

    def ConnectPort(self) -> None:
        self.calls.append(("ConnectPort", ()))
        if self.connect_error is not None:
            raise self.connect_error

    def EnablePiper(self) -> bool:
        self.calls.append(("EnablePiper", ()))
        self.enable_called.set()
        return self.enable_results.pop(0) if self.enable_results else self.enabled

    def MotionCtrl_2(self, *args: object) -> None:
        self.calls.append(("MotionCtrl_2", args))

    def JointCtrl(self, *args: object) -> None:
        self.calls.append(("JointCtrl", args))

    def GripperCtrl(self, *args: object) -> None:
        self.calls.append(("GripperCtrl", args))

    def EmergencyStop(self, *args: object) -> None:
        self.calls.append(("EmergencyStop", args))

    def get_connect_status(self) -> bool:
        return self.connected

    def GetArmEnableStatus(self) -> list[bool]:
        return [self.enabled] * 6

    def GetArmLowSpdInfoMsgs(self) -> SimpleNamespace:
        self.low_speed_timestamp += 0.01
        motors = {
            f"motor_{index}": SimpleNamespace(
                foc_status=SimpleNamespace(driver_enable_status=self.enabled)
            )
            for index in range(1, 7)
        }
        return SimpleNamespace(time_stamp=self.low_speed_timestamp, **motors)

    def GetArmJointMsgs(self) -> SimpleNamespace:
        joint_state = SimpleNamespace(
            **{f"joint_{index}": value for index, value in enumerate(self.joints_mdeg, 1)}
        )
        return SimpleNamespace(time_stamp=123.5, Hz=100.0, joint_state=joint_state)

    def GetArmGripperMsgs(self) -> SimpleNamespace:
        foc_status = SimpleNamespace(
            voltage_too_low=False,
            motor_overheating=False,
            driver_overcurrent=False,
            driver_overheating=False,
            sensor_status=False,
            driver_error_status=False,
            driver_enable_status=True,
            homing_status=True,
        )
        state = SimpleNamespace(
            grippers_angle=self.gripper_um,
            grippers_effort=1000,
            foc_status=foc_status,
        )
        return SimpleNamespace(time_stamp=123.5, Hz=100.0, gripper_state=state)

    def GetArmEndPoseMsgs(self) -> SimpleNamespace:
        pose = SimpleNamespace(
            X_axis=1000,
            Y_axis=2000,
            Z_axis=3000,
            RX_axis=4000,
            RY_axis=5000,
            RZ_axis=6000,
        )
        return SimpleNamespace(time_stamp=123.5, Hz=100.0, end_pose=pose)

    def GetArmStatus(self) -> SimpleNamespace:
        status = SimpleNamespace(
            ctrl_mode=1,
            arm_status=0 if self.fault_code == 0 else 9,
            mode_feed=1,
            teach_status=0,
            motion_status=0,
            trajectory_num=0,
            err_code=self.fault_code,
        )
        return SimpleNamespace(time_stamp=123.5, Hz=100.0, arm_status=status)


def test_constructor_does_not_create_or_connect_sdk() -> None:
    created = []
    PiperArmDriver(arm_config(), sdk_factory=lambda _: created.append(True))
    assert created == []


def test_connect_creates_v2_sdk_lazily_and_connects() -> None:
    sdk = FakeSdk()
    driver = PiperArmDriver(arm_config(), sdk_factory=lambda can_port: (can_port, sdk)[1])
    driver.connect()
    assert sdk.calls == [("ConnectPort", ())]


def test_enable_completes_handshake_then_selects_joint_mode() -> None:
    sdk = FakeSdk()
    sdk.enable_results = [False, True]
    driver = PiperArmDriver(arm_config(), sdk=sdk)
    driver.connect()
    sdk.calls.clear()
    driver.enable()
    assert sdk.calls == [
        ("EnablePiper", ()),
        ("EnablePiper", ()),
        ("EnablePiper", ()),
        ("MotionCtrl_2", (0x01, 0x01, 100, 0x00)),
    ]


def test_enable_rejects_stale_success_when_motor_feedback_stays_disabled() -> None:
    sdk = FakeSdk()
    sdk.enable_results = [True]
    sdk.enabled = False
    driver = PiperArmDriver(arm_config(), sdk=sdk, enable_timeout_s=0.02)
    driver.connect()
    sdk.calls.clear()

    with pytest.raises(
        TimeoutError,
        match=r"enable handshake timed out: disabled motors \[1, 2, 3, 4, 5, 6\]",
    ):
        driver.enable()

    assert not any(name == "MotionCtrl_2" for name, _args in sdk.calls)


def test_recovery_enable_waits_for_fresh_disabled_then_stable_enabled_feedback() -> None:
    class TransitioningSdk(FakeSdk):
        def __init__(self) -> None:
            super().__init__()
            self.samples = iter(
                [
                    (10.0, True),
                    (10.0, True),
                    (11.0, False),
                    (12.0, False),
                    (13.0, True),
                    (14.0, True),
                    (15.0, True),
                ]
            )
            self.last_sample = (10.0, True)
            self.low_speed_reads = 0
            self.motion_ctrl_low_speed_reads: int | None = None

        def GetArmLowSpdInfoMsgs(self) -> SimpleNamespace:
            self.last_sample = next(self.samples, self.last_sample)
            self.low_speed_reads += 1
            timestamp, enabled = self.last_sample
            motors = {
                f"motor_{index}": SimpleNamespace(
                    foc_status=SimpleNamespace(driver_enable_status=enabled)
                )
                for index in range(1, 7)
            }
            return SimpleNamespace(time_stamp=timestamp, **motors)

        def GetArmEnableStatus(self) -> list[bool]:
            return [True] * 6

        def MotionCtrl_2(self, *args: object) -> None:
            self.motion_ctrl_low_speed_reads = self.low_speed_reads
            super().MotionCtrl_2(*args)

    sdk = TransitioningSdk()
    driver = PiperArmDriver(arm_config(), sdk=sdk, enable_timeout_s=0.2)
    driver.connect()
    driver.emergency_stop()
    driver.recover_emergency()
    sdk.calls.clear()

    driver.enable()

    assert sdk.motion_ctrl_low_speed_reads == 7


def test_emergency_stop_preempts_enable_wait_and_aborts_ready_transition() -> None:
    sdk = FakeSdk()
    sdk.enable_results = []
    sdk.enabled = False
    driver = PiperArmDriver(arm_config(), sdk=sdk, enable_timeout_s=10)
    driver.connect()
    errors: list[BaseException] = []

    def run_enable() -> None:
        try:
            driver.enable()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_enable)
    thread.start()
    assert sdk.enable_called.wait(0.1)
    started = time.monotonic()
    driver.emergency_stop()
    elapsed = time.monotonic() - started
    thread.join(0.1)

    assert elapsed < 0.05
    assert not thread.is_alive()
    assert len(errors) == 1 and "emergency" in str(errors[0]).lower()
    assert ("EmergencyStop", (0x01,)) in sdk.calls
    assert not any(name == "MotionCtrl_2" for name, _ in sdk.calls)


def test_recovery_cannot_revalidate_an_inflight_enable() -> None:
    sdk = FakeSdk()
    sdk.enable_results = [False, True]
    driver = PiperArmDriver(arm_config(), sdk=sdk, enable_timeout_s=10)
    driver.connect()
    controlled_event = ControlledEvent()
    driver._emergency_latched = controlled_event  # type: ignore[attr-defined]
    errors: list[BaseException] = []

    def run_enable() -> None:
        try:
            driver.enable()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_enable)
    thread.start()
    assert controlled_event.wait_started.wait(0.1)
    driver.emergency_stop()
    driver.recover_emergency()
    controlled_event.release_wait.set()
    thread.join(0.1)

    assert not thread.is_alive()
    assert len(errors) == 1 and "emergency" in str(errors[0]).lower()
    assert not any(name == "MotionCtrl_2" for name, _ in sdk.calls)
    with pytest.raises(RuntimeError, match="enabled"):
        driver.command_joints((0, 0, 0, 0, 0, 0))


def test_failed_reconnect_revokes_previous_ready_state() -> None:
    sdk = FakeSdk()
    driver = ready_driver(arm_config(), sdk)
    sdk.connect_error = RuntimeError("CAN reconnect failed")
    with pytest.raises(RuntimeError, match="CAN reconnect failed"):
        driver.connect()
    with pytest.raises(RuntimeError, match="connected"):
        driver.command_joints((0, 0, 0, 0, 0, 0))
    assert not any(name == "JointCtrl" for name, _ in sdk.calls)


def test_successful_reconnect_requires_enable_again() -> None:
    sdk = FakeSdk()
    driver = ready_driver(arm_config(), sdk)
    driver.connect()
    with pytest.raises(RuntimeError, match="enabled"):
        driver.command_joints((0, 0, 0, 0, 0, 0))
    assert not any(name == "JointCtrl" for name, _ in sdk.calls)
    driver.enable()
    driver.command_joints((0, 0, 0, 0, 0, 0))
    assert sdk.calls[-1][0] == "JointCtrl"


def test_commands_require_successful_connect_and_enable() -> None:
    sdk = FakeSdk()
    driver = PiperArmDriver(arm_config(), sdk=sdk)
    with pytest.raises(RuntimeError, match="connected"):
        driver.command_joints((0, 0, 0, 0, 0, 0))
    driver.connect()
    with pytest.raises(RuntimeError, match="enabled"):
        driver.command_gripper(0.02)
    driver.enable()
    driver.command_joints((0, 0, 0, 0, 0, 0))
    assert sdk.calls[-1][0] == "JointCtrl"


def test_enable_timeout_does_not_make_driver_ready() -> None:
    sdk = FakeSdk()
    sdk.enable_results = [False]
    sdk.enabled = False
    driver = PiperArmDriver(arm_config(), sdk=sdk, enable_timeout_s=0)
    driver.connect()
    with pytest.raises(TimeoutError, match="timed out"):
        driver.enable()
    with pytest.raises(RuntimeError, match="enabled"):
        driver.command_joints((0, 0, 0, 0, 0, 0))
    assert not any(name == "MotionCtrl_2" for name, _ in sdk.calls)


def test_emergency_stop_revokes_ready_state() -> None:
    sdk = FakeSdk()
    driver = PiperArmDriver(arm_config(), sdk=sdk)
    driver.connect()
    driver.enable()
    driver.emergency_stop()
    with pytest.raises(RuntimeError, match="emergency"):
        driver.command_joints((0, 0, 0, 0, 0, 0))
    with pytest.raises(RuntimeError, match="emergency"):
        driver.command_gripper(0.02)


def test_feedback_subtracts_calibration_and_converts_units() -> None:
    sdk = FakeSdk()
    sdk.joints_mdeg = (11000, 22000, -33000, 44000, 55000, 66000)
    driver = PiperArmDriver(arm_config((1, 2, -3, 4, 5, 6)), sdk=sdk)
    feedback = driver.read_feedback()
    assert feedback.joints_deg == (10, 20, -30, 40, 50, 60)
    assert feedback.gripper_position == 0.035
    assert feedback.tcp_pose == (0.001, 0.002, 0.003, 4.0, 5.0, 6.0)
    assert feedback.received_at_s == 123.5
    assert feedback.connected and feedback.enabled
    assert feedback.fault_code == 0 and not feedback.stale


def test_feedback_restarts_dead_piper_sdk_receive_thread() -> None:
    class ReceiverThread:
        def __init__(self, alive: bool) -> None:
            self.alive = alive

        def is_alive(self) -> bool:
            return self.alive

    class RestartableSdk(FakeSdk):
        def __init__(self) -> None:
            super().__init__()
            self._C_PiperInterface_V2__can_deal_th = ReceiverThread(True)

        def ConnectPort(
            self, can_init: bool = False, piper_init: bool = True
        ) -> None:
            self.calls.append(("ConnectPort", (can_init, piper_init)))
            self._C_PiperInterface_V2__can_deal_th = ReceiverThread(True)

        def DisconnectPort(self) -> None:
            self.calls.append(("DisconnectPort", ()))

    sdk = RestartableSdk()
    driver = PiperArmDriver(arm_config(), sdk=sdk)
    driver.connect()
    sdk.calls.clear()
    sdk._C_PiperInterface_V2__can_deal_th.alive = False

    feedback = driver.read_feedback()

    assert not feedback.stale
    assert sdk.calls == [
        ("DisconnectPort", ()),
        ("ConnectPort", (True, False)),
    ]


def test_feedback_does_not_reconnect_live_piper_sdk_receive_thread() -> None:
    class LiveThread:
        def is_alive(self) -> bool:
            return True

    sdk = FakeSdk()
    sdk._C_PiperInterface_V2__can_deal_th = LiveThread()  # type: ignore[attr-defined]
    driver = PiperArmDriver(arm_config(), sdk=sdk)
    driver.connect()
    sdk.calls.clear()

    feedback = driver.read_feedback()

    assert not feedback.stale
    assert sdk.calls == []


def test_feedback_replaces_live_sdk_when_heartbeat_stops() -> None:
    class DisconnectableSdk(FakeSdk):
        def DisconnectPort(self) -> None:
            self.calls.append(("DisconnectPort", ()))

    class ReplacementSdk(FakeSdk):
        def ConnectPort(
            self, can_init: bool = False, piper_init: bool = True
        ) -> None:
            self.calls.append(("ConnectPort", (can_init, piper_init)))

    now = [0.0]
    stalled = DisconnectableSdk()
    replacement = ReplacementSdk()
    driver = PiperArmDriver(
        arm_config(),
        sdk=stalled,
        sdk_factory=lambda _can_port: replacement,
        monotonic=lambda: now[0],
    )
    driver.connect()
    stalled.calls.clear()
    driver.read_feedback()

    now[0] = 0.6
    driver.read_feedback()

    assert stalled.calls == [("DisconnectPort", ())]
    assert replacement.calls == [("ConnectPort", (False, False))]
    replacement.calls.clear()
    assert not driver.read_feedback().stale
    assert replacement.calls == []


def test_raw_joint_feedback_preserves_sdk_millidegrees_without_commands() -> None:
    sdk = FakeSdk()
    sdk.joints_mdeg = (11123, -22456, 33789, -44001, 55234, -66567)
    driver = PiperArmDriver(arm_config((1, 2, -3, 4, 5, 6)), sdk=sdk)
    driver.connect()
    sdk.calls.clear()

    feedback = driver.read_raw_joint_feedback()

    assert feedback == RawJointFeedback(
        joints_mdeg=(11123, -22456, 33789, -44001, 55234, -66567),
        connected=True,
        fault_code=0,
        stale=False,
        received_at_s=123.5,
    )
    assert sdk.calls == []


def test_raw_joint_feedback_uses_joint_frame_freshness() -> None:
    sdk = FakeSdk()
    original_joint = sdk.GetArmJointMsgs
    sdk.GetArmJointMsgs = lambda: _with_zero_time(original_joint())  # type: ignore[method-assign]
    driver = PiperArmDriver(arm_config(), sdk=sdk)
    driver.connect()

    feedback = driver.read_raw_joint_feedback()

    assert feedback.received_at_s == 0.0
    assert feedback.stale


def test_raw_joint_feedback_requires_fresh_status_report() -> None:
    sdk = FakeSdk()
    original_status = sdk.GetArmStatus
    sdk.GetArmStatus = lambda: _with_zero_time(original_status())  # type: ignore[method-assign]
    driver = PiperArmDriver(arm_config(), sdk=sdk)
    driver.connect()

    feedback = driver.read_raw_joint_feedback()

    assert feedback.stale


@pytest.mark.parametrize("invalid", [True, "11000", 11000.5])
def test_raw_joint_feedback_rejects_non_integer_sdk_joint_values(
    invalid: object,
) -> None:
    sdk = FakeSdk()
    sdk.joints_mdeg = (invalid, 22000, 33000, 44000, 55000, 66000)  # type: ignore[assignment]
    driver = PiperArmDriver(arm_config(), sdk=sdk)
    driver.connect()

    with pytest.raises(ValueError, match="integer millidegrees"):
        driver.read_raw_joint_feedback()


def test_feedback_copies_each_sdk_wrapper_before_later_getters_mutate_it() -> None:
    sdk = FakeSdk()
    shared = sdk.GetArmJointMsgs()

    def joints():  # type: ignore[no-untyped-def]
        return shared

    def gripper():  # type: ignore[no-untyped-def]
        shared.joint_state.joint_1 = 99000
        shared.time_stamp = 999.0
        shared.Hz = 0.0
        return FakeSdk().GetArmGripperMsgs()

    sdk.GetArmJointMsgs = joints  # type: ignore[method-assign]
    sdk.GetArmGripperMsgs = gripper  # type: ignore[method-assign]

    feedback = PiperArmDriver(arm_config(), sdk=sdk).read_feedback()

    assert feedback.joints_deg[0] == 11.0
    assert feedback.received_at_s == 123.5
    assert not feedback.stale


def test_feedback_is_frozen() -> None:
    feedback = PiperArmDriver(arm_config(), sdk=FakeSdk()).read_feedback()
    with pytest.raises(FrozenInstanceError):
        feedback.fault_code = 3  # type: ignore[misc]


def test_initial_zero_timestamps_return_stale_feedback() -> None:
    sdk = FakeSdk()
    original_joint = sdk.GetArmJointMsgs
    original_gripper = sdk.GetArmGripperMsgs
    original_pose = sdk.GetArmEndPoseMsgs
    original_status = sdk.GetArmStatus
    sdk.GetArmJointMsgs = lambda: _with_zero_time(original_joint())  # type: ignore[method-assign]
    sdk.GetArmGripperMsgs = lambda: _with_zero_time(original_gripper())  # type: ignore[method-assign]
    sdk.GetArmEndPoseMsgs = lambda: _with_zero_time(original_pose())  # type: ignore[method-assign]
    sdk.GetArmStatus = lambda: _with_zero_time(original_status())  # type: ignore[method-assign]
    feedback = PiperArmDriver(arm_config(), sdk=sdk).read_feedback()
    assert feedback.received_at_s == 0.0
    assert feedback.stale


def test_positive_timestamps_are_fresh_when_sdk_fps_counter_misses_window() -> None:
    sdk = FakeSdk()

    def zero_hz(message: SimpleNamespace) -> SimpleNamespace:
        message.Hz = 0.0
        return message

    sdk.GetArmJointMsgs = lambda: zero_hz(FakeSdk().GetArmJointMsgs())  # type: ignore[method-assign]
    sdk.GetArmGripperMsgs = lambda: zero_hz(FakeSdk().GetArmGripperMsgs())  # type: ignore[method-assign]
    sdk.GetArmEndPoseMsgs = lambda: zero_hz(FakeSdk().GetArmEndPoseMsgs())  # type: ignore[method-assign]
    sdk.GetArmStatus = lambda: zero_hz(FakeSdk().GetArmStatus())  # type: ignore[method-assign]

    feedback = PiperArmDriver(arm_config(), sdk=sdk).read_feedback()

    assert feedback.received_at_s == 123.5
    assert not feedback.stale


def test_disabled_gripper_freshness_ignores_gripper_feedback() -> None:
    sdk = FakeSdk()
    sdk.GetArmGripperMsgs = lambda: _with_zero_time(  # type: ignore[method-assign]
        FakeSdk().GetArmGripperMsgs()
    )
    feedback = PiperArmDriver(
        arm_config(gripper_enabled=False), sdk=sdk
    ).read_feedback()
    assert not feedback.stale
    assert feedback.received_at_s == 123.5
    assert feedback.gripper_position == 0.0


def test_enabled_gripper_freshness_requires_gripper_feedback() -> None:
    sdk = FakeSdk()
    sdk.GetArmGripperMsgs = lambda: _with_zero_time(  # type: ignore[method-assign]
        FakeSdk().GetArmGripperMsgs()
    )
    feedback = PiperArmDriver(arm_config(gripper_enabled=True), sdk=sdk).read_feedback()
    assert feedback.stale


def test_command_adds_calibration_in_millidegrees() -> None:
    sdk = FakeSdk()
    driver = ready_driver(arm_config((1, 2, -3, 4, 5, 6)), sdk)
    driver.command_joints(
        (10, 20, -30, 40, 50, 60)
    )
    assert sdk.calls == [("JointCtrl", (11000, 22000, -33000, 44000, 55000, 66000))]


def test_gripper_converts_metres_and_nm() -> None:
    sdk = FakeSdk()
    ready_driver(arm_config(), sdk).command_gripper(0.035)
    assert sdk.calls == [("GripperCtrl", (35000, 1000, 0x01, 0))]


def test_emergency_recovery_and_hold_use_only_safe_sdk_calls() -> None:
    sdk = FakeSdk()
    driver = ready_driver(arm_config((1, 2, -3, 4, 5, 6)), sdk)
    feedback = ArmFeedback(joints_deg=(10, 20, -30, 40, 50, 60), enabled=True)
    driver.emergency_stop()
    driver.recover_emergency()
    driver.hold(feedback)
    assert sdk.calls == [
        ("EmergencyStop", (0x01,)),
        ("EmergencyStop", (0x02,)),
        ("JointCtrl", (11000, 22000, -33000, 44000, 55000, 66000)),
    ]
    assert not hasattr(ArmDriver, "disable")
    assert not hasattr(ArmDriver, "reset")


def test_hold_clamps_feedback_within_ten_degrees_of_physical_limits() -> None:
    sdk = FakeSdk()
    driver = ready_driver(arm_config(), sdk)

    driver.hold(
        ArmFeedback(
            joints_deg=(0, -2.718, 0.163, 0, 0, 0),
            connected=True,
            enabled=True,
        )
    )

    assert sdk.calls == [("JointCtrl", (0, 0, 0, 0, 0, 0))]


def test_hold_rejects_feedback_more_than_ten_degrees_outside_physical_limits() -> None:
    sdk = FakeSdk()
    driver = ready_driver(arm_config(), sdk)

    with pytest.raises(ValueError, match=r"joint 2.*10 degrees"):
        driver.hold(
            ArmFeedback(
                joints_deg=(0, -10.001, 0, 0, 0, 0),
                connected=True,
                enabled=True,
            )
        )

    assert sdk.calls == []


def test_rejects_nonfinite_joint_commands() -> None:
    driver = PiperArmDriver(arm_config(), sdk=FakeSdk())
    with pytest.raises(ValueError, match="finite"):
        driver.command_joints((float("nan"), 20, -30, 0, 0, 0))


@pytest.mark.parametrize(
    "target",
    [
        (-141, 20, -30, 0, 0, 0),
        (141, 20, -30, 0, 0, 0),
        (0, -1, -30, 0, 0, 0),
        (0, 161, -30, 0, 0, 0),
    ],
)
def test_joint_logical_range_is_rejected_before_sdk(target: tuple[float, ...]) -> None:
    sdk = FakeSdk()
    driver = ready_driver(arm_config(), sdk)
    with pytest.raises(ValueError, match="logical range"):
        driver.command_joints(target)  # type: ignore[arg-type]
    assert sdk.calls == []


def test_calibrated_physical_range_is_rejected_before_sdk() -> None:
    sdk = FakeSdk()
    inconsistent = arm_config().model_copy(
        update={"calibration_joints_pos_deg": (20, 0, 0, 0, 0, 0)}
    )
    driver = ready_driver(inconsistent, sdk)
    with pytest.raises(ValueError, match="physical range"):
        driver.command_joints((140, 20, -30, 0, 0, 0))
    assert sdk.calls == []


@pytest.mark.parametrize("position", [-0.001, 0.071])
def test_gripper_range_is_rejected_before_sdk(position: float) -> None:
    sdk = FakeSdk()
    driver = ready_driver(arm_config(), sdk)
    with pytest.raises(ValueError, match="gripper range"):
        driver.command_gripper(position)
    assert sdk.calls == []


def test_disabled_gripper_command_is_rejected_before_sdk() -> None:
    sdk = FakeSdk()
    driver = ready_driver(arm_config(gripper_enabled=False), sdk)
    with pytest.raises(ValueError, match="disabled"):
        driver.command_gripper(0.0)
    assert sdk.calls == []


def _with_zero_time(message: SimpleNamespace) -> SimpleNamespace:
    message.time_stamp = 0.0
    message.Hz = 0.0
    return message


def ready_driver(config: ArmConfig, sdk: FakeSdk) -> PiperArmDriver:
    driver = PiperArmDriver(config, sdk=sdk)
    driver.connect()
    driver.enable()
    sdk.calls.clear()
    return driver


class ControlledEvent:
    """Pause enable's wait until emergency and recovery have both completed."""

    def __init__(self) -> None:
        self._is_set = False
        self._lock = threading.Lock()
        self.wait_started = threading.Event()
        self.release_wait = threading.Event()

    def set(self) -> None:
        with self._lock:
            self._is_set = True

    def clear(self) -> None:
        with self._lock:
            self._is_set = False

    def is_set(self) -> bool:
        with self._lock:
            return self._is_set

    def wait(self, timeout: float | None = None) -> bool:
        self.wait_started.set()
        if not self.release_wait.wait(0.1 if timeout is None else max(0.1, timeout)):
            raise TimeoutError("test did not release controlled enable wait")
        return self.is_set()
