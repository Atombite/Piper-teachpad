# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Safe adapter for the Piper SDK V2 interface."""

from __future__ import annotations

import logging
import math
from numbers import Integral, Real
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

from piper_control.config import ArmConfig, PiperGripperConfig
from piper_control.types import (
    ArmFeedback,
    GripperDiagnostics,
    Joint6,
    RawJointFeedback,
    TcpPose6,
)


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingOperation:
    operation: str
    started_at_s: float
    thread_name: str
    thread_id: int
    next_report_at_s: float
    sdk_stage: str | None = None
    stage_started_at_s: float | None = None
    reported: bool = False
    last_reported_stage: str | None = None


class _OperationWatchdog:
    """Report driver calls that remain blocked past the feedback deadline."""

    def __init__(
        self,
        *,
        side_name: str,
        can_port: str,
        timeout_s: float,
        monotonic: Callable[[], float],
    ) -> None:
        self._side_name = side_name
        self._can_port = can_port
        self._timeout_s = timeout_s
        self._repeat_s = max(5.0, timeout_s * 10.0)
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_token = 0
        self._pending: dict[int, _PendingOperation] = {}
        self._thread_tokens: dict[int, int] = {}

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"piper-driver-watchdog-{self._side_name}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)

    @contextmanager
    def track(self, name: str):  # type: ignore[no-untyped-def]
        started_at_s = self._monotonic()
        current = threading.current_thread()
        with self._lock:
            token = self._next_token
            self._next_token += 1
            thread_id = threading.get_ident()
            self._pending[token] = _PendingOperation(
                operation=name,
                started_at_s=started_at_s,
                thread_name=current.name,
                thread_id=thread_id,
                next_report_at_s=started_at_s + self._timeout_s,
            )
            self._thread_tokens[thread_id] = token
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            completed_at_s = self._monotonic()
            with self._lock:
                pending = self._pending.pop(token, None)
                if self._thread_tokens.get(thread_id) == token:
                    self._thread_tokens.pop(thread_id, None)
            if pending is not None:
                elapsed_s = max(0.0, completed_at_s - pending.started_at_s)
                if pending.reported:
                    LOGGER.warning(
                        "event=driver_operation_recovered side=%s can_port=%s "
                        "operation=%s blocked_stage=%s elapsed_s=%.6f "
                        "outcome=%s thread=%s thread_id=%s",
                        self._side_name,
                        self._can_port,
                        pending.operation,
                        pending.last_reported_stage or "none",
                        elapsed_s,
                        "failed" if failed else "completed",
                        pending.thread_name,
                        pending.thread_id,
                    )

    @contextmanager
    def stage(self, name: str):  # type: ignore[no-untyped-def]
        thread_id = threading.get_ident()
        previous_stage: str | None = None
        previous_stage_started_at_s: float | None = None
        token: int | None = None
        with self._lock:
            token = self._thread_tokens.get(thread_id)
            pending = None if token is None else self._pending.get(token)
            if pending is not None:
                previous_stage = pending.sdk_stage
                previous_stage_started_at_s = pending.stage_started_at_s
                pending.sdk_stage = name
                pending.stage_started_at_s = self._monotonic()
        try:
            yield
        finally:
            if token is not None:
                with self._lock:
                    pending = self._pending.get(token)
                    if pending is not None:
                        pending.sdk_stage = previous_stage
                        pending.stage_started_at_s = previous_stage_started_at_s

    def _run(self) -> None:
        poll_s = min(0.1, max(0.01, self._timeout_s / 4.0))
        while not self._stop.is_set():
            self._wake.wait(poll_s)
            self._wake.clear()
            now_s = self._monotonic()
            with self._lock:
                for pending in self._pending.values():
                    if now_s < pending.next_report_at_s:
                        continue
                    elapsed_s = max(0.0, now_s - pending.started_at_s)
                    stage_elapsed_s = (
                        0.0
                        if pending.stage_started_at_s is None
                        else max(0.0, now_s - pending.stage_started_at_s)
                    )
                    pending.reported = True
                    pending.last_reported_stage = pending.sdk_stage
                    pending.next_report_at_s = now_s + self._repeat_s
                    # Keep the rare transition log ordered before completion can
                    # remove this operation and emit its recovery event.
                    LOGGER.error(
                        "event=driver_operation_blocked side=%s can_port=%s "
                        "operation=%s sdk_stage=%s elapsed_s=%.6f "
                        "stage_elapsed_s=%.6f thread=%s thread_id=%s",
                        self._side_name,
                        self._can_port,
                        pending.operation,
                        pending.sdk_stage or "none",
                        elapsed_s,
                        stage_elapsed_s,
                        pending.thread_name,
                        pending.thread_id,
                    )


def _tracked_driver_operation(name: str):  # type: ignore[no-untyped-def]
    def decorate(method):  # type: ignore[no-untyped-def]
        @wraps(method)
        def wrapped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            with self._operation_watchdog.track(name):
                return method(self, *args, **kwargs)

        return wrapped

    return decorate


def _create_sdk(can_port: str) -> Any:
    # Import and construction are intentionally delayed until connect().
    from piper_sdk import C_PiperInterface_V2

    return C_PiperInterface_V2(can_name=can_port)


def _sdk_integer(
    value: object, field: str, *, description: str = "an integer"
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"Piper SDK {field} must contain {description}")
    return int(value)


def _sdk_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Piper SDK {field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Piper SDK {field} must be a finite number")
    return result


def _normalized_fault_code(
    err_code: int, arm_status: int, *, emergency_stop_expected: bool
) -> int:
    """Return faults while excluding a stop initiated by this driver."""
    if err_code:
        return err_code
    if arm_status == 0x01 and emergency_stop_expected:
        return 0
    return arm_status


class PiperArmDriver:
    """Translate service units to a serialized, safe Piper V2 SDK boundary."""

    _HOLD_LIMIT_CORRECTION_DEG = 10.0
    _ENABLE_STABLE_SAMPLES = 3
    _ENABLE_POLL_INTERVAL_S = 0.01

    def __init__(
        self,
        config: ArmConfig,
        *,
        sdk: Any | None = None,
        sdk_factory: Callable[[str], Any] = _create_sdk,
        enable_timeout_s: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._sdk = sdk
        self._sdk_factory = sdk_factory
        self._enable_timeout_s = enable_timeout_s
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._connected = False
        self._ready = False
        self._emergency_latched = threading.Event()
        self._enable_generation = 0
        self._recovery_enable_baseline_s: float | None = None
        self._feedback_heartbeat_s: float | None = None
        self._feedback_progress_at_s = monotonic()
        self._operation_watchdog = _OperationWatchdog(
            side_name=config.name,
            can_port=config.can_port,
            timeout_s=config.feedback_timeout_s,
            monotonic=monotonic,
        )

    @_tracked_driver_operation("connect")
    def connect(self) -> None:
        self._operation_watchdog.start()
        with self._lock:
            self._enable_generation += 1
            self._connected = False
            self._ready = False
            if self._sdk is None:
                self._sdk = self._sdk_call(
                    "sdk.construct_interface",
                    self._sdk_factory,
                    self._config.can_port,
                )
            self._sdk_call("sdk.ConnectPort", self._sdk.ConnectPort)
            self._connected = True

    @_tracked_driver_operation("enable")
    def enable(self) -> None:
        with self._lock:
            sdk = self._require_connected()
            self._raise_if_emergency()
            self._ready = False
            self._enable_generation += 1
            generation = self._enable_generation
            recovery_baseline = self._recovery_enable_baseline_s
        deadline = self._monotonic() + self._enable_timeout_s

        if recovery_baseline is not None:
            LOGGER.info("event=piper_recovery_phase phase=waiting_for_disabled_feedback")
            while True:
                with self._lock:
                    self._raise_if_enable_cancelled(generation)
                    timestamp, enable_status = self._read_enable_feedback(sdk)
                if timestamp > recovery_baseline and not any(enable_status):
                    LOGGER.info(
                        "event=piper_recovery_phase phase=disabled_feedback_confirmed"
                    )
                    break
                self._wait_for_enable_poll(
                    generation,
                    deadline,
                    "Piper recovery release feedback timed out",
                )
        else:
            with self._lock:
                self._raise_if_enable_cancelled(generation)
                timestamp, _enable_status = self._read_enable_feedback(sdk)

        last_timestamp = timestamp
        stable_samples = 0
        LOGGER.info("event=piper_recovery_phase phase=enable_requested")
        while True:
            with self._lock:
                self._raise_if_enable_cancelled(generation)
                self._sdk_call("sdk.EnablePiper", sdk.EnablePiper)
                timestamp, enable_status = self._read_enable_feedback(sdk)
                self._raise_if_enable_cancelled(generation)
            if timestamp > last_timestamp:
                last_timestamp = timestamp
                stable_samples = stable_samples + 1 if all(enable_status) else 0
            if stable_samples >= self._ENABLE_STABLE_SAMPLES:
                break
            disabled_motors = [
                index
                for index, enabled in enumerate(enable_status, 1)
                if not enabled
            ]
            self._wait_for_enable_poll(
                generation,
                deadline,
                "Piper enable handshake timed out: disabled motors "
                f"{disabled_motors}",
            )
        with self._lock:
            self._raise_if_enable_cancelled(generation)
            self._sdk_call(
                "sdk.MotionCtrl_2",
                sdk.MotionCtrl_2,
                0x01,
                0x01,
                100,
                0x00,
            )
            self._raise_if_enable_cancelled(generation)
            self._ready = True
            self._recovery_enable_baseline_s = None
        LOGGER.info("event=piper_recovery_phase phase=motion_mode_restored")
        LOGGER.info("event=piper_recovery_phase phase=enable_confirmed")

    def _wait_for_enable_poll(
        self, generation: int, deadline: float, timeout_message: str
    ) -> None:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise TimeoutError(timeout_message)
        self._emergency_latched.wait(
            min(self._ENABLE_POLL_INTERVAL_S, remaining)
        )
        with self._lock:
            self._raise_if_enable_cancelled(generation)

    @staticmethod
    def _enable_status(feedback: Any) -> tuple[float, tuple[bool, ...]]:
        timestamp = _sdk_number(feedback.time_stamp, "low-speed timestamp")
        enabled = tuple(
            bool(getattr(feedback, f"motor_{index}").foc_status.driver_enable_status)
            for index in range(1, 7)
        )
        return timestamp, enabled

    def _read_enable_feedback(self, sdk: Any) -> tuple[float, tuple[bool, ...]]:
        feedback = self._sdk_call(
            "sdk.GetArmLowSpdInfoMsgs", sdk.GetArmLowSpdInfoMsgs
        )
        return self._enable_status(feedback)

    def _sdk_call(self, name: str, operation, *args):  # type: ignore[no-untyped-def]
        with self._operation_watchdog.stage(name):
            return operation(*args)

    def _recover_dead_sdk_receiver(self, sdk: Any) -> None:
        """Restart the SDK CAN socket when its receive thread has exited.

        piper_sdk 0.6.x leaves its connected flag set when ReadCan exits after
        a decode exception.  A plain ConnectPort call then returns without
        starting a replacement thread, so feedback remains stale forever even
        though SocketCAN continues receiving frames.
        """
        receiver = getattr(
            sdk,
            "_C_PiperInterface_V2__can_deal_th",
            None,
        )
        if receiver is None or receiver.is_alive():
            return
        LOGGER.error(
            "event=piper_sdk_receiver_dead side=%s can_port=%s action=reconnect",
            self._config.name,
            self._config.can_port,
        )
        self._sdk_call("sdk.DisconnectPort", sdk.DisconnectPort)
        self._sdk_call(
            "sdk.ConnectPort",
            sdk.ConnectPort,
            True,
            False,
        )
        replacement = getattr(
            sdk,
            "_C_PiperInterface_V2__can_deal_th",
            None,
        )
        if replacement is None or not replacement.is_alive():
            raise RuntimeError("Piper SDK CAN receive thread failed to restart")
        LOGGER.info(
            "event=piper_sdk_receiver_recovered side=%s can_port=%s",
            self._config.name,
            self._config.can_port,
        )

    def _replace_stalled_sdk(self, sdk: Any) -> None:
        """Replace an SDK instance whose live receive thread is wedged."""
        LOGGER.error(
            "event=piper_sdk_feedback_stalled side=%s can_port=%s "
            "action=replace_sdk",
            self._config.name,
            self._config.can_port,
        )
        self._sdk_call("sdk.DisconnectPort", sdk.DisconnectPort)

        # C_PiperInterface_V2 is a per-CAN singleton.  Remove only this exact
        # stale instance so construction can produce fresh locks and threads.
        sdk_type = type(sdk)
        instances = getattr(sdk_type, "_instances", None)
        singleton_lock = getattr(sdk_type, "_lock", None)
        if isinstance(instances, dict) and singleton_lock is not None:
            with singleton_lock:
                if instances.get(self._config.can_port) is sdk:
                    del instances[self._config.can_port]

        self._connected = False
        replacement = self._sdk_call(
            "sdk.construct_interface",
            self._sdk_factory,
            self._config.can_port,
        )
        try:
            self._sdk_call(
                "sdk.ConnectPort",
                replacement.ConnectPort,
                False,
                False,
            )
        except BaseException:
            self._ready = False
            raise
        self._sdk = replacement
        self._connected = True
        self._feedback_heartbeat_s = None
        self._feedback_progress_at_s = self._monotonic()
        LOGGER.info(
            "event=piper_sdk_feedback_recovered side=%s can_port=%s",
            self._config.name,
            self._config.can_port,
        )

    def _observe_feedback_heartbeat(
        self, sdk: Any, timestamps: tuple[float, ...]
    ) -> None:
        # The aggregate is only as fresh as its oldest required report.  Using
        # the newest timestamp hides a partially wedged SDK callback where one
        # report type keeps updating while another remains frozen.
        heartbeat_s = min(timestamps, default=0.0)
        now_s = self._monotonic()
        if (
            self._feedback_heartbeat_s is None
            or heartbeat_s > self._feedback_heartbeat_s
        ):
            self._feedback_heartbeat_s = heartbeat_s
            self._feedback_progress_at_s = now_s
            return
        if now_s - self._feedback_progress_at_s < self._config.feedback_timeout_s:
            return
        self._replace_stalled_sdk(sdk)

    @_tracked_driver_operation("read_feedback")
    def read_feedback(self) -> ArmFeedback:
        """Read a best-effort aggregate of independently updated SDK reports.

        Piper exposes mutable wrappers for separate CAN report types.  Each
        report's scalars are copied immediately after its getter returns so a
        later getter cannot mutate values already sampled.  Reports of
        different types are not guaranteed to belong to one atomic CAN frame.
        """
        with self._lock:
            sdk = self._require_sdk()
            self._recover_dead_sdk_receiver(sdk)
            joint_message = self._sdk_call(
                "sdk.GetArmJointMsgs", sdk.GetArmJointMsgs
            )
            joint_state = joint_message.joint_state
            raw_joints = tuple(
                getattr(joint_state, f"joint_{index}") / 1000.0
                for index in range(1, 7)
            )
            joint_timestamp = float(joint_message.time_stamp)

            gripper_position = 0.0
            gripper_timestamp: float | None = None
            piper_gripper = self._config.gripper
            builtin_gripper_enabled = (
                isinstance(piper_gripper, PiperGripperConfig)
                and piper_gripper.enabled
            )
            if builtin_gripper_enabled:
                gripper_message = self._sdk_call(
                    "sdk.GetArmGripperMsgs", sdk.GetArmGripperMsgs
                )
                gripper_position = (
                    float(gripper_message.gripper_state.grippers_angle) / 1_000_000.0
                )
                gripper_timestamp = float(gripper_message.time_stamp)

            pose_message = self._sdk_call(
                "sdk.GetArmEndPoseMsgs", sdk.GetArmEndPoseMsgs
            )
            pose = pose_message.end_pose
            tcp_pose: TcpPose6 = (
                float(pose.X_axis) / 1_000_000.0,
                float(pose.Y_axis) / 1_000_000.0,
                float(pose.Z_axis) / 1_000_000.0,
                float(pose.RX_axis) / 1000.0,
                float(pose.RY_axis) / 1000.0,
                float(pose.RZ_axis) / 1000.0,
            )
            pose_timestamp = float(pose_message.time_stamp)

            status_message = self._sdk_call(
                "sdk.GetArmStatus", sdk.GetArmStatus
            )
            status = status_message.arm_status
            fault_code = _normalized_fault_code(
                int(status.err_code),
                int(status.arm_status),
                emergency_stop_expected=self._emergency_latched.is_set(),
            )
            status_timestamp = float(status_message.time_stamp)

            connected = bool(
                self._sdk_call("sdk.get_connect_status", sdk.get_connect_status)
            )
            enabled = all(
                bool(value)
                for value in self._sdk_call(
                    "sdk.GetArmEnableStatus", sdk.GetArmEnableStatus
                )
            )

            joints = tuple(
                raw - calibration
                for raw, calibration in zip(
                    raw_joints,
                    self._config.calibration_joints_pos_deg,
                    strict=True,
                )
            )
            timestamps = [
                joint_timestamp,
                pose_timestamp,
                status_timestamp,
            ]
            if gripper_timestamp is not None:
                timestamps.append(gripper_timestamp)
            received_at = min((value for value in timestamps if value > 0), default=0.0)
            self._observe_feedback_heartbeat(sdk, tuple(timestamps))
            return ArmFeedback(
                joints_deg=joints,  # type: ignore[arg-type]
                gripper_position=gripper_position,
                gripper=GripperDiagnostics(
                    backend="piper",
                    enabled=builtin_gripper_enabled and enabled,
                    communication_ok=(
                        not builtin_gripper_enabled
                        or (gripper_timestamp is not None and gripper_timestamp > 0)
                    ),
                    fault_code=0,
                    warning_code=0,
                    feedback_age_s=None,
                ),
                tcp_pose=tcp_pose,
                connected=connected,
                enabled=enabled,
                fault_code=fault_code,
                stale=any(timestamp <= 0 for timestamp in timestamps),
                received_at_s=received_at,
            )

    @_tracked_driver_operation("read_raw_joint_feedback")
    def read_raw_joint_feedback(self) -> RawJointFeedback:
        """Read one joint report without applying configured calibration."""

        with self._lock:
            sdk = self._require_connected()
            self._recover_dead_sdk_receiver(sdk)
            joint_message = self._sdk_call(
                "sdk.GetArmJointMsgs", sdk.GetArmJointMsgs
            )
            joint_state = joint_message.joint_state
            joints_mdeg = tuple(
                _sdk_integer(
                    getattr(joint_state, f"joint_{index}"),
                    f"joint_{index}",
                    description="integer millidegrees",
                )
                for index in range(1, 7)
            )
            received_at = _sdk_number(joint_message.time_stamp, "joint timestamp")

            status_message = self._sdk_call(
                "sdk.GetArmStatus", sdk.GetArmStatus
            )
            status = status_message.arm_status
            status_timestamp = _sdk_number(
                status_message.time_stamp, "status timestamp"
            )
            fault_code = _sdk_integer(
                status.err_code or status.arm_status,
                "fault code",
            )
            connected = bool(
                self._sdk_call("sdk.get_connect_status", sdk.get_connect_status)
            )

            self._observe_feedback_heartbeat(
                sdk,
                (received_at, status_timestamp),
            )

            return RawJointFeedback(
                joints_mdeg=joints_mdeg,  # type: ignore[arg-type]
                connected=connected,
                fault_code=fault_code,
                stale=(
                    received_at <= 0
                    or status_timestamp <= 0
                ),
                received_at_s=received_at,
            )

    @_tracked_driver_operation("command_joints")
    def command_joints(self, joints_deg: Joint6) -> None:
        values = self._validate_joints(joints_deg)
        for index, (value, minimum, maximum) in enumerate(
            zip(
                values,
                self._config.min_joints_deg,
                self._config.max_joints_deg,
                strict=True,
            ),
            1,
        ):
            if not minimum <= value <= maximum:
                raise ValueError(f"joint {index} is outside configured logical range")
        raw_millidegrees = self._calibrated_millidegrees(values)
        with self._lock:
            sdk = self._require_ready()
            self._sdk_call("sdk.JointCtrl", sdk.JointCtrl, *raw_millidegrees)

    def _calibrated_millidegrees(self, values: Joint6) -> tuple[int, ...]:
        raw_degrees = tuple(
            logical + calibration
            for logical, calibration in zip(
                values, self._config.calibration_joints_pos_deg, strict=True
            )
        )
        for index, (value, minimum, maximum) in enumerate(
            zip(
                raw_degrees,
                ArmConfig.PHYSICAL_MIN_JOINTS_DEG,
                ArmConfig.PHYSICAL_MAX_JOINTS_DEG,
                strict=True,
            ),
            1,
        ):
            if not minimum <= value <= maximum:
                raise ValueError(f"joint {index} is outside Piper physical range")
        return tuple(round(value * 1000) for value in raw_degrees)

    def _hold_millidegrees(self, values: Joint6) -> tuple[int, ...]:
        raw_degrees = tuple(
            logical + calibration
            for logical, calibration in zip(
                values, self._config.calibration_joints_pos_deg, strict=True
            )
        )
        clamped: list[float] = []
        for index, (value, minimum, maximum) in enumerate(
            zip(
                raw_degrees,
                ArmConfig.PHYSICAL_MIN_JOINTS_DEG,
                ArmConfig.PHYSICAL_MAX_JOINTS_DEG,
                strict=True,
            ),
            1,
        ):
            safe_value = min(max(value, minimum), maximum)
            correction = abs(safe_value - value)
            if correction > self._HOLD_LIMIT_CORRECTION_DEG:
                raise ValueError(
                    f"joint {index} HOLD correction exceeds "
                    f"{self._HOLD_LIMIT_CORRECTION_DEG:g} degrees: "
                    f"{value:g} outside [{minimum:g}, {maximum:g}]"
                )
            clamped.append(safe_value)
        return tuple(round(value * 1000) for value in clamped)

    @_tracked_driver_operation("command_gripper")
    def command_gripper(self, position: float) -> None:
        if not math.isfinite(position):
            raise ValueError("gripper position must be finite")
        gripper = self._config.gripper
        if not isinstance(gripper, PiperGripperConfig):
            raise ValueError("Piper driver requires a Piper gripper backend")
        if not gripper.enabled:
            raise ValueError("gripper is disabled")
        if not (
            gripper.min_position_m
            <= position
            <= gripper.max_position_m
        ):
            raise ValueError("position is outside configured gripper range")
        with self._lock:
            sdk = self._require_ready()
            self._sdk_call(
                "sdk.GripperCtrl",
                sdk.GripperCtrl,
                round(position * 1_000_000),
                round(gripper.effort_nm * 1000),
                0x01,
                0,
            )

    @_tracked_driver_operation("emergency_stop")
    def emergency_stop(self) -> None:
        with self._lock:
            self._enable_generation += 1
            self._emergency_latched.set()
            self._ready = False
            self._recovery_enable_baseline_s = None
            sdk = self._require_sdk()
            self._sdk_call("sdk.EmergencyStop.stop", sdk.EmergencyStop, 0x01)

    @_tracked_driver_operation("recover_emergency")
    def recover_emergency(self) -> None:
        with self._lock:
            sdk = self._require_sdk()
            baseline, _enable_status = self._read_enable_feedback(sdk)
            self._sdk_call(
                "sdk.EmergencyStop.release", sdk.EmergencyStop, 0x02
            )
            self._ready = False
            self._recovery_enable_baseline_s = baseline
            self._emergency_latched.clear()
        LOGGER.info("event=piper_recovery_phase phase=release_requested")

    @_tracked_driver_operation("hold")
    def hold(self, feedback: ArmFeedback) -> None:
        # HOLD only reissues the current logical joint position. It does not
        # alter enable state or invoke a controller reset.
        values = self._validate_joints(feedback.joints_deg)
        raw_millidegrees = self._hold_millidegrees(values)
        with self._lock:
            sdk = self._require_connected()
            self._sdk_call("sdk.JointCtrl", sdk.JointCtrl, *raw_millidegrees)

    def shutdown(self) -> None:
        """Piper SDK lifetime remains process-owned; no power-off is issued."""
        self._operation_watchdog.stop()

    def _require_sdk(self) -> Any:
        if self._sdk is None:
            raise RuntimeError("Piper driver is not connected")
        return self._sdk

    def _require_connected(self) -> Any:
        sdk = self._require_sdk()
        if not self._connected:
            raise RuntimeError("Piper driver is not connected")
        return sdk

    def _require_ready(self) -> Any:
        sdk = self._require_connected()
        self._raise_if_emergency()
        if not self._ready:
            raise RuntimeError("Piper driver is not enabled and ready")
        return sdk

    def _raise_if_emergency(self) -> None:
        if self._emergency_latched.is_set():
            raise RuntimeError("Piper driver emergency stop is latched")

    def _raise_if_enable_cancelled(self, generation: int) -> None:
        if generation != self._enable_generation:
            raise RuntimeError("Piper enable cancelled by emergency or state change")
        self._raise_if_emergency()

    @staticmethod
    def _validate_joints(joints_deg: Joint6) -> Joint6:
        if len(joints_deg) != 6:
            raise ValueError("joint command must contain six values")
        values = tuple(float(value) for value in joints_deg)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("joint command values must be finite")
        return values  # type: ignore[return-value]
