# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Fixed-rate deterministic controller for one Piper arm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import logging
from math import ceil, isfinite
from threading import Event, Lock, RLock, Thread
import time

from piper_control.config import ArmConfig, ArmSide
from piper_control.drivers.base import ArmDriver
from piper_control.interpolation import step_axes
from piper_control.lifecycle import ShutdownResult
from piper_control.mailbox import LatestValue
from piper_control.types import (
    ArmFeedback,
    ArmState,
    ControlMode,
    Joint6,
    ServiceState,
    ArmStateSnapshot,
    ZERO_JOINTS,
)
from piper_control.vectors import GripperCommand, ServoCommand


LOGGER = logging.getLogger(__name__)


class StateConflict(RuntimeError):
    """Raised when a request is incompatible with the controller state."""


class ArmController:
    """Own one arm and publish one immutable snapshot per control tick."""

    _HOMING_STABLE_CYCLES = 3
    _RECOVERY_STABLE_CYCLES = 3
    _RECOVERY_RELEASE_STABLE_DURATION_S = 0.3
    _RECOVERY_ENABLED_STABLE_DURATION_S = 0.5
    _RECOVERY_ENABLE_SETTLE_TIMEOUT_S = 2.0
    _OVERRUN_LIMIT = 3
    _ARRIVAL_TOLERANCE_DEG = 0.01
    # Piper joints can settle slightly beyond a configured boundary even
    # though feedback itself is quantized to one millidegree.  Keep command
    # limits strict while allowing this small feedback-only settling margin.
    _FEEDBACK_JOINT_TOLERANCE_DEG = 2.0
    _FEEDBACK_GRIPPER_TOLERANCE_M = 0.000001
    _MAX_COMMAND_LEAD_DEG = 5.0

    def __init__(
        self,
        side: ArmSide,
        config: ArmConfig,
        driver: ArmDriver,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._side = side
        self._side_name = side.value
        self._config = config
        self._driver = driver
        self._clock = clock
        self._sleep = sleep

        self._state_lock = RLock()
        self._tick_lock = Lock()
        self._emergency_lock = Lock()
        self._emergency_operation_lock = Lock()
        self._emergency_latched = Event()
        self._emergency_reason: str | None = None
        self._emergency_generation = 0
        self._emergency_delivered = False
        self._recovery_in_progress = False
        self._connected = False
        self._snapshot = ArmStateSnapshot(side=side)
        self._snapshot_published_at_s = self._clock()
        self._state = ServiceState.STARTING
        self._control_mode = ControlMode.LOCAL
        self._fault_reason: str | None = None
        self._servo_mailbox: LatestValue[ServoCommand] = LatestValue()
        self._gripper_mailbox: LatestValue[GripperCommand] = LatestValue()
        self._target: Joint6 = ZERO_JOINTS
        self._commanded_q: Joint6 | None = None
        self._commanded_gripper: float | None = None
        self._joint_speed: Joint6 = config.default_joints_speed_deg_s
        self._gripper_target = 0.0
        self._gripper_speed = (
            config.gripper.default_speed if config.gripper.enabled else 0.0
        )
        self._previous_feedback: ArmFeedback | None = None
        self._latest_feedback: ArmFeedback | None = None
        self._last_valid_feedback: ArmFeedback | None = None
        self._last_progress_receive_timestamp: float | None = None
        self._receive_timestamp_changed_at: float | None = None
        self._homing_stable_cycles = 0
        self._homing_started_at_s: float | None = None
        self._homing_trigger: str | None = None
        self._overrun_cycles = 0
        self._suppress_next_scheduler_overrun = False
        self._recovery_requested = False
        self._recovery_request_generation: int | None = None
        self._shutdown_requested = Event()
        self._shutdown_started_at: float | None = None
        self._shutdown_deadline: float | None = None
        self._shutdown_transition_started = Event()
        self._disconnect_hold = Event()
        self._disconnect_hold_needs_fresh_latch = False
        self._shutdown_result_lock = Lock()
        self._shutdown_result: ShutdownResult | None = None
        self._safety_errors: list[str] = []
        self._wake_event = Event()
        self._thread: Thread | None = None

    def snapshot(self) -> ArmStateSnapshot:
        # ArmStateSnapshot is immutable and writers replace the whole reference
        # while holding _state_lock.  Readers must remain lock-free so process
        # shutdown can still observe the last published state when an SDK call
        # is blocked inside a control tick that owns _state_lock.
        snapshot = self._snapshot
        if (
            snapshot.arm.feedback_fresh
            and self._clock() - self._snapshot_published_at_s
            >= self._config.feedback_timeout_s
        ):
            return replace(
                snapshot,
                arm=replace(snapshot.arm, feedback_fresh=False),
            )
        return snapshot

    def _reject_expired_feedback(self, command_name: str) -> None:
        snapshot = self.snapshot()
        if (
            snapshot.state is ServiceState.RUNNING
            and not snapshot.arm.feedback_fresh
        ):
            raise StateConflict(f"{command_name} commands require fresh feedback")

    def submit_servo(self, command: ServoCommand) -> None:
        # Fail closed before taking _state_lock: an SDK call may be blocked in
        # tick() while holding that lock, but stale feedback must still reject
        # new motion immediately.
        self._reject_expired_feedback("servo")
        with self._state_lock:
            with self._emergency_lock:
                if self._emergency_latched.is_set():
                    raise StateConflict("servo commands rejected during emergency stop")
                if self._state is not ServiceState.RUNNING:
                    raise StateConflict("servo commands require RUNNING state")
                self._reject_expired_feedback("servo")
                self._disconnect_hold.clear()
                self._disconnect_hold_needs_fresh_latch = False
                self._servo_mailbox.put(command)

    def submit_gripper(self, command: GripperCommand) -> None:
        self._reject_expired_feedback("gripper")
        with self._state_lock:
            with self._emergency_lock:
                if self._emergency_latched.is_set():
                    raise StateConflict("gripper commands rejected during emergency stop")
                if self._state is not ServiceState.RUNNING:
                    raise StateConflict("gripper commands require RUNNING state")
                self._reject_expired_feedback("gripper")
                self._disconnect_hold.clear()
                self._disconnect_hold_needs_fresh_latch = False
                self._gripper_mailbox.put(command)

    def request_homing(self) -> None:
        LOGGER.info("event=homing_request_received side=%s", self._side_name)
        with self._state_lock:
            with self._emergency_lock:
                if self._emergency_latched.is_set():
                    raise StateConflict("homing rejected during emergency stop")
                if self._state is not ServiceState.RUNNING:
                    raise StateConflict("homing requires RUNNING state")
                self._servo_mailbox.take()
                self._gripper_mailbox.take()
                self._state = ServiceState.HOMING
                self._homing_stable_cycles = 0
                self._homing_started_at_s = self._clock()
                self._homing_trigger = "request"
                self._target = self._config.reset_joints_pos_deg
                self._joint_speed = self._config.default_joints_speed_deg_s
                sample = self._last_valid_feedback
                self._commanded_q = sample.joints_deg if sample is not None else None
                if sample is not None and self._config.gripper.enabled:
                    self._commanded_gripper = sample.gripper_position
                    self._gripper_target = sample.gripper_position
                    self._gripper_speed = 0.0
                self._snapshot = replace(
                    self._snapshot,
                    state=self._state,
                    moving=False,
                    hold=True,
                    target_q=self._target,
                    target_qd=ZERO_JOINTS,
                    target_gripper=self._gripper_target,
                    target_gripper_speed=0.0,
                )
        LOGGER.info(
            "event=homing_started side=%s trigger=request", self._side_name
        )
        self._wake_event.set()

    def request_hold(self) -> None:
        """Enter the explicit mode-switch HOLD at the latest safe feedback."""

        started_at_s = self._clock()
        LOGGER.info("event=hold_requested side=%s", self._side_name)
        with self._state_lock:
            with self._emergency_lock:
                if self._emergency_latched.is_set():
                    raise StateConflict("HOLD rejected during emergency stop")
            if self._state not in (ServiceState.RUNNING, ServiceState.HOLD):
                raise StateConflict("HOLD requires RUNNING or HOLD state")
            self._servo_mailbox.take()
            self._gripper_mailbox.take()
            errors = self._hold_best_effort(
                preserve_gripper_target=self._state is ServiceState.HOLD
            )
            if errors:
                reason = "mode-switch HOLD failed: " + "; ".join(errors)
                self._enter_fault(reason, self._latest_feedback)
                self._snapshot = replace(
                    self._snapshot,
                    state=ServiceState.EMERGENCY_STOP,
                    emergency_stopped=True,
                    moving=False,
                    hold=True,
                    fault_reason=self._fault_reason,
                    target_q=self._target,
                    target_qd=ZERO_JOINTS,
                    target_gripper=self._gripper_target,
                    target_gripper_speed=0.0,
                )
                raise StateConflict(reason)
            self._state = ServiceState.HOLD
            self._snapshot = replace(
                self._snapshot,
                state=ServiceState.HOLD,
                moving=False,
                hold=True,
                target_q=self._target,
                target_qd=ZERO_JOINTS,
                target_gripper=self._gripper_target,
                target_gripper_speed=0.0,
            )
        LOGGER.info(
            "event=hold_completed side=%s elapsed_s=%.6f",
            self._side_name,
            self._clock() - started_at_s,
        )
        self._wake_event.set()

    def change_control_mode(self, mode: ControlMode) -> None:
        """Change identity class from explicit HOLD and resume RUNNING."""

        if not isinstance(mode, ControlMode):
            raise TypeError("mode must be a ControlMode")
        with self._state_lock:
            with self._emergency_lock:
                if self._emergency_latched.is_set():
                    raise StateConflict("control mode change rejected during emergency stop")
            if self._state is not ServiceState.HOLD:
                raise StateConflict("control mode change requires HOLD state")
            arm = self._snapshot.arm
            if (
                not arm.connected
                or not arm.enabled
                or arm.fault_code != 0
                or not arm.feedback_fresh
            ):
                raise StateConflict("control mode change requires healthy feedback")
            self._control_mode = mode
            self._state = ServiceState.RUNNING
            self._snapshot = replace(
                self._snapshot,
                control_mode=mode,
                state=ServiceState.RUNNING,
                moving=False,
                hold=True,
                target_qd=ZERO_JOINTS,
                target_gripper_speed=0.0,
            )
        self._wake_event.set()

    def request_recovery(self) -> None:
        LOGGER.info("event=recovery_request_received side=%s", self._side_name)
        with self._state_lock:
            if self._state is not ServiceState.EMERGENCY_STOP:
                raise StateConflict("recovery requires EMERGENCY_STOP state")
            with self._emergency_lock:
                if not self._emergency_latched.is_set():
                    raise StateConflict("recovery requires latched emergency stop")
                self._recovery_request_generation = self._emergency_generation
                self._recovery_requested = True
        LOGGER.info("event=recovery_queued side=%s", self._side_name)
        self._wake_event.set()

    def emergency_stop(self, reason: str) -> None:
        """Latch and send an arm emergency stop without taking the state lock."""

        if not reason:
            raise ValueError("emergency stop reason must not be empty")
        self._latch_emergency(reason)
        self._deliver_pending_emergency()
        self._wake_event.set()

    def publish_runtime_fault(self, reason: str) -> None:
        """Publish a terminal worker fault without waiting for another tick.

        This boundary is used after the owning control worker has terminated.
        It latches emergency semantics first, then atomically replaces the
        immutable public snapshot under the normal state lock. Hardware E-stop
        and HOLD attempts remain asynchronous so a blocked SDK cannot prevent
        fault observability.
        """

        if not reason:
            raise ValueError("runtime fault reason must not be empty")
        self._latch_emergency(reason)
        with self._state_lock:
            self._apply_emergency_latch(self._latest_feedback)
            self._publish_emergency_snapshot()
        self._start_timeout_safety_attempts()
        self._wake_event.set()

    def _latch_emergency(self, reason: str) -> int:
        with self._emergency_lock:
            already_latched = self._emergency_latched.is_set()
            self._emergency_generation += 1
            if not already_latched:
                self._emergency_reason = reason
                self._emergency_delivered = False
            elif (
                self._recovery_in_progress
                or self._recovery_request_generation is not None
            ):
                self._emergency_delivered = False
            self._emergency_latched.set()
            generation = self._emergency_generation
        self._servo_mailbox.take()
        self._gripper_mailbox.take()
        return generation

    def _deliver_pending_emergency(self) -> tuple[str, ...]:
        with self._emergency_operation_lock:
            return self._deliver_pending_emergency_locked()

    def _deliver_pending_emergency_locked(self) -> tuple[str, ...]:
        errors: list[str] = []
        with self._emergency_lock:
            pending = (
                self._emergency_latched.is_set() and not self._emergency_delivered
            )
        if pending:
            try:
                self._driver.emergency_stop()
            except Exception as exc:
                errors.append(
                    f"{self._side_name} emergency stop failed: {exc}"
                )
            else:
                with self._emergency_lock:
                    if self._emergency_latched.is_set():
                        self._emergency_delivered = self._connected
        self._record_safety_errors(errors)
        return tuple(errors)

    def _force_emergency_locked(self) -> tuple[str, ...]:
        with self._emergency_lock:
            self._emergency_delivered = False
            self._emergency_latched.set()
        return self._deliver_pending_emergency_locked()

    def client_disconnected(self) -> None:
        """Immediately discard pending motion and HOLD the latest feedback."""

        with self._state_lock:
            self._servo_mailbox.take()
            self._gripper_mailbox.take()
            self._disconnect_hold.set()
            self._disconnect_hold_needs_fresh_latch = (
                self._state is ServiceState.RUNNING
            )
            if (
                self._shutdown_requested.is_set()
                or self._state is ServiceState.SHUTTING_DOWN
            ):
                return
            if self._latest_feedback is not None:
                errors = self._hold_best_effort(
                    preserve_gripper_target=self._state is ServiceState.HOLD
                )
                if errors:
                    self._enter_fault("disconnect HOLD failed", self._latest_feedback)
                self._snapshot = replace(
                    self._snapshot,
                    moving=False,
                    hold=True,
                    target_q=self._target,
                    target_qd=ZERO_JOINTS,
                    target_gripper=self._gripper_target,
                    target_gripper_speed=0.0,
                )

    def shutdown_gracefully(
        self,
        *,
        wait: Callable[[float], object] | None = None,
    ) -> ShutdownResult:
        """Home a healthy controller, or HOLD immediately on fault/timeout."""

        started_at = self._clock()
        self._shutdown_started_at = started_at
        self._shutdown_deadline = (
            started_at + self._config.shutdown_home_timeout_s
        )
        caller_deadline = time.monotonic() + self._config.shutdown_home_timeout_s

        done = Event()
        result: list[ShutdownResult] = []

        def perform_shutdown() -> None:
            try:
                started_normally = self._request_shutdown_at(
                    started_at, self._shutdown_deadline
                )
                if not started_normally:
                    with self._state_lock:
                        errors = self._hold_best_effort()
                    result.append(
                        ShutdownResult(
                            homed=False,
                            faulted=True,
                            errors=self._result_errors(errors),
                        )
                    )
                else:
                    result.append(
                        self._run_shutdown_sequence(
                            wait or self._wait_for_shutdown_period
                        )
                    )
            except Exception as exc:
                error = f"shutdown worker failed: {exc}"
                self._record_safety_errors((error,))
                self._latch_emergency(error)
                self._deliver_pending_emergency()
                result.append(
                    ShutdownResult(
                        homed=False,
                        faulted=True,
                        errors=self._result_errors((error,)),
                    )
                )
            finally:
                done.set()

        worker = Thread(
            target=perform_shutdown,
            name="piper-shutdown",
            daemon=True,
        )
        worker.start()
        remaining = max(0.0, caller_deadline - time.monotonic())
        if not done.wait(remaining):
            timeout_error = "shutdown homing timeout during blocking SDK operation"
            self._record_safety_errors((timeout_error,))
            self._latch_emergency("shutdown homing timeout")
            Thread(
                target=self._deliver_pending_emergency,
                name="piper-shutdown-emergency",
                daemon=True,
            ).start()
            return ShutdownResult(
                homed=False,
                timed_out=True,
                errors=self._result_errors((timeout_error,)),
            )
        return result[0]

    def _run_shutdown_sequence(
        self, wait_for_period: Callable[[float], object]
    ) -> ShutdownResult:
        period = 1.0 / self._config.control_hz
        while True:
            if self._shutdown_deadline_reached():
                return self._finish_shutdown_timeout()
            snapshot = self.tick(dt=period)
            if self._shutdown_deadline_reached():
                return self._finish_shutdown_timeout()
            if snapshot.state is ServiceState.EMERGENCY_STOP:
                with self._state_lock:
                    errors = self._hold_best_effort()
                return ShutdownResult(
                    homed=False,
                    faulted=True,
                    errors=self._result_errors(errors),
                )
            if self._homing_stable_cycles >= self._HOMING_STABLE_CYCLES:
                with self._state_lock:
                    errors = self._hold_best_effort()
                    if errors:
                        self._enter_fault("final shutdown HOLD failed", self._latest_feedback)
                return ShutdownResult(
                    homed=not errors,
                    faulted=bool(errors),
                    errors=self._result_errors(errors),
                )
            wait_for_period(period)
            if self._shutdown_deadline_reached():
                return self._finish_shutdown_timeout()

    def _shutdown_deadline_reached(self) -> bool:
        return (
            self._shutdown_deadline is not None
            and self._clock() >= self._shutdown_deadline
        )

    def _finish_shutdown_timeout(self) -> ShutdownResult:
        error = "shutdown homing timeout"
        self._record_safety_errors((error,))
        self._latch_emergency(error)
        self._deliver_pending_emergency()
        with self._state_lock:
            self._apply_emergency_latch(self._latest_feedback)
            errors = self._hold_best_effort()
            self._snapshot = replace(
                self._snapshot,
                moving=False,
                hold=True,
                fault_reason=self._fault_reason,
                target_qd=ZERO_JOINTS,
                target_gripper_speed=0.0,
            )
        return ShutdownResult(
            homed=False,
            timed_out=True,
            errors=self._result_errors(errors),
        )

    def _wait_for_shutdown_period(self, period: float) -> None:
        self._wake_event.wait(period)
        self._wake_event.clear()

    def request_shutdown(self) -> None:
        started_at = self._clock()
        deadline = started_at + self._config.shutdown_home_timeout_s
        self._request_shutdown_at(started_at, deadline)
        self._shutdown_transition_started.set()
        self._wake_event.set()

    def request_shutdown_async(self) -> None:
        """Request production shutdown without waiting for tick/state/SDK locks."""

        if self._shutdown_requested.is_set():
            return
        started_at = self._clock()
        self._shutdown_started_at = started_at
        self._shutdown_deadline = (
            started_at + self._config.shutdown_home_timeout_s
        )
        self._servo_mailbox.take()
        self._gripper_mailbox.take()
        self._shutdown_requested.set()
        self._wake_event.set()

    def shutdown_result(self) -> ShutdownResult | None:
        """Return the control-thread shutdown outcome without taking state_lock."""

        with self._shutdown_result_lock:
            return self._shutdown_result

    def request_timeout_safety_async(self, reason: str) -> None:
        """Latch emergency and asynchronously attempt per-arm E-stop and HOLD.

        This is reserved for process shutdown when the tick owner is blocked in
        SDK I/O. Emergency-stop and HOLD attempts use separate daemon threads.
        No power-off or controller-reset primitive is part of the driver contract.
        """

        self._latch_emergency(reason)
        self._start_timeout_safety_attempts()

    def _start_timeout_safety_attempts(self) -> None:
        """Attempt already-latched emergency safety without blocking caller."""

        feedback = self._last_valid_feedback

        def attempt(operation: str) -> None:
            try:
                if operation == "emergency":
                    self._deliver_pending_emergency()
                else:
                    assert feedback is not None
                    self._driver.hold(feedback)
            except Exception as exc:
                self._record_safety_errors(
                    (f"{self._side_name} asynchronous {operation} failed: {exc}",)
                )

        Thread(
            target=attempt,
            args=("emergency",),
            name=f"piper-timeout-emergency-{self._side_name}",
            daemon=True,
        ).start()
        if feedback is not None:
            Thread(
                target=attempt,
                args=("HOLD",),
                name=f"piper-timeout-hold-{self._side_name}",
                daemon=True,
            ).start()

    def _request_shutdown_at(
        self, started_at: float, deadline: float | None
    ) -> bool:
        with self._state_lock:
            self._servo_mailbox.take()
            self._gripper_mailbox.take()
            self._shutdown_started_at = started_at
            self._shutdown_deadline = deadline
            self._shutdown_requested.set()
            if (
                self._emergency_latched.is_set()
                or self._state is ServiceState.EMERGENCY_STOP
            ):
                self._apply_emergency_latch(self._latest_feedback)
                return False
            self._state = ServiceState.SHUTTING_DOWN
            self._homing_stable_cycles = 0
            self._target = self._config.reset_joints_pos_deg
            self._joint_speed = self._config.default_joints_speed_deg_s
            sample = self._last_valid_feedback
            if sample is not None:
                self._commanded_q = sample.joints_deg
                self._gripper_target = sample.gripper_position
            self._gripper_speed = 0.0
            self._snapshot = replace(
                self._snapshot,
                state=self._state,
                moving=self._state is ServiceState.SHUTTING_DOWN,
                hold=self._state is not ServiceState.SHUTTING_DOWN,
                target_q=self._target,
                target_gripper=self._gripper_target,
                target_qd=ZERO_JOINTS,
                target_gripper_speed=0.0,
            )
        return True

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = Thread(
                target=self.run,
                name="piper-control",
                daemon=True,
            )
            self._thread.start()

    def run(self) -> None:
        """Run until shutdown, then release every driver resource exactly once."""

        try:
            self._run_until_shutdown()
        finally:
            try:
                self._driver.shutdown()
            except Exception as exc:
                error = f"{self._side_name} driver shutdown failed: {exc}"
                self._record_safety_errors((error,))
                current = self.shutdown_result()
                self._set_shutdown_result(
                    ShutdownResult(
                        homed=False if current is None else current.homed,
                        timed_out=False if current is None else current.timed_out,
                        faulted=True,
                        errors=self._result_errors((error,)),
                    )
                )

    def _run_until_shutdown(self) -> None:
        """Run ticks at the configured frequency until shutdown is requested."""

        period = 1.0 / self._config.control_hz
        last_tick = self._clock()
        deadline = last_tick
        while True:
            observed_before_wait = self._clock()
            delay = deadline - observed_before_wait
            if delay > period * 1.5:
                LOGGER.warning(
                    "event=control_scheduler_deadline_reset side=%s "
                    "delay_s=%.6f period_s=%.6f",
                    self._side_name,
                    delay,
                    period,
                )
                deadline = observed_before_wait + period
                delay = period
            woke_early = False
            if delay > 0:
                woke_early = self._wake_event.wait(delay)
                self._wake_event.clear()
            now = self._clock()
            elapsed = period if deadline == last_tick else max(now - last_tick, 0.0)
            self.tick(dt=elapsed)
            if self._shutdown_requested.is_set():
                if self._state is ServiceState.EMERGENCY_STOP:
                    with self._state_lock:
                        errors = self._hold_best_effort()
                    self._set_shutdown_result(
                        ShutdownResult(
                            homed=False,
                            faulted=True,
                            errors=self._result_errors(errors),
                        )
                    )
                    return
                if self._homing_stable_cycles >= self._HOMING_STABLE_CYCLES:
                    with self._state_lock:
                        errors = self._hold_best_effort()
                        if errors:
                            self._enter_fault(
                                "final shutdown HOLD failed", self._latest_feedback
                            )
                    self._set_shutdown_result(
                        ShutdownResult(
                            homed=not errors,
                            faulted=bool(errors),
                            errors=self._result_errors(errors),
                        )
                    )
                    return
                if self._shutdown_deadline_reached():
                    self._set_shutdown_result(self._finish_shutdown_timeout())
                    return
            deadline += period
            finished_at = self._clock()
            if woke_early:
                # Command and safety events intentionally interrupt the
                # periodic sleep.  Rebase after servicing one so repeated
                # wakeups cannot accumulate future deadlines and eventually
                # park the control loop for an unbounded interval.
                deadline = finished_at + period
                last_tick = finished_at
            elif deadline <= finished_at:
                deadline = finished_at + period
                last_tick = finished_at
            else:
                last_tick = now

    def _set_shutdown_result(self, result: ShutdownResult) -> None:
        with self._shutdown_result_lock:
            self._shutdown_result = result

    def tick(self, dt: float | None = None) -> ArmStateSnapshot:
        """Execute exactly one deterministic control cycle."""

        with self._tick_lock, self._state_lock:
            cycle_started_at = self._clock()
            raw_dt = 1.0 / self._config.control_hz if dt is None else float(dt)
            if not isfinite(raw_dt) or raw_dt <= 0:
                raise ValueError("dt must be a positive finite number")
            suppress_scheduler_gap = self._suppress_next_scheduler_overrun
            self._suppress_next_scheduler_overrun = False
            scheduler_dt = (
                1.0 / self._config.control_hz
                if suppress_scheduler_gap
                else raw_dt
            )
            max_dt = 2.0 / self._config.control_hz
            actual_dt = min(scheduler_dt, max_dt)
            if (
                self._shutdown_requested.is_set()
                and not self._shutdown_transition_started.is_set()
            ):
                started_at = self._shutdown_started_at or self._clock()
                self._request_shutdown_at(started_at, self._shutdown_deadline)
                self._shutdown_transition_started.set()
            if scheduler_dt > max_dt:
                self._overrun_cycles += 1
                if self._overrun_cycles == 1:
                    LOGGER.warning(
                        "event=control_overrun dt=%s limit=%s",
                        scheduler_dt,
                        max_dt,
                        extra={
                            "event": "control_overrun",
                            "dt": scheduler_dt,
                            "limit": max_dt,
                        },
                    )
            else:
                self._overrun_cycles = 0

            previous_feedback = self._latest_feedback
            try:
                if self._state is ServiceState.STARTING:
                    self._start_driver()
                    feedback = self._read_startup_feedback()
                elif self._emergency_latched.is_set():
                    self._connect_missing_for_emergency()
                    self._deliver_pending_emergency()
                    feedback = self._driver.read_feedback()
                else:
                    feedback = self._driver.read_feedback()
            except Exception as exc:
                self._enter_fault(f"arm I/O error: {exc}", self._latest_feedback)
                return self._publish(self._latest_feedback, actual_dt)

            raw_feedback = feedback
            if self._feedback_numeric_fault(raw_feedback) is None:
                self._last_valid_feedback = raw_feedback
            feedback = self._finite_feedback(raw_feedback, previous_feedback)
            self._latest_feedback = feedback
            cycle_delay = max(scheduler_dt, self._clock() - cycle_started_at)
            if cycle_delay >= self._config.feedback_timeout_s:
                if self._overrun_cycles == 0:
                    LOGGER.warning(
                        "event=control_overrun dt=%s limit=%s",
                        cycle_delay,
                        max_dt,
                        extra={
                            "event": "control_overrun",
                            "dt": cycle_delay,
                            "limit": max_dt,
                        },
                    )
                self._overrun_cycles = max(
                    self._overrun_cycles,
                    self._OVERRUN_LIMIT,
                )
            unhealthy = self._health_fault(raw_feedback)
            if unhealthy is None:
                unhealthy = self._feedback_timeout_fault(raw_feedback, self._clock())
            try:
                self._apply_emergency_latch(feedback)
                if (
                    self._state is ServiceState.EMERGENCY_STOP
                    and self._recovery_requested
                ):
                    feedback = self._recover(feedback, raw_feedback)
                elif unhealthy is not None:
                    self._enter_fault(unhealthy, feedback)
                elif self._overrun_cycles >= self._OVERRUN_LIMIT:
                    if self._overrun_cycles == self._OVERRUN_LIMIT:
                        LOGGER.error(
                            "event=control_overrun_fault cycles=%s",
                            self._overrun_cycles,
                            extra={
                                "event": "control_overrun_fault",
                                "cycles": self._overrun_cycles,
                            },
                        )
                    self._enter_fault("control loop overrun", feedback)
                elif self._state is ServiceState.EMERGENCY_STOP:
                    self._hold_best_effort()
                elif self._state is ServiceState.SHUTTING_DOWN:
                    self._advance_shutdown_state(feedback, actual_dt)
                else:
                    self._advance_normal_state(feedback, actual_dt)
            except Exception as exc:
                self._enter_fault(f"arm I/O error: {exc}", feedback)

            return self._publish(feedback, actual_dt)

    def _start_driver(self) -> None:
        if self._emergency_latched.is_set():
            self._deliver_pending_emergency()
        if not self._connected:
            self._driver.connect()
            with self._emergency_lock:
                self._connected = True
        if self._emergency_latched.is_set():
            self._deliver_pending_emergency()
            return
        self._driver.enable()
        if self._emergency_latched.is_set():
            self._deliver_pending_emergency()

    def _read_startup_feedback(self) -> ArmFeedback:
        deadline = self._clock() + self._config.feedback_timeout_s
        period = 1.0 / self._config.control_hz
        while True:
            feedback = self._driver.read_feedback()
            if not feedback.stale or self._emergency_latched.is_set():
                return feedback
            remaining = deadline - self._clock()
            if remaining <= 0:
                return feedback
            self._sleep(min(period, remaining))

    def _connect_missing_for_emergency(self) -> None:
        with self._emergency_lock:
            connected = self._connected
        if connected:
            return
        try:
            self._driver.connect()
        except Exception as exc:
            self._record_safety_errors(
                (f"{self._side_name} emergency connect failed: {exc}",)
            )
            return
        with self._emergency_lock:
            self._connected = True
        self._deliver_pending_emergency()

    def _finite_feedback(
        self,
        sample: ArmFeedback,
        previous: ArmFeedback | None,
    ) -> ArmFeedback:
        old = self._last_valid_feedback or previous or ArmFeedback()
        safe_joints = all(isfinite(value) for value in sample.joints_deg) and all(
            minimum - self._FEEDBACK_JOINT_TOLERANCE_DEG
            <= value
            <= maximum + self._FEEDBACK_JOINT_TOLERANCE_DEG
            for value, minimum, maximum in zip(
                sample.joints_deg,
                self._config.min_joints_deg,
                self._config.max_joints_deg,
                strict=True,
            )
        )
        safe_gripper = isfinite(sample.gripper_position) and (
            not self._config.gripper.enabled
            or self._config.gripper.physical_min_position
            - self._FEEDBACK_GRIPPER_TOLERANCE_M
            <= sample.gripper_position
            <= self._config.gripper.physical_max_position
            + self._FEEDBACK_GRIPPER_TOLERANCE_M
        )
        return replace(
            sample,
            joints_deg=(
                sample.joints_deg if safe_joints else old.joints_deg
            ),
            gripper_position=(
                sample.gripper_position if safe_gripper else old.gripper_position
            ),
            tcp_pose=(
                sample.tcp_pose
                if all(isfinite(value) for value in sample.tcp_pose)
                else old.tcp_pose
            ),
            received_at_s=(
                sample.received_at_s
                if isfinite(sample.received_at_s)
                else old.received_at_s
            ),
            robot_timestamp_s=(
                sample.robot_timestamp_s
                if sample.robot_timestamp_s is None
                or isfinite(sample.robot_timestamp_s)
                else old.robot_timestamp_s
            ),
        )

    def _health_fault(self, sample: ArmFeedback) -> str | None:
        gripper_fault = self._gripper_health_fault(
            sample, require_enabled=True
        )
        if gripper_fault is not None:
            return gripper_fault
        if not sample.connected:
            return f"{self._side_name} arm disconnected"
        if not sample.enabled:
            return f"{self._side_name} arm disabled"
        if sample.stale:
            return f"{self._side_name} arm feedback stale"
        if sample.fault_code:
            return f"{self._side_name} arm fault code {sample.fault_code}"
        numeric_fault = self._feedback_numeric_fault(sample)
        if numeric_fault is not None:
            return numeric_fault
        return None

    def _feedback_numeric_fault(self, sample: ArmFeedback) -> str | None:
        numeric = (
            *sample.joints_deg,
            sample.gripper_position,
            *sample.tcp_pose,
            sample.received_at_s,
        )
        if sample.robot_timestamp_s is not None:
            numeric = (*numeric, sample.robot_timestamp_s)
        if not all(isfinite(value) for value in numeric):
            return f"{self._side_name} arm feedback is non-finite"
        for index, (value, minimum, maximum) in enumerate(
            zip(
                sample.joints_deg,
                self._config.min_joints_deg,
                self._config.max_joints_deg,
                strict=True,
            ),
            1,
        ):
            if not (
                minimum - self._FEEDBACK_JOINT_TOLERANCE_DEG
                <= value
                <= maximum + self._FEEDBACK_JOINT_TOLERANCE_DEG
            ):
                return (
                    f"{self._side_name} joint {index} feedback {value:g} is outside "
                    f"[{minimum:g}, {maximum:g}] degrees"
                )
        gripper = self._config.gripper
        if gripper.enabled and not (
            gripper.physical_min_position - self._FEEDBACK_GRIPPER_TOLERANCE_M
            <= sample.gripper_position
            <= gripper.physical_max_position + self._FEEDBACK_GRIPPER_TOLERANCE_M
        ):
            return (
                f"{self._side_name} gripper feedback "
                f"{sample.gripper_position:g} is outside "
                f"[{gripper.physical_min_position:g}, "
                f"{gripper.physical_max_position:g}] {gripper.position_unit}"
            )
        return None

    def _feedback_timeout_fault(
        self,
        sample: ArmFeedback,
        observed_at: float,
    ) -> str | None:
        previous_timestamp = self._last_progress_receive_timestamp
        if previous_timestamp is None or sample.received_at_s > previous_timestamp:
            self._last_progress_receive_timestamp = sample.received_at_s
            self._receive_timestamp_changed_at = observed_at
            return None
        assert self._receive_timestamp_changed_at is not None
        if observed_at - self._receive_timestamp_changed_at > self._config.feedback_timeout_s:
            return f"{self._side_name} arm feedback timeout"
        return None

    def _advance_normal_state(
        self, feedback: ArmFeedback, dt: float
    ) -> None:
        if self._state is ServiceState.STARTING:
            if self._config.auto_initialize:
                self._state = ServiceState.HOMING
                self._homing_started_at_s = self._clock()
                self._homing_trigger = "automatic"
                LOGGER.info(
                    "event=homing_started side=%s trigger=automatic",
                    self._side_name,
                )
                self._target = self._config.reset_joints_pos_deg
                self._joint_speed = self._config.default_joints_speed_deg_s
                if self._config.gripper.enabled:
                    self._gripper_target = feedback.gripper_position
                    self._gripper_speed = 0.0
                    self._commanded_gripper = feedback.gripper_position
            else:
                self._state = ServiceState.RUNNING
                self._target = feedback.joints_deg
                self._commanded_q = feedback.joints_deg
                self._gripper_target = (
                    feedback.gripper_position
                    if self._config.gripper.enabled
                    else 0.0
                )
                errors = self._hold_best_effort()
                if errors:
                    raise RuntimeError("; ".join(errors))
                return
            if self._commanded_q is None:
                self._commanded_q = feedback.joints_deg

        if self._state is ServiceState.HOMING:
            arrived = all(
                abs(current - target) <= self._ARRIVAL_TOLERANCE_DEG
                for current, target in zip(
                    feedback.joints_deg,
                    self._config.reset_joints_pos_deg,
                    strict=True,
                )
            )
            self._homing_stable_cycles = (
                self._homing_stable_cycles + 1 if arrived else 0
            )
            homing_elapsed_s = (
                None
                if self._homing_started_at_s is None
                else self._clock() - self._homing_started_at_s
            )
            homing_timed_out = (
                homing_elapsed_s is not None
                and homing_elapsed_s >= self._config.homing_timeout_s
            )
            if self._homing_stable_cycles >= self._HOMING_STABLE_CYCLES:
                LOGGER.info(
                    "event=homing_completed side=%s trigger=%s "
                    "elapsed_s=%.6f stable_cycles=%s",
                    self._side_name,
                    self._homing_trigger or "unknown",
                    0.0
                    if self._homing_started_at_s is None
                    else self._clock() - self._homing_started_at_s,
                    self._homing_stable_cycles,
                )
                self._state = ServiceState.RUNNING
                self._homing_started_at_s = None
                self._homing_trigger = None
            elif homing_timed_out:
                LOGGER.warning(
                    "event=homing_timeout side=%s trigger=%s elapsed_s=%.3f",
                    self._side_name,
                    self._homing_trigger or "unknown",
                    homing_elapsed_s,
                )
                self._target = feedback.joints_deg
                self._gripper_target = (
                    feedback.gripper_position
                    if self._config.gripper.enabled
                    else 0.0
                )
                self._state = ServiceState.RUNNING
                self._homing_started_at_s = None
                self._homing_trigger = None
                errors = self._hold_best_effort()
                if errors:
                    raise RuntimeError("; ".join(errors))
                return

        if self._state is ServiceState.HOLD:
            errors = self._hold_best_effort(preserve_gripper_target=True)
            if errors:
                raise RuntimeError("; ".join(errors))
            return

        if self._state is ServiceState.RUNNING:
            if self._disconnect_hold.is_set():
                errors = self._hold_best_effort(
                    preserve_gripper_target=(
                        not self._disconnect_hold_needs_fresh_latch
                    )
                )
                self._disconnect_hold_needs_fresh_latch = False
                if errors:
                    raise RuntimeError("; ".join(errors))
                return
            servo = self._servo_mailbox.take()
            if servo is not None:
                self._target = servo.target
                self._joint_speed = servo.speed
            gripper = self._gripper_mailbox.take()
            if gripper is not None:
                self._gripper_target = gripper.target
                self._gripper_speed = gripper.speed

        self._command_interpolated(feedback, dt)

    def _advance_shutdown_state(
        self, feedback: ArmFeedback, dt: float
    ) -> None:
        arrived = all(
            abs(current - target) <= self._ARRIVAL_TOLERANCE_DEG
            for current, target in zip(
                feedback.joints_deg,
                self._config.reset_joints_pos_deg,
                strict=True,
            )
        )
        self._homing_stable_cycles = (
            self._homing_stable_cycles + 1 if arrived else 0
        )
        if self._homing_stable_cycles >= self._HOMING_STABLE_CYCLES:
            errors = self._hold_best_effort()
            if errors:
                self._enter_fault("final shutdown HOLD failed", feedback)
            return
        self._command_shutdown_joints(feedback, dt)

    def _command_shutdown_joints(
        self, feedback: ArmFeedback, dt: float
    ) -> None:
        if self._shutdown_deadline_reached():
            self._latch_emergency("shutdown homing timeout")
            self._apply_emergency_latch(feedback)
            return
        if self._emergency_latched.is_set():
            self._apply_emergency_latch(feedback)
            return
        commanded = self._commanded_q or feedback.joints_deg
        advanced = step_axes(commanded, self._target, self._joint_speed, dt)
        next_joints = self._bounded_joint_command(
            tuple(
                max(
                    actual - self._MAX_COMMAND_LEAD_DEG,
                    min(value, actual + self._MAX_COMMAND_LEAD_DEG),
                )
                for value, actual in zip(
                    advanced, feedback.joints_deg, strict=True
                )
            )
        )
        self._driver.command_joints(next_joints)
        self._commanded_q = next_joints
        if self._shutdown_deadline_reached():
            self._latch_emergency("shutdown homing timeout")
            self._apply_emergency_latch(feedback)
            return
        if self._emergency_latched.is_set():
            self._apply_emergency_latch(feedback)

    def _command_interpolated(
        self, feedback: ArmFeedback, dt: float
    ) -> None:
        if self._emergency_latched.is_set():
            self._apply_emergency_latch(feedback)
            return
        commanded = self._commanded_q or feedback.joints_deg
        advanced = step_axes(commanded, self._target, self._joint_speed, dt)
        next_joints = self._bounded_joint_command(
            tuple(
                max(
                    actual - self._MAX_COMMAND_LEAD_DEG,
                    min(value, actual + self._MAX_COMMAND_LEAD_DEG),
                )
                for value, actual in zip(
                    advanced, feedback.joints_deg, strict=True
                )
            )
        )
        self._driver.command_joints(next_joints)
        self._commanded_q = next_joints
        if self._emergency_latched.is_set():
            self._apply_emergency_latch(feedback)
            return
        if self._config.gripper.enabled:
            command_min = self._config.gripper.command_min_position
            command_max = self._config.gripper.command_max_position
            if (
                self._gripper_speed == 0.0
                and not command_min <= self._gripper_target <= command_max
            ):
                # Some external grippers can report a wider physical range than
                # callers are allowed to command.  HOLD is already applied by
                # the driver; never replay that physical-only value through the
                # public command clamp, which would create unintended motion.
                return
            commanded_gripper = (
                self._commanded_gripper
                if self._commanded_gripper is not None
                else feedback.gripper_position
            )
            advanced_gripper = step_axes(
                (commanded_gripper,),
                (self._gripper_target,),
                (self._gripper_speed,),
                dt,
            )[0]
            next_gripper = max(
                command_min,
                min(
                    max(
                        feedback.gripper_position
                        - self._config.gripper.command_lead_limit,
                        min(
                            advanced_gripper,
                            feedback.gripper_position
                            + self._config.gripper.command_lead_limit,
                        ),
                    ),
                    command_max,
                ),
            )
            self._driver.command_gripper(next_gripper)
            self._commanded_gripper = next_gripper

    def _bounded_joint_command(self, joints: tuple[float, ...]) -> Joint6:
        return tuple(
            max(minimum, min(value, maximum))
            for value, minimum, maximum in zip(
                joints,
                self._config.min_joints_deg,
                self._config.max_joints_deg,
                strict=True,
            )
        )  # type: ignore[return-value]

    def _hold_best_effort(
        self, *, preserve_gripper_target: bool = False
    ) -> tuple[str, ...]:
        errors: list[str] = []
        sample = self._last_valid_feedback
        if sample is None:
            errors.append(
                f"{self._side_name} HOLD unavailable: no valid feedback"
            )
        else:
            self._target = sample.joints_deg
            self._commanded_q = sample.joints_deg
            hold_sample = sample
            if self._config.gripper.enabled:
                if preserve_gripper_target:
                    hold_sample = replace(
                        sample,
                        gripper_position=self._gripper_target,
                    )
                else:
                    self._gripper_target = sample.gripper_position
                self._commanded_gripper = self._gripper_target
            else:
                self._commanded_gripper = None
                self._gripper_target = 0.0
            self._gripper_speed = 0.0
            try:
                self._driver.hold(hold_sample)
            except Exception as exc:
                errors.append(f"{self._side_name} HOLD failed: {exc}")
        self._record_safety_errors(errors)
        return tuple(errors)

    def _enter_fault(
        self, reason: str, feedback: ArmFeedback | None
    ) -> None:
        if not self._emergency_latched.is_set():
            LOGGER.error(
                "event=controller_fault side=%s state=%s reason=%s",
                self._side_name,
                self._state.value,
                reason,
            )
        self._latch_emergency(reason)
        self._apply_emergency_latch(feedback)
        self._publish_emergency_snapshot()
        self._deliver_pending_emergency()
        self._wake_event.set()

    def _publish_emergency_snapshot(self) -> None:
        self._snapshot = replace(
            self._snapshot,
            state=ServiceState.EMERGENCY_STOP,
            emergency_stopped=True,
            moving=False,
            hold=True,
            fault_reason=self._fault_reason,
            target_q=self._target,
            target_qd=ZERO_JOINTS,
            target_gripper=self._gripper_target,
            target_gripper_speed=0.0,
        )

    def _apply_emergency_latch(
        self, feedback: ArmFeedback | None
    ) -> None:
        if not self._emergency_latched.is_set():
            return
        self._state = ServiceState.EMERGENCY_STOP
        if self._fault_reason is None:
            self._fault_reason = self._emergency_reason
        self._servo_mailbox.take()
        self._gripper_mailbox.take()
        sample = self._last_valid_feedback
        if sample is not None:
            self._target = sample.joints_deg
            self._gripper_target = (
                sample.gripper_position if self._config.gripper.enabled else 0.0
            )

    def _recover(
        self, feedback: ArmFeedback, raw_feedback: ArmFeedback
    ) -> ArmFeedback:
        recovery_error: str | None = None
        recovered_feedback: ArmFeedback | None = None
        with self._emergency_operation_lock:
            with self._emergency_lock:
                generation = self._recovery_request_generation
                request_is_current = (
                    generation is not None
                    and generation == self._emergency_generation
                    and self._emergency_latched.is_set()
                )
                self._recovery_requested = False
                self._recovery_request_generation = None
                self._recovery_in_progress = request_is_current
            if not request_is_current:
                self._deliver_pending_emergency_locked()
                self._state = ServiceState.EMERGENCY_STOP
                return feedback
            assert generation is not None
            recovery_started_at_s = self._clock()
            LOGGER.info("event=recovery_started side=%s", self._side_name)
            recovery_input, recovery_error = self._wait_for_recovery_input(
                raw_feedback, generation
            )
            if recovery_input is not None:
                feedback = self._finite_feedback(recovery_input, feedback)
            if recovery_error is None:
                with self._emergency_lock:
                    changed = self._emergency_generation != generation
                if changed:
                    recovery_error = "new emergency requested during recovery"
                else:
                    try:
                        self._driver.recover_emergency()
                    except Exception as exc:
                        recovery_error = (
                            f"{self._side_name} emergency recovery failed: {exc}"
                        )
                    with self._emergency_lock:
                        changed = self._emergency_generation != generation
                    if changed:
                        recovery_error = "new emergency requested during recovery"

            # Releasing the Piper emergency stop is asynchronous.  Do not
            # re-enable while status/error feedback is still settling: doing
            # so can make the SDK briefly aggregate all six joints as lost.
            if recovery_error is None:
                released_feedback, recovery_error = (
                    self._wait_for_released_feedback(feedback, generation)
                )
                if released_feedback is not None:
                    feedback = self._finite_feedback(released_feedback, feedback)

            # EmergencyStop(0x02) clears the controller-side stop but the
            # production driver deliberately remains not-ready.  Re-establish
            # enable/motion mode before clearing the controller latch. No
            # ordinary command can run while this transaction owns the
            # emergency operation lock and the latch remains set.
            if recovery_error is None:
                with self._emergency_lock:
                    changed = self._emergency_generation != generation
                if changed:
                    recovery_error = "new emergency requested during recovery"
                else:
                    try:
                        self._driver.enable()
                    except Exception as exc:
                        recovery_error = (
                            f"{self._side_name} post-recovery enable failed: {exc}"
                        )
                    with self._emergency_lock:
                        changed = self._emergency_generation != generation
                    if changed:
                        recovery_error = "new emergency requested during recovery"

            if recovery_error is None:
                recovered_feedback, recovery_error = (
                    self._wait_for_recovered_feedback(feedback, generation)
                )

            if recovery_error is None:
                assert recovered_feedback is not None
                if self._feedback_numeric_fault(recovered_feedback) is None:
                    self._last_valid_feedback = recovered_feedback
                self._latest_feedback = recovered_feedback
                self._last_progress_receive_timestamp = (
                    recovered_feedback.received_at_s
                )
                self._receive_timestamp_changed_at = self._clock()
                hold_errors = self._hold_best_effort()
                if hold_errors:
                    recovery_error = (
                        "recovery HOLD failed: " + "; ".join(hold_errors)
                    )

            if recovery_error is None:
                with self._emergency_lock:
                    if self._emergency_generation != generation:
                        recovery_error = "new emergency requested during recovery"
                    else:
                        self._emergency_reason = None
                        self._emergency_latched.clear()
                        self._emergency_delivered = False
                        self._recovery_in_progress = False
            if recovery_error is not None:
                if recovery_error != "new emergency requested during recovery":
                    self._fault_reason = recovery_error
                self._record_safety_errors((recovery_error,))
                self._force_emergency_locked()
                with self._emergency_lock:
                    self._recovery_in_progress = False

            self._overrun_cycles = 0
            self._suppress_next_scheduler_overrun = True

        if recovery_error is not None:
            LOGGER.error(
                "event=recovery_failed side=%s elapsed_s=%.6f reason=%s",
                self._side_name,
                self._clock() - recovery_started_at_s,
                recovery_error,
            )
            self._state = ServiceState.EMERGENCY_STOP
            return feedback
        self._fault_reason = None
        self._safety_errors.clear()
        self._state = ServiceState.RUNNING
        LOGGER.info(
            "event=recovery_completed side=%s elapsed_s=%.6f",
            self._side_name,
            self._clock() - recovery_started_at_s,
        )
        assert recovered_feedback is not None
        return recovered_feedback

    def _wait_for_recovery_input(
        self, initial: ArmFeedback, generation: int
    ) -> tuple[ArmFeedback | None, str | None]:
        initial_error = self._recovery_input_fault(initial)
        if initial_error is None:
            return initial, None

        attempts = max(
            1,
            ceil(self._config.feedback_timeout_s * self._config.control_hz),
        )
        last_error = initial_error
        last_timestamp = initial.received_at_s
        stable_cycles = 0
        for _attempt in range(attempts):
            with self._emergency_lock:
                if self._emergency_generation != generation:
                    return None, "new emergency requested during recovery"
            self._sleep(1.0 / self._config.control_hz)
            try:
                feedback = self._driver.read_feedback()
            except Exception as exc:
                stable_cycles = 0
                last_error = f"recovery feedback failed: {exc}"
                continue

            input_error = self._recovery_input_fault(feedback)
            timestamp_advanced = feedback.received_at_s > last_timestamp
            if isfinite(feedback.received_at_s):
                last_timestamp = max(last_timestamp, feedback.received_at_s)
            if input_error is not None:
                stable_cycles = 0
                last_error = input_error
            elif not timestamp_advanced:
                stable_cycles = 0
                last_error = (
                    f"recovery rejected: {self._side_name} arm feedback "
                    "did not advance"
                )
            else:
                stable_cycles += 1
                if stable_cycles >= self._RECOVERY_STABLE_CYCLES:
                    return feedback, None
                last_error = (
                    "recovery feedback has not remained stable for "
                    f"{self._RECOVERY_STABLE_CYCLES} cycles"
                )
        return None, last_error

    def _wait_for_released_feedback(
        self, previous: ArmFeedback, generation: int
    ) -> tuple[ArmFeedback | None, str | None]:
        return self._wait_for_stable_recovery_feedback(
            previous,
            generation,
            duration_s=self._RECOVERY_RELEASE_STABLE_DURATION_S,
            acquisition_timeout_s=self._config.feedback_timeout_s,
            require_enabled=False,
            phase="post-release",
        )

    def _wait_for_recovered_feedback(
        self, previous: ArmFeedback, generation: int
    ) -> tuple[ArmFeedback | None, str | None]:
        return self._wait_for_stable_recovery_feedback(
            previous,
            generation,
            duration_s=self._RECOVERY_ENABLED_STABLE_DURATION_S,
            acquisition_timeout_s=self._RECOVERY_ENABLE_SETTLE_TIMEOUT_S,
            require_enabled=True,
            phase="post-recovery",
        )

    def _wait_for_stable_recovery_feedback(
        self,
        previous: ArmFeedback,
        generation: int,
        *,
        duration_s: float,
        acquisition_timeout_s: float,
        require_enabled: bool,
        phase: str,
    ) -> tuple[ArmFeedback | None, str | None]:
        observation_budget_s = acquisition_timeout_s + duration_s
        deadline = self._clock() + observation_budget_s
        attempts = max(
            1,
            ceil(observation_budget_s * self._config.control_hz),
        ) + 1
        last_error = f"{phase} feedback unavailable"
        stable_cycles = 0
        stable_started_at: float | None = None
        last_healthy_timestamp = previous.received_at_s
        for attempt in range(attempts):
            with self._emergency_lock:
                if self._emergency_generation != generation:
                    return None, "new emergency requested during recovery"
            read_started_at = self._clock()
            try:
                feedback = self._driver.read_feedback()
            except Exception as exc:
                stable_cycles = 0
                stable_started_at = None
                last_error = f"{phase} feedback failed: {exc}"
            else:
                if (
                    self._clock() - read_started_at
                    > self._config.feedback_timeout_s
                ):
                    return None, (
                        f"{phase} feedback deadline exceeded: {last_error}"
                    )
                if self._clock() > deadline:
                    return None, (
                        f"{phase} feedback deadline exceeded: {last_error}"
                    )
                unhealthy = (
                    self._health_fault(feedback)
                    if require_enabled
                    else self._released_feedback_fault(feedback)
                )
                if unhealthy is not None:
                    stable_cycles = 0
                    stable_started_at = None
                    last_error = f"{phase} health check failed: {unhealthy}"
                else:
                    if feedback.received_at_s > last_healthy_timestamp:
                        if stable_cycles == 0:
                            stable_started_at = feedback.received_at_s
                        stable_cycles += 1
                        last_healthy_timestamp = feedback.received_at_s
                        assert stable_started_at is not None
                        stable_duration = (
                            feedback.received_at_s - stable_started_at
                        )
                        if (
                            stable_cycles >= self._RECOVERY_STABLE_CYCLES
                            and stable_duration
                            >= duration_s - 1e-9
                        ):
                            return feedback, None
                        last_error = (
                            f"{phase} feedback has not remained stable for "
                            f"{duration_s:g} seconds"
                        )
                    else:
                        stable_cycles = 0
                        stable_started_at = None
                        last_error = (
                            f"{phase} health check failed: "
                            f"{self._side_name} arm feedback did not advance"
                        )
            if attempt + 1 < attempts:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return None, (
                        f"{phase} feedback deadline exceeded: {last_error}"
                    )
                self._sleep(
                    min(1.0 / self._config.control_hz, remaining)
                )
        return None, last_error

    def _released_feedback_fault(self, sample: ArmFeedback) -> str | None:
        if not sample.connected:
            return f"{self._side_name} arm disconnected"
        if sample.fault_code:
            return f"{self._side_name} arm fault code {sample.fault_code}"
        numeric_fault = self._feedback_numeric_fault(sample)
        if numeric_fault is not None:
            return numeric_fault
        return None

    def _gripper_health_fault(
        self, sample: ArmFeedback, *, require_enabled: bool
    ) -> str | None:
        diagnostics = sample.gripper
        if not self._config.gripper.enabled or diagnostics.backend == "unknown":
            return None
        if not diagnostics.communication_ok:
            return f"{self._side_name} gripper communication unavailable"
        if diagnostics.fault_code:
            return (
                f"{self._side_name} gripper fault code "
                f"{diagnostics.fault_code}"
            )
        if diagnostics.warning_code:
            return (
                f"{self._side_name} gripper warning code "
                f"{diagnostics.warning_code}"
            )
        if (
            require_enabled
            and diagnostics.backend != "piper"
            and not diagnostics.enabled
        ):
            return f"{self._side_name} gripper disabled"
        return None

    def _recovery_input_fault(
        self, sample: ArmFeedback
    ) -> str | None:
        if not sample.connected:
            return f"recovery rejected: {self._side_name} arm disconnected"
        if sample.fault_code:
            return (
                f"recovery rejected: {self._side_name} arm fault code "
                f"{sample.fault_code}"
            )
        numeric_error = self._feedback_numeric_fault(sample)
        if numeric_error is not None:
            return f"recovery rejected: {numeric_error}"
        return None

    def _record_safety_errors(self, errors) -> None:  # type: ignore[no-untyped-def]
        for error in errors:
            if error not in self._safety_errors:
                self._safety_errors.append(error)
                LOGGER.error("event=safety_error error=%s", error)

    def _result_errors(self, errors=()) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
        self._record_safety_errors(errors)
        return tuple(self._safety_errors)

    def _publish(
        self, feedback: ArmFeedback | None, dt: float
    ) -> ArmStateSnapshot:
        previous = self._previous_feedback
        sample = feedback or ArmFeedback()
        actual_qd: Joint6 = (
            tuple(
                (current - old) / dt
                for current, old in zip(
                    sample.joints_deg,
                    previous.joints_deg,
                    strict=True,
                )
            )
            if feedback is not None and previous is not None
            else ZERO_JOINTS
        )  # type: ignore[assignment]
        actual_gripper = (
            sample.gripper_position if self._config.gripper.enabled else 0.0
        )
        actual_gripper_speed = (
            (sample.gripper_position - previous.gripper_position) / dt
            if self._config.gripper.enabled
            and feedback is not None
            and previous is not None
            else 0.0
        )

        if feedback is not None and self._state in (
            ServiceState.HOMING,
            ServiceState.RUNNING,
            ServiceState.SHUTTING_DOWN,
        ):
            next_joints = step_axes(
                sample.joints_deg, self._target, self._joint_speed, dt
            )
            target_qd: Joint6 = tuple(
                (next_value - current) / dt
                for next_value, current in zip(
                    next_joints, sample.joints_deg, strict=True
                )
            )  # type: ignore[assignment]
            if self._config.gripper.enabled:
                next_gripper = step_axes(
                    (sample.gripper_position,),
                    (self._gripper_target,),
                    (self._gripper_speed,),
                    dt,
                )[0]
                target_gripper_speed = (
                    next_gripper - sample.gripper_position
                ) / dt
            else:
                target_gripper_speed = 0.0
        else:
            target_qd = ZERO_JOINTS
            target_gripper_speed = 0.0

        observed_at = self._clock()
        robot_timestamp = (
            sample.robot_timestamp_s
            if feedback is not None and sample.robot_timestamp_s is not None
            else sample.received_at_s - self._config.receive_latency_s
            if feedback is not None
            else 0.0
        )
        arm = ArmState(
            receive_timestamp=sample.received_at_s,
            robot_timestamp=robot_timestamp,
            connected=sample.connected,
            enabled=sample.enabled,
            fault_code=sample.fault_code,
            feedback_fresh=(
                feedback is not None
                and not sample.stale
                and (
                    self._receive_timestamp_changed_at is None
                    or observed_at - self._receive_timestamp_changed_at
                    <= self._config.feedback_timeout_s
                )
            ),
        )
        moving = any(target_qd) or bool(target_gripper_speed)
        snapshot = ArmStateSnapshot(
            side=self._side,
            control_mode=self._control_mode,
            state=self._state,
            emergency_stopped=self._state is ServiceState.EMERGENCY_STOP,
            moving=moving,
            hold=not moving,
            fault_reason=self._fault_reason,
            actual_q=sample.joints_deg,
            actual_qd=actual_qd,
            target_q=self._target,
            target_qd=target_qd,
            actual_tcp_pose=sample.tcp_pose,
            actual_tcp_speed=ZERO_JOINTS,
            actual_gripper=actual_gripper,
            actual_gripper_speed=actual_gripper_speed,
            target_gripper=self._gripper_target,
            target_gripper_speed=target_gripper_speed,
            robot_receive_timestamp=sample.received_at_s,
            robot_timestamp=robot_timestamp,
            arm=arm,
            gripper=sample.gripper,
        )
        with self._state_lock:
            self._snapshot = snapshot
            self._snapshot_published_at_s = observed_at
        self._previous_feedback = feedback
        return snapshot
