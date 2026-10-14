# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Deterministic in-memory implementation of the safe arm driver."""

from __future__ import annotations

from dataclasses import replace
from math import inf, nextafter
from threading import Event, RLock
import time
from typing import Callable

from piper_control.types import ArmFeedback, Joint6


class SimulatedArmDriver:
    def __init__(
        self,
        feedback: ArmFeedback | None = None,
        *,
        auto_timestamp: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lock = RLock()
        self._feedback = feedback or ArmFeedback()
        self._auto_timestamp = auto_timestamp
        self._clock = clock
        self._emergency_latched = Event()
        self.connect_calls = 0
        self.enable_calls = 0
        self.emergency_stop_calls = 0
        self.recover_emergency_calls = 0
        self.hold_calls = 0
        self.shutdown_calls = 0
        self.joint_commands: list[Joint6] = []
        self.gripper_commands: list[float] = []

    @property
    def disable_calls(self) -> int:
        return 0

    @property
    def reset_calls(self) -> int:
        return 0

    def connect(self) -> None:
        with self._lock:
            self.connect_calls += 1
            self._feedback = replace(self._feedback, connected=True)

    def enable(self) -> None:
        with self._lock:
            if self._emergency_latched.is_set():
                raise RuntimeError("simulated driver emergency stop is latched")
            self.enable_calls += 1
            self._feedback = replace(self._feedback, enabled=True)

    def read_feedback(self) -> ArmFeedback:
        with self._lock:
            if self._auto_timestamp:
                timestamp = max(
                    float(self._clock()),
                    nextafter(self._feedback.received_at_s, inf),
                )
                self._feedback = replace(
                    self._feedback,
                    received_at_s=timestamp,
                )
            return self._feedback

    def command_joints(self, joints_deg: Joint6) -> None:
        values = tuple(float(value) for value in joints_deg)
        if len(values) != 6:
            raise ValueError("joint command must contain six values")
        with self._lock:
            if self._emergency_latched.is_set():
                raise RuntimeError("simulated driver emergency stop is latched")
            self.joint_commands.append(values)  # type: ignore[arg-type]
            self._feedback = replace(self._feedback, joints_deg=values)

    def command_gripper(self, position: float) -> None:
        with self._lock:
            if self._emergency_latched.is_set():
                raise RuntimeError("simulated driver emergency stop is latched")
            self.gripper_commands.append(float(position))
            self._feedback = replace(
                self._feedback, gripper_position=float(position)
            )

    def emergency_stop(self) -> None:
        with self._lock:
            self.emergency_stop_calls += 1
            self._emergency_latched.set()

    def recover_emergency(self) -> None:
        with self._lock:
            self.recover_emergency_calls += 1
            self._emergency_latched.clear()

    def hold(self, feedback: ArmFeedback) -> None:
        with self._lock:
            self.hold_calls += 1
            values = tuple(float(value) for value in feedback.joints_deg)
            if len(values) != 6:
                raise ValueError("joint command must contain six values")
            self.joint_commands.append(values)  # type: ignore[arg-type]
            self._feedback = replace(self._feedback, joints_deg=values)

    def shutdown(self) -> None:
        with self._lock:
            self.shutdown_calls += 1

    def set_feedback(self, feedback: ArmFeedback) -> None:
        with self._lock:
            self._feedback = feedback

    def set_fault(self, fault_code: int) -> None:
        with self._lock:
            self._feedback = replace(self._feedback, fault_code=fault_code)

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._feedback = replace(self._feedback, enabled=enabled)

    def set_feedback_stale(self, stale: bool) -> None:
        with self._lock:
            self._feedback = replace(self._feedback, stale=stale)
