# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Narrow safe driver boundary used by the controller."""

from __future__ import annotations

from typing import Protocol

from piper_control.types import ArmFeedback, Joint6


class ArmDriver(Protocol):
    """Arm operations intentionally excluding power-off and reset APIs."""

    def connect(self) -> None: ...

    def enable(self) -> None: ...

    def read_feedback(self) -> ArmFeedback: ...

    def command_joints(self, joints_deg: Joint6) -> None: ...

    def command_gripper(self, position: float) -> None: ...

    def emergency_stop(self) -> None: ...

    def recover_emergency(self) -> None: ...

    def hold(self, feedback: ArmFeedback) -> None: ...

    def shutdown(self) -> None: ...
