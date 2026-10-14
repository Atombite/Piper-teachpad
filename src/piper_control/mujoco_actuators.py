# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Shared actuator tuning for the Piper MuJoCo model."""

from __future__ import annotations

from .config import ArmSide
from .mujoco_model import DualArmModel


_SIDES = (ArmSide.LEFT, ArmSide.RIGHT)
_JOINT_POSITION_KP_SCALE = 10.0
_GRIPPER_POSITION_KP = 400.0


def configure_piper_actuators(dual: DualArmModel) -> None:
    """Stiffen position tracking while preserving model force limits."""

    for side in _SIDES:
        for actuator_id in dual.bindings[side].joint_actuator_ids:
            dual.model.actuator_gainprm[actuator_id, 0] *= _JOINT_POSITION_KP_SCALE
            dual.model.actuator_biasprm[actuator_id, 1] *= _JOINT_POSITION_KP_SCALE
        actuator_id = dual.bindings[side].gripper_actuator_id
        dual.model.actuator_gainprm[actuator_id, 0] = _GRIPPER_POSITION_KP
        dual.model.actuator_biasprm[actuator_id, 1] = -_GRIPPER_POSITION_KP
