# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from piper_control.config import ArmSide, load_config
from piper_control.mujoco_model import build_dual_arm_model


CONFIG_PATH = Path(__file__).parents[1] / "config" / "piper-control.mujoco.yaml"


@pytest.fixture
def dual():
    service = load_config(CONFIG_PATH)
    assert service.mujoco is not None
    return build_dual_arm_model(service.mujoco, service.arms)


def test_configure_piper_actuators_preserves_forces_and_sets_expected_gains(
    dual,
) -> None:
    from piper_control.mujoco_actuators import configure_piper_actuators

    before_force = dual.model.actuator_forcerange.copy()
    joint_id = dual.bindings[ArmSide.LEFT].joint_actuator_ids[0]
    before_joint_gain = float(dual.model.actuator_gainprm[joint_id, 0])

    configure_piper_actuators(dual)

    assert np.array_equal(dual.model.actuator_forcerange, before_force)
    assert dual.model.actuator_gainprm[joint_id, 0] == pytest.approx(
        before_joint_gain * 10.0
    )
    for side in (ArmSide.LEFT, ArmSide.RIGHT):
        actuator_id = dual.bindings[side].gripper_actuator_id
        assert dual.model.actuator_gainprm[actuator_id, 0] == 400.0
        assert dual.model.actuator_biasprm[actuator_id, 1] == -400.0
