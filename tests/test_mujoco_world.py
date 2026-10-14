# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from piper_control.config import ArmSide, load_config
from piper_control.controller import ArmController
from piper_control.drivers.mujoco import MujocoArmDriver
from piper_control.mujoco_model import build_dual_arm_model
from piper_control.mujoco_world import MujocoWorld
from piper_control.types import ServiceState
from piper_control.vectors import ServoCommand


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "config" / "piper-control.mujoco.yaml"


@pytest.fixture
def world_inputs():
    service = load_config(CONFIG_PATH)
    assert service.mujoco is not None
    dual = build_dual_arm_model(service.mujoco, service.arms)
    return service.mujoco, dual


@pytest.fixture
def world(world_inputs):
    config, dual = world_inputs
    return MujocoWorld(dual, config, threading.Event(), clock=lambda: 123.5)


def _advance(world: MujocoWorld, count: int = 300) -> None:
    for index in range(count):
        world.step_once(10.0 + index * 0.002)


def test_deterministic_prestart_steps_publish_finite_immutable_samples(world):
    before_right = world.read_sample(ArmSide.RIGHT)
    world.set_enabled(ArmSide.LEFT, True)
    world.set_target(
        ArmSide.LEFT,
        (0.1, 1.0, -1.0, 0.1, 0.0, 0.0),
        0.035,
    )

    _advance(world, 200)

    after_left = world.read_sample(ArmSide.LEFT)
    after_right = world.read_sample(ArmSide.RIGHT)
    assert after_left.joints_rad[0] != pytest.approx(0.0)
    assert after_left.simulation_time_s > 0.0
    assert after_left.received_at_s == pytest.approx(10.0 + 199 * 0.002)
    assert after_right.command_generation == before_right.command_generation
    assert len(after_left.joints_rad) == 6
    assert len(after_left.tcp_position_m) == 3
    assert len(after_left.tcp_quat_wxyz) == 4
    assert math.sqrt(
        sum(value * value for value in after_left.tcp_quat_wxyz)
    ) == pytest.approx(1.0)
    assert all(
        math.isfinite(value)
        for value in (
            *after_left.joints_rad,
            after_left.gripper_opening_m,
            *after_left.tcp_position_m,
            *after_left.tcp_quat_wxyz,
            after_left.simulation_time_s,
            after_left.received_at_s,
        )
    )
    assert after_left.gripper_opening_m == pytest.approx(
        2.0
        * world._data.qpos[
            world._dual.bindings[ArmSide.LEFT].gripper_qpos_adr
        ]
    )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        after_left.enabled = False


def test_incremental_gripper_setpoints_reach_distinct_api_targets(world):
    for side in ArmSide:
        world.set_enabled(side, True)

    # A 20 mm/s API command advances the model target by 0.14 mm per 10 ms
    # controller cycle. Reproduce that closed-loop command shape, including the
    # five 2 ms physics steps that happen between controller updates.
    for index in range(800):
        left = world.read_sample(ArmSide.LEFT).gripper_opening_m
        right = world.read_sample(ArmSide.RIGHT).gripper_opening_m
        world.set_gripper_target(ArmSide.LEFT, min(0.07, left + 0.00014))
        world.set_gripper_target(ArmSide.RIGHT, min(0.035, right + 0.00014))
        for step in range(5):
            world.step_once(index * 0.01 + step * 0.002)

    assert world.read_sample(ArmSide.LEFT).gripper_opening_m == pytest.approx(
        0.07, abs=0.003
    )
    assert world.read_sample(ArmSide.RIGHT).gripper_opening_m == pytest.approx(
        0.035, abs=0.002
    )


def test_production_speed_joint_ramps_track_independent_targets_and_hold(
    world_inputs,
):
    config, dual = world_inputs
    actuator_force_ranges = dual.model.actuator_forcerange.copy()
    actuator_velocity_bias = dual.model.actuator_biasprm[:, 2].copy()
    dof_damping = dual.model.dof_damping.copy()
    joint_ranges = dual.model.jnt_range.copy()
    world = MujocoWorld(dual, config, threading.Event(), clock=lambda: 0.0)
    goals_deg = {
        ArmSide.LEFT: (0.0, 45.0, -45.0, 5.0, 5.0, 5.0),
        ArmSide.RIGHT: (0.0, 55.0, -55.0, -5.0, -5.0, -5.0),
    }
    maximum_cycle_step = math.radians(10.0 / 100.0)
    for side in ArmSide:
        world.set_enabled(side, True)

    for cycle in range(1500):
        for side in ArmSide:
            actual = world.read_sample(side).joints_rad
            goal = tuple(math.radians(value) for value in goals_deg[side])
            command = tuple(
                position
                + max(
                    -maximum_cycle_step,
                    min(maximum_cycle_step, target - position),
                )
                for position, target in zip(actual, goal)
            )
            world.set_joint_target(side, command)
        for step in range(5):
            world.step_once(cycle * 0.01 + step * 0.002)

    reached = {
        side: tuple(
            math.degrees(value) for value in world.read_sample(side).joints_rad
        )
        for side in ArmSide
    }
    for side in ArmSide:
        assert reached[side] == pytest.approx(goals_deg[side], abs=2.0)
    assert reached[ArmSide.RIGHT][1] - reached[ArmSide.LEFT][1] == pytest.approx(
        10.0, abs=2.0
    )

    for side in ArmSide:
        world.hold(side)
    for step in range(1000):
        world.step_once(15.0 + step * 0.002)
    for side in ArmSide:
        held = tuple(
            math.degrees(value) for value in world.read_sample(side).joints_rad
        )
        assert held == pytest.approx(reached[side], abs=0.2)

    np.testing.assert_array_equal(dual.model.actuator_forcerange, actuator_force_ranges)
    np.testing.assert_array_equal(
        dual.model.actuator_biasprm[:, 2], actuator_velocity_bias
    )
    np.testing.assert_array_equal(dual.model.dof_damping, dof_damping)
    np.testing.assert_array_equal(dual.model.jnt_range, joint_ranges)


def test_production_speed_shutdown_homes_without_crossing_joint_limits(world_inputs):
    config, dual = world_inputs
    clock = _AdvancingClock()
    world = MujocoWorld(dual, config, threading.Event(), clock=clock)
    controllers = {
        side: ArmController(
            side,
            arm,
            MujocoArmDriver(
                side,
                arm,
                world,
                gripper_model_max_m=config.gripper_model_max_m,
            ),
            clock=clock,
            sleep=lambda _seconds: None,
        )
        for side, arm in load_config(CONFIG_PATH).items()
    }

    def cycle() -> None:
        for controller in controllers.values():
            controller.tick(dt=0.01)
        for _ in range(5):
            clock.advance(0.002)
            world.step_once(clock())

    for _ in range(10):
        cycle()
        if all(
            controller.snapshot().state is ServiceState.RUNNING
            for controller in controllers.values()
        ):
            break
    assert all(
        controller.snapshot().state is ServiceState.RUNNING
        for controller in controllers.values()
    )

    goals = {
        ArmSide.LEFT: (0.0, 45.0, -45.0, 5.0, 5.0, 5.0),
        ArmSide.RIGHT: (0.0, 55.0, -55.0, -5.0, -5.0, -5.0),
    }
    service = load_config(CONFIG_PATH)
    for side, controller in controllers.items():
        controller.submit_servo(
            ServoCommand(goals[side], service.arms[side].default_joints_speed_deg_s)
        )
    for _ in range(800):
        cycle()
    for side, controller in controllers.items():
        assert controller.snapshot().actual_q == pytest.approx(goals[side], abs=2.0)
        controller.request_shutdown_async()

    for _ in range(1000):
        cycle()
        assert all(
            controller.snapshot().state is not ServiceState.EMERGENCY_STOP
            for controller in controllers.values()
        ), {
            side.value: controller.snapshot().fault_reason
            for side, controller in controllers.items()
        }
        if all(
            controller._homing_stable_cycles >= 3
            for controller in controllers.values()
        ):
            break

    assert all(
        controller._homing_stable_cycles >= 3
        for controller in controllers.values()
    )
    for controller in controllers.values():
        snapshot = controller.snapshot()
        assert snapshot.state is ServiceState.SHUTTING_DOWN
        assert snapshot.actual_q == pytest.approx((0.0,) * 6, abs=0.01)
        assert snapshot.fault_reason is None


@pytest.mark.parametrize("side", list(ArmSide))
@pytest.mark.parametrize("boundary", ["lower", "upper"])
def test_physics_step_projects_gripper_soft_limit_penetration(
    world_inputs,
    monkeypatch,
    side: ArmSide,
    boundary: str,
):
    config, dual = world_inputs
    world = MujocoWorld(dual, config, threading.Event(), clock=lambda: 0.0)
    binding = dual.bindings[side]
    joint = dual.model.joint(f"{side.value}_joint7")
    dof_adr = int(joint.dofadr[0])
    finger_maximum = config.gripper_model_max_m / 2.0
    if boundary == "lower":
        penetrated = -0.001
        outward_velocity = -1.0
        expected = 0.0
    else:
        penetrated = finger_maximum + 0.001
        outward_velocity = 1.0
        expected = finger_maximum

    def penetrate_soft_limit(_model, data):
        data.qpos[binding.gripper_qpos_adr] = penetrated
        data.qvel[dof_adr] = outward_velocity

    monkeypatch.setattr(mujoco, "mj_step", penetrate_soft_limit)

    world.step_once(0.0)

    assert world._data.qpos[binding.gripper_qpos_adr] == pytest.approx(expected)
    assert world._data.qvel[dof_adr] == pytest.approx(0.0)
    assert world.read_sample(side).gripper_opening_m == pytest.approx(2.0 * expected)


def test_latest_concurrent_target_wins_independently_per_side(world):
    for side in ArmSide:
        world.set_enabled(side, True)
    barrier = threading.Barrier(3)

    def write(side: ArmSide, sign: float) -> None:
        barrier.wait(timeout=1.0)
        for index in range(1, 51):
            world.set_target(side, (sign * index / 500.0, 0.5, -0.5, 0, 0, 0), 0.0)

    left = threading.Thread(target=write, args=(ArmSide.LEFT, 1.0))
    right = threading.Thread(target=write, args=(ArmSide.RIGHT, -1.0))
    left.start()
    right.start()
    barrier.wait(timeout=1.0)
    left.join(timeout=1.0)
    right.join(timeout=1.0)
    assert not left.is_alive()
    assert not right.is_alive()
    _advance(world)

    left_sample = world.read_sample(ArmSide.LEFT)
    right_sample = world.read_sample(ArmSide.RIGHT)
    assert left_sample.joints_rad[0] > 0.0
    assert right_sample.joints_rad[0] < 0.0
    assert left_sample.command_generation == 50
    assert right_sample.command_generation == 50


def test_read_sample_does_not_wait_for_world_state_lock(world):
    completed = threading.Event()

    def read() -> None:
        world.read_sample(ArmSide.LEFT)
        completed.set()

    with world._lock:
        reader = threading.Thread(target=read)
        reader.start()
        assert completed.wait(0.2)
    reader.join(timeout=1.0)
    assert not reader.is_alive()


def test_emergency_and_recovery_are_isolated_per_side(world):
    for side in ArmSide:
        world.set_enabled(side, True)
    world.set_target(ArmSide.LEFT, (0.08, 0.5, -0.5, 0, 0, 0), 0.02)
    world.set_target(ArmSide.RIGHT, (-0.08, 0.5, -0.5, 0, 0, 0), 0.02)
    _advance(world, 150)
    left_generation = world.read_sample(ArmSide.LEFT).command_generation
    right_generation = world.read_sample(ArmSide.RIGHT).command_generation

    world.emergency_stop(ArmSide.LEFT)
    world.set_target(ArmSide.LEFT, (-0.4, 0.5, -0.5, 0, 0, 0), 0.06)
    world.set_target(ArmSide.RIGHT, (-0.2, 0.5, -0.5, 0, 0, 0), 0.04)
    world.step_once(20.0)

    left = world.read_sample(ArmSide.LEFT)
    right = world.read_sample(ArmSide.RIGHT)
    assert left.emergency_latched is True
    assert right.emergency_latched is False
    assert left.command_generation == left_generation + 1
    assert right.command_generation == right_generation + 1

    world.recover(ArmSide.LEFT)
    world.set_target(ArmSide.LEFT, (-0.2, 0.5, -0.5, 0, 0, 0), 0.04)
    world.step_once(20.1)
    assert world.read_sample(ArmSide.LEFT).emergency_latched is False
    assert world.read_sample(ArmSide.LEFT).command_generation == left_generation + 2
    assert world.read_sample(ArmSide.RIGHT).command_generation == right_generation + 1


def test_hold_freezes_only_requested_side_at_published_position(world):
    for side in ArmSide:
        world.set_enabled(side, True)
        world.set_target(side, (0.15, 0.5, -0.5, 0, 0, 0), 0.03)
    _advance(world, 80)
    held = world.read_sample(ArmSide.LEFT)
    right_generation = world.read_sample(ArmSide.RIGHT).command_generation

    world.hold(ArmSide.LEFT)
    world.set_target(ArmSide.RIGHT, (-0.2, 0.5, -0.5, 0, 0, 0), 0.05)
    world.step_once(30.0)

    left_binding = world._dual.bindings[ArmSide.LEFT]
    assert tuple(
        world._data.ctrl[index] for index in left_binding.joint_actuator_ids
    ) == pytest.approx(held.joints_rad)
    assert world._data.ctrl[left_binding.gripper_actuator_id] == pytest.approx(
        held.gripper_opening_m / 2.0
    )
    assert world.read_sample(ArmSide.RIGHT).command_generation == right_generation + 1


def test_partial_targets_reject_emergency_and_retain_frozen_counterpart(world):
    world.set_enabled(ArmSide.LEFT, True)
    world.set_target(ArmSide.LEFT, (0.2, 0.5, -0.5, 0, 0, 0), 0.06)
    frozen = world.read_sample(ArmSide.LEFT)
    right_before = world.read_sample(ArmSide.RIGHT).command_generation

    world.emergency_stop(ArmSide.LEFT)
    with pytest.raises(RuntimeError, match="emergency stop"):
        world.set_joint_target(ArmSide.LEFT, (-0.2, 0.5, -0.5, 0, 0, 0))
    world.recover(ArmSide.LEFT)
    world.set_gripper_target(ArmSide.LEFT, 0.04)

    state = world._states[ArmSide.LEFT]
    assert state.joints_rad == frozen.joints_rad
    assert state.gripper_opening_m == 0.04
    assert state.command_generation == 3
    assert world.read_sample(ArmSide.RIGHT).command_generation == right_before


def test_partial_joint_target_retains_gripper_frozen_by_emergency(world):
    world.set_enabled(ArmSide.LEFT, True)
    world.set_target(ArmSide.LEFT, (0.2, 0.5, -0.5, 0, 0, 0), 0.06)
    frozen = world.read_sample(ArmSide.LEFT)

    world.emergency_stop(ArmSide.LEFT)
    with pytest.raises(RuntimeError, match="emergency stop"):
        world.set_gripper_target(ArmSide.LEFT, 0.05)
    world.recover(ArmSide.LEFT)
    world.set_joint_target(ArmSide.LEFT, (-0.2, 0.5, -0.5, 0, 0, 0))

    state = world._states[ArmSide.LEFT]
    assert state.joints_rad == (-0.2, 0.5, -0.5, 0, 0, 0)
    assert state.gripper_opening_m == frozen.gripper_opening_m
    assert state.command_generation == 3


@pytest.mark.parametrize("freeze", ["emergency", "hold"])
def test_inflight_publication_cannot_undo_atomic_freeze_before_partial_command(
    world, monkeypatch, freeze: str
):
    world.set_enabled(ArmSide.LEFT, True)
    world.set_target(ArmSide.LEFT, (0.2, 0.5, -0.5, 0, 0, 0), 0.06)
    frozen = world.read_sample(ArmSide.LEFT)
    right_before = world.read_sample(ArmSide.RIGHT)
    entered = threading.Event()
    release = threading.Event()
    real_step = mujoco.mj_step

    def blocked_step(model, data):
        entered.set()
        assert release.wait(1.0)
        for _ in range(20):
            real_step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", blocked_step)
    stepper = threading.Thread(target=world.step_once, args=(20.0,))
    stepper.start()
    assert entered.wait(1.0)

    try:
        if freeze == "emergency":
            world.emergency_stop(ArmSide.LEFT)
            world.recover(ArmSide.LEFT)
        else:
            world.hold(ArmSide.LEFT)
    finally:
        release.set()
        stepper.join(timeout=1.0)
    assert not stepper.is_alive()
    raced_sample = world.read_sample(ArmSide.LEFT)
    assert raced_sample.joints_rad != pytest.approx(frozen.joints_rad)

    world.set_gripper_target(ArmSide.LEFT, 0.04)

    state = world._states[ArmSide.LEFT]
    assert state.joints_rad == frozen.joints_rad
    assert state.gripper_opening_m == 0.04
    assert world.read_sample(ArmSide.RIGHT).command_generation == (
        right_before.command_generation
    )


def test_step_failure_is_stored_and_requests_process_shutdown(world_inputs, monkeypatch):
    config, dual = world_inputs
    shutdown = threading.Event()
    world = MujocoWorld(
        dual,
        config.model_copy(update={"realtime": False, "viewer": False}),
        shutdown,
    )
    failure = RuntimeError("step exploded")

    def fail_step(model, data):
        raise failure

    monkeypatch.setattr(mujoco, "mj_step", fail_step)
    world.start()

    assert world.join(1.0)
    assert world.failure is failure
    assert shutdown.is_set()
    assert world.running is False


def test_blocked_physics_worker_allows_bounded_nonzero_process_exit() -> None:
    child = dedent(
        f"""
        import sys
        import threading
        from pathlib import Path

        from piper_control.config import load_config
        from piper_control.mujoco_model import build_dual_arm_model
        from piper_control.mujoco_world import MujocoWorld

        service = load_config(Path({str(CONFIG_PATH)!r}))
        assert service.mujoco is not None
        config = service.mujoco.model_copy(
            update={{"viewer": False, "realtime": False}}
        )
        dual = build_dual_arm_model(config, service.arms)
        world = MujocoWorld(dual, config, threading.Event())
        worker_blocked = threading.Event()

        def block_worker(_now_s):
            worker_blocked.set()
            threading.Event().wait()

        world._step_locked = block_worker
        world.start()
        assert worker_blocked.wait(2.0)
        world.request_shutdown()
        if world.join(0.1):
            raise SystemExit(2)
        raise SystemExit(23)
        """
    )
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
    }
    started_at = time.monotonic()

    result = subprocess.run(
        [sys.executable, "-c", child],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )

    assert result.returncode == 23, result.stderr
    assert time.monotonic() - started_at < 5.0


class _ClosedViewer:
    def __init__(self) -> None:
        self.closed = False
        self.sync_calls = 0

    def is_running(self) -> bool:
        return False

    def sync(self) -> None:
        self.sync_calls += 1

    def close(self) -> None:
        self.closed = True


class _OpenViewer(_ClosedViewer):
    def is_running(self) -> bool:
        return not self.closed


def test_closing_viewer_requests_process_shutdown_but_keeps_physics_running(
    world_inputs,
):
    config, dual = world_inputs
    shutdown = threading.Event()
    viewer = _ClosedViewer()
    world = MujocoWorld(dual, config, shutdown, viewer_factory=lambda model, data: viewer)

    world.start()
    before = world.read_sample(ArmSide.LEFT).simulation_time_s

    try:
        deadline = time.monotonic() + 0.2
        while (
            world.read_sample(ArmSide.LEFT).simulation_time_s < before + 0.02
            and time.monotonic() < deadline
        ):
            time.sleep(0.002)
        assert world.viewer_closed is True
        assert shutdown.is_set()
        assert viewer.closed is True
        assert viewer.sync_calls == 0
        assert world.running is True
        assert world.read_sample(ArmSide.LEFT).simulation_time_s >= before + 0.02
        assert not world.join(0.01)
        assert world.failure is None
    finally:
        world.request_shutdown()
        assert world.join(1.0)


def test_process_shutdown_closes_viewer_but_keeps_physics_running(world_inputs):
    config, dual = world_inputs
    shutdown = threading.Event()
    viewer = _OpenViewer()
    world = MujocoWorld(dual, config, shutdown, viewer_factory=lambda model, data: viewer)
    world.start()
    before = world.read_sample(ArmSide.LEFT).simulation_time_s

    try:
        shutdown.set()
        deadline = time.monotonic() + 0.2
        while (
            (
                not viewer.closed
                or world.read_sample(ArmSide.LEFT).simulation_time_s < before + 0.02
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.002)
        assert viewer.closed is True
        assert world.viewer_closed is False
        assert world.running is True
        assert world.read_sample(ArmSide.LEFT).simulation_time_s >= before + 0.02
        assert world.failure is None
    finally:
        world.request_shutdown()
        assert world.join(1.0)


class _AdvancingClock:
    def __init__(self) -> None:
        self.value = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.value += seconds


class _VsyncViewer:
    def __init__(self, clock, step_count, stop) -> None:
        self._clock = clock
        self._step_count = step_count
        self._stop = stop
        self.sync_calls = 0
        self.steps_at_sync: list[int] = []
        self.closed = False

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        self.sync_calls += 1
        self.steps_at_sync.append(self._step_count())
        if self.sync_calls == 1:
            self._clock.advance(0.2)
        else:
            self._stop()

    def close(self) -> None:
        self.closed = True


def test_long_vsync_delay_is_caught_up_before_next_render(world_inputs, monkeypatch):
    config, dual = world_inputs
    clock = _AdvancingClock()
    shutdown = threading.Event()
    steps = 0
    original_step = mujoco.mj_step
    world: MujocoWorld

    viewer = _VsyncViewer(
        clock,
        lambda: steps,
        lambda: world.request_shutdown(),
    )

    def count_step(model, data):
        nonlocal steps
        original_step(model, data)
        steps += 1
        clock.advance(0.0005)

    monkeypatch.setattr(mujoco, "mj_step", count_step)
    world = MujocoWorld(
        dual,
        config.model_copy(update={"realtime": True}),
        shutdown,
        clock=clock,
        viewer_factory=lambda model, data: viewer,
    )

    world.start()

    assert world.join(1.0)
    assert viewer.sync_calls == 2
    assert viewer.steps_at_sync[0] == 1
    assert 100 <= viewer.steps_at_sync[1] <= 160
    assert viewer.closed is True
    assert world.failure is None


def test_headless_world_never_calls_viewer_factory(world_inputs):
    config, dual = world_inputs
    shutdown = threading.Event()
    factory_called = threading.Event()
    headless = config.model_copy(update={"viewer": False, "realtime": True})
    world = MujocoWorld(
        dual,
        headless,
        shutdown,
        viewer_factory=lambda model, data: factory_called.set(),
    )

    world.start()
    time.sleep(0.02)
    world.request_shutdown()

    assert world.join(1.0)
    assert not factory_called.is_set()
    assert world.failure is None


def test_process_shutdown_notification_does_not_stop_physics_before_world_request(
    world_inputs,
):
    config, dual = world_inputs
    shutdown = threading.Event()
    world = MujocoWorld(
        dual,
        config.model_copy(update={"viewer": False, "realtime": True}),
        shutdown,
    )
    world.start()
    before = world.read_sample(ArmSide.LEFT).simulation_time_s

    try:
        shutdown.set()
        deadline = time.monotonic() + 0.2
        while (
            world.running
            and world.read_sample(ArmSide.LEFT).simulation_time_s < before + 0.02
            and time.monotonic() < deadline
        ):
            time.sleep(0.002)
        assert world.running is True
        assert world.read_sample(ArmSide.LEFT).simulation_time_s >= before + 0.02
    finally:
        world.request_shutdown()
        assert world.join(1.0)


def test_viewer_creation_failure_is_published_before_start_returns(world_inputs):
    config, dual = world_inputs
    shutdown = threading.Event()
    failure = RuntimeError("no display")

    def fail_viewer(model, data):
        raise failure

    world = MujocoWorld(dual, config, shutdown, viewer_factory=fail_viewer)
    world.start()

    assert world.failure is failure
    assert world.running is False
    assert shutdown.is_set()
    assert world.join(1.0)


def test_step_once_is_rejected_after_live_thread_has_started(world_inputs):
    config, dual = world_inputs
    world = MujocoWorld(
        dual,
        config.model_copy(update={"viewer": False}),
        threading.Event(),
    )
    world.start()
    with pytest.raises(RuntimeError, match="before start"):
        world.step_once(1.0)
    world.request_shutdown()
    assert world.join(1.0)


def test_concurrent_prestart_steps_never_overlap_mj_step(world, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    guard = threading.Lock()
    active = 0
    maximum_active = 0
    calls = 0

    def blocking_step(model, data):
        nonlocal active, maximum_active, calls
        with guard:
            calls += 1
            call = calls
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if call == 1:
                entered.set()
                assert release.wait(1.0)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(mujoco, "mj_step", blocking_step)
    errors: list[BaseException] = []

    def step(now_s: float) -> None:
        try:
            world.step_once(now_s)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=step, args=(1.0,))
    second = threading.Thread(target=step, args=(2.0,))
    first.start()
    assert entered.wait(1.0)
    second.start()
    time.sleep(0.02)
    with guard:
        overlap_before_release = maximum_active
    release.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert calls == 2
    assert overlap_before_release == 1
    assert maximum_active == 1


def test_start_waits_for_prestart_step_and_rejects_queued_step_once(
    world_inputs, monkeypatch
):
    config, dual = world_inputs
    world = MujocoWorld(
        dual,
        config.model_copy(update={"viewer": False, "realtime": True}),
        threading.Event(),
    )
    entered = threading.Event()
    release = threading.Event()
    guard = threading.Lock()
    active = 0
    maximum_active = 0
    calls = 0

    def blocking_step(model, data):
        nonlocal active, maximum_active, calls
        with guard:
            calls += 1
            call = calls
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if call == 1:
                entered.set()
                assert release.wait(1.0)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(mujoco, "mj_step", blocking_step)
    prestart_errors: list[BaseException] = []
    queued_errors: list[BaseException] = []
    start_returned = threading.Event()
    queued_started = threading.Event()

    def prestart_step() -> None:
        try:
            world.step_once(1.0)
        except BaseException as exc:
            prestart_errors.append(exc)

    def start() -> None:
        world.start()
        start_returned.set()

    def queued_step() -> None:
        queued_started.set()
        try:
            world.step_once(2.0)
        except BaseException as exc:
            queued_errors.append(exc)

    first = threading.Thread(target=prestart_step)
    starter = threading.Thread(target=start)
    queued = threading.Thread(target=queued_step)
    first.start()
    assert entered.wait(1.0)
    starter.start()

    deadline = time.monotonic() + 1.0
    while True:
        with world._lock:
            start_linearized = world._started
        if start_linearized:
            break
        if time.monotonic() >= deadline:
            pytest.fail("start() did not linearize within one second")
        time.sleep(0.01)

    assert not start_returned.is_set()
    queued.start()
    assert queued_started.wait(1.0)
    with guard:
        overlap_before_release = maximum_active
    start_completed_before_release = start_returned.is_set()
    release.set()
    first.join(timeout=1.0)
    starter.join(timeout=1.0)
    queued.join(timeout=1.0)
    world.request_shutdown()
    assert world.join(1.0)

    assert not first.is_alive()
    assert not starter.is_alive()
    assert not queued.is_alive()
    assert prestart_errors == []
    assert len(queued_errors) == 1
    assert isinstance(queued_errors[0], RuntimeError)
    assert "before start" in str(queued_errors[0])
    assert start_completed_before_release is False
    assert overlap_before_release == 1
    assert maximum_active == 1


def test_inflight_step_publishes_new_physics_with_latest_mailbox_metadata(
    world, monkeypatch
):
    world.set_enabled(ArmSide.LEFT, True)
    world.set_target(ArmSide.LEFT, (0.1, 0.5, -0.5, 0, 0, 0), 0.02)
    world.emergency_stop(ArmSide.LEFT)
    world.set_enabled(ArmSide.RIGHT, True)
    world.set_target(ArmSide.RIGHT, (-0.1, 0.5, -0.5, 0, 0, 0), 0.02)
    before = world.read_sample(ArmSide.RIGHT)
    left_generation = world.read_sample(ArmSide.LEFT).command_generation
    original_step = mujoco.mj_step
    stepped = threading.Event()
    release = threading.Event()

    def blocking_step(model, data):
        original_step(model, data)
        stepped.set()
        assert release.wait(1.0)

    monkeypatch.setattr(mujoco, "mj_step", blocking_step)
    errors: list[BaseException] = []

    def step() -> None:
        try:
            world.step_once(77.0)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=step)
    worker.start()
    assert stepped.wait(1.0)

    world.recover(ArmSide.LEFT)
    world.set_target(ArmSide.LEFT, (0.2, 0.5, -0.5, 0, 0, 0), 0.03)
    world.set_enabled(ArmSide.LEFT, False)
    world.emergency_stop(ArmSide.RIGHT)
    world.set_enabled(ArmSide.RIGHT, False)
    release.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert errors == []
    left = world.read_sample(ArmSide.LEFT)
    right = world.read_sample(ArmSide.RIGHT)
    assert left.simulation_time_s > 0.0
    assert left.received_at_s == pytest.approx(77.0)
    assert right.simulation_time_s > before.simulation_time_s
    assert left.enabled is False
    assert left.emergency_latched is False
    assert left.command_generation == left_generation + 1
    assert right.enabled is False
    assert right.emergency_latched is True
    assert right.command_generation == 2
