# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe ownership and stepping of the shared dual-arm MuJoCo world."""

from __future__ import annotations

import importlib
import math
import threading
import time
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable, Protocol, TypeAlias, cast

import numpy as np

from .config import ArmSide, MujocoConfig
from .mujoco_actuators import configure_piper_actuators
from .mujoco_model import DualArmModel


JointVector: TypeAlias = tuple[float, float, float, float, float, float]
Vector3: TypeAlias = tuple[float, float, float]
Quaternion: TypeAlias = tuple[float, float, float, float]
_SIDES = (ArmSide.LEFT, ArmSide.RIGHT)
_VIEWER_SYNC_INTERVAL_S = 1.0 / 60.0


class _PassiveViewer(Protocol):
    def is_running(self) -> bool: ...

    def sync(self) -> None: ...

    def close(self) -> None: ...


ViewerFactory: TypeAlias = Callable[[object, object], _PassiveViewer]


@dataclass(frozen=True, slots=True)
class MujocoArmSample:
    joints_rad: JointVector
    gripper_opening_m: float
    tcp_position_m: Vector3
    tcp_quat_wxyz: Quaternion
    simulation_time_s: float
    received_at_s: float
    connected: bool
    enabled: bool
    emergency_latched: bool
    command_generation: int


@dataclass(frozen=True, slots=True)
class _ArmState:
    joints_rad: JointVector
    gripper_opening_m: float
    enabled: bool = False
    emergency_latched: bool = False
    command_generation: int = 0


class MujocoWorld:
    """Own one ``MjData`` and expose immutable per-arm command mailboxes."""

    def __init__(
        self,
        dual: DualArmModel,
        config: MujocoConfig,
        shutdown_event: threading.Event,
        *,
        clock: Callable[[], float] = time.monotonic,
        viewer_factory: ViewerFactory | None = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise RuntimeError(
                "MuJoCo support requires the optional 'mujoco' dependency"
            ) from exc

        self._mujoco = mujoco
        self._dual = dual
        self._config = config
        self._shutdown_event = shutdown_event
        self._clock = clock
        self._viewer_factory = viewer_factory
        self._lock = threading.RLock()
        self._step_lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._owner_ident: int | None = None
        self._failure: Exception | None = None
        self._viewer_closed = False

        # The Menagerie position actuators are too soft for the controller's
        # feedback-relative setpoint ramps: their intentionally small errors do
        # not produce enough force to track under gravity. Preserve force
        # limits, damping, and the stock relative joint gains while stiffening
        # only the position terms.
        configure_piper_actuators(dual)

        self._data = mujoco.MjData(dual.model)
        mujoco.mj_forward(dual.model, self._data)
        states: dict[ArmSide, _ArmState] = {}
        for side in _SIDES:
            binding = dual.bindings[side]
            states[side] = _ArmState(
                joints_rad=cast(
                    JointVector,
                    tuple(
                        float(self._data.qpos[address])
                        for address in binding.joint_qpos_adr
                    ),
                ),
                gripper_opening_m=2.0
                * float(self._data.qpos[binding.gripper_qpos_adr]),
            )
        self._states = states
        received_at_s = float(clock())
        self._samples = MappingProxyType(
            {
                side: self._build_sample(side, states[side], received_at_s)
                for side in _SIDES
            }
        )

    @property
    def running(self) -> bool:
        return self._running.is_set()

    @property
    def failure(self) -> Exception | None:
        return self._failure

    @property
    def viewer_closed(self) -> bool:
        return self._viewer_closed

    def start(self) -> None:
        """Start the sole live-data owner and wait for viewer initialization."""

        with self._lock:
            if self._started:
                raise RuntimeError("MuJoCo world has already been started")
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="mujoco-physics",
                # Native MuJoCo/Viewer calls cannot be interrupted safely. The
                # service still requests and joins normal shutdown, but a
                # timed-out native worker must not hold the failed process open.
                daemon=True,
            )
            thread = self._thread
        with self._step_lock:
            thread.start()
        if not self._ready.wait(10.0):
            failure = TimeoutError("MuJoCo world did not become ready within 10 seconds")
            self._record_failure(failure)
            self._stop.set()

    def read_sample(self, side: ArmSide) -> MujocoArmSample:
        """Return the latest immutable object without acquiring the state lock."""

        return self._samples[side]

    def set_enabled(self, side: ArmSide, enabled: bool) -> None:
        with self._lock:
            state = self._states[side]
            if not enabled:
                sample = self._samples[side]
                state = replace(
                    state,
                    joints_rad=sample.joints_rad,
                    gripper_opening_m=sample.gripper_opening_m,
                )
            self._states[side] = replace(state, enabled=bool(enabled))
            self._publish_metadata_locked(side)

    def set_target(
        self,
        side: ArmSide,
        joints_rad: tuple[float, ...],
        gripper_opening_m: float,
    ) -> None:
        joints = self._validated_target(side, joints_rad, gripper_opening_m)
        with self._lock:
            state = self._states[side]
            if state.emergency_latched:
                return
            self._states[side] = replace(
                state,
                joints_rad=joints,
                gripper_opening_m=float(gripper_opening_m),
                command_generation=state.command_generation + 1,
            )

    def set_joint_target(
        self,
        side: ArmSide,
        joints_rad: tuple[float, ...],
    ) -> None:
        """Atomically update one side's joints while preserving its gripper."""

        joints = self._validated_joints(side, joints_rad)
        with self._lock:
            state = self._states[side]
            self._raise_if_emergency_latched(side, state)
            self._states[side] = replace(
                state,
                joints_rad=joints,
                command_generation=state.command_generation + 1,
            )

    def set_gripper_target(
        self,
        side: ArmSide,
        gripper_opening_m: float,
    ) -> None:
        """Atomically update one side's gripper while preserving its joints."""

        opening = self._validated_gripper_target(gripper_opening_m)
        with self._lock:
            state = self._states[side]
            self._raise_if_emergency_latched(side, state)
            self._states[side] = replace(
                state,
                gripper_opening_m=opening,
                command_generation=state.command_generation + 1,
            )

    def emergency_stop(self, side: ArmSide) -> None:
        with self._lock:
            state = self._states[side]
            sample = self._samples[side]
            self._states[side] = replace(
                state,
                joints_rad=sample.joints_rad,
                gripper_opening_m=sample.gripper_opening_m,
                emergency_latched=True,
                command_generation=state.command_generation + 1,
            )
            self._publish_metadata_locked(side)

    def recover(self, side: ArmSide) -> None:
        with self._lock:
            self._states[side] = replace(
                self._states[side], emergency_latched=False
            )
            self._publish_metadata_locked(side)

    def hold(self, side: ArmSide) -> None:
        with self._lock:
            state = self._states[side]
            sample = self._samples[side]
            self._states[side] = replace(
                state,
                joints_rad=sample.joints_rad,
                gripper_opening_m=sample.gripper_opening_m,
                command_generation=state.command_generation + 1,
            )

    def request_shutdown(self) -> None:
        self._stop.set()

    def join(self, timeout_s: float) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=max(0.0, timeout_s))
        return not thread.is_alive()

    def step_once(self, now_s: float) -> None:
        """Advance deterministically before ``start``; intended for tests only."""

        with self._step_lock:
            with self._lock:
                if self._started:
                    raise RuntimeError("step_once is only permitted before start")
            self._step_locked(float(now_s))

    def _validated_target(
        self,
        side: ArmSide,
        joints_rad: tuple[float, ...],
        gripper_opening_m: float,
    ) -> JointVector:
        joints = self._validated_joints(side, joints_rad)
        self._validated_gripper_target(gripper_opening_m)
        return joints

    def _validated_joints(
        self,
        side: ArmSide,
        joints_rad: tuple[float, ...],
    ) -> JointVector:
        if len(joints_rad) != 6:
            raise ValueError("MuJoCo joint target must contain exactly six values")
        joints = tuple(float(value) for value in joints_rad)
        if not all(math.isfinite(value) for value in joints):
            raise ValueError("MuJoCo joint target values must be finite")
        for value, (minimum, maximum) in zip(
            joints, self._dual.bindings[side].joint_ranges_rad
        ):
            if value < minimum or value > maximum:
                raise ValueError("MuJoCo joint target is outside the model range")
        return cast(JointVector, joints)

    def _validated_gripper_target(self, gripper_opening_m: float) -> float:
        opening = float(gripper_opening_m)
        if (
            not math.isfinite(opening)
            or not 0.0 <= opening <= self._config.gripper_model_max_m
        ):
            raise ValueError("MuJoCo gripper target is outside the model range")
        return opening

    @staticmethod
    def _raise_if_emergency_latched(side: ArmSide, state: _ArmState) -> None:
        if state.emergency_latched:
            raise RuntimeError(f"{side.value} MuJoCo emergency stop is latched")

    def _publish_metadata_locked(self, side: ArmSide) -> None:
        state = self._states[side]
        samples = dict(self._samples)
        samples[side] = replace(
            samples[side],
            enabled=state.enabled,
            emergency_latched=state.emergency_latched,
            command_generation=state.command_generation,
        )
        self._samples = MappingProxyType(samples)

    def _step_locked(self, now_s: float) -> None:
        if self._started and threading.get_ident() != self._owner_ident:
            raise RuntimeError("only the physics thread may step a started world")
        with self._lock:
            states = dict(self._states)

        for side in _SIDES:
            state = states[side]
            binding = self._dual.bindings[side]
            if state.enabled:
                joint_targets = state.joints_rad
                gripper_target = state.gripper_opening_m / 2.0
            else:
                joint_targets = cast(
                    JointVector,
                    tuple(
                        float(self._data.qpos[address])
                        for address in binding.joint_qpos_adr
                    ),
                )
                gripper_target = float(self._data.qpos[binding.gripper_qpos_adr])
            for actuator_id, target in zip(binding.joint_actuator_ids, joint_targets):
                self._data.ctrl[actuator_id] = target
            self._data.ctrl[binding.gripper_actuator_id] = gripper_target

        self._mujoco.mj_step(self._dual.model, self._data)
        projected = False
        for side in _SIDES:
            binding = self._dual.bindings[side]
            for qpos_adr, dof_adr, (minimum, maximum) in zip(
                binding.joint_qpos_adr,
                binding.joint_dof_adr,
                binding.joint_safety_ranges_rad,
                strict=True,
            ):
                position = float(self._data.qpos[qpos_adr])
                bounded = min(max(position, minimum), maximum)
                if bounded == position:
                    continue
                self._data.qpos[qpos_adr] = bounded
                velocity = float(self._data.qvel[dof_adr])
                if (bounded == minimum and velocity < 0.0) or (
                    bounded == maximum and velocity > 0.0
                ):
                    self._data.qvel[dof_adr] = 0.0
                projected = True
            gripper_minimum = 0.0
            gripper_maximum = self._config.gripper_model_max_m / 2.0
            gripper_position = float(
                self._data.qpos[binding.gripper_qpos_adr]
            )
            bounded_gripper = min(
                max(gripper_position, gripper_minimum), gripper_maximum
            )
            if bounded_gripper != gripper_position:
                self._data.qpos[binding.gripper_qpos_adr] = bounded_gripper
                gripper_velocity = float(
                    self._data.qvel[binding.gripper_dof_adr]
                )
                if (
                    bounded_gripper == gripper_minimum
                    and gripper_velocity < 0.0
                ) or (
                    bounded_gripper == gripper_maximum
                    and gripper_velocity > 0.0
                ):
                    self._data.qvel[binding.gripper_dof_adr] = 0.0
                projected = True
        if projected:
            self._mujoco.mj_forward(self._dual.model, self._data)
        samples = {
            side: self._build_sample(side, states[side], now_s) for side in _SIDES
        }
        with self._lock:
            samples = {
                side: replace(
                    samples[side],
                    enabled=self._states[side].enabled,
                    emergency_latched=self._states[side].emergency_latched,
                    command_generation=self._states[side].command_generation,
                )
                for side in _SIDES
            }
            self._samples = MappingProxyType(samples)

    def _build_sample(
        self,
        side: ArmSide,
        state: _ArmState,
        received_at_s: float,
    ) -> MujocoArmSample:
        binding = self._dual.bindings[side]
        fingers = self._data.xpos[list(binding.finger_body_ids)]
        tcp = 0.5 * (fingers[0] + fingers[1])
        quaternion = np.empty(4, dtype=float)
        self._mujoco.mju_mat2Quat(quaternion, self._data.xmat[binding.wrist_body_id])
        return MujocoArmSample(
            joints_rad=cast(
                JointVector,
                tuple(float(self._data.qpos[address]) for address in binding.joint_qpos_adr),
            ),
            gripper_opening_m=2.0
            * float(self._data.qpos[binding.gripper_qpos_adr]),
            tcp_position_m=cast(Vector3, tuple(float(value) for value in tcp)),
            tcp_quat_wxyz=cast(
                Quaternion, tuple(float(value) for value in quaternion)
            ),
            simulation_time_s=float(self._data.time),
            received_at_s=float(received_at_s),
            connected=True,
            enabled=state.enabled,
            emergency_latched=state.emergency_latched,
            command_generation=state.command_generation,
        )

    def _create_viewer(self) -> _PassiveViewer:
        if self._viewer_factory is not None:
            return self._viewer_factory(self._dual.model, self._data)
        viewer_module = importlib.import_module("mujoco.viewer")
        return viewer_module.launch_passive(self._dual.model, self._data)

    def _record_failure(self, failure: Exception) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = failure
        self._shutdown_event.set()

    def _run(self) -> None:
        viewer: _PassiveViewer | None = None
        self._owner_ident = threading.get_ident()
        try:
            if self._config.viewer:
                viewer = self._create_viewer()
            self._running.set()
            self._ready.set()
            deadline = self._clock()
            next_viewer_sync = deadline
            while not self._stop.is_set():
                if viewer is not None and not viewer.is_running():
                    self._viewer_closed = True
                    self._shutdown_event.set()
                if viewer is not None and self._shutdown_event.is_set():
                    closed_viewer, viewer = viewer, None
                    closed_viewer.close()
                with self._step_lock:
                    self._step_locked(self._clock())
                deadline += self._config.timestep_s
                now = self._clock()
                physics_caught_up = not self._config.realtime or deadline >= now
                if (
                    viewer is not None
                    and now >= next_viewer_sync
                    and physics_caught_up
                ):
                    viewer.sync()
                    next_viewer_sync = self._clock() + _VIEWER_SYNC_INTERVAL_S
                if self._config.realtime:
                    self._stop.wait(max(0.0, deadline - self._clock()))
        except Exception as exc:
            self._record_failure(exc)
        finally:
            if viewer is not None:
                try:
                    viewer.close()
                except Exception as exc:
                    self._record_failure(exc)
            self._running.clear()
            self._ready.set()
            self._owner_ident = None
