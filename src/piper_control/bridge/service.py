# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Single-writer coordinator for one arm-local Bridge process."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
import time
from typing import Any

from piper_control.bridge.arbitration import EnqueueResult, PriorityArbiter, QueuedCommand
from piper_control.bridge.config import BridgeConfig
from piper_control.bridge.motion import (
    GripperMotionPlan,
    JointMotionPlan,
    plan_gripper_motion,
    plan_joint_motion,
    step_gripper_motion,
    step_joint_motion,
)
from piper_control.bridge.protocol import (
    BRIDGE_COMMANDS,
    BridgeGripperRequest,
    BridgeServoRequest,
)
from piper_control.bridge.upstream import (
    PiperControlClient,
    UpstreamError,
    UpstreamSnapshot,
    probe_control_mode,
)
from piper_control.config import ArmSide
from piper_control.protocol import ProtocolError


Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
UpstreamFactory = Callable[[str], Any]
ModeProbe = Callable[[str, float], Awaitable[str]]
_JOINT_SETTLED_TOLERANCE_DEG = 0.01


class BridgeService:
    """Coordinate arbitration, bounded interpolation, and upstream recovery."""

    def __init__(
        self,
        config: BridgeConfig,
        *,
        side: ArmSide,
        upstream_factory: UpstreamFactory = PiperControlClient,
        mode_probe: ModeProbe = probe_control_mode,
        mode_probe_interval_s: float = 1.0,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.config = config
        self.side = side
        self._upstream_factory = upstream_factory
        self._mode_probe = mode_probe
        self._mode_probe_interval_s = mode_probe_interval_s
        self._clock = clock
        self._sleep = sleep
        self._period = 1.0 / float(config.control_hz)
        self._arbiter = PriorityArbiter(
            queue_capacity=config.queue_capacity,
            command_ttl_s=float(config.command_ttl_s),
            owner_timeout_s=float(config.owner_timeout_s),
        )
        self._snapshot: UpstreamSnapshot | None = None
        self._cached_state: dict[str, Any] | None = None
        self._cached_base_info: dict[str, Any] | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._wakeup = asyncio.Event()
        self._generation = 0

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="piper-control-bridge")

    async def stop(self) -> None:
        task, self._task = self._task, None
        self._stopping = True
        self._wakeup.set()
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._clear_control()

    async def submit(
        self,
        connection_id: str,
        request: BridgeServoRequest | BridgeGripperRequest,
    ) -> EnqueueResult:
        if self._snapshot is None:
            raise ProtocolError("UPSTREAM_UNAVAILABLE", "upstream is unavailable or unsafe")
        previous_owner = self._arbiter.owner
        result = self._arbiter.enqueue(connection_id, request, self._clock())
        if result.accepted:
            if self._arbiter.owner != previous_owner:
                self._generation += 1
            self._wakeup.set()
        return result

    async def release(self, connection_id: str, priority: int) -> None:
        owner_changed = self._arbiter.release(connection_id, priority, self._clock())
        if owner_changed:
            self._generation += 1
        self._wakeup.set()

    async def disconnect(self, connection_id: str) -> None:
        owner_changed = self._arbiter.disconnect(connection_id, self._clock())
        if owner_changed:
            self._generation += 1
        self._wakeup.set()

    async def state(self) -> dict[str, Any]:
        payload = dict(self._cached_state or {})
        payload["bridge"] = self._bridge_status()
        return payload

    async def base_info(self) -> dict[str, Any]:
        payload = dict(self._cached_base_info or {})
        payload["supported_commands"] = list(BRIDGE_COMMANDS)
        payload["bridge"] = self._bridge_status()
        return payload

    def _bridge_status(self) -> dict[str, Any]:
        owner = self._arbiter.owner
        return {
            "upstream_available": self._snapshot is not None,
            "cached_at": None if self._snapshot is None else self._snapshot.received_at,
            "owner": None
            if owner is None
            else {"connection_id": owner.connection_id, "priority": owner.priority},
        }

    def _clear_control(self) -> None:
        self._snapshot = None
        self._arbiter.clear()
        self._generation += 1
        self._wakeup.set()

    async def _run(self) -> None:
        reconnect_delay = float(self.config.reconnect_initial_s)
        while not self._stopping:
            client = self._upstream_factory(self.config.upstream_url)
            mode_mismatch = False
            try:
                await client.connect()
                base_info = await client.get_base_info()
                state = await client.get_state()
                self._cached_base_info = dict(base_info)
                self._accept_snapshot(state, base_info)
                reconnect_delay = float(self.config.reconnect_initial_s)
                await self._run_connected(client)
            except asyncio.CancelledError:
                raise
            except UpstreamError as exc:
                self._clear_control()
                mode_mismatch = exc.code == "IDENTITY_MISMATCH"
            except Exception:
                self._clear_control()
            finally:
                with suppress(Exception):
                    await client.close()
            if mode_mismatch and not self._stopping:
                await self._wait_for_remote_mode()
                reconnect_delay = float(self.config.reconnect_initial_s)
                continue
            if not self._stopping:
                await self._sleep(reconnect_delay)
                reconnect_delay = min(
                    float(self.config.reconnect_max_s), reconnect_delay * 2.0
                )

    async def _wait_for_remote_mode(self) -> None:
        while not self._stopping:
            try:
                mode = await self._mode_probe(
                    self.config.upstream_url,
                    1.0,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                mode = None
            if mode == "remote":
                return
            await self._sleep(self._mode_probe_interval_s)

    def _accept_snapshot(self, state: dict[str, Any], base_info: dict[str, Any]) -> UpstreamSnapshot:
        snapshot = UpstreamSnapshot.build(
            state, base_info, self._clock(), expected_side=self.side
        )
        self._snapshot = snapshot
        self._cached_state = dict(state)
        self._cached_base_info = dict(base_info)
        return snapshot

    async def _refresh(self, client: Any) -> UpstreamSnapshot:
        if self._cached_base_info is None:
            raise ProtocolError("UPSTREAM_UNAVAILABLE", "base information is unavailable")
        state = await client.get_state()
        return self._accept_snapshot(state, self._cached_base_info)

    async def _run_connected(self, client: Any) -> None:
        while not self._stopping:
            command = self._arbiter.next_command(self._clock())
            if command is None:
                await self._hold_once(client)
                await self._wait_period()
                continue
            await self._execute(client, command)
            self._arbiter.complete(command, self._clock())

    async def _hold_once(self, client: Any) -> None:
        generation = self._generation
        snapshot = await self._refresh(client)
        if generation != self._generation:
            return
        await client.servo_j(snapshot.actual_joints, snapshot.joint_default_speed)
        if generation != self._generation:
            return
        snapshot = await self._refresh(client)
        if generation != self._generation:
            return
        await client.move_gripper(snapshot.actual_gripper, snapshot.gripper_default_speed)

    async def _execute(self, client: Any, command: QueuedCommand) -> None:
        generation = self._generation
        snapshot = await self._refresh(client)
        if self._arbiter.owner != command.lane or generation != self._generation:
            return
        request = command.request
        if isinstance(request, BridgeServoRequest):
            speed = request.speed or snapshot.joint_default_speed
            plan = plan_joint_motion(
                snapshot.actual_joints,
                request.target_joints,
                speed,
                snapshot.joint_max_speed,
                command.deadline,
                self._clock(),
            )
            if plan is not None:
                await self._execute_joints(client, command, generation, plan)
        else:
            speed = request.speed or snapshot.gripper_default_speed
            plan = plan_gripper_motion(
                snapshot.actual_gripper,
                request.target_position,
                speed,
                snapshot.gripper_max_speed,
                command.deadline,
                self._clock(),
            )
            if plan is not None:
                await self._execute_gripper(client, command, generation, plan)

    def _still_owns(self, command: QueuedCommand, generation: int) -> bool:
        return self._generation == generation and self._arbiter.owner == command.lane

    async def _execute_joints(
        self,
        client: Any,
        command: QueuedCommand,
        generation: int,
        plan: JointMotionPlan,
    ) -> None:
        commanded: tuple[float, ...] | None = None
        while self._still_owns(command, generation):
            snapshot = await self._refresh(client)
            if not self._still_owns(command, generation):
                return
            if commanded is None:
                commanded = snapshot.actual_joints
            elif commanded == plan.target and all(
                abs(actual - expected) <= _JOINT_SETTLED_TOLERANCE_DEG
                for actual, expected in zip(
                    snapshot.actual_joints, plan.target, strict=True
                )
            ):
                return
            target = step_joint_motion(
                commanded, plan, dt=self._period, now=self._clock()
            )
            if target is None:
                return
            await client.servo_j(target, plan.speed)
            commanded = target
            await self._wait_period()

    async def _execute_gripper(
        self,
        client: Any,
        command: QueuedCommand,
        generation: int,
        plan: GripperMotionPlan,
    ) -> None:
        while self._still_owns(command, generation):
            snapshot = await self._refresh(client)
            if not self._still_owns(command, generation):
                return
            target = step_gripper_motion(
                snapshot.actual_gripper, plan, dt=self._period, now=self._clock()
            )
            if target is None:
                return
            await client.move_gripper(target, plan.speed)
            if target == plan.target:
                return
            await self._wait_period()

    async def _wait_period(self) -> None:
        if self._wakeup.is_set():
            self._wakeup.clear()
            return
        wake_task = asyncio.create_task(self._wakeup.wait())
        sleep_task = asyncio.create_task(self._sleep(self._period))
        try:
            _done, pending = await asyncio.wait(
                (wake_task, sleep_task), return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in pending:
                with suppress(asyncio.CancelledError):
                    await task
        finally:
            for task in (wake_task, sleep_task):
                if not task.done():
                    task.cancel()
            self._wakeup.clear()
