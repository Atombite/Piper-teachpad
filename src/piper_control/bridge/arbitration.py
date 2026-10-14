# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Pure deterministic priority arbitration for one Bridge-controlled arm."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite

from piper_control.bridge.protocol import BridgeGripperRequest, BridgeServoRequest


MotionRequest = BridgeServoRequest | BridgeGripperRequest


@dataclass(frozen=True, slots=True)
class LaneId:
    connection_id: str
    priority: int


@dataclass(frozen=True, slots=True)
class QueuedCommand:
    request: MotionRequest
    lane: LaneId
    received_at: float
    deadline: float
    sequence: int


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    accepted: bool
    queued: bool
    code: str | None = None
    preempted: LaneId | None = None


class PriorityArbiter:
    """Own per-lane queues and select one global arm-and-gripper owner."""

    def __init__(
        self,
        *,
        queue_capacity: int,
        command_ttl_s: float = 1.0,
        owner_timeout_s: float = 3.0,
    ) -> None:
        if isinstance(queue_capacity, bool) or not isinstance(queue_capacity, int):
            raise TypeError("queue_capacity must be an integer")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        for label, value in (
            ("command_ttl_s", command_ttl_s),
            ("owner_timeout_s", owner_timeout_s),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} must be a number")
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be positive and finite")
        self.queue_capacity = queue_capacity
        self.command_ttl_s = float(command_ttl_s)
        self.owner_timeout_s = float(owner_timeout_s)
        self._queues: dict[LaneId, deque[QueuedCommand]] = {}
        self._sequence = 0
        self.owner: LaneId | None = None
        self.owner_idle_deadline: float | None = None

    @staticmethod
    def _checked_now(now: float) -> float:
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise TypeError("now must be a number")
        if not isfinite(now):
            raise ValueError("now must be finite")
        return float(now)

    def _expire(self, now: float) -> None:
        empty: list[LaneId] = []
        for lane, queue in self._queues.items():
            if queue:
                retained = deque(command for command in queue if command.deadline > now)
                if len(retained) != len(queue):
                    self._queues[lane] = retained
                queue = retained
            if not queue:
                empty.append(lane)
        for lane in empty:
            self._queues.pop(lane, None)
        if (
            self.owner is not None
            and self.owner_idle_deadline is not None
            and now >= self.owner_idle_deadline
        ):
            self.owner = None
            self.owner_idle_deadline = None

    def _select_owner(self) -> None:
        candidates = [
            (lane, queue[0])
            for lane, queue in self._queues.items()
            if queue
        ]
        if not candidates:
            self.owner = None
            self.owner_idle_deadline = None
            return
        lane, _first = min(
            candidates,
            key=lambda item: (-item[0].priority, item[1].sequence),
        )
        self.owner = lane
        latest_received = max(command.received_at for command in self._queues[lane])
        self.owner_idle_deadline = latest_received + self.owner_timeout_s

    def enqueue(
        self, connection_id: str, request: MotionRequest, now: float
    ) -> EnqueueResult:
        now = self._checked_now(now)
        if not isinstance(connection_id, str) or not connection_id:
            raise ValueError("connection_id must be a non-empty string")
        if not isinstance(request, (BridgeServoRequest, BridgeGripperRequest)):
            raise TypeError("request must be a Bridge motion request")
        self._expire(now)
        if self.owner is None:
            self._select_owner()

        lane = LaneId(connection_id, request.priority)
        queue = self._queues.setdefault(lane, deque())
        if len(queue) >= self.queue_capacity:
            return EnqueueResult(
                accepted=False,
                queued=False,
                code="QUEUE_FULL",
            )

        command = QueuedCommand(
            request=request,
            lane=lane,
            received_at=now,
            deadline=now + self.command_ttl_s,
            sequence=self._sequence,
        )
        self._sequence += 1
        queue.append(command)

        preempted: LaneId | None = None
        if self.owner is None:
            self.owner = lane
            self.owner_idle_deadline = now + self.owner_timeout_s
        elif lane == self.owner:
            self.owner_idle_deadline = now + self.owner_timeout_s
        elif lane.priority > self.owner.priority:
            preempted = self.owner
            self.owner = lane
            self.owner_idle_deadline = now + self.owner_timeout_s
        return EnqueueResult(
            accepted=True,
            queued=True,
            preempted=preempted,
        )

    def next_command(self, now: float) -> QueuedCommand | None:
        now = self._checked_now(now)
        self._expire(now)
        if self.owner is None:
            self._select_owner()
        if self.owner is None:
            return None
        queue = self._queues.get(self.owner)
        if not queue:
            return None
        command = queue.popleft()
        if not queue:
            self._queues.pop(self.owner, None)
        return command

    def complete(self, command: QueuedCommand, now: float) -> None:
        if not isinstance(command, QueuedCommand):
            raise TypeError("command must be a QueuedCommand")
        now = self._checked_now(now)
        self._expire(now)
        if self.owner is None:
            self._select_owner()

    def release(self, connection_id: str, priority: int, now: float) -> bool:
        previous_owner = self.owner
        now = self._checked_now(now)
        self._expire(now)
        lane = LaneId(connection_id, priority)
        self._queues.pop(lane, None)
        if self.owner == lane:
            self.owner = None
            self.owner_idle_deadline = None
        if self.owner is None:
            self._select_owner()
        return self.owner != previous_owner

    def disconnect(self, connection_id: str, now: float) -> bool:
        previous_owner = self.owner
        now = self._checked_now(now)
        self._expire(now)
        removed = {
            lane for lane in self._queues if lane.connection_id == connection_id
        }
        for lane in removed:
            self._queues.pop(lane, None)
        if self.owner is not None and self.owner.connection_id == connection_id:
            self.owner = None
            self.owner_idle_deadline = None
        if self.owner is None:
            self._select_owner()
        return self.owner != previous_owner

    def clear(self) -> None:
        self._queues.clear()
        self.owner = None
        self.owner_idle_deadline = None
