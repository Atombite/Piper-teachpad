# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from piper_control.bridge.arbitration import LaneId, PriorityArbiter
from piper_control.bridge.protocol import parse_bridge_request


def servo(priority: int = 0, marker: float = 0.0):
    return parse_bridge_request(
        {
            "id": f"servo-{priority}-{marker}",
            "cmd": "servoJ",
            "target_joints": [marker] * 6,
            "priority": priority,
        }
    )


def gripper(priority: int = 0, marker: float = 0.0):
    return parse_bridge_request(
        {
            "id": f"gripper-{priority}-{marker}",
            "cmd": "move_gripper",
            "target_position": marker,
            "priority": priority,
        }
    )


def test_higher_priority_preempts_globally_across_joint_and_gripper() -> None:
    arbiter = PriorityArbiter(queue_capacity=500)
    low = arbiter.enqueue("a", servo(0, 1), now=0.0)
    assert low.accepted is True
    assert arbiter.owner == LaneId("a", 0)

    high = arbiter.enqueue("b", gripper(100, 0.4), now=0.1)
    assert high.accepted is True
    assert high.preempted == LaneId("a", 0)
    assert arbiter.owner == LaneId("b", 100)
    assert arbiter.next_command(0.1).request.cmd == "move_gripper"

    reverse = PriorityArbiter(queue_capacity=500)
    reverse.enqueue("a", gripper(0, 0.2), now=0.0)
    reverse.enqueue("b", servo(100, 2), now=0.1)
    assert reverse.owner == LaneId("b", 100)
    assert reverse.next_command(0.1).request.cmd == "servoJ"


def test_lower_priority_waits_and_fifo_is_preserved_within_a_lane() -> None:
    arbiter = PriorityArbiter(queue_capacity=10)
    arbiter.enqueue("high", servo(10, 1), now=0.0)
    arbiter.enqueue("low", servo(0, 2), now=0.1)
    arbiter.enqueue("low", gripper(0, 0.3), now=0.2)

    assert arbiter.next_command(0.2).lane == LaneId("high", 10)
    assert arbiter.next_command(0.2) is None
    arbiter.release("high", 10, now=0.2)
    first = arbiter.next_command(0.2)
    second = arbiter.next_command(0.2)
    assert first.request.cmd == "servoJ"
    assert second.request.cmd == "move_gripper"


def test_expiration_includes_queue_time_and_owner_retains_for_three_seconds() -> None:
    arbiter = PriorityArbiter(
        queue_capacity=10, command_ttl_s=1.0, owner_timeout_s=3.0
    )
    arbiter.enqueue("owner", servo(10, 1), now=0.0)
    command = arbiter.next_command(0.0)
    arbiter.complete(command, now=0.1)
    arbiter.enqueue("waiting", servo(0, 2), now=0.5)

    assert arbiter.next_command(2.9) is None
    assert arbiter.owner == LaneId("owner", 10)
    assert arbiter.owner_idle_deadline == pytest.approx(3.0)
    assert arbiter.next_command(3.0) is None
    assert arbiter.owner is None


def test_deadline_is_absolute_and_preempted_completion_cannot_reclaim_owner() -> None:
    arbiter = PriorityArbiter(queue_capacity=10, command_ttl_s=1.0)
    arbiter.enqueue("low", servo(0, 1), now=0.0)
    old = arbiter.next_command(0.0)
    arbiter.enqueue("waiting", servo(-1, 2), now=0.0)
    arbiter.enqueue("high", gripper(100, 0.2), now=0.5)
    assert arbiter.owner == LaneId("high", 100)

    arbiter.complete(old, now=0.6)
    assert arbiter.owner == LaneId("high", 100)
    arbiter.release("high", 100, now=1.0)
    assert arbiter.owner is None


def test_owner_activity_refreshes_idle_deadline_but_completion_does_not() -> None:
    arbiter = PriorityArbiter(queue_capacity=10, owner_timeout_s=3.0)
    arbiter.enqueue("a", servo(0, 1), now=0.0)
    first = arbiter.next_command(0.0)
    arbiter.complete(first, now=1.0)
    assert arbiter.owner_idle_deadline == pytest.approx(3.0)
    arbiter.enqueue("a", gripper(0, 0.2), now=2.5)
    assert arbiter.owner_idle_deadline == pytest.approx(5.5)


def test_same_priority_owner_is_sticky_then_hands_off_to_earliest_valid_lane() -> None:
    arbiter = PriorityArbiter(queue_capacity=10)
    arbiter.enqueue("owner", servo(7, 1), now=0.0)
    arbiter.enqueue("earlier", servo(7, 2), now=0.1)
    arbiter.enqueue("later", servo(7, 3), now=0.2)
    assert arbiter.owner == LaneId("owner", 7)

    arbiter.release("owner", 7, now=0.3)
    assert arbiter.owner == LaneId("earlier", 7)
    assert arbiter.next_command(0.3).request.target_joints == (2.0,) * 6


def test_external_priority_zero_is_a_normal_owner() -> None:
    arbiter = PriorityArbiter(queue_capacity=10)
    arbiter.enqueue("external", servo(0, 1), now=0.0)
    assert arbiter.owner == LaneId("external", 0)
    assert arbiter.next_command(0.0) is not None


def test_expired_entries_are_removed_before_queue_full_decision() -> None:
    arbiter = PriorityArbiter(queue_capacity=2, command_ttl_s=1.0)
    assert arbiter.enqueue("a", servo(0, 1), now=0.0).accepted
    assert arbiter.enqueue("a", servo(0, 2), now=0.1).accepted
    rejected = arbiter.enqueue("a", servo(0, 3), now=0.2)
    assert rejected.accepted is False
    assert rejected.code == "QUEUE_FULL"

    accepted = arbiter.enqueue("a", servo(0, 4), now=1.01)
    assert accepted.accepted is True


def test_non_integral_control_rate_capacity_is_supplied_as_ceil_498() -> None:
    arbiter = PriorityArbiter(queue_capacity=498)
    for index in range(498):
        assert arbiter.enqueue("a", servo(0, float(index)), now=0.0).accepted
    assert arbiter.enqueue("a", servo(0, 999), now=0.0).code == "QUEUE_FULL"


def test_release_clears_only_named_lane_and_disconnect_clears_all_client_lanes() -> None:
    arbiter = PriorityArbiter(queue_capacity=10)
    arbiter.enqueue("a", servo(10, 1), now=0.0)
    arbiter.enqueue("a", servo(1, 2), now=0.0)
    arbiter.enqueue("b", servo(5, 3), now=0.0)

    arbiter.release("a", 10, now=0.1)
    assert arbiter.owner == LaneId("b", 5)
    arbiter.release("b", 5, now=0.1)
    assert arbiter.owner == LaneId("a", 1)

    arbiter.enqueue("a", servo(20, 4), now=0.2)
    arbiter.enqueue("b", servo(15, 5), now=0.2)
    arbiter.disconnect("a", now=0.3)
    assert arbiter.owner == LaneId("b", 15)


def test_clear_removes_queues_owner_and_deadline() -> None:
    arbiter = PriorityArbiter(queue_capacity=10)
    arbiter.enqueue("a", servo(1, 1), now=0.0)
    arbiter.clear()
    assert arbiter.owner is None
    assert arbiter.owner_idle_deadline is None
    assert arbiter.next_command(0.0) is None
