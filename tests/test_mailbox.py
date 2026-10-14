# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from piper_control.mailbox import LatestValue


def test_latest_overwrites_and_take_clears_value():
    box = LatestValue[int]()

    box.put(1)
    box.put(2)

    assert box.take() == 2
    assert box.take() is None


def test_simultaneous_consumers_take_one_token_at_most_once():
    box = LatestValue[object]()
    token = object()
    consumer_count = 8
    start = Barrier(consumer_count + 1)
    box.put(token)

    def take_after_barrier():
        start.wait()
        return box.take()

    with ThreadPoolExecutor(max_workers=consumer_count) as executor:
        futures = [executor.submit(take_after_barrier) for _ in range(consumer_count)]
        start.wait()
        results = [future.result() for future in futures]

    assert sum(value is token for value in results) == 1
    assert sum(value is None for value in results) == consumer_count - 1


def test_mailbox_remains_usable_after_concurrent_production_and_consumption():
    box = LatestValue[object]()
    start = Barrier(3)

    def produce():
        start.wait()
        for value in range(1000):
            box.put(value)

    def consume():
        start.wait()
        for _ in range(1000):
            box.take()

    with ThreadPoolExecutor(max_workers=2) as executor:
        producer = executor.submit(produce)
        consumer = executor.submit(consume)
        start.wait()
        producer.result()
        consumer.result()

    sentinel = object()
    box.put(sentinel)
    assert box.take() is sentinel
    assert box.take() is None
