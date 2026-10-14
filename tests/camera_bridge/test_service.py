# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from threading import Barrier, Event, Lock, Thread
import time
from uuid import UUID

import pytest

from piper_control.camera_bridge.config import (
    CameraBridgeConfig,
    CameraName,
    StreamName,
)
from piper_control.camera_bridge.frames import CameraFrameSet, EncodedFrame
from piper_control.camera_bridge.publisher import CameraStatus
from piper_control.camera_bridge.service import CameraBridgeService, CameraFault
from piper_control.camera_bridge.sources.base import (
    DeviceDisconnected,
    FatalFrameError,
    FrameTimeout,
)


def service_config(
    *,
    frame_timeout_ms: int = 200,
    startup_timeout_ms: int = 10000,
    reconnect_timeout_ms: int = 1000,
) -> CameraBridgeConfig:
    color = {
        "width": 640,
        "height": 480,
        "fps": 30,
        "capture_format": "bgr8",
        "encoding": "jpeg",
    }
    depth = {
        "width": 640,
        "height": 480,
        "fps": 30,
        "capture_format": "z16",
        "encoding": "raw",
    }
    infrared = {
        "width": 640,
        "height": 480,
        "fps": 30,
        "capture_format": "y8",
        "encoding": "raw",
    }
    return CameraBridgeConfig.model_validate(
        {
            "bind": "tcp://127.0.0.1:15555",
            "queue_size": 2,
            "jpeg_quality": 90,
            "recovery": {
                "frame_timeout_ms": frame_timeout_ms,
                "startup_timeout_ms": startup_timeout_ms,
                "max_consecutive_frame_failures": 3,
                "reconnect_timeout_ms": reconnect_timeout_ms,
                "max_reconnect_attempts": 1,
            },
            "cameras": {
                "left_wrist": {
                    "driver": "v4l2",
                    "device": (
                        "/dev/v4l/by-id/"
                        "LEFT_WRIST_CAMERA_EXAMPLE-video-index0"
                    ),
                    "streams": {
                        "color": {**color, "capture_format": "mjpeg"}
                    },
                },
                "right_wrist": {
                    "driver": "realsense",
                    "serial": "RIGHT_WRIST_SERIAL_EXAMPLE",
                    "streams": {"color": color, "depth": depth},
                },
                "head": {
                    "driver": "realsense",
                    "serial": "HEAD_SERIAL_EXAMPLE",
                    "streams": {
                        "color": color,
                        "depth": depth,
                        "infrared_1": infrared,
                        "infrared_2": infrared,
                    },
                },
            },
        }
    )


def camera_frameset(camera: CameraName, sequence: int) -> CameraFrameSet:
    frame = EncodedFrame(
        stream=StreamName.COLOR,
        payload=f"{camera.value}-{sequence}".encode(),
        width=640,
        height=480,
        encoding="jpeg",
        pixel_format="bgr8",
    )
    return CameraFrameSet(
        camera=camera,
        frameset_sequence=sequence,
        host_monotonic_ns=sequence,
        host_unix_ns=sequence,
        device_timestamp_ms=float(sequence),
        frames=(frame,),
    )


class SequenceSource:
    def __init__(
        self,
        camera: CameraName,
        outcomes: list[object] | None = None,
        *,
        open_error: Exception | None = None,
        open_outcomes: list[object] | None = None,
    ) -> None:
        self.camera = camera
        self.identity = f"identity-{camera.value}"
        self.outcomes = deque(outcomes or [])
        self.open_error = open_error
        self.open_outcomes = deque(open_outcomes or [])
        self.open_count = 0
        self.close_count = 0
        self.read_count = 0
        self.present = True
        self._lock = Lock()

    def open(self, _first_frame_timeout_ms: int) -> CameraFrameSet:
        with self._lock:
            self.open_count += 1
            if self.open_outcomes:
                outcome = self.open_outcomes.popleft()
                if isinstance(outcome, Exception):
                    raise outcome
                assert isinstance(outcome, CameraFrameSet)
                return outcome
            if self.open_error is not None:
                raise self.open_error
            return camera_frameset(self.camera, self.open_count * 100)

    def read(self, _timeout_ms: int) -> CameraFrameSet:
        with self._lock:
            self.read_count += 1
            if self.outcomes:
                outcome = self.outcomes.popleft()
                if isinstance(outcome, Exception):
                    raise outcome
                assert isinstance(outcome, CameraFrameSet)
                return outcome
            sequence = self.read_count
        time.sleep(0.002)
        return camera_frameset(self.camera, sequence)

    def close(self) -> None:
        with self._lock:
            self.close_count += 1

    def read_startup(self, _timeout_ms: int) -> CameraFrameSet:
        return camera_frameset(self.camera, self.open_count * 100 + 1)

    def is_present(self) -> bool:
        return self.present


class RecordingPublisher:
    def __init__(self) -> None:
        self.open_count = 0
        self.close_count = 0
        self.framesets: list[tuple[UUID, CameraFrameSet]] = []
        self.statuses: list[CameraStatus] = []
        self.reset_count = 0
        self._lock = Lock()

    def open(self) -> None:
        self.open_count += 1

    def publish_frameset(self, session_id: UUID, frameset: CameraFrameSet) -> int:
        with self._lock:
            self.framesets.append((session_id, frameset))
        return len(frameset.frames)

    def publish_status(self, status: CameraStatus) -> bool:
        with self._lock:
            self.statuses.append(status)
        return True

    def reset_sequences(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.close_count += 1


def sources(**overrides: SequenceSource) -> dict[CameraName, SequenceSource]:
    result = {
        name: SequenceSource(name)
        for name in (
            CameraName.LEFT_WRIST,
            CameraName.RIGHT_WRIST,
            CameraName.HEAD,
        )
    }
    result.update({CameraName(key): value for key, value in overrides.items()})
    return result


def wait_until(predicate, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("condition was not reached before deadline")


def test_startup_opens_and_primes_all_sources_before_publisher() -> None:
    events: list[str] = []

    class OrderedSource(SequenceSource):
        def open(self, timeout_ms: int) -> CameraFrameSet:
            events.append(f"open:{self.camera.value}")
            return super().open(timeout_ms)

    class OrderedPublisher(RecordingPublisher):
        def open(self) -> None:
            events.append("publisher:open")
            super().open()

    camera_sources = {
        name: OrderedSource(name)
        for name in (
            CameraName.LEFT_WRIST,
            CameraName.RIGHT_WRIST,
            CameraName.HEAD,
        )
    }
    publisher = OrderedPublisher()
    service = CameraBridgeService(service_config(), camera_sources, publisher)

    service.start()
    service.stop()

    assert events[:4] == [
        "open:left_wrist",
        "open:right_wrist",
        "open:head",
        "publisher:open",
    ]
    assert publisher.open_count == 1


def test_startup_uses_dedicated_driver_warmup_budget() -> None:
    open_timeouts: dict[CameraName, int] = {}

    class TimeoutRecordingSource(SequenceSource):
        def open(self, timeout_ms: int) -> CameraFrameSet:
            open_timeouts[self.camera] = timeout_ms
            return super().open(timeout_ms)

    camera_sources = {
        camera: TimeoutRecordingSource(camera) for camera in CameraName
    }
    service = CameraBridgeService(
        service_config(frame_timeout_ms=200, reconnect_timeout_ms=1000),
        camera_sources,
        RecordingPublisher(),
    )

    service.start()
    service.stop()

    assert open_timeouts == {camera: 10000 for camera in CameraName}


def test_startup_immediately_retries_one_driver_frame_timeout(caplog) -> None:
    right = SequenceSource(
        CameraName.RIGHT_WRIST,
        open_outcomes=[
            FrameTimeout("cold SDK produced no frame"),
            camera_frameset(CameraName.RIGHT_WRIST, 200),
        ],
    )
    publisher = RecordingPublisher()
    service = CameraBridgeService(
        service_config(), sources(right_wrist=right), publisher
    )

    service.start()
    service.stop()

    assert right.open_count == 2
    assert right.close_count == 2
    assert publisher.open_count == 1
    assert "event=camera_startup_retry" in caplog.text
    assert "camera=right_wrist" in caplog.text


def test_startup_second_driver_frame_timeout_fails_without_third_attempt() -> None:
    head = SequenceSource(
        CameraName.HEAD,
        open_outcomes=[
            FrameTimeout("cold SDK produced no frame"),
            FrameTimeout("second attempt also timed out"),
        ],
    )
    publisher = RecordingPublisher()
    service = CameraBridgeService(service_config(), sources(head=head), publisher)

    with pytest.raises(CameraFault, match="head") as error:
        service.start()

    assert head.open_count == 2
    assert error.value.reconnect_attempts == 1
    assert publisher.open_count == 0


def test_startup_failure_closes_open_sources_and_never_opens_publisher() -> None:
    right = SequenceSource(
        CameraName.RIGHT_WRIST,
        open_error=DeviceDisconnected("right_wrist absent"),
    )
    camera_sources = sources(right_wrist=right)
    publisher = RecordingPublisher()
    service = CameraBridgeService(service_config(), camera_sources, publisher)

    with pytest.raises(CameraFault, match="right_wrist"):
        service.start()

    assert camera_sources[CameraName.LEFT_WRIST].close_count == 1
    assert right.close_count == 1
    assert camera_sources[CameraName.HEAD].open_count == 0
    assert publisher.open_count == 0


def test_shutdown_cancels_blocked_startup_without_opening_later_sources() -> None:
    open_started = Event()
    release = Event()

    class BlockingStartupSource(SequenceSource):
        def open(self, timeout_ms: int) -> CameraFrameSet:
            self.open_count += 1
            open_started.set()
            release.wait(1.0)
            return camera_frameset(self.camera, 100)

    left = BlockingStartupSource(CameraName.LEFT_WRIST)
    camera_sources = sources(left_wrist=left)
    publisher = RecordingPublisher()
    service = CameraBridgeService(
        service_config(startup_timeout_ms=1000), camera_sources, publisher
    )
    shutdown = Event()
    result: list[int] = []
    thread = Thread(target=lambda: result.append(service.run(shutdown)))
    thread.start()
    assert open_started.wait(0.2)

    shutdown.set()
    thread.join(0.3)
    exited_promptly = not thread.is_alive()
    release.set()
    thread.join(1.0)

    assert exited_promptly
    assert result == [0]
    assert camera_sources[CameraName.RIGHT_WRIST].open_count == 0
    assert camera_sources[CameraName.HEAD].open_count == 0
    assert publisher.open_count == 0


def test_startup_refreshes_all_primed_frames_before_enabling_publication() -> None:
    allow_reads = Event()
    read_started = {camera: Event() for camera in CameraName}

    class StalePrimeSource(SequenceSource):
        def open(self, timeout_ms: int) -> CameraFrameSet:
            with self._lock:
                self.open_count += 1
            return camera_frameset(self.camera, 1)

        def read_startup(self, timeout_ms: int) -> CameraFrameSet:
            read_started[self.camera].set()
            allow_reads.wait(1.0)
            with self._lock:
                self.read_count += 1
                sequence = self.read_count + 1
            return camera_frameset(self.camera, sequence)

    camera_sources = {camera: StalePrimeSource(camera) for camera in CameraName}
    publisher = RecordingPublisher()
    service = CameraBridgeService(service_config(), camera_sources, publisher)
    start_thread = Thread(target=service.start)
    start_thread.start()

    assert all(event.wait(0.3) for event in read_started.values())
    allow_reads.set()
    start_thread.join(1.0)
    wait_until(
        lambda: {frameset.camera for _session, frameset in publisher.framesets}
        == set(CameraName)
    )
    service.stop()

    first_by_camera: dict[CameraName, CameraFrameSet] = {}
    for _session, frameset in publisher.framesets:
        first_by_camera.setdefault(frameset.camera, frameset)
    assert set(first_by_camera) == set(CameraName)
    assert all(frame.frameset_sequence > 1 for frame in first_by_camera.values())


def test_healthy_run_publishes_each_camera_independently_and_stops_cleanly() -> None:
    camera_sources = sources()
    publisher = RecordingPublisher()
    service = CameraBridgeService(service_config(), camera_sources, publisher)
    shutdown = Event()
    result: list[int] = []
    thread = Thread(target=lambda: result.append(service.run(shutdown)))
    thread.start()

    wait_until(
        lambda: {item.camera for _session, item in publisher.framesets}
        == set(CameraName)
    )
    shutdown.set()
    thread.join(1.0)

    assert result == [0]
    assert not thread.is_alive()
    assert publisher.close_count == 1
    assert all(source.close_count == 1 for source in camera_sources.values())
    assert service.worker_threads_alive == 0


def run_service_until_exit(
    service: CameraBridgeService, timeout_s: float = 1.0
) -> tuple[Thread, list[int], Event]:
    shutdown = Event()
    result: list[int] = []
    thread = Thread(target=lambda: result.append(service.run(shutdown)))
    thread.start()
    thread.join(timeout_s)
    return thread, result, shutdown


def test_third_consecutive_frame_timeout_faults_without_reopen() -> None:
    left = SequenceSource(
        CameraName.LEFT_WRIST,
        outcomes=[FrameTimeout("late 1"), FrameTimeout("late 2"), FrameTimeout("late 3")],
    )
    camera_sources = sources(left_wrist=left)
    service = CameraBridgeService(service_config(), camera_sources, RecordingPublisher())

    thread, result, _shutdown = run_service_until_exit(service)

    assert not thread.is_alive()
    assert result == [1]
    assert left.open_count == 1
    assert service.faults[0].category == "frame_timeout"
    assert service.faults[0].consecutive_failures == 3
    assert service.faults[0].reconnect_attempts == 0


def test_blocked_read_probes_absence_and_uses_single_reconnect() -> None:
    release = Event()

    class UnpluggedDuringReadSource(SequenceSource):
        def open(self, timeout_ms: int) -> CameraFrameSet:
            if self.open_count:
                self.open_count += 1
                raise DeviceDisconnected("D405 remains absent")
            return super().open(timeout_ms)

        def read(self, timeout_ms: int) -> CameraFrameSet:
            self.present = False
            release.wait(1.0)
            raise FrameTimeout("late SDK classification")

        def close(self) -> None:
            super().close()
            release.set()

    right = UnpluggedDuringReadSource(CameraName.RIGHT_WRIST)
    service = CameraBridgeService(
        service_config(frame_timeout_ms=20, reconnect_timeout_ms=100),
        sources(right_wrist=right),
        RecordingPublisher(),
    )

    thread, result, _shutdown = run_service_until_exit(service)
    release.set()

    assert not thread.is_alive()
    assert result == [1]
    assert right.open_count == 2
    assert service.faults[0].category == "reconnect_failed"
    assert service.faults[0].reconnect_attempts == 1


def test_reconnect_waits_for_abandoned_read_to_finish_before_reopening() -> None:
    close_called = Event()
    finish_read = Event()

    class SlowlyStoppingSource(SequenceSource):
        def __init__(self) -> None:
            super().__init__(CameraName.RIGHT_WRIST)
            self.active_reads = 0
            self.overlap_detected = False

        def open(self, timeout_ms: int) -> CameraFrameSet:
            with self._lock:
                if self.active_reads:
                    self.overlap_detected = True
            self.present = True
            return super().open(timeout_ms)

        def read(self, timeout_ms: int) -> CameraFrameSet:
            with self._lock:
                self.read_count += 1
                call = self.read_count
                self.active_reads += 1
                if self.active_reads > 1:
                    self.overlap_detected = True
            try:
                if call == 1:
                    self.present = False
                    finish_read.wait(1.0)
                    raise DeviceDisconnected("old read stopped")
                time.sleep(0.002)
                return camera_frameset(self.camera, call)
            finally:
                with self._lock:
                    self.active_reads -= 1

        def close(self) -> None:
            super().close()
            close_called.set()

    right = SlowlyStoppingSource()
    service = CameraBridgeService(
        service_config(frame_timeout_ms=20, reconnect_timeout_ms=200),
        sources(right_wrist=right),
        RecordingPublisher(),
    )
    initial_session = service.session_id
    shutdown = Event()
    result: list[int] = []
    thread = Thread(target=lambda: result.append(service.run(shutdown)))
    thread.start()

    assert close_called.wait(0.2)
    time.sleep(0.03)
    assert right.open_count == 1
    finish_read.set()
    wait_until(lambda: service.session_id != initial_session)
    shutdown.set()
    thread.join(1.0)

    assert result == [0]
    assert right.open_count == 2
    assert right.overlap_detected is False


def test_blocked_presence_probe_cannot_bypass_fail_fast_recovery() -> None:
    read_release = Event()
    probe_started = Event()
    probe_release = Event()

    class BlockingPresenceSource(SequenceSource):
        def open(self, timeout_ms: int) -> CameraFrameSet:
            if self.open_count:
                self.open_count += 1
                raise DeviceDisconnected("driver remains unavailable")
            return super().open(timeout_ms)

        def read(self, timeout_ms: int) -> CameraFrameSet:
            read_release.wait(1.0)
            raise FrameTimeout("late read result")

        def is_present(self) -> bool:
            probe_started.set()
            probe_release.wait(1.0)
            return False

        def close(self) -> None:
            super().close()
            read_release.set()

    right = BlockingPresenceSource(CameraName.RIGHT_WRIST)
    service = CameraBridgeService(
        service_config(frame_timeout_ms=20, reconnect_timeout_ms=100),
        sources(right_wrist=right),
        RecordingPublisher(),
    )
    shutdown = Event()
    result: list[int] = []
    thread = Thread(target=lambda: result.append(service.run(shutdown)))
    thread.start()

    assert probe_started.wait(0.2)
    thread.join(0.3)
    exited_within_bound = not thread.is_alive()
    probe_release.set()
    thread.join(1.0)

    assert exited_within_bound
    assert result == [1]
    assert right.open_count == 2
    assert service.faults[0].category == "reconnect_failed"
    assert "presence probe exceeded" in service.faults[0].cause


def test_presence_probe_error_uses_the_single_reconnect_path() -> None:
    read_release = Event()

    class FailedPresenceSource(SequenceSource):
        def open(self, timeout_ms: int) -> CameraFrameSet:
            if self.open_count:
                self.open_count += 1
                raise DeviceDisconnected("driver remains unavailable")
            return super().open(timeout_ms)

        def read(self, timeout_ms: int) -> CameraFrameSet:
            read_release.wait(1.0)
            raise FrameTimeout("late read result")

        def is_present(self) -> bool:
            raise RuntimeError("SDK enumeration failed")

        def close(self) -> None:
            super().close()
            read_release.set()

    right = FailedPresenceSource(CameraName.RIGHT_WRIST)
    service = CameraBridgeService(
        service_config(frame_timeout_ms=20, reconnect_timeout_ms=100),
        sources(right_wrist=right),
        RecordingPublisher(),
    )

    thread, result, _shutdown = run_service_until_exit(service)

    assert not thread.is_alive()
    assert result == [1]
    assert right.open_count == 2
    assert service.faults[0].category == "reconnect_failed"
    assert "SDK enumeration failed" in service.faults[0].cause


def test_present_device_with_blocked_read_keeps_three_timeout_policy() -> None:
    release = Event()

    class StalledPresentSource(SequenceSource):
        def read(self, timeout_ms: int) -> CameraFrameSet:
            release.wait(1.0)
            raise FrameTimeout("late read result")

        def close(self) -> None:
            super().close()
            release.set()

    left = StalledPresentSource(CameraName.LEFT_WRIST)
    service = CameraBridgeService(
        service_config(frame_timeout_ms=20, reconnect_timeout_ms=100),
        sources(left_wrist=left),
        RecordingPublisher(),
    )

    thread, result, _shutdown = run_service_until_exit(service)

    assert not thread.is_alive()
    assert result == [1]
    assert left.open_count == 1
    assert service.faults[0].category == "frame_timeout"
    assert service.faults[0].consecutive_failures == 3


def test_slow_presence_probe_is_charged_to_absolute_failure_deadline() -> None:
    release = Event()
    probe_started_at: list[float] = []

    class SlowlyProbedPresentSource(SequenceSource):
        def read(self, timeout_ms: int) -> CameraFrameSet:
            release.wait(1.0)
            raise FrameTimeout("late read result")

        def is_present(self) -> bool:
            probe_started_at.append(time.monotonic())
            time.sleep(0.04)
            return True

        def close(self) -> None:
            super().close()
            release.set()

    left = SlowlyProbedPresentSource(CameraName.LEFT_WRIST)
    service = CameraBridgeService(
        service_config(frame_timeout_ms=100, reconnect_timeout_ms=100),
        sources(left_wrist=left),
        RecordingPublisher(),
    )

    thread, result, _shutdown = run_service_until_exit(service)

    assert not thread.is_alive()
    assert result == [1]
    assert len(probe_started_at) == 3
    assert probe_started_at[1] - probe_started_at[0] < 0.12
    assert probe_started_at[2] - probe_started_at[1] < 0.12
    assert service.faults[0].category == "frame_timeout"


def test_device_disappearing_in_third_interval_still_uses_reconnect() -> None:
    release = Event()
    probe_count = 0

    class LateUnplugSource(SequenceSource):
        def open(self, timeout_ms: int) -> CameraFrameSet:
            if self.open_count:
                self.open_count += 1
                raise DeviceDisconnected("device remains absent")
            return super().open(timeout_ms)

        def read(self, timeout_ms: int) -> CameraFrameSet:
            release.wait(1.0)
            raise FrameTimeout("late read result")

        def is_present(self) -> bool:
            nonlocal probe_count
            probe_count += 1
            return probe_count < 3

        def close(self) -> None:
            super().close()
            release.set()

    right = LateUnplugSource(CameraName.RIGHT_WRIST)
    service = CameraBridgeService(
        service_config(frame_timeout_ms=20, reconnect_timeout_ms=100),
        sources(right_wrist=right),
        RecordingPublisher(),
    )

    thread, result, _shutdown = run_service_until_exit(service)

    assert not thread.is_alive()
    assert result == [1]
    assert probe_count == 3
    assert right.open_count == 2
    assert service.faults[0].category == "reconnect_failed"
    assert service.faults[0].reconnect_attempts == 1


def test_valid_frameset_resets_consecutive_timeout_count() -> None:
    left = SequenceSource(
        CameraName.LEFT_WRIST,
        outcomes=[
            FrameTimeout("late 1"),
            FrameTimeout("late 2"),
            camera_frameset(CameraName.LEFT_WRIST, 9),
            FrameTimeout("late 3"),
            FrameTimeout("late 4"),
        ],
    )
    camera_sources = sources(left_wrist=left)
    service = CameraBridgeService(service_config(), camera_sources, RecordingPublisher())
    shutdown = Event()
    result: list[int] = []
    thread = Thread(target=lambda: result.append(service.run(shutdown)))
    thread.start()

    wait_until(lambda: left.read_count >= 5)
    assert service.faults == ()
    shutdown.set()
    thread.join(1.0)

    assert result == [0]
    assert left.open_count == 1


def test_deterministic_frame_error_faults_immediately() -> None:
    head = SequenceSource(
        CameraName.HEAD,
        outcomes=[FatalFrameError("depth profile changed")],
    )
    service = CameraBridgeService(
        service_config(), sources(head=head), RecordingPublisher()
    )

    thread, result, _shutdown = run_service_until_exit(service)

    assert not thread.is_alive()
    assert result == [1]
    assert service.faults[0].category == "fatal_frame_error"
    assert service.faults[0].consecutive_failures == 0


def test_disconnect_pauses_all_output_and_successful_reconnect_rotates_session() -> None:
    reconnected = camera_frameset(CameraName.RIGHT_WRIST, 200)
    right = SequenceSource(
        CameraName.RIGHT_WRIST,
        outcomes=[DeviceDisconnected("USB link lost")],
        open_outcomes=[camera_frameset(CameraName.RIGHT_WRIST, 100), reconnected],
    )
    publisher = RecordingPublisher()
    service = CameraBridgeService(service_config(), sources(right_wrist=right), publisher)
    initial_session = service.session_id
    shutdown = Event()
    result: list[int] = []
    thread = Thread(target=lambda: result.append(service.run(shutdown)))
    thread.start()

    wait_until(lambda: service.session_id != initial_session)
    new_session = service.session_id
    wait_until(
        lambda: any(
            session == new_session and frameset.camera is CameraName.RIGHT_WRIST
            for session, frameset in publisher.framesets
        )
    )
    shutdown.set()
    thread.join(1.0)

    assert result == [0]
    assert right.open_count == 2
    assert publisher.reset_count >= 2
    assert any(status.state == "reconnecting" for status in publisher.statuses)
    assert service.faults == ()


def test_overlapping_reconnects_keep_gate_closed_until_every_camera_is_online() -> None:
    service = CameraBridgeService(
        service_config(), sources(), RecordingPublisher()
    )
    initial_session = service.session_id
    service._publish_gate.set()

    assert service._begin_reconnect(CameraName.RIGHT_WRIST, "right lost")
    assert service._begin_reconnect(CameraName.HEAD, "head lost")
    service._finish_reconnect(
        CameraName.RIGHT_WRIST,
        camera_frameset(CameraName.RIGHT_WRIST, 200),
    )

    assert not service._publish_gate.is_set()
    assert service.session_id == initial_session
    service._finish_reconnect(
        CameraName.HEAD,
        camera_frameset(CameraName.HEAD, 200),
    )
    assert service._publish_gate.is_set()
    assert service.session_id != initial_session


def test_session_cutover_waits_for_inflight_publication() -> None:
    publish_entered = Event()
    release_publish = Event()
    order: list[str] = []

    class BlockingPublisher(RecordingPublisher):
        def publish_frameset(
            self, session_id: UUID, frameset: CameraFrameSet
        ) -> int:
            publish_entered.set()
            release_publish.wait(1.0)
            order.append(f"publish:{session_id}")
            return super().publish_frameset(session_id, frameset)

    publisher = BlockingPublisher()
    service = CameraBridgeService(service_config(), sources(), publisher)
    service.start()
    assert publish_entered.wait(0.3)

    def reconnect() -> None:
        service._begin_reconnect(CameraName.RIGHT_WRIST, "USB lost")
        service._finish_reconnect(
            CameraName.RIGHT_WRIST,
            camera_frameset(CameraName.RIGHT_WRIST, 200),
        )
        order.append("cutover")

    reconnect_thread = Thread(target=reconnect)
    reconnect_thread.start()
    time.sleep(0.03)
    release_publish.set()
    reconnect_thread.join(1.0)
    service.stop()

    assert order.index(next(item for item in order if item.startswith("publish:"))) < order.index("cutover")


def test_failed_reconnect_faults_after_exactly_one_attempt() -> None:
    right = SequenceSource(
        CameraName.RIGHT_WRIST,
        outcomes=[DeviceDisconnected("USB link lost")],
        open_outcomes=[
            camera_frameset(CameraName.RIGHT_WRIST, 100),
            DeviceDisconnected("serial still absent"),
        ],
    )
    service = CameraBridgeService(
        service_config(), sources(right_wrist=right), RecordingPublisher()
    )

    thread, result, _shutdown = run_service_until_exit(service)

    assert not thread.is_alive()
    assert result == [1]
    assert right.open_count == 2
    assert service.faults[0].category == "reconnect_failed"
    assert service.faults[0].reconnect_attempts == 1


def test_late_reconnect_faults_at_monotonic_deadline() -> None:
    release = Event()

    class SlowReconnectSource(SequenceSource):
        def open(self, timeout_ms: int) -> CameraFrameSet:
            if self.open_count:
                self.open_count += 1
                release.wait(1.0)
                return camera_frameset(self.camera, 200)
            return super().open(timeout_ms)

    right = SlowReconnectSource(
        CameraName.RIGHT_WRIST,
        outcomes=[DeviceDisconnected("USB link lost")],
    )
    service = CameraBridgeService(
        service_config(frame_timeout_ms=20, reconnect_timeout_ms=40),
        sources(right_wrist=right),
        RecordingPublisher(),
    )

    thread, result, _shutdown = run_service_until_exit(service)
    release.set()

    assert not thread.is_alive()
    assert result == [1]
    assert service.faults[0].category == "reconnect_timeout"
    assert service.faults[0].reconnect_attempts == 1


def test_wall_clock_jumps_do_not_change_reconnect_deadline() -> None:
    wall_values = iter([10**18, 1, 10**19, 2] * 100)
    right = SequenceSource(
        CameraName.RIGHT_WRIST,
        outcomes=[DeviceDisconnected("USB link lost")],
        open_outcomes=[
            camera_frameset(CameraName.RIGHT_WRIST, 100),
            camera_frameset(CameraName.RIGHT_WRIST, 200),
        ],
    )
    service = CameraBridgeService(
        service_config(frame_timeout_ms=20, reconnect_timeout_ms=100),
        sources(right_wrist=right),
        RecordingPublisher(),
        unix_ns=lambda: next(wall_values),
    )
    initial_session = service.session_id
    shutdown = Event()
    result: list[int] = []
    thread = Thread(target=lambda: result.append(service.run(shutdown)))
    thread.start()

    wait_until(lambda: service.session_id != initial_session)
    shutdown.set()
    thread.join(1.0)

    assert result == [0]
    assert service.faults == ()


def test_shutdown_during_reconnect_returns_promptly() -> None:
    reconnect_started = Event()
    release = Event()

    class BlockingReconnectSource(SequenceSource):
        def open(self, timeout_ms: int) -> CameraFrameSet:
            if self.open_count:
                self.open_count += 1
                reconnect_started.set()
                release.wait(1.0)
                return camera_frameset(self.camera, 200)
            return super().open(timeout_ms)

    right = BlockingReconnectSource(
        CameraName.RIGHT_WRIST,
        outcomes=[DeviceDisconnected("USB link lost")],
    )
    service = CameraBridgeService(
        service_config(reconnect_timeout_ms=1000),
        sources(right_wrist=right),
        RecordingPublisher(),
    )
    shutdown = Event()
    result: list[int] = []
    thread = Thread(target=lambda: result.append(service.run(shutdown)))
    thread.start()
    assert reconnect_started.wait(0.5)

    shutdown.set()
    thread.join(0.3)
    release.set()

    assert not thread.is_alive()
    assert result == [0]
    assert service.faults == ()


def test_simultaneous_fatal_errors_are_both_logged_and_shutdown_is_idempotent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    barrier = Barrier(2)

    class BarrierFaultSource(SequenceSource):
        def read(self, timeout_ms: int) -> CameraFrameSet:
            barrier.wait(timeout=0.5)
            raise FatalFrameError(f"broken {self.identity}")

    left = BarrierFaultSource(CameraName.LEFT_WRIST)
    right = BarrierFaultSource(CameraName.RIGHT_WRIST)
    service = CameraBridgeService(
        service_config(),
        sources(left_wrist=left, right_wrist=right),
        RecordingPublisher(),
    )
    caplog.set_level("ERROR")

    thread, result, _shutdown = run_service_until_exit(service)
    service.stop()

    assert not thread.is_alive()
    assert result == [1]
    assert {fault.identity for fault in service.faults} == {
        left.identity,
        right.identity,
    }
    assert left.identity in caplog.text
    assert right.identity in caplog.text


def test_publisher_exception_faults_service_instead_of_silently_losing_vision() -> None:
    class BrokenPublisher(RecordingPublisher):
        def publish_frameset(
            self, session_id: UUID, frameset: CameraFrameSet
        ) -> int:
            raise RuntimeError("PUB socket failed")

    service = CameraBridgeService(service_config(), sources(), BrokenPublisher())

    thread, result, shutdown = run_service_until_exit(service, timeout_s=0.5)
    if thread.is_alive():
        shutdown.set()
        thread.join(1.0)

    assert result == [1]
    assert service.faults[0].category == "publisher_error"
    assert "PUB socket failed" in service.faults[0].cause


def test_publisher_open_failure_closes_all_started_sources() -> None:
    class BrokenPublisher(RecordingPublisher):
        def open(self) -> None:
            raise RuntimeError("bind failed")

    camera_sources = sources()
    service = CameraBridgeService(
        service_config(), camera_sources, BrokenPublisher()
    )

    thread, result, _shutdown = run_service_until_exit(service)

    assert not thread.is_alive()
    assert result == [1]
    assert service.faults[0].camera is None
    assert service.faults[0].category == "publisher_startup_error"
    assert all(source.close_count == 1 for source in camera_sources.values())


def test_readiness_notification_failure_faults_and_cleans_up_service() -> None:
    camera_sources = sources()
    publisher = RecordingPublisher()

    def fail_readiness() -> None:
        raise OSError("systemd notify socket unavailable")

    service = CameraBridgeService(
        service_config(),
        camera_sources,
        publisher,
        on_ready=fail_readiness,
    )

    thread, result, _shutdown = run_service_until_exit(service)

    assert not thread.is_alive()
    assert result == [1]
    assert service.faults[0].camera is None
    assert service.faults[0].category == "readiness_error"
    assert all(source.close_count == 1 for source in camera_sources.values())
    assert publisher.close_count == 1


def test_blocking_source_close_cannot_hold_shutdown_past_cleanup_deadline() -> None:
    close_started = Event()
    release = Event()

    class BlockingCloseSource(SequenceSource):
        def close(self) -> None:
            close_started.set()
            release.wait(1.0)
            super().close()

    left = BlockingCloseSource(CameraName.LEFT_WRIST)
    service = CameraBridgeService(
        service_config(reconnect_timeout_ms=100),
        sources(left_wrist=left),
        RecordingPublisher(),
    )
    shutdown = Event()
    result: list[int] = []
    thread = Thread(target=lambda: result.append(service.run(shutdown)))
    thread.start()
    wait_until(lambda: service.worker_threads_alive == 3)

    shutdown.set()
    assert close_started.wait(0.2)
    thread.join(0.4)
    exited_within_bound = not thread.is_alive()
    release.set()
    thread.join(1.0)

    assert exited_within_bound
    assert result == [0]
