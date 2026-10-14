# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Threaded camera-bridge lifecycle and safety supervision."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Callable, Generic, Literal, Mapping, TypeVar
from uuid import UUID, uuid4

from piper_control.camera_bridge.config import CameraBridgeConfig, CameraName
from piper_control.camera_bridge.frames import CameraFrameSet, LatestFrameQueue
from piper_control.camera_bridge.publisher import CameraPublisher, CameraStatus
from piper_control.camera_bridge.sources.base import (
    CameraSource,
    DeviceDisconnected,
    FatalFrameError,
    FrameTimeout,
)


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CameraFault(RuntimeError):
    camera: CameraName | None
    identity: str
    category: str
    cause: str
    consecutive_failures: int = 0
    reconnect_attempts: int = 0
    last_success_monotonic_ns: int | None = None

    def __str__(self) -> str:
        owner = self.camera.value if self.camera is not None else "service"
        return f"{owner} {self.category}: {self.cause}"


class _StartupCancelled(RuntimeError):
    """The process received shutdown while the startup barrier was active."""


class _BoundedCall(Generic[T]):
    """One daemon-owned driver call which is never invoked concurrently."""

    def __init__(
        self,
        function: Callable[[], T],
        *,
        name: str,
        on_late_result: Callable[[], None] | None = None,
    ) -> None:
        self._function = function
        self._on_late_result = on_late_result
        self._finished = Event()
        self._abandoned = Event()
        self._late_result_handled = Event()
        self._value: T | None = None
        self._error: Exception | None = None
        self._thread = Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._value = self._function()
        except Exception as exc:  # The worker classifies source exceptions.
            self._error = exc
        finally:
            self._finished.set()
            self._handle_late_result()

    def wait(
        self,
        timeout_ms: int,
        stop_event: Event,
        monotonic_ns: Callable[[], int],
    ) -> tuple[Literal["complete", "timeout", "stopped"], T | Exception | None]:
        deadline_ns = monotonic_ns() + timeout_ms * 1_000_000
        while True:
            if self._finished.is_set():
                if self._error is not None:
                    return "complete", self._error
                return "complete", self._value
            if stop_event.is_set():
                # Preserve a second failure completed concurrently with the first.
                if self._finished.wait(0.01):
                    continue
                return "stopped", None
            remaining_ns = deadline_ns - monotonic_ns()
            if remaining_ns <= 0:
                return "timeout", None
            self._finished.wait(min(remaining_ns / 1_000_000_000, 0.01))

    def abandon(self) -> None:
        self._abandoned.set()
        self._handle_late_result()

    def _handle_late_result(self) -> None:
        if (
            self._finished.is_set()
            and self._abandoned.is_set()
            and self._on_late_result is not None
            and not self._late_result_handled.is_set()
        ):
            self._late_result_handled.set()
            self._on_late_result()


class CaptureWorker:
    """Read one source and apply the configured fail-fast recovery policy."""

    def __init__(
        self,
        camera: CameraName,
        source: CameraSource,
        output: LatestFrameQueue[CameraFrameSet],
        *,
        frame_timeout_ms: int,
        max_consecutive_failures: int,
        reconnect_timeout_ms: int,
        initial_last_success_ns: int,
        monotonic_ns: Callable[[], int],
        generation: Callable[[], int],
        on_reconnecting: Callable[[CameraName, str], bool],
        on_reconnected: Callable[[CameraName, CameraFrameSet], None],
        close_source: Callable[[CameraName], None],
        mark_source_opening: Callable[[CameraName], None],
        on_fault: Callable[[CameraFault], None],
    ) -> None:
        self.camera = camera
        self.source = source
        self.output = output
        self.frame_timeout_ms = frame_timeout_ms
        self.max_consecutive_failures = max_consecutive_failures
        self.reconnect_timeout_ms = reconnect_timeout_ms
        self.last_success_ns = initial_last_success_ns
        self.monotonic_ns = monotonic_ns
        self.generation = generation
        self.on_reconnecting = on_reconnecting
        self.on_reconnected = on_reconnected
        self.close_source = close_source
        self.mark_source_opening = mark_source_opening
        self.on_fault = on_fault

    def run(self, stop_event: Event) -> None:
        consecutive_failures = 0
        pending: _BoundedCall[CameraFrameSet] | None = None
        pending_generation: int | None = None
        next_failure_deadline_ns: int | None = None
        while not stop_event.is_set():
            if pending is None:
                pending_generation = self.generation()
                pending = _BoundedCall(
                    lambda: self.source.read(self.frame_timeout_ms),
                    name=f"camera-read-{self.camera.value}",
                )
                next_failure_deadline_ns = (
                    self.monotonic_ns() + self.frame_timeout_ms * 1_000_000
                )
            assert next_failure_deadline_ns is not None
            state, outcome = pending.wait(
                self._remaining_ms(next_failure_deadline_ns),
                stop_event,
                self.monotonic_ns,
            )
            if state == "stopped":
                pending.abandon()
                return
            if state == "timeout":
                consecutive_failures += 1
                terminal_timeout = (
                    consecutive_failures >= self.max_consecutive_failures
                )
                if terminal_timeout:
                    probe_timeout_ms = max(1, self.frame_timeout_ms // 2)
                else:
                    next_failure_deadline_ns += (
                        self.frame_timeout_ms * 1_000_000
                    )
                    probe_timeout_ms = self._remaining_ms(
                        next_failure_deadline_ns
                    )
                if probe_timeout_ms <= 0:
                    continue
                probe = _BoundedCall(
                    self.source.is_present,
                    name=f"camera-presence-{self.camera.value}",
                )
                probe_state, probe_outcome = probe.wait(
                    probe_timeout_ms,
                    stop_event,
                    self.monotonic_ns,
                )
                if probe_state == "stopped":
                    probe.abandon()
                    pending.abandon()
                    return
                if probe_state == "timeout":
                    probe.abandon()
                    present = False
                    absence_cause = (
                        f"presence probe exceeded {probe_timeout_ms} ms "
                        "after frame timeout"
                    )
                elif isinstance(probe_outcome, bool):
                    present = probe_outcome
                    absence_cause = (
                        "device disappeared while waiting for a frame"
                    )
                else:
                    assert isinstance(probe_outcome, Exception)
                    present = False
                    absence_cause = (
                        "presence probe failed after frame timeout: "
                        f"{type(probe_outcome).__name__}: {probe_outcome}"
                    )
                if not present:
                    pending.abandon()
                    interrupted_read = pending
                    pending = None
                    if not self._recover(
                        absence_cause,
                        stop_event,
                        interrupted_read=interrupted_read,
                    ):
                        return
                    consecutive_failures = 0
                    continue
                if terminal_timeout:
                    pending.abandon()
                    self._fault(
                        "frame_timeout",
                        f"no complete frame for {consecutive_failures} consecutive "
                        f"{self.frame_timeout_ms} ms intervals",
                        consecutive_failures=consecutive_failures,
                    )
                    return
                # Keep waiting on the same call; never read one source concurrently.
                continue

            completed_generation = pending_generation
            pending = None
            pending_generation = None
            next_failure_deadline_ns = None
            if isinstance(outcome, CameraFrameSet):
                if completed_generation != self.generation():
                    continue
                consecutive_failures = 0
                self.last_success_ns = outcome.host_monotonic_ns
                try:
                    self.output.put(outcome)
                except RuntimeError:
                    if stop_event.is_set():
                        return
                    raise
                continue
            if isinstance(outcome, FrameTimeout):
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    self._fault(
                        "frame_timeout",
                        str(outcome),
                        consecutive_failures=consecutive_failures,
                    )
                    return
                continue
            if isinstance(outcome, DeviceDisconnected):
                if not self._recover(str(outcome), stop_event):
                    return
                consecutive_failures = 0
                continue
            if isinstance(outcome, FatalFrameError):
                self._fault("fatal_frame_error", str(outcome))
                return
            assert isinstance(outcome, Exception)
            self._fault("unexpected_capture_error", str(outcome))
            return

    def _recover(
        self,
        cause: str,
        stop_event: Event,
        *,
        interrupted_read: _BoundedCall[CameraFrameSet] | None = None,
    ) -> bool:
        if not self.on_reconnecting(self.camera, cause):
            return False
        deadline_ns = (
            self.monotonic_ns() + self.reconnect_timeout_ms * 1_000_000
        )
        close_call = _BoundedCall(
            lambda: self.close_source(self.camera),
            name=f"camera-close-{self.camera.value}",
        )
        state, outcome = close_call.wait(
            self._remaining_ms(deadline_ns), stop_event, self.monotonic_ns
        )
        if state == "stopped":
            close_call.abandon()
            return False
        if state == "timeout":
            close_call.abandon()
            self._fault(
                "reconnect_timeout",
                "closing the disconnected camera exceeded the reconnect deadline",
                reconnect_attempts=1,
            )
            return False
        if isinstance(outcome, Exception):
            self._fault(
                "reconnect_failed",
                f"failed to close disconnected camera: {outcome}",
                reconnect_attempts=1,
            )
            return False
        if interrupted_read is not None:
            state, _outcome = interrupted_read.wait(
                self._remaining_ms(deadline_ns),
                stop_event,
                self.monotonic_ns,
            )
            if state == "stopped":
                return False
            if state == "timeout":
                self._fault(
                    "reconnect_timeout",
                    "the interrupted camera read did not stop before the reconnect deadline",
                    reconnect_attempts=1,
                )
                return False
        self.mark_source_opening(self.camera)
        remaining_ms = self._remaining_ms(deadline_ns)
        if remaining_ms <= 0:
            self._fault(
                "reconnect_timeout",
                "no reconnect deadline remained after closing the camera",
                reconnect_attempts=1,
            )
            return False
        call = _BoundedCall(
            lambda: self.source.open(remaining_ms),
            name=f"camera-reconnect-{self.camera.value}",
            on_late_result=self.source.close,
        )
        state, outcome = call.wait(
            remaining_ms, stop_event, self.monotonic_ns
        )
        if state == "stopped":
            call.abandon()
            return False
        if state == "timeout":
            call.abandon()
            self._fault(
                "reconnect_timeout",
                f"reopen plus first frame exceeded {self.reconnect_timeout_ms} ms",
                reconnect_attempts=1,
            )
            return False
        if isinstance(outcome, CameraFrameSet):
            self.last_success_ns = outcome.host_monotonic_ns
            self.on_reconnected(self.camera, outcome)
            return True
        assert isinstance(outcome, Exception)
        self._fault(
            "reconnect_failed",
            f"{cause}; reopen failed: {outcome}",
            reconnect_attempts=1,
        )
        return False

    def _remaining_ms(self, deadline_ns: int) -> int:
        remaining_ns = deadline_ns - self.monotonic_ns()
        if remaining_ns <= 0:
            return 0
        return (remaining_ns + 999_999) // 1_000_000

    def _fault(
        self,
        category: str,
        cause: str,
        *,
        consecutive_failures: int = 0,
        reconnect_attempts: int = 0,
    ) -> None:
        self.on_fault(
            CameraFault(
                camera=self.camera,
                identity=self.source.identity,
                category=category,
                cause=cause,
                consecutive_failures=consecutive_failures,
                reconnect_attempts=reconnect_attempts,
                last_success_monotonic_ns=self.last_success_ns,
            )
        )


class CameraBridgeService:
    def __init__(
        self,
        config: CameraBridgeConfig,
        sources: Mapping[CameraName, CameraSource],
        publisher: CameraPublisher,
        *,
        uuid_factory: Callable[[], UUID] = uuid4,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        unix_ns: Callable[[], int] = time.time_ns,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self.sources = dict(sources)
        self.publisher = publisher
        self._uuid_factory = uuid_factory
        self._monotonic_ns = monotonic_ns
        self._unix_ns = unix_ns
        self._on_ready = on_ready or (lambda: None)
        self._queues = {
            camera: LatestFrameQueue[CameraFrameSet]() for camera in CameraName
        }
        self._stop_event = Event()
        self._publish_gate = Event()
        self._status_events: Queue[CameraStatus] = Queue()
        self._worker_threads: list[Thread] = []
        self._publisher_thread: Thread | None = None
        self._state_lock = Lock()
        self._session_id = self._uuid_factory()
        self._faults: list[CameraFault] = []
        self._camera_states = {camera: "online" for camera in CameraName}
        self._source_open = {camera: False for camera in CameraName}
        self._generation = 0
        self._started = False
        self._stopped = False

    @property
    def session_id(self) -> UUID:
        with self._state_lock:
            return self._session_id

    @property
    def faults(self) -> tuple[CameraFault, ...]:
        with self._state_lock:
            return tuple(self._faults)

    @property
    def worker_threads_alive(self) -> int:
        return sum(thread.is_alive() for thread in self._worker_threads)

    def start(self, shutdown_event: Event | None = None) -> None:
        if self._started:
            return
        shutdown = shutdown_event or Event()
        primed: list[tuple[CameraName, CameraFrameSet]] = []
        attempted: list[CameraName] = []
        startup_reconnect_attempts = 0
        try:
            for camera in CameraName:
                source = self.sources[camera]
                attempted.append(camera)
                startup_reconnect_attempts = 0
                frameset = self._bounded_startup_open(camera, shutdown)
                if isinstance(frameset, FrameTimeout):
                    startup_reconnect_attempts = 1
                    self._bounded_close_direct(camera)
                    LOGGER.warning(
                        "event=camera_startup_retry camera=%s identity=%s "
                        "attempt=1 cause=%r",
                        camera.value,
                        source.identity,
                        str(frameset),
                    )
                    frameset = self._bounded_startup_open(camera, shutdown)
                    if isinstance(frameset, FrameTimeout):
                        raise frameset
                self._set_source_open(camera, True)
                primed.append((camera, frameset))
            primed = self._refresh_startup_frames(shutdown)
            try:
                self.publisher.open()
            except Exception as exc:
                raise CameraFault(
                    camera=None,
                    identity=self.config.bind,
                    category="publisher_startup_error",
                    cause=f"{type(exc).__name__}: {exc}",
                ) from exc
        except Exception as exc:
            self._close_all_sources_bounded(attempted, force=True)
            if isinstance(exc, _StartupCancelled):
                raise
            if isinstance(exc, CameraFault):
                raise
            failed_camera = camera if "camera" in locals() else None
            identity = (
                self.sources[failed_camera].identity
                if failed_camera is not None
                else self.config.bind
            )
            raise CameraFault(
                camera=failed_camera,
                identity=identity,
                category="startup_failure",
                cause=str(exc),
                reconnect_attempts=startup_reconnect_attempts,
            ) from exc
        for camera, frameset in primed:
            self._queues[camera].put(frameset)
        self._started = True
        self._publisher_thread = Thread(
            target=self._run_publisher,
            name="camera-publisher",
            daemon=True,
        )
        self._publisher_thread.start()
        recovery = self.config.recovery
        for camera, first_frameset in primed:
            worker = CaptureWorker(
                camera,
                self.sources[camera],
                self._queues[camera],
                frame_timeout_ms=recovery.frame_timeout_ms,
                max_consecutive_failures=recovery.max_consecutive_frame_failures,
                reconnect_timeout_ms=recovery.reconnect_timeout_ms,
                initial_last_success_ns=first_frameset.host_monotonic_ns,
                monotonic_ns=self._monotonic_ns,
                generation=self._current_generation,
                on_reconnecting=self._begin_reconnect,
                on_reconnected=self._finish_reconnect,
                close_source=self._close_source,
                mark_source_opening=lambda name: self._set_source_open(name, True),
                on_fault=self._record_fault,
            )
            thread = Thread(
                target=worker.run,
                args=(self._stop_event,),
                name=f"camera-capture-{camera.value}",
                daemon=True,
            )
            self._worker_threads.append(thread)
            thread.start()
        self._publish_gate.set()
        try:
            self._on_ready()
        except Exception as exc:
            raise CameraFault(
                camera=None,
                identity=self.config.bind,
                category="readiness_error",
                cause=f"{type(exc).__name__}: {exc}",
            ) from exc

    def run(self, shutdown_event: Event) -> int:
        try:
            self.start(shutdown_event)
        except _StartupCancelled:
            self.stop()
            return 0
        except CameraFault as fault:
            self._record_fault(fault)
            self.stop()
            return 1
        while not shutdown_event.is_set() and not self._stop_event.wait(0.01):
            pass
        faulted = bool(self.faults)
        self.stop()
        return 1 if faulted else 0

    def stop(self) -> None:
        with self._state_lock:
            if self._stopped:
                return
            self._stopped = True
        self._publish_gate.clear()
        self._stop_event.set()
        deadline = time.monotonic() + 1.0
        for thread in self._worker_threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._publisher_thread is not None:
            self._publisher_thread.join(
                timeout=max(0.0, deadline - time.monotonic())
            )
        for queue in self._queues.values():
            queue.close()
        self._close_all_sources_bounded(CameraName)
        close_publisher = _BoundedCall(
            self.publisher.close, name="camera-publisher-close"
        )
        state, outcome = close_publisher.wait(
            self.config.recovery.reconnect_timeout_ms,
            Event(),
            self._monotonic_ns,
        )
        if state == "timeout":
            close_publisher.abandon()
            LOGGER.warning("event=camera_publisher_close_timeout")
        elif isinstance(outcome, Exception):
            LOGGER.warning("event=camera_publisher_close_failed cause=%r", outcome)
        LOGGER.info("event=camera_bridge_stopped faulted=%s", bool(self.faults))

    def _bounded_startup_open(
        self, camera: CameraName, shutdown_event: Event
    ) -> CameraFrameSet | FrameTimeout:
        timeout_ms = self.config.recovery.startup_timeout_ms
        source = self.sources[camera]
        call = _BoundedCall(
            lambda: source.open(timeout_ms),
            name=f"camera-startup-{camera.value}",
            on_late_result=source.close,
        )
        state, outcome = call.wait(timeout_ms, shutdown_event, self._monotonic_ns)
        if state == "stopped":
            call.abandon()
            raise _StartupCancelled(f"startup cancelled while opening {camera.value}")
        if state == "timeout":
            call.abandon()
            raise FrameTimeout(
                f"{camera.value} startup open exceeded {timeout_ms} ms"
            )
        if isinstance(outcome, FrameTimeout):
            return outcome
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, CameraFrameSet)
        return outcome

    def _refresh_startup_frames(
        self, shutdown_event: Event
    ) -> list[tuple[CameraName, CameraFrameSet]]:
        timeout_ms = self.config.recovery.startup_timeout_ms
        deadline_ns = self._monotonic_ns() + timeout_ms * 1_000_000
        calls = {
            camera: _BoundedCall(
                lambda source=self.sources[camera]: source.read_startup(timeout_ms),
                name=f"camera-startup-refresh-{camera.value}",
            )
            for camera in CameraName
        }
        refreshed: list[tuple[CameraName, CameraFrameSet]] = []
        for camera in CameraName:
            remaining_ns = deadline_ns - self._monotonic_ns()
            remaining_ms = max(0, (remaining_ns + 999_999) // 1_000_000)
            state, outcome = calls[camera].wait(
                remaining_ms, shutdown_event, self._monotonic_ns
            )
            if state == "stopped":
                for call in calls.values():
                    call.abandon()
                raise _StartupCancelled("startup cancelled during fresh-frame barrier")
            if state == "timeout":
                for call in calls.values():
                    call.abandon()
                raise FrameTimeout(
                    f"{camera.value} fresh startup frame exceeded {timeout_ms} ms"
                )
            if isinstance(outcome, Exception):
                raise outcome
            assert isinstance(outcome, CameraFrameSet)
            refreshed.append((camera, outcome))
        return refreshed

    def _bounded_close_direct(self, camera: CameraName) -> None:
        call = _BoundedCall(
            self.sources[camera].close,
            name=f"camera-cleanup-{camera.value}",
        )
        state, outcome = call.wait(
            self.config.recovery.reconnect_timeout_ms,
            Event(),
            self._monotonic_ns,
        )
        if state == "timeout":
            call.abandon()
            LOGGER.warning(
                "event=camera_close_timeout camera=%s identity=%s",
                camera.value,
                self.sources[camera].identity,
            )
        elif isinstance(outcome, Exception):
            LOGGER.warning(
                "event=camera_close_failed camera=%s identity=%s cause=%r",
                camera.value,
                self.sources[camera].identity,
                outcome,
            )

    def _close_all_sources_bounded(self, cameras, *, force: bool = False) -> None:
        calls: dict[CameraName, _BoundedCall[None]] = {}
        for camera in cameras:
            with self._state_lock:
                should_close = self._source_open[camera]
                self._source_open[camera] = False
            if should_close or force:
                calls[camera] = _BoundedCall(
                    self.sources[camera].close,
                    name=f"camera-cleanup-{camera.value}",
                )
        deadline_ns = (
            self._monotonic_ns()
            + self.config.recovery.reconnect_timeout_ms * 1_000_000
        )
        for camera, call in calls.items():
            remaining_ns = deadline_ns - self._monotonic_ns()
            remaining_ms = max(0, (remaining_ns + 999_999) // 1_000_000)
            state, outcome = call.wait(remaining_ms, Event(), self._monotonic_ns)
            if state == "timeout":
                call.abandon()
                LOGGER.warning(
                    "event=camera_close_timeout camera=%s identity=%s",
                    camera.value,
                    self.sources[camera].identity,
                )
            elif isinstance(outcome, Exception):
                LOGGER.warning(
                    "event=camera_close_failed camera=%s identity=%s cause=%r",
                    camera.value,
                    self.sources[camera].identity,
                    outcome,
                )

    def _begin_reconnect(self, camera: CameraName, cause: str) -> bool:
        with self._state_lock:
            if self._stopped:
                return False
            if all(state != "reconnecting" for state in self._camera_states.values()):
                self._generation += 1
            self._camera_states[camera] = "reconnecting"
            self._publish_gate.clear()
            for queue in self._queues.values():
                queue.clear()
        self._queue_status(camera, "reconnecting", cause=cause)
        LOGGER.warning(
            "event=camera_reconnecting camera=%s identity=%s attempt=1 cause=%r",
            camera.value,
            self.sources[camera].identity,
            cause,
        )
        return not self._stop_event.is_set()

    def _finish_reconnect(
        self, camera: CameraName, first_frameset: CameraFrameSet
    ) -> None:
        with self._state_lock:
            if self._stopped or self._stop_event.is_set():
                return
            self._camera_states[camera] = "online"
            self._queues[camera].put(first_frameset)
            all_online = all(
                state == "online" for state in self._camera_states.values()
            )
            if all_online:
                self._session_id = self._uuid_factory()
                self._publish_gate.set()
            session_id = self._session_id
        self._queue_status(camera, "online")
        LOGGER.info(
            "event=camera_reconnected camera=%s identity=%s session_id=%s attempt=1",
            camera.value,
            self.sources[camera].identity,
            session_id,
        )

    def _record_fault(self, fault: CameraFault) -> None:
        with self._state_lock:
            self._faults.append(fault)
            if fault.camera is not None:
                self._camera_states[fault.camera] = "fault"
        camera_label = fault.camera.value if fault.camera is not None else "service"
        LOGGER.error(
            "event=camera_fault camera=%s identity=%s category=%s "
            "last_success_monotonic_ns=%s consecutive_failures=%s "
            "reconnect_attempts=%s cause=%r "
            "action='inspect camera USB connection, power, and driver; manual intervention required'",
            camera_label,
            fault.identity,
            fault.category,
            fault.last_success_monotonic_ns,
            fault.consecutive_failures,
            fault.reconnect_attempts,
            fault.cause,
        )
        if fault.camera is not None:
            self._queue_status(
                fault.camera,
                "fault",
                category=fault.category,
                cause=fault.cause,
            )
        self._publish_gate.clear()
        for queue in self._queues.values():
            queue.clear()
        self._stop_event.set()

    def _run_publisher(self) -> None:
        try:
            self._publisher_loop()
        except Exception as exc:
            if not self._stop_event.is_set():
                self._record_fault(
                    CameraFault(
                        camera=None,
                        identity=self.config.bind,
                        category="publisher_error",
                        cause=f"{type(exc).__name__}: {exc}",
                    )
                )

    def _publisher_loop(self) -> None:
        last_session: UUID | None = None
        last_heartbeat_ns = 0
        while not self._stop_event.is_set():
            self._publish_pending_status()
            now_ns = self._monotonic_ns()
            if now_ns - last_heartbeat_ns >= 1_000_000_000:
                with self._state_lock:
                    session_id = self._session_id
                    states = dict(self._camera_states)
                for camera in CameraName:
                    self.publisher.publish_status(
                        CameraStatus(
                            session_id=session_id,
                            camera=camera,
                            state=states[camera],
                            host_monotonic_ns=now_ns,
                            host_unix_ns=self._unix_ns(),
                            identity=self.sources[camera].identity,
                        )
                    )
                last_heartbeat_ns = now_ns
            published = False
            with self._state_lock:
                session_id = self._session_id
                if session_id != last_session:
                    self.publisher.reset_sequences()
                    last_session = session_id
                if self._publish_gate.is_set():
                    for camera in CameraName:
                        if not self._publish_gate.is_set():
                            break
                        frameset = self._queues[camera].pop_latest(timeout_s=0)
                        if frameset is not None:
                            self.publisher.publish_frameset(session_id, frameset)
                            published = True
            if not published:
                self._stop_event.wait(0.001)
        self._publish_pending_status()

    def _queue_status(
        self,
        camera: CameraName,
        state: Literal["online", "reconnecting", "fault"],
        *,
        category: str | None = None,
        cause: str | None = None,
    ) -> None:
        self._status_events.put(
            CameraStatus(
                session_id=self.session_id,
                camera=camera,
                state=state,
                host_monotonic_ns=self._monotonic_ns(),
                host_unix_ns=self._unix_ns(),
                identity=self.sources[camera].identity,
                error_category=category,
                error=cause,
            )
        )

    def _publish_pending_status(self) -> None:
        while True:
            try:
                status = self._status_events.get_nowait()
            except Empty:
                return
            self.publisher.publish_status(status)

    def _set_source_open(self, camera: CameraName, value: bool) -> None:
        with self._state_lock:
            self._source_open[camera] = value

    def _current_generation(self) -> int:
        with self._state_lock:
            return self._generation

    def _close_source(self, camera: CameraName) -> None:
        with self._state_lock:
            should_close = self._source_open[camera]
            self._source_open[camera] = False
        if should_close:
            self.sources[camera].close()
