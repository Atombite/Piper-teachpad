# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from threading import Event, Lock, Thread
import time

import pytest

from piper_control.config import ArmSide, MujocoConfig, ServiceConfig
from piper_control.lifecycle import ShutdownResult
from piper_control.drivers import PiperArmDriver, SimulatedArmDriver
from piper_control.main import (
    ArmRuntime,
    RuntimeDependencies,
    _make_driver,
    build_parser,
    run_service,
)
from piper_control.types import ArmStateSnapshot, ServiceState
from test_config import arm_config


def _config(*, left_timeout: float = 0.2, right_timeout: float = 0.2) -> ServiceConfig:
    left = arm_config("left_arm", 8765, "can1")
    right = arm_config("right_arm", 9876, "can0")
    left["shutdown_home_timeout_s"] = left_timeout
    right["shutdown_home_timeout_s"] = right_timeout
    return ServiceConfig.model_validate({"arms": {"left": left, "right": right}})


def _mujoco_config(*, viewer: bool = True) -> ServiceConfig:
    config = _config()
    mujoco = MujocoConfig.model_validate(
        {
            "model_path": "/models/piper.xml",
            "timestep_s": 0.002,
            "realtime": False,
            "viewer": viewer,
            "gripper_model_max_m": 0.07,
            "bases": {
                "left": {"position_m": [-0.2, 0.0, 0.0], "rpy_deg": [0, 0, 0]},
                "right": {"position_m": [0.2, 0.0, 0.0], "rpy_deg": [0, 0, 0]},
            },
        }
    )
    return config.model_copy(update={"mujoco": mujoco})


@pytest.mark.parametrize("side", [ArmSide.LEFT, ArmSide.RIGHT])
@pytest.mark.parametrize("simulate", [False, True])
def test_make_driver_selects_whole_arm_backend(side: ArmSide, simulate: bool) -> None:
    config = _config().arms[side]
    driver = _make_driver(side, config, simulate)
    if simulate:
        assert isinstance(driver, SimulatedArmDriver)
        before = driver.read_feedback().received_at_s
        assert driver.read_feedback().received_at_s > before
        driver.command_gripper(0.04)
        assert driver.read_feedback().gripper_position == pytest.approx(0.04)
    else:
        assert isinstance(driver, PiperArmDriver)


class _LockContext:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def __enter__(self) -> "_LockContext":
        self._events.append("lock")
        return self

    def __exit__(self, *_args: object) -> None:
        self._events.append("unlock")


class _Controller:
    def __init__(
        self,
        side: ArmSide,
        *,
        fail: Event | None = None,
        stop_on_shutdown: bool = True,
        shutdown_raises: bool = False,
    ) -> None:
        self.side = side
        self.fail = fail
        self.stop_on_shutdown = stop_on_shutdown
        self.shutdown_raises = shutdown_raises
        self.stop = Event()
        self.running = Event()
        self.shutdown_requested = Event()
        self.requested_at: float | None = None
        self.finished_at: float | None = None
        self.timeout_safety: list[str] = []
        self.emergency_reasons: list[str] = []
        self._snapshot = ArmStateSnapshot(side=side, state=ServiceState.RUNNING)

    def run(self) -> None:
        self.running.set()
        while not self.stop.wait(0.001):
            if self.fail is not None and self.fail.is_set():
                raise RuntimeError(f"{self.side.value} control fault")

    def snapshot(self) -> ArmStateSnapshot:
        return self._snapshot

    def request_shutdown_async(self) -> None:
        if self.shutdown_raises:
            raise RuntimeError("shutdown request failed")
        self.requested_at = time.monotonic()
        self.shutdown_requested.set()
        if self.stop_on_shutdown:
            if self._snapshot.state is not ServiceState.EMERGENCY_STOP:
                self._snapshot = replace(
                    self._snapshot, state=ServiceState.SHUTTING_DOWN
                )
            self.stop.set()

    def request_shutdown(self) -> None:
        self.request_shutdown_async()

    def emergency_stop(self, reason: str) -> None:
        self.emergency_reasons.append(reason)
        self._snapshot = replace(
            self._snapshot,
            state=ServiceState.EMERGENCY_STOP,
            emergency_stopped=True,
            fault_reason=reason,
            hold=True,
        )

    def publish_runtime_fault(self, reason: str) -> None:
        self.request_timeout_safety_async(reason)

    def shutdown_result(self) -> ShutdownResult | None:
        self.finished_at = time.monotonic()
        if self.stop.is_set():
            return ShutdownResult(homed=True)
        return None

    def request_timeout_safety_async(self, reason: str) -> None:
        self.timeout_safety.append(reason)
        self.emergency_stop(reason)


class _Server:
    def __init__(self, side: ArmSide, *, shutdown_event: Event | None = None) -> None:
        self.side = side
        self.started = False
        self.should_exit = False
        self.running = Event()
        self._shutdown_event = shutdown_event

    def run(self) -> None:
        self.started = True
        self.running.set()
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        while not self.should_exit:
            time.sleep(0.001)


class _World:
    def __init__(
        self,
        events: list[str],
        *,
        start_raises: bool = False,
        request_raises: bool = False,
        join_result: bool = True,
        join_raises: bool = False,
    ) -> None:
        self.events = events
        self.start_raises = start_raises
        self.request_raises = request_raises
        self.join_result = join_result
        self.join_raises = join_raises
        self.failure: Exception | None = None
        self.viewer_closed = False
        self.running = False

    def start(self) -> None:
        self.events.append("world_start")
        if self.start_raises:
            raise RuntimeError("world startup failed")
        self.running = True

    def request_shutdown(self) -> None:
        self.events.append("world_shutdown")
        self.running = False
        if self.request_raises:
            raise RuntimeError("world shutdown request failed")

    def join(self, timeout_s: float) -> bool:
        assert timeout_s >= 0
        self.events.append("world_join")
        if self.join_raises:
            raise RuntimeError("world join failed")
        return self.join_result


class _RaisingExitServer(_Server):
    def __init__(self, side: ArmSide, events: list[str]) -> None:
        self._should_exit = False
        self._events = events
        super().__init__(side)

    @property
    def should_exit(self) -> bool:
        return self._should_exit

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._should_exit = value
        if value:
            self._events.append(f"{self.side.value}_web_stop_attempted")
            raise RuntimeError("server exit mutation failed")


def test_shutdown_result_log_exposes_controller_fault_reason_and_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="piper_control.main")
    controller = _Controller(ArmSide.RIGHT)
    server = _Server(ArmSide.RIGHT)
    runtime = ArmRuntime(
        ArmSide.RIGHT,
        _config().arms[ArmSide.RIGHT],
        controller,
        server,
    )
    runtime.start()
    assert controller.running.wait(0.5)
    assert server.running.wait(0.5)
    controller._snapshot = ArmStateSnapshot(
        side=ArmSide.RIGHT,
        state=ServiceState.EMERGENCY_STOP,
        emergency_stopped=True,
        fault_reason="right joint 3 feedback crossed its upper limit",
        hold=True,
    )
    controller.shutdown_result = lambda: ShutdownResult(  # type: ignore[method-assign]
        homed=False,
        faulted=True,
        errors=("right emergency stop latched",),
    )

    runtime.begin_shutdown()
    assert runtime.finish_shutdown() is False

    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "shutdown_result"
    ]
    assert len(records) == 1
    assert records[0].fault_reason == "right joint 3 feedback crossed its upper limit"
    assert records[0].shutdown_errors == "right emergency stop latched"


def test_parser_rejects_removed_control_hz_override() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--config", "/etc/piper.yaml", "--control-hz", "50"])


@pytest.mark.parametrize(
    "modes",
    [
        ("--simulate", "--mujoco"),
        ("--simulate", "--mujoco-headless"),
        ("--mujoco", "--mujoco-headless"),
    ],
)
def test_parser_rejects_multiple_runtime_modes(modes: tuple[str, str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--config", "/etc/piper.yaml", *modes])


def test_parser_keeps_hardware_as_default_mode() -> None:
    args = build_parser().parse_args(["--config", "/etc/piper.yaml"])
    assert (args.simulate, args.mujoco, args.mujoco_headless) == (False, False, False)


def test_parser_accepts_explicit_arm_side() -> None:
    args = build_parser().parse_args(
        ["--config", "/etc/piper.yaml", "--side", "left"]
    )
    assert args.side is ArmSide.LEFT


def test_mujoco_constructs_one_shared_world_and_one_driver_per_side() -> None:
    config = _mujoco_config()
    shutdown = Event()
    events: list[str] = []
    model = object()
    worlds: list[tuple[object, MujocoConfig, Event]] = []
    drivers: list[tuple[ArmSide, object, object, float]] = []

    def make_world(built_model, mujoco_config, stop_event):  # type: ignore[no-untyped-def]
        worlds.append((built_model, mujoco_config, stop_event))
        return _World(events)

    def make_mujoco_driver(  # type: ignore[no-untyped-def]
        side, arm, world, *, gripper_model_max_m
    ):
        driver = object()
        drivers.append((side, arm, world, gripper_model_max_m))
        return driver

    server_count = 0

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        nonlocal server_count
        server_count += 1
        return _Server(app, shutdown_event=shutdown if server_count == 2 else None)

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext(events),
        driver_factory=lambda *_args: pytest.fail("hardware driver constructed"),
        controller_factory=lambda side, _arm, _driver: _Controller(side),
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
        mujoco_model_factory=lambda mujoco, arms: (
            model
            if mujoco is config.mujoco and arms is config.arms
            else pytest.fail("wrong model factory inputs")
        ),
        mujoco_world_factory=make_world,
        mujoco_driver_factory=make_mujoco_driver,
    )

    assert run_service(
        config,
        simulate=False,
        mujoco=True,
        lock_path="unused",
        shutdown_event=shutdown,
        dependencies=deps,
    ) == 0
    assert len(worlds) == 1
    world = drivers[0][2]
    assert [(side, shared is world, maximum) for side, _arm, shared, maximum in drivers] == [
        (ArmSide.RIGHT, True, 0.07),
        (ArmSide.LEFT, True, 0.07),
    ]
    assert events == ["lock", "world_start", "world_shutdown", "world_join", "unlock"]


def test_mujoco_headless_overrides_viewer_without_mutating_config() -> None:
    config = _mujoco_config(viewer=True)
    shutdown = Event()
    received: list[MujocoConfig] = []
    server_count = 0

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        nonlocal server_count
        server_count += 1
        return _Server(app, shutdown_event=shutdown if server_count == 2 else None)

    def make_world(_model, mujoco_config, _event):  # type: ignore[no-untyped-def]
        received.append(mujoco_config)
        return _World([])

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext([]),
        controller_factory=lambda side, _arm, _driver: _Controller(side),
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
        mujoco_model_factory=lambda *_args: object(),
        mujoco_world_factory=make_world,
        mujoco_driver_factory=lambda *_args, **_kwargs: object(),
    )

    assert run_service(
        config,
        simulate=False,
        mujoco_headless=True,
        lock_path="unused",
        shutdown_event=shutdown,
        dependencies=deps,
    ) == 0
    assert received[0].viewer is False
    assert config.mujoco is not None and config.mujoco.viewer is True


def test_mujoco_requires_config_before_constructing_runtime_components() -> None:
    constructed: list[str] = []
    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext([]),
        driver_factory=lambda *_args: constructed.append("driver"),
        controller_factory=lambda *_args: constructed.append("controller"),
        app_factory=lambda *_args: constructed.append("app"),
        server_factory=lambda *_args: constructed.append("server"),
        mujoco_model_factory=lambda *_args: constructed.append("model"),
        mujoco_world_factory=lambda *_args: constructed.append("world"),
        mujoco_driver_factory=lambda *_args, **_kwargs: constructed.append("mujoco_driver"),
    )

    assert run_service(
        _config(), simulate=False, mujoco=True, lock_path="unused", dependencies=deps
    ) == 1
    assert constructed == []


def test_dual_config_constructs_independent_arm_runtimes_under_one_lock() -> None:
    config = _config()
    events: list[str] = []
    shutdown = Event()
    drivers: list[tuple[ArmSide, object, bool]] = []
    controllers: list[tuple[ArmSide, object, object]] = []
    apps: list[tuple[ArmSide, object, object]] = []
    servers: list[tuple[ArmSide, str, int]] = []

    def make_driver(side, arm, simulate):  # type: ignore[no-untyped-def]
        assert events == ["lock"]
        driver = object()
        drivers.append((side, arm, simulate))
        return driver

    def make_controller(side, arm, driver):  # type: ignore[no-untyped-def]
        controller = _Controller(side)
        controllers.append((side, arm, driver))
        return controller

    def make_app(controller, side, arm):  # type: ignore[no-untyped-def]
        apps.append((side, arm, controller))
        return side

    def make_server(app, host, port):  # type: ignore[no-untyped-def]
        side = app
        servers.append((side, host, port))
        return _Server(side, shutdown_event=shutdown if len(servers) == 2 else None)

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext(events),
        driver_factory=make_driver,
        controller_factory=make_controller,
        app_factory=make_app,
        server_factory=make_server,
    )

    assert run_service(config, simulate=True, lock_path="unused", shutdown_event=shutdown, dependencies=deps) == 0
    assert events == ["lock", "unlock"]
    assert [(side.value, arm.can_port, simulate) for side, arm, simulate in drivers] == [
        ("right", "can0", True),
        ("left", "can1", True),
    ]
    assert [side for side, _arm, _driver in controllers] == [ArmSide.RIGHT, ArmSide.LEFT]
    assert [side for side, _arm, _controller in apps] == [ArmSide.RIGHT, ArmSide.LEFT]
    assert servers == [
        (ArmSide.RIGHT, "127.0.0.1", 9876),
        (ArmSide.LEFT, "127.0.0.1", 8765),
    ]
    assert controllers[0][1] is config.arms[ArmSide.RIGHT]
    assert controllers[1][1] is config.arms[ArmSide.LEFT]


def test_side_selection_constructs_only_the_requested_runtime() -> None:
    config = _config()
    shutdown = Event()
    constructed: list[ArmSide] = []

    def make_controller(side, _arm, _driver):  # type: ignore[no-untyped-def]
        constructed.append(side)
        return _Controller(side)

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext([]),
        driver_factory=lambda _side, _arm, _simulate: object(),
        controller_factory=make_controller,
        app_factory=lambda _controller, side, _arm: side,
        server_factory=lambda side, _host, _port: _Server(
            side, shutdown_event=shutdown
        ),
    )

    assert run_service(
        config,
        simulate=True,
        side=ArmSide.LEFT,
        lock_path="unused",
        shutdown_event=shutdown,
        dependencies=deps,
    ) == 0
    assert constructed == [ArmSide.LEFT]


def test_missing_selected_side_fails_before_runtime_construction() -> None:
    config = _config().model_copy(
        update={"arms": {ArmSide.LEFT: _config().arms[ArmSide.LEFT]}}
    )
    constructed: list[str] = []
    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext([]),
        driver_factory=lambda *_args: constructed.append("driver"),
        controller_factory=lambda *_args: constructed.append("controller"),
        app_factory=lambda *_args: constructed.append("app"),
        server_factory=lambda *_args: constructed.append("server"),
    )

    assert run_service(
        config,
        simulate=True,
        side=ArmSide.RIGHT,
        lock_path="unused",
        dependencies=deps,
    ) == 1
    assert constructed == []


def test_left_worker_fault_does_not_stop_healthy_right_runtime() -> None:
    shutdown = Event()
    left_fault = Event()
    controllers: dict[ArmSide, _Controller] = {}
    servers: dict[ArmSide, _Server] = {}
    result: list[int] = []

    def make_controller(side, _arm, _driver):  # type: ignore[no-untyped-def]
        controller = _Controller(side, fail=left_fault if side is ArmSide.LEFT else None)
        controllers[side] = controller
        return controller

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        server = _Server(app)
        servers[app] = server
        return server

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext([]),
        driver_factory=lambda *_args: object(),
        controller_factory=make_controller,
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
    )
    runner = Thread(
        target=lambda: result.append(run_service(_config(), simulate=True, lock_path="unused", shutdown_event=shutdown, dependencies=deps)),
        daemon=True,
    )
    runner.start()
    deadline = time.monotonic() + 0.5
    while (len(controllers) < 2 or len(servers) < 2) and time.monotonic() < deadline:
        time.sleep(0.001)
    assert set(controllers) == {ArmSide.RIGHT, ArmSide.LEFT}
    assert set(servers) == {ArmSide.RIGHT, ArmSide.LEFT}
    assert all(controller.running.wait(0.5) for controller in controllers.values())
    assert all(server.running.wait(0.5) for server in servers.values())

    left_fault.set()
    assert controllers[ArmSide.LEFT].shutdown_requested.wait(0.5)
    deadline = time.monotonic() + 0.5
    while (
        controllers[ArmSide.LEFT].snapshot().state
        is not ServiceState.EMERGENCY_STOP
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert runner.is_alive()
    left_snapshot = controllers[ArmSide.LEFT].snapshot()
    assert left_snapshot.state is ServiceState.EMERGENCY_STOP
    assert left_snapshot.emergency_stopped is True
    assert left_snapshot.fault_reason == "control worker failed: left control fault"
    assert controllers[ArmSide.RIGHT].snapshot().state is ServiceState.RUNNING
    assert controllers[ArmSide.RIGHT].emergency_reasons == []
    assert controllers[ArmSide.RIGHT].running.is_set()
    assert not controllers[ArmSide.RIGHT].shutdown_requested.is_set()
    assert not servers[ArmSide.LEFT].should_exit
    assert servers[ArmSide.LEFT].running.is_set()
    assert not servers[ArmSide.RIGHT].should_exit

    shutdown.set()
    runner.join(1)
    assert not runner.is_alive()
    assert result == [1]


def test_later_factory_failure_cleans_all_constructed_runtimes_before_unlock(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("ERROR", logger="piper_control.main")
    events: list[str] = []
    controllers: dict[ArmSide, _Controller] = {}
    servers: dict[ArmSide, _Server] = {}

    def make_controller(side, _arm, _driver):  # type: ignore[no-untyped-def]
        controller = _Controller(side)
        controllers[side] = controller
        return controller

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        if app is ArmSide.LEFT:
            raise RuntimeError("left server construction failed")
        server = _Server(app)
        servers[app] = server
        return server

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext(events),
        driver_factory=lambda *_args: object(),
        controller_factory=make_controller,
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
    )

    result: int | None = None
    try:
        result = run_service(
            _config(), simulate=True, lock_path="unused", dependencies=deps
        )
    except RuntimeError:
        pass
    finally:
        for controller in controllers.values():
            controller.stop.set()
        for server in servers.values():
            server.should_exit = True

    assert result == 1
    assert events == ["lock", "unlock"]
    assert set(controllers) == {ArmSide.RIGHT, ArmSide.LEFT}
    assert not any(controller.running.is_set() for controller in controllers.values())
    assert all(
        controller.shutdown_requested.is_set() for controller in controllers.values()
    )
    assert servers[ArmSide.RIGHT].should_exit
    orchestration_log = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "orchestration_failure"
    )
    assert (
        orchestration_log.side,
        orchestration_log.can_port,
        orchestration_log.port,
    ) == ("left", "can1", 8765)


def test_partial_runtime_shutdown_failure_log_retains_arm_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("ERROR", logger="piper_control.main")

    def make_controller(side, _arm, _driver):  # type: ignore[no-untyped-def]
        return _Controller(side, shutdown_raises=side is ArmSide.LEFT)

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        if app is ArmSide.LEFT:
            raise RuntimeError("left server construction failed")
        return _Server(app)

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext([]),
        driver_factory=lambda *_args: object(),
        controller_factory=make_controller,
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
    )

    assert run_service(_config(), simulate=True, lock_path="unused", dependencies=deps) == 1
    partial_log = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "partial_runtime_shutdown_failure"
    )
    assert (partial_log.side, partial_log.can_port, partial_log.port) == (
        "left",
        "can1",
        8765,
    )


def test_simultaneous_runtime_faults_batch_shutdown_requests_before_finishing() -> None:
    fault = Event()
    controllers: dict[ArmSide, _Controller] = {}
    servers: dict[ArmSide, _Server] = {}

    def make_controller(side, _arm, _driver):  # type: ignore[no-untyped-def]
        controller = _Controller(side)
        controllers[side] = controller
        return controller

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        server = _Server(app)

        def run() -> None:
            server.started = True
            server.running.set()
            fault.wait(1)

        server.run = run  # type: ignore[method-assign]
        servers[app] = server
        return server

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext([]),
        driver_factory=lambda *_args: object(),
        controller_factory=make_controller,
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
    )
    result: list[int] = []
    runner = Thread(
        target=lambda: result.append(
            run_service(
                _config(),
                simulate=True,
                lock_path="unused",
                dependencies=deps,
            )
        ),
        daemon=True,
    )
    runner.start()
    deadline = time.monotonic() + 0.5
    while len(servers) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert all(server.running.wait(0.5) for server in servers.values())
    fault.set()
    runner.join(1)

    assert not runner.is_alive()
    requested = [controller.requested_at for controller in controllers.values()]
    finished = [controller.finished_at for controller in controllers.values()]
    assert all(timestamp is not None for timestamp in requested + finished)
    assert max(requested) <= min(finished)  # type: ignore[type-var]
    assert result == [1]


def test_shutdown_requests_all_arms_before_waiting_and_uses_per_arm_deadlines() -> None:
    shutdown = Event()
    controllers: dict[ArmSide, _Controller] = {}
    servers_ready = Event()
    server_count = 0
    count_lock = Lock()

    def make_controller(side, _arm, _driver):  # type: ignore[no-untyped-def]
        controller = _Controller(side, stop_on_shutdown=side is ArmSide.RIGHT)
        controllers[side] = controller
        return controller

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        nonlocal server_count
        server = _Server(app)
        original_run = server.run

        def run() -> None:
            nonlocal server_count
            with count_lock:
                server_count += 1
                if server_count == 2:
                    servers_ready.set()
            original_run()

        server.run = run  # type: ignore[method-assign]
        return server

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext([]),
        driver_factory=lambda *_args: object(),
        controller_factory=make_controller,
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
    )
    result: list[int] = []
    runner = Thread(
        target=lambda: result.append(run_service(_config(left_timeout=0.04, right_timeout=0.2), simulate=True, lock_path="unused", shutdown_event=shutdown, dependencies=deps)),
        daemon=True,
    )
    runner.start()
    assert servers_ready.wait(0.5)
    shutdown.set()
    runner.join(1)

    assert not runner.is_alive()
    assert all(controller.requested_at is not None for controller in controllers.values())
    assert controllers[ArmSide.RIGHT].stop.is_set()
    assert controllers[ArmSide.LEFT].timeout_safety == ["service shutdown deadline exceeded"]
    assert controllers[ArmSide.RIGHT].timeout_safety == []
    assert result == [1]


def test_world_failure_faults_both_arms_and_world_stops_after_arm_workers() -> None:
    shutdown = Event()
    events: list[str] = []
    controllers: dict[ArmSide, _Controller] = {}
    servers: dict[ArmSide, _Server] = {}
    world = _World(events)

    def make_controller(side, _arm, _driver):  # type: ignore[no-untyped-def]
        controller = _Controller(side)
        original_request = controller.request_shutdown_async
        original_result = controller.shutdown_result

        def request_shutdown() -> None:
            events.append(f"{side.value}_shutdown_requested")
            original_request()

        def shutdown_result():  # type: ignore[no-untyped-def]
            events.append(f"{side.value}_shutdown_finished")
            return original_result()

        controller.request_shutdown_async = request_shutdown  # type: ignore[method-assign]
        controller.shutdown_result = shutdown_result  # type: ignore[method-assign]
        controllers[side] = controller
        return controller

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        server = _Server(app)
        original_run = server.run

        def run() -> None:
            try:
                original_run()
            finally:
                events.append(f"{app.value}_web_closed")

        server.run = run  # type: ignore[method-assign]
        servers[app] = server
        return server

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext(events),
        controller_factory=make_controller,
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
        mujoco_model_factory=lambda *_args: object(),
        mujoco_world_factory=lambda *_args: world,
        mujoco_driver_factory=lambda *_args, **_kwargs: object(),
    )
    result: list[int] = []
    runner = Thread(
        target=lambda: result.append(
            run_service(
                _mujoco_config(),
                simulate=False,
                mujoco=True,
                lock_path="unused",
                shutdown_event=shutdown,
                dependencies=deps,
            )
        ),
        daemon=True,
    )
    runner.start()
    deadline = time.monotonic() + 0.5
    while len(servers) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert all(server.running.wait(0.5) for server in servers.values())

    world.failure = RuntimeError("physics exploded")
    shutdown.set()  # The real world sets the shared event when recording a failure.
    runner.join(1)

    assert not runner.is_alive()
    assert result == [1]
    assert all(
        controller.timeout_safety == ["MuJoCo world failed: physics exploded"]
        for controller in controllers.values()
    )
    request_indexes = [
        events.index(f"{side.value}_shutdown_requested") for side in ArmSide
    ]
    finish_indexes = [events.index(f"{side.value}_shutdown_finished") for side in ArmSide]
    web_indexes = [events.index(f"{side.value}_web_closed") for side in ArmSide]
    assert max(request_indexes) < min(finish_indexes)
    assert max(web_indexes) < events.index("world_shutdown") < events.index("world_join")
    assert events[-1] == "unlock"


def test_world_failure_during_arm_shutdown_faults_both_and_returns_nonzero() -> None:
    shutdown = Event()
    events: list[str] = []
    controllers: dict[ArmSide, _Controller] = {}
    servers: dict[ArmSide, _Server] = {}
    world = _World(events)

    def make_controller(side, _arm, _driver):  # type: ignore[no-untyped-def]
        controller = _Controller(side)
        original_request = controller.request_shutdown_async
        original_result = controller.shutdown_result

        def request_shutdown() -> None:
            events.append(f"{side.value}_shutdown_requested")
            original_request()

        def shutdown_result():  # type: ignore[no-untyped-def]
            result = original_result()
            events.append(f"{side.value}_shutdown_finished")
            if side is ArmSide.LEFT and world.failure is None:
                world.failure = RuntimeError("physics failed during shutdown")
                events.append("world_failed")
            return result

        controller.request_shutdown_async = request_shutdown  # type: ignore[method-assign]
        controller.shutdown_result = shutdown_result  # type: ignore[method-assign]
        controllers[side] = controller
        return controller

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        server = _Server(app)
        servers[app] = server
        return server

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext(events),
        controller_factory=make_controller,
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
        mujoco_model_factory=lambda *_args: object(),
        mujoco_world_factory=lambda *_args: world,
        mujoco_driver_factory=lambda *_args, **_kwargs: object(),
    )
    result: list[int] = []
    runner = Thread(
        target=lambda: result.append(
            run_service(
                _mujoco_config(),
                simulate=False,
                mujoco=True,
                lock_path="unused",
                shutdown_event=shutdown,
                dependencies=deps,
            )
        ),
        daemon=True,
    )
    runner.start()
    deadline = time.monotonic() + 0.5
    while len(servers) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert all(server.running.wait(0.5) for server in servers.values())

    shutdown.set()
    runner.join(1)

    assert not runner.is_alive()
    assert result == [1]
    assert all(
        controller.timeout_safety
        == ["MuJoCo world failed: physics failed during shutdown"]
        for controller in controllers.values()
    )
    request_indexes = [
        events.index(f"{side.value}_shutdown_requested") for side in ArmSide
    ]
    assert max(request_indexes) < events.index("world_failed")
    assert events.index("world_failed") < events.index("world_shutdown")
    assert events.index("world_shutdown") < events.index("world_join")


def test_world_startup_failure_cleans_world_under_lock_without_starting_web() -> None:
    events: list[str] = []
    world = _World(events, start_raises=True)
    server_calls: list[object] = []
    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext(events),
        server_factory=lambda *_args: server_calls.append(object()),
        mujoco_model_factory=lambda *_args: object(),
        mujoco_world_factory=lambda *_args: world,
        mujoco_driver_factory=lambda *_args, **_kwargs: pytest.fail(
            "driver constructed after world startup failed"
        ),
    )

    assert run_service(
        _mujoco_config(),
        simulate=False,
        mujoco=True,
        lock_path="unused",
        dependencies=deps,
    ) == 1
    assert server_calls == []
    assert events == ["lock", "world_start", "world_shutdown", "world_join", "unlock"]


def test_single_arm_fault_does_not_stop_shared_world_or_peer_runtime() -> None:
    shutdown = Event()
    left_fault = Event()
    events: list[str] = []
    controllers: dict[ArmSide, _Controller] = {}
    servers: dict[ArmSide, _Server] = {}
    world = _World(events)

    def make_controller(side, _arm, _driver):  # type: ignore[no-untyped-def]
        controller = _Controller(
            side, fail=left_fault if side is ArmSide.LEFT else None
        )
        controllers[side] = controller
        return controller

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        server = _Server(app)
        servers[app] = server
        return server

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext(events),
        controller_factory=make_controller,
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
        mujoco_model_factory=lambda *_args: object(),
        mujoco_world_factory=lambda *_args: world,
        mujoco_driver_factory=lambda *_args, **_kwargs: object(),
    )
    result: list[int] = []
    runner = Thread(
        target=lambda: result.append(
            run_service(
                _mujoco_config(),
                simulate=False,
                mujoco_headless=True,
                lock_path="unused",
                shutdown_event=shutdown,
                dependencies=deps,
            )
        ),
        daemon=True,
    )
    runner.start()
    deadline = time.monotonic() + 0.5
    while len(servers) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert all(server.running.wait(0.5) for server in servers.values())

    left_fault.set()
    assert controllers[ArmSide.LEFT].shutdown_requested.wait(0.5)
    assert runner.is_alive()
    assert world.running
    assert "world_shutdown" not in events
    assert controllers[ArmSide.RIGHT].running.is_set()
    assert not controllers[ArmSide.RIGHT].shutdown_requested.is_set()

    shutdown.set()
    runner.join(1)
    assert not runner.is_alive()
    assert result == [1]


def test_viewer_close_safely_stops_both_arms_without_runtime_fault() -> None:
    shutdown = Event()
    events: list[str] = []
    controllers: dict[ArmSide, _Controller] = {}
    servers: dict[ArmSide, _Server] = {}
    world = _World(events)

    def make_controller(side, _arm, _driver):  # type: ignore[no-untyped-def]
        controller = _Controller(side)
        controllers[side] = controller
        return controller

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        server = _Server(app)
        servers[app] = server
        return server

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext(events),
        controller_factory=make_controller,
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
        mujoco_model_factory=lambda *_args: object(),
        mujoco_world_factory=lambda *_args: world,
        mujoco_driver_factory=lambda *_args, **_kwargs: object(),
    )
    result: list[int] = []
    runner = Thread(
        target=lambda: result.append(
            run_service(
                _mujoco_config(),
                simulate=False,
                mujoco=True,
                lock_path="unused",
                shutdown_event=shutdown,
                dependencies=deps,
            )
        ),
        daemon=True,
    )
    runner.start()
    deadline = time.monotonic() + 0.5
    while len(servers) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert all(server.running.wait(0.5) for server in servers.values())

    world.viewer_closed = True
    shutdown.set()
    runner.join(1)

    assert not runner.is_alive()
    assert result == [0]
    assert all(controller.shutdown_requested.is_set() for controller in controllers.values())
    assert all(controller.timeout_safety == [] for controller in controllers.values())
    assert events[-3:] == ["world_shutdown", "world_join", "unlock"]


@pytest.mark.parametrize(
    ("world_options", "expected_event"),
    [
        ({"request_raises": True}, "mujoco_world_shutdown_failure"),
        ({"join_result": False}, "mujoco_world_shutdown_timeout"),
        ({"join_raises": True}, "mujoco_world_join_failure"),
    ],
)
def test_world_cleanup_failures_are_bounded_and_do_not_escape(
    world_options: dict[str, bool],
    expected_event: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("ERROR", logger="piper_control.main")
    shutdown = Event()
    events: list[str] = []
    world = _World(events, **world_options)
    server_count = 0

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        nonlocal server_count
        server_count += 1
        return _Server(app, shutdown_event=shutdown if server_count == 2 else None)

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext(events),
        controller_factory=lambda side, _arm, _driver: _Controller(side),
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
        mujoco_model_factory=lambda *_args: object(),
        mujoco_world_factory=lambda *_args: world,
        mujoco_driver_factory=lambda *_args, **_kwargs: object(),
    )

    assert run_service(
        _mujoco_config(),
        simulate=False,
        mujoco=True,
        lock_path="unused",
        shutdown_event=shutdown,
        dependencies=deps,
    ) == 1
    assert events.count("world_shutdown") == 1
    assert events.count("world_join") == 1
    assert events[-1] == "unlock"
    assert any(
        getattr(record, "event", None) == expected_event for record in caplog.records
    )


def test_snapshot_failure_during_first_arm_shutdown_does_not_skip_peer_or_world() -> None:
    shutdown = Event()
    events: list[str] = []
    controllers: dict[ArmSide, _Controller] = {}
    world = _World(events)
    server_count = 0

    def make_controller(side, _arm, _driver):  # type: ignore[no-untyped-def]
        controller = _Controller(side)
        original_snapshot = controller.snapshot

        def snapshot():  # type: ignore[no-untyped-def]
            if side is ArmSide.RIGHT and shutdown.is_set():
                events.append("right_snapshot_failed")
                raise RuntimeError("snapshot failed")
            return original_snapshot()

        controller.snapshot = snapshot  # type: ignore[method-assign]
        controllers[side] = controller
        return controller

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        nonlocal server_count
        server_count += 1
        return _Server(app, shutdown_event=shutdown if server_count == 2 else None)

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext(events),
        controller_factory=make_controller,
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
        mujoco_model_factory=lambda *_args: object(),
        mujoco_world_factory=lambda *_args: world,
        mujoco_driver_factory=lambda *_args, **_kwargs: object(),
    )

    assert run_service(
        _mujoco_config(),
        simulate=False,
        mujoco=True,
        lock_path="unused",
        shutdown_event=shutdown,
        dependencies=deps,
    ) == 1
    assert all(controller.shutdown_requested.is_set() for controller in controllers.values())
    assert events.index("right_snapshot_failed") < events.index("world_shutdown")
    assert events[-3:] == ["world_shutdown", "world_join", "unlock"]


def test_server_stop_failure_does_not_skip_peer_or_world_cleanup() -> None:
    shutdown = Event()
    events: list[str] = []
    controllers: dict[ArmSide, _Controller] = {}
    world = _World(events)
    server_count = 0

    def make_controller(side, _arm, _driver):  # type: ignore[no-untyped-def]
        controller = _Controller(side)
        controllers[side] = controller
        return controller

    def make_server(app, _host, _port):  # type: ignore[no-untyped-def]
        nonlocal server_count
        server_count += 1
        if app is ArmSide.RIGHT:
            return _RaisingExitServer(app, events)
        return _Server(app, shutdown_event=shutdown if server_count == 2 else None)

    deps = RuntimeDependencies(
        lock_factory=lambda _path: _LockContext(events),
        controller_factory=make_controller,
        app_factory=lambda _controller, side, _arm: side,
        server_factory=make_server,
        mujoco_model_factory=lambda *_args: object(),
        mujoco_world_factory=lambda *_args: world,
        mujoco_driver_factory=lambda *_args, **_kwargs: object(),
    )

    assert run_service(
        _mujoco_config(),
        simulate=False,
        mujoco=True,
        lock_path="unused",
        shutdown_event=shutdown,
        dependencies=deps,
    ) == 1
    assert all(controller.shutdown_requested.is_set() for controller in controllers.values())
    assert "right_web_stop_attempted" in events
    assert events[-3:] == ["world_shutdown", "world_join", "unlock"]
