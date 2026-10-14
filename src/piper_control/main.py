# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Process composition for independent Piper arm runtimes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import faulthandler
from pathlib import Path
import logging
import signal
from threading import Event, Thread
import time
from typing import Any, Callable, Sequence

import uvicorn

from piper_control.api import create_app
from piper_control.config import (
    ArmConfig,
    ArmSide,
    ServiceConfig,
    load_config,
)
from piper_control.controller import ArmController
from piper_control.drivers import (
    PiperArmDriver,
    SimulatedArmDriver,
)
from piper_control.lifecycle import SingletonLock, install_signal_handlers
from piper_control.types import ArmStateSnapshot, ServiceState


LOGGER = logging.getLogger(__name__)
DEFAULT_LOCK_PATH = "/run/piper-control/piper-control.lock"
WEB_SHUTDOWN_TIMEOUT_S = 2.0


def _make_driver(side: ArmSide, config: ArmConfig, simulate: bool) -> Any:
    del side
    if simulate:
        return SimulatedArmDriver(auto_timestamp=True)
    return PiperArmDriver(config)


def _make_server(app: Any, host: str, port: int) -> uvicorn.Server:
    return uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))


def _build_mujoco_model(config: Any, arms: Any) -> Any:
    from piper_control.mujoco_model import build_dual_arm_model

    return build_dual_arm_model(config, arms)


def _make_mujoco_world(model: Any, config: Any, shutdown_event: Event) -> Any:
    from piper_control.mujoco_world import MujocoWorld

    return MujocoWorld(model, config, shutdown_event)


def _make_mujoco_driver(
    side: ArmSide,
    config: ArmConfig,
    world: Any,
    *,
    gripper_model_max_m: float,
) -> Any:
    from piper_control.drivers.mujoco import MujocoArmDriver

    return MujocoArmDriver(
        side,
        config,
        world,
        gripper_model_max_m=gripper_model_max_m,
    )


@dataclass(frozen=True)
class RuntimeDependencies:
    """Factories used to construct one isolated runtime per configured arm."""

    lock_factory: Callable[[str | Path], Any] = SingletonLock
    driver_factory: Callable[[ArmSide, ArmConfig, bool], Any] = _make_driver
    controller_factory: Callable[[ArmSide, ArmConfig, Any], Any] = ArmController
    server_factory: Callable[[Any, str, int], Any] = _make_server
    app_factory: Callable[[Any, ArmSide, ArmConfig], Any] = create_app
    mujoco_model_factory: Callable[[Any, Any], Any] = _build_mujoco_model
    mujoco_world_factory: Callable[[Any, Any, Event], Any] = _make_mujoco_world
    mujoco_driver_factory: Callable[..., Any] = _make_mujoco_driver


def _log_event(event: str, **fields: object) -> None:
    LOGGER.info(
        "event=%s %s",
        event,
        " ".join(f"{key}={value}" for key, value in fields.items()),
        extra={"event": event, **fields},
    )


@dataclass
class ArmRuntime:
    """Coordinate the workers and bounded shutdown for one Piper arm."""

    side: ArmSide
    config: ArmConfig
    controller: Any
    server: Any
    control_thread: Thread = field(init=False)
    web_thread: Thread = field(init=False)
    errors: list[tuple[str, BaseException]] = field(default_factory=list)
    readiness: bool = False
    terminal_safety_result: bool | None = None
    _shutdown_started: Event = field(default_factory=Event, init=False)
    _shutdown_deadline: float | None = field(default=None, init=False)
    _pre_shutdown: ArmStateSnapshot | None = field(default=None, init=False)
    _control_started: bool = field(default=False, init=False)
    _web_started: bool = field(default=False, init=False)
    _controller_faulted: bool | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.control_thread = Thread(
            target=self._guarded("control", self.controller.run),
            name=f"control_thread_{self.side.value}",
            daemon=True,
        )
        self.web_thread = Thread(
            target=self._guarded("web", self.server.run),
            name=f"ws_serve_thread_{self.side.value}",
            daemon=True,
        )

    def _fields(self) -> dict[str, object]:
        return {
            "side": self.side.value,
            "can_port": self.config.can_port,
            "port": self.config.server.port,
        }

    def _guarded(self, component: str, target: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                target()
            except BaseException as exc:
                self.errors.append((component, exc))
            else:
                if not self._shutdown_started.is_set():
                    self.errors.append(
                        (component, RuntimeError(f"{component} worker returned early"))
                    )

        return run

    def start(self) -> None:
        self.control_thread.start()
        self._control_started = True
        self.web_thread.start()
        self._web_started = True
        _log_event("runtime_started", **self._fields())

    def ready(self) -> bool:
        if self.readiness:
            return True
        snapshot = self.controller.snapshot()
        self.readiness = (
            snapshot.state is not ServiceState.STARTING
            and bool(getattr(self.server, "started", False))
        )
        if self.readiness:
            _log_event(
                "runtime_ready",
                **self._fields(),
                state=snapshot.state.value,
                fault=snapshot.fault_reason or "none",
            )
        return self.readiness

    def worker_alive(self) -> bool:
        return self.control_thread.is_alive() and self.web_thread.is_alive()

    def expose_worker_fault(self, reason: str) -> None:
        """Publish a worker crash through the controller's safe fault APIs."""

        def transition() -> None:
            try:
                self.controller.publish_runtime_fault(reason)
            except BaseException as exc:
                self.errors.append(("fault_transition", exc))

        Thread(
            target=transition,
            name=f"fault_transition_thread_{self.side.value}",
            daemon=True,
        ).start()

    def _shutdown_snapshot(self) -> ArmStateSnapshot:
        try:
            return self.controller.snapshot()
        except Exception as exc:
            self.errors.append(("shutdown_snapshot", exc))
            return ArmStateSnapshot(
                side=self.side,
                state=ServiceState.EMERGENCY_STOP,
                emergency_stopped=True,
                fault_reason=f"shutdown snapshot failed: {exc}",
            )

    def begin_shutdown(self, *, stop_web: bool = True) -> None:
        """Request shutdown without waiting for either worker."""

        if not self._shutdown_started.is_set():
            self._shutdown_deadline = (
                time.monotonic() + self.config.shutdown_home_timeout_s
            )
            self._shutdown_started.set()
            self._pre_shutdown = self._shutdown_snapshot()
            try:
                request = getattr(self.controller, "request_shutdown_async", None)
                if request is not None:
                    request()
                else:
                    Thread(
                        target=self.controller.request_shutdown,
                        name=f"shutdown_request_thread_{self.side.value}",
                        daemon=True,
                    ).start()
            except BaseException as exc:
                self.errors.append(("shutdown", exc))
            _log_event("shutdown_requested", **self._fields())
        if stop_web:
            try:
                self.server.should_exit = True
            except Exception as exc:
                self.errors.append(("web_shutdown", exc))

    def finish_shutdown(self, *, stop_web: bool = True) -> bool:
        """Wait only within this arm's independent shutdown budgets."""

        if not self._shutdown_started.is_set():
            self.begin_shutdown(stop_web=stop_web)
        assert self._shutdown_deadline is not None
        faulted = bool(self.errors) or bool(self._controller_faulted)
        final_snapshot = self._pre_shutdown or self._shutdown_snapshot()
        controller_result = None
        if self._controller_faulted is None:
            if self._control_started:
                self.control_thread.join(
                    max(0.0, self._shutdown_deadline - time.monotonic())
                )
            if self._control_started and self.control_thread.is_alive():
                faulted = True
                self.errors.append(
                    ("control", TimeoutError("control thread did not stop"))
                )
                timeout_safety = getattr(
                    self.controller, "request_timeout_safety_async", None
                )
                if timeout_safety is not None:
                    timeout_safety("service shutdown deadline exceeded")
            else:
                final_snapshot = self._shutdown_snapshot()
                controller_result = getattr(
                    self.controller, "shutdown_result", lambda: None
                )()
                if controller_result is not None and controller_result.exit_code != 0:
                    faulted = True
                if (
                    final_snapshot.state is ServiceState.EMERGENCY_STOP
                    or not final_snapshot.hold
                ):
                    faulted = True
            self._controller_faulted = faulted

        if stop_web and self._web_started:
            self.web_thread.join(WEB_SHUTDOWN_TIMEOUT_S)
            if self.web_thread.is_alive():
                faulted = True
                self.errors.append(
                    ("web", TimeoutError("web server thread did not stop"))
                )

        for component, exc in self.errors:
            LOGGER.error(
                "event=component_failure component=%s error=%r side=%s can_port=%s port=%s",
                component,
                exc,
                self.side.value,
                self.config.can_port,
                self.config.server.port,
            )
        self.terminal_safety_result = not faulted
        _log_event(
            "shutdown_result",
            result="failure" if faulted else "success",
            **self._fields(),
            state=final_snapshot.state.value,
            hold=final_snapshot.hold,
            actual_q=final_snapshot.actual_q,
            fault_reason=final_snapshot.fault_reason or "none",
            shutdown_errors=(
                "; ".join(controller_result.errors)
                if controller_result is not None and controller_result.errors
                else "none"
            ),
        )
        return not faulted


def run_service(
    config: ServiceConfig,
    *,
    simulate: bool,
    lock_path: str | Path,
    side: ArmSide | None = None,
    mujoco: bool = False,
    mujoco_headless: bool = False,
    shutdown_event: Event | None = None,
    dependencies: RuntimeDependencies | None = None,
    startup_timeout_s: float = 10.0,
) -> int:
    """Run all configured arms while isolating their faults and deadlines."""

    deps = dependencies or RuntimeDependencies()
    stop_requested = shutdown_event or Event()
    runtimes: list[ArmRuntime] = []
    controllers: list[tuple[ArmSide, ArmConfig, Any]] = []
    degraded: set[ArmSide] = set()
    finish_threads: list[Thread] = []
    orchestration_failed = False
    world: Any | None = None
    shared_fault_published = False
    current_side: ArmSide | None = None
    current_config: ArmConfig | None = None
    selected_items = tuple(
        (candidate, arm)
        for candidate, arm in config.items()
        if side is None or candidate is side
    )

    def begin_shutdown_batch(batch: list[ArmRuntime], *, stop_web: bool) -> None:
        nonlocal orchestration_failed
        for runtime in batch:
            try:
                runtime.begin_shutdown(stop_web=stop_web)
            except Exception as exc:
                orchestration_failed = True
                runtime.errors.append(("shutdown_begin", exc))
                LOGGER.exception(
                    "event=shutdown_begin_failure side=%s can_port=%s port=%s",
                    runtime.side.value,
                    runtime.config.can_port,
                    runtime.config.server.port,
                    extra={"event": "shutdown_begin_failure", **runtime._fields()},
                )

    def finish_shutdown_batch(
        batch: list[ArmRuntime], *, stop_web: bool, wait: bool
    ) -> list[Thread]:
        threads = [
            Thread(
                target=runtime.finish_shutdown,
                kwargs={"stop_web": stop_web},
                name=f"shutdown_finish_thread_{runtime.side.value}",
                daemon=True,
            )
            for runtime in batch
        ]
        for thread in threads:
            thread.start()
        finish_threads.extend(threads)
        if wait:
            for thread in threads:
                thread.join()
        return threads

    def shutdown_batch(
        batch: list[ArmRuntime], *, stop_web: bool, wait: bool
    ) -> list[Thread]:
        begin_shutdown_batch(batch, stop_web=stop_web)
        return finish_shutdown_batch(batch, stop_web=stop_web, wait=wait)

    def degrade_batch(batch: list[ArmRuntime], reason: str) -> None:
        affected = [runtime for runtime in batch if runtime.side not in degraded]
        batch_to_finish: list[ArmRuntime] = []
        while affected:
            for runtime in affected:
                degraded.add(runtime.side)
                _log_event("runtime_degraded", reason=reason, **runtime._fields())
                component, error = runtime.errors[-1]
                runtime.expose_worker_fault(f"{component} worker failed: {error}")
            begin_shutdown_batch(affected, stop_web=False)
            batch_to_finish.extend(affected)
            stop_requested.wait(0.01)
            affected = [
                runtime
                for runtime in runtimes
                if runtime.side not in degraded and not runtime.worker_alive()
            ]
        finish_shutdown_batch(batch_to_finish, stop_web=False, wait=False)

    def publish_world_fault(failure: BaseException) -> None:
        nonlocal orchestration_failed, shared_fault_published
        if shared_fault_published:
            return
        shared_fault_published = True
        orchestration_failed = True
        reason = f"MuJoCo world failed: {failure}"
        _log_event("mujoco_world_failure", reason=reason)
        for runtime in runtimes:
            try:
                runtime.controller.publish_runtime_fault(reason)
            except BaseException as exc:
                runtime.errors.append(("fault_transition", exc))

    with deps.lock_factory(lock_path):
        try:
            if not selected_items:
                requested = side.value if side is not None else "unknown"
                raise ValueError(f"selected arm side is not configured: {requested}")
            selected_modes = sum((bool(simulate), bool(mujoco), bool(mujoco_headless)))
            if selected_modes > 1:
                raise ValueError(
                    "simulate, mujoco, and mujoco_headless modes are mutually exclusive"
                )
            mujoco_selected = bool(mujoco or mujoco_headless)
            mujoco_config = config.mujoco
            if mujoco_selected:
                if mujoco_config is None:
                    raise ValueError("MuJoCo mode requires a mujoco configuration")
                if mujoco_headless and mujoco_config.viewer:
                    mujoco_config = mujoco_config.model_copy(
                        update={"viewer": False}
                    )
                model = deps.mujoco_model_factory(mujoco_config, config.arms)
                world = deps.mujoco_world_factory(model, mujoco_config, stop_requested)
                _log_event(
                    "mujoco_world_starting",
                    mode="mujoco_headless" if mujoco_headless else "mujoco",
                    model_path=mujoco_config.model_path,
                    timestep_s=mujoco_config.timestep_s,
                    viewer=mujoco_config.viewer,
                    left_port=config.arms[ArmSide.LEFT].server.port,
                    right_port=config.arms[ArmSide.RIGHT].server.port,
                )
                world.start()
                if world.failure is not None:
                    raise RuntimeError(f"MuJoCo world startup failed: {world.failure}")

            for runtime_side, arm_config in selected_items:
                current_side = runtime_side
                current_config = arm_config
                _log_event(
                    "configuration_summary",
                    mode=(
                        "mujoco_headless"
                        if mujoco_headless
                        else "mujoco"
                        if mujoco
                        else "simulate"
                        if simulate
                        else "hardware"
                    ),
                    control_hz=arm_config.control_hz,
                    side=runtime_side.value,
                    can_port=arm_config.can_port,
                    port=arm_config.server.port,
                )
                if mujoco_selected:
                    assert world is not None and mujoco_config is not None
                    driver = deps.mujoco_driver_factory(
                        runtime_side,
                        arm_config,
                        world,
                        gripper_model_max_m=mujoco_config.gripper_model_max_m,
                    )
                else:
                    driver = deps.driver_factory(runtime_side, arm_config, simulate)
                controller = deps.controller_factory(runtime_side, arm_config, driver)
                controllers.append((runtime_side, arm_config, controller))
                app = deps.app_factory(controller, runtime_side, arm_config)
                server = deps.server_factory(
                    app, arm_config.server.host, arm_config.server.port
                )
                runtimes.append(
                    ArmRuntime(
                        side=runtime_side,
                        config=arm_config,
                        controller=controller,
                        server=server,
                    )
                )

            for runtime in runtimes:
                current_side = runtime.side
                current_config = runtime.config
                runtime.start()

            startup_deadline = time.monotonic() + startup_timeout_s
            while True:
                if world is not None and world.failure is not None:
                    publish_world_fault(world.failure)
                    break
                if world is not None and world.viewer_closed:
                    stop_requested.set()
                    break
                if stop_requested.is_set():
                    break
                pending = [
                    runtime
                    for runtime in runtimes
                    if runtime.side not in degraded and not runtime.ready()
                ]
                failed = [runtime for runtime in pending if not runtime.worker_alive()]
                if failed:
                    degrade_batch(failed, "worker exited before readiness")
                pending = [
                    runtime
                    for runtime in runtimes
                    if runtime.side not in degraded and not runtime.readiness
                ]
                if not pending:
                    break
                if time.monotonic() >= startup_deadline:
                    for runtime in pending:
                        runtime.errors.append(
                            ("startup", TimeoutError("startup readiness timeout"))
                        )
                    degrade_batch(pending, "startup readiness timeout")
                    break
                stop_requested.wait(0.01)

            while True:
                if world is not None and world.failure is not None:
                    publish_world_fault(world.failure)
                    break
                if world is not None and world.viewer_closed:
                    stop_requested.set()
                    break
                if stop_requested.is_set():
                    break
                healthy = [
                    runtime
                    for runtime in runtimes
                    if runtime.side not in degraded
                ]
                failed = [runtime for runtime in healthy if not runtime.worker_alive()]
                if failed:
                    degrade_batch(failed, "worker exited after readiness")
                if len(degraded) == len(runtimes):
                    break
                stop_requested.wait(0.01)
        except Exception as exc:
            orchestration_failed = True
            LOGGER.exception(
                "event=orchestration_failure error=%r side=%s can_port=%s port=%s",
                exc,
                current_side.value if current_side is not None else "unknown",
                current_config.can_port if current_config is not None else "unknown",
                current_config.server.port if current_config is not None else "unknown",
                extra={
                    "event": "orchestration_failure",
                    "side": current_side.value if current_side is not None else "unknown",
                    "can_port": (
                        current_config.can_port
                        if current_config is not None
                        else "unknown"
                    ),
                    "port": (
                        current_config.server.port
                        if current_config is not None
                        else "unknown"
                    ),
                },
            )
        finally:
            begin_shutdown_batch(runtimes, stop_web=True)
            runtime_controllers = {id(runtime.controller) for runtime in runtimes}
            for side, arm_config, controller in controllers:
                if id(controller) in runtime_controllers:
                    continue
                try:
                    request = getattr(controller, "request_shutdown_async", None)
                    if request is None:
                        request = controller.request_shutdown
                    request()
                except Exception:
                    LOGGER.exception(
                        "event=partial_runtime_shutdown_failure side=%s can_port=%s port=%s",
                        side.value,
                        arm_config.can_port,
                        arm_config.server.port,
                        extra={
                            "event": "partial_runtime_shutdown_failure",
                            "side": side.value,
                            "can_port": arm_config.can_port,
                            "port": arm_config.server.port,
                        },
                    )
            for thread in finish_threads:
                thread.join()
            shutdown_batch(runtimes, stop_web=True, wait=True)
            if world is not None:
                if world.failure is not None:
                    publish_world_fault(world.failure)
                try:
                    world.request_shutdown()
                except Exception:
                    orchestration_failed = True
                    LOGGER.exception(
                        "event=mujoco_world_shutdown_failure",
                        extra={"event": "mujoco_world_shutdown_failure"},
                    )
                try:
                    world_joined = world.join(WEB_SHUTDOWN_TIMEOUT_S)
                except Exception:
                    orchestration_failed = True
                    LOGGER.exception(
                        "event=mujoco_world_join_failure",
                        extra={"event": "mujoco_world_join_failure"},
                    )
                else:
                    if not world_joined:
                        orchestration_failed = True
                        LOGGER.error(
                            "event=mujoco_world_shutdown_timeout timeout_s=%s",
                            WEB_SHUTDOWN_TIMEOUT_S,
                            extra={"event": "mujoco_world_shutdown_timeout"},
                        )
                if world.failure is not None:
                    publish_world_fault(world.failure)

        return (
            0
            if not orchestration_failed
            and all(runtime.terminal_safety_result for runtime in runtimes)
            else 1
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Piper control service")
    parser.add_argument("--config", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--simulate", action="store_true")
    modes.add_argument("--mujoco", action="store_true")
    modes.add_argument("--mujoco-headless", action="store_true")
    parser.add_argument("--side", type=ArmSide, choices=tuple(ArmSide))
    parser.add_argument("--lock-path", default=DEFAULT_LOCK_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    faulthandler.register(signal.SIGUSR2, all_threads=True)
    try:
        config = load_config(Path(args.config))
        shutdown_event = Event()
        install_signal_handlers(shutdown_event)
        return run_service(
            config,
            simulate=args.simulate,
            mujoco=args.mujoco,
            mujoco_headless=args.mujoco_headless,
            side=args.side,
            lock_path=args.lock_path,
            shutdown_event=shutdown_event,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("event=startup_failure error=%r", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
