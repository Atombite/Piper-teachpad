# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Safe SocketCAN administration driven only by validated service config."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Protocol, Sequence

from piper_control.config import ServiceConfig, load_config


CAN_BITRATE = 1_000_000


class CanCommandError(RuntimeError):
    """A requested CAN inspection or mutation could not be completed."""


class CommandResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str]) -> CommandResult: ...


class SubprocessRunner:
    def run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
        )


@dataclass(frozen=True, slots=True)
class CanState:
    up: bool
    bitrate: int | None
    details: str


def _run(runner: CommandRunner, argv: Sequence[str]) -> CommandResult:
    result = runner.run(argv)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "command failed"
        raise CanCommandError(f"{' '.join(argv)}: {detail}")
    return result


def _inspect(interface: str, runner: CommandRunner) -> CanState:
    result = _run(
        runner,
        ("ip", "-details", "-statistics", "link", "show", "dev", interface),
    )
    first_line = result.stdout.splitlines()[0] if result.stdout.splitlines() else ""
    flags = (
        first_line.split("<", 1)[1].split(">", 1)[0].split(",")
        if "<" in first_line and ">" in first_line
        else []
    )
    bitrate_match = re.search(r"\bbitrate\s+(\d+)\b", result.stdout)
    return CanState(
        up="UP" in flags,
        bitrate=int(bitrate_match.group(1)) if bitrate_match else None,
        details=result.stdout,
    )


def _interfaces(config: ServiceConfig) -> tuple[str, ...]:
    return tuple(arm.can_port for _side, arm in config.items())


def can_up(config: ServiceConfig, runner: CommandRunner | None = None) -> int:
    actual_runner = runner or SubprocessRunner()
    for interface in _interfaces(config):
        state = _inspect(interface, actual_runner)
        if state.up and state.bitrate == CAN_BITRATE:
            continue
        if state.up:
            _run(actual_runner, ("ip", "link", "set", "dev", interface, "down"))
        if state.bitrate != CAN_BITRATE:
            _run(
                actual_runner,
                (
                    "ip",
                    "link",
                    "set",
                    "dev",
                    interface,
                    "type",
                    "can",
                    "bitrate",
                    str(CAN_BITRATE),
                ),
            )
        _run(actual_runner, ("ip", "link", "set", "dev", interface, "up"))
    return 0


def can_status(config: ServiceConfig, runner: CommandRunner | None = None) -> int:
    actual_runner = runner or SubprocessRunner()
    for interface in _interfaces(config):
        state = _inspect(interface, actual_runner)
        print(state.details, end="" if state.details.endswith("\n") else "\n")
    return 0


def can_down(config: ServiceConfig, runner: CommandRunner | None = None) -> int:
    actual_runner = runner or SubprocessRunner()
    for interface in _interfaces(config):
        state = _inspect(interface, actual_runner)
        if state.up:
            _run(actual_runner, ("ip", "link", "set", "dev", interface, "down"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Administer configured Piper CAN links")
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=("up", "status", "down"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(Path(args.config))
        return {"up": can_up, "status": can_status, "down": can_down}[args.command](config)
    except (OSError, ValueError, CanCommandError) as exc:
        print(f"piper-control-can: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
