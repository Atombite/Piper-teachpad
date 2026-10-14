# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Strict configuration models for independent Piper Bridge processes."""

from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

import yaml
from typing import Annotated, Any
from pydantic import (
    Field,
    FiniteFloat,
    Strict,
    StrictInt,
    field_validator,
    model_validator,
)

from piper_control.config import ArmSide, FrozenModel


StrictFiniteFloat = Annotated[FiniteFloat, Strict()]


class BridgeEndpointConfig(FrozenModel):
    """TCP listener used by one downstream Bridge service."""

    host: str = Field(min_length=1)
    port: StrictInt = Field(ge=1, le=65535)


class BridgeConfig(FrozenModel):
    """Configuration for one arm-local Bridge instance."""

    listen: BridgeEndpointConfig
    upstream_url: str = Field(min_length=1)
    control_hz: StrictFiniteFloat = Field(gt=0)
    command_ttl_s: StrictFiniteFloat = Field(default=1.0, gt=0)
    owner_timeout_s: StrictFiniteFloat = Field(default=3.0, gt=0)
    reconnect_initial_s: StrictFiniteFloat = Field(default=0.25, gt=0)
    reconnect_max_s: StrictFiniteFloat = Field(default=5.0, gt=0)

    @field_validator("upstream_url")
    @classmethod
    def validate_upstream_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
            raise ValueError("upstream_url must be an absolute WebSocket URL")
        if parsed.path != "/ws/control" or parsed.query or parsed.fragment:
            raise ValueError("upstream_url must target /ws/control")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("upstream_url contains an invalid port") from exc
        if port is None:
            raise ValueError("upstream_url must include an explicit port")
        return value

    @property
    def queue_capacity(self) -> int:
        return math.ceil(float(self.control_hz) * 5.0)

    @model_validator(mode="after")
    def validate_reconnect_backoff(self) -> "BridgeConfig":
        if self.reconnect_max_s < self.reconnect_initial_s:
            raise ValueError("reconnect maximum must not be below initial delay")
        return self


class BridgeServiceConfig(FrozenModel):
    """Side-indexed configuration for launching independent Bridge instances."""

    bridges: dict[ArmSide, BridgeConfig]

    @model_validator(mode="after")
    def validate_bridges(self) -> "BridgeServiceConfig":
        if not self.bridges:
            raise ValueError("configuration must contain at least one bridge")
        ports = [bridge.listen.port for bridge in self.bridges.values()]
        if len(ports) != len(set(ports)):
            raise ValueError("bridge listen ports must be unique")
        upstream_urls = [bridge.upstream_url for bridge in self.bridges.values()]
        if len(upstream_urls) != len(set(upstream_urls)):
            raise ValueError("bridge upstream URLs must be unique")
        object.__setattr__(self, "bridges", MappingProxyType(dict(self.bridges)))
        return self

    def items(self) -> tuple[tuple[ArmSide, BridgeConfig], ...]:
        return tuple(
            (side, self.bridges[side])
            for side in (ArmSide.RIGHT, ArmSide.LEFT)
            if side in self.bridges
        )


def load_bridge_config(path: Path) -> BridgeServiceConfig:
    """Load Bridge YAML without rewriting URLs or other endpoint values."""

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"failed to parse YAML configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"YAML configuration {path} must contain a mapping")
    return BridgeServiceConfig.model_validate(raw)
