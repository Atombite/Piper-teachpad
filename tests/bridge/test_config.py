# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from piper_control.bridge.config import BridgeServiceConfig, load_bridge_config
from piper_control.config import ArmSide


def valid_config() -> dict[str, object]:
    return {
        "bridges": {
            "left": {
                "listen": {"host": "127.0.0.1", "port": 18765},
                "upstream_url": "ws://127.0.0.1:8765/ws/control",
                "control_hz": 100.0,
            },
            "right": {
                "listen": {"host": "127.0.0.1", "port": 19876},
                "upstream_url": "ws://127.0.0.1:9876/ws/control",
                "control_hz": 99.5,
            },
        }
    }


def write_yaml(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "bridge.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_loads_strict_frozen_side_configs_and_derived_queue_capacity(
    tmp_path: Path,
) -> None:
    config = load_bridge_config(write_yaml(tmp_path, valid_config()))

    left = config.bridges[ArmSide.LEFT]
    right = config.bridges[ArmSide.RIGHT]
    assert left.listen.host == "127.0.0.1"
    assert left.listen.port == 18765
    assert left.upstream_url == "ws://127.0.0.1:8765/ws/control"
    assert left.command_ttl_s == 1.0
    assert left.owner_timeout_s == 3.0
    assert left.reconnect_initial_s == 0.25
    assert left.reconnect_max_s == 5.0
    assert left.queue_capacity == 500
    assert right.queue_capacity == 498
    assert [side for side, _bridge in config.items()] == [
        ArmSide.RIGHT,
        ArmSide.LEFT,
    ]
    with pytest.raises(ValidationError, match="frozen"):
        left.control_hz = 50
    with pytest.raises(TypeError):
        config.bridges[ArmSide.LEFT] = right


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update({"unexpected": True}), "extra"),
        (lambda raw: raw["bridges"]["left"].update({"unexpected": True}), "extra"),
        (lambda raw: raw["bridges"]["left"]["listen"].update({"extra": 1}), "extra"),
        (lambda raw: raw["bridges"]["left"].update({"control_hz": 0}), "greater than 0"),
        (lambda raw: raw["bridges"]["left"].update({"control_hz": float("nan")}), "finite"),
        (lambda raw: raw["bridges"]["left"].update({"command_ttl_s": 0}), "greater than 0"),
        (lambda raw: raw["bridges"]["left"].update({"owner_timeout_s": float("inf")}), "finite"),
        (lambda raw: raw["bridges"]["left"].update({"reconnect_initial_s": 0}), "greater than 0"),
        (lambda raw: raw["bridges"]["left"].update({"reconnect_initial_s": 2, "reconnect_max_s": 1}), "reconnect maximum"),
        (lambda raw: raw["bridges"]["left"].update({"upstream_url": "http://127.0.0.1"}), "WebSocket"),
    ],
)
def test_rejects_unknown_or_unsafe_configuration(mutate, message: str) -> None:
    raw = deepcopy(valid_config())
    mutate(raw)
    with pytest.raises(ValidationError, match=message):
        BridgeServiceConfig.model_validate(raw)


def test_rejects_duplicate_listen_ports() -> None:
    raw = valid_config()
    raw["bridges"]["right"]["listen"]["port"] = 18765  # type: ignore[index]
    with pytest.raises(ValidationError, match="listen ports must be unique"):
        BridgeServiceConfig.model_validate(raw)


def test_rejects_duplicate_upstream_controller_urls() -> None:
    raw = valid_config()
    raw["bridges"]["right"]["upstream_url"] = raw["bridges"]["left"]["upstream_url"]  # type: ignore[index]
    with pytest.raises(ValidationError, match="upstream URLs must be unique"):
        BridgeServiceConfig.model_validate(raw)


def test_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_bridge_config(write_yaml(tmp_path, ["not", "a", "mapping"]))
