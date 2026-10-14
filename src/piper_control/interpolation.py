# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Pure interpolation helpers for control-loop axes."""

from __future__ import annotations

from collections.abc import Sequence


def step_axes(
    current: Sequence[float],
    target: Sequence[float],
    speed: Sequence[float],
    dt: float,
) -> tuple[float, ...]:
    """Move every axis toward its target by at most ``speed * dt``."""

    try:
        return tuple(
            current_value
            + max(
                -speed_value * dt,
                min(speed_value * dt, target_value - current_value),
            )
            for current_value, target_value, speed_value in zip(
                current, target, speed, strict=True
            )
        )
    except ValueError as exc:
        raise ValueError("current, target, and speed dimensions must match") from exc
