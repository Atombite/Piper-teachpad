# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from piper_control.interpolation import step_axes


def test_step_axes_limits_each_axis_without_overshooting():
    assert step_axes((0.0, 10.0), (5.0, 0.0), (2.0, 4.0), 0.5) == (
        1.0,
        8.0,
    )


def test_step_axes_stops_exactly_at_nearby_target():
    assert step_axes((0.0, 2.0), (0.25, 1.75), (2.0, 4.0), 0.5) == (
        0.25,
        1.75,
    )


@pytest.mark.parametrize(
    ("current", "target", "speed"),
    [
        ((0.0,), (1.0, 2.0), (1.0, 1.0)),
        ((0.0, 1.0), (1.0,), (1.0, 1.0)),
        ((0.0, 1.0), (1.0, 2.0), (1.0,)),
    ],
)
def test_step_axes_rejects_mismatched_dimensions(current, target, speed):
    with pytest.raises(ValueError):
        step_axes(current, target, speed, 0.1)
