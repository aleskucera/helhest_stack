"""`obstacle_clearance`: the safety metric the robust-margin change is judged by.

The margin exists so the robot never drives into a wall, and the closed-loop harness could not see
that -- it scored a run by whether it REACHED, which a robot scraping a wall still does. These pin
the metric to distances worked out by hand, because a safety metric that is quietly off by a few
centimetres is worse than none.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import helhest.worlds as w
from helhest.worlds import Box
from helhest.worlds import footprint
from helhest.worlds import obstacle_clearance

# front axle at the origin, rear wheel 0.75 behind, r = 0.35, half_track 0.365, wheel_width 0.10
FP = footprint(
    SimpleNamespace(rear_offset=0.75, wheel_radius=0.35, half_track=0.365, wheel_width=0.1)
)


def test_footprint_is_the_bare_robot():
    """No margin: padding the robot would turn a graze into a pass."""
    assert FP == pytest.approx((-1.10, 0.35, -0.415, 0.415))


# gap: wall halves at x = 6.0 +- 0.2, spanning |y| >= 0.9. Near face at x = 5.8.
@pytest.mark.parametrize(
    "x, y, yaw, want, what",
    [
        (5.8 - 0.35 - 0.5, 3.2, 0.0, 0.5, "nose 0.5 m short of the wall"),
        (5.8 - 0.35 + 0.1, 3.2, 0.0, -0.1, "nose 0.1 m INTO the wall"),
        (5.8 - 1.10 - 0.3, 3.2, math.pi, 0.3, "reversed: rear 0.3 m short"),
        (6.0, 0.0, 0.0, 0.9 - 0.415, "threading the gap: lateral to both halves"),
    ],
)
def test_distances_worked_by_hand(x, y, yaw, want, what):
    assert obstacle_clearance("gap", x, y, yaw, FP) == pytest.approx(want, abs=1e-6), what


def test_a_box_corner_inside_the_robot_is_caught(monkeypatch):
    """A 45-degree box whose corner pokes 1 cm into the robot's side, between two boundary samples
    -- the case the corner check exists for. Sampling the robot's edge alone would report clear."""
    monkeypatch.setitem(w.OBSTACLES, "corner", (Box(0.0, 0.0, 0.3, 0.3, yaw=math.pi / 4),))
    top = 0.3 * math.sqrt(2.0)  # the diamond's upper corner
    x = 0.375  # centres the robot's 1.45 m length over the corner
    y = (top - 0.01) + 0.415  # lower side 1 cm below the corner
    assert obstacle_clearance("corner", x, y, 0.0, FP) == pytest.approx(-0.01, abs=1e-6)


def test_a_world_without_solids_has_no_wall_to_hit():
    assert obstacle_clearance("bumpy", 5.0, 0.0, 0.0, FP) == float("inf")
