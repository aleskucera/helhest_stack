"""The commander bridge's route logic: the same arrival test and sequence walk as crl_commander,
so the Robotour follower sees the behaviour it was written against."""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

_PATH = pathlib.Path(__file__).resolve().parents[2] / "ros/tools/commander_bridge.py"
_spec = importlib.util.spec_from_file_location("commander_bridge", _PATH)
cb = importlib.util.module_from_spec(_spec)
sys.modules["commander_bridge"] = cb  # dataclasses look their module up here
_spec.loader.exec_module(cb)

ROUTE = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]


def test_arrival_is_a_box_in_the_robots_frame():
    # 2 m ahead and 2 m left of a robot facing +y is inside a 2.5 x 2.5 box...
    assert cb.reached((0.0, 0.0, math.pi / 2), (-2.0, 2.0), 2.5, 2.5)
    # ...3 m to its side is not, although 3 m ahead would be outside the box too
    assert not cb.reached((0.0, 0.0, math.pi / 2), (3.0, 0.0), 2.5, 2.5)


def test_a_sequence_starts_after_the_nearest_waypoint():
    assert cb.start_index((9.0, 1.0), ROUTE, from_next=True) == 2
    assert cb.start_index((9.0, 1.0), ROUTE, from_next=False) == 1
    # at the end of an open route: sent to the last waypoint, which it has already reached
    assert cb.start_index((31.0, 0.0), ROUTE, from_next=True) == 3
    assert cb.start_index((0.0, 0.0), [], from_next=True) is None


def test_a_loop_route_starts_at_its_beginning():
    # kolecko2: the start and end of a loop lie side by side and the robot, between them, was
    # nearest the LAST waypoint -- the sequence ended before it began, 20 times in a row
    loop = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0), (0.0, 3.0)]
    assert cb.start_index((0.0, 2.0), loop, from_next=True) == 1


def test_the_walk_advances_on_arrival_and_ends():
    w = cb.Walker(box_x=2.5, box_y=2.5, timeout_s=180.0)
    assert w.begin((1.0, 0.0), ROUTE, from_next=True, now=0.0) and w.index == 1
    assert w.step((5.0, 0.0, 0.0), ROUTE, now=1.0) == "active"
    assert w.step((9.0, 0.5, 0.0), ROUTE, now=2.0) == "advanced" and w.index == 2
    assert w.step((19.0, 0.0, 0.0), ROUTE, now=3.0) == "advanced" and w.index == 3
    assert w.step((29.0, 0.0, 0.0), ROUTE, now=4.0) == "done" and w.index is None


def test_a_waypoint_is_skipped_after_the_timeout_and_a_loop_wraps():
    w = cb.Walker(timeout_s=180.0, loop=True)
    w.begin((1.0, 0.0), ROUTE, from_next=True, now=0.0)
    assert w.step((1.0, 0.0, 0.0), ROUTE, now=100.0) == "active"
    assert w.step((1.0, 0.0, 0.0), ROUTE, now=181.0) == "advanced" and w.index == 2
    w.index = 3
    assert w.step((30.0, 0.0, 0.0), ROUTE, now=200.0) == "advanced" and w.index == 0


def test_a_waypoint_is_lifted_along_the_ellipsoid_normal():
    # a GPX point at altitude 0 near Temesvar, lifted to the robot's height
    lat, lon = math.radians(49.37), math.radians(14.26)
    point = cb.geodetic_to_ecef(lat, lon, 0.0)
    lat2, lon2, h2 = cb.ecef_to_geodetic(*cb.lift_to_height(point, 454.0))
    assert abs(h2 - 454.0) < 1e-3
    # same place on the ground: < 1 mm of latitude / longitude change
    assert abs(lat2 - lat) * cb.WGS84_A < 1e-3 and abs(lon2 - lon) * cb.WGS84_A < 1e-3
