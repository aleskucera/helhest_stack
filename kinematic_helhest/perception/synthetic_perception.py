"""Synthetic perception stand-in for the rolling-map planner.

Mimics terrain_toolkit's eventual `TerrainMapGPU` contract exactly — a device
elevation `wp.array` + `resolution` + `bounds` — but sources it by cropping a
static world `Heightmap` into a robot-centered window. Lets the whole rolling-map
architecture (navigate.py, the integrated viewer) be built and tested with NO
terrain_toolkit dependency; Phase 4 swaps `crop_window` for the real pipeline.

Frame: robot-attached. The window is centered on the robot and rotated so +x is
the robot's heading; the robot plans at local (0, 0, 0). Elevation keeps absolute
(gravity-aligned) world z — only the XY is re-indexed — so the kinematic settle is
unchanged.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class DeviceMap:
    """The generic perception contract navigate.py consumes (duck-typed): a device
    elevation grid + grid metadata. terrain_toolkit's TerrainMapGPU is the same shape."""

    elevation: object        # device wp.array2d(float32) [ny, nx], row=y, col=x
    resolution: float        # meters / cell
    bounds: tuple            # (xmin, xmax, ymin, ymax) in the local (robot) frame
    traversability: object = None  # optional device cost grid; None for the stand-in


def crop_window(world_hm, robot_xy, robot_yaw, half_extent, res, device="cpu"):
    """Robot-attached local window cropped + rotated from a world `Heightmap`.

    `robot_xy` world position, `robot_yaw` heading [rad], `half_extent` window
    half-size [m], `res` window cell size [m]. Returns a `DeviceMap` whose origin
    cell-center grid is centered on the robot with +x along the heading.
    """
    n = int(round(2.0 * half_extent / res))
    lc = -half_extent + (np.arange(n) + 0.5) * res     # cell centers, local axis
    LX, LY = np.meshgrid(lc, lc)                         # [n, n]: col=x, row=y
    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    WX = robot_xy[0] + c * LX - s * LY                  # rotate local -> world
    WY = robot_xy[1] + s * LX + c * LY
    H = world_hm.sample(WX, WY).astype(np.float32)      # bilinear from the world
    H_wp = wp.array(np.ascontiguousarray(H), dtype=wp.float32, device=device)
    return DeviceMap(H_wp, float(res), (-half_extent, half_extent, -half_extent, half_extent))


def to_local(goal, state):
    """Project a world goal (x, y) into the robot frame at world `state` (x, y, yaw)."""
    dx, dy = goal[0] - state[0], goal[1] - state[1]
    c, s = np.cos(state[2]), np.sin(state[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy], np.float64)  # R(-yaw) . delta
