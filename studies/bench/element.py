"""The wheel structuring element, shared by every Clark-family benchmark's contact choke point:
`wheel_offset_table(...) -> _footprint_cells(...)`.

Two elements:

  sphere    the wheel's contact disk (`helhest.engine.envelope.wheel_offset_table`), YAW-
            INVARIANT: one [K] candidate table shared by every node.
  cylinder  the ruler-measured 0.10 m-wide tread, a rotated rectangle 2*radius (along travel) x
            2*half_width (across) -- NOT yaw-invariant, so it is one [K] table PER NODE HEADING,
            quantized to 32 bins so the table is built once per bin and gathered rather than
            rebuilt per node (`clark_conv.py`'s original derivation; K = 5-7 vs the disk's 37).

`clark_conv.py` and `clark_hinge_fast.py` proved this recipe correct against `clark.py`'s exact
(non-convolution) machinery for the sphere, and reported it as a PHYSICS change (not an error) for
the cylinder. Every other benchmark's own choke point now calls through this ONE copy instead of
re-deriving it.
"""

from __future__ import annotations

import math

import numpy as np

from helhest.engine import RobotParams
from helhest.engine.envelope import wheel_offset_table

WHEEL_HALF_WIDTH = 0.05  # [m] half of the ruler-measured 0.10 m tread (engine/robot.py)
N_YAW_BINS = 32  # heading quantization the cylinder table is shared/gathered at


def cylinder_offsets(
    cell: float, radius: float, half_width: float, yaw: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cylinder wheel's structuring element at heading `yaw`, mirroring
    `helhest.engine.envelope.cylinder_offset_table`: the rotated rectangle
    2*radius (along travel) x 2*half_width (across), capped by the along-travel offset alone."""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    env_radius = int(math.ceil(math.hypot(radius, half_width) / cell))
    dy_l, dx_l, cap_l = [], [], []
    for dy in range(-env_radius, env_radius + 1):
        for dx in range(-env_radius, env_radius + 1):
            wx, wy = dx * cell, dy * cell
            along = wx * cos_y + wy * sin_y
            across = -wx * sin_y + wy * cos_y
            if abs(along) <= radius and abs(across) <= half_width:
                dy_l.append(dy)
                dx_l.append(dx)
                cap_l.append(math.sqrt(radius**2 - along**2) - radius)
    return np.array(dy_l, np.int64), np.array(dx_l, np.int64), np.array(cap_l, np.float64)


def element_offsets(
    element: str, cell: float, rp: RobotParams, yaw: np.ndarray, n_bins: int = N_YAW_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The structuring element's candidate offsets, at the `wheel_offset_table` choke point every
    sphere-only benchmark shares.

    sphere:   off_dy/off_dx/off_cap are [K], shared by every node (yaw-invariant); `yaw` unused.
    cylinder: off_dy/off_dx/off_cap are [N, K], one row per entry of `yaw` (radians, any shape --
              raveled to [N]), quantized to `n_bins` heading bins. K is truncated to the smallest
              per-bin table (yaw-ragged: a diagonal heading clips a few more cells than an axis-
              aligned one), matching every node to the SAME K so they stack into one array.
    """
    if element == "sphere":
        env_radius = int(np.ceil(rp.wheel_radius / cell))
        off_dy, off_dx, off_cap = wheel_offset_table(env_radius, cell, rp.wheel_radius)
        return np.asarray(off_dy, np.int64), np.asarray(off_dx, np.int64), np.asarray(off_cap)
    if element == "cylinder":
        yaw = np.asarray(yaw).ravel()
        bins = np.round(yaw / (2 * np.pi) * n_bins).astype(np.int64) % n_bins
        tables = {
            b: cylinder_offsets(cell, rp.wheel_radius, WHEEL_HALF_WIDTH, 2 * np.pi * b / n_bins)
            for b in np.unique(bins)
        }
        k_min = min(len(t[0]) for t in tables.values())
        off_dy = np.stack([tables[b][0][:k_min] for b in bins])
        off_dx = np.stack([tables[b][1][:k_min] for b in bins])
        off_cap = np.stack([tables[b][2][:k_min] for b in bins])
        return off_dy, off_dx, off_cap
    raise ValueError(f"unknown element {element!r}")


def element_offsets_single(
    element: str, cell: float, rp: RobotParams, yaw: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`element_offsets` for callers with no per-node pose context -- e.g. `softgrad.py`'s and
    `order2.py`'s whole-grid roll mechanism, which applies ONE table uniformly to every cell and
    so has no per-cell heading to give the cylinder its real orientation. Always returns [K]
    arrays at a single reference heading: exact for the sphere (genuinely yaw-invariant), a
    DECLARED approximation for the cylinder (aligned with the world x-axis by default, `yaw=0`)."""
    off_dy, off_dx, off_cap = element_offsets(element, cell, rp, np.array([yaw]))
    if off_dy.ndim == 1:
        return off_dy, off_dx, off_cap
    return off_dy[0], off_dx[0], off_cap[0]


def broadcast_cap(off_cap: np.ndarray) -> np.ndarray:
    """`off_cap` ready to add to a [N, K] candidate-means array: the sphere's shared [K] table
    broadcasts over nodes via a leading axis; the cylinder's per-node [N, K] table already is
    node-aligned and passes through unchanged."""
    return off_cap if off_cap.ndim == 2 else off_cap[None, :]
