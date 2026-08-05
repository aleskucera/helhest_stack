"""Labelled non-uniform terrain for the adjoint-sensitivity studies.

ONE scene, four y-lanes, each carrying a different local terrain character, so every cell
carries a REGION label and the flat-vs-curb split of Study A is a lookup rather than a
judgement call. Regions are a property of the CELL, not of the rollout: a rollout's
gradient support is stratified by looking up each support cell's label.

Why the lanes and not a single feature: on flat ground the terrain slope (gx, gy) is zero,
which zeroes `J[i,1] = dp[2] - gx*dp[0] - gy*dp[1]` back to `dp[2]` and kills the whole
`adj_pose[0]/[1]` accumulation in the settle adjoint. A flat-terrain gradient check
therefore exercises none of those terms -- it cannot fail. Every feature below is also
rotated off the grid axes so a sign error in x cannot be cancelled by symmetry in y.

  FLAT   y0[0]  h = 0, non-uniform friction only (incl. a sharp low-mu patch)
  SLOPE  y0[1]  constant 10 deg slope, in-plane direction 15 deg off +x
  CURB   y0[2]  sharp 0.15 m step along a line 20 deg off the y-axis
  ROCK   y0[3]  isolated sub-wheel-scale rocks, laterally offset from the wheel track

The lanes are windowed with a smooth taper so the boundaries between them carry no
artificial cliff, and the taper starts (1.2 m from the lane centre) outside the reach of
the robot's widest wheel-envelope contact (half-track 0.365 + wheel radius 0.35 = 0.715 m).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Region codes. FLAT covers every smooth cell inside a lane, including the raised plateau
# past the curb -- "flat" here means "smooth, no switch expected", not "at z = 0".
FLAT = 0
SLOPE = 1
CURB = 2
ROCK = 3
OTHER = 4  # between lanes: the taper bands, never touched by a wheel
REGION_NAMES = ("flat", "slope", "curb", "rock", "other")

XLIM = (-1.5, 4.5)
YLIM = (-6.0, 6.0)
LANE_Y = (-4.5, -1.5, 1.5, 4.5)  # flat, slope, curb, rock
LANE_INNER = 1.2  # [m] full-strength half-width of a lane
LANE_OUTER = 1.5  # [m] taper reaches zero here; lanes touch, no cliff

SLOPE_DEG = 10.0  # slope-lane grade
SLOPE_DIR_DEG = 15.0  # in-plane slope direction, off +x
SLOPE_START_X = 0.0

CURB_HEIGHT = 0.15  # [m] step rise
CURB_X = 1.5  # [m] step location at the lane centreline
CURB_DIR_DEG = 20.0  # step edge tilted off the y-axis
CURB_BAND = 0.35  # [m] cells within one wheel radius of the edge are labelled CURB

# (world x, dy from lane centre, height, radius, sharp?)
ROCKS = (
    (0.9, 0.30, 0.20, 0.12, False),  # smooth bump, offset onto the left wheel track
    (1.8, -0.28, 0.18, 0.12, True),  # sharp disk, offset onto the right wheel track
    (2.6, 0.05, 0.16, 0.10, False),  # smooth bump, on the centreline (straddled)
)
ROCK_BAND = 0.35  # [m] cells within one wheel radius of a rock centre


@dataclass(frozen=True)
class Scene:
    """Raw elevation, friction and region labels on a shared [ny, nx] grid."""

    elevation: np.ndarray  # [ny, nx] float64, raw heights [m]
    friction: np.ndarray  # [ny, nx] float64, mu
    region: np.ndarray  # [ny, nx] int8, one of the region codes
    cell: float
    origin_x: float
    origin_y: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape


def _grid(cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre world coordinates, matching helhest.heightmap's convention."""
    nx = int(round((XLIM[1] - XLIM[0]) / cell)) + 1
    ny = int(round((YLIM[1] - YLIM[0]) / cell)) + 1
    xs = XLIM[0] + (np.arange(nx) + 0.5) * cell
    ys = YLIM[0] + (np.arange(ny) + 0.5) * cell
    return np.meshgrid(xs, ys)  # [ny, nx]


def _taper(dy: np.ndarray) -> np.ndarray:
    """Smoothstep lane window: 1 inside LANE_INNER, 0 outside LANE_OUTER."""
    t = np.clip((LANE_OUTER - np.abs(dy)) / (LANE_OUTER - LANE_INNER), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_scene(cell: float = 0.05) -> Scene:
    """The four-lane labelled scene. `cell` = 0.05 m resolves the curb edge to ~7 cells
    per wheel radius, which is what makes the dilation arg-max non-trivial there."""
    XX, YY = _grid(cell)
    H = np.zeros_like(XX)
    region = np.full(XX.shape, OTHER, np.int8)

    # --- lane 0: flat -----------------------------------------------------------------
    inside0 = np.abs(YY - LANE_Y[0]) <= LANE_INNER
    region[inside0] = FLAT

    # --- lane 1: constant slope, direction rotated off +x ------------------------------
    a = np.radians(SLOPE_DIR_DEG)
    s = (XX - SLOPE_START_X) * np.cos(a) + (YY - LANE_Y[1]) * np.sin(a)
    ramp = np.tan(np.radians(SLOPE_DEG)) * np.clip(s, 0.0, 4.0)
    H += ramp * _taper(YY - LANE_Y[1])
    inside1 = np.abs(YY - LANE_Y[1]) <= LANE_INNER
    region[inside1] = FLAT
    region[inside1 & (s > 0.0) & (s < 4.0)] = SLOPE

    # --- lane 2: sharp step, edge rotated off the y-axis -------------------------------
    c = np.radians(CURB_DIR_DEG)
    d = (XX - CURB_X) * np.cos(c) + (YY - LANE_Y[2]) * np.sin(c)
    H += CURB_HEIGHT * (d > 0.0) * _taper(YY - LANE_Y[2])
    inside2 = np.abs(YY - LANE_Y[2]) <= LANE_INNER
    region[inside2] = FLAT
    region[inside2 & (np.abs(d) <= CURB_BAND)] = CURB

    # --- lane 3: isolated sub-wheel-scale rocks ----------------------------------------
    inside3 = np.abs(YY - LANE_Y[3]) <= LANE_INNER
    region[inside3] = FLAT
    win3 = _taper(YY - LANE_Y[3])
    for rx, ry, height, radius, sharp in ROCKS:
        cx, cy = rx, LANE_Y[3] + ry
        r2 = (XX - cx) ** 2 + (YY - cy) ** 2
        if sharp:
            H += height * (r2 <= radius**2) * win3
        else:
            H += height * np.exp(-r2 / (2.0 * radius**2)) * win3
        region[inside3 & (r2 <= ROCK_BAND**2)] = ROCK

    return Scene(H, _friction(XX, YY), region, cell, XLIM[0], YLIM[0])


def _friction(XX: np.ndarray, YY: np.ndarray) -> np.ndarray:
    """Non-uniform mu in BOTH axes plus one sharp low-mu patch.

    Non-uniformity is the point: friction is sampled at the pose-dependent contact point,
    so a uniform field zeroes sample_field's POSITION gradient contribution and hides any
    error in it -- the exact failure mode that once cost ~47% on this path.
    """
    mu = 0.55 + 0.20 * np.sin(0.8 * XX + 0.5) * np.cos(0.6 * YY)
    patch = (XX - 1.2) ** 2 + (YY - LANE_Y[0]) ** 2 <= 0.35**2
    mu[patch] = 0.25
    return np.clip(mu, 0.20, 0.95)


# --- rollouts ---------------------------------------------------------------------------
# B = 8: two per lane. Terrain is per-rollout in DifferentiableSimulator, so every rollout
# owns an independent [ny, nx] grad slice -- one perturbed forward pass yields B independent
# finite differences. Straight runs cover ~1.68 m in T=16 steps at dt=0.1, w=3 rad/s.

N_STEPS = 16
DT = 0.1

# (label, x0, y-lane index, yaw0, wheel omega (L, R, rear))
#
# The first rollout of each lane APPROACHES its feature from flat ground; the second STARTS
# on it. That second start is what gives level A1 -- which sees only the init settle -- a
# non-flat pose to test. With every rollout starting on flat ground A1 degenerates into a
# flat-terrain check, which is the failure mode this whole scene exists to avoid.
ROLLOUTS = (
    ("flat-straight", -0.50, 0, 0.00, (3.0, 3.0, 3.0)),
    ("flat-turn", -0.50, 0, 0.00, (2.7, 3.3, 3.0)),  # turns across the low-mu patch
    ("slope-climb", -0.50, 1, 0.00, (3.0, 3.0, 3.0)),  # approaches, then climbs
    ("slope-traverse", 1.00, 1, 0.25, (3.0, 3.0, 3.0)),  # starts ON the slope, oblique -> roll
    ("curb-headon", 0.70, 2, 0.00, (3.0, 3.0, 3.0)),  # approaches, both front wheels together
    ("curb-oblique", 1.45, 2, -0.25, (3.0, 3.0, 3.0)),  # starts AT the edge, one wheel first
    ("rock-straddle", 0.50, 3, 0.00, (3.0, 3.0, 3.0)),  # approaches, then straddles
    ("rock-turn", 0.85, 3, 0.12, (2.8, 3.2, 3.0)),  # starts ON the first rock
)


def rollouts() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Start poses [B, 3] (x, y, yaw), wheel commands [T, B, 3], and per-rollout labels."""
    poses = np.array(
        [[x0, LANE_Y[lane], yaw] for _, x0, lane, yaw, _ in ROLLOUTS],
        np.float32,
    )
    omega = np.tile(
        np.array([om for *_, om in ROLLOUTS], np.float32),
        (N_STEPS, 1, 1),
    )  # [T, B, 3], constant in time
    return poses, omega, tuple(name for name, *_ in ROLLOUTS)
