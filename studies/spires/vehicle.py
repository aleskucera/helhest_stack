"""The deployed vehicle: cylinder structuring element, bilinear contact stencil, tripod settle.

VENDORED, NOT IMPORTED -- and that was forced, not a preference. This worktree
(`study/oxford-spires`) branched before the contact-height fix landed: its own
`src/helhest/engine/envelope.py` has no `cyl_table`, no `N_YAW_BINS` and no
`wheel_half_width`, and `src/helhest/risk/` (which holds `settle_map`) does not exist here at
all. Importing the deployed tables would therefore mean importing them from ANOTHER worktree
by absolute path, which is neither reproducible off this machine nor recorded by this repo's
git history. So the four definitions this study needs are copied here verbatim, with the
source file and line of each stated below, and `verify_fold_parity.py` asserts the copies are
bit-identical to the originals when the study worktree is on the path.

Provenance (worktree `helhest_stack-study`, branch `study/adjoint-sensitivity`, HEAD 31ed976):
  cylinder_offset_table  src/helhest/engine/envelope.py:89
  cyl_table, N_YAW_BINS  src/helhest/engine/envelope.py:121-136
  yaw_bin                src/helhest/engine/step.py:609  (bins over [0, PI), +0.5 then floor)
  element_offsets        studies/bench/element.py:143    (cylinder branch, SENTINEL pad)
  blend_stencil          studies/bench/element.py:226    (_locate's cell-centre convention)
  stencil_cells          studies/bench/element.py:269
  blend_weights          studies/bench/element.py:282
  broadcast_cap          studies/bench/element.py:330
  settle_map             src/helhest/risk/settle.py:13
  RobotParams defaults   src/helhest/engine/robot.py:81-103 (identical on BOTH branches)
  DERIV_WZ/WPITCH/WROLL  studies/adjoint/harness.py:73-75

The three defects `element.py` documents (yaw binned over [0, 2*PI); K truncated to the batch
minimum instead of sentinel-padded; the single-cell `np.round` read instead of the engine's
bilinear blend) are all in their FIXED form here. Do not re-derive any of them.
"""

from __future__ import annotations

import functools
import math

import numpy as np

# --- robot (RobotParams defaults; identical on both worktree branches) ------------------------
WHEEL_RADIUS = 0.35  # [m]
WHEEL_WIDTH = 0.10  # [m] ruler-measured tread
HALF_WIDTH = 0.5 * WHEEL_WIDTH  # `wheel_half_width(rp)`
HALF_TRACK = 0.365  # [m] b
REAR_OFFSET = 0.75  # [m] l

# --- cost weights (harness.py DERIV_W*) ------------------------------------------------------
W_Z, W_PITCH, W_ROLL = 1.0, 0.7, 0.5

# --- element ---------------------------------------------------------------------------------
N_YAW_BINS = 32
PAD_CAP = -1.0e3  # [m] sentinel cap: a padded candidate that cannot win any fold
MAX_K = 8  # the deployed cylinder needs K = 5-7 at a 0.10 m cell; 8 is the checked bound


def cylinder_offset_table(
    cell_size: float, wheel_radius: float, half_width: float, yaw: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Structuring element of a CYLINDER wheel heading `yaw`, as (dy, dx, cap) offset lists.

    The rotated rectangle 2*wheel_radius (along travel) x 2*half_width (across), with
    cap = sqrt(wheel_radius^2 - u^2) - wheel_radius for u the ALONG-travel offset alone.
    """
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    env_radius = int(math.ceil(math.hypot(wheel_radius, half_width) / cell_size))
    dy_l, dx_l, cap_l = [], [], []
    for dy in range(-env_radius, env_radius + 1):
        for dx in range(-env_radius, env_radius + 1):
            wx, wy = dx * cell_size, dy * cell_size
            along = wx * cos_yaw + wy * sin_yaw
            across = -wx * sin_yaw + wy * cos_yaw
            if abs(along) <= wheel_radius and abs(across) <= half_width:
                dy_l.append(dy)
                dx_l.append(dx)
                cap_l.append(math.sqrt(wheel_radius**2 - along**2) - wheel_radius)
    return np.array(dy_l, np.int32), np.array(dx_l, np.int32), np.array(cap_l, np.float32)


@functools.lru_cache(maxsize=None)
def cyl_table(cell_size: float, wheel_radius: float, half_width: float, b: int, n_bins: int):
    """`cylinder_offset_table` for yaw bin `b` of `n_bins` over [0, PI), memoised."""
    return cylinder_offset_table(cell_size, wheel_radius, half_width, math.pi * b / n_bins)


def yaw_bins(yaw: np.ndarray, n_bins: int = N_YAW_BINS) -> np.ndarray:
    """`engine/step.py::yaw_bin`, verbatim and vectorized: floor(yaw / (PI/n) + 0.5) into [0, n).

    Bins span [0, PI), NOT [0, 2*PI): the element at yaw and yaw+PI is the same shape.
    """
    k = np.floor(np.asarray(yaw, np.float64) / (np.pi / n_bins) + 0.5).astype(np.int64)
    return ((k % n_bins) + n_bins) % n_bins


def element_offsets(
    cell: float, yaw: np.ndarray, n_bins: int = N_YAW_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-node [N, K] cylinder candidate offsets and caps at the deployed tread.

    Ragged per-bin tables are padded up to the largest K with a SENTINEL cap (`PAD_CAP`), never
    by duplicating a real offset: Clark's fold re-applies its normal approximation at every
    fold, so a duplicated candidate is folded twice and the approximation applied twice
    (measured shift up to 2.7 cm; study-repo commit 0e41164). A `PAD_CAP` candidate drives
    alpha to +inf, so the fold returns the running moments unchanged. Pad slots reuse a real
    (dy, dx) so every gather stays in bounds.
    """
    bins = yaw_bins(yaw, n_bins)
    tables = {b: cyl_table(cell, WHEEL_RADIUS, HALF_WIDTH, int(b), n_bins) for b in np.unique(bins)}
    k_max = max(len(t[0]) for t in tables.values())
    if k_max > MAX_K:
        raise ValueError(f"K={k_max} exceeds MAX_K={MAX_K} at cell={cell}")
    uniq = np.array(sorted(tables))
    t_dy = np.empty((len(uniq), k_max), np.int64)
    t_dx = np.empty((len(uniq), k_max), np.int64)
    t_cap = np.empty((len(uniq), k_max), np.float64)
    for j, b in enumerate(uniq):
        a_, b_, c_ = tables[b]
        pad = k_max - len(a_)
        t_dy[j] = np.concatenate([a_, np.repeat(a_[-1:], pad)])
        t_dx[j] = np.concatenate([b_, np.repeat(b_[-1:], pad)])
        t_cap[j] = np.concatenate([c_, np.full(pad, PAD_CAP)])
    row = np.searchsorted(uniq, bins)
    return t_dy[row], t_dx[row], t_cap[row]


def blend_stencil(
    wx: np.ndarray, wy: np.ndarray, origin_x: float, origin_y: float,
    cell: float, ny: int, nx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The FOUR cells the engine's bilinear read touches, and their weights. [N, 4] each.

    Cell-centre convention is `engine/terrain.py::_locate`'s exactly: (x - origin)/cell - 0.5,
    floor, fractional part -- `origin` is the grid's MIN CORNER and cell i's centre is at
    origin + (i + 0.5)*cell, which is the same convention `elevation_belief.py` writes its
    windows in (`self._wx = self.xmin + (jj + 0.5) * cell`). Corner order is
    (00, 10, 01, 11) = (+0,+0), (+1,+0), (+0,+1), (+1,+1) in (x, y).
    """
    fx = (np.asarray(wx, np.float64) - origin_x) / cell - 0.5
    fy = (np.asarray(wy, np.float64) - origin_y) / cell - 0.5
    xi = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
    yi = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
    tx = np.clip(fx - xi, 0.0, 1.0)
    ty = np.clip(fy - yi, 0.0, 1.0)
    ix = np.stack([xi, xi + 1, xi, xi + 1], axis=-1)
    iy = np.stack([yi, yi, yi + 1, yi + 1], axis=-1)
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=-1)
    return iy, ix, w


def stencil_cells(
    iy: np.ndarray, ix: np.ndarray, off_dy: np.ndarray, off_dx: np.ndarray, ny: int, nx: int
) -> np.ndarray:
    """Absolute [ny*nx]-flat candidate index for every stencil node. [4N, K].

    Node index is `4*n + corner`, so the flattened ordering lines up with a blend weight vector
    built as `(c[:, None] * w).ravel()`.
    """
    ry, rx = iy.ravel()[:, None], ix.ravel()[:, None]
    return np.clip(ry + off_dy, 0, ny - 1) * nx + np.clip(rx + off_dx, 0, nx - 1)


def blend_weights(c_nodes: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-node coefficient [N] and `blend_stencil` weights [N, 4] -> per-stencil-node [4N]."""
    return (np.asarray(c_nodes, np.float64)[:, None] * np.asarray(w, np.float64)).ravel()


def broadcast_cap(off_cap: np.ndarray) -> np.ndarray:
    return off_cap if off_cap.ndim == 2 else off_cap[None, :]


# --- settle ----------------------------------------------------------------------------------
def settle_map(b: float = HALF_TRACK, ell: float = REAR_OFFSET) -> np.ndarray:
    """d(z, pitch, roll) / d(env_L, env_R, env_rear), rows (z, pitch, roll), cols (L, R, rear)."""
    return np.array(
        [
            [0.5, 0.5, 0.0],
            [-0.5 / ell, -0.5 / ell, 1.0 / ell],
            [1.0 / (2.0 * b), -1.0 / (2.0 * b), 0.0],
        ]
    )


def settle_weights() -> np.ndarray:
    """Per-wheel constant coefficient of J_settle = sum_t w . (z, pitch, roll); order (L, R, rear).

    J_settle is LINEAR and TIME-INDEPENDENT in the three contact heights because the tripod
    settle is (harness.py `_terms_kernel` term 1; clark.py declared approximation (b)):
        (z, pitch, roll) = settle_map @ (env + wheel_radius),  J_t = w . (z, pitch, roll).
    settle_map @ 1 = (1, 0, 0), so the constant is n_steps * W_Z * wheel_radius (see
    `SETTLE_CONST_PER_STEP`).
    """
    return np.array([W_Z, W_PITCH, W_ROLL]) @ settle_map()


SETTLE_CONST_PER_STEP = W_Z * WHEEL_RADIUS

WHEEL_BODY_XY = np.array([[0.0, HALF_TRACK], [0.0, -HALF_TRACK], [-REAR_OFFSET, 0.0]])
