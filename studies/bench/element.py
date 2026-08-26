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

CONTACT-HEIGHT DEFECTS FIXED 2026-08-26
---------------------------------------
Three separate mismatches made the modelled contact height differ from the height the engine
actually rests a wheel on, by a median of 2.84 cm. Two were in `element_offsets` (yaw binned over
[0, 2*pi) instead of the engine's [0, PI); K truncated to the batch minimum instead of padded)
and are fixed there. The third was the single-cell read, fixed by `blend_stencil` below.

Corrected, the modelled height matches the engine to 0.06 cm, and both benchmarks that feed the
paper's SectionV moved from undemonstrated to decisive against every baseline (`clark.py`:
clark_cvar vs step 32/52 p = 0.126 -> 42/47 p = 2.5e-08; `realistic_sigma.py` hybrid/all, which
had `passed = False` on its own pre-registered criteria, 35/56 p = 0.081 -> 45/51 p = 1.8e-08).
Pre-registrations and artifacts live in the clark_paper repo under
`theory/notes/measurements/clark_paper_benchmark_*` and `clark_realistic_sigma_*`.

ALL CONVERTED as of 0e41164 -- these six called `clark._footprint_cells` directly and now read
`blend_stencil`/`stencil_cells`:

    clark_conv.py, clark_fast.py, clark_full.py, clark_grad.py, clark_hinge.py,
    clark_hinge_fast.py

They each contract nodes differently, so the conversion was not uniform; the guard was to
establish the relationships they are documented to reproduce on the UNCONVERTED code and require
them to survive (`clark_blend_invariants.py`, in the clark_paper repo -- all hold). Note that the
two `fold_weights`-based relationships (conv, hinge_fast) were exact before and are now ~5e-7
relative: the sentinel pad that replaced the K truncation duplicates a real offset, so `a`
collapses to 0 and the two implementations take their degenerate branch by slightly different
arithmetic. `clark.gate2_clark_vs_mc` is single-cell still, but deliberately: it tests the fold,
not the contact (see its docstring).
"""

from __future__ import annotations

import math

import numpy as np

from helhest.engine import RobotParams
from helhest.engine.envelope import cyl_table as _cyl_table
from helhest.engine.envelope import cylinder_offset_table
from helhest.engine.envelope import N_YAW_BINS
from helhest.engine.envelope import wheel_half_width
from helhest.engine.envelope import wheel_offset_table

# [m] half the ruler-measured tread, DERIVED from the robot rather than restated. It was a
# literal 0.05 beside `RobotParams.wheel_width = 0.10` -- the same measurement stored twice, and
# agreeing only until someone widened one of them (the half-width is also the lateral
# safety-margin dial, so widening it is a thing people do).
WHEEL_HALF_WIDTH = wheel_half_width(RobotParams())
PAD_CAP = -1.0e3  # [m] sentinel cap: a padded candidate that cannot win any fold (see below)


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


def crowned_offsets(
    cell: float, radius: float, half_width: float, crown_radius: float, yaw: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The wheel as a TIRE: a surface of revolution whose tread is crowned across its width.

    `cylinder_offsets` caps by the along-travel coordinate ALONE, so every cell of an
    across-tread column carries the identical cap and the envelope's max over them is an EXACT
    tie wherever the ground is level across the tread. A real tire is curved across its tread
    too, which breaks that tie by construction rather than by relying on terrain roughness.

    Model the tread profile as a circular arc of radius `crown_radius` (R_c). The wheel's radius
    at axial offset `across` is then

        rho(across) = radius - (R_c - sqrt(R_c^2 - across^2))

    and the cap -- the height of the tire surface above its own lowest point, negated -- is

        cap(along, across) = sqrt(rho(across)^2 - along^2) - radius.

    ONE parameter spans both elements the study has used, which is why it is worth carrying as a
    parameter rather than a third element:

        R_c = radius    ->  rho = sqrt(radius^2 - across^2), cap = sqrt(radius^2 - across^2
                            - along^2) - radius: the SPHERE, exactly.
        R_c -> infinity ->  rho = radius: the flat-tread CYLINDER, exactly.

    A measured tire sits between them, and the across-tread cap step it supplies is
    approximately half_width^2 / (2 R_c) -- the quantity that sets the contact validity radius.
    """
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    env_radius = int(math.ceil(math.hypot(radius, half_width) / cell))
    dy_l, dx_l, cap_l = [], [], []
    for dy in range(-env_radius, env_radius + 1):
        for dx in range(-env_radius, env_radius + 1):
            wx, wy = dx * cell, dy * cell
            along = wx * cos_y + wy * sin_y
            across = -wx * sin_y + wy * cos_y
            if abs(across) > half_width:
                continue
            if math.isinf(crown_radius):
                rho = radius
            else:
                if abs(across) > crown_radius:
                    continue
                rho = radius - (crown_radius - math.sqrt(crown_radius**2 - across**2))
            if rho <= 0.0 or abs(along) > rho:
                continue
            dy_l.append(dy)
            dx_l.append(dx)
            cap_l.append(math.sqrt(rho**2 - along**2) - radius)
    return np.array(dy_l, np.int64), np.array(dx_l, np.int64), np.array(cap_l, np.float64)


def element_offsets(
    element: str, cell: float, rp: RobotParams, yaw: np.ndarray, n_bins: int = N_YAW_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The structuring element's candidate offsets, at the `wheel_offset_table` choke point every
    sphere-only benchmark shares.

    sphere:   off_dy/off_dx/off_cap are [K], shared by every node (yaw-invariant); `yaw` unused.
    cylinder: off_dy/off_dx/off_cap are [N, K], one row per entry of `yaw` (radians, any shape --
              raveled to [N]), quantized to `n_bins` heading bins.

    TWO DEFECTS FIXED HERE (2026-08-26). Both made the modelled contact height differ from the
    one the engine actually rests on; together with the single-cell read they accounted for a
    2.84 cm median mismatch, and correcting all three moved the benchmark's decision result from
    undemonstrated to p < 1e-7 (clark_paper_benchmark_PREREG.md, clark_realistic_sigma_PREREG.md
    in the clark_paper repo). Neither was a modelling choice; both were oversights.

      1. YAW BINNING. This binned over [0, 2*pi) in `n_bins`, while the engine's stack spans
         [0, PI) (`engine/step.py::yaw_bin`) because a capsule at yaw and yaw+PI is the same
         shape. That was half the engine's angular resolution AND misaligned bin centres. The
         engine's exact expression is used now, and the table comes from the engine's own
         `cylinder_offset_table` rather than this module's re-derivation, so the two cannot
         drift apart again.

      2. K TRUNCATION. The per-bin tables are yaw-ragged (a diagonal heading clips a few more
         cells than an axis-aligned one). This truncated every node to the SMALLEST K in the
         batch, so a plan mixing headings dropped 2 of 7 cells from every axis-aligned node --
         and dropped them from one END of the (dy, dx) scan, losing the footprint's front rather
         than trimming it symmetrically. Now the rows are PADDED up to the largest K instead.

    The pad is a SENTINEL, not a duplicate. Duplicating a real candidate looks free because a
    true maximum is idempotent -- max(M, x, x) = max(M, x) -- but Clark's fold is not the true
    maximum: it re-applies its normal approximation at every fold, so a duplicated candidate is
    folded twice and the approximation applied twice, shifting the mean by up to 2.7 cm
    (measured). A cap of `PAD_CAP` drives alpha to +inf instead, so the fold returns the running
    mean, variance and covariance row unchanged -- exact to machine precision, and independent
    of how many pads are added. Pad slots reuse a real (dy, dx) so every gather stays in bounds;
    they take zero fold weight (verified: max weight 0.000e+00).
    """
    if element == "sphere":
        env_radius = int(np.ceil(rp.wheel_radius / cell))
        off_dy, off_dx, off_cap = wheel_offset_table(env_radius, cell, rp.wheel_radius)
        return np.asarray(off_dy, np.int64), np.asarray(off_dx, np.int64), np.asarray(off_cap)
    if element == "cylinder":
        yaw = np.asarray(yaw).ravel()
        # `engine/step.py::yaw_bin`, verbatim: floor(yaw / (PI/n) + 0.5), wrapped into [0, n).
        k = np.floor(yaw / (np.pi / n_bins) + 0.5).astype(np.int64)
        bins = ((k % n_bins) + n_bins) % n_bins
        tables = {b: _cyl_table(cell, rp.wheel_radius, WHEEL_HALF_WIDTH, int(b), n_bins)
                  for b in np.unique(bins)}
        k_max = max(len(t[0]) for t in tables.values())
        # Build one padded row PER DISTINCT BIN, then gather. The per-node Python loop this
        # replaces cost ~14% of the estimator's runtime once the bilinear blend quadrupled the
        # node count; distinct bins are at most `n_bins` and typically ~19 on a real plan.
        uniq = np.array(sorted(tables))
        n_u = len(uniq)
        t_dy = np.empty((n_u, k_max), np.int64)
        t_dx = np.empty((n_u, k_max), np.int64)
        t_cap = np.empty((n_u, k_max), np.float64)
        for j, b in enumerate(uniq):
            a_, b_, c_ = tables[b]
            pad = k_max - len(a_)
            t_dy[j] = np.concatenate([a_, np.repeat(a_[-1:], pad)])
            t_dx[j] = np.concatenate([b_, np.repeat(b_[-1:], pad)])
            t_cap[j] = np.concatenate([c_, np.full(pad, PAD_CAP)])
        row = np.searchsorted(uniq, bins)
        return t_dy[row], t_dx[row], t_cap[row]
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


def blend_stencil(
    wx: np.ndarray,
    wy: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    ny: int,
    nx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The FOUR cells the engine's bilinear read touches, and their weights. [N, 4] each.

    THE THIRD DEFECT (fixed 2026-08-26). `_footprint_cells` snaps a wheel to ONE cell with
    `np.round`, so the modelled contact height is a single dilated cell's value. The engine does
    not do that: `engine/envelope.py` dilates the elevation at EVERY cell, then `sample_field`
    reads that dilated field with a BILINEAR sample at the wheel's exact position
    (`engine/step.py`). The height a wheel rests on is therefore a weighted blend of four
    neighbouring dilated cells.

    The difference is not a small bias. It is SCATTER -- 2.73 cm of it on the belief map with no
    noise at all (mean only +0.34 cm) -- and it makes the modelled height 3.3x jumpier in time
    than the engine's, because a discrete maximum steps abruptly as the winning cell changes
    while a bilinear sample slides continuously. Any cost with an acceleration term divides that
    jumpiness by dt^4 = 1e-4.

    Cell-centre convention is `_locate`'s, exactly: (x - origin)/cell - 0.5, floor, fractional
    part. Corner order is (00, 10, 01, 11) = (+0,+0), (+1,+0), (+0,+1), (+1,+1) in (x, y).

    Callers must index the returned (iy, ix) DIRECTLY and never route them back through
    `_footprint_cells`: `np.round` is banker's rounding, so a cell centre at index+0.5 lands on
    index for even indices and index+1 for odd ones, silently displacing half the stencil.
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

    `iy`/`ix` are `blend_stencil`'s [N, 4]; node index is `4*n + corner`, so the flattened
    ordering lines up with a blend weight vector built as `(c[:, None] * w).ravel()`.
    `off_dy`/`off_dx` must already be per-node for the 4N nodes (build them on a yaw array
    repeated 4x, e.g. `np.repeat(yaw_per_node, 4)`), or [K] for the yaw-invariant sphere."""
    ry, rx = iy.ravel()[:, None], ix.ravel()[:, None]
    return np.clip(ry + off_dy, 0, ny - 1) * nx + np.clip(rx + off_dx, 0, nx - 1)


def blend_weights(c_nodes: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Fold a per-node coefficient vector [N] and `blend_stencil`'s weights [N, 4] into the
    effective per-stencil-node coefficients [4N], matching `stencil_cells`' 4*n + corner order.

    A weighted sum of maxima needs each maximum's mean and the covariance BETWEEN maxima, which
    the Clark machinery already produces -- so the blend costs 4x the nodes and no new
    approximation."""
    return (np.asarray(c_nodes, np.float64)[:, None] * np.asarray(w, np.float64)).ravel()


def dedupe_nodes(
    cell_flat: np.ndarray, cap_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the DISTINCT nodes, and the map back. `(first, back)` with
    `first` selecting one representative each and `x[first][back] == x` for any per-node array.

    A max-node is fully determined by its candidate cells and their caps. After the bilinear
    blend many nodes repeat: consecutive timesteps' 2x2 stencils overlap and the yaw bins are
    5.6 deg wide, so ~42% of a plan's 480 blended nodes are exact duplicates (measured
    480 -> 279). The fold is the dominant cost, so folding only the distinct set is a direct
    saving, and exact -- these are the same random variable, not merely a close one."""
    key = np.concatenate(
        [np.asarray(cell_flat, np.int64),
         # float64 first: caps arrive as float32 from the sphere table, and a bitwise view
         # needs a fixed 8-byte element. Exact equality is the right test here -- two nodes
         # share a table row or they do not.
         np.ascontiguousarray(
             np.broadcast_to(cap_rows, cell_flat.shape), dtype=np.float64
         ).view(np.int64)],
        axis=1,
    )
    _u, first, back = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return first, back.ravel()


def blend_matrix(w: np.ndarray) -> np.ndarray:
    """`blend_stencil`'s weights [N, 4] as the dense contraction W [N, 4N], `W[i, 4i:4i+4] = w[i]`.

    For callers that need the CONTRACTED moments rather than a single coefficient vector: the
    contact means are `W @ mean_nodes` and their covariance is `W @ Sigma_nodes @ W.T`. Downstream
    code that was written against one node per (wheel, timestep) then works unchanged."""
    w = np.asarray(w, np.float64)
    n = w.shape[0]
    W = np.zeros((n, 4 * n))
    W[np.repeat(np.arange(n), 4), np.arange(4 * n)] = w.ravel()
    return W


def broadcast_cap(off_cap: np.ndarray) -> np.ndarray:
    """`off_cap` ready to add to a [N, K] candidate-means array: the sphere's shared [K] table
    broadcasts over nodes via a leading axis; the cylinder's per-node [N, K] table already is
    node-aligned and passes through unchanged."""
    return off_cap if off_cap.ndim == 2 else off_cap[None, :]
