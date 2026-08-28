"""E0 and E1 of the frozen risk-calibration prereg, DESIGN SITE ONLY.

Protocol: clark_paper/PREREG_risk_calibration.md (frozen 2026-08-28). This module implements
the two experiments on the design site (keble-college-*) and nothing else. The held-out set
(virgin-*) is locked by pre-registration and this file REFUSES to open it: every path that
reaches the filesystem goes through `_forbid_held_out`, which raises on any path containing
"virgin", and the site filter defaults to the design site.

WHAT IS COMPUTED, per design window
-----------------------------------
1. TRAJECTORY. The window's ground-truth pose track (`gt_poses`, 4x4) resampled to a fixed
   `RESAMPLE_M` = 0.10 m along-track spacing -- ONE CELL, so consecutive samples never skip a
   cell of the 0.10 m map and the sum over steps is a well-defined line integral of the tilt
   cost at the map's own resolution. Each sample is (x, y, yaw); z and the out-of-plane
   attitude of the handheld rig are discarded, because the vehicle's own settle supplies them.

2. CONTACT. The deployed measured-tread cylinder (`vehicle.py`), one candidate table per yaw
   bin over [0, PI), sentinel-padded to the ragged maximum. Support height per wheel is
   max_k (h_c(k) + kappa_k) over the footprint, read at the engine's BILINEAR blend of four
   cells rather than one rounded cell -- so each (wheel, step) contributes four max-nodes.

3. COST. J = sum_t (1.0 z_t + 0.7 pitch_t + 0.5 roll_t), the `settle` functional of
   studies/adjoint/harness.py, with (z, pitch, roll) the affine tripod map of the three
   contact heights (`vehicle.settle_map`). Identical for every arm and for the truth.

4. TRUTH. The same functional on the TLS survey MAX raster
   (`ground_truth_map/<site>/tls_max_raster.npz`, built by `tls_raster.py`), exact max, no
   Monte Carlo. Prereg amendment A1(3) names that raster specifically: the max-per-cell
   convention is the one the belief's own mu layer uses and the one the drive-on-max contact
   model assumes. The `tls_mean_raster.npz` sitting next to it is NEVER truth here; it is read
   for one purpose only, the overhang QC flag below.

5. OVERHANG QC FLAG (prereg amendment A1(2), geometry only -- no belief, no arm, no score).
   A window is flagged iff max(tls_max - tls_mean) > `OVERHANG_M` = 1.0 m over the cells
   within one cell of its resampled track (the cells containing the track samples, dilated by
   a 3x3 neighbourhood; only cells where both rasters are finite). Both the belief and the TLS
   max raster alias overhanging structure -- an archway, a gate -- onto the walkable ground
   cell beneath it, and the two rasters disagree by metres exactly there. Flags are computed
   for every window before any scoring; the pre-registered criteria are evaluated on unflagged
   windows and the flagged ones are reported separately and unconditionally.

6. DISTINCT TRACKS (prereg amendment A1(3)). `keble-college-02-default` and
   `keble-college-02-unclamped` are the SAME 15 trajectories under two `max_variance` settings
   of `window_runner2.py`, so the 44 design windows are 29 distinct tracks. Everything is
   COMPUTED for all 44 -- the pair is a free clamp ablation -- but every headline statistic is
   taken over the 29 distinct tracks, with the `-unclamped` half reported separately as the
   ablation it is.

THE BELIEF, AND THE ONE CHOICE THAT MOVES EVERY NUMBER
------------------------------------------------------
The marginal per cell is the window's PUBLISHED FUSED READOUT (`mu`, `sigma`) -- not `raw_sd`.
Three reasons, and the choice is stated here because it changes every number below:

  (i)  the prereg names it: "Belief: the windows' published (mu, sigma)";
  (ii) `mu` is itself the fused readout (the neighbour mixture of III-D), so pairing it with
       `raw_sd` would take the mean from one distribution and the sd from another;
  (iii) `sigma` is what a consumer of this elevation map actually reads.

`raw_sd` and `meas_sd` are not discarded -- they are used to SPLIT that same sigma into parts
with different spatial coherence, which is what the fold needs and what a single marginal sd
cannot supply. The split is exact by construction (the three parts sum to sigma^2 in every
cell), so the model reproduces the published marginal in every cell and only adds structure:

    var_ind(c) = min(meas_sd^2, sigma^2)                    independent per cell
    var_gcm(c) = clip(raw_sd^2 - meas_sd^2, 0, sigma^2 - var_ind)   globally coherent
    var_loc(c) = sigma^2 - var_ind - var_gcm                 coherent over the fusion kernel

  * var_ind is the pure Kalman measurement variance (`elevation_belief.var_meas`): each cell's
    own range noise, independent between cells by construction of the sensor model.
  * var_gcm is exactly what the eq.-20 motion update added on top of it, i.e. the q_z
    random-walk accumulation of the VILENS drift calibrated on this same design site
    (`calibrate_drift.py`; rates frozen in `elevation_belief.DriftRates`). A z drift is one
    scalar error of the whole window's anchor, so it is COMMON-MODE across every cell: a
    rank-1 term with correlation 1, and it therefore does NOT average out along a trajectory.
    This is the drift-informed part of the model.
  * var_loc is whatever the III-D neighbour mixture added at readout. That mixture is driven
    by the horizontal pose uncertainty var_x/var_y, which the SAME calibration produces from
    q_x, q_y and q_yaw, and the window stores its realized kernel widths as `sx`, `sy`. Two
    cells' mixtures draw on the same terrain only while their kernels overlap, so the
    coherence of this part is modelled by the kernel's own normalized autocorrelation --
    a Gaussian of sd sqrt(2)*s per axis. (Same construction as `clark.py::rho1_table`, which
    builds its correlation from the noise generator's own kernel rather than assuming one.)

    Cov(a, b) = delta_ab var_ind(a)
              + sqrt(var_gcm(a) var_gcm(b))
              + rho(a, b) sqrt(var_loc(a) var_loc(b)),
    rho = exp(-(dx cell)^2 / (4 sx^2)) exp(-(dy cell)^2 / (4 sy^2)).

  Each of the three terms is PSD (a nonnegative diagonal, a rank-1 outer product, and a
  Gaussian kernel conjugated by a diagonal), so the sum is PSD with no truncation anywhere --
  unlike clark.py's declared approximation (d), this study needs no finite-support cutoff.

  A KNOWN LIMIT, stated because the model cannot express it. The data audit
  (clark_paper/theory/RISK_CAL_DATA_AUDIT.md section 4) measured a consistent -0.11 m median
  offset between mu and the TLS raster on large flat well-observed areas. The common-mode term
  above is a zero-mean rank-1 VARIANCE: it says a whole-window vertical offset of that scale is
  a plausible draw, which is why the trajectory cost's sd does not shrink with the number of
  steps. It does not say the offset has a sign. A persistent bias therefore shows up in E1 as a
  consistent shift of z, not as inflated sd, and per amendment A1(3) it is NOT corrected here.

  Cells never touched by a scan (`count` == 0) have no Kalman moments at all: `raw_sd` and
  `meas_sd` are NaN there while `mu`/`sigma` may still be finite, because the mixture can read
  them off observed neighbours. For such a cell the split degenerates to var_ind = var_gcm = 0,
  var_loc = sigma^2: all of its uncertainty came from the mixture, which is exactly true.
  Candidates with NO finite (mu, sigma) -- or no TLS truth -- cannot be scored at all, and the
  policy is to drop the whole STEP (all 3 wheels, all 4 corners, all K candidates) from both
  the belief cost and the truth cost, so the two always sum over the identical step set. Per
  window the retained fraction is recorded, and a window enters the metrics only if at least
  `MIN_RETAINED` of its steps survive.

ARMS (prereg section E1; identical inputs, identical element, identical cost)
  clark       full covariance through the fold, node weights from the fold trace
  clark-diag  same fold with a DIAGONAL Sigma (ablation: what the off-diagonal buys)
  fosm        max of means; the winner's one-hot weights propagated through the SAME full Sigma
  mean-map    point prediction only (max of means), scored on error, no variance

Var[J] for every arm is w^T Sigma w over the candidate cells, where w is that arm's effective
per-candidate weight -- for clark, the product of the fold's own phi / (1-phi) factors, which
is algebraically identical to what `clark.py::clark_quad_form` contracts (see
`verify_fold_parity.py`, which checks it against clark.py to machine precision).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .elevation_belief import DriftRates
from .tls_raster import load_or_build
from .vehicle import blend_stencil
from .vehicle import blend_weights
from .vehicle import element_offsets
from .vehicle import SETTLE_CONST_PER_STEP
from .vehicle import settle_weights
from .vehicle import stencil_cells
from .vehicle import WHEEL_BODY_XY

# --- the pre-registration's hard boundary ----------------------------------------------------
DESIGN_SITE = "keble-college-*"  # the ONLY site this module will open
DESIGN_TLS_SITE = "keble-college"
HELD_OUT_TOKEN = "virgin"
OUT_ROOT = Path("/home/kuceral4/data/oxford_spires/out")

RESAMPLE_M = 0.10  # along-track resample spacing = one map cell
OVERHANG_M = 1.0  # [m] prereg A1(2): max(tls_max - tls_mean) above this flags a window
DUPLICATE_DIR = "keble-college-02-unclamped"  # same 15 tracks as -02-default (audit A1)
MIN_RETAINED = 0.50  # a window needs this fraction of its steps scoreable to be reported
SIGMA_FLOOR = 1.0e-4  # [m] a cell with sigma below this is treated as unusable (gates.py)
N_MC_DEFAULT = 20_000  # prereg E0: >= 20,000 draws
MC_CHUNK = 250
MC_SEED = 20260828


class HeldOutViolation(RuntimeError):
    """Raised on any attempt to touch the pre-registered held-out set."""


def _forbid_held_out(path) -> None:
    """The single choke point. Every filesystem path in this module passes through here."""
    if HELD_OUT_TOKEN in str(path).lower():
        raise HeldOutViolation(
            f"{path!r} is in the pre-registered HELD-OUT set (virgin-*). "
            "PREREG_risk_calibration.md locks it until the design-site pipeline is frozen; "
            "this module implements the design site only."
        )


def window_dirs(site: str = DESIGN_SITE, root: Path = OUT_ROOT) -> list[Path]:
    """Window directories for `site`. Defaults to the DESIGN site; refuses the held-out set."""
    _forbid_held_out(site)
    _forbid_held_out(root)
    dirs = sorted(d for d in root.glob(site) if d.is_dir())
    for d in dirs:
        _forbid_held_out(d)
    if not dirs:
        raise FileNotFoundError(f"no window directories matched {site!r} under {root}")
    return dirs


def window_files(site: str = DESIGN_SITE, root: Path = OUT_ROOT) -> list[Path]:
    out: list[Path] = []
    for d in window_dirs(site, root):
        for f in sorted(d.glob("window_*.npz")):
            _forbid_held_out(f)
            out.append(f)
    return out


def tls_site_of(window_file: Path) -> str:
    _forbid_held_out(window_file)
    if not window_file.parent.name.startswith("keble-college-"):
        raise HeldOutViolation(
            f"{window_file.parent.name!r} is not a design-site window directory; "
            "this module opens the design site only."
        )
    return DESIGN_TLS_SITE


# --- one window ------------------------------------------------------------------------------
@dataclass
class Window:
    path: Path
    mu: np.ndarray
    sigma: np.ndarray
    raw_sd: np.ndarray
    meas_sd: np.ndarray
    count: np.ndarray
    x0: float
    y0: float
    cell: float
    sx: float
    sy: float
    gt_poses: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.mu.shape


def load_window(path: Path) -> Window:
    _forbid_held_out(path)
    d = np.load(path)
    return Window(
        path=path,
        mu=d["mu"].astype(np.float64),
        sigma=d["sigma"].astype(np.float64),
        raw_sd=d["raw_sd"].astype(np.float64),
        meas_sd=d["meas_sd"].astype(np.float64),
        count=d["count"],
        x0=float(d["xmin"]),
        y0=float(d["ymin"]),
        cell=float(d["cell"]),
        sx=float(d["sx"]),
        sy=float(d["sy"]),
        gt_poses=d["gt_poses"],
    )


_TLS_CACHE: dict[tuple[str, str], dict] = {}


def tls_layer(site: str, layer: str) -> dict:
    """The site's TLS raster. `layer` is "max" (the TRUTH, prereg A1(3)) or "mean" (QC only)."""
    _forbid_held_out(site)
    key = (site, layer)
    if key in _TLS_CACHE:
        return _TLS_CACHE[key]
    if layer == "max":
        d = load_or_build(site)  # ground_truth_map/<site>/tls_max_raster.npz
    elif layer == "mean":
        p = Path(f"/home/kuceral4/data/oxford_spires/ground_truth_map/{site}/tls_mean_raster.npz")
        _forbid_held_out(p)
        z = np.load(p)
        d = {"H": z["H"], "x0": float(z["x0"]), "y0": float(z["y0"]), "cell": float(z["cell"])}
    else:
        raise ValueError(layer)
    _TLS_CACHE[key] = d
    return d


def _tls_index(win: Window, tls: dict):
    """Window cell centres -> the TLS cell that CONTAINS them, and the in-range mask.

    Both lattices are 0.10 m but their origins differ by a sub-cell amount, so this is the
    containing-cell map `gates.py` uses. No interpolation: the TLS layers are per-cell
    reductions and interpolating them would invent surfaces between survey cells.
    """
    tny, tnx = tls["H"].shape
    ny, nx = win.shape
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    wx = win.x0 + (jj + 0.5) * win.cell
    wy = win.y0 + (ii + 0.5) * win.cell
    tj = np.floor((wx - tls["x0"]) / tls["cell"]).astype(np.int64)
    ti = np.floor((wy - tls["y0"]) / tls["cell"]).astype(np.int64)
    inside = (tj >= 0) & (tj < tnx) & (ti >= 0) & (ti < tny)
    return ti, tj, inside


def tls_on_window(win: Window, site: str, layer: str = "max") -> np.ndarray:
    """A TLS layer resampled onto the window's own lattice (NaN outside coverage)."""
    tls = tls_layer(site, layer)
    ti, tj, inside = _tls_index(win, tls)
    out = np.full(win.shape, np.nan)
    out[inside] = tls["H"][ti[inside], tj[inside]]
    return out


def overhang_flag(
    win: Window, track: np.ndarray, tls_max: np.ndarray, tls_mean: np.ndarray
) -> tuple[bool, float, int]:
    """Prereg A1(2): flag iff max(tls_max - tls_mean) > OVERHANG_M near the resampled track.

    "Near" is the cells containing the resampled track samples DILATED BY ONE CELL (a 3x3
    Chebyshev neighbourhood), evaluated only where both rasters are finite. Geometry and the
    two truth rasters only -- no belief, no arm, no score enters this.
    """
    ny, nx = win.shape
    j0 = np.floor((track[:, 0] - win.x0) / win.cell).astype(np.int64)
    i0 = np.floor((track[:, 1] - win.y0) / win.cell).astype(np.int64)
    di, dj = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij")
    ii = (i0[:, None] + di.ravel()[None, :]).ravel()
    jj = (j0[:, None] + dj.ravel()[None, :]).ravel()
    keep = (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
    flat = np.unique(ii[keep] * nx + jj[keep])
    d = tls_max.ravel()[flat] - tls_mean.ravel()[flat]
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return False, float("nan"), 0
    return bool(d.max() > OVERHANG_M), float(d.max()), int(len(d))


# --- trajectory ------------------------------------------------------------------------------
def resample_track(gt_poses: np.ndarray, spacing: float = RESAMPLE_M) -> np.ndarray:
    """(x, y, yaw) resampled to `spacing` along the XY arc length of the GT pose track. [T, 3].

    Yaw is unwrapped before interpolation so a wrap does not interpolate the long way round;
    the returned yaw is NOT re-wrapped, because `vehicle.yaw_bins` wraps into [0, PI) itself.
    """
    p = gt_poses[:, :3, 3]
    yaw = np.unwrap(np.arctan2(gt_poses[:, 1, 0], gt_poses[:, 0, 0]))
    seg = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], np.diff(s) > 0])  # strictly increasing knots for np.interp
    s, p, yaw = s[keep], p[keep], yaw[keep]
    if s[-1] < spacing:
        return np.zeros((0, 3))
    sq = np.arange(0.0, s[-1], spacing)
    return np.stack(
        [np.interp(sq, s, p[:, 0]), np.interp(sq, s, p[:, 1]), np.interp(sq, s, yaw)], axis=1
    )


def build_nodes(win: Window, track: np.ndarray):
    """Stencil nodes for one track. Returns (cell_flat [4*3T, K], caps [4*3T, K], c_eff [4*3T]).

    Node order is wheel-major then time (`n = wheel*T + t`), and each node becomes four stencil
    nodes at `4n + corner` -- the ordering `blend_weights` and `stencil_cells` assume.
    """
    ny, nx = win.shape
    t_steps = len(track)
    x, y, yaw = track[:, 0], track[:, 1], track[:, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    wx = np.stack([x + WHEEL_BODY_XY[w, 0] * c - WHEEL_BODY_XY[w, 1] * s for w in range(3)])
    wy = np.stack([y + WHEEL_BODY_XY[w, 0] * s + WHEEL_BODY_XY[w, 1] * c for w in range(3)])
    iy, ix, w_blend = blend_stencil(wx.ravel(), wy.ravel(), win.x0, win.y0, win.cell, ny, nx)
    off_dy, off_dx, off_cap = element_offsets(win.cell, np.repeat(np.tile(yaw, 3), 4))
    cell_flat = stencil_cells(iy, ix, off_dy, off_dx, ny, nx)
    c_eff = blend_weights(np.repeat(settle_weights(), t_steps), w_blend)
    return cell_flat, off_cap, c_eff


# --- the belief ------------------------------------------------------------------------------
def cell_variance_split(win: Window) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(var_ind, var_gcm, var_loc), flat over the window grid. They sum to sigma^2 exactly.

    See the module docstring. NaN Kalman moments (an untouched cell whose mu came from the
    neighbour mixture alone) contribute zero to the first two parts, so all of that cell's
    variance is mixture variance.
    """
    var_tot = np.nan_to_num(win.sigma.ravel() ** 2, nan=0.0)
    meas_v = np.nan_to_num(win.meas_sd.ravel() ** 2, nan=0.0)
    raw_v = np.nan_to_num(win.raw_sd.ravel() ** 2, nan=0.0)
    var_ind = np.minimum(meas_v, var_tot)
    var_gcm = np.clip(raw_v - meas_v, 0.0, np.maximum(var_tot - var_ind, 0.0))
    var_loc = np.maximum(var_tot - var_ind - var_gcm, 0.0)
    return var_ind, var_gcm, var_loc


def cell_covariance(
    u_cells: np.ndarray, win: Window, var_ind, var_gcm, var_loc
) -> np.ndarray:
    """Dense [|U|, |U|] covariance of the belief over the candidate cells `u_cells`."""
    ny, nx = win.shape
    iy, ix = u_cells // nx, u_cells % nx
    sx = max(win.sx, 1.0e-3)
    sy = max(win.sy, 1.0e-3)
    dx = (ix[:, None] - ix[None, :]) * win.cell
    dy = (iy[:, None] - iy[None, :]) * win.cell
    rho = np.exp(-(dx**2) / (4.0 * sx**2) - (dy**2) / (4.0 * sy**2))
    g = np.sqrt(var_gcm[u_cells])
    s_loc = np.sqrt(var_loc[u_cells])
    C = np.outer(g, g) + rho * np.outer(s_loc, s_loc)
    C[np.diag_indices_from(C)] += var_ind[u_cells]
    return C


# --- the fold --------------------------------------------------------------------------------
def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26, max abs error 1.5e-7. Byte-for-byte the rational
    approximation `studies/bench/clark.py::_erf` uses, deliberately -- so a parity check
    against clark.py isolates the ALGEBRA and not two different erf implementations."""
    a1, a2, a3, a4, a5, p = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911,
    )
    sgn = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sgn * y


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _phi_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def clark_fold(means: np.ndarray, cov_self: np.ndarray):
    """Clark's moment-matched max over each row's K candidates. means [N, K], cov [N, K, K].

    Returns (mean[N], var[N], lam[N, K]) where `lam` is the fold's EFFECTIVE LINEAR WEIGHT of
    each candidate in the ORIGINAL column order: cov(max_n, Y) = sum_k lam[n, k] cov(X_nk, Y)
    for any Y. That identity is what makes a universe basis unnecessary here -- the [N, K, |U|]
    tracking array `clark.py::clark_build` carries is exactly `lam` contracted against
    Sigma[:, U], and `clark.py::clark_quad_form` contracts it again against the cost weights.
    Carrying `lam` instead costs O(N K) memory instead of O(N K |U|) and is algebraically the
    same object (checked to machine precision by `verify_fold_parity.py`).

    Candidates are folded in DESCENDING order of mean, and every mean is shifted by the node's
    largest mean before folding and shifted back afterwards. The fold is exactly equivariant
    under a common shift (alpha depends only on mean differences and variances), so the shift
    changes no result and removes the cancellation of ~6 m absolute elevations against ~0.2 m
    of spread.
    """
    n, k = means.shape
    order = np.argsort(-means, axis=1)
    m_s = np.take_along_axis(means, order, axis=1)
    shift = m_s[:, :1].copy()
    m_s = m_s - shift
    cov_s = np.take_along_axis(cov_self, order[:, :, None], axis=1)
    cov_s = np.take_along_axis(cov_s, order[:, None, :], axis=2)
    v_s = np.diagonal(cov_s, axis1=1, axis2=2).copy()

    mean_run = m_s[:, 0].copy()
    var_run = v_s[:, 0].copy()
    lam = np.zeros((n, k))
    lam[:, 0] = 1.0
    cov_run = cov_s[:, 0, :].copy()  # cov(running max, each raw candidate)
    for i in range(1, k):
        m2, v2 = m_s[:, i], v_s[:, i]
        c12 = cov_run[:, i]
        a = np.sqrt(np.maximum(var_run + v2 - 2.0 * c12, 0.0))
        degenerate = a < 1e-9
        safe_a = np.where(degenerate, 1.0, a)
        alpha = np.where(degenerate, np.sign(mean_run - m2) * 1.0e6, (mean_run - m2) / safe_a)
        phi_a = _norm_cdf(alpha)
        phi_na = 1.0 - phi_a
        pdf_a = _phi_pdf(alpha)
        new_mean = mean_run * phi_a + m2 * phi_na + a * pdf_a
        new_ex2 = (
            (mean_run**2 + var_run) * phi_a + (m2**2 + v2) * phi_na
            + (mean_run + m2) * a * pdf_a
        )
        var_run = np.maximum(new_ex2 - new_mean**2, 0.0)
        mean_run = new_mean
        cov_run = cov_run * phi_a[:, None] + cov_s[:, i, :] * phi_na[:, None]
        lam *= phi_a[:, None]
        lam[:, i] = phi_na
    lam_orig = np.empty_like(lam)
    np.put_along_axis(lam_orig, order, lam, axis=1)
    return mean_run + shift[:, 0], var_run, lam_orig


# --- arms ------------------------------------------------------------------------------------
def _aggregate(u_idx: np.ndarray, w_cand: np.ndarray, n_u: int) -> np.ndarray:
    """Sum candidate weights onto their distinct cells. Two candidates of one node may point at
    the same cell (the sentinel pad reuses a real offset), and nodes overlap heavily, so the
    quadratic form must be taken over CELLS, not over candidate slots."""
    return np.bincount(u_idx.ravel(), weights=w_cand.ravel(), minlength=n_u)


def arm_moments(means, c_eff, u_idx, C, n_u):
    """(E, sd) for every arm, WITHOUT the per-step settle constant (the caller adds it).
    `means` already includes the element caps."""
    out = {}
    # clark -- full covariance in the fold and in the contraction
    cov_self = C[u_idx[:, :, None], u_idx[:, None, :]]
    m_n, _v_n, lam = clark_fold(means, cov_self)
    a = _aggregate(u_idx, c_eff[:, None] * lam, n_u)
    out["clark"] = (float(c_eff @ m_n), math.sqrt(max(float(a @ C @ a), 0.0)))
    # clark-diag -- the SAME fold with the off-diagonal removed
    var_u = np.diag(C)
    cov_d = np.zeros_like(cov_self)
    kk = np.arange(means.shape[1])
    cov_d[:, kk, kk] = var_u[u_idx]
    m_d, _v_d, lam_d = clark_fold(means, cov_d)
    a_d = _aggregate(u_idx, c_eff[:, None] * lam_d, n_u)
    out["clark-diag"] = (float(c_eff @ m_d), math.sqrt(max(float(a_d @ (var_u * a_d)), 0.0)))
    # fosm -- max of means, the winner's one-hot weights through the SAME full covariance
    win_k = np.argmax(means, axis=1)
    m_f = means[np.arange(means.shape[0]), win_k]
    lam_f = np.zeros_like(means)
    lam_f[np.arange(means.shape[0]), win_k] = 1.0
    a_f = _aggregate(u_idx, c_eff[:, None] * lam_f, n_u)
    out["fosm"] = (float(c_eff @ m_f), math.sqrt(max(float(a_f @ C @ a_f), 0.0)))
    # mean-map -- the same point prediction, no variance at all
    out["mean-map"] = (float(c_eff @ m_f), None)
    return out


# --- per-window assembly ---------------------------------------------------------------------
@dataclass
class WindowCase:
    """Everything both experiments need, built once so E0 and E1 consume the SAME bytes."""

    name: str
    seq: str
    distinct: bool
    overhang: bool
    overhang_gap_m: float
    overhang_cells: int
    track_len_m: float
    n_steps_total: int
    n_steps: int
    retained: float
    u_cells: np.ndarray
    u_idx: np.ndarray
    mu_u: np.ndarray
    caps: np.ndarray
    means: np.ndarray
    c_eff: np.ndarray
    C: np.ndarray
    const: float
    j_tls: float
    frac_count0: float
    sigma_med: float
    raw_sd_med: float
    meas_sd_med: float


def build_case(path: Path):
    _forbid_held_out(path)
    win = load_window(path)
    site = tls_site_of(path)
    tls = tls_on_window(win, site, "max")  # THE TRUTH (prereg A1(3))
    track = resample_track(win.gt_poses)
    if len(track) == 0:
        return None
    # the QC flag is geometry + the two truth rasters, computed BEFORE anything is scored
    flagged, gap, n_qc = overhang_flag(win, track, tls, tls_on_window(win, site, "mean"))
    cell_flat, caps, c_eff = build_nodes(win, track)
    ny, nx = win.shape
    t_steps = len(track)

    mu_f = win.mu.ravel()
    sg_f = win.sigma.ravel()
    tl_f = tls.ravel()
    usable = np.isfinite(mu_f) & np.isfinite(sg_f) & (sg_f > SIGMA_FLOOR) & np.isfinite(tl_f)

    node_ok = usable[cell_flat].all(axis=1)  # [4 * 3T]
    node_ok = node_ok.reshape(3 * t_steps, 4).all(axis=1)  # [3T], wheel-major
    step_ok = node_ok.reshape(3, t_steps).all(axis=0)  # [T]
    n_keep = int(step_ok.sum())
    if n_keep == 0:
        return None

    sel_nodes = np.repeat(np.tile(step_ok, 3), 4)
    cell_flat = cell_flat[sel_nodes]
    caps = caps[sel_nodes]
    c_eff = c_eff[sel_nodes]

    u_cells, inv = np.unique(cell_flat.ravel(), return_inverse=True)
    u_idx = inv.reshape(cell_flat.shape)
    var_ind, var_gcm, var_loc = cell_variance_split(win)
    C = cell_covariance(u_cells, win, var_ind, var_gcm, var_loc)
    mu_u = mu_f[u_cells]
    means = mu_u[u_idx] + caps

    env_tls = (tl_f[u_cells][u_idx] + caps).max(axis=1)
    const = n_keep * SETTLE_CONST_PER_STEP
    j_tls = float(c_eff @ env_tls) + const

    return WindowCase(
        name=f"{path.parent.name}/{path.stem}",
        seq=path.parent.name,
        distinct=path.parent.name != DUPLICATE_DIR,
        overhang=flagged,
        overhang_gap_m=gap,
        overhang_cells=n_qc,
        track_len_m=float(len(track) * RESAMPLE_M),
        n_steps_total=t_steps,
        n_steps=n_keep,
        retained=n_keep / t_steps,
        u_cells=u_cells,
        u_idx=u_idx,
        mu_u=mu_u,
        caps=caps,
        means=means,
        c_eff=c_eff,
        C=C,
        const=const,
        j_tls=j_tls,
        frac_count0=float((win.count.ravel()[u_cells] == 0).mean()),
        sigma_med=float(np.median(win.sigma.ravel()[u_cells])),
        raw_sd_med=float(np.nanmedian(win.raw_sd.ravel()[u_cells])),
        meas_sd_med=float(np.nanmedian(win.meas_sd.ravel()[u_cells])),
    )


def case_arms(case: WindowCase) -> dict[str, tuple[float, float | None]]:
    raw = arm_moments(case.means, case.c_eff, case.u_idx, case.C, len(case.u_cells))
    return {k: (e + case.const, sd) for k, (e, sd) in raw.items()}


def psd_factor(C: np.ndarray) -> tuple[np.ndarray, float]:
    """A with A A^T = C, via the symmetric eigendecomposition. NO jitter is added.

    A Cholesky would need a regularizer: cells that no scan ever touched have var_ind = 0
    exactly (their whole variance is mixture variance, see the module docstring), and the
    Gaussian coherence kernel is near-singular between adjacent cells, so C is PSD but can be
    numerically singular. `eigh` factors a singular PSD matrix exactly; negative eigenvalues
    are pure roundoff and are clipped to zero, and the most negative one is reported with each
    window so the reader can see how far from PSD the assembled matrix ever got.
    """
    w, V = np.linalg.eigh(C)
    return V * np.sqrt(np.maximum(w, 0.0))[None, :], float(w.min())


# --- E0: Monte Carlo from the same belief -----------------------------------------------------
def mc_cost(case: WindowCase, n_draws: int = N_MC_DEFAULT, seed: int = MC_SEED, chunk: int = MC_CHUNK):
    """Sample-mean and sample-sd of the EXACT cost over draws of the belief itself.

    The draws are generated from `case.C` and `case.mu_u` -- the very arrays the clark arm
    folds, not a re-derivation of them -- so E0 measures the fold's approximation and nothing
    else. Circular by design; that is the point (prereg E0).
    """
    L, min_eig = psd_factor(case.C)
    rng = np.random.default_rng(seed)
    total = 0.0
    total_sq = 0.0
    n_done = 0
    while n_done < n_draws:
        m = min(chunk, n_draws - n_done)
        z = rng.standard_normal((len(case.u_cells), m))
        h = case.mu_u[:, None] + L @ z  # [|U|, m]
        env = (h[case.u_idx] + case.caps[:, :, None]).max(axis=1)  # [N, m]
        j = case.c_eff @ env + case.const
        total += j.sum()
        total_sq += (j * j).sum()
        n_done += m
    mean = total / n_done
    var = max(total_sq / n_done - mean * mean, 0.0) * n_done / max(n_done - 1, 1)
    return mean, math.sqrt(var), n_done, min_eig


# --- experiment drivers -----------------------------------------------------------------------
def _iqr(v: np.ndarray) -> list[float]:
    return [float(np.percentile(v, 25)), float(np.percentile(v, 75))]


def _load_all_cases(site: str, root: Path, limit: int | None = None):
    files = window_files(site, root)
    if limit:
        files = files[:limit]
    cases, skipped = [], []
    for f in files:
        c = build_case(f)
        if c is None:
            skipped.append({"window": f"{f.parent.name}/{f.stem}", "reason": "empty track"})
            continue
        if c.retained < MIN_RETAINED:
            row = _meta(c)
            row["reason"] = f"retained {c.retained:.3f} < MIN_RETAINED {MIN_RETAINED}"
            skipped.append(row)
            continue
        cases.append(c)
    return cases, skipped


def _meta(c: WindowCase) -> dict:
    return {
        "window": c.name, "sequence": c.seq, "distinct": c.distinct,
        "overhang_flag": c.overhang, "overhang_gap_m": c.overhang_gap_m,
        "overhang_qc_cells": c.overhang_cells, "track_len_m": c.track_len_m,
        "n_steps": c.n_steps, "n_steps_total": c.n_steps_total, "retained": c.retained,
        "n_cells": int(len(c.u_cells)), "frac_cells_count0": c.frac_count0,
        "median_sigma_on_candidates_m": c.sigma_med,
        "median_raw_sd_on_candidates_m": c.raw_sd_med,
        "median_meas_sd_on_candidates_m": c.meas_sd_med,
    }


def _groups(rows: list[dict]) -> dict[str, list[dict]]:
    """The reporting partition fixed by prereg amendment A1: headline statistics are over the
    DISTINCT tracks, split by the geometry-only overhang flag; the `-unclamped` duplicates are
    kept as a separate clamp ablation and never pooled with the rest."""
    dis = [r for r in rows if r["distinct"]]
    return {
        "distinct_unflagged": [r for r in dis if not r["overhang_flag"]],
        "distinct_flagged": [r for r in dis if r["overhang_flag"]],
        "distinct_all": dis,
        "ablation_unclamped": [r for r in rows if not r["distinct"]],
    }


def run_e0(site: str, root: Path, n_draws: int, out_dir: Path, limit: int | None = None) -> dict:
    t_start = time.time()
    cases, skipped = _load_all_cases(site, root, limit)
    rows = []
    for c in cases:
        arms = case_arms(c)
        e_c, sd_c = arms["clark"]
        e_mc, sd_mc, n_used, min_eig = mc_cost(c, n_draws)
        row = _meta(c)
        row.update({
            "n_draws": n_used, "min_eig_C": min_eig,
            "E_clark": e_c, "sd_clark": sd_c, "E_mc": e_mc, "sd_mc": sd_mc,
            "rel_err_E": abs(e_c - e_mc) / abs(e_mc),
            "rel_err_sd": abs(sd_c - sd_mc) / sd_mc,
            "E_err_in_sd": abs(e_c - e_mc) / sd_mc,
            "mc_se_of_mean_in_sd": 1.0 / math.sqrt(n_used),
        })
        rows.append(row)
        print(f"  {c.name}{' [overhang]' if c.overhang else ''}: E {e_c:.3f} vs {e_mc:.3f}  "
              f"sd {sd_c:.3f} vs {sd_mc:.3f}  relE {row['rel_err_E']:.2e} "
              f"relSD {row['rel_err_sd']:.4f}", flush=True)

    def summary(pick: list[dict]) -> dict:
        if not pick:
            return {"n": 0}
        re_e = np.array([r["rel_err_E"] for r in pick])
        re_s = np.array([r["rel_err_sd"] for r in pick])
        e_sd = np.array([r["E_err_in_sd"] for r in pick])
        return {
            "n": len(pick),
            "median_rel_err_E": float(np.median(re_e)), "iqr_rel_err_E": _iqr(re_e),
            "median_rel_err_sd": float(np.median(re_s)), "iqr_rel_err_sd": _iqr(re_s),
            "max_rel_err_sd": float(re_s.max()),
            "median_E_err_in_sd_units": float(np.median(e_sd)),
            "iqr_E_err_in_sd_units": _iqr(e_sd),
            "E0_i_passed": bool(np.median(re_e) <= 0.05 and np.median(re_s) <= 0.05),
        }

    groups = _groups(rows)
    res = {
        "experiment": "E0 (fold approximation error, DESIGN SITE)",
        "prereg": "clark_paper/PREREG_risk_calibration.md frozen 2026-08-28 + amendment A1",
        "site_filter": site, "held_out_touched": False,
        "truth_raster": "tls_max_raster.npz", "qc_raster": "tls_mean_raster.npz",
        "overhang_threshold_m": OVERHANG_M,
        "n_windows_computed": len(rows),
        "n_draws_per_window": n_draws, "mc_seed": MC_SEED, "resample_m": RESAMPLE_M,
        "primary": summary(groups["distinct_unflagged"]),
        "groups": {k: summary(v) for k, v in groups.items()},
        "skipped": skipped,
        "windows": rows,
        "runtime_s": time.time() - t_start,
        "drift_rates_frozen": vars(DriftRates()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e0_design.json").write_text(json.dumps(res, indent=2))
    return res


ARMS_SCORED = ("clark", "clark-diag", "fosm")


def run_e1(site: str, root: Path, out_dir: Path, limit: int | None = None) -> dict:
    t_start = time.time()
    cases, skipped = _load_all_cases(site, root, limit)
    rows = []
    for c in cases:
        arms = case_arms(c)
        row = _meta(c)
        row["J_tls"] = c.j_tls
        for arm in ARMS_SCORED:
            e, sd = arms[arm]
            z = (c.j_tls - e) / sd
            row[arm] = {"E": e, "sd": sd, "z": z,
                        "nll": 0.5 * math.log(2.0 * math.pi * sd * sd) + 0.5 * z * z}
        e_mm, _ = arms["mean-map"]
        row["mean-map"] = {"E": e_mm, "sd": None, "err": c.j_tls - e_mm}
        rows.append(row)
        print(f"  {c.name}{' [overhang]' if c.overhang else ''}: J_tls {c.j_tls:.2f}  "
              f"clark z {row['clark']['z']:+.2f}  diag z {row['clark-diag']['z']:+.2f}  "
              f"fosm z {row['fosm']['z']:+.2f}", flush=True)

    def summary(pick: list[dict]) -> dict:
        if not pick:
            return {"n": 0}
        agg = {"n": len(pick)}
        for arm in ARMS_SCORED:
            z = np.array([r[arm]["z"] for r in pick])
            nll = np.array([r[arm]["nll"] for r in pick])
            agg[arm] = {
                "mean_nll": float(nll.mean()),
                "cov1": float(np.mean(np.abs(z) <= 1)),
                "cov2": float(np.mean(np.abs(z) <= 2)),
                "sd_ratio": float(np.std(z, ddof=1)) if len(z) > 1 else float("nan"),
                "mean_z": float(z.mean()), "median_z": float(np.median(z)),
                "median_sd": float(np.median([r[arm]["sd"] for r in pick])),
            }
        # paired NLL differences, the E1-iii statistic (design-site dry run only)
        for a, b in (("clark", "fosm"), ("clark", "clark-diag")):
            d = np.array([r[a]["nll"] - r[b]["nll"] for r in pick])
            agg[f"nll_diff_{a}_minus_{b}"] = {
                "mean": float(d.mean()), "median": float(np.median(d)),
                "n_negative": int((d < 0).sum()),
            }
        err = np.array([r["mean-map"]["err"] for r in pick])
        agg["mean-map"] = {"median_abs_err": float(np.median(np.abs(err))),
                           "mean_err": float(err.mean())}
        return agg

    groups = _groups(rows)
    seqs = sorted({r["sequence"] for r in rows if r["distinct"]})
    res = {
        "experiment": "E1 DESIGN-SITE DRY RUN -- NOT the pre-registered held-out test",
        "prereg": "clark_paper/PREREG_risk_calibration.md frozen 2026-08-28 + amendment A1",
        "site_filter": site, "held_out_touched": False,
        "truth_raster": "tls_max_raster.npz", "qc_raster": "tls_mean_raster.npz",
        "overhang_threshold_m": OVERHANG_M,
        "resample_m": RESAMPLE_M,
        "cost_weights": {"wz": 1.0, "wpitch": 0.7, "wroll": 0.5},
        "n_windows_computed": len(rows),
        "primary": summary(groups["distinct_unflagged"]),
        "groups": {k: summary(v) for k, v in groups.items()},
        "by_sequence_distinct_unflagged": {
            s_: summary([r for r in groups["distinct_unflagged"] if r["sequence"] == s_])
            for s_ in seqs
        },
        "skipped": skipped,
        "windows": rows,
        "runtime_s": time.time() - t_start,
        "drift_rates_frozen": vars(DriftRates()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e1_design_dryrun.json").write_text(json.dumps(res, indent=2))
    return res


def design_set_audit(root: Path = OUT_ROOT) -> dict:
    """Provenance facts about the design set that every number below depends on.

    The prereg's amendment A1(3) says `keble-college-02-default` and `-unclamped` are the same
    15 trajectories under two `max_variance` settings. They are more than that: on this data
    the two runs are bit-identical in every map layer, so the clamp ablation is a NO-OP. That
    is consistent with the mechanism -- `elevation_belief.measure_scan` applies the
    `max_variance` clamp only to cells it just measured, whose Kalman variance is ~(0.7 cm)^2,
    three orders below the 9e-4 m^2 ceiling, and the q_z motion inflation that does exceed the
    ceiling is added afterwards and never re-clamped. The clamp therefore never binds.
    """
    a_dir, b_dir = root / "keble-college-02-default", root / DUPLICATE_DIR
    _forbid_held_out(a_dir)
    _forbid_held_out(b_dir)
    layers = ("mu", "sigma", "raw_h", "raw_sd", "meas_sd", "count")
    per_window = []
    for fa in sorted(a_dir.glob("window_*.npz")):
        fb = b_dir / fa.name
        if not fb.exists():
            continue
        da, db = np.load(fa), np.load(fb)
        per_window.append({
            "window": fa.stem,
            "gt_poses_identical": bool(np.array_equal(da["gt_poses"], db["gt_poses"])),
            "map_layers_identical": bool(
                all(np.array_equal(da[k], db[k], equal_nan=True) for k in layers)
            ),
        })
    flags = []
    for f in window_files():
        w = load_window(f)
        tr = resample_track(w.gt_poses)
        site = tls_site_of(f)
        fl, gap, n = overhang_flag(
            w, tr, tls_on_window(w, site, "max"), tls_on_window(w, site, "mean")
        )
        flags.append({
            "window": f"{f.parent.name}/{f.stem}", "distinct": f.parent.name != DUPLICATE_DIR,
            "overhang_flag": fl, "overhang_gap_m": gap, "overhang_qc_cells": n,
            "track_len_m": float(len(tr) * RESAMPLE_M),
        })
    n_dis = [x for x in flags if x["distinct"]]
    return {
        "overhang_flag": {
            "rule": "max(tls_max - tls_mean) > OVERHANG_M over the track cells dilated 3x3",
            "threshold_m": OVERHANG_M,
            "n_windows_all": len(flags),
            "n_flagged_all": sum(x["overhang_flag"] for x in flags),
            "n_windows_distinct": len(n_dis),
            "n_flagged_distinct": sum(x["overhang_flag"] for x in n_dis),
            "per_window": flags,
        },
        "pair": [a_dir.name, b_dir.name],
        "layers_compared": list(layers),
        "n_windows": len(per_window),
        "n_gt_poses_identical": sum(w["gt_poses_identical"] for w in per_window),
        "n_map_layers_identical": sum(w["map_layers_identical"] for w in per_window),
        "clamp_ablation_is_a_no_op": all(w["map_layers_identical"] for w in per_window),
        "per_window": per_window,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiment", choices=["e0", "e1", "both", "audit"])
    ap.add_argument("--site", default=DESIGN_SITE, help="site glob (DESIGN site only)")
    ap.add_argument("--root", default=str(OUT_ROOT))
    ap.add_argument("--draws", type=int, default=N_MC_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "out" / "risk_calibration"),
    )
    args = ap.parse_args()
    out_dir = Path(args.out)
    root = Path(args.root)
    if args.experiment == "audit":
        a = design_set_audit(root)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "design_set_audit.json").write_text(json.dumps(a, indent=2))
        print(json.dumps({k: v for k, v in a.items() if k != "per_window"}, indent=2))
        return
    if args.experiment in ("e0", "both"):
        r = run_e0(args.site, root, args.draws, out_dir, args.limit)
        print("\nE0 by group:")
        for g, a in r["groups"].items():
            print(f"  {g:20s} {a}")
    if args.experiment in ("e1", "both"):
        r = run_e1(args.site, root, out_dir, args.limit)
        print("\nE1 dry run by group:")
        for g, a in r["groups"].items():
            print(f"  {g:20s} {json.dumps(a)}")


if __name__ == "__main__":
    main()
