"""The three things that actually make a robot's height map wrong.

Section 7's ranking experiment modelled exactly one of them, and crudely: a binary disc of
"observed", inside which the map was the ground truth EXACTLY, and revealing a cell handed over
the exact truth as well. That is a perfect sensor with perfect localisation, and it flatters
every sensing policy at once. This module supplies the other two, and a better version of the
first, so the section-7 conclusions can be re-tested against a belief that is wrong for the
reasons a real one is wrong.

The three sources are not variations on one theme -- they have completely different STRUCTURE,
and Study B already showed structure is what decides the answer (i.i.d. per-cell noise biased
the wheel envelope by 3.4 sigma, correlated noise by 0.4 sigma, an 8x difference from the
correlation alone). So each is generated with the structure it really has:

  OCCLUSION      geometric. Ray-cast 2.5-D visibility from the sensor: a cell is seen only if
                 nothing between it and the sensor rises above the line of sight. Produces
                 SHADOWS behind ridges, which a disc cannot. Unobserved cells are inpainted
                 flat, as the production map does.

  SENSOR         short-correlated, range-dependent. A blurred white field (correlation length
                 `CORR_LEN`, matching `studies/adjoint/sigma.py`) scaled by a per-cell standard
                 deviation that grows linearly with range. Independent per-cell noise would be
                 unrealistically benign here: the wheel envelope is a max over ~37 cells and
                 averages it away.

  LOCALISATION   rank-3, maximally correlated, and IRREDUCIBLE BY LOOKING. A pose error does not
                 corrupt cells one at a time -- it writes the whole measured patch into the map
                 at the wrong place, so the belief is a rigidly shifted and rotated copy of the
                 truth. This is the source a per-cell sigma field cannot represent, and the one
                 that no amount of sensing can fix: revealing more cells just delivers more
                 measurements carrying the same pose error.

WHAT THE POLICIES ARE TOLD. Each source contributes a per-cell sigma, and the total is their
quadrature sum -- so sigma stops being the heuristic saliency weight of section 7 and becomes a
statement about the noise actually injected.

The localisation term is the interesting one, and building it turned up a result worth stating
before any policy is run. The textbook marginal is first order: a displacement d produces a
height error of about grad(h) . d, so the per-cell sigma is `|grad h| * displacement`. THAT
MODEL IS INVALID AT REALISTIC POSE ERROR. A 2 deg heading error over a 6 m lever arm displaces
the map by ~2 cells while this terrain decorrelates in ~1 (autocorrelation 0.63 at one cell),
so the displaced map is nearly independent of the truth rather than a small perturbation of it:
a three-parameter first-order fit explains only 23% of the error field it generates. It is the
same lesson Study B found for map noise -- first order runs out well before realistic magnitudes
-- reappearing for pose error. The marginal is therefore capped at the decorrelated limit,
sqrt(2) * local terrain std, because you cannot be more wrong than the local relief.

So the per-cell sigma handed to the policies is a LOOSE marginal of a rank-3 error, blind both
to its correlation and to its non-linearity. That is not a shortcoming of the experiment; it is
the modelling gap the experiment exists to measure.
"""

from __future__ import annotations

import numpy as np

CORR_LEN = 0.15  # [m] spatial correlation length of sensor error (matches sigma.py)
SENSOR_BASE = 0.010  # [m] height noise at zero range
SENSOR_PER_M = 0.006  # [m per m] growth with range
LOC_SIGMA_XY = 0.06  # [m] translation error of the pose the map was written at
LOC_SIGMA_YAW = np.radians(2.0)  # [rad] heading error
SENSOR_HEIGHT = 0.55  # [m] lidar above the ground plane
MAX_RANGE = 6.0  # [m] beyond this nothing is observed
RAY_STEPS = 48
SIGMA_UNOBS_FRONTIER = 0.02  # [m] baseline uncertainty right at the edge of what was observed
SIGMA_UNOBS_KD = 0.05  # [m per m] growth with distance to the nearest observed cell (dominant term)
SIGMA_UNOBS_KR = 0.2  # [m per m] growth with local relief (roughness), secondary term
SIGMA_UNOBS_CEIL = 0.35  # [m] soft ceiling the tanh saturates toward -- never reached exactly
# [m] uniform fallback for --flat-sigma: a mid-field magnitude, deliberately structureless
SIGMA_UNOBS_FLAT = 0.245

SOURCES = ("clean", "sensor", "localisation", "occlusion", "all")


def _bilinear(field: np.ndarray, gy: np.ndarray, gx: np.ndarray) -> np.ndarray:
    """Sample `field` at fractional index coordinates, clamped at the border."""
    ny, nx = field.shape
    y0 = np.clip(np.floor(gy).astype(int), 0, ny - 1)
    x0 = np.clip(np.floor(gx).astype(int), 0, nx - 1)
    y1, x1 = np.clip(y0 + 1, 0, ny - 1), np.clip(x0 + 1, 0, nx - 1)
    fy, fx = np.clip(gy - y0, 0, 1), np.clip(gx - x0, 0, 1)
    return (
        field[y0, x0] * (1 - fy) * (1 - fx)
        + field[y1, x0] * fy * (1 - fx)
        + field[y0, x1] * (1 - fy) * fx
        + field[y1, x1] * fy * fx
    )


def visibility(truth: np.ndarray, XX: np.ndarray, YY: np.ndarray, cell: float) -> np.ndarray:
    """2.5-D ray-cast line of sight from a sensor at the origin. True where the cell is seen.

    A cell is visible iff no sample along the ray to it subtends a larger elevation angle than
    the cell itself -- the standard height-field visibility test. Unlike a radius, this casts
    SHADOWS: the ground behind a ridge is unobserved however close it is.
    """
    rng_field = np.hypot(XX, YY)
    seen = rng_field <= MAX_RANGE
    # Sample the ray from the sensor (at world origin, per rng_field above) to each cell, at
    # RAY_STEPS fractions of the way out. XX.min()/YY.min() are the CENTRES of cell (0, 0), so
    # no half-cell offset -- and the ray must start at the SENSOR's index, not the map corner.
    frac = np.linspace(0.0, 1.0, RAY_STEPS + 1)[1:-1].reshape(-1, 1, 1)
    gy = (YY - YY.min()) / cell
    gx = (XX - XX.min()) / cell
    gy_s = (0.0 - YY.min()) / cell
    gx_s = (0.0 - XX.min()) / cell
    h_along = _bilinear(truth, gy_s + frac * (gy - gy_s), gx_s + frac * (gx - gx_s))  # [S, ny, nx]
    # Elevation angle of each sample and of the target, from the sensor.
    ang_along = (h_along - SENSOR_HEIGHT) / np.maximum(frac * rng_field, 1e-6)
    ang_target = (truth - SENSOR_HEIGHT) / np.maximum(rng_field, 1e-6)
    blocked = (ang_along > ang_target[None] + 1e-4).any(axis=0)
    return seen & ~blocked


def correlated_field(shape: tuple[int, int], cell: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance field with correlation length CORR_LEN: white noise, separably blurred."""
    sd = max(CORR_LEN / cell, 0.5)
    r = int(np.ceil(3 * sd))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sd) ** 2)
    k /= k.sum()
    w = rng.standard_normal(shape)
    for axis in (0, 1):  # separable, wrap-around -- the patch is a torus for noise purposes
        w = sum(kk * np.roll(w, d, axis=axis) for d, kk in zip(range(-r, r + 1), k))
    return w / (w.std() + 1e-12)


def _gradient_mag(h: np.ndarray, cell: float) -> np.ndarray:
    gy, gx = np.gradient(h, cell)
    return np.hypot(gy, gx)


def apply_pose_error(truth, XX, YY, cell, dx, dy, dyaw):
    """Resample `truth` as if it had been written into the map at a pose wrong by (dx, dy, dyaw).

    Exposed so the verifier can reproduce the localisation field EXACTLY from three numbers,
    which is a direct proof of rank-3 structure -- stronger than regressing against a
    first-order gradient model, which is invalid at these displacements anyway.
    """
    c, s = np.cos(dyaw), np.sin(dyaw)
    xs, ys = XX * c - YY * s + dx, XX * s + YY * c + dy
    # XX.min()/YY.min() are already the CENTRES of cell (0, 0) (see ranking.build_case), so the
    # index of world x is (x - XX.min()) / cell exactly -- no half-cell offset.
    return _bilinear(truth, (ys - YY.min()) / cell, (xs - XX.min()) / cell)


def build_belief(
    truth: np.ndarray,
    XX: np.ndarray,
    YY: np.ndarray,
    cell: float,
    seed: int,
    source: str,
    flat_sigma: bool = False,
):
    """Return (belief, measured, observed, sigma) for one noise configuration.

    `belief`   what the robot currently thinks the map is (unobserved cells inpainted flat)
    `measured` what revealing a cell would hand over -- NOT the truth once a sensor or a pose
               error is in play. This is the change that matters most: section 7 let a reveal
               deliver ground truth, which is a perfect sensor and inflates every policy.
    `sigma`    the per-cell marginal the policies are told, consistent with what was injected
    `info`     the injected pose error, so `verify_noise` can prove the field really is rank-3
    """
    if source not in SOURCES:
        raise ValueError(f"unknown noise source {source!r}, expected one of {SOURCES}")
    rng = np.random.default_rng(50_000 + seed)
    rng_field = np.hypot(XX, YY)
    info: dict = {"pose_error": (0.0, 0.0, 0.0)}
    want = {"sensor", "all"}
    want_loc = {"localisation", "all"}

    # --- occlusion --------------------------------------------------------------------
    if source in ("occlusion", "all"):
        observed = visibility(truth, XX, YY, cell)
    else:
        observed = rng_field <= 1.5  # section 7's disc, kept so arms differ in ONE thing

    # --- localisation: the whole patch lands at the wrong pose -------------------------
    measured = truth.copy()
    disp = np.zeros_like(truth)
    if source in want_loc:
        dx, dy = LOC_SIGMA_XY * rng.standard_normal(2)
        dyaw = LOC_SIGMA_YAW * rng.standard_normal()
        measured = apply_pose_error(truth, XX, YY, cell, dx, dy, dyaw)
        info["pose_error"] = (float(dx), float(dy), float(dyaw))
        # marginal displacement magnitude: translation, plus a yaw lever arm growing with range
        disp = np.hypot(LOC_SIGMA_XY, rng_field * LOC_SIGMA_YAW)

    # --- sensor: correlated, range-growing height noise --------------------------------
    sens_sd = np.zeros_like(truth)
    if source in want:
        sens_sd = SENSOR_BASE + SENSOR_PER_M * rng_field
        measured = measured + sens_sd * correlated_field(truth.shape, cell, rng)

    belief = np.where(observed, measured, 0.0)

    # --- what the policies are told ----------------------------------------------------
    # Unobserved cells. Distance-to-nearest-observed-cell is the physically sensible DOMINANT
    # term -- a real mapper is least sure about what it has never come near -- plus a secondary,
    # truth-derived roughness term (a deliberate leak: it favours the entropy baseline, since
    # entropy is nothing BUT sigma; `flat_sigma` removes it for every policy at once). Both are
    # summed and passed through a soft tanh saturation toward SIGMA_UNOBS_CEIL rather than a hard
    # clip: a hard clip put ~63-65% of unobserved cells EXACTLY at the cap, so entropy's top-M
    # choice there was really the random tie-break, not a discrimination the baseline earned.
    if flat_sigma:
        sigma = np.full_like(truth, SIGMA_UNOBS_FLAT)
    else:
        dist_unobs = _distance_to_observed(observed, cell)
        rough = _local_relief(truth)
        raw = SIGMA_UNOBS_FRONTIER + SIGMA_UNOBS_KD * dist_unobs + SIGMA_UNOBS_KR * rough
        sigma = SIGMA_UNOBS_CEIL * np.tanh(raw / SIGMA_UNOBS_CEIL)
    # The first-order term |grad h| * displacement is only valid while the shift is small
    # against the terrain's own correlation length. It is NOT, at these magnitudes -- a 2 deg
    # yaw error over a 6 m lever arm displaces the map by ~2 cells while the terrain decorrelates
    # in ~1, so the displaced map is nearly independent of the truth and the linear term runs
    # away. Cap it at the decorrelated limit: you cannot be more wrong than the local relief.
    grad_belief = _gradient_mag(belief, cell)
    decorrelated = np.sqrt(2.0) * _local_std(truth)
    loc_sd = np.minimum(grad_belief * disp, decorrelated) if disp.any() else disp
    obs_sd = np.sqrt(sens_sd**2 + loc_sd**2 + 1e-6)
    sigma = np.where(observed, obs_sd, sigma)
    return (
        belief.astype(np.float32),
        measured.astype(np.float32),
        observed,
        sigma.astype(np.float64),
        info,
    )


def _local_std(h: np.ndarray, rad: int = 3) -> np.ndarray:
    """Standard deviation of the terrain in a local window -- the decorrelated error ceiling."""
    n = (2 * rad + 1) ** 2
    s1 = sum(
        np.roll(np.roll(h, dy, 0), dx, 1)
        for dy in range(-rad, rad + 1)
        for dx in range(-rad, rad + 1)
    )
    s2 = sum(
        np.roll(np.roll(h, dy, 0), dx, 1) ** 2
        for dy in range(-rad, rad + 1)
        for dx in range(-rad, rad + 1)
    )
    return np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))


_DT_INF = 1e20


def _dt1d(f: np.ndarray) -> np.ndarray:
    """Exact 1-D squared distance transform (Felzenszwalb & Huttenlocher lower envelope of
    parabolas). `f[i]` is 0 at a source and `_DT_INF` elsewhere; no scipy in this venv."""
    n = len(f)
    d = np.zeros(n)
    v = np.zeros(n, dtype=int)
    z = np.zeros(n + 1)
    k = 0
    v[0] = 0
    z[0], z[1] = -_DT_INF, _DT_INF
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        k += 1
        v[k] = q
        z[k], z[k + 1] = s, _DT_INF
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def _distance_to_observed(observed: np.ndarray, cell: float) -> np.ndarray:
    """[m] distance from every cell to the nearest True cell in `observed` (0 where observed)."""
    ny, nx = observed.shape
    f = np.where(observed, 0.0, _DT_INF)
    for c in range(nx):
        f[:, c] = _dt1d(f[:, c])
    for r in range(ny):
        f[r, :] = _dt1d(f[r, :])
    return np.sqrt(f) * cell


def _local_relief(h: np.ndarray, rad: int = 2) -> np.ndarray:
    lo = np.full_like(h, np.inf)
    hi = np.full_like(h, -np.inf)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            s = np.roll(np.roll(h, dy, 0), dx, 1)
            lo, hi = np.minimum(lo, s), np.maximum(hi, s)
    return hi - lo
