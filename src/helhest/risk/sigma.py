"""The per-cell height uncertainty the contact estimator consumes, and the spatial correlation
of the error it assumes.

WHY THIS IS HERE. `MppiGpu` carries a belief (`sim.elevation`) and an observed-cell mask
(`MppiGpu.measured`) but no uncertainty, and the contact estimator cannot run without one. This
module is the study's noise model (`studies/bench/noise.py`) moved into the library, so the map
the deployed planner is handed is built by the SAME code whose accuracy the paper measured.

THE MODEL, in one line per term:

  observed cells      sd = sqrt(sensor^2 + localisation^2)
      sensor          `sensor_base + sensor_per_m * range` -- short-correlated, range-growing
      localisation    `|grad h| * displacement`, CAPPED at the decorrelated limit
                      `sqrt(2) * local terrain sd`. The cap is not a safety factor: a 2 deg
                      heading error over a 6 m lever arm displaces the map by more than the
                      terrain's own correlation length, so the first-order marginal runs away
                      while the true error saturates at the local relief. You cannot be more
                      wrong than the ground varies.
  unobserved cells    a soft-saturating `frontier + k_d * distance_to_observed + k_r * relief`,
                      passed through `ceil * tanh(raw / ceil)`. Distance to the nearest observed
                      cell is the dominant term -- a mapper is least sure about what it has never
                      come near. The tanh rather than a hard clip because clipping put ~63% of
                      unobserved cells EXACTLY at the cap, which turns any top-M selection there
                      into a random tie-break.

THE ONE DEVIATION FROM THE STUDY, DECLARED. `studies/bench/noise.py` evaluates the roughness and
the decorrelation cap on the GROUND TRUTH -- a deliberate oracle leak, present so the entropy
baseline it was comparing against was not handicapped. A robot has no truth, so both are
evaluated on the BELIEF here. Where the belief is inpainted this reads flat and the roughness
term under-reports; the distance-to-observed term, which dominates in exactly that region, does
not depend on it.

Everything in here is HOST-side and runs once per perception frame, not in the refine loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SigmaParams:
    """The uncertainty model's constants. Defaults are `studies/bench/noise.py` verbatim, which
    is what every number in the paper was measured under -- change them and the study's accuracy
    figures no longer describe the map you are handing the estimator."""

    sensor_base: float = 0.010  # [m] height noise at zero range
    sensor_per_m: float = 0.006  # [m per m] growth with range
    loc_sigma_xy: float = 0.06  # [m] translation error of the pose the map was written at
    loc_sigma_yaw: float = np.radians(2.0)  # [rad] heading error -> a lever arm growing with range
    unobs_frontier: float = 0.02  # [m] uncertainty right at the edge of what was observed
    unobs_k_dist: float = 0.05  # [m per m] growth with distance to the nearest observed cell
    unobs_k_relief: float = 0.2  # [m per m] growth with local relief (roughness)
    unobs_ceil: float = 0.35  # [m] soft ceiling the tanh saturates toward -- never reached
    corr_len: float = 0.15  # [m] spatial correlation length of the sensor error
    relief_radius: int = 2  # [cells] window the relief is measured over
    local_sd_radius: int = 3  # [cells] window the decorrelation cap is measured over


# --- the correlation the estimator's quadratic form is built on --------------------------------


def gauss_kernel(corr_len: float, cell: float) -> tuple[np.ndarray, int]:
    """Normalised 1-D Gaussian taps for a correlation length in metres."""
    if corr_len <= 0.0:
        return np.ones(1, np.float32), 0
    sd = corr_len / cell
    radius = max(int(np.ceil(3.0 * sd)), 1)
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (k / sd) ** 2)
    return (w / w.sum()).astype(np.float32), radius


def rho1_table(corr_len: float, cell: float) -> np.ndarray:
    """rho1[|lag|] = normalised autocorrelation of the 1-D blur kernel `gauss_kernel` builds.

    Index by abs(lag) in cells; EXACTLY zero for |lag| >= len(table) (the kernel's finite
    support), which is what makes the estimator's cross-cell covariance decay to a hard zero
    rather than an assumed one.
    """
    w, radius = gauss_kernel(corr_len, cell)
    support = 2 * radius  # convolving w with itself has support 2*radius
    ac = np.array([np.dot(w[: len(w) - lag], w[lag:]) for lag in range(support + 1)])
    return (ac / ac[0]).astype(np.float64)


# --- the per-cell marginal ----------------------------------------------------------------------

_DT_INF = 1e20


def _dt1d(f: np.ndarray) -> np.ndarray:
    """Exact 1-D squared distance transform (Felzenszwalb & Huttenlocher lower envelope of
    parabolas). `f[i]` is 0 at a source and `_DT_INF` elsewhere."""
    n = len(f)
    d = np.zeros(n)
    v = np.zeros(n, dtype=int)
    z = np.zeros(n + 1)
    k = 0
    v[0] = 0
    z[0], z[1] = -_DT_INF, _DT_INF
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = _DT_INF
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def distance_to_observed(observed: np.ndarray, cell: float) -> np.ndarray:
    """[m] distance from every cell to the nearest True cell in `observed` (0 where observed)."""
    ny, nx = observed.shape
    f = np.where(observed, 0.0, _DT_INF)
    for c in range(nx):
        f[:, c] = _dt1d(f[:, c])
    for r in range(ny):
        f[r, :] = _dt1d(f[r, :])
    return np.sqrt(f) * cell


def _local_relief(h: np.ndarray, rad: int) -> np.ndarray:
    lo = np.full_like(h, np.inf)
    hi = np.full_like(h, -np.inf)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            s = np.roll(np.roll(h, dy, 0), dx, 1)
            lo, hi = np.minimum(lo, s), np.maximum(hi, s)
    return hi - lo


def _local_sd(h: np.ndarray, rad: int) -> np.ndarray:
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


def map_sigma(
    belief: np.ndarray,
    observed: np.ndarray,
    cell: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    sensor_xy: tuple[float, float] = (0.0, 0.0),
    params: SigmaParams = SigmaParams(),
) -> np.ndarray:
    """Per-cell height standard deviation [m] for a belief map, as float32 [ny, nx].

    `belief`    [ny, nx] the map the planner rolls out on (inpainted where blind)
    `observed`  [ny, nx] truthy where the cell carries real data -- the SAME mask
                `MppiGpu.set_measured` takes
    `sensor_xy` world position the ranges are measured from (the sensor, not the map centre)
    """
    h = np.asarray(belief, np.float64)
    obs = np.asarray(observed).astype(bool)
    if h.shape != obs.shape:
        raise ValueError(f"belief {h.shape} and observed {obs.shape} must have the same shape")
    p = params
    ny, nx = h.shape
    xs = origin_x + (np.arange(nx) + 0.5) * cell
    ys = origin_y + (np.arange(ny) + 0.5) * cell
    rng_field = np.hypot(xs[None, :] - sensor_xy[0], ys[:, None] - sensor_xy[1])

    # observed: sensor noise in quadrature with a pose-error marginal capped at decorrelation
    sens_sd = p.sensor_base + p.sensor_per_m * rng_field
    disp = np.hypot(p.loc_sigma_xy, rng_field * p.loc_sigma_yaw)
    gy, gx = np.gradient(h, cell)
    loc_sd = np.minimum(np.hypot(gy, gx) * disp, np.sqrt(2.0) * _local_sd(h, p.local_sd_radius))
    obs_sd = np.sqrt(sens_sd**2 + loc_sd**2 + 1e-6)

    # unobserved: dominated by how far the cell is from anything ever seen
    raw = (
        p.unobs_frontier
        + p.unobs_k_dist * distance_to_observed(obs, cell)
        + p.unobs_k_relief * _local_relief(h, p.relief_radius)
    )
    unobs_sd = p.unobs_ceil * np.tanh(raw / p.unobs_ceil)
    return np.where(obs, obs_sd, unobs_sd).astype(np.float32)
