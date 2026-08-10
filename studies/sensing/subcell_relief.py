"""Model the discretization error the same way the method models everything else: through the max.

    .venv/bin/python -m studies.sensing.subcell_relief

THE PROBLEM. The wheel does not rest on the max of the grid's cell VALUES; it rests on the max
of the true surface. `lidar_belief.py --diagnose` showed the difference is real: on 10 cm cells
over 12 cm RMS terrain, a rasterized cell holds the mean height of the returns in it -- an area
average -- while the contact model reads it as the height AT the cell centre, and the two differ
by ~2 cm. A max over sampled points is BELOW the max over the surface those points came from, so
the grid is systematically optimistic about contact height, and it is most optimistic exactly
where the terrain is roughest.

THE FIX IS THE METHOD'S OWN THESIS, ONE LEVEL DOWN, because max is associative:

    max over the footprint  =  max over cells of ( max WITHIN the cell )

The current model silently replaces the inner max with the stored cell value. So instead of a
correction bolted on afterwards, we give the fold the within-cell max as its candidate:

    h(x) = h_c + eta(x),  eta zero-mean within the cell, variance tau_c^2  (the sub-grid relief)
    m_c  = h_c + tau_c a(n),   v_c = tau_c^2 b(n)     n = independent bumps per cell

and a(n), b(n) come from Clark's own recursion over n iid candidates rather than from an
imported extreme-value approximation, so the two levels are the same machinery.

TWO THINGS THAT DECIDE WHETHER IT WORKS, both handled here.
  * tau must be the relief left AFTER removing the local plane. A 10 cm cell on a 20 deg slope
    varies by 3.6 cm from the gradient alone, but that is the smooth field the neighbouring cells
    already encode -- counting it as sub-grid relief would double-count every hillside.
  * n is not guessable from first principles for broadband terrain, so it is FIT as a single
    scalar on one half of the map and validated on the other.

WHAT IS COMPARED, over many wheel placements, against the true footprint max computed on a 5x
finer surface:
  mean layer     the current model: max over coarse cells of (cell mean + cap)
  max layer      the free alternative: the mapper already computes a per-cell max, use it
  corrected      the model above
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..adjoint.generalise import fractal_terrain
from ..bench.clark import clark_build
from ..bench.ranking import CELL
from ..bench.ranking import OUT
from helhest.engine import RobotParams

SUB = 5  # the true surface is this many times finer than the belief grid


def _iid_max_moments(tau: float, n: int) -> tuple[float, float]:
    """E[max] and Var[max] of `n` iid N(0, tau^2), via the same Clark fold the method uses."""
    if n <= 1:
        return 0.0, tau * tau
    means = np.zeros((1, n))
    sigmas = np.full((1, n), tau)
    cov = np.zeros((1, n, n))
    cov[0, np.arange(n), np.arange(n)] = tau * tau
    mean_n, var_n, *_ = clark_build(means, sigmas, cov, cov)
    return float(mean_n[0]), float(var_n[0])


def build_surfaces(
    seed: int, n_coarse: int, beta: float = 1.8, hits_per_cell: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One fine surface, and the coarse views of it a mapper could produce.

    `beta` sets how much power the surface has below the grid's Nyquist -- the whole magnitude
    of this effect is set by that, so it is swept rather than fixed. `hits_per_cell` samples
    only that many of the SUB^2 sub-cells before taking the max, which is what a real mapper
    sees: the max layer is only as good as the returns that landed in the cell."""
    fine = fractal_terrain(n_coarse * SUB, n_coarse * SUB, CELL / SUB, seed=seed, beta=beta)
    blocks = fine.reshape(n_coarse, SUB, n_coarse, SUB).transpose(0, 2, 1, 3)
    coarse_mean = blocks.mean(axis=(2, 3))
    if hits_per_cell is None:
        coarse_max = blocks.max(axis=(2, 3))
    else:
        flat = blocks.reshape(n_coarse, n_coarse, SUB * SUB)
        rng = np.random.default_rng(seed + 5)
        pick = rng.integers(0, SUB * SUB, (n_coarse, n_coarse, hits_per_cell))
        coarse_max = np.take_along_axis(flat, pick, axis=2).max(axis=2)

    # tau: the within-cell spread AFTER removing the plane the coarse grid already represents.
    # The bilinear interpolation of `coarse_mean` evaluated at the fine positions IS that plane.
    gy, gx = np.meshgrid(np.arange(n_coarse * SUB), np.arange(n_coarse * SUB), indexing="ij")
    fy = np.clip((gy + 0.5) / SUB - 0.5, 0, n_coarse - 1.001)
    fx = np.clip((gx + 0.5) / SUB - 0.5, 0, n_coarse - 1.001)
    iy, ix = fy.astype(int), fx.astype(int)
    ty, tx = fy - iy, fx - ix
    plane = (
        coarse_mean[iy, ix] * (1 - tx) * (1 - ty)
        + coarse_mean[iy, ix + 1] * tx * (1 - ty)
        + coarse_mean[iy + 1, ix] * (1 - tx) * ty
        + coarse_mean[iy + 1, ix + 1] * tx * ty
    )
    resid = (fine - plane).reshape(n_coarse, SUB, n_coarse, SUB).transpose(0, 2, 1, 3)
    tau = resid.std(axis=(2, 3))
    return fine, coarse_mean, coarse_max, tau


def footprint_tables(rp: RobotParams) -> tuple[tuple, tuple]:
    """(dy, dx, cap) for the wheel disk at the coarse and at the fine resolution."""

    def table(cell: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = int(np.ceil(rp.wheel_radius / cell))
        dy, dx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        d2 = (dy * cell) ** 2 + (dx * cell) ** 2
        keep = d2 <= rp.wheel_radius**2
        cap = np.sqrt(np.maximum(rp.wheel_radius**2 - d2[keep], 0.0)) - rp.wheel_radius
        return dy[keep], dx[keep], cap

    return table(CELL), table(CELL / SUB)


def compare(
    seed: int, n_coarse: int, n_probe: int, beta: float = 1.8, hits: int | None = None
) -> dict:
    rp = RobotParams()
    fine, c_mean, c_max, tau = build_surfaces(seed, n_coarse, beta, hits)
    (cdy, cdx, ccap), (fdy, fdx, fcap) = footprint_tables(rp)
    k = len(cdy)

    rng = np.random.default_rng(seed + 99)
    margin = 6
    iy = rng.integers(margin, n_coarse - margin, n_probe)
    ix = rng.integers(margin, n_coarse - margin, n_probe)

    # truth: the max of the real surface over the wheel's disk, at fine resolution
    fy = iy[:, None] * SUB + SUB // 2 + fdy[None, :]
    fx = ix[:, None] * SUB + SUB // 2 + fdx[None, :]
    truth = (fine[fy, fx] + fcap[None, :]).max(axis=1)

    cy = iy[:, None] + cdy[None, :]
    cx = ix[:, None] + cdx[None, :]
    plain_mean = (c_mean[cy, cx] + ccap[None, :]).max(axis=1)
    plain_max = (c_max[cy, cx] + ccap[None, :]).max(axis=1)

    # --- the corrected model, with n fit on the first half and reported on the second ---------
    tau_probe = tau[cy, cx]
    half = n_probe // 2
    best_n, best_err = 1, np.inf
    for n in range(1, 33):
        a, b = _iid_max_moments(1.0, n)
        means = c_mean[cy[:half], cx[:half]] + ccap[None, :] + a * tau_probe[:half]
        sig = np.sqrt(b) * tau_probe[:half]
        cov = np.zeros((half, k, k))
        idx = np.arange(k)
        cov[:, idx, idx] = sig**2
        pred, _v, *_ = clark_build(means, sig, cov, cov)
        err = abs(float(np.mean(pred - truth[:half])))
        if err < best_err:
            best_n, best_err = n, err

    a, b = _iid_max_moments(1.0, best_n)
    means = c_mean[cy, cx] + ccap[None, :] + a * tau_probe
    sig = np.sqrt(b) * tau_probe
    cov = np.zeros((n_probe, k, k))
    idx = np.arange(k)
    cov[:, idx, idx] = sig**2
    corrected, _v, *_ = clark_build(means, sig, cov, cov)

    def stats(pred: np.ndarray, sl: slice) -> dict:
        e = (pred - truth)[sl]
        return {"bias_cm": float(e.mean() * 100), "rms_cm": float(np.sqrt((e**2).mean()) * 100)}

    hold = slice(half, n_probe)
    return {
        "seed": seed, "n_probe": n_probe, "sub": SUB, "k_coarse": int(k),
        "beta": beta, "hits_per_cell": hits,
        "tau_median_cm": float(np.median(tau) * 100),
        "fitted_n_bumps": best_n,
        "held_out": {
            "mean layer (current)": stats(plain_mean, hold),
            "max layer (free)": stats(plain_max, hold),
            "relief-corrected": stats(corrected, hold),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cells", type=int, default=90)
    ap.add_argument("--probes", type=int, default=400)
    ap.add_argument("--hetero", action="store_true",
                    help="terrain whose roughness varies in space, as real ground does")
    ap.add_argument("--tau-v2", action="store_true",
                    help="the within-scan de-meaned tau estimator")
    ap.add_argument("--from-returns", action="store_true",
                    help="estimate tau from the returns instead of the true surface")
    ap.add_argument("--sweep", action="store_true",
                    help="how the effect scales with sub-grid roughness and with hit count")
    args = ap.parse_args()

    if args.hetero:
        import warp as wp

        wp.init()
        print("=== does tau track roughness when roughness is really there? ===")
        rows = []
        for contrast in (1.0, 3.0, 6.0):
            r = tau_heterogeneous(0, args.cells, "cuda:0", contrast)
            rows.append(r)
            print(f"  contrast {contrast:4.1f}x  tau true {r['tau_true_median_cm']:5.2f} cm "
                  f"(spread {r['tau_true_spread']:.2f}, reliability "
                  f"{r['target_split_half_reliability']:+.2f})   est "
                  f"{r['tau_hat_median_cm']:5.2f} cm   corr(tau) {r['corr_tau']:+.3f}   "
                  f"corr(amplitude) {r['corr_amplitude']:+.3f}")
            print(f"                 pooled 3x3 {r['corr_tau_pooled_3x3']:+.3f}   "
                  f"5x5 {r['corr_tau_pooled_5x5']:+.3f}   9x9 {r['corr_tau_pooled_9x9']:+.3f}")
        (OUT / "subcell_tau_hetero.json").write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {OUT / 'subcell_tau_hetero.json'}")
        return

    if args.tau_v2:
        import warp as wp

        wp.init()
        print("=== tau estimated with the drift removed as common mode ===")
        rows = []
        for beta in (1.8, 2.4, 3.0):
            r = tau_v2(0, args.cells, args.probes, "cuda:0", beta)
            rows.append(r)
            print(f"\n  beta {beta}   {r['n_probe']} probes")
            print(f"    tau: true {r['tau_true_median_cm']:.2f} cm, estimated "
                  f"{r['tau_hat_median_cm']:.2f} cm, corr {r['tau_corr']:+.3f}")
            for name, v in r["held_out"].items():
                print(f"    {name:26s} bias {v['bias_cm']:+7.2f}  RMS {v['rms_cm']:6.2f}")
        (OUT / "subcell_tau_v2.json").write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {OUT / 'subcell_tau_v2.json'}")
        return

    if args.from_returns:
        import warp as wp

        wp.init()
        print("=== tau estimated from the returns, not from the true surface ===")
        rows = []
        for beta in (1.8, 2.4, 3.0):
            r = from_returns(0, args.cells, args.probes, "cuda:0", beta)
            rows.append(r)
            print(f"\n  beta {beta}   {r['n_probe']} probes, median "
                  f"{r['median_hits_per_cell']:.0f} hits/cell")
            print(f"    tau: true {r['tau_true_median_cm']:.2f} cm, estimated "
                  f"{r['tau_hat_median_cm']:.2f} cm, corr {r['tau_corr']:+.2f}")
            for name, v in r["held_out"].items():
                print(f"    {name:26s} bias {v['bias_cm']:+7.2f}  RMS {v['rms_cm']:6.2f}")
        (OUT / "subcell_from_returns.json").write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {OUT / 'subcell_from_returns.json'}")
        return

    if args.sweep:
        print("=== how big is this, really? ===")
        print("  beta = terrain spectrum (higher = smoother below the grid); "
              "hits = returns per cell\n")
        print(f"  {'beta':>5} {'hits':>5} {'tau[cm]':>8} | {'mean layer':>18} "
              f"{'max layer':>18} {'corrected':>18}")
        out = []
        for beta in (1.8, 2.4, 3.0):
            for hits in (None, 8, 3):
                r = compare(0, args.cells, args.probes, beta, hits)
                h = r["held_out"]
                out.append(r)
                cells = "   ".join(
                    f"{h[k]['bias_cm']:+6.2f}/{h[k]['rms_cm']:5.2f}"
                    for k in ("mean layer (current)", "max layer (free)", "relief-corrected")
                )
                hl = "all" if hits is None else str(hits)
                print(f"  {beta:>5.1f} {hl:>5} {r['tau_median_cm']:>8.2f} |  {cells}")
        (OUT / "subcell_relief_sweep.json").write_text(json.dumps(out, indent=1))
        print(f"\nwrote {OUT / 'subcell_relief_sweep.json'}  (bias/RMS in cm, held-out half)")
        return

    print("=== predicting the TRUE footprint max from a coarse grid ===")
    print("    (bias < 0 means the model sits BELOW the real contact height: optimistic)")
    rows = []
    for seed in range(args.seeds):
        r = compare(seed, args.cells, args.probes)
        rows.append(r)
        h = r["held_out"]
        print(f"  seed {seed}  tau {r['tau_median_cm']:.2f} cm, n={r['fitted_n_bumps']:2d}   "
              + "   ".join(f"{k}: {v['bias_cm']:+.2f}/{v['rms_cm']:.2f}" for k, v in h.items()))
    agg = {
        name: {
            "bias_cm": float(np.mean([r["held_out"][name]["bias_cm"] for r in rows])),
            "rms_cm": float(np.mean([r["held_out"][name]["rms_cm"] for r in rows])),
        }
        for name in rows[0]["held_out"]
    }
    print("\n  held-out mean over seeds        bias [cm]   RMS [cm]")
    for name, v in agg.items():
        print(f"    {name:26s}   {v['bias_cm']:+7.2f}   {v['rms_cm']:7.2f}")
    path = OUT / "subcell_relief.json"
    path.write_text(json.dumps({"per_seed": rows, "aggregate": agg}, indent=1))
    print(f"\nwrote {path}")



# --- the deployment question: can tau be estimated from the returns themselves? ---------------
def _rasterize_coarse(cloud: np.ndarray, n_coarse: int, x0: float, y0: float) -> dict:
    """Bin a cloud into the coarse grid, keeping the second moment as well as the first.

    The mapper already computes mean/max/count; the only new accumulator is sum-of-squares, and
    that is what turns a heightmap into a roughness map."""
    ix = np.floor((cloud[:, 0] - x0) / CELL).astype(np.int64)
    iy = np.floor((cloud[:, 1] - y0) / CELL).astype(np.int64)
    keep = (ix >= 0) & (ix < n_coarse) & (iy >= 0) & (iy < n_coarse)
    ix, iy, z = ix[keep], iy[keep], cloud[keep, 2]
    shape = (n_coarse, n_coarse)
    cnt = np.zeros(shape)
    s1 = np.zeros(shape)
    s2 = np.zeros(shape)
    mx = np.full(shape, -np.inf)
    np.add.at(cnt, (iy, ix), 1.0)
    np.add.at(s1, (iy, ix), z)
    np.add.at(s2, (iy, ix), z * z)
    np.maximum.at(mx, (iy, ix), z)
    # A max is the least robust statistic there is: one bad return owns the cell. Quantiles of
    # the same returns cost a sort and nothing else, so they come out of the same pass.
    flat_id = iy * n_coarse + ix
    order = np.lexsort((z, flat_id))
    zs, ids = z[order], flat_id[order]
    start = np.searchsorted(ids, np.arange(n_coarse * n_coarse), side="left")
    cnt_flat = cnt.ravel().astype(np.int64)
    quant = {}
    for q in (0.90, 0.95):
        idx = start + np.maximum((cnt_flat * q).astype(np.int64) - 0, 0)
        idx = np.clip(idx, 0, len(zs) - 1)
        v = np.where(cnt_flat > 0, zs[idx], np.nan).reshape(shape)
        quant[f"q{int(q*100)}"] = v
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = np.maximum(s2 / cnt - mean**2, 0.0) * np.where(cnt > 1, cnt / (cnt - 1), np.nan)
    return {"count": cnt, "mean": mean, "var": var, "max": mx, **quant}


def from_returns(seed: int, n_coarse: int, n_probe: int, device: str, beta: float = 2.4) -> dict:
    """Everything as before, but tau is ESTIMATED from the returns instead of read off the true
    surface -- which is the only version that could ever be deployed.

        tau^2 = var(hit heights) - var(sensor) - var(slope)

    The sensor term is calibrated by flying the identical trajectory over a PLANE, where the true
    relief is zero by construction, so whatever within-cell spread survives is the sensor's. The
    slope term is s^2 c^2 / 12, the variance a linear ramp contributes across a cell -- without
    it every hillside would report its own gradient as roughness.
    """
    from .lidar_belief import LidarSim
    from .lidar_belief import NoiseParams

    fine_cell = CELL / SUB
    fine = fractal_terrain(n_coarse * SUB, n_coarse * SUB, fine_cell, seed=seed, beta=beta)
    flat = np.zeros_like(fine)
    p = NoiseParams()
    xs = np.linspace(1.0, 8.0, 40)
    traj = np.stack([xs, np.full_like(xs, 4.5), np.zeros_like(xs)], axis=1)

    def fly(surface: np.ndarray) -> np.ndarray:
        sim = LidarSim(surface, 0.0, 0.0, fine_cell, device)
        rng = np.random.default_rng(seed)
        steps = np.array([p.pose_step_xy, p.pose_step_xy, p.pose_step_z, p.pose_step_yaw,
                          p.pose_step_pitch, p.pose_step_pitch])
        drift = np.zeros(6)
        out = []
        for k, (x, y, yaw) in enumerate(traj):
            drift = drift + rng.normal(0.0, steps)
            pts = sim.scan((float(x), float(y), float(yaw)), p, seed * 7919 + k)
            if len(pts) == 0:
                continue
            a = pts.numpy()
            dxp, dyp = a[:, 0] - x, a[:, 1] - y
            c, sn = np.cos(drift[3]), np.sin(drift[3])
            out.append(np.stack([
                c * dxp - sn * dyp + x + drift[0],
                sn * dxp + c * dyp + y + drift[1],
                a[:, 2] + drift[2] - drift[4] * dxp + drift[5] * dyp,
            ], axis=1))
        return np.concatenate(out, axis=0)

    obs = _rasterize_coarse(fly(fine), n_coarse, 0.0, 0.0)
    cal = _rasterize_coarse(fly(flat), n_coarse, 0.0, 0.0)

    # the slope the coarse map already represents
    gy, gx = np.gradient(np.nan_to_num(obs["mean"]), CELL)
    var_slope = (gx**2 + gy**2) * CELL**2 / 12.0
    tau_hat = np.sqrt(np.maximum(obs["var"] - np.nan_to_num(cal["var"]) - var_slope, 0.0))

    # the oracle tau, for reference only
    _f, _cm, _cx, tau_true = build_surfaces(seed, n_coarse, beta, None)

    rp = RobotParams()
    (cdy, cdx, ccap), (fdy, fdx, fcap) = footprint_tables(rp)
    k = len(cdy)
    # a count==0 hole is filled by nan_to_num above, faking a cliff at every neighbour the
    # gradient stencil (+-1 in y or x) reads it from -- drop those neighbours from usable too,
    # not just the hole itself, or var_slope zero-clamps tau_hat in a ring around every hole.
    hole = obs["count"] == 0
    touches_hole = hole.copy()
    touches_hole[1:, :] |= hole[:-1, :]
    touches_hole[:-1, :] |= hole[1:, :]
    touches_hole[:, 1:] |= hole[:, :-1]
    touches_hole[:, :-1] |= hole[:, 1:]
    usable = (obs["count"] >= 3) & np.isfinite(obs["mean"]) & ~touches_hole
    rng = np.random.default_rng(seed + 7)
    margin = 6
    cand = np.argwhere(usable[margin:-margin, margin:-margin]) + margin
    pick = cand[rng.choice(len(cand), size=min(n_probe, len(cand)), replace=False)]
    iy, ix = pick[:, 0], pick[:, 1]
    ok = np.ones(len(iy), bool)
    for d_y, d_x in zip(cdy, cdx):
        ok &= usable[iy + d_y, ix + d_x]
    iy, ix = iy[ok], ix[ok]
    n = len(iy)

    fy = iy[:, None] * SUB + SUB // 2 + fdy[None, :]
    fx = ix[:, None] * SUB + SUB // 2 + fdx[None, :]
    truth = (fine[fy, fx] + fcap[None, :]).max(axis=1)
    cy, cx = iy[:, None] + cdy[None, :], ix[:, None] + cdx[None, :]

    def fold(tau_field: np.ndarray, n_bumps: int) -> np.ndarray:
        a, b = _iid_max_moments(1.0, n_bumps)
        t = tau_field[cy, cx]
        means = obs["mean"][cy, cx] + ccap[None, :] + a * t
        sig = np.sqrt(b) * t
        cov = np.zeros((n, k, k))
        idx = np.arange(k)
        cov[:, idx, idx] = sig**2
        pred, *_ = clark_build(means, sig, cov, cov)
        return pred

    half = n // 2

    def fit_n(tau_field: np.ndarray) -> int:
        best, be = 1, np.inf
        for nb in range(1, 33):
            e = abs(float(np.mean((fold(tau_field, nb) - truth)[:half])))
            if e < be:
                best, be = nb, e
        return best

    preds = {
        "mean layer (current)": (obs["mean"][cy, cx] + ccap[None, :]).max(axis=1),
        "max layer (free)": (obs["max"][cy, cx] + ccap[None, :]).max(axis=1),
        "p95 layer (free)": (obs["q95"][cy, cx] + ccap[None, :]).max(axis=1),
        "p90 layer (free)": (obs["q90"][cy, cx] + ccap[None, :]).max(axis=1),
        "corrected, tau ESTIMATED": fold(tau_hat, fit_n(tau_hat)),
        "corrected, tau ORACLE": fold(tau_true, fit_n(tau_true)),
    }
    hold = slice(half, n)
    res = {
        "seed": seed, "beta": beta, "n_probe": int(n),
        "median_hits_per_cell": float(np.median(obs["count"][usable])),
        "tau_true_median_cm": float(np.median(tau_true[usable]) * 100),
        "tau_hat_median_cm": float(np.median(tau_hat[usable]) * 100),
        "tau_corr": float(np.corrcoef(tau_hat[usable], tau_true[usable])[0, 1]),
        "held_out": {
            name: {
                "bias_cm": float(((v - truth)[hold]).mean() * 100),
                "rms_cm": float(np.sqrt((((v - truth)[hold]) ** 2).mean()) * 100),
            }
            for name, v in preds.items()
        },
    }
    return res


# --- a tau estimator with per-cell skill --------------------------------------------------------
def _residual_variance(
    cloud: np.ndarray, scan_id: np.ndarray, coarse_mean: np.ndarray, n_coarse: int,
    block: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell variance of the returns AFTER removing everything that is not sub-grid relief.

    Three subtractions, in the order that matters:
      1. the grid-scale surface -- each hit's height minus the bilinear interpolation of the
         coarse map at the hit's own (x, y). Whatever the grid can already represent leaves.
      2. the per-scan offset -- de-mean within (scan, block). A scan's drift is one rigid
         transform, so it is COMMON MODE inside a scan and cancels exactly here. Pooling all
         scans together, as the previous estimator did, let inter-scan disagreement masquerade
         as roughness, and that disagreement is proportional to local slope, which is why the
         estimate had no per-cell skill.
      3. nothing else: what remains is sub-grid relief plus sensor noise, and the sensor part is
         measured separately on a plane.

    The block is 3x3 cells only to have enough returns per scan to form a mean; the squared
    residuals are still accumulated per CELL, so the output keeps full resolution.
    """
    ix = np.floor((cloud[:, 0]) / CELL).astype(np.int64)
    iy = np.floor((cloud[:, 1]) / CELL).astype(np.int64)
    keep = (ix >= 1) & (ix < n_coarse - 1) & (iy >= 1) & (iy < n_coarse - 1)
    ix, iy, z, sid = ix[keep], iy[keep], cloud[keep, 2], scan_id[keep]

    fx = np.clip(cloud[keep, 0] / CELL - 0.5, 0, n_coarse - 1.001)
    fy = np.clip(cloud[keep, 1] / CELL - 0.5, 0, n_coarse - 1.001)
    jx, jy = fx.astype(int), fy.astype(int)
    tx, ty = fx - jx, fy - jy
    cm = np.nan_to_num(coarse_mean)
    surface = (
        cm[jy, jx] * (1 - tx) * (1 - ty) + cm[jy, jx + 1] * tx * (1 - ty)
        + cm[jy + 1, jx] * (1 - tx) * ty + cm[jy + 1, jx + 1] * tx * ty
    )
    r = z - surface

    nb = (n_coarse + block - 1) // block
    key = (sid * nb + iy // block) * nb + ix // block
    uniq, inv = np.unique(key, return_inverse=True)
    gsum = np.zeros(len(uniq))
    gcnt = np.zeros(len(uniq))
    np.add.at(gsum, inv, r)
    np.add.at(gcnt, inv, 1.0)
    r = r - (gsum / np.maximum(gcnt, 1.0))[inv]
    usable = gcnt[inv] >= 2  # a group of one carries no variance information

    shape = (n_coarse, n_coarse)
    ss = np.zeros(shape)
    cn = np.zeros(shape)
    np.add.at(ss, (iy[usable], ix[usable]), r[usable] ** 2)
    np.add.at(cn, (iy[usable], ix[usable]), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = ss / cn
    return var, cn


def tau_v2(seed: int, n_coarse: int, n_probe: int, device: str, beta: float = 2.4) -> dict:
    """The estimator above, scored the same way: does it track tau, and does the correction it
    feeds beat the one built on the flat-calibrated estimate?"""
    from .lidar_belief import LidarSim
    from .lidar_belief import NoiseParams

    fine_cell = CELL / SUB
    fine = fractal_terrain(n_coarse * SUB, n_coarse * SUB, fine_cell, seed=seed, beta=beta)
    flat = np.zeros_like(fine)
    p = NoiseParams()
    xs = np.linspace(1.0, 8.0, 40)
    traj = np.stack([xs, np.full_like(xs, 4.5), np.zeros_like(xs)], axis=1)

    def fly(surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sim = LidarSim(surface, 0.0, 0.0, fine_cell, device)
        rng = np.random.default_rng(seed)
        steps = np.array([p.pose_step_xy, p.pose_step_xy, p.pose_step_z, p.pose_step_yaw,
                          p.pose_step_pitch, p.pose_step_pitch])
        drift = np.zeros(6)
        pts_all, ids = [], []
        for k, (x, y, yaw) in enumerate(traj):
            drift = drift + rng.normal(0.0, steps)
            pts = sim.scan((float(x), float(y), float(yaw)), p, seed * 7919 + k)
            if len(pts) == 0:
                continue
            a = pts.numpy()
            dxp, dyp = a[:, 0] - x, a[:, 1] - y
            c, sn = np.cos(drift[3]), np.sin(drift[3])
            pts_all.append(np.stack([
                c * dxp - sn * dyp + x + drift[0],
                sn * dxp + c * dyp + y + drift[1],
                a[:, 2] + drift[2] - drift[4] * dxp + drift[5] * dyp,
            ], axis=1))
            ids.append(np.full(len(a), k))
        return np.concatenate(pts_all), np.concatenate(ids)

    cloud, sid = fly(fine)
    cloud_f, sid_f = fly(flat)
    obs = _rasterize_coarse(cloud, n_coarse, 0.0, 0.0)
    var_r, cnt_r = _residual_variance(cloud, sid, obs["mean"], n_coarse)
    obs_f = _rasterize_coarse(cloud_f, n_coarse, 0.0, 0.0)
    var_s, _ = _residual_variance(cloud_f, sid_f, obs_f["mean"], n_coarse)
    tau_hat = np.sqrt(np.maximum(np.nan_to_num(var_r) - np.nan_to_num(var_s), 0.0))

    _f, _cm, _cx, tau_true = build_surfaces(seed, n_coarse, beta, None)
    rp = RobotParams()
    (cdy, cdx, ccap), (fdy, fdx, fcap) = footprint_tables(rp)
    k = len(cdy)
    usable = (obs["count"] >= 3) & np.isfinite(obs["mean"]) & (cnt_r >= 3)
    rng = np.random.default_rng(seed + 7)
    margin = 6
    cand = np.argwhere(usable[margin:-margin, margin:-margin]) + margin
    pick = cand[rng.choice(len(cand), size=min(n_probe, len(cand)), replace=False)]
    iy, ix = pick[:, 0], pick[:, 1]
    ok = np.ones(len(iy), bool)
    for d_y, d_x in zip(cdy, cdx):
        ok &= usable[iy + d_y, ix + d_x]
    iy, ix = iy[ok], ix[ok]
    n = len(iy)
    fy = iy[:, None] * SUB + SUB // 2 + fdy[None, :]
    fx = ix[:, None] * SUB + SUB // 2 + fdx[None, :]
    truth = (fine[fy, fx] + fcap[None, :]).max(axis=1)
    cy, cx = iy[:, None] + cdy[None, :], ix[:, None] + cdx[None, :]

    def fold(tau_field: np.ndarray, nb: int) -> np.ndarray:
        a, b = _iid_max_moments(1.0, nb)
        t = tau_field[cy, cx]
        means = obs["mean"][cy, cx] + ccap[None, :] + a * t
        sig = np.sqrt(b) * t
        cov = np.zeros((n, k, k))
        idx = np.arange(k)
        cov[:, idx, idx] = sig**2
        pred, *_ = clark_build(means, sig, cov, cov)
        return pred

    half = n // 2

    def fit_n(tf: np.ndarray) -> int:
        best, be = 1, np.inf
        for nb in range(1, 33):
            e = abs(float(np.mean((fold(tf, nb) - truth)[:half])))
            if e < be:
                best, be = nb, e
        return best

    preds = {
        "mean layer (current)": (obs["mean"][cy, cx] + ccap[None, :]).max(axis=1),
        "corrected, tau v2": fold(tau_hat, fit_n(tau_hat)),
        "corrected, tau ORACLE": fold(tau_true, fit_n(tau_true)),
    }
    hold = slice(half, n)
    m = usable
    return {
        "seed": seed, "beta": beta, "n_probe": int(n),
        "tau_true_median_cm": float(np.median(tau_true[m]) * 100),
        "tau_hat_median_cm": float(np.median(tau_hat[m]) * 100),
        "tau_corr": float(np.corrcoef(tau_hat[m], tau_true[m])[0, 1]),
        "held_out": {
            name: {
                "bias_cm": float(((v - truth)[hold]).mean() * 100),
                "rms_cm": float(np.sqrt((((v - truth)[hold]) ** 2).mean()) * 100),
            }
            for name, v in preds.items()
        },
    }


def heterogeneous_surface(
    n_coarse: int, seed: int, contrast: float = 6.0
) -> tuple[np.ndarray, np.ndarray]:
    """A surface whose ROUGHNESS varies in space, which is the case real terrain presents and
    the fractal does not.

    A homogeneous fractal has statistically constant sub-grid roughness: its per-cell tau varies
    only by sampling noise (measured split-half reliability 0.21-0.50), so asking an estimator to
    track it per cell is asking it to predict noise. Real ground has gravel beside packed soil
    and grass beside bare earth. Here the sub-grid component is multiplied by a smooth amplitude
    field spanning `contrast`, so tau is a real, recoverable property of place.

    Returns the fine surface and the amplitude field at coarse resolution.
    """
    rng = np.random.default_rng(seed)
    nf = n_coarse * SUB
    # large scale: a coarse fractal upsampled bilinearly, so it has NO sub-grid content
    base_c = fractal_terrain(n_coarse, n_coarse, CELL, seed=seed, beta=2.0)
    gy, gx = np.meshgrid(np.arange(nf), np.arange(nf), indexing="ij")
    fy = np.clip((gy + 0.5) / SUB - 0.5, 0, n_coarse - 1.001)
    fx = np.clip((gx + 0.5) / SUB - 0.5, 0, n_coarse - 1.001)
    iy, ix = fy.astype(int), fx.astype(int)
    ty, tx = fy - iy, fx - ix
    base = (
        base_c[iy, ix] * (1 - tx) * (1 - ty) + base_c[iy, ix + 1] * tx * (1 - ty)
        + base_c[iy + 1, ix] * (1 - tx) * ty + base_c[iy + 1, ix + 1] * tx * ty
    )
    # Roughness amplitude, as PATCHES rather than a gentle gradient. A min-max mapped fractal
    # concentrates in the middle -- "6x contrast" then means a 25% spread, which is not what
    # gravel beside packed soil looks like. Thresholding gives genuinely distinct regions.
    field = fractal_terrain(n_coarse, n_coarse, CELL, seed=seed + 31, beta=3.2)
    rough_region = (field > np.median(field)).astype(np.float64)
    k = np.array([0.25, 0.5, 0.25])
    for _ in range(2):  # soften the boundaries so they are not a step in one cell
        rough_region = np.apply_along_axis(np.convolve, 0, rough_region, k, mode="same")
        rough_region = np.apply_along_axis(np.convolve, 1, rough_region, k, mode="same")
        rough_region /= rough_region.max()
    amp_c = 1.0 + (contrast - 1.0) * rough_region
    amp = amp_c[iy, ix]
    # sub-grid component, with each coarse cell's own mean removed so it cannot move the map
    hf = rng.normal(0.0, 1.0, (nf, nf))
    hf = hf - hf.reshape(n_coarse, SUB, n_coarse, SUB).mean(axis=(1, 3)).repeat(SUB, 0).repeat(SUB, 1)
    return base + 0.012 * amp * hf, amp_c


def tau_heterogeneous(seed: int, n_coarse: int, device: str, contrast: float = 6.0) -> dict:
    """Does the estimator track roughness when roughness is actually there to be tracked?"""
    from .lidar_belief import LidarSim
    from .lidar_belief import NoiseParams

    fine_cell = CELL / SUB
    fine, amp = heterogeneous_surface(n_coarse, seed, contrast)
    flat = np.zeros_like(fine)
    p = NoiseParams()
    xs = np.linspace(1.0, 8.0, 40)
    traj = np.stack([xs, np.full_like(xs, 4.5), np.zeros_like(xs)], axis=1)

    def fly(surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sim = LidarSim(surface, 0.0, 0.0, fine_cell, device)
        rng = np.random.default_rng(seed)
        steps = np.array([p.pose_step_xy, p.pose_step_xy, p.pose_step_z, p.pose_step_yaw,
                          p.pose_step_pitch, p.pose_step_pitch])
        drift = np.zeros(6)
        pts, ids = [], []
        for k, (x, y, yaw) in enumerate(traj):
            drift = drift + rng.normal(0.0, steps)
            a = sim.scan((float(x), float(y), float(yaw)), p, seed * 7919 + k).numpy()
            dxp, dyp = a[:, 0] - x, a[:, 1] - y
            c, sn = np.cos(drift[3]), np.sin(drift[3])
            pts.append(np.stack([
                c * dxp - sn * dyp + x + drift[0], sn * dxp + c * dyp + y + drift[1],
                a[:, 2] + drift[2] - drift[4] * dxp + drift[5] * dyp,
            ], axis=1))
            ids.append(np.full(len(a), k))
        return np.concatenate(pts), np.concatenate(ids)

    cl, sid = fly(fine)
    clf, sidf = fly(flat)
    obs = _rasterize_coarse(cl, n_coarse, 0.0, 0.0)
    var_r, cnt_r = _residual_variance(cl, sid, obs["mean"], n_coarse)
    obs_f = _rasterize_coarse(clf, n_coarse, 0.0, 0.0)
    var_s, _ = _residual_variance(clf, sidf, obs_f["mean"], n_coarse)
    tau_hat = np.sqrt(np.maximum(np.nan_to_num(var_r) - np.nan_to_num(var_s), 0.0))

    # The target must be the relief left after removing the plane the grid already represents.
    # Taking the raw within-cell std instead lets the local GRADIENT dominate, which is a
    # property of slope rather than of roughness -- it swamped the signal on the first attempt.
    true_mean = fine.reshape(n_coarse, SUB, n_coarse, SUB).transpose(0, 2, 1, 3).mean(axis=(2, 3))
    gy2, gx2 = np.meshgrid(np.arange(n_coarse * SUB), np.arange(n_coarse * SUB), indexing="ij")
    fy2 = np.clip((gy2 + 0.5) / SUB - 0.5, 0, n_coarse - 1.001)
    fx2 = np.clip((gx2 + 0.5) / SUB - 0.5, 0, n_coarse - 1.001)
    iy2, ix2 = fy2.astype(int), fx2.astype(int)
    ty2, tx2 = fy2 - iy2, fx2 - ix2
    plane = (
        true_mean[iy2, ix2] * (1 - tx2) * (1 - ty2) + true_mean[iy2, ix2 + 1] * tx2 * (1 - ty2)
        + true_mean[iy2 + 1, ix2] * (1 - tx2) * ty2 + true_mean[iy2 + 1, ix2 + 1] * tx2 * ty2
    )
    blocks = (fine - plane).reshape(n_coarse, SUB, n_coarse, SUB).transpose(0, 2, 1, 3)
    tau_true = blocks.std(axis=(2, 3))
    # split-half reliability of the target, so the correlation has a ceiling to be judged against
    flat_sub = blocks.reshape(n_coarse, n_coarse, SUB * SUB)
    perm = np.random.default_rng(0).permutation(SUB * SUB)
    rel = float(np.corrcoef(
        flat_sub[:, :, perm[:12]].std(axis=2).ravel(),
        flat_sub[:, :, perm[12:24]].std(axis=2).ravel(),
    )[0, 1])

    m = (obs["count"] >= 3) & (cnt_r >= 3) & np.isfinite(obs["mean"])

    # tau is a smooth field -- roughness is a property of a patch of ground, not of a 10 cm
    # square -- so pooling it over neighbours is not a fudge, it is using what we know about the
    # quantity. A within-cell MAX could not be pooled this way; a variance can.
    def _pool(a: np.ndarray, w: int) -> np.ndarray:
        k = np.ones(w) / w
        out = np.apply_along_axis(np.convolve, 0, a, k, mode="same")
        return np.apply_along_axis(np.convolve, 1, out, k, mode="same")

    pooled = {
        f"corr_tau_pooled_{w}x{w}": float(
            np.corrcoef(_pool(tau_hat, w)[m], _pool(tau_true, w)[m])[0, 1]
        )
        for w in (3, 5, 9)
    }
    return {
        "seed": seed, "contrast": contrast, "n_cells": int(m.sum()), **pooled,
        "target_split_half_reliability": rel,
        "tau_true_median_cm": float(np.median(tau_true[m]) * 100),
        "tau_true_spread": float(tau_true[m].std() / tau_true[m].mean()),
        "tau_hat_median_cm": float(np.median(tau_hat[m]) * 100),
        "corr_tau": float(np.corrcoef(tau_hat[m], tau_true[m])[0, 1]),
        "corr_amplitude": float(np.corrcoef(tau_hat[m], amp[m])[0, 1]),
    }

if __name__ == "__main__":
    main()
