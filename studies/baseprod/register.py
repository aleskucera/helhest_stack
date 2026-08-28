"""Item 1: per-traverse vertical registration of the drone DSM against the RTK track, plus
the void-fill validity mask. Truth-side only: no belief, no arm, no score enters here.

Ported from the spike (`~/data/baseprod_audit/spike/register.py`) with the traverse directory
and the DSM path taken from `paths`. The numerics are unchanged, and in particular the frozen
model-selection rule below is the rule PREREG_baseprod.md freezes -- a RULE, not an order.

Model. The audit measured RTK-minus-DSM at -0.18 m in zone A and +0.56 m in zone B, a sign
flip over ~400 m: a low-frequency doming error in the Pix4D bundle, not a datum offset. Over
one traverse footprint that field is well approximated by a low-order polynomial in (x, y).
The gradient does NOT transfer between zones (spike section 1), so every traverse -- held-out
included -- is registered against its own RTK track.

The comparison is footprint to footprint: the truth side is the DSM averaged over each of the
six wheel patches (r = 0.05 m discs) and then over the six wheels, the same spatial average
the suspension takes. r_wheel is a pure constant and is EXACTLY confounded with the constant
term of the correction; it is carried as that constant and never separated.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window as RWindow

from .constants import N_BLOCKS, WHEEL_PATCH_R
from .geometry import WHEELS, gnss_xyz, load_tf, wheel_xyz
from .paths import DSM, KEY, OUT, traverse_dir

NODATA = -10000.0
BUFFER = 2.0   # [m] corridor buffer around the track bbox

# --- the frozen model-selection rule ----------------------------------------------------------
SELECTION_CANDIDATES = ("plane", "quadratic")
SELECTION_RULE = (
    "Fit each candidate in {plane, quadratic} by least squares on the six-wheel footprint "
    "statistic D(x, y); score each by leave-one-contiguous-block-out cross-validation with "
    "N_BLOCKS = 5 contiguous along-track blocks of equal fix count, taken in RTK time order "
    "with no shuffling; select the candidate with the minimum CV RMSE. 'constant' is fitted "
    "and reported as a reference row but is not a candidate."
)


# --------------------------------------------------------------------------- poses
def rover_poses(tdir) -> pd.DataFrame:
    """RTK fixes with a body orientation: yaw from the track, roll/pitch from the IMU."""
    tdir = str(tdir)
    g = pd.read_csv(f"{tdir}/GNSS.csv")
    g = g[["Timestamp", "Altitude", "UTM_Easting", "UTM_Northing",
           "Position_Covariance_0", "Position_Covariance_8"]].copy()
    g.columns = ["t", "alt", "x", "y", "cov_h", "cov_v"]

    # course over ground, smoothed over ~1.5 s so stationary jitter does not spin the heading
    k = 7
    xs = g.x.rolling(k, center=True, min_periods=1).mean().to_numpy()
    ys = g.y.rolling(k, center=True, min_periods=1).mean().to_numpy()
    yaw = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    g["yaw"] = yaw

    imu = pd.read_csv(f"{tdir}/IMU.csv",
                      usecols=["Timestamp", "Orientation_X", "Orientation_Y",
                               "Orientation_Z", "Orientation_W"])
    q = imu[["Orientation_X", "Orientation_Y", "Orientation_Z", "Orientation_W"]].to_numpy()
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x_, y_, z_, w_ = q.T
    up_imu = np.stack([2 * (x_ * z_ - w_ * y_),
                       2 * (y_ * z_ + w_ * x_),
                       1 - 2 * (x_ * x_ + y_ * y_)], axis=1)
    # `imu_link` is NOT aligned with `base_link`: link_mast -> imu_link is a 180 deg rotation
    # about Z. Taking the IMU's raw roll/pitch as the body's INVERTS BOTH SIGNS, which doubles
    # rather than removes the tilt (spike: it put the depth cloud up to 1.2 m off). The
    # rotation is read from the TF tree, never hard-coded.
    R_bi = np.eye(3)
    tf = load_tf(tdir)
    for a, b in [("base_link", "link_mast"), ("link_mast", "imu_link")]:
        R_bi = R_bi @ tf[(a, b)][:3, :3]
    up = up_imu @ R_bi.T
    roll = np.arctan2(up[:, 1], up[:, 2])
    pitch = -np.arcsin(np.clip(up[:, 0], -1, 1))
    it = imu.Timestamp.to_numpy().astype(np.float64)
    g["roll"] = np.interp(g.t.to_numpy().astype(np.float64), it, roll)
    g["pitch"] = np.interp(g.t.to_numpy().astype(np.float64), it, pitch)
    return g


def body_R(yaw, pitch, roll):
    """[N, 3, 3] ZYX rotation body->world."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.empty((len(yaw), 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


# --------------------------------------------------------------------------- raster
class Corridor:
    """One windowed, full-resolution read of the DSM over a track's bounding box."""

    def __init__(self, xs, ys, buffer=BUFFER):
        self.ds = rasterio.open(DSM)
        x0, x1 = xs.min() - buffer, xs.max() + buffer
        y0, y1 = ys.min() - buffer, ys.max() + buffer
        r1, c0 = self.ds.index(x0, y1)
        r0, c1 = self.ds.index(x1, y0)
        row0, row1 = max(0, min(r1, r0)), min(self.ds.height, max(r1, r0) + 1)
        col0, col1 = max(0, min(c0, c1)), min(self.ds.width, max(c0, c1) + 1)
        self.win = RWindow(col0, row0, col1 - col0, row1 - row0)
        self.nbytes = (row1 - row0) * (col1 - col0) * 4
        self.a = self.ds.read(1, window=self.win).astype(np.float32)
        self.a[self.a == NODATA] = np.nan
        self.tr = self.ds.window_transform(self.win)
        self.res = self.ds.res[0]
        self.row0, self.col0 = row0, col0

    def rc(self, x, y):
        col = (np.asarray(x) - self.tr.c) / self.tr.a
        row = (np.asarray(y) - self.tr.f) / self.tr.e
        return row, col

    def disc_mean(self, x, y, radius, mask=None):
        """Mean DSM over a disc of `radius` about each (x, y). NaN if fully invalid."""
        row, col = self.rc(x, y)
        k = int(np.ceil(radius / self.res))
        dd = np.arange(-k, k + 1)
        DR, DC = np.meshgrid(dd, dd, indexing="ij")
        keep = (DR**2 + DC**2) * self.res**2 <= radius**2
        DR, DC = DR[keep], DC[keep]
        r = np.round(row)[:, None] + DR[None, :]
        c = np.round(col)[:, None] + DC[None, :]
        ok = (r >= 0) & (r < self.a.shape[0]) & (c >= 0) & (c < self.a.shape[1])
        r = np.clip(r, 0, self.a.shape[0] - 1).astype(np.int32)
        c = np.clip(c, 0, self.a.shape[1] - 1).astype(np.int32)
        v = self.a[r, c]
        if mask is not None:
            ok = ok & ~mask[r, c]
        v = np.where(ok & np.isfinite(v), v, np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanmean(v, axis=1), np.isfinite(v).mean(axis=1)


def voidfill_mask(a: np.ndarray) -> np.ndarray:
    """Pix4D constant-value fill: a pixel bit-identical to all four neighbours."""
    m = np.zeros(a.shape, bool)
    core = a[1:-1, 1:-1]
    eq = ((core == a[:-2, 1:-1]) & (core == a[2:, 1:-1])
          & (core == a[1:-1, :-2]) & (core == a[1:-1, 2:]))
    m[1:-1, 1:-1] = eq & np.isfinite(core)
    return m | ~np.isfinite(a)


# --------------------------------------------------------------------------- fitting
def design(x, y, order):
    x = (x - x.mean()) / max(np.ptp(x), 1e-9)
    y = (y - y.mean()) / max(np.ptp(y), 1e-9)
    cols = [np.ones_like(x)]
    if order >= 1:
        cols += [x, y]
    if order >= 2:
        cols += [x * x, x * y, y * y]
    return np.stack(cols, axis=1)


def block_cv(x, y, d, order, n_blocks=N_BLOCKS):
    """RMSE of a leave-one-contiguous-block-out fit. Blocks are along-track segments."""
    n = len(d)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    err = []
    for i in range(n_blocks):
        te = np.zeros(n, bool)
        te[edges[i]:edges[i + 1]] = True
        A = design(x, y, order)
        try:
            beta, *_ = np.linalg.lstsq(A[~te], d[~te], rcond=None)
        except np.linalg.LinAlgError:
            return float("nan")
        err.append(d[te] - A[te] @ beta)
    e = np.concatenate(err)
    return float(np.sqrt(np.mean(e**2)))


def semivariogram(x, y, r, lags):
    rng = np.random.default_rng(0)
    n = len(r)
    m = min(n, 2500)
    idx = rng.choice(n, m, replace=False) if n > m else np.arange(n)
    dx = x[idx][:, None] - x[idx][None, :]
    dy = y[idx][:, None] - y[idx][None, :]
    dist = np.hypot(dx, dy)
    dv = 0.5 * (r[idx][:, None] - r[idx][None, :]) ** 2
    iu = np.triu_indices(len(idx), 1)
    dist, dv = dist[iu], dv[iu]
    out = []
    for lo, hi in zip(lags[:-1], lags[1:]):
        s = (dist >= lo) & (dist < hi)
        if s.sum() > 50:
            out.append({"lag_m": 0.5 * (lo + hi), "gamma": float(dv[s].mean()),
                        "sd_equiv_m": float(np.sqrt(dv[s].mean())), "n": int(s.sum())})
    return out


def npz_path(name: str):
    return OUT / "registration" / f"register_{KEY.get(name, name)}.npz"


# --------------------------------------------------------------------------- driver
def run(name: str) -> dict:
    tdir = traverse_dir(name)
    key = KEY.get(name, name)
    tf = load_tf(tdir)
    wb = wheel_xyz(tf)
    p_gnss = gnss_xyz(tf)
    P = np.stack([wb[w] for w in WHEELS])

    g = rover_poses(tdir)
    R = body_R(g.yaw.to_numpy(), g.pitch.to_numpy(), g.roll.to_numpy())
    base = np.stack([g.x.to_numpy(), g.y.to_numpy(), g.alt.to_numpy()], axis=1)
    base = base - np.einsum("nij,j->ni", R, p_gnss)
    axles = base[:, None, :] + np.einsum("nij,wj->nwi", R, P)   # [N, 6, 3]

    cor = Corridor(axles[:, :, 0].ravel(), axles[:, :, 1].ravel())
    fill = voidfill_mask(cor.a)
    fill_frac = float(fill.mean())
    nan_frac = float((~np.isfinite(cor.a)).mean())

    dm = np.empty((len(g), 6))
    cv = np.empty((len(g), 6))
    for w in range(6):
        dm[:, w], cv[:, w] = cor.disc_mean(axles[:, w, 0], axles[:, w, 1],
                                           WHEEL_PATCH_R, mask=fill)
    good = np.isfinite(dm).all(axis=1)
    D = (axles[:, :, 2] - dm).mean(axis=1)

    pt, _ = cor.disc_mean(g.x.to_numpy(), g.y.to_numpy(), cor.res * 0.5)
    audit_like = g.alt.to_numpy() - pt

    x = axles[:, :, 0].mean(axis=1)[good]
    y = axles[:, :, 1].mean(axis=1)[good]
    d = D[good]

    res = {"traverse": str(name), "key": key, "n_fix": int(len(g)), "n_used": int(good.sum()),
           "corridor_px": [int(cor.win.height), int(cor.win.width)],
           "corridor_MB": round(cor.nbytes / 1e6, 1),
           "voidfill_frac_corridor": round(fill_frac, 5),
           "dsm_nan_frac_corridor": round(nan_frac, 5),
           "wheel_samples_hit_by_fill": round(float(1 - cv.mean()), 5),
           "audit_like_antenna_minus_dsm_median_m": round(float(np.nanmedian(audit_like)), 4),
           "audit_like_std_m": round(float(np.nanstd(audit_like)), 4),
           "D_raw_median_m": round(float(np.median(d)), 4),
           "D_raw_std_m": round(float(np.std(d)), 4),
           "D_raw_p5_p95_m": [round(float(np.percentile(d, 5)), 4),
                              round(float(np.percentile(d, 95)), 4)],
           "models": {}}

    best = None
    plane_ref = None
    for order, mname in [(0, "constant"), (1, "plane"), (2, "quadratic")]:
        A = design(x, y, order)
        beta, *_ = np.linalg.lstsq(A, d, rcond=None)
        r_in = d - A @ beta
        cvr = block_cv(x, y, d, order)
        res["models"][mname] = {
            "n_par": int(A.shape[1]),
            "insample_rmse_m": round(float(np.sqrt(np.mean(r_in**2))), 4),
            "insample_sd_m": round(float(np.std(r_in)), 4),
            "blockcv_rmse_m": round(cvr, 4),
            "const_term_m": round(float(beta[0]), 4),
        }
        if order >= 1:
            sx, sy = max(np.ptp(x), 1e-9), max(np.ptp(y), 1e-9)
            res["models"][mname]["tilt_mm_per_m"] = [round(float(beta[1] / sx * 1000), 2),
                                                     round(float(beta[2] / sy * 1000), 2)]
        if mname in SELECTION_CANDIDATES and (best is None or cvr < best[1]):
            best = (mname, cvr, order, beta, r_in)
        if mname == "plane":
            plane_ref = (beta, r_in, cvr)

    mname, cvr, order, beta, r_in = best
    res["selection_rule"] = SELECTION_RULE
    res["selection_candidates"] = list(SELECTION_CANDIDATES)
    res["chosen_model"] = mname
    res["chosen_blockcv_rmse_m"] = round(cvr, 4)
    pb, pr, pcv = plane_ref
    res["plane_only_alternative"] = {
        "sd_m": round(float(np.std(pr)), 4),
        "blockcv_rmse_m": round(pcv, 4),
        "penalty_vs_selected_blockcv_m": round(pcv - cvr, 4),
    }
    res["residual_after_registration"] = {
        "sd_m": round(float(np.std(r_in)), 4),
        "mad_m": round(float(np.median(np.abs(r_in - np.median(r_in)))) * 1.4826, 4),
        "p5_p95_m": [round(float(np.percentile(r_in, 5)), 4),
                     round(float(np.percentile(r_in, 95)), 4)],
    }
    res["r_wheel_plus_offset_estimate_m"] = round(float(beta[0]), 4)
    res["semivariogram"] = semivariogram(
        x, y, r_in, np.array([0, .5, 1, 2, 4, 8, 16, 32, 64, 128]))

    p = npz_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, x=x, y=y, D=d, resid=r_in, beta=beta, order=order,
                        axles=axles[good], base=base[good],
                        yaw=g.yaw.to_numpy()[good], t=g.t.to_numpy()[good],
                        pitch=g.pitch.to_numpy()[good], roll=g.roll.to_numpy()[good])
    return res


def run_and_record(names) -> dict:
    out = {}
    for n in names:
        out[KEY.get(n, n)] = run(n)
    d = OUT / "registration"
    d.mkdir(parents=True, exist_ok=True)
    prev = {}
    fp = d / "registration.json"
    if fp.exists():
        prev = json.loads(fp.read_text())
    prev.update(out)
    fp.write_text(json.dumps(prev, indent=2))
    return out
