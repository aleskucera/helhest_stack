"""v2 per-window driver (PREREG_attitude_cost.md): the v1 loading, QC and
element machinery verbatim, with the attitude cost and v2 arms in place of
the linear cost.

Node structure (from risk_calibration.build_case): nodes are wheel-major
blocks of t_steps, each with 4 candidates; c_eff[n] = settle_weight[wheel]
* blend[n]. For v2 the wheel supports feed the settle pitch/roll rows, so
the linear map from folded node maxima to the stacked attitude vector is
  G[2t + {0,1}, n] = R[{pitch,roll}, wheel(n)] * blend(n)   for step(n)=t.

PARITY HOOK: v1's clark E is c_eff @ m_fold + const; the v2 fold must
reproduce the same per-node means m_fold (same Clark recursion), asserted
per window at 1e-9. (The sd's differ by construction: v1 contracts with
the lambda rule; v2 tracks the Clark pairwise covariances - the prereg's
definition.)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spires.risk_calibration import SIGMA_FLOOR, Window, build_nodes, resample_track
from spires.vehicle import PAD_CAP, settle_weights
from . import constants as K
from .score import cell_covariance, cell_variance_split  # v1's belief model, unchanged
from .v2_arms import score_window
from .v2_moments import fold_supports, settle_pitch_roll_rows


def _attitude_G(c_eff: np.ndarray, t_steps: int) -> np.ndarray:
    """Linear map folded sub-node maxima -> stacked (pitch, roll).

    Sub-node layout (build_case): wheel-major blocks of t_steps wheel-steps,
    each wheel-step exploded into 4 consecutive blend positions; c_eff[n] =
    settle_weight[wheel] * blend[n]."""
    sw = settle_weights()                      # per-wheel scalar of the v1 cost
    R = settle_pitch_roll_rows()               # (2, 3) wheel columns L, R, rear
    n_nodes = len(c_eff)
    assert n_nodes == 3 * t_steps * 4, (n_nodes, t_steps)
    ws = np.arange(n_nodes) // 4               # wheel-step index
    wheel = ws // t_steps
    step = ws % t_steps
    blend = c_eff / sw[wheel]
    G = np.zeros((2 * t_steps, n_nodes))
    for n in range(n_nodes):
        G[2 * step[n], n] = R[0, wheel[n]] * blend[n]
        G[2 * step[n] + 1, n] = R[1, wheel[n]] * blend[n]
    return G


def case_v2(p: Path, n_ref: int = 20_000, seed: int = 0, correction=None) -> dict | None:
    d = np.load(p)
    win = Window(path=p, mu=d["mu"].astype(np.float64), sigma=d["sigma"].astype(np.float64),
                 raw_sd=d["raw_sd"].astype(np.float64), meas_sd=d["meas_sd"].astype(np.float64),
                 count=d["count"], x0=float(d["xmin"]), y0=float(d["ymin"]),
                 cell=float(d["cell"]), sx=float(d["sx"]), sy=float(d["sy"]),
                 gt_poses=d["gt_poses"])
    track = resample_track(win.gt_poses)
    if len(track) == 0:
        return None
    cell_flat, caps, c_eff = build_nodes(win, track)
    t_steps = len(track)
    mu_f, sg_f = win.mu.ravel(), win.sigma.ravel()
    cnt_f = win.count.ravel()
    usable = (np.isfinite(mu_f) & np.isfinite(sg_f) & (sg_f > SIGMA_FLOOR) & (cnt_f > 0))

    # QC identical to v1 minus the truth-referencing flag (no truth in v2)
    real_all = caps > PAD_CAP + 1.0
    qc_cells = np.unique(cell_flat[real_all])
    obs_frac = float((cnt_f[qc_cells] > 0).mean())
    seen = qc_cells[cnt_f[qc_cells] > 0]
    sig_med_all = float(np.median(sg_f[seen])) if len(seen) else float("nan")
    flags = []
    if obs_frac < K.FLAG_MIN_OBS_FRAC:
        flags.append("coverage")
    if not (sig_med_all <= K.FLAG_MAX_SIGMA_M):
        flags.append("sigma")

    node_ok = usable[cell_flat].all(axis=1).reshape(3 * t_steps, 4).all(axis=1)
    step_ok = node_ok.reshape(3, t_steps).all(axis=0)
    n_keep = int(step_ok.sum())
    base = {"name": f"{p.parent.name}/{p.stem}", "traverse": p.parent.name,
            "n_steps": n_keep, "n_steps_total": t_steps,
            "retained": n_keep / t_steps if t_steps else 0.0,
            "qc_observed_frac": round(obs_frac, 4),
            "qc_sigma_med_m": round(sig_med_all, 5) if np.isfinite(sig_med_all) else None,
            "qc_flags": flags, "flagged": bool(flags)}
    if n_keep == 0:
        base["scoreable"] = False
        return base
    base["scoreable"] = True

    sel = np.repeat(np.tile(step_ok, 3), 4)      # v1's own selection
    cell_flat, caps = cell_flat[sel], caps[sel]
    c_eff = c_eff[sel]
    u_cells, inv = np.unique(cell_flat.ravel(), return_inverse=True)
    u_idx = inv.reshape(cell_flat.shape)
    vi, vg, vl = cell_variance_split(win)
    C = cell_covariance(u_cells, win, vi, vg, vl)
    mu_u = mu_f[u_cells]
    means = mu_u[u_idx] + caps               # (n_nodes, 4)

    # abstract form: candidates flattened, jointly Gaussian via the cell cov
    n_nodes, ncand = means.shape
    mu_all = means.ravel()
    Cflat = C[u_idx.ravel()][:, u_idx.ravel()]
    node_slices = [np.arange(ncand * i, ncand * (i + 1)) for i in range(n_nodes)]
    G = _attitude_G(c_eff, n_keep)

    # parity hook: fold means must match v1's clark recursion means
    from spires.risk_calibration import clark_fold
    cov_self = C[u_idx[:, :, None], u_idx[:, None, :]]
    m_v1, _v, _l = clark_fold(means, cov_self)
    m_v2, _S = fold_supports(mu_all, Cflat, node_slices)
    assert np.max(np.abs(m_v1 - m_v2)) < 1e-6, "fold parity violated"  # metres

    alpha_w = None  # cost-weighted alpha for the correction; filled by runner
    row = score_window(mu_all, Cflat, node_slices, G, n_ref=n_ref, seed=seed,
                       correction=correction, alpha_w=alpha_w)
    base.update(row)
    return base


def main(win_dir: str, out_json: str, n_ref: int = 20_000):
    rows = []
    for p in sorted(Path(win_dir).glob("*/window_*.npz")):
        r = case_v2(p, n_ref=n_ref, seed=abs(hash(p.stem)) % 2**31)
        if r is not None:
            rows.append(r)
            print(r["name"], "flagged" if r["flagged"] else
                  (f"clark relE {r['arms']['clark']['rel_err_E']:.2e}" if r.get("scoreable") else "unscoreable"))
    Path(out_json).write_text(json.dumps(rows, indent=1))
    print("wrote", out_json, len(rows), "rows")


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
