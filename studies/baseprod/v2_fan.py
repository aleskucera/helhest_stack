"""v2 fan experiment (PREREG_attitude_cost.md 5b): sample K spline action
sequences at the window's entry pose, roll them through 2-D unicycle
kinematics, gate on the belief (mean-map attitude > 0.35 rad; any step with
unusable element cells), rank survivors by each arm's CVaR, referee with
SHARED terrain draws.

Candidates are scored over the full horizon; a candidate with ANY unusable
step is gated (strict form of the prereg's observed-fraction rule -- keeps
costs length-comparable; final constants are F1 business).
"""
from __future__ import annotations

import numpy as np

from spires.risk_calibration import SIGMA_FLOOR, build_nodes
from spires.vehicle import PAD_CAP
from .score import cell_covariance, cell_variance_split
from .v2_moments import gaussian_cvar
from .v2_arms import arm_clark, arm_clark_diag, arm_fosm, arm_mean_map, arm_step_form
from .v2_score import _attitude_G

ATT_GATE_RAD = 0.35
STEP_M = 0.10


def sample_actions(rng, K, T, v_max, w_max, knots=5):
    """(v, omega) piecewise-linear profiles: (K, knots) each."""
    v = rng.uniform(0.15 * v_max, v_max, (K, knots))
    w = rng.uniform(-w_max, w_max, (K, knots))
    return v, w


def rollout(pose, v_kn, w_kn, T, dt=0.05):
    """Unicycle rollout of one candidate, resampled to STEP_M. -> (S,3)."""
    n = int(T / dt)
    tt = np.linspace(0, 1, len(v_kn))
    ts = np.linspace(0, 1, n)
    v = np.interp(ts, tt, v_kn)
    w = np.interp(ts, tt, w_kn)
    x, y, psi = pose
    P = [(x, y, psi)]
    for i in range(n):
        psi = psi + w[i] * dt
        x = x + v[i] * np.cos(psi) * dt
        y = y + v[i] * np.sin(psi) * dt
        P.append((x, y, psi))
    P = np.array(P)
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < STEP_M:
        return None
    grid = np.arange(0, s[-1], STEP_M)
    out = np.stack([np.interp(grid, s, P[:, i]) for i in range(3)], axis=1)
    return out


def fan(win, K=32, T=3.0, v_max=2.6, w_max=1.0, seed=0, n_ref=20_000,
        correction=None, window_alpha_fn=None):
    """Score one fan. Returns record with per-arm pick + regret, or None."""
    rng = np.random.default_rng(seed)
    entry = win.gt_poses[0]
    yaw0 = np.arctan2(entry[1, 0], entry[0, 0])
    pose0 = (float(entry[0, 3]), float(entry[1, 3]), float(yaw0))
    vs, ws_ = sample_actions(rng, K, T, v_max, w_max)

    mu_f, sg_f, cnt_f = win.mu.ravel(), win.sigma.ravel(), win.count.ravel()
    usable = (np.isfinite(mu_f) & np.isfinite(sg_f) & (sg_f > SIGMA_FLOOR) & (cnt_f > 0))
    vi, vg, vl = cell_variance_split(win)

    cands = []
    n_att_gated = n_obs_gated = 0
    for k in range(K):
        track = rollout(pose0, vs[k], ws_[k], T)
        if track is None:
            n_obs_gated += 1
            continue
        try:
            cell_flat, caps, c_eff = build_nodes(win, track)
        except (IndexError, ValueError):
            n_obs_gated += 1          # left the grid
            continue
        t = len(track)
        node_ok = usable[cell_flat].all(axis=1).reshape(3 * t, 4).all(axis=1)
        if not node_ok.all():
            n_obs_gated += 1          # strict: every step fully usable
            continue
        # mean-map attitude gate
        u_cells, inv = np.unique(cell_flat.ravel(), return_inverse=True)
        u_idx = inv.reshape(cell_flat.shape)
        means = mu_f[u_cells][u_idx] + caps
        G = _attitude_G(c_eff, t)
        e0 = means.max(axis=1)
        x0 = G @ e0
        if np.abs(x0).max() > ATT_GATE_RAD:
            n_att_gated += 1
            continue
        cands.append(dict(track=track, cell_flat=cell_flat, caps=caps,
                          c_eff=c_eff, u_cells=u_cells, u_idx=u_idx,
                          means=means, G=G, t=t))
    rec = {"K": K, "n_att_gated": n_att_gated, "n_obs_gated": n_obs_gated,
           "n_valid": len(cands)}
    if len(cands) < 4:
        rec["scoreable"] = False
        return rec
    rec["scoreable"] = True

    # shared-terrain referee over the union of cells
    all_cells = np.unique(np.concatenate([c["u_cells"] for c in cands]))
    pos = {c: i for i, c in enumerate(all_cells)}
    C_un = cell_covariance(all_cells, win, vi, vg, vl)
    L = np.linalg.cholesky(C_un + 1e-12 * np.eye(len(C_un)))
    mu_un = mu_f[all_cells]
    cvar_mc = np.zeros(len(cands))
    chunk = 2048
    samples = [np.empty(n_ref) for _ in cands]
    done = 0
    rng2 = np.random.default_rng(seed + 1)
    while done < n_ref:
        b = min(chunk, n_ref - done)
        h = mu_un + rng2.standard_normal((b, len(mu_un))) @ L.T
        for ci, c in enumerate(cands):
            hc = h[:, [pos[x] for x in c["u_cells"]]]
            sup = (hc[:, c["u_idx"]] + c["caps"]).max(axis=2)
            x = sup @ c["G"].T
            samples[ci][done:done + b] = np.einsum("ij,ij->i", x, x)
        done += b
    q = 0.9
    for ci in range(len(cands)):
        srt = np.sort(samples[ci])
        cvar_mc[ci] = srt[int(np.ceil(q * n_ref)):].mean()
    best = float(cvar_mc.min())

    # arms rank survivors
    arm_cvar = {a: np.zeros(len(cands)) for a in
                ("clark-corr", "clark", "clark-diag", "fosm", "step-form",
                 "mean-map", "mc-2", "mc-32")}
    for ci, c in enumerate(cands):
        mu_all = c["means"].ravel()
        Cc = C_un[np.ix_([pos[x] for x in c["u_cells"]], [pos[x] for x in c["u_cells"]])]
        Cflat = Cc[c["u_idx"].ravel()][:, c["u_idx"].ravel()]
        nc = c["means"].shape[1]
        sl = [np.arange(nc * i, nc * (i + 1)) for i in range(c["means"].shape[0])]
        G = c["G"]
        E, sd = arm_clark(mu_all, Cflat, sl, G)
        arm_cvar["clark"][ci] = gaussian_cvar(E, sd)
        corr = correction(win, c) if correction else 1.0
        arm_cvar["clark-corr"][ci] = gaussian_cvar(E, sd * corr)
        E, sd = arm_clark_diag(mu_all, Cflat, sl, G)
        arm_cvar["clark-diag"][ci] = gaussian_cvar(E, sd)
        E, sd = arm_fosm(mu_all, Cflat, sl, G)
        arm_cvar["fosm"][ci] = gaussian_cvar(E, sd)
        E, sd = arm_step_form(mu_all, Cflat, sl, G)
        arm_cvar["step-form"][ci] = gaussian_cvar(E, sd)
        E, _ = arm_mean_map(mu_all, sl, G)
        arm_cvar["mean-map"][ci] = E
        for nN, nm in ((2, "mc-2"), (32, "mc-32")):
            sub = samples[ci][rng2.integers(0, n_ref, nN)]
            arm_cvar[nm][ci] = gaussian_cvar(float(sub.mean()),
                                             float(sub.std(ddof=1)) if nN > 1 else 0.0)
    rec["regret"] = {}
    rec["top1"] = {}
    for a, cv in arm_cvar.items():
        pick = int(np.argmin(cv))
        rec["regret"][a] = float(cvar_mc[pick] - best)
        rec["top1"][a] = bool(cvar_mc[pick] == best)
    rec["cvar_mc_spread"] = float(cvar_mc.max() - best)
    return rec
