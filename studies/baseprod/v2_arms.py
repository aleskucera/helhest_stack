"""v2 arms (PREREG_attitude_cost.md section 3) on the abstract window form.

A window is (mu_all, C_all, node_slices, G):
  mu_all, C_all   joint Gaussian of all candidate entries (v1's means/cov
                  gathered per candidate),
  node_slices     candidate indices per node (wheel-step),
  G (2T x n_nodes) linear map from folded node maxima to the stacked
                  attitude vector (settle pitch/roll rows x blend weights).

Cost L = x'x with x = G e, e the node maxima. clark's e is the fold's
joint Gaussian (v2_moments.fold_supports); the quadratic step is exact.
"""
from __future__ import annotations

import numpy as np

from .v2_moments import fold_supports, onehot_supports, quadratic_moments, gaussian_cvar


def _quad(m, S, G):
    return quadratic_moments(m, S, G.T @ G)


def arm_clark(mu_all, C_all, node_slices, G, correction=None, alpha_w=None,
              fast=None):
    if fast is not None:
        from .v2_moments import fold_supports_fast
        means, cov_self, C, u_idx = fast
        m, S = fold_supports_fast(means, cov_self, C, u_idx)
    else:
        m, S = fold_supports(mu_all, C_all, node_slices)
    E, V = _quad(m, S, G)
    sd = np.sqrt(V)
    if correction is not None:
        sd = sd * float(correction(alpha_w))
    return E, sd


def arm_clark_diag(mu_all, C_all, node_slices, G):
    m, S = fold_supports(mu_all, np.diag(np.diag(C_all)), node_slices)
    E, V = _quad(m, S, G)
    return E, np.sqrt(V)


def arm_fosm(mu_all, C_all, node_slices, G):
    m, S = onehot_supports(mu_all, C_all, node_slices)
    E, V = _quad(m, S, G)
    return E, np.sqrt(V)


def arm_mean_map(mu_all, node_slices, G):
    e0 = np.array([mu_all[idx].max() for idx in node_slices])
    x0 = G @ e0
    return float(x0 @ x0), None


def arm_step_form(mu_all, C_all, node_slices, G):
    """Deployed-practice analog: mean-map cost + per-node first-order
    sigma-sum spread through the cost's gradient at the mean map."""
    e0 = np.array([mu_all[idx].max() for idx in node_slices])
    x0 = G @ e0
    E = float(x0 @ x0)
    g = 2.0 * G.T @ x0                       # dL/de at the mean map
    w = np.array([idx[np.argmax(mu_all[idx])] for idx in node_slices])
    sd = float(np.abs(g) @ np.sqrt(np.diag(C_all)[w]))
    return E, sd


def sample_costs(mu_all, C_all, node_slices, G, n, seed, chunk=4096):
    """MC through the true pipeline: draw candidates, max per node, attitude
    quadratic. Returns the n cost samples (the referee, and the mc-N arms
    subsample from the same construction with their own seeds)."""
    rng = np.random.default_rng(seed)
    Lc = np.linalg.cholesky(C_all + 1e-12 * np.eye(len(C_all)))
    out = np.empty(n)
    done = 0
    nodes = [np.asarray(idx) for idx in node_slices]
    while done < n:
        b = min(chunk, n - done)
        h = mu_all + rng.standard_normal((b, len(mu_all))) @ Lc.T
        e = np.stack([h[:, idx].max(axis=1) for idx in nodes], axis=1)
        x = e @ G.T
        out[done:done + b] = np.einsum("ij,ij->i", x, x)
        done += b
    return out


def score_window(mu_all, C_all, node_slices, G, n_ref=20_000, seed=0,
                 correction=None, alpha_w=None, q=0.9, fast=None):
    """All eight arms + the referee for one window. Returns a dict in the
    v1 row idiom: per-arm (E, sd), referee stats, per-arm referee errors."""
    ref = sample_costs(mu_all, C_all, node_slices, G, n_ref, seed)
    E_mc, sd_mc = float(ref.mean()), float(ref.std(ddof=1))
    tail = np.sort(ref)[int(np.ceil(q * len(ref))):]
    cvar_mc = float(tail.mean())

    arms = {}
    arms["clark"] = arm_clark(mu_all, C_all, node_slices, G, fast=fast)
    E_c, sd_c = arms["clark"]
    corr = float(correction(alpha_w)) if correction is not None else 1.0
    arms["clark-corr"] = (E_c, sd_c * corr)
    arms["clark-diag"] = arm_clark_diag(mu_all, C_all, node_slices, G)
    arms["fosm"] = arm_fosm(mu_all, C_all, node_slices, G)
    arms["step-form"] = arm_step_form(mu_all, C_all, node_slices, G)
    arms["mean-map"] = arm_mean_map(mu_all, node_slices, G)
    rng = np.random.default_rng(seed + 1)
    for nN, name in ((2, "mc-2"), (32, "mc-32")):
        sub = sample_costs(mu_all, C_all, node_slices, G, nN, seed + 100 + nN)
        arms[name] = (float(sub.mean()), float(sub.std(ddof=1)) if nN > 1 else None)

    row = {"mc": {"n_draws": n_ref, "E_mc": E_mc, "sd_mc": sd_mc,
                  "cvar_mc": cvar_mc, "cvar_q": q},
           "arms": {}}
    for name, (E, sd) in arms.items():
        d = {"E": E, "sd": sd,
             "rel_err_E": abs(E / E_mc - 1.0) if E_mc != 0 else float("inf")}
        if sd is not None:
            d["rel_err_sd"] = abs(sd / sd_mc - 1.0)
            d["sd_ratio_to_mc"] = sd / sd_mc
            d["cvar_arm"] = gaussian_cvar(E, sd, q)
        else:
            d["cvar_arm"] = E  # the risk-blind floor
        d["cvar_abs_err"] = abs(d["cvar_arm"] - cvar_mc)
        row["arms"][name] = d
    return row
