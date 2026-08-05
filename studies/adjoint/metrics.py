"""Probe-cell selection and the stratified adjoint-vs-FD statistics.

Four design points that the obvious version gets wrong:

**Probe the ZERO set too.** Checking only the cells where the adjoint is already nonzero
cannot detect a *missing* gradient term -- the adjoint reports zero, is never probed, and
passes. The historic `sample_field` position-gradient bug in this repo was exactly a dropped
term. `probe_cells` adds a control set of cells the adjoint calls zero but that sit within a
few cells of its support, where a misplaced contribution would land.

**Score only ACTIVE pairs.** Each probe cell is compared against all B rollouts, but a cell
lies in only one or two rollouts' support; the rest are zero-vs-zero and would drag every
percentile to exactly 0. Pairs where both the adjoint and the FD are negligible are counted
and set aside, not averaged in.

**Split smooth from kinked before scoring.** With one-sided differences in hand, the kink
ratio |D+ - D-| / (|D+| + |D-|) says whether the forward is differentiable at that cell at
all. A correctness claim only means something on smooth pairs; on kinked pairs the central
difference reports the AVERAGE of two different one-sided slopes and no adjoint can match it.

**Guard the relative-error denominator.** An unguarded |a - f| / |f| explodes on cells whose
true sensitivity is ~0, which are the cells that matter least; the group's own scale is the
right floor.
"""

from __future__ import annotations

import numpy as np

from .scene import REGION_NAMES

KAPPA = 1e-2  # relative-error floor, as a fraction of the group's largest |g_fd|
# A pair counts only if max(|adj|, |fd|) exceeds this fraction of the group's scale. 1% is
# not arbitrary: below it the float32 cancellation noise of the finite difference itself
# (|J| * 6e-8 / eps) is comparable to the signal, so both the relative error AND the kink
# ratio become noise. A cell at 1% of the peak sensitivity is also below any threshold a
# sensing policy would act on, so nothing decision-relevant is being excluded.
ACTIVE_FRAC = 1e-2
KINK_TOL = 0.05  # |D+ - D-| / (|D+| + |D-|) above this = the forward is kinked here


def probe_cells(
    grads: np.ndarray,
    region: np.ndarray,
    per_region: int = 160,
    n_zero: int = 160,
    halo: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick cells to finite-difference, stratified by region.

    `grads` is [N_TERMS, B, ny, nx]. Each region contributes a top-|gradient| quota PER TERM
    plus a random sample from the region's whole support. Ranking per term matters: the
    terms differ in support size by two orders of magnitude (one init settle touches ~12
    cells, the belly clearance ~2000), so a single pooled ranking would let the largest term
    crowd the smallest one out of the probe set entirely.

    Returns (cells [n, 2] as (iy, ix), is_zero_control [n] bool).
    """
    rng = np.random.default_rng(seed)
    n_terms = grads.shape[0]
    per_term = np.abs(grads).max(axis=1)  # [N_TERMS, ny, nx]
    strength = per_term.max(axis=0)
    support = strength > 0.0
    quota_top = max(per_region // (2 * n_terms), 1)

    picked: list[np.ndarray] = []
    for code in range(len(REGION_NAMES) - 1):  # OTHER is never touched by a wheel
        in_region = support & (region == code)
        if not in_region.any():
            continue
        for k in range(n_terms):
            idx = np.argwhere(in_region & (per_term[k] > 0.0))
            if len(idx):
                order = np.argsort(-per_term[k][idx[:, 0], idx[:, 1]])
                picked.append(idx[order][:quota_top])
        idx = np.argwhere(in_region)
        n_rand = min(per_region // 2, len(idx))
        picked.append(idx[rng.choice(len(idx), n_rand, replace=False)])
    cells = np.unique(np.concatenate(picked), axis=0)

    # Zero control: adjoint says exactly 0, but within `halo` cells of the support, so a
    # misplaced stencil or a dropped term lands here rather than nowhere.
    near = np.zeros_like(support)
    iy, ix = np.nonzero(support)
    ny, nx = support.shape
    for dy in range(-halo, halo + 1):
        for dx in range(-halo, halo + 1):
            near[np.clip(iy + dy, 0, ny - 1), np.clip(ix + dx, 0, nx - 1)] = True
    zero_idx = np.argwhere(near & ~support & (region != len(REGION_NAMES) - 1))
    n_zero = min(n_zero, len(zero_idx))
    zeros = zero_idx[rng.choice(len(zero_idx), n_zero, replace=False)] if n_zero else zero_idx

    all_cells = np.concatenate([cells, zeros])
    is_zero = np.zeros(len(all_cells), bool)
    is_zero[len(cells) :] = True
    return all_cells, is_zero


def compare(
    grads: np.ndarray,
    d_plus: np.ndarray,
    d_minus: np.ndarray,
    cells: np.ndarray,
    is_zero: np.ndarray,
    region: np.ndarray,
    term: int,
) -> dict[str, dict[str, float]]:
    """Stratified adjoint-vs-FD statistics for one term, keyed by region name.

    `grads` [N_TERMS, B, ny, nx]; `d_plus`/`d_minus` [n_cells, N_TERMS, B]. Every (cell,
    rollout) pair is one sample. The zero-control cells are pooled into their own group.
    """
    adj = grads[term][:, cells[:, 0], cells[:, 1]].T  # [n_cells, B]
    dp, dm = d_plus[:, term, :], d_minus[:, term, :]
    central = 0.5 * (dp + dm)
    codes = region[cells[:, 0], cells[:, 1]]

    scale = float(np.abs(central[~is_zero]).max()) if (~is_zero).any() else 0.0
    out: dict[str, dict[str, float]] = {}
    for code in range(len(REGION_NAMES) - 1):
        sel = (codes == code) & ~is_zero
        if sel.any():
            out[REGION_NAMES[code]] = _stats(adj[sel], central[sel], dp[sel], dm[sel], scale)
    if is_zero.any():
        out["zero-control"] = _stats(
            adj[is_zero], central[is_zero], dp[is_zero], dm[is_zero], scale
        )
    return out


def _stats(
    a: np.ndarray, f: np.ndarray, dp: np.ndarray, dm: np.ndarray, scale: float
) -> dict[str, float]:
    """Split the pairs into inactive / kinked / smooth, then score the smooth ones.

    The regression slope is reported alongside the percentiles because it is the statistic
    that would have read ~0.53 for the historic ~47% friction-gradient error -- far more
    legible than any percentile. A slope of exactly 0.5 on kinked pairs is the signature of
    a central difference straddling a kink, not of a halved gradient.
    """
    a, f, dp, dm = a.ravel(), f.ravel(), dp.ravel(), dm.ravel()
    floor = KAPPA * max(scale, 1e-30)
    active = np.maximum(np.abs(a), np.abs(f)) > ACTIVE_FRAC * max(scale, 1e-30)
    kink = np.abs(dp - dm) / np.maximum(np.abs(dp) + np.abs(dm), 1e-30)
    smooth = active & (kink <= KINK_TOL)

    st = {
        "n_pairs": int(a.size),
        "n_active": int(active.sum()),
        "n_smooth": int(smooth.sum()),
        "kinked_frac": float((active & ~smooth).sum() / max(active.sum(), 1)),
        "kink_median_active": float(np.median(kink[active])) if active.any() else 0.0,
        "max_abs_grad": float(np.abs(f).max()),
        # False zeros: the adjoint says exactly zero where the FD sees real sensitivity.
        "false_zero_frac": float(
            ((a == 0.0) & (np.abs(f) > ACTIVE_FRAC * max(scale, 1e-30))).sum() / max(a.size, 1)
        ),
        "false_zero_worst": (
            float(np.abs(f[a == 0.0]).max() / max(scale, 1e-30)) if (a == 0.0).any() else 0.0
        ),
    }
    st.update(_fit(a[smooth], f[smooth], floor, ""))
    st.update(_fit(a[active & ~smooth], f[active & ~smooth], floor, "kinked_"))
    # On kinked pairs, does the adjoint at least agree with ONE of the one-sided slopes?
    if (active & ~smooth).any():
        k = active & ~smooth
        best = np.minimum(np.abs(a[k] - dp[k]), np.abs(a[k] - dm[k]))
        st["kinked_onesided_rel_p90"] = float(
            np.percentile(best / (np.maximum(np.abs(dp[k]), np.abs(dm[k])) + floor), 90)
        )
    return st


def _fit(a: np.ndarray, f: np.ndarray, floor: float, prefix: str) -> dict[str, float]:
    keys = ("rel_l2", "rel_median", "rel_p90", "rel_max", "slope", "cosine")
    if a.size == 0:
        return {f"{prefix}{k}": float("nan") for k in keys}
    rel = np.abs(a - f) / (np.abs(f) + floor)
    denom = float(np.dot(f, f))
    na, nf = float(np.linalg.norm(a)), float(np.linalg.norm(f))
    return {
        # rel_l2 is the primary correctness number: a per-group vector relative error, which
        # unlike a percentile is not dominated by the smallest-magnitude pairs in the group.
        f"{prefix}rel_l2": float(np.linalg.norm(a - f) / nf) if nf > 0.0 else float("nan"),
        f"{prefix}rel_median": float(np.median(rel)),
        f"{prefix}rel_p90": float(np.percentile(rel, 90)),
        f"{prefix}rel_max": float(rel.max()),
        f"{prefix}slope": float(np.dot(a, f) / denom) if denom > 0.0 else float("nan"),
        f"{prefix}cosine": float(np.dot(a, f) / (na * nf)) if na * nf > 0.0 else float("nan"),
    }


def plateau(errors: list[float], tol: float = 3.0, run: int = 3) -> tuple[bool, int]:
    """Is there a plateau in the error-vs-epsilon curve?

    THE load-bearing diagnostic of Study A. A correct adjoint differencing a smooth forward
    shows a flat middle band: too small an epsilon is float32 cancellation noise, too large
    is truncation error. Large error WITH a plateau means the adjoint is wrong. Large error
    WITHOUT one means no finite difference converges at that cell -- a property of the
    physics, not a bug, and precisely what Study B is about.

    Returns (has_plateau, index of the plateau centre, or of the minimum if there is none).
    """
    e = np.asarray(errors, float)
    e = np.where(np.isnan(e), np.inf, e)  # no smooth pairs at this eps -> not a candidate
    e = np.where(e <= 0.0, np.finfo(float).tiny, e)  # an exact 0 is a plateau of its own
    if not np.isfinite(e).any():
        return False, 0
    for i in range(len(e) - run + 1):
        window = e[i : i + run]
        if np.isfinite(window).all() and window.max() / window.min() <= tol:
            return True, i + run // 2
    return False, int(np.argmin(e))
