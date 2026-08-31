"""v2 attitude-cost moments (PREREG_attitude_cost.md sections 2-3).

Pure-numpy reference implementation of the v2 scoring math:

  1. Clark fold over each node's candidates, tracking covariance against
     an augmented universe (raw candidates + already-folded supports), so
     the output is the JOINT Gaussian (m, S) of all supports in a window.
  2. The attitude map: per step, (pitch, roll) are linear in the three
     supports (settle-map rows; the constant vanishes: settle_map @ 1 =
     (1,0,0)), so the stacked attitude vector is x = T e.
  3. The cost L = x'x = e'Ae, A = T'T, a Gaussian quadratic form with
     EXACT moments E[L] = tr(AS) + m'Am, Var[L] = 2tr((AS)^2) + 4m'ASAm.

Step 1 is the campaign's single approximation; steps 2-3 are exact given
(m, S). `python -m studies.baseprod.v2_moments` runs the verification:
(a) the quadratic-form formulas against MC on their own Gaussian
(implementation exactness; runner freeze condition C1), (b) a full-chain
preview against true-max MC on synthetic windows (the deficit the
design-phase correction derivation will quantify on real windows).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# vehicle geometry (mirrors spires/vehicle.py; asserted against it before F1)
HALF_TRACK = 0.365
REAR_OFFSET = 0.75


def settle_pitch_roll_rows() -> np.ndarray:
    """d(pitch, roll)/d(env_L, env_R, env_rear)."""
    b, ell = HALF_TRACK, REAR_OFFSET
    return np.array([[-0.5 / ell, -0.5 / ell, 1.0 / ell],
                     [1.0 / (2.0 * b), -1.0 / (2.0 * b), 0.0]])


def _clark_pair(m1, v1, m2, v2, c12, cz1, cz2):
    """max of jointly Gaussian (X1, X2): matched mean/var and Cov(max, Z)
    for tracked Z given cz1 = Cov(X1, Z), cz2 = Cov(X2, Z). Clark (1961)."""
    a2 = v1 + v2 - 2.0 * c12
    if a2 < 1e-16:
        return (m1, v1, cz1) if m1 >= m2 else (m2, v2, cz2)
    a = np.sqrt(a2)
    al = (m1 - m2) / a
    Phi, phi = norm.cdf(al), norm.pdf(al)
    m = m1 * Phi + m2 * (1 - Phi) + a * phi
    m2nd = (m1 * m1 + v1) * Phi + (m2 * m2 + v2) * (1 - Phi) + (m1 + m2) * a * phi
    v = max(m2nd - m * m, 1e-14)
    return m, v, Phi * cz1 + (1 - Phi) * cz2


def fold_supports(mu_all, C_all, node_slices):
    """Fold every node; return (m, S) of the joint support Gaussian.

    mu_all (N,), C_all (N, N): all candidates of all nodes, jointly Gaussian.
    node_slices: list of index arrays, one per node (its candidates in mu_all).
    """
    n = len(node_slices)
    N = len(mu_all)
    # augmented universe: raw candidates then folded supports
    M = np.zeros((N + n, N + n))
    M[:N, :N] = C_all
    mu_aug = np.concatenate([mu_all, np.zeros(n)])
    for i, idx in enumerate(node_slices):
        order = idx[np.argsort(-mu_all[idx])]      # stable on ties, like v1
        j0 = order[0]
        shift = mu_aug[j0]                         # v1's per-node shift: the fold
        m_run, v_run = 0.0, M[j0, j0]              # is shift-equivariant, and the
        c_run = M[j0, :N + i].copy()               # shift avoids 400 m elevations
        for j in order[1:]:                        # cancelling against cm spreads
            m_run, v_run, c_run = _clark_pair(
                m_run, v_run, mu_aug[j] - shift, M[j, j],
                c_run[j] if j < N + i else 0.0,
                c_run, M[j, :N + i])
        k = N + i
        M[k, :N + i] = c_run
        M[:N + i, k] = c_run
        M[k, k] = v_run
        mu_aug[k] = m_run + shift
    m = mu_aug[N:]
    S = M[N:, N:]
    return m, S


def onehot_supports(mu_all, C_all, node_slices):
    """fosm's supports: the mean-map winner per node, moments exact for
    that (frozen) selection."""
    w = np.array([idx[np.argmax(mu_all[idx])] for idx in node_slices])
    return mu_all[w].copy(), C_all[np.ix_(w, w)].copy()


def attitude_operator(n_steps):
    """T mapping stacked supports (L,R,rear per step) to stacked (pitch, roll)."""
    R = settle_pitch_roll_rows()
    T = np.zeros((2 * n_steps, 3 * n_steps))
    for t in range(n_steps):
        T[2 * t:2 * t + 2, 3 * t:3 * t + 3] = R
    return T


def quadratic_moments(m, S, A):
    """Exact E, Var of e'Ae for e ~ N(m, S)."""
    AS = A @ S
    E = np.trace(AS) + m @ A @ m
    V = 2.0 * np.trace(AS @ AS) + 4.0 * m @ A @ S @ A @ m
    return float(E), float(max(V, 0.0))


def attitude_cost_moments(m, S, n_steps):
    T = attitude_operator(n_steps)
    return quadratic_moments(m, S, T.T @ T)


def gaussian_cvar(E, sd, q=0.9):
    return E + norm.pdf(norm.ppf(q)) / (1.0 - q) * sd


# ------------------------------- verification -------------------------------

def _rand_window(rng, n_steps=4, k=4, rho=0.6):
    n_nodes = 3 * n_steps
    N = n_nodes * k
    # correlated cell field: exponential-decay covariance over a line of cells
    x = np.arange(N) * 0.35
    C = 0.03 ** 2 * np.exp(-np.abs(x[:, None] - x[None, :]) / 2.0)
    C += np.diag(rng.uniform(0.0005, 0.004, N) ** 1) * 0.5
    mu = rng.normal(0.0, 0.05, N)
    # tighten some contests: make candidates within a node close
    for i in range(n_nodes):
        mu[i * k:(i + 1) * k] += rng.normal(0, 0.02)
    slices = [np.arange(i * k, (i + 1) * k) for i in range(n_nodes)]
    return mu, C, slices, n_steps


def selftest(seed=0, n_mc=400_000):
    rng = np.random.default_rng(seed)
    print("== (a) quadratic-form exactness on its own Gaussian ==")
    mu, C, slices, T = _rand_window(rng)
    m, S = fold_supports(mu, C, slices)
    E, V = attitude_cost_moments(m, S, T)
    L = np.linalg.cholesky(S + 1e-12 * np.eye(len(S)))
    e = m + rng.standard_normal((n_mc, len(m))) @ L.T
    Top = attitude_operator(T)
    x = e @ Top.T
    Ls = np.einsum("ij,ij->i", x, x)
    print(f"   E  formula {E:.6e}  MC {Ls.mean():.6e}  rel {abs(E/Ls.mean()-1):.2e}")
    print(f"   sd formula {np.sqrt(V):.6e}  MC {Ls.std():.6e}  rel {abs(np.sqrt(V)/Ls.std()-1):.2e}")

    print("== (b) full chain vs true-max MC (deficit preview, synthetic) ==")
    for s2 in range(3):
        mu, C, slices, T = _rand_window(np.random.default_rng(100 + s2))
        m, S = fold_supports(mu, C, slices)
        E, V = attitude_cost_moments(m, S, T)
        mo, So = onehot_supports(mu, C, slices)
        Ef, Vf = attitude_cost_moments(mo, So, T)
        Lc = np.linalg.cholesky(C + 1e-12 * np.eye(len(C)))
        h = mu + np.random.default_rng(7).standard_normal((n_mc, len(mu))) @ Lc.T
        sup = np.stack([h[:, idx].max(axis=1) for idx in slices], axis=1)
        x = sup @ attitude_operator(T).T
        Ls = np.einsum("ij,ij->i", x, x)
        print(f"   [{s2}] clark relE {abs(E/Ls.mean()-1):.3e}  sd-ratio {np.sqrt(V)/Ls.std():.4f}"
              f"   | fosm relE {abs(Ef/Ls.mean()-1):.3e}  sd-ratio {np.sqrt(Vf)/Ls.std():.4f}")


if __name__ == "__main__":
    selftest()


def fold_supports_fast(means, cov_self, C, u_idx, lam_fold=None):
    """GPU-shaped equivalent of fold_supports for the real-window layout.

    Uses the vectorized v1 recursion (clark_fold) for per-node (m, v, lam),
    then the SSTA identity: Cov(sup_i, sup_j) = lam_i C lam_j' for i != j,
    with Clark's exact variance on the diagonal. Returns (m, S). This is
    the formulation the Warp port implements: never materialize node-space
    S at O(n^2) when the cost only needs x-space -- callers can contract
    G S G' as (G Lam) C (G Lam)' + G diag(v - lamClam) G'.
    """
    from spires.risk_calibration import clark_fold
    m, v, lam = clark_fold(means, cov_self)
    n, k = means.shape
    # W[i, :] = lam_i gathered to unique-cell space
    n_u = C.shape[0]
    W = np.zeros((n, n_u))
    rows = np.repeat(np.arange(n), k)
    np.add.at(W, (rows, u_idx.ravel()), lam.ravel())
    S = W @ C @ W.T
    np.fill_diagonal(S, v)
    return m, S
