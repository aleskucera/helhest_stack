"""Robust trajectory evaluation (CVaR over wheel-slip scenarios) -> emergent clearance.

An obstacle is not a static cell -- it's where the settle tilts / high-centers. So instead of
a hand-set clearance margin, score each candidate trajectory by how it holds up under the
DISTURBANCE: roll it out under K correlated wheel-slip scenarios and take the CVaR (mean of the
worst beta-tail) of its cost. A path that hugs the wall is cheap for the exact nominal, but its
slip-fan drifts into the wall (tilt / infeasible settle) -> bad CVaR; a path with margin stays
feasible across the fan -> good CVaR. So the planner prefers clearance WITHOUT being told a
margin -- it falls out of robustness.

Trick: wheel slip just scales the commanded wheel speeds, so each scenario is baked into the
omega buffer (no engine change). B candidates x K scenarios = B*K rollouts; scenario 0 is the
undisturbed nominal. The existing rollout+settle scores all B*K; we CVaR-reduce the K per
candidate in numpy.

Demo: a fan of arc candidates toward a goal past demo_terrain's wall. The nominal-best candidate
hugs the wall (its fan clips); the CVaR-best keeps clear.

Run:  python -m kinematic_helhest.control.robust [--device cuda] [--out /tmp/robust.png]
"""
import argparse

import numpy as np
import warp as wp

from .. import heightmap as hmmod
from ..engine import GridParams
from ..engine import RobotParams
from ..engine import Simulator
from ..engine import SolverParams

_CLEAR_MARGIN, _RESID_TOL, _WMAX = 0.05, 1e-2, 4.0


def _arc_candidates(turns, T, base=2.5):
    """One constant-(wL, wR) arc per turn rate: wL = base - turn, wR = base + turn (turn > 0
    curves up/left). A fan over `turns` sweeps from straight-into-the-wall to wide-and-clear."""
    cand = np.zeros((len(turns), T, 2), np.float32)
    cand[:, :, 0] = (base - turns)[:, None]
    cand[:, :, 1] = (base + turns)[:, None]
    return np.clip(cand, 0.0, _WMAX)


def _build_omega(cand, slips):
    """cand [B, T, 2] x slips [K, 2] -> omega [T, B*K, 3]; rollout b*K + k = candidate b under
    slip scenario k (wheel speeds scaled by slip -> the unmodeled disturbance, baked in)."""
    B, T, _ = cand.shape
    eff = cand[:, None, :, :] * slips[None, :, None, :]      # [B, K, T, 2]
    eff = eff.reshape(B * len(slips), T, 2)
    rear = eff.mean(2, keepdims=True)
    return np.ascontiguousarray(np.concatenate([eff, rear], 2).transpose(1, 0, 2), np.float32)


def evaluate(scene, mu, start, goal, cand, slips, beta=0.5, T=70, dt=0.1, device="cuda",
             w_term=3.0, w_inv=2.0e3):
    """Roll out every candidate x scenario; return per-candidate nominal cost, CVaR cost, the
    scenario paths [T+1, B, K, 2], and the worst-scenario min clearance per candidate."""
    B, K = len(cand), len(slips)
    sim = Simulator(RobotParams(), SolverParams(dt=dt, k_turn=2.0, newton_iters=6, atol=1e-4),
                    GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0), B * K, T, device)
    sim.set_terrain(wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device))
    sim.set_friction(mu)
    controlled, _, clearance, residual = sim.rollout(_build_omega(cand, slips), start)

    goal = np.asarray(goal[:2], np.float64)
    d_term = np.linalg.norm(controlled[-1, :, :2] - goal, axis=1)            # [B*K]
    clear_viol = np.maximum(_CLEAR_MARGIN - clearance, 0.0).sum(0)            # [B*K]
    resid_viol = np.maximum(residual - _RESID_TOL, 0.0).sum(0)               # [B*K]
    J = (w_term * d_term ** 2 + w_inv * (clear_viol + resid_viol)).reshape(B, K)

    nominal = J[:, 0]                                                         # scenario 0 = no slip
    m = max(1, int(round(beta * K)))                                         # worst beta-tail size
    cvar = np.sort(J, axis=1)[:, -m:].mean(1)                                # CVaR = mean of worst m
    paths = controlled[:, :, :2].reshape(T + 1, B, K, 2)
    min_clear = clearance.reshape(-1, B, K).min(0).min(1)                    # worst-scenario clearance [B]
    bad = (clearance < _CLEAR_MARGIN) | (residual > _RESID_TOL)              # [T, B*K] infeasible step
    badstep = bad.reshape(-1, B, K)                                          # [T, B, K]
    return nominal, cvar, paths, min_clear, badstep


def _plot(scene, goal, paths, i_nom, i_cvar, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nx, ny = scene.nx, scene.ny
    ext = [scene.x0, scene.x0 + nx * scene.cell, scene.y0, scene.y0 + ny * scene.cell]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(scene.H, origin="lower", extent=ext, cmap="terrain", alpha=0.9)
    for idx, color, label in [(i_nom, "crimson", "nominal-best (hugs)"),
                              (i_cvar, "seagreen", "CVaR-best (clears)")]:
        fan = paths[:, idx]                       # [T+1, K, 2]
        for k in range(fan.shape[1]):             # the slip fan
            ax.plot(fan[:, k, 0], fan[:, k, 1], "-", color=color, lw=0.5, alpha=0.25)
        ax.plot(fan[:, 0, 0], fan[:, 0, 1], "-", color=color, lw=2.5, label=label)  # nominal (k=0)
    ax.plot(0, 0, "o", color="k", ms=8)
    ax.plot(*goal[:2], "*", color="red", ms=18, label="goal")
    ax.set_xlim(-0.5, 5.0); ax.set_ylim(-2.0, 3.0); ax.axis("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(loc="upper left")
    ax.set_title("Robust eval: CVaR over slip scenarios picks the trajectory whose fan stays clear")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")


def _animate(scene, goal, paths, badstep, i_nom, i_cvar, out, stride=2, fps=12):
    """Two-panel GIF: watch the K slip scenarios roll out under each arc. A scenario's marker
    turns red once it has high-centered (infeasible settle). The nominal-best panel accumulates
    red; the CVaR-best panel stays green -- the disturbance fan clearing the wall, animated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import PillowWriter

    ext = [scene.x0, scene.x0 + scene.nx * scene.cell, scene.y0, scene.y0 + scene.ny * scene.cell]
    Tp1, K = paths.shape[0], paths.shape[2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    panels = [(i_nom, "nominal-best (hugs)"), (i_cvar, "CVaR-best (clears)")]
    frames = list(range(2, Tp1 + 1, stride))

    def draw(t):
        for ax, (idx, title) in zip(axes, panels):
            ax.clear()
            ax.imshow(scene.H, origin="lower", extent=ext, cmap="terrain", alpha=0.9)
            fan, bad = paths[:, idx], badstep[:, idx]  # [T+1, K, 2], [T, K]
            clipped = bad[: min(t, bad.shape[0])].any(0)  # [K] has scenario k clipped by now
            for k in range(K):
                ax.plot(fan[:t, k, 0], fan[:t, k, 1], "-", color="0.35", lw=0.4, alpha=0.4)
                ax.plot(fan[t - 1, k, 0], fan[t - 1, k, 1], "o", ms=4,
                        color="crimson" if clipped[k] else "seagreen")
            ax.plot(0, 0, "o", color="k", ms=7)
            ax.plot(*goal[:2], "*", color="red", ms=16)
            ax.set_xlim(-0.5, 5.0); ax.set_ylim(-2.0, 3.0)
            ax.set_xlabel("x [m]")
            ax.set_title(f"{title}\n{int(clipped.sum())}/{K} scenarios high-centered")
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    anim.save(out, writer=PillowWriter(fps=fps))
    print(f"saved {out}  ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/robust.png")
    ap.add_argument("--B", type=int, default=48, help="arc candidates")
    ap.add_argument("--K", type=int, default=16, help="slip scenarios per candidate")
    ap.add_argument("--beta", type=float, default=0.5, help="CVaR tail fraction (smaller = more conservative)")
    ap.add_argument("--slip-lo", type=float, default=0.6, help="min wheel-speed retained in a scenario")
    ap.add_argument("--animate", action="store_true", help="save a GIF of the slip fans rolling out")
    args = ap.parse_args()

    scene = hmmod.demo_terrain()
    mu = hmmod.Heightmap(np.full((scene.ny, scene.nx), 0.8, np.float32), (scene.x0, scene.y0), scene.cell)
    start, goal = np.array([0.0, 0.0, 0.0], np.float32), np.array([4.0, 1.6])

    turns = np.linspace(0.0, 1.8, args.B)
    cand = _arc_candidates(turns, T=70)
    rng = np.random.default_rng(0)
    slips = np.ones((args.K, 2), np.float32)
    slips[1:] = rng.uniform(args.slip_lo, 1.0, (args.K - 1, 2))  # scenario 0 = no slip

    nominal, cvar, paths, min_clear, badstep = evaluate(
        scene, mu, start, goal, cand, slips, beta=args.beta, device=args.device)
    i_nom, i_cvar = int(np.argmin(nominal)), int(np.argmin(cvar))
    print(f"B={args.B} candidates, K={args.K} slip scenarios, CVaR beta={args.beta}")
    print(f"  nominal-best: arc#{i_nom} turn={turns[i_nom]:.2f}  nominal={nominal[i_nom]:.1f} "
          f"cvar={cvar[i_nom]:.1f}  worst-scenario min-clear={min_clear[i_nom]:+.3f} m")
    print(f"  CVaR-best   : arc#{i_cvar} turn={turns[i_cvar]:.2f}  nominal={nominal[i_cvar]:.1f} "
          f"cvar={cvar[i_cvar]:.1f}  worst-scenario min-clear={min_clear[i_cvar]:+.3f} m")
    print(f"  -> CVaR picks a {'wider' if turns[i_cvar] > turns[i_nom] else 'tighter'} arc; "
          f"worst-case clearance {min_clear[i_nom]:+.3f} -> {min_clear[i_cvar]:+.3f} m")
    if args.animate:
        out = args.out[:-4] + ".gif" if args.out.endswith(".png") else args.out
        _animate(scene, goal, paths, badstep, i_nom, i_cvar, out)
    else:
        _plot(scene, goal, paths, i_nom, i_cvar, args.out)


if __name__ == "__main__":
    main()
