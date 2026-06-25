"""Tube / robust MPPI demo (option F): hold the real robot on the MPPI nominal under
wheel slip the planner does not model.

MPPI plans on the clean kinematic model, but the world robot's wheels SLIP (asymmetric,
unmodeled), so the planned controls produce a different motion than intended. We replan
SLOWLY (every `replan_every` steps -- planning is expensive) and, between replans, drive
the slipping world either:

  * OPEN-LOOP  -- apply the planned wheel speeds directly (no feedback between replans), or
  * TUBE       -- a fast ancillary tracker (pure-pursuit on the nominal path) re-aims at
                  the plan every step, keeping the robot inside a tube around it.

The tube's value is exactly the gap between these: open-loop drifts off the nominal within
each replan window (slip accumulates); the tracker holds it on. This is the sim stand-in
for sim-to-real -- the disturbance IS the model error a real skid-steer would have.

Run:  python -m kinematic_helhest.control.tube [--device cuda] [--out /tmp/tube.png]
"""
import argparse

import numpy as np
import warp as wp

from .. import heightmap as hmmod
from ..engine import GridParams
from ..engine import RobotParams
from ..engine import Simulator
from ..engine import SolverParams
from .mppi_gpu import MppiGpu


def _track(state, nominal_poses, nominal_U, local_t, k_lat=3.0, k_yaw=2.0, wmax=4.0):
    """Ancillary tracker: the planned control (feedforward) + a feedback correction on the
    pose error, run every step to reject the slip between (slow) replans. The feedforward is
    the nominal wheel speeds; the feedback turns toward the reference pose -- lateral + heading
    error in the robot frame drive a differential wheel correction (the tube-keeping law)."""
    ti = min(local_t, len(nominal_U) - 1)
    wl_n, wr_n = float(nominal_U[ti, 0]), float(nominal_U[ti, 1])
    ref = nominal_poses[min(local_t, len(nominal_poses) - 1)]
    yaw = float(state[2])
    dx, dy = ref[0] - state[0], ref[1] - state[1]
    e_lat = -np.sin(yaw) * dx + np.cos(yaw) * dy             # lateral error, +left of heading
    e_yaw = (float(ref[2]) - yaw + np.pi) % (2 * np.pi) - np.pi
    corr = k_lat * e_lat + k_yaw * e_yaw                     # differential turn-rate correction
    return np.array([np.clip(wl_n - corr, 0.0, wmax), np.clip(wr_n + corr, 0.0, wmax)], np.float32)


def _world_step(world, state, wheels, slip):
    """Step the disturbed world one tick: the commanded wheels SLIP (retain slip[i] of their
    speed) before the kinematics see them -- the unmodeled disturbance."""
    wl = float(wheels[0]) * slip[0]
    wr = float(wheels[1]) * slip[1]
    omega = np.array([[[wl, wr, 0.5 * (wl + wr)]]], np.float32)  # [T=1, B=1, 3]
    controlled, _, _, _ = world.rollout(omega, tuple(float(v) for v in state))
    return controlled[1, 0].astype(np.float32)


def run(scene, mu, start, goal, slip, tube, T=60, B=8192, n_refine=3, replan_every=30,
        max_steps=300, dt=0.1, goal_tol=0.3, device="cuda", seed=0):
    """One drive of the slipping world toward the goal; tube=False is open-loop. Returns the
    driven path, the per-step cross-track error from the nominal, reached, final distance."""
    params = SolverParams(dt=dt, k_turn=2.0, newton_iters=6, atol=1e-4)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    terr = wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)

    plan_sim = Simulator(RobotParams(), params, grid, B, T, device)  # clean model the planner uses
    plan_sim.set_terrain(terr); plan_sim.set_friction(mu)
    world = Simulator(RobotParams(), params, grid, 1, 1, device)      # the disturbed "real" robot
    world.set_terrain(terr); world.set_friction(mu)

    w = dict(term=3.0, run=0.3, head=2.0, invalid=1e5, eff=2e-3, smooth=2e-3)
    drv = MppiGpu(plan_sim, 0.5, 4.0, w, 0.05, 1e-2, seed, sigma_knot=1.0, n_knots=4)
    drv.reset_nominal(1.5)

    goal = np.asarray(goal[:2], np.float64)
    state = np.asarray(start, np.float32)
    path = [state.copy()]
    track_err = []
    nominal_poses, nominal_U, local_t = None, None, 0
    reached = False
    for k in range(max_steps):
        if np.linalg.norm(state[:2] - goal) < goal_tol:
            reached = True
            break
        if k % replan_every == 0:  # slow replan from the actual (drifted) world state
            drv.replan(state, goal, n_refine)
            nominal_poses = plan_sim.controlled[:, 0].numpy().copy()  # [T+1, 3]
            nominal_U = drv.nominal().copy()                          # [T, 2]
            local_t = 0
        # cross-track error = distance from the world pose to the nominal path (the tube width)
        track_err.append(float(np.min(np.linalg.norm(nominal_poses[:, :2] - state[:2], axis=1))))
        if tube:
            wheels = _track(state, nominal_poses, nominal_U, local_t)
        else:
            wheels = nominal_U[min(local_t, len(nominal_U) - 1)]  # open-loop planned control
        state = _world_step(world, state, wheels, slip)
        path.append(state.copy())
        local_t += 1
    return np.array(path), np.array(track_err), reached, float(np.linalg.norm(state[:2] - goal))


def _plot(scene, goal, open_path, tube_path, open_err, tube_err, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(open_path[:, 0], open_path[:, 1], "-", color="crimson", lw=2, label="open-loop (drifts)")
    ax1.plot(tube_path[:, 0], tube_path[:, 1], "-", color="seagreen", lw=2, label="tube (tracked)")
    ax1.plot(0, 0, "o", color="k", ms=8); ax1.plot(*goal[:2], "*", color="red", ms=18, label="goal")
    ax1.set_title("Driven path under wheel slip"); ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.legend(loc="upper left"); ax1.axis("equal"); ax1.grid(alpha=0.3)
    ax2.plot(open_err, color="crimson", lw=1.5, label="open-loop")
    ax2.plot(tube_err, color="seagreen", lw=1.5, label="tube")
    ax2.set_title("Cross-track error from the nominal (tube width)")
    ax2.set_xlabel("step"); ax2.set_ylabel("error [m]"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/tube.png")
    ap.add_argument("--slip-l", type=float, default=0.9, help="left-wheel speed retained (1=no slip)")
    ap.add_argument("--slip-r", type=float, default=0.7, help="right-wheel speed retained (1=no slip)")
    ap.add_argument("--replan-every", type=int, default=50,
                    help="steps between (slow) replans -- a long window is where a tube matters")
    args = ap.parse_args()

    half, cell = 7.0, 0.06
    n = int(2 * half / cell)
    scene = hmmod.Heightmap(np.zeros((n, n), np.float32), (-half, -half), cell)  # flat: isolate the slip
    mu = hmmod.Heightmap(np.full((n, n), 0.8, np.float32), (-half, -half), cell)  # matching-dims friction
    start, goal = np.array([0.0, 0.0, 0.0], np.float32), np.array([6.0, 0.0])
    slip = (args.slip_l, args.slip_r)

    print(f"wheel slip retention L={slip[0]} R={slip[1]}, replan every {args.replan_every} steps")
    op, oe, orch, od = run(scene, mu, start, goal, slip, tube=False, replan_every=args.replan_every, device=args.device)
    tp, te, trch, td = run(scene, mu, start, goal, slip, tube=True, replan_every=args.replan_every, device=args.device)
    print(f"  open-loop: reached={orch} final_dist={od:.2f} steps={len(op)-1}  "
          f"cross-track mean={oe.mean():.3f} max={oe.max():.3f} m")
    print(f"  tube     : reached={trch} final_dist={td:.2f} steps={len(tp)-1}  "
          f"cross-track mean={te.mean():.3f} max={te.max():.3f} m")
    print(f"  tube tightens the track {oe.mean() / max(te.mean(), 1e-6):.1f}x (mean cross-track)")
    _plot(scene, goal, op, tp, oe, te, args.out)


if __name__ == "__main__":
    main()
