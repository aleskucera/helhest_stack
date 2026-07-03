"""Motor actuator lag comparison demo (Warp).

Runs the same open-loop wheel-speed programme through the Warp ForwardSimulator twice —
once with instantaneous motors (tau_motor = 0) and once with a 0.5 s first-order lag —
and plots the differences in a three-panel figure:

  Panel 1 -- Wheel speed: commanded vs actual (left wheel) for each tau.
             Shows the first-order lag filter response to a step command.
  Panel 2 -- XY trajectory: top-down path for each tau.
             Shows how lag causes the robot to overshoot straight before a turn
             takes effect, compounding into a different exit path.
  Panel 3 -- Forward distance x(t): accumulates the positional gap between an
             instantaneous-motor robot and a laggy one.

    python demos/motor_lag_comparison.py
    python demos/motor_lag_comparison.py --out /tmp/motor_lag.png
    python demos/motor_lag_comparison.py --device cuda:0
"""

import argparse

import numpy as np
import warp as wp

from helhest import dynamics
from helhest import friction as friction_mod
from helhest import heightmap as hm_mod
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams

# --- scenario parameters ---
DT = 0.1  # [s] control timestep
PHASES = [
    # (n_steps, omega_cmd [L, R, rear])
    (30, [2.5,  2.5, 2.5]),   # straight forward  -- shows acceleration ramp-up
    (30, [2.5, -1.0, 0.75]),  # asymmetric turn left -- shows turn-response delay
    (30, [2.5,  2.5, 2.5]),   # straight forward again -- lag from turn exit compounds
]

TAUS   = [0.0, 0.5]
COLORS = ["steelblue", "crimson"]
LABELS = ["tau = 0 s  (instantaneous)", "tau = 0.5 s  (lagged)"]

# terrain large enough that the robot never leaves the grid
XLIM = (-2.0, 14.0)
YLIM = (-8.0,  8.0)


def _build_setpoints() -> np.ndarray:
    """Concatenate phase blocks into a [T, 3] setpoints array."""
    blocks = [
        np.tile(np.asarray(cmd, np.float64), (n, 1))
        for n, cmd in PHASES
    ]
    return np.vstack(blocks)


def _omega_actual_log(setpoints: np.ndarray, tau: float) -> np.ndarray:
    """Compute the omega_actual trajectory analytically (left wheel only).

    Mirrors the device motor_lag_step @wp.func: alpha = min(dt/max(tau, 1e-6), 1).
    Returns shape [T] — effective left-wheel speed used at each step.
    """
    alpha = min(DT / max(tau, 1e-6), 1.0)
    T = len(setpoints)
    oa = 0.0
    log = np.empty(T)
    for t in range(T):
        oa = oa + alpha * (setpoints[t, 0] - oa)
        log[t] = oa
    return log


def run_warp(
    tau: float,
    setpoints: np.ndarray,
    scene: hm_mod.Heightmap,
    mu: hm_mod.Heightmap,
    grid: GridParams,
    device: str,
) -> np.ndarray:
    """Run one ForwardSimulator rollout and return controlled [T+1, 3] (x, y, yaw)."""
    T = len(setpoints)
    solver = dynamics.planning_solver(dt=DT)
    solver.tau_motor = tau
    sim = ForwardSimulator(dynamics.robot_params(), solver, grid, batch_size=1, n_steps=T, device=device)
    sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    sim.set_friction(mu)
    wheel_omega = np.ascontiguousarray(setpoints[:, None, :], np.float32)  # [T, 1, 3]
    controlled, *_ = sim.rollout(wheel_omega, (0.0, 0.0, 0.0), np.zeros(3))
    return controlled[:, 0, :]  # [T+1, 3]


def run(out: str | None = None, device: str = "cpu") -> None:
    import matplotlib.pyplot as plt

    setpoints = _build_setpoints()
    T = len(setpoints)
    time = np.arange(T) * DT  # [s]

    scene = hm_mod.flat(xlim=XLIM, ylim=YLIM)
    mu    = friction_mod.uniform(0.8, xlim=XLIM, ylim=YLIM)
    grid  = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)

    # phase boundary times for shading
    phase_times = np.cumsum([0] + [n * DT for n, _ in PHASES])  # [0, 3, 6, 9]
    # step indices where phases begin/end (for trajectory markers)
    phase_idx = [sum(n for n, _ in PHASES[:k]) for k in range(len(PHASES) + 1)]

    # collect warp rollouts
    results: dict[float, np.ndarray] = {}
    for tau in TAUS:
        results[tau] = run_warp(tau, setpoints, scene, mu, grid, device)

    # --- figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Motor actuator lag: Warp simulation — tau = 0 s vs tau = 0.5 s",
        fontsize=12,
    )

    # ---------- Panel 1: wheel speed (left wheel) ----------
    ax = axes[0]
    ax.set_title("Left-wheel speed: commanded vs actual")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("wheel speed [rad/s]")

    for i, (t0, t1) in enumerate(zip(phase_times[:-1], phase_times[1:])):
        ax.axvspan(t0, t1, alpha=0.07, color="gray", zorder=0)

    ax.step(time, setpoints[:, 0], color="gray", linewidth=1.5, linestyle="--",
            where="post", label="omega_cmd", zorder=5)
    for tau, color, label in zip(TAUS, COLORS, LABELS):
        oa = _omega_actual_log(setpoints, tau)
        ax.plot(time, oa, color=color, linewidth=2.0, label=label)
    ax.legend(fontsize=9)
    ax.set_xlim(0, time[-1])

    # ---------- Panel 2: XY trajectory ----------
    ax = axes[1]
    ax.set_title("XY trajectory (top view)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")

    for tau, color, label in zip(TAUS, COLORS, LABELS):
        pose = results[tau]  # [T+1, 3]; index 0 is the start pose (0,0,0)
        xs, ys = pose[:, 0], pose[:, 1]
        ax.plot(xs, ys, color=color, linewidth=2.0, label=label)
        # arrow at the end showing final heading
        dx = xs[-1] - xs[-2]
        dy = ys[-1] - ys[-2]
        ax.annotate("", xy=(xs[-1], ys[-1]),
                    xytext=(xs[-1] - dx * 3, ys[-1] - dy * 3),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    # phase-change markers on the tau=0 reference path
    ref = results[0.0]
    markers = ["^", "s"]
    for idx, marker in zip(phase_idx[1:-1], markers):
        ax.plot(ref[idx, 0], ref[idx, 1], marker, color="black", ms=8, zorder=6)

    ax.plot(ref[0, 0], ref[0, 1], "o", color="black", ms=9, zorder=7, label="start")
    ax.legend(fontsize=9)

    # ---------- Panel 3: forward distance x(t) ----------
    ax = axes[2]
    ax.set_title("Forward distance over time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("x [m]")

    for i, (t0, t1) in enumerate(zip(phase_times[:-1], phase_times[1:])):
        ax.axvspan(t0, t1, alpha=0.07, color="gray", zorder=0, label=f"phase {i + 1}")

    # controlled[0] is the initial pose; time axis: 0, dt, 2*dt, ..., T*dt
    time_full = np.concatenate([[0.0], time + DT])  # [T+1] shifted so index 0 = t=0
    for tau, color, label in zip(TAUS, COLORS, LABELS):
        xs = results[tau][:, 0]
        ax.plot(time_full, xs, color=color, linewidth=2.0, label=label)

    ax.legend(fontsize=9)
    ax.set_xlim(0, time_full[-1])

    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"saved {out}")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="save figure to this path instead of showing it")
    ap.add_argument("--device", default="cpu", help="Warp device (default: cpu)")
    args = ap.parse_args()
    run(out=args.out, device=args.device)


if __name__ == "__main__":
    main()
