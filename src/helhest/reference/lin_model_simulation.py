"""Open-loop EKF predict replay on flat ground (no ROS, no measurement update).

Replicates the predict block in elevation_node_ekf.py using predict_q6d,
jacobian_F_6d_analytical, and EKF6D.predict. Input is a CSV of wheel speeds and
optional gyro yaw rate, or a built-in synthetic maneuver sequence.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.figure import Figure

from helhest import dynamics
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.filtering.ekf import EKF6D
from helhest.filtering.jacobian import jacobian_F_6d_analytical
from helhest.filtering.jacobian import predict_q6d
from helhest.model import HALF_TRACK
from helhest.model import WHEEL_RADIUS

# ---------------------------------------------------------------------------
# Knobs — edit these to change input source and simulator tuning.
# ---------------------------------------------------------------------------
INPUT_CSV: pathlib.Path | None = None
DEVICE: str = "cuda:0"
INIT_POSE: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, psi [m, m, rad]
FRICTION: float = 0.8
K_TURN: float = dynamics.K_TURN_INDOOR  # matches elevation_node_ekf _build_ekf default
ROS_JOINT_ORDER: bool = False  # True if CSV columns are ROS [omega_l, omega_rear, omega_r]
SHOW_SIGMA_BAND: bool = True
QUIVER_EVERY: int = 5
OUT_PNG: pathlib.Path | None = None  # None -> plt.show()

_SIG_P0 = np.array([0.10, 0.10, np.deg2rad(2.0), 0.30, 0.30, 0.20])
_SIG_Q = np.array([0.02, 0.02, np.deg2rad(0.5), 0.15, 0.15, 0.10])
_SIG_R_ICP = np.array([0.05, 0.05, np.deg2rad(1.0)])
P0 = np.diag(_SIG_P0**2)
Q = np.diag(_SIG_Q**2)
R_ICP = np.diag(_SIG_R_ICP**2)

_REQUIRED_COLS = ("omega_l", "omega_r", "omega_rear")


def _reorder_ros_joint_velocities(u: np.ndarray) -> np.ndarray:
    """ROS /joint_states [ω_L, ω_rear, ω_R] -> model [ω_L, ω_R, ω_rear]."""
    out = u.copy()
    out[..., 1] = u[..., 2]
    out[..., 2] = u[..., 1]
    return out


def load_commands(
    path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load per-step wheel commands and optional gyro from a named-header CSV.

    Returns (t_sec [T], u [T,3], gyro_z [T] or None when column absent / all NaN).
    """
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.ndim == 0:
        data = np.array([data])

    names = set(data.dtype.names or ())
    missing = [c for c in _REQUIRED_COLS if c not in names]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}; have {sorted(names)}")

    u = np.column_stack([data["omega_l"], data["omega_r"], data["omega_rear"]]).astype(np.float64)
    if ROS_JOINT_ORDER:
        u = _reorder_ros_joint_velocities(u)

    if "t_sec" in names:
        t_sec = np.asarray(data["t_sec"], dtype=np.float64)
    else:
        t_sec = np.arange(len(u), dtype=np.float64) * dynamics.DT

    gyro_z: np.ndarray | None = None
    if "gyro_z" in names:
        gyro_z = np.asarray(data["gyro_z"], dtype=np.float64)
        if np.all(np.isnan(gyro_z)):
            gyro_z = None

    return t_sec, u, gyro_z


def synthetic_commands() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straight -> left arc -> right arc -> spin-in-place at dynamics.DT."""
    dt = dynamics.DT
    alpha = 1.0 + K_TURN * FRICTION
    b = HALF_TRACK
    r = WHEEL_RADIUS

    segments: list[tuple[int, float, float, float]] = [
        (30, 2.0, 2.0, 2.0),  # straight
        (40, 2.0, 3.5, 2.75),  # left arc
        (40, 3.5, 2.0, 2.75),  # right arc
        (25, -2.5, 2.5, 0.0),  # spin in place
    ]

    rows_u: list[np.ndarray] = []
    rows_gyro: list[float] = []
    for n_steps, w_l, w_r, w_rear in segments:
        for _ in range(n_steps):
            u_row = np.array([w_l, w_r, w_rear], dtype=np.float64)
            rows_u.append(u_row)
            wz_wheel = r * (w_r - w_l) / (2.0 * b * alpha)
            rows_gyro.append(float(wz_wheel))

    u = np.stack(rows_u, axis=0)
    gyro_z = np.array(rows_gyro, dtype=np.float64)
    t_sec = np.arange(len(u), dtype=np.float64) * dt
    return t_sec, u, gyro_z


def build_predict_sim(device: str, friction: float, k_turn: float) -> ForwardSimulator:
    """Flat-ground ForwardSimulator matching elevation_node_ekf._build_ekf."""
    cell = 0.1
    n = int(round(8.0 / cell))
    grid = GridParams(n, n, cell, -0.5 * n * cell, -0.5 * n * cell)
    elev0 = wp.zeros((n, n), dtype=wp.float32, device=device)
    robot = dynamics.robot_params()
    solver = dynamics.planning_solver(k_turn=k_turn)
    sim = ForwardSimulator(robot, solver, grid, 1, 1, device)
    sim.set_terrain(elev0)
    sim.set_uniform_friction(friction)
    return sim


def _dt_ratios(t_sec: np.ndarray) -> np.ndarray:
    """Per-step dt_ratio = clip(Δt / DT, 0.5, 3.0); first step uses 1.0."""
    dt_ratio = np.ones(len(t_sec), dtype=np.float64)
    if len(t_sec) < 2:
        return dt_ratio
    dt = np.diff(t_sec)
    dt_ratio[1:] = np.clip(dt / dynamics.DT, 0.5, 3.0)
    return dt_ratio


def simulate(
    t_sec: np.ndarray,
    u: np.ndarray,
    gyro_z: np.ndarray | None,
    sim: ForwardSimulator,
    init_pose: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run open-loop EKF predict for T steps.

    Returns (t_hist [T+1], states [T+1, 6], sigmas [T+1, 6] = sqrt(diag(P))).
    """
    t_steps = len(u)
    if t_steps == 0:
        raise ValueError("empty command sequence")

    x0, y0, psi0 = init_pose
    ekf = EKF6D(np.array([x0, y0, psi0, 0.0, 0.0, 0.0], dtype=np.float64), P0, Q, R_ICP, R_ICP)

    dt_ratio = _dt_ratios(t_sec)
    states = np.empty((t_steps + 1, 6), dtype=np.float64)
    sigmas = np.empty((t_steps + 1, 6), dtype=np.float64)
    states[0] = ekf.x
    sigmas[0] = np.sqrt(np.diag(ekf.P))

    for k in range(t_steps):
        ratio = float(dt_ratio[k])
        omega_z: float | None = None
        if gyro_z is not None and not np.isnan(gyro_z[k]):
            omega_z = float(gyro_z[k])

        off_x, off_y = float(ekf.x[0]), float(ekf.x[1])
        q_local = ekf.x.copy()
        q_local[0] = 0.0
        q_local[1] = 0.0

        x_pred = predict_q6d(q_local, u[k], sim, omega_z=omega_z)
        F = jacobian_F_6d_analytical(q_local, x_pred, dynamics.DT)

        x_pred[0] = ratio * x_pred[0] + off_x
        x_pred[1] = ratio * x_pred[1] + off_y
        dpsi = (x_pred[2] - q_local[2] + np.pi) % (2.0 * np.pi) - np.pi
        x_pred[2] = q_local[2] + ratio * dpsi
        F[0, 2] *= ratio
        F[1, 2] *= ratio

        ekf.predict(F, x_pred, q_scale=ratio)
        states[k + 1] = ekf.x
        sigmas[k + 1] = np.sqrt(np.diag(ekf.P))

    t_hist = np.empty(t_steps + 1, dtype=np.float64)
    dt0 = float(t_sec[1] - t_sec[0]) if len(t_sec) > 1 else dynamics.DT
    t_hist[0] = float(t_sec[0]) - dt0
    t_hist[1:] = t_sec[:t_steps]

    return t_hist, states, sigmas


def plot_dashboard(
    t_hist: np.ndarray,
    states: np.ndarray,
    sigmas: np.ndarray,
) -> Figure:
    """Four-panel dashboard: x, y, psi vs time and xy trajectory."""
    x = states[:, 0]
    y = states[:, 1]
    psi_deg = np.rad2deg(np.unwrap(states[:, 2]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_x, ax_y = axes[0, 0], axes[0, 1]
    ax_psi, ax_xy = axes[1, 0], axes[1, 1]

    color = "#1f77b4"

    state_label = "Predicted state (open-loop EKF f, no ICP/odom)"
    sigma_label = (
        "±1σ band: √diag(P) after each predict "
        "(grows without measurement updates)"
    )

    def _time_panel(ax: plt.Axes, values: np.ndarray, sigma: np.ndarray, ylabel: str) -> None:
        ax.plot(t_hist, values, color=color, linewidth=1.2, label=state_label)
        if SHOW_SIGMA_BAND:
            ax.fill_between(
                t_hist,
                values - sigma,
                values + sigma,
                color=color,
                alpha=0.2,
                label=sigma_label,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(loc="best", fontsize=8)

    _time_panel(ax_x, x, sigmas[:, 0], "x [m]")
    _time_panel(ax_y, y, sigmas[:, 1], "y [m]")
    _time_panel(ax_psi, psi_deg, np.rad2deg(sigmas[:, 2]), "ψ [deg]")
    ax_x.set_xlabel("t [s]")
    ax_y.set_xlabel("t [s]")
    ax_psi.set_xlabel("t [s]")

    # Right y-axis on the ψ panel showing σ(ψ) alone, so the growth is legible
    # even though |ψ| >> σ(ψ) keeps the ±1σ band invisible on the left axis.
    sigma_psi_deg = np.rad2deg(sigmas[:, 2])
    ax_psi_r = ax_psi.twinx()
    ax_psi_r.plot(
        t_hist,
        sigma_psi_deg,
        color="orange",
        linewidth=1.2,
        linestyle="--",
        label="σ(ψ) [deg]",
    )
    ax_psi_r.set_ylabel("σ(ψ) [deg]", color="orange")
    ax_psi_r.tick_params(axis="y", labelcolor="orange")
    ax_psi_r.legend(loc="upper left", fontsize=8)

    ax_xy.plot(x, y, color=color, linewidth=1.2, label="Path in the horizontal plane (x, y)")
    ax_xy.scatter(
        x[0],
        y[0],
        marker="o",
        s=80,
        color=color,
        edgecolors="black",
        zorder=5,
        label="Circle: start pose (initial x, y)",
    )
    ax_xy.scatter(
        x[-1],
        y[-1],
        marker="X",
        s=80,
        color=color,
        edgecolors="black",
        zorder=5,
        label="X: end pose (final x, y after last step)",
    )

    step = max(1, QUIVER_EVERY)
    idx = np.arange(0, len(x), step)
    dx = np.cos(states[idx, 2])
    dy = np.sin(states[idx, 2])
    ax_xy.quiver(
        x[idx],
        y[idx],
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=3.0,
        width=0.004,
        color=color,
        label=f"Arrows: heading ψ (every {step} steps)",
    )

    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_xy.legend(loc="best", fontsize=8)

    fig.suptitle("Helhest EKF predict (open loop, flat ground)")
    fig.tight_layout()
    return fig


def _verify_dt_ratio(sim: ForwardSimulator) -> None:
    """One doubled inter-step interval should double xy displacement on that step."""
    dt = dynamics.DT
    u = np.tile(np.array([2.0, 2.0, 2.0], dtype=np.float64), (2, 1))
    t_doubled = np.array([0.0, 2.0 * dt], dtype=np.float64)
    _, states_a, sig_a = simulate(t_doubled, u, None, sim, INIT_POSE)

    t_uniform = np.array([0.0, dt], dtype=np.float64)
    _, states_b, sig_b = simulate(t_uniform, u, None, sim, INIT_POSE)

    d_a = np.linalg.norm(states_a[2, :2] - states_a[1, :2])
    d_b = np.linalg.norm(states_b[2, :2] - states_b[1, :2])
    ratio = d_a / max(d_b, 1e-9)
    p_growth_a = float(np.sum(sig_a[2] ** 2) - np.sum(sig_a[1] ** 2))
    p_growth_b = float(np.sum(sig_b[2] ** 2) - np.sum(sig_b[1] ** 2))
    print(
        f"[verify] dt_ratio: step-2 disp ratio={ratio:.2f} (expect ~2.0), "
        f"P sigma-sum growth ratio={p_growth_a / max(p_growth_b, 1e-12):.2f}"
    )


def _verify_synthetic(states: np.ndarray, u: np.ndarray, gyro_z: np.ndarray) -> None:
    """Print sanity checks for the built-in synthetic sequence."""
    alpha = 1.0 + K_TURN * FRICTION
    r = WHEEL_RADIUS
    b = HALF_TRACK

    # Straight segment: first 30 steps, w_L == w_R == 2
    sl = slice(0, 31)
    psi_span = np.max(states[sl, 2]) - np.min(states[sl, 2])
    disp = np.linalg.norm(states[30, :2] - states[0, :2])
    v_cmd = r * 2.0
    print(f"[verify] straight: |Δψ|={np.rad2deg(psi_span):.3f}° disp={disp:.3f} m (expect ~{v_cmd * 30 * dynamics.DT:.2f} m)")

    # Left arc segment: steps 30–70
    arc = slice(30, 71)
    wz = float(np.mean(gyro_z[30:70]))
    v = r * (2.0 + 3.5) / 2.0
    r_fit = v / abs(wz) if abs(wz) > 1e-6 else float("inf")
    chord = np.linalg.norm(states[70, :2] - states[30, :2])
    print(f"[verify] left arc: mean gyro_z={wz:.3f} rad/s, v≈{v:.3f} m/s, R=v/|ω|≈{r_fit:.2f} m, chord={chord:.2f} m")

    # Spin: last 25 steps
    spin = slice(-26, None)
    xy_span = np.max(np.linalg.norm(states[spin, :2] - states[spin.start, :2], axis=1))
    dpsi = states[-1, 2] - states[spin.start, 2]
    print(f"[verify] spin: max radial drift={xy_span:.3f} m, Δψ={np.rad2deg(dpsi):.1f}°")

    # Wheel-only yaw on a differential step (no gyro would use alpha)
    w_l, w_r = 2.0, 3.5
    wz_wheel = r * (w_r - w_l) / (2.0 * b * alpha)
    wz_ideal = r * (w_r - w_l) / (2.0 * b)
    print(f"[verify] alpha={alpha:.2f}: wheel yaw rate {wz_wheel:.3f} vs ideal {wz_ideal:.3f} rad/s")


def main() -> None:
    if INPUT_CSV is not None:
        t_sec, u, gyro_z = load_commands(INPUT_CSV)
    else:
        t_sec, u, gyro_z = synthetic_commands()

    sim = build_predict_sim(DEVICE, FRICTION, K_TURN)
    t_hist, states, sigmas = simulate(t_sec, u, gyro_z, sim, INIT_POSE)

    if INPUT_CSV is None:
        assert gyro_z is not None
        _verify_synthetic(states, u, gyro_z)
        _verify_dt_ratio(sim)

    fig = plot_dashboard(t_hist, states, sigmas)
    if OUT_PNG is not None:
        fig.savefig(OUT_PNG, dpi=150)
        print(f"saved {OUT_PNG}")
    elif plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
