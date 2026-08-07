"""Fit the wheel actuator response (tau_motor) from bags, and show its overshoot.

Run:  python scripts/fit_actuator_lag.py bags/out_experiment_goal_unreachable0

The engine models the actuator as a first-order lag (`step.motor_lag_step`) with `tau_motor`
defaulting to 0, i.e. the simulated wheel lands exactly on the command every step. Three
candidate models are fitted here to find out what the real one does:

  delay     pure transport lag, by cross-correlation.
  1st order tau, by simulating omega_m[k+1] = omega_m[k] + (dt/tau)(omega_c[k] - omega_m[k])
            and minimising RMSE. This is what the engine can represent today.
  2nd order (omega_n, zeta), which CAN overshoot: peak overshoot = exp(-pi zeta / sqrt(1-zeta^2)).

MEASURED (out_experiment_goal_unreachable0/1, the only post-fix bags with /joint_states): the
response is essentially PURE DELAY of 189-249 ms plus a fast lag of 20-50 ms, with NO overshoot --
the second-order fit lands at zeta 0.85-1.00 and does not beat first order on RMSE.

That matters for what to change in the engine, because the two are not interchangeable: at
dt = 0.1 s a 0.03 s lag is a no-op (the blend saturates at 1.0), so `tau_motor` is the wrong knob.
The effect that dominates is a ~2-control-tick transport delay, which neither a first-order lag
nor any instantaneous model can express -- it needs a command delay line.

WARNING ABOUT INSTANTANEOUS RATIOS. Comparing measured/commanded sample by sample on a
continuously varying command reports ratios of 1.3-1.7 and peaks above 3, which look like
overshoot and are not: they are lag. Delay-aligning collapses the peaks (3.4 -> 1.8) and the
model fit finds no oscillation. `_trace` plots the raw and aligned traces so this is visible
rather than inferred.

Needs /joint_states, so it only runs on bags that recorded measured wheel speeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

JOINTS = ("left_wheel_j", "right_wheel_j", "rear_wheel_j")
STEP_THRESHOLD = 1.0  # [rad/s] command jump that counts as a step
SETTLE_WINDOW = 1.0  # [s] of response to inspect after a step


def _load(bag: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(time, commanded, measured) wheel speeds on the /joint_states clock, plus its period."""
    cmd_t, cmd, st_t, meas = [], [], [], []
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic in ("/cmd_joints", "/joint_states")]
        for conn, stamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if len(msg.velocity) < 3:
                continue
            idx = [list(msg.name).index(j) for j in JOINTS]
            row = [msg.velocity[i] for i in idx]
            if conn.topic == "/cmd_joints":
                cmd_t.append(stamp * 1e-9)
                cmd.append(row)
            else:
                st_t.append(stamp * 1e-9)
                meas.append(row)
    st_t, meas = np.asarray(st_t), np.asarray(meas)
    cmd_t, cmd = np.asarray(cmd_t), np.asarray(cmd)
    if st_t.size == 0 or cmd_t.size == 0:
        return st_t, np.empty((0, 3)), np.empty((0, 3)), 0.0
    on_grid = np.stack([np.interp(st_t, cmd_t, cmd[:, i]) for i in range(3)], axis=1)
    return st_t, on_grid, meas, float(np.median(np.diff(st_t)))


def _delay(command: np.ndarray, response: np.ndarray, dt: float, max_lag: float = 0.6) -> float:
    best_shift, best = 0, -2.0
    for shift in range(int(max_lag / dt)):
        end = len(command) - shift
        corr = float(np.corrcoef(command[:end], response[shift:])[0, 1])
        if corr > best:
            best_shift, best = shift, corr
    return best_shift * dt


def _simulate_first_order(command: np.ndarray, tau: float, dt: float) -> np.ndarray:
    out = np.empty_like(command)
    state = command[0]
    blend = min(dt / max(tau, 1e-6), 1.0)
    for k, target in enumerate(command):
        state += blend * (target - state)
        out[k] = state
    return out


def _simulate_second_order(command: np.ndarray, omega_n: float, zeta: float, dt: float):
    """Discrete second-order velocity loop -- the simplest model that can overshoot."""
    out = np.empty_like(command)
    position, velocity = command[0], 0.0
    for k, target in enumerate(command):
        accel = omega_n**2 * (target - position) - 2.0 * zeta * omega_n * velocity
        velocity += accel * dt
        position += velocity * dt
        out[k] = position
    return out


def _downsample(values: np.ndarray, columns: int) -> np.ndarray:
    edges = np.linspace(0, len(values), columns + 1).astype(int)
    return np.array([values[i:j].mean() for i, j in zip(edges[:-1], edges[1:])])


def _sparkline(values: np.ndarray, low: float, high: float, rows: int = 10) -> list[str]:
    """Tiny ASCII plot: one column per sample, `rows` tall."""
    grid = [[" "] * len(values) for _ in range(rows)]
    for col, v in enumerate(values):
        frac = (v - low) / (high - low + 1e-9)
        grid[int(round((1.0 - np.clip(frac, 0, 1)) * (rows - 1)))][col] = "*"
    return ["".join(r) for r in grid]


def _overlay(command: np.ndarray, response: np.ndarray, title: str, label: str) -> None:
    low = min(command.min(), response.min())
    high = max(command.max(), response.max())
    print(f"  {title}")
    for a, b in zip(_sparkline(command, low, high), _sparkline(response, low, high)):
        merged = "".join(
            "#" if p == "*" and q == "*" else ("-" if p == "*" else ("*" if q == "*" else " "))
            for p, q in zip(a, b)
        )
        print("   |" + merged)
    print(f"   +{'-' * len(command)}   - commanded   * {label}   # both\n")


def analyse(bag: Path) -> None:
    t, command, measured, dt = _load(bag)
    if measured.size == 0:
        print(f"{bag.name}: no /joint_states -- cannot measure the actuator response")
        return
    print(f"=== {bag.name}   {len(t)} samples @ {1 / dt:.0f} Hz")

    for i, joint in enumerate(JOINTS):
        c, m = command[:, i], measured[:, i]
        delay = _delay(c, m, dt)
        shift = int(round(delay / dt))
        c_aligned = c[: len(c) - shift] if shift else c
        m_aligned = m[shift:] if shift else m

        taus = np.arange(0.02, 1.0, 0.01)
        errs = [
            np.sqrt(np.mean((_simulate_first_order(c_aligned, tau, dt) - m_aligned) ** 2))
            for tau in taus
        ]
        tau_best, tau_rmse = float(taus[int(np.argmin(errs))]), float(np.min(errs))

        best = (np.inf, 0.0, 0.0)
        for omega_n in np.arange(2.0, 30.0, 1.0):
            for zeta in np.arange(0.1, 1.3, 0.05):
                rmse = np.sqrt(
                    np.mean((_simulate_second_order(c_aligned, omega_n, zeta, dt) - m_aligned) ** 2)
                )
                if rmse < best[0]:
                    best = (float(rmse), float(omega_n), float(zeta))
        rmse2, omega_n, zeta = best
        overshoot = 100.0 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)) if zeta < 1.0 else 0.0
        print(
            f"  {joint:14s} delay {delay * 1000:5.0f} ms | 1st order tau {tau_best:.2f} s "
            f"RMSE {tau_rmse:.3f} | 2nd order wn {omega_n:4.1f} zeta {zeta:.2f} "
            f"RMSE {rmse2:.3f} -> overshoot {overshoot:.0f}%"
        )

    _trace(command, measured, dt)

    # measured overshoot on real command steps
    print("\n  measured step responses (command jump > 1 rad/s, response over the next 1 s):")
    window = int(SETTLE_WINDOW / dt)
    overshoots = []
    examples = []
    for i, joint in enumerate(JOINTS):
        c, m = command[:, i], measured[:, i]
        jumps = np.where(np.abs(np.diff(c)) > STEP_THRESHOLD)[0]
        for k in jumps:
            if k + window >= len(c) or k < 5:
                continue
            target = c[k + 1]
            start = m[k]
            if abs(target - start) < STEP_THRESHOLD:
                continue
            seg = m[k : k + window]
            peak = seg.max() if target > start else seg.min()
            over = 100.0 * (peak - target) / (target - start)
            overshoots.append(over)
            examples.append((over, joint, k, start, target, seg, c[k : k + window]))
    if not overshoots:
        print("    no clean steps found")
        return
    o = np.array(overshoots)
    print(
        f"    {len(o)} steps: median overshoot {np.median(o):.0f}%, "
        f"p90 {np.percentile(o, 90):.0f}%, max {o.max():.0f}%, "
        f"fraction overshooting {100 * (o > 5).mean():.0f}%"
    )
    examples.sort(key=lambda e: -e[0])
    for over, joint, k, start, target, seg, cseg in examples[:2]:
        low = min(seg.min(), cseg.min())
        high = max(seg.max(), cseg.max())
        print(f"\n    {joint} step {start:.1f} -> {target:.1f} rad/s, overshoot {over:.0f}%")
        for row_m, row_c in zip(_sparkline(seg, low, high), _sparkline(cseg, low, high)):
            merged = "".join(
                "#" if a == "*" and b == "*" else ("*" if a == "*" else ("-" if b == "*" else " "))
                for a, b in zip(row_m, row_c)
            )
            print(f"      |{merged}")
        print(f"      +{'-' * len(seg)}  ({SETTLE_WINDOW:.0f} s)   * measured  - commanded")


def _trace(command: np.ndarray, measured: np.ndarray, dt: float, seconds: float = 3.0) -> None:
    """Plot the busiest window raw and delay-aligned -- the picture that distinguishes lag from
    overshoot. An instantaneous measured/commanded ratio cannot: on a continuously varying command
    a pure lag produces ratios far above 1 with no overshoot anywhere."""
    c, m = command[:, 0], measured[:, 0]
    delay = _delay(c, m, dt)
    shift = int(round(delay / dt))
    width = int(seconds / dt)
    if len(c) < width + shift + 10:
        return
    spread = [np.ptp(c[k : k + width]) for k in range(0, len(c) - width - shift, 10)]
    k = int(np.argmax(spread)) * 10
    cols = 96
    print(f"\n  {JOINTS[0]}, busiest {seconds:.0f} s, measured delay {delay * 1000:.0f} ms")
    _overlay(
        _downsample(c[k : k + width], cols), _downsample(m[k : k + width], cols), "RAW", "measured"
    )
    _overlay(
        _downsample(c[k : k + width], cols),
        _downsample(m[k + shift : k + shift + width], cols),
        f"SHIFTED BACK {delay * 1000:.0f} ms",
        "measured (shifted)",
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    for name in sys.argv[1:]:
        analyse(Path(name))
