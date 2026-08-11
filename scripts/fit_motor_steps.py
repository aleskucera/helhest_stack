"""Identify the drive motors from `calibrate_drive.py steps`, in the air and on the ground.

    python scripts/fit_motor_steps.py ~/bags/steps_air
    python scripts/fit_motor_steps.py ~/bags/steps_air ~/bags/steps_ground --plot /tmp/motors.png

WHY TWO BAGS. Every actuator number this project has is confounded. `MOTOR_TAU = 0.19 s` was fitted
from planner-driven bags, where the command is never held. `compact` has held steps but they are
all breakaway-from-rest in a spin: measured 2026-08-10, the wheel creeps under 30% of command for
~600 ms, then breaks loose to 175%, then rings -- not a first-order lag at any parameter. Wheels
off the ground removes breakaway, load and slip, so what is left IS the motor plus wheel inertia.
The ground run then adds them back one experiment at a time, and the DIFFERENCE is the answer.

Per step this reports:
  dead     command edge -> the wheel passing 10% of the step [ms]. The planner models this as
           COMMAND_DELAY, currently 0.04 s.
  tau      first-order fit over the rise. What the engine can represent (`step.motor_lag_step`).
  rise     10% -> 90% [ms], model-free, so it stands even where the first-order fit does not.
  peak     highest speed reached / commanded. >1 means overshoot -- which a first-order lag CANNOT
           produce, so a peak well above 1 says the model is the wrong shape, not mistuned.
  gain     settled speed / commanded. In the air this should be ~1; on the ground, what is missing
           is load and slip.
  effort   |effort| while settling. /joint_states carries it (+-1353 on this robot, units unknown
           but monotonic in torque), so air-vs-ground at MATCHED speed prices the terrain directly.

A step is only used if the command was held long enough to settle and the previous level was held
long enough to have settled first -- otherwise the "step" starts from a moving target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

DRIVE = ("left_wheel_j", "right_wheel_j")
SETTLE_FRAC = 0.75  # sample the settled value over the last quarter of the hold


def read(bag: Path) -> dict:
    """(t, cmd) per drive joint plus measured velocity and effort, on the bag clock."""
    tc, cmd, tj, vel, eff = [], [], [], [], []
    names: list[str] | None = None
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic in ("/cmd_joints", "/joint_states")]
        if not conns:
            raise SystemExit(f"{bag}: needs /cmd_joints and /joint_states")
        for conn, stamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == "/cmd_joints":
                if len(msg.velocity) >= 3:
                    tc.append(stamp * 1e-9)
                    cmd.append(list(msg.velocity))
            else:
                if names is None:
                    names = list(msg.name)
                if len(msg.velocity) >= 3:
                    tj.append(stamp * 1e-9)
                    vel.append(list(msg.velocity))
                    eff.append(list(msg.effort) if len(msg.effort) >= 3 else [0.0] * 3)
    return {
        "tc": np.asarray(tc),
        "cmd": np.asarray(cmd),
        "tj": np.asarray(tj),
        "vel": np.asarray(vel),
        "eff": np.asarray(eff),
        "names": names or list(DRIVE),
    }


def edges(tc: np.ndarray, c: np.ndarray, min_jump: float, min_hold: float) -> list[tuple]:
    """(t_edge, from, to, t_end) for command changes that were HELD either side."""
    out = []
    changes = [0] + [k for k in range(1, len(c)) if abs(c[k] - c[k - 1]) > 1e-6] + [len(c)]
    for i in range(1, len(changes) - 1):
        k = changes[i]
        t_edge = tc[k]
        held_before = t_edge - tc[changes[i - 1]]
        t_end = tc[changes[i + 1] - 1]
        if abs(c[k] - c[k - 1]) < min_jump or held_before < min_hold or t_end - t_edge < min_hold:
            continue
        out.append((t_edge, float(c[k - 1]), float(c[k]), t_end))
    return out


def fit_step(t: np.ndarray, v: np.ndarray, e: np.ndarray, edge: tuple) -> dict | None:
    t_edge, lo, hi, t_end = edge
    win = (t >= t_edge - 0.30) & (t <= t_end)
    if win.sum() < 25:
        return None
    tt, vv, ee = t[win] - t_edge, v[win], np.abs(e[win])
    base = float(np.median(vv[tt <= 0.0]))
    settled = float(np.median(vv[tt >= SETTLE_FRAC * (t_end - t_edge)]))
    span = settled - base
    # Normalise by the COMMANDED CHANGE, never by the absolute command: half the steps in this
    # program go back to zero, and dividing by that makes peak/gain explode.
    span_cmd = hi - lo
    if abs(span) < 0.15:  # the wheel never actually moved -- below breakaway, say so
        return {
            "cmd_from": lo, "cmd_to": hi, "moved": False,
            "dead": np.nan, "tau": np.nan, "rise": np.nan,
            "peak": 0.0, "gain": span / span_cmd,
            "effort": float(np.median(ee[tt >= SETTLE_FRAC * (t_end - t_edge)])),
        }

    def crossing(frac: float) -> float:
        hit = np.where((vv - base) * np.sign(span) >= abs(frac * span))[0]
        return float(tt[hit[0]]) if len(hit) else np.nan

    t10, t90 = crossing(0.10), crossing(0.90)
    # first-order tau over the rise, by direct search -- cheaper and more robust than a log fit,
    # which blows up wherever the response briefly reverses (it does, on the ground).
    rise = (tt >= 0.0) & (tt <= min(1.5, t_end - t_edge))
    best_tau, best_err = np.nan, np.inf
    for tau in np.arange(0.02, 0.81, 0.01):
        model = base + span * (1.0 - np.exp(-np.maximum(tt[rise] - (t10 if t10 == t10 else 0.0), 0.0) / tau))
        err = float(np.mean((model - vv[rise]) ** 2))
        if err < best_err:
            best_tau, best_err = float(tau), err
    return {
        "cmd_from": lo, "cmd_to": hi, "moved": True,
        "dead": t10 * 1e3, "tau": best_tau, "rise": (t90 - t10) * 1e3,
        "peak": float(np.max((vv[tt >= 0.0] - base) * np.sign(span_cmd))) / abs(span_cmd),
        "gain": span / span_cmd,
        "effort": float(np.median(ee[tt >= SETTLE_FRAC * (t_end - t_edge)])),
    }


def analyse(bag: Path) -> list[dict]:
    d = read(bag)
    rows = []
    for j in DRIVE:
        if j not in d["names"]:
            continue
        col = d["names"].index(j)
        for e in edges(d["tc"], d["cmd"][:, col], min_jump=0.25, min_hold=1.2):
            r = fit_step(d["tj"], d["vel"][:, col], d["eff"][:, col], e)
            if r:
                r["joint"] = j
                rows.append(r)
    return rows


def report(tag: str, rows: list[dict]) -> None:
    moved = [r for r in rows if r["moved"]]
    still = [r for r in rows if not r["moved"]]
    print(f"\n=== {tag} ===  {len(rows)} usable steps, {len(still)} of which never moved the wheel")
    if still:
        amps = sorted({abs(r["cmd_to"] - r["cmd_from"]) for r in still})
        print(f"  did not break loose at steps of: {', '.join(f'{a:.1f}' for a in amps)} rad/s")
    if not moved:
        return
    print(f"{'|step|':>8}{'n':>4}{'dead ms':>10}{'tau s':>9}{'rise ms':>10}"
          f"{'peak':>9}{'gain':>8}{'|effort|':>10}")
    buckets = [(0.0, 0.8), (0.8, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 99.0)]
    for lo, hi in buckets:
        sel = [r for r in moved if lo <= abs(r["cmd_to"] - r["cmd_from"]) < hi]
        if not sel:
            continue
        g = lambda k: np.nanmedian([r[k] for r in sel])  # noqa: E731
        print(f"{np.median([abs(r['cmd_to']-r['cmd_from']) for r in sel]):>8.1f}{len(sel):>4}"
              f"{g('dead'):>10.0f}{g('tau'):>9.2f}{g('rise'):>10.0f}"
              f"{g('peak'):>9.2f}{g('gain'):>8.2f}{g('effort'):>10.0f}")
    # Per joint as well as pooled: this drivetrain has a documented left/right asymmetry (the left
    # drive joint is mirror-mounted), and a median over both wheels would hide it.
    for j in DRIVE:
        sel = [r for r in moved if r["joint"] == j]
        if not sel:
            continue
        g = lambda k: np.nanmedian([r[k] for r in sel])  # noqa: E731
        print(f"{j.replace('_wheel_j',''):>8}{len(sel):>4}{g('dead'):>10.0f}{g('tau'):>9.2f}"
              f"{g('rise'):>10.0f}{g('peak'):>9.2f}{g('gain'):>8.2f}{g('effort'):>10.0f}")
    print(f"{'ALL':>8}{len(moved):>4}{np.nanmedian([r['dead'] for r in moved]):>10.0f}"
          f"{np.nanmedian([r['tau'] for r in moved]):>9.2f}"
          f"{np.nanmedian([r['rise'] for r in moved]):>10.0f}"
          f"{np.nanmedian([r['peak'] for r in moved]):>9.2f}"
          f"{np.nanmedian([r['gain'] for r in moved]):>8.2f}"
          f"{np.nanmedian([r['effort'] for r in moved]):>10.0f}")


def compare(a: list[dict], b: list[dict]) -> None:
    ma = [r for r in a if r["moved"]]
    mb = [r for r in b if r["moved"]]
    if not (ma and mb):
        return
    print("\n=== air -> ground: what the terrain adds ===")
    for key, label, unit in (("dead", "dead time", "ms"), ("tau", "tau", "s"),
                             ("rise", "rise 10-90", "ms"), ("peak", "peak/cmd", ""),
                             ("gain", "settled/cmd", ""), ("effort", "|effort|", "")):
        va, vb = np.nanmedian([r[key] for r in ma]), np.nanmedian([r[key] for r in mb])
        print(f"  {label:>12}: {va:8.2f} -> {vb:8.2f} {unit}")
    print("\n  Read it this way: the AIR column is the actuator, so that is what COMMAND_DELAY and\n"
          "  MOTOR_TAU should be set from. Whatever the ground column adds on top is terrain --\n"
          "  load, breakaway and slip -- and belongs in the traction model, not the motor one.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("air", help="bag with the wheels OFF the ground")
    ap.add_argument("ground", nargs="?", help="bag with the wheels ON the ground")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    rows_air = analyse(Path(args.air))
    report(Path(args.air).name + "  (AIR)" if args.ground else Path(args.air).name, rows_air)
    rows_gnd = []
    if args.ground:
        rows_gnd = analyse(Path(args.ground))
        report(Path(args.ground).name + "  (GROUND)", rows_gnd)
        compare(rows_air, rows_gnd)
    if args.plot:
        _plot(Path(args.air), Path(args.ground) if args.ground else None, args.plot)


def _plot(air: Path, ground: Path | None, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bags = [(air, "air")] + ([(ground, "ground")] if ground else [])
    fig, axes = plt.subplots(len(bags), 2, figsize=(14, 4.5 * len(bags)), squeeze=False)
    for row, (bag, tag) in enumerate(bags):
        d = read(bag)
        col = d["names"].index(DRIVE[0]) if DRIVE[0] in d["names"] else 0
        ev = edges(d["tc"], d["cmd"][:, col], min_jump=0.25, min_hold=1.2)
        ax = axes[row][0]
        for t_edge, lo, hi, t_end in ev:
            if abs(hi - lo) < 1.0:
                continue
            w = (d["tj"] >= t_edge - 0.2) & (d["tj"] <= t_edge + 1.5)
            if w.sum() < 20:
                continue
            base = float(np.median(d["vel"][:, col][d["tj"] <= t_edge][-10:]))
            ax.plot((d["tj"][w] - t_edge) * 1e3,
                    (d["vel"][w, col] - base) / (hi - base), lw=0.7, color="#adb5bd")
        ax.axhline(1.0, ls="--", color="k", lw=1)
        ax.axvline(0.0, ls=":", color="k", lw=1)
        ax.set_ylim(-0.4, 2.0)
        ax.set_xlabel("ms after the command edge")
        ax.set_ylabel("fraction of step")
        ax.set_title(f"{tag}: normalised step responses")
        ax2 = axes[row][1]
        ax2.plot(d["tj"] - d["tj"][0], d["vel"][:, col], lw=0.7, color="#0b7285", label="measured")
        ax2.step(d["tc"] - d["tj"][0], d["cmd"][:, col], where="post", lw=1.0,
                 color="#e8590c", label="commanded")
        ax2.set_xlabel("s")
        ax2.set_ylabel("rad/s")
        ax2.set_title(f"{tag}: {bag.name}")
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
