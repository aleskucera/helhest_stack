"""Calibrate VILENS random-walk drift rates on the DESIGN site (keble) only.

For horizons h, compare VILENS relative motion against GT relative motion,
expressed in the starting body frame (so the rates are frame-independent).
A random walk predicts var(err(h)) = q * h; q is fit by least squares through
the origin over several horizons. The frozen rates feed the eq.-20 motion update
on the virgin sites, blind.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .loaders import GtTrajectory

HORIZONS_S = (2.0, 5.0, 10.0, 20.0)


def _yaw(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def increment_errors(seq_dir: Path, horizon: float) -> np.ndarray:
    """Rows of (ex, ey, ez, eyaw): VILENS-vs-GT relative-motion error over `horizon`."""
    gt = GtTrajectory(str(seq_dir / "trajectory" / "gt-tum.txt"))
    vs = GtTrajectory(str(seq_dir / "trajectory" / "vilens-slam-tum.txt"))
    t0 = max(gt.t[0], vs.t[0])
    t1 = min(gt.t[-1], vs.t[-1]) - horizon
    rows = []
    for ts in np.arange(t0, t1, horizon / 2.0):  # 50% overlapped starts
        Ga, Gb = gt.pose_at(ts), gt.pose_at(ts + horizon)
        Va, Vb = vs.pose_at(ts), vs.pose_at(ts + horizon)
        G_rel = np.linalg.inv(Ga) @ Gb
        V_rel = np.linalg.inv(Va) @ Vb
        E = np.linalg.inv(G_rel) @ V_rel
        rows.append([E[0, 3], E[1, 3], E[2, 3], _yaw(E)])
    return np.array(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dirs", nargs="+")
    args = ap.parse_args()
    print(f"{'horizon':>8s} {'sd_x':>7s} {'sd_y':>7s} {'sd_z':>7s} {'sd_yaw':>8s}  [cm, deg]")
    var_by_h = {h: [] for h in HORIZONS_S}
    for h in HORIZONS_S:
        errs = np.concatenate([increment_errors(Path(d), h) for d in args.seq_dirs])
        v = errs.var(axis=0)
        var_by_h[h] = v
        print(
            f"{h:8.0f} {np.sqrt(v[0])*100:7.1f} {np.sqrt(v[1])*100:7.1f} "
            f"{np.sqrt(v[2])*100:7.1f} {np.degrees(np.sqrt(v[3])):8.2f}"
        )
    # least-squares slope through origin: q = sum(h*var) / sum(h^2)
    hs = np.array(HORIZONS_S)
    V = np.stack([var_by_h[h] for h in HORIZONS_S])
    q = (hs[:, None] * V).sum(0) / (hs**2).sum()
    print("\nrandom-walk rates q = var/s (frozen into the prereg):")
    print(f"  q_x   = {q[0]:.3e} m^2/s   (sd at 20 s: {np.sqrt(q[0]*20)*100:.1f} cm)")
    print(f"  q_y   = {q[1]:.3e} m^2/s   (sd at 20 s: {np.sqrt(q[1]*20)*100:.1f} cm)")
    print(f"  q_z   = {q[2]:.3e} m^2/s   (sd at 20 s: {np.sqrt(q[2]*20)*100:.1f} cm)")
    print(f"  q_yaw = {q[3]:.3e} rad^2/s (sd at 20 s: {np.degrees(np.sqrt(q[3]*20)):.2f} deg)")


if __name__ == "__main__":
    main()
