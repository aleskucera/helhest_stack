"""Trajectory quality: how far a drive_sim run is from the reference (reference.py), and the
shape measures that reach time and contact never see.

    python studies/closed_loop/quality.py --world box RUN.npz [RUN.npz ...]

Per run: reached, time, the distance to the reference (discrete Frechet, and the mean and max
lateral offset of the driven path from it), the driven path's curvature (RMS and peak) and
curvature rate (RMS), the body's clearance along the way (mean, minimum, and the fraction of
frames closer than the reference's clearance), the symmetry of the path about the midpoint plane
between start and goal (mean mirror error -- meaningful where the world is symmetric), the number
of yaw reversals and of stops, and the command slew RMS from the recorded commands.

The curvature of the driven path is taken from the trail, smoothed over `--window` seconds
first: at 14.5 Hz the raw trail's second difference is noise.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import reference as R  # noqa: E402

DT = 1.0 / 14.5
WHEEL_R = 0.35


@dataclasses.dataclass
class Quality:
    run: str
    reached: bool
    time_s: float
    frechet_m: float
    offset_mean_m: float
    offset_max_m: float
    kappa_rms: float
    kappa_max: float
    kappa_rate_rms: float
    clearance_mean_m: float
    clearance_min_m: float
    below_reference_frac: float
    mirror_error_m: float
    reversals: int
    stops: int
    cmd_slew_rms: float


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Discrete Frechet distance between two polylines [N, 2] and [M, 2]."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    n, m = d.shape
    ca = np.full((n, m), np.inf)
    ca[0, 0] = d[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, n):
        prev = ca[i - 1]
        row = ca[i]
        for j in range(1, m):
            row[j] = max(min(prev[j], prev[j - 1], row[j - 1]), d[i, j])
    return float(ca[-1, -1])


def offsets(trail: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """[N] distance from each trail point to the reference polyline."""
    a, b = ref[:-1], ref[1:]
    ab = b - a
    ab2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    out = np.empty(len(trail))
    for i, p in enumerate(trail):
        t = np.clip(np.sum((p - a) * ab, axis=1) / ab2, 0.0, 1.0)
        q = a + t[:, None] * ab
        out[i] = np.min(np.linalg.norm(q - p, axis=1))
    return out


def smooth_trail(trail: np.ndarray, window: int) -> np.ndarray:
    k = np.ones(window) / window
    pad = window // 2
    x = np.pad(trail[:, 0], pad, mode="edge")
    y = np.pad(trail[:, 1], pad, mode="edge")
    return np.column_stack(
        [np.convolve(x, k, "valid")[: len(trail)], np.convolve(y, k, "valid")[: len(trail)]]
    )


def mirror_error(trail: np.ndarray, start: np.ndarray, goal: np.ndarray) -> float:
    """Mean distance between the path and its mirror image about the plane through the midpoint
    of start and goal, perpendicular to the start-goal line. 0 for a perfectly symmetric path."""
    axis = goal - start
    L = np.linalg.norm(axis)
    if L < 1e-6:
        return float("nan")
    u = axis / L
    v = np.array([-u[1], u[0]])
    along = (trail - start) @ u
    across = (trail - start) @ v
    keep = (along > 0.0) & (along < L)
    along, across = along[keep], across[keep]
    order = np.argsort(along)
    along, across = along[order], across[order]
    half = np.linspace(0.05 * L, 0.45 * L, 60)
    near = np.interp(half, along, across)
    far = np.interp(L - half, along, across)
    return float(np.mean(np.abs(near - far)))


def assess(fn: str, ref: R.Reference, window_s: float = 0.5) -> Quality:
    z = np.load(fn)
    trail = z["trail"]
    goal = z["goal"]
    reached = bool(z["reached"])
    n = (
        int(np.argmax(np.hypot(trail[:, 0] - goal[0], trail[:, 1] - goal[1]) < 0.4))
        if reached
        else len(trail)
    )
    t = trail[:n]
    sm = smooth_trail(t, max(3, int(round(window_s / DT)) | 1))
    # resample the smoothed trail by arc length so curvature is geometric, not speed-dependent
    seg = np.hypot(*np.diff(sm, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    moving = s[-1] > 0.5
    rs = R.resample(sm, ref.params.spacing_m) if moving else sm
    k = R.curvature(rs)
    dk = np.diff(k) / ref.params.spacing_m
    cl = z["clearance"][:n]
    cl = cl[np.isfinite(cl)]
    off = offsets(t[:: max(1, n // 400)], ref.points)
    yaw = z["yaw_log"][: n - 1, 2] if "yaw_log" in z else np.zeros(1)
    sign = np.sign(np.where(np.abs(yaw) > 0.2, yaw, 0.0))
    sign = sign[sign != 0]
    reversals = int(np.sum(np.diff(sign) != 0)) if len(sign) else 0
    speed = np.hypot(*np.diff(t, axis=0).T) / DT
    stopped = speed < 0.1
    stops = 0
    i = int(np.argmax(speed > 0.3)) if (speed > 0.3).any() else len(speed)
    while i < len(stopped):
        if stopped[i]:
            j = i
            while j < len(stopped) and stopped[j]:
                j += 1
            if (j - i) * DT >= 1.0:
                stops += 1
            i = j
        else:
            i += 1
    h = z["hist_meta"]
    h = h[h[:, 0] < n]
    if len(h) > 2:
        stride = float(h[1, 0] - h[0, 0]) * DT
        slew = np.diff(h[:, 4:6], axis=0) / stride
        cmd_slew = float(np.sqrt(np.mean(slew**2)))
    else:
        cmd_slew = float("nan")
    start = np.array(R.W.WORLDS[ref.world][1][:2], float)
    return Quality(
        run=fn.rsplit("/", 1)[-1],
        reached=reached,
        time_s=n * DT,
        frechet_m=frechet(t[:: max(1, n // 300)], ref.points),
        offset_mean_m=float(off.mean()),
        offset_max_m=float(off.max()),
        kappa_rms=float(np.sqrt(np.mean(k**2))),
        kappa_max=float(np.abs(k).max()),
        kappa_rate_rms=float(np.sqrt(np.mean(dk**2))) if len(dk) else 0.0,
        clearance_mean_m=float(cl.mean()) if len(cl) else float("nan"),
        clearance_min_m=float(cl.min()) if len(cl) else float("nan"),
        below_reference_frac=(
            float(np.mean(cl < ref.params.clearance_m)) if len(cl) else float("nan")
        ),
        mirror_error_m=mirror_error(t, start, np.asarray(goal, float)),
        reversals=reversals,
        stops=stops,
        cmd_slew_rms=cmd_slew,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--world", required=True)
    p.add_argument("--clearance", type=float, default=0.8)
    p.add_argument("--radius", type=float, default=1.5)
    p.add_argument("--clothoid", type=float, default=1.5)
    p.add_argument("--window", type=float, default=0.5, help="[s] trail smoothing for curvature")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    prm = R.ReferenceParams(clearance_m=a.clearance, radius_min_m=a.radius, clothoid_m=a.clothoid)
    ref = R.reference_for_world(a.world, prm)
    print(
        f"reference {a.world}: {ref.s[-1]:.2f} m, kappa rms {np.sqrt(np.mean(ref.kappa**2)):.2f} "
        f"max {np.abs(ref.kappa).max():.2f}, clearance min {ref.clearance.min():.2f}"
    )
    cols = [f.name for f in dataclasses.fields(Quality)]
    print(" ".join(f"{c:>14s}" for c in cols))
    out = []
    for fn in a.runs:
        q = assess(fn, ref, a.window)
        out.append(dataclasses.asdict(q))
        print(
            " ".join(
                (
                    f"{str(getattr(q, c))[:14]:>14s}"
                    if isinstance(getattr(q, c), (str, bool, int))
                    else f"{getattr(q, c):14.3f}"
                )
                for c in cols
            )
        )
    if a.json:
        json.dump(
            {
                "reference": {
                    "points": ref.points.tolist(),
                    "kappa": ref.kappa.tolist(),
                    "clearance": ref.clearance.tolist(),
                },
                "runs": out,
            },
            open(a.json, "w"),
        )


if __name__ == "__main__":
    main()
