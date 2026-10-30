"""Score every recorded reference curve (extract.py output) on what can be judged without the map:

shape    curvature from the points (max |kappa|), S-bends (sign changes of curvature beyond
         a threshold, over the first 6 m), total turning (a loop turns > 300 deg)
start    the curve's first heading against the robot's yaw (a pivot asked at the start)
flicker  how far this frame's curve lies from last frame's over the next 4 m ahead of the
         robot (mean and max distance from this curve's points to last frame's polyline)
follow   where the robot actually drove over the next 2 s, against this frame's curve
         (mean and max distance of the future poses to the curve)
empty    frames that published no curve (MPPI descended V instead)

  uv run --no-project --with numpy python score.py out/*.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def wrap(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def resample(p: np.ndarray, step: float) -> np.ndarray:
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < step:
        return p
    q = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], 1)


def dist_to_polyline(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to a polyline."""
    if len(poly) < 2:
        return np.full(len(pts), np.nan)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L2 = np.maximum((ab**2).sum(1), 1e-12)
    ap = pts[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(2) / L2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2).min(1)


def shape(p: np.ndarray) -> dict:
    q = resample(p, 0.25)
    if len(q) < 4:
        return dict(len=0.0, kmax=0.0, s_bends=0, turn_deg=0.0, turn_abs_deg=0.0)
    d = np.diff(q, axis=0)
    h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    k = np.diff(h) / 0.25
    if len(k) >= 4:  # over 1 m, so the marker's rounding does not count as bends
        k = np.convolve(k, np.ones(4) / 4, mode="same")
    k6 = k[: int(6.0 / 0.25)]
    sig = np.sign(np.where(np.abs(k6) > 0.15, k6, 0))  # past R ~ 6.7 m counts as a bend
    sig = sig[sig != 0]
    return dict(
        len=float(np.linalg.norm(d, axis=1).sum()),
        kmax=float(np.abs(k).max()),
        s_bends=int(np.count_nonzero(np.diff(sig))),
        turn_deg=float(np.degrees(h[-1] - h[0])),
        turn_abs_deg=float(np.degrees(np.abs(np.diff(h)).sum())),
    )


def score(path: Path) -> dict:
    d = np.load(path)
    off = np.concatenate([[0], np.cumsum(d["ref_n"])])
    refs = [d["ref_xy"][off[i] : off[i + 1]] for i in range(len(d["ref_n"]))]
    t = d["ref_t"]
    od = d["odom"]
    frames = []
    prev = None
    for i, (ti, p) in enumerate(zip(t, refs)):
        j = min(np.searchsorted(od[:, 0], ti), len(od) - 1)
        x, y, yaw, v = od[j, 1], od[j, 2], od[j, 3], od[j, 4]
        f = dict(
            i=i,
            t=float(ti - t[0]),
            x=float(x),
            y=float(y),
            yaw=float(yaw),
            v=float(v),
            empty=len(p) < 3,
        )
        if len(p) >= 3:
            f.update(shape(p))
            d0 = p[min(2, len(p) - 1)] - p[0]
            f["start_err_deg"] = float(np.degrees(wrap(np.arctan2(d0[1], d0[0]) - yaw)))
            q = resample(p, 0.1)
            s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
            ahead = q[s <= 4.0]
            if prev is not None and len(prev) >= 3:
                dd = dist_to_polyline(ahead, prev)
                f["flick_mean"] = float(np.nanmean(dd))
                f["flick_max"] = float(np.nanmax(dd))
            fut = od[(od[:, 0] > ti) & (od[:, 0] <= ti + 2.0)]
            if len(fut) >= 5 and np.abs(fut[:, 4]).mean() > 0.2:
                dd = dist_to_polyline(fut[:, 1:3], q)
                f["follow_mean"] = float(dd.mean())
                f["follow_max"] = float(dd.max())
            prev = p
        frames.append(f)
    return dict(bag=path.stem, frames=frames)


def summary(res: dict) -> dict:
    fr = res["frames"]
    full = [f for f in fr if not f["empty"]]

    def pct(key: str, q: float) -> float:
        a = np.array([f[key] for f in full if key in f])
        return float(np.percentile(a, q)) if len(a) else float("nan")

    def frac(cond) -> float:
        return float(np.mean([cond(f) for f in full])) if full else float("nan")

    return dict(
        bag=res["bag"],
        frames=len(fr),
        empty_pct=100.0 * float(np.mean([f["empty"] for f in fr])),
        len_med=pct("len", 50),
        kmax_p95=pct("kmax", 95),
        s_bend_pct=100 * frac(lambda f: f["s_bends"] >= 1),
        loop_pct=100 * frac(lambda f: f["turn_abs_deg"] > 300),
        pivot_pct=100 * frac(lambda f: abs(f.get("start_err_deg", 0)) > 30),
        flick_p50=pct("flick_mean", 50),
        flick_p95=pct("flick_mean", 95),
        jump_pct=100 * frac(lambda f: f.get("flick_max", 0) > 0.5),
        follow_p50=pct("follow_mean", 50),
        follow_p95=pct("follow_mean", 95),
        follow_bad_pct=100 * frac(lambda f: f.get("follow_max", 0) > 0.75),
    )


def main() -> None:
    allres = [score(Path(p)) for p in sys.argv[1:]]
    out = Path(sys.argv[1]).parent / "scores.json"
    out.write_text(json.dumps(allres))
    rows = [summary(r) for r in allres]
    keys = list(rows[0])
    print(" ".join(f"{k[:12]:>12s}" for k in keys))
    for r in rows:
        print(
            " ".join(
                f"{v:>12.2f}" if isinstance(v, float) else f"{str(v)[-12:]:>12s}"
                for v in r.values()
            )
        )


if __name__ == "__main__":
    main()
