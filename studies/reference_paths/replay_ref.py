"""Rerun the reference stage offline on recorded frames (the node's plan_ref_record or drive_sim's
--ref-record) and score every curve against the map the planner saw.

    python replay_ref.py <rec.npz> [--set key=value ...] [--out scored.npz]

`--set` overrides ReferencePathParams fields (a variant); with none, the replay is the deployed
stage and its curves are checked against the recorded ones (they must agree: same inputs, same
warm-start chain). Warm starts chain through the frames in order, as on the robot.

Per-frame metrics, map-aware:
  empty      no usable curve (< 3 points): MPPI descends V that frame
  wall_m     the least clearance of the wheels (front and rear axle discs) along the curve, in m
             past the half-width, measured on the settle's every-heading forbidden distance
  veto       points of the curve the settle forbids at the curve's own heading
  detour     curve length over the taut string's length
  end_gap    curve end to the taut string's end (did it get where V said to go)
  v_drop     V at the robot minus V at the curve's end: progress the curve buys, in V's metres
  kmax, s_bends, turn_abs_deg, start_err_deg    shape, as in score.py
  flick      mean distance of this curve's first 4 m from last frame's curve (world frame)
  ms         the stage's own time this frame
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

from helhest.control.reference_path import ReferencePath
from helhest.control.reference_path import ReferencePathParams
from helhest.grid import Grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import dist_to_polyline  # noqa: E402
from score import resample  # noqa: E402
from score import shape  # noqa: E402
from score import wrap  # noqa: E402


def ragged(d: np.lib.npyio.NpzFile, key: str) -> list[np.ndarray]:
    n = d[f"{key}_n"]
    off = np.concatenate([[0], np.cumsum(n)])
    return [d[key][off[i] : off[i + 1]] for i in range(len(n))]


def make_grid(row: np.ndarray) -> Grid:
    g = Grid()
    g.origin_x, g.origin_y, g.cell_size = float(row[0]), float(row[1]), float(row[2])
    g.cells_x, g.cells_y = int(row[3]), int(row[4])
    return g


def sample(field: np.ndarray, g: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Bilinear sample of a [ny, nx] field on a centre-origin grid row, clamped."""
    fx = (p[:, 0] - g[0]) / g[2]
    fy = (p[:, 1] - g[1]) / g[2]
    ny, nx = field.shape
    x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    ax = np.clip(fx - x0, 0, 1)
    ay = np.clip(fy - y0, 0, 1)
    return (
        (1 - ax) * (1 - ay) * field[y0, x0]
        + ax * (1 - ay) * field[y0, x0 + 1]
        + (1 - ax) * ay * field[y0 + 1, x0]
        + ax * ay * field[y0 + 1, x0 + 1]
    )


def veto_count(blocked: np.ndarray, g: np.ndarray, p: np.ndarray) -> int:
    """Curve points (past the first two) whose cell is forbidden at the curve's own heading."""
    if len(p) < 4:
        return 0
    nt = blocked.shape[2]
    d = np.diff(p, axis=0)
    th = np.arctan2(d[:, 1], d[:, 0])
    q = p[2:-1]
    th = th[2:]
    c = np.clip(np.round((q[:, 0] - g[0]) / g[2]).astype(int), 0, blocked.shape[1] - 1)
    r = np.clip(np.round((q[:, 1] - g[1]) / g[2]).astype(int), 0, blocked.shape[0] - 1)
    t = np.round((th % (2 * np.pi)) / (2 * np.pi / nt)).astype(int) % nt
    return int(blocked[r, c, t].sum())


def metrics(curve, taut, d, i, half_width, rear, prev_world) -> dict:
    """The map-aware score of one frame's curve."""
    m: dict = {"empty": len(curve) < 3}
    st = d["state"][i]
    if m["empty"]:
        return m
    g = d["sgrid"][i]
    clear = d["clear"][i]
    V = d["V"][i]
    vmin = V.min(2)
    q = resample(curve, 0.1)
    dq = np.diff(q, axis=0)
    th = np.arctan2(dq[:, 1], dq[:, 0])
    rear_pts = q[1:] - rear * np.stack([np.cos(th), np.sin(th)], 1)
    wall = np.concatenate([sample(clear, g, q[1:]), sample(clear, g, rear_pts)]) - half_width
    m["wall_m"] = float(wall.min())
    m["veto"] = veto_count(d["blocked"][i], g, curve)
    L = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
    Lt = float(np.linalg.norm(np.diff(taut, axis=0), axis=1).sum()) if len(taut) > 1 else 0.0
    m["detour"] = L / Lt if Lt > 0.5 else float("nan")
    m["end_gap"] = float(np.linalg.norm(curve[-1] - taut[-1])) if len(taut) else float("nan")
    m["v_drop"] = float(sample(vmin, g, st[None, :2])[0] - sample(vmin, g, curve[-1:])[0])
    m.update(shape(curve))
    d0 = curve[min(2, len(curve) - 1)] - curve[0]
    m["start_err_deg"] = float(np.degrees(wrap(np.arctan2(d0[1], d0[0]) - st[2])))
    world = curve + d["origin"][i][None]
    if prev_world is not None and len(prev_world) >= 3:
        qw = resample(world, 0.1)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(qw, axis=0), axis=1))])
        m["flick"] = float(np.nanmean(dist_to_polyline(qw[s <= 4.0], prev_world)))
    return m


def run(rec: Path, overrides: dict, check: bool = True) -> list[dict]:
    d = np.load(rec)
    base = json.loads(str(d["params"]))
    fields = {f.name for f in dataclasses.fields(ReferencePathParams)}
    prm = ReferencePathParams(**{k: v for k, v in base.items() if k in fields})
    prm = dataclasses.replace(prm, **overrides)
    nt = d["V"].shape[3]
    # the stage subtracts forbidden_growth_m from the half-width it is given
    hw_in = float(d["half_width"]) + base.get("forbidden_growth_m", 0.3)
    rp = ReferencePath(nt, hw_in, float(d["rear"]), prm, device="cuda:0")
    hw = rp.half_width
    shp = d["V"].shape[1:]
    V = wp.zeros(shp, dtype=wp.float32)
    B = wp.zeros(shp, dtype=wp.float32)
    C = wp.zeros(d["clear"].shape[1:], dtype=wp.float32)
    rec_curves = ragged(d, "curve")
    out, prev_world, mism = [], None, 0
    for i in range(len(d["t"])):
        wp.copy(V, wp.array(d["V"][i], dtype=wp.float32))
        wp.copy(B, wp.array(d["blocked"][i].astype(np.float32), dtype=wp.float32))
        wp.copy(C, wp.array(d["clear"][i], dtype=wp.float32))
        g = make_grid(d["sgrid"][i])
        wp.synchronize()
        t0 = time.perf_counter()
        n = rp.update(V, g, C, g, d["state"][i], goal_xy=d["goal"][i], blocked=B)
        ms = (time.perf_counter() - t0) * 1e3
        curve = rp.curve() if n >= 3 else np.zeros((0, 2), np.float32)
        taut = rp.taut.numpy()[: int(rp.n_taut.numpy()[0])]
        if check and not overrides:
            rc = rec_curves[i]
            if len(rc) != len(curve) or (len(rc) and np.abs(rc - curve).max() > 1e-3):
                mism += 1
        m = metrics(curve, taut, d, i, hw, float(d["rear"]), prev_world)
        m.update(i=i, t=float(d["t"][i] - d["t"][0]), ms=ms, cold=int(rp.cold.numpy()[0]))
        st = d["state"][i]
        m.update(x=float(st[0] + d["origin"][i][0]), y=float(st[1] + d["origin"][i][1]))
        out.append(m)
        if not m["empty"]:
            prev_world = curve + d["origin"][i][None]
    if check and not overrides:
        print(f"replay vs recorded curves: {mism} of {len(out)} frames differ")
    return out


def summarize(fr: list[dict]) -> dict:
    full = [f for f in fr if not f["empty"]]

    def pct(k: str, q: float) -> float:
        a = np.array([f[k] for f in full if k in f and np.isfinite(f[k])])
        return float(np.percentile(a, q)) if len(a) else float("nan")

    def frac(c) -> float:
        return 100.0 * float(np.mean([c(f) for f in full])) if full else float("nan")

    return dict(
        frames=len(fr),
        empty=100.0 * float(np.mean([f["empty"] for f in fr])),
        wall_neg=frac(lambda f: f["wall_m"] < 0.0),
        wall_p5=pct("wall_m", 5),
        veto=frac(lambda f: f["veto"] > 0),
        detour_p95=pct("detour", 95),
        end_gap_p95=pct("end_gap", 95),
        s_bend=frac(lambda f: f["s_bends"] >= 1),
        loop=frac(lambda f: f["turn_abs_deg"] > 300),
        kmax_p95=pct("kmax", 95),
        flick_p95=pct("flick", 95),
        jump=frac(lambda f: f.get("flick", 0) > 0.3),
        cold=frac(lambda f: f["cold"] == 1),
        ms_p50=float(np.median([f["ms"] for f in fr])),
        ms_max=float(np.max([f["ms"] for f in fr])),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("rec", nargs="+")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    ov = {}
    for kv in a.set:
        k, v = kv.split("=")
        ov[k] = type(getattr(ReferencePathParams(), k))(float(v))
    wp.init()
    allfr = {}
    for r in a.rec:
        fr = run(Path(r), ov)
        allfr[Path(r).stem] = fr
        s = summarize(fr)
        print(Path(r).stem, " ".join(f"{k}={v:.2f}" for k, v in s.items()))
    if a.out:
        Path(a.out).write_text(json.dumps(allfr))


if __name__ == "__main__":
    main()
