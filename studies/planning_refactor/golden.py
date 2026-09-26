"""Freeze what `CostToGo` and `CoarseRouter` compute, so the planning refactor is held to it.

    python studies/planning_refactor/golden.py record   # arrays -> golden/, hashes -> hashes.json
    python studies/planning_refactor/golden.py check    # recompute; on a mismatch, say where

The bar is bit-identical, so the check compares SHA-256 hashes of every field. A hash pins one
machine's GPU and Warp build (transcendentals are not bit-portable across architectures), which
is why this lives here and not in tests/: record and check on the same machine. Every run compiles
from scratch into its own cache (a few minutes), see `compute`. The arrays are
kept locally (gitignored) only to diagnose a mismatch -- how many cells, how large, which flips.

Five configurations over 8 stress-world windows (3 poses each) and 6 real-bag frames:

    A  deployed: the params file's cost-to-go (clearance route, robust tube, pivots) + a
       world-anchored coarse layer
    B  A without the clearance route (the tube vetoes), no coarse layer
    C  B + the z-margin (z_veto 2, charge 0.5) with sigma and drift supplied -- the sim-only path
    D  B + the step gate (0.3 m), with blind cells in the measured mask
    E  A with a window-bound coarse layer and the goal 30 m ahead, so the ring is seeded
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np
import warp as wp

from helhest import dynamics
from helhest import worlds as W
from helhest.engine import GridParams
from helhest.planner_config import planner_config
from helhest.planning.coarse import CoarseRouter
from helhest.planning.costtogo import CostToGo

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "studies/clearance"))
import worlds as _study_worlds  # noqa: E402,F401  registers corridor24 / corridor26 / cornerL24

N = 66  # routing window [cells]
CELL = 0.24  # routing cell = map cell 0.08 x plan_lat_coarsen 3
RANGE_M = 6.0  # past this from the robot a world window reads as unmeasured
WORLD_NAMES = [
    "slalom",
    "false_door",
    "pocket",
    "cornerL24",
    "corridor24",
    "corridor26",
    "pillars",
    "gap",
]
BAGS = [
    ("in_speed_new", (5, 30, 60)),
    ("out_new", (100, 350, 600)),
]
FIELDS = ("V", "V_escape", "blocked", "hazard", "zmargin", "doubt", "_pose_cost", "_seeds")


def _deployed_params() -> dict:
    """The resolved plan_* config the robot ran on the recorded bags (no yaml needed)."""
    d = np.load(REPO / "studies/bag_replay/out2/in_speed_new.npz", allow_pickle=True)
    return json.loads(str(d["plan_config"]))


def _world_windows() -> list[dict]:
    out = []
    for name in WORLD_NAMES:
        build, start, goal = W.WORLDS[name]
        hm = build(cell=CELL)
        for frac in (0.0, 0.4, 0.8):
            rx = start[0] + frac * (goal[0] - start[0])
            ry = start[1] + frac * (goal[1] - start[1])
            x0, y0 = rx - N * CELL / 2, ry - N * CELL / 2
            h = np.zeros((N, N), np.float32)
            inside = np.zeros((N, N), bool)
            c0 = int(round((x0 - hm.x0) / CELL))
            r0 = int(round((y0 - hm.y0) / CELL))
            for r in range(N):
                rr = r0 + r
                if not 0 <= rr < hm.ny:
                    continue
                for c in range(N):
                    cc = c0 + c
                    if 0 <= cc < hm.nx:
                        h[r, c] = hm.H[rr, cc]
                        inside[r, c] = True
            yy, xx = np.mgrid[0:N, 0:N]
            dist = np.hypot((xx - N / 2) * CELL, (yy - N / 2) * CELL)
            measured = (inside & (dist <= RANGE_M)).astype(np.float32)
            out.append(
                dict(
                    name=f"{name}_{frac:.1f}",
                    h=h,
                    measured=measured,
                    origin=(x0, y0),
                    goal=(float(goal[0]), float(goal[1])),
                )
            )
    return out


def _bag_windows() -> list[dict]:
    out = []
    for bag, frames in BAGS:
        d = np.load(REPO / f"studies/bag_replay/out2/{bag}.npz", allow_pickle=True)
        for i in frames:
            h8 = d["hist_h"][i]
            s8 = d["hist_seen"][i].astype(bool)
            k = 3
            n8 = (h8.shape[0] // k) * k
            hb = np.where(s8, h8, -np.inf)[:n8, :n8].reshape(n8 // k, k, n8 // k, k)
            sb = s8[:n8, :n8].reshape(n8 // k, k, n8 // k, k).any(axis=(1, 3))
            h = np.where(sb, hb.max(axis=(1, 3)), 0.0).astype(np.float32)
            meta = d["hist_meta"][i]
            out.append(
                dict(
                    name=f"{bag}_{i}",
                    h=np.ascontiguousarray(h[:N, :N]),
                    measured=np.ascontiguousarray(sb[:N, :N].astype(np.float32)),
                    origin=(float(meta[7]), float(meta[8])),
                    goal=(float(d["hist_goal"][i][0]), float(d["hist_goal"][i][1])),
                )
            )
    return out


def _belief(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic measurement sd and drift: sd grows with range, drift with range, -1 unmeasured."""
    yy, xx = np.mgrid[0:N, 0:N]
    dist = np.hypot((xx - N / 2) * CELL, (yy - N / 2) * CELL)
    sd = (0.004 + 0.003 * dist).astype(np.float32)
    drift = np.where(case["measured"] > 0.5, 2.0e-5 * dist * dist, -1.0).astype(np.float32)
    return sd, drift


def _configs() -> dict[str, dict]:
    cfg = planner_config(_deployed_params())
    base = dict(cfg.costtogo)
    no_route = {k: v for k, v in base.items() if k != "clearance"}
    return {
        "A": dict(ctg=base, coarse="memory"),
        "B": dict(ctg=no_route, coarse=None),
        "C": dict(ctg={**no_route, "z_veto": 2.0, "charge_per_sigma": 0.5}, coarse=None, sd=True),
        "D": dict(ctg={**no_route, "obstacle_step_m": 0.3}, coarse=None),
        "E": dict(ctg=base, coarse="window", far_goal=True),
    }


def _run_config(key: str, spec: dict, cases: list[dict], device: str) -> dict[str, np.ndarray]:
    robot = dynamics.robot_params(0.1)
    grid = GridParams(N, N, CELL, 0.0, 0.0)
    ctg = CostToGo(grid, robot, dynamics.planning_solver(), **spec["ctg"], device=device)
    coarse = None
    if spec["coarse"] == "memory":
        nm = int(round(60.0 / CELL))
        memory = GridParams(nm, nm, CELL, -0.5 * nm * CELL, -0.5 * nm * CELL)
        coarse = CoarseRouter(grid, factor=2, bridge_m=1.2, memory_grid=memory, device=device)
    elif spec["coarse"] == "window":
        coarse = CoarseRouter(grid, factor=2, bridge_m=1.2, device=device)
    if coarse is not None:
        cg = coarse.grid
        ctg.set_coarse(GridParams(cg.cells_x, cg.cells_y, cg.cell_size, 0.0, 0.0))

    out: dict[str, np.ndarray] = {}
    for case in cases:
        x0, y0 = case["origin"]
        gx, gy = case["goal"]
        if spec.get("far_goal"):
            gx, gy = x0 + N * CELL / 2 + 30.0, y0 + N * CELL / 2
        h = wp.array(case["h"], dtype=wp.float32, device=device)
        m = wp.array(case["measured"], dtype=wp.float32, device=device)
        kw = {}
        if spec.get("sd"):
            sd, drift = _belief(case)
            kw = dict(
                sigma=wp.array(sd, dtype=wp.float32, device=device),
                drift=wp.array(drift, dtype=wp.float32, device=device),
            )
        vc, corigin = None, None
        if coarse is not None:
            if coarse.persistent:
                vc = coarse.solve(h, m, (gx, gy), (x0, y0))
                # as the node and drive_sim do: the anchored grid's origin, in this window's frame
                corigin = (coarse.grid.origin_x - x0, coarse.grid.origin_y - y0)
            else:
                vc = coarse.solve(h, m, (gx - x0, gy - y0))
                corigin = (0.0, 0.0)
            out[f"{case['name']}/coarse_V"] = vc.numpy().copy()
        ctg.compute(h, (gx - x0, gy - y0), measured=m, coarse_value=vc, coarse_origin=corigin, **kw)
        for f in FIELDS:
            out[f"{case['name']}/{f}"] = getattr(ctg, f).numpy().copy()
        bearing = ctg.descent_bearing(N * CELL / 2, N * CELL / 2, 1.0)
        out[f"{case['name']}/descent"] = np.array([bearing], np.float32)
    return out


def _hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def compute(device: str = "cuda:0") -> dict[str, dict[str, np.ndarray]]:
    # A private kernel cache, emptied every run. Warp's shared cache served a binary that a fresh
    # compile of the same source does not reproduce (one ill-conditioned settle pose moved), so
    # both record and check must build from the source as it stands.
    cache = HERE / "golden" / "kernel_cache"
    shutil.rmtree(cache, ignore_errors=True)
    wp.config.kernel_cache_dir = str(cache)
    wp.init()
    cases = _world_windows() + _bag_windows()
    return {k: _run_config(k, s, cases, device) for k, s in _configs().items()}


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    gdir = HERE / "golden"
    hfile = HERE / "hashes.json"
    res = compute()
    hashes = {k: {n: _hash(a) for n, a in fields.items()} for k, fields in res.items()}
    if mode == "record":
        gdir.mkdir(exist_ok=True)
        for k, fields in res.items():
            np.savez_compressed(gdir / f"{k}.npz", **fields)
        hfile.write_text(json.dumps(hashes, indent=1, sort_keys=True) + "\n")
        n = sum(len(v) for v in hashes.values())
        print(f"recorded {n} fields over {len(hashes)} configs -> {hfile.name}, arrays in golden/")
        return
    want = json.loads(hfile.read_text())
    bad = 0
    for k in sorted(want):
        ref = np.load(gdir / f"{k}.npz") if (gdir / f"{k}.npz").exists() else None
        for n in sorted(want[k]):
            got = hashes.get(k, {}).get(n)
            if got == want[k][n]:
                continue
            bad += 1
            if got is None:
                print(f"{k} {n}: MISSING")
                continue
            if ref is None:
                print(f"{k} {n}: hash differs (no local arrays to diff)")
                continue
            a, b = ref[n], res[k][n]
            fin = np.isfinite(a) & np.isfinite(b)
            diff = np.abs(np.where(fin, a - b, 0.0))
            print(
                f"{k} {n}: {int((a != b).sum())} of {a.size} differ, max |d| {diff.max():.3g}, "
                f"non-finite mismatch {int((np.isfinite(a) != np.isfinite(b)).sum())}"
            )
    total = sum(len(v) for v in want.values())
    print(f"{'OK' if bad == 0 else 'FAIL'}: {total - bad}/{total} fields bit-identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
