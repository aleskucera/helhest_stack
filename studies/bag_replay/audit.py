"""Audit the coarse "which way" map the node built on a real bag.

Reads a node recording (elevation_node `plan_debug_record`, see run_bag.sh) and answers, per
bag, the questions that decide whether the coarse layer can be trusted on the robot:

  - how much of the map it sealed, and whether the robot's own path ran through sealed blocks
    (it should not: the robot drove there, so a sealed block on the path is a false wall);
  - whether the tall structure the map saw (cells `elevated_m` above their block's floor) sits
    in sealed blocks (it should: a wall in a passable block is a wall the router will drive at);
  - how many blocks the shadow bridge invented, and whether the path crossed one;
  - how often the coarse and the fine field had no route at the robot's own pose.

One PNG per bag -- the final accumulated map with the sealed / bridged blocks, the path and the
goals over it, and the final coarse value beside it -- plus a JSON with the numbers.

  python studies/bag_replay/audit.py studies/bag_replay/out/*.npz --out studies/bag_replay/out
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# CoarseRouter's own defaults (helhest/planning/coarse.py); the node passes neither
MIN_PASS_FRACTION = 0.5
ELEVATED_M = 0.5
REACH_M = 0.3  # plan_reach_radius on the robot


def audit(npz: pathlib.Path, out_dir: pathlib.Path) -> dict:
    d = np.load(npz)
    name = npz.stem
    cell = float(d["cell"])
    ccell = float(d["coarse_cell"])
    fac = int(round(ccell / cell))
    cx0, cy0 = (float(x) for x in d["coarse_bounds"])
    h, seen_f = d["final_h"], d["final_seen"] > 0
    seen = d["coarse_seen"] > 0.5
    passable = d["coarse_passable"]
    bridged = d["coarse_bridged"] > 0.5
    floor = d["coarse_floor"]
    sealed = seen & (passable < MIN_PASS_FRACTION)
    cv = d["coarse_V"]
    cap = float(cv.max())
    unreach = cv >= 0.9 * cap

    # tall structure: fine cells well above their block's floor, in blocks the router has seen
    fr = np.minimum(np.arange(h.shape[0]) // fac, floor.shape[0] - 1)
    fc = np.minimum(np.arange(h.shape[1]) // fac, floor.shape[1] - 1)
    floor_f = floor[fr[:, None], fc[None, :]]
    tall = seen_f & (h - floor_f > ELEVATED_M) & seen[fr[:, None], fc[None, :]]
    tall_sealed = tall & sealed[fr[:, None], fc[None, :]]

    # the path, block by block
    trail = d["trail"]
    tr = np.clip(((trail[:, 1] - cy0) / ccell).astype(int), 0, sealed.shape[0] - 1)
    tc = np.clip(((trail[:, 0] - cx0) / ccell).astype(int), 0, sealed.shape[1] - 1)
    on_sealed = sealed[tr, tc]
    on_bridged = bridged[tr, tc]
    on_unseen = ~seen[tr, tc]
    blocks_visited = {(int(r), int(c)) for r, c in zip(tr, tc)}
    sealed_visited = sorted({(r, c) for r, c in blocks_visited if sealed[r, c]})

    # per recorded frame: a route at the robot's own pose, coarse and fine
    meta = d["hist_meta"]
    hcv = d["hist_cv"]
    rr = np.clip(((meta[:, 2] - cy0) / ccell).astype(int), 0, hcv.shape[1] - 1)
    rc = np.clip(((meta[:, 1] - cx0) / ccell).astype(int), 0, hcv.shape[2] - 1)
    cv_here = hcv[np.arange(len(meta)), rr, rc]
    coarse_dead = cv_here >= 0.9 * 1.0e30 if hcv.max() > 1.0e29 else cv_here >= 0.9 * hcv.max()
    vcap = float(np.nanmax(d["hist_v"]))
    fine_dead = meta[:, 11] >= 0.9 * vcap
    # (a "commanded zero while far from the goal" count was tried here and dropped: on a replay
    # the robot moves as recorded whatever the node commands, so the turn-first brake's
    # stop-before-spin holds a zero for as long as the bag's robot keeps driving.)

    # goals: one segment per distinct goal, closest approach in each
    goals = d["hist_goal"]
    segs = []
    start = 0
    for i in range(1, len(goals) + 1):
        if i == len(goals) or np.any(goals[i] != goals[start]):
            dist = meta[start:i, 6]
            segs.append(
                dict(
                    goal=[round(float(goals[start][0]), 2), round(float(goals[start][1]), 2)],
                    frames=int(i - start),
                    closest_m=round(float(dist.min()), 2),
                    reached=bool(dist.min() < REACH_M),
                )
            )
            start = i

    n_seen = int(seen.sum())
    stats = dict(
        bag=name,
        frames_recorded=int(len(meta)),
        frames_planned=int(len(trail)),
        bag_time_s=round(float(d["hist_t"][-1] - d["hist_t"][0]), 1),
        path_m=round(float(np.hypot(*np.diff(trail, axis=0).T).sum()), 1),
        blocks=dict(
            seen=n_seen,
            sealed=int(sealed.sum()),
            bridged=int(bridged.sum()),
            unreachable_seen=int((unreach & seen).sum()),
        ),
        tall_cells=int(tall.sum()),
        tall_in_sealed_frac=(
            round(float(tall_sealed.sum() / tall.sum()), 3) if tall.any() else None
        ),
        path=dict(
            samples=int(len(trail)),
            in_sealed=int(on_sealed.sum()),
            in_bridged=int(on_bridged.sum()),
            in_unseen=int(on_unseen.sum()),
            sealed_blocks_crossed=len(sealed_visited),
        ),
        no_route_frames=dict(
            coarse=int(coarse_dead.sum()),
            fine_own_heading=int(fine_dead.sum()),
        ),
        goals=segs,
        reached=int(sum(s["reached"] for s in segs)),
    )

    # --- the picture: the seen part of the map, blocks over it, the path and the goals
    ys, xs = np.nonzero(seen_f)
    m = int(round(2.0 / cell))
    r0, r1 = max(ys.min() - m, 0), min(ys.max() + m, h.shape[0])
    c0, c1 = max(xs.min() - m, 0), min(xs.max() + m, h.shape[1])
    ext = (cx0 + c0 * cell, cx0 + c1 * cell, cy0 + r0 * cell, cy0 + r1 * cell)
    hv = np.where(seen_f, h, np.nan)[r0:r1, c0:c1]
    lo, hi = np.nanpercentile(hv, [2, 98])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    ax = axes[0]
    ax.imshow(
        hv, origin="lower", extent=ext, cmap="terrain", vmin=lo, vmax=hi, interpolation="nearest"
    )
    br0, br1, bc0, bc1 = r0 // fac, (r1 + fac - 1) // fac, c0 // fac, (c1 + fac - 1) // fac
    bext = (cx0 + bc0 * ccell, cx0 + bc1 * ccell, cy0 + br0 * ccell, cy0 + br1 * ccell)
    overlay = np.zeros((br1 - br0, bc1 - bc0, 4), np.float32)
    overlay[sealed[br0:br1, bc0:bc1]] = (0.85, 0.1, 0.1, 0.45)
    overlay[bridged[br0:br1, bc0:bc1]] = (1.0, 0.6, 0.0, 0.55)
    ax.imshow(overlay, origin="lower", extent=bext, interpolation="nearest")
    ty, tx = np.nonzero(tall[r0:r1, c0:c1])
    ax.scatter(cx0 + (c0 + tx + 0.5) * cell, cy0 + (r0 + ty + 0.5) * cell, s=1, c="k", alpha=0.5)
    ax.plot(trail[:, 0], trail[:, 1], color="#1f5fd0", lw=1.5)
    ax.plot(trail[0, 0], trail[0, 1], "o", color="#20a040", ms=8)
    for s in segs:
        ax.plot(*s["goal"], marker="*", ms=12, color="#b5179e", mec="k")
    ax.legend(
        handles=[
            Patch(color=(0.85, 0.1, 0.1, 0.45), label="sealed block"),
            Patch(color=(1.0, 0.6, 0.0, 0.55), label="shadow-bridged block"),
            Patch(color="k", label=f"cell > {ELEVATED_M} m above its block's floor"),
        ],
        loc="upper right",
        fontsize=8,
    )
    ax.set_title(
        f"{name}: final map, path {stats['path_m']} m, {len(segs)} goals ({stats['reached']} reached)"
    )
    ax = axes[1]
    cvv = np.where(unreach, np.nan, cv)[br0:br1, bc0:bc1]
    im = ax.imshow(cvv, origin="lower", extent=bext, cmap="viridis", interpolation="nearest")
    ax.imshow(
        np.where(bridged | sealed, 1.0, np.nan)[br0:br1, bc0:bc1],
        origin="lower",
        extent=bext,
        cmap="gray",
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    ax.plot(trail[:, 0], trail[:, 1], color="w", lw=1.2)
    fig.colorbar(im, ax=ax, shrink=0.7, label="coarse value at the last goal [m]")
    ax.set_title("final coarse cost-to-go (grey: sealed or bridged; blank: no route)")
    png = out_dir / f"{name}_audit.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    (out_dir / f"{name}_audit.json").write_text(json.dumps(stats, indent=1))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", nargs="+", type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("studies/bag_replay/out"))
    a = p.parse_args()
    rows = [audit(f, a.out) for f in a.npz]
    print(
        "| bag | bag s | path m | goals reached | blocks seen | sealed | bridged | "
        "tall cells in sealed | path samples in sealed / bridged / unseen | "
        "frames no coarse route at robot / no fine route at own heading |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in rows:
        b, pa, nr = s["blocks"], s["path"], s["no_route_frames"]
        print(
            f"| {s['bag']} | {s['bag_time_s']} | {s['path_m']} | {s['reached']}/{len(s['goals'])} | "
            f"{b['seen']} | {b['sealed']} | {b['bridged']} | {s['tall_in_sealed_frac']} of {s['tall_cells']} | "
            f"{pa['in_sealed']} / {pa['in_bridged']} / {pa['in_unseen']} of {pa['samples']} | "
            f"{nr['coarse']} / {nr['fine_own_heading']} of {s['frames_recorded']} |"
        )


if __name__ == "__main__":
    main()
