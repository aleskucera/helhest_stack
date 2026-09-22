"""Judge the PLAN, not the driving: from each recorded pose, does the policy reach the goal?

The instrument that should have existed before any of the controller work. A run's outcome mixes
the planner and the controller, and every attribution made from reach/no-reach during this
study turned out to be confounded -- three commits changed two or three things each, and a
carrot follower built to isolate the planner measured its own pure-pursuit bugs twice before
producing a usable number.

This needs no controller and no simulator. It replays each recorded map, rebuilds the field, and
walks the lattice's own policy from the pose the robot actually held. The question it answers --
"was there a plan from here" -- is the planner's alone.

What it found, on six worlds, no routing layer vs the 0.2 m one (`ab_A` vs `ab_C`):

    world     no routing layer   with it     what MPPI did
    gap             100%            96%      reached both
    slalom           32%            89%      fail -> reach
    pillars          61%            94%      fail -> reach
    pocket           67%            79%      fail -> reach
    ridge            74%            78%      reached both
    bumpy            35%            39%      reached both

The three worlds the routing layer flipped for MPPI are the three where plan usability jumps.
The three MPPI reached either way are the three where it barely moves. So the layer's benefit is
a PLANNER effect, not the controller exploiting a lattice cost.

And `bumpy` is the reverse case worth knowing: the planner has no route 6 frames in 10 either
way, yet MPPI reaches it comfortably. There the controller carries the run -- which is the same
`bumpy` where MPPI drives through vetoed poses 20% of the time, including 1.7 s at 9.5 degrees
past the nose-down limit. It reaches that world by overriding the plan.

"Usable" is a proxy: the policy ends within a metre of the goal, or runs most of the way to the
window edge toward it. The absolute percentages move with that threshold. The A-vs-C differences
do not -- and they were predicted by nothing, then matched the independent reach outcomes
world for world, which is why they are worth believing. The two arms drove different
trajectories, so frames are not matched pose for pose.

  python studies/closed_loop/plan_quality.py --a out/ab_A --c out/ab_C
"""

import argparse

import numpy as np, warp as wp
from helhest import dynamics
from helhest.engine import GridParams
from helhest.planning.coarse import CoarseRouter
from helhest.planning.costtogo import CostToGo
from helhest.planning.lattice_solver import trace_optimal

wp.init()
NT, DT = 16, 1.0 / 14.5
dev = lambda v: wp.array(np.ascontiguousarray(v, np.float32), dtype=wp.float32)


def judge(npz, coarsen):
    d = np.load(npz)
    h, seen, meta = d["hist_h"], d["hist_seen"], d["hist_meta"]
    cell = float(d["cell"])
    n = h.shape[1]
    nr = d["hist_blk"].shape[1]
    off_r = n // 2 - nr // 2
    goal = d["goal"]
    grid = GridParams(cells_x=nr, cells_y=nr, cell_size=cell, origin_x=0.0, origin_y=0.0)
    bg = GridParams(cells_x=n, cells_y=n, cell_size=cell, origin_x=0.0, origin_y=0.0)
    ctg = CostToGo(
        grid,
        dynamics.robot_params(0.10),
        dynamics.planning_solver(dt=DT, command_delay=0.0),
        n_theta=NT,
        z_veto=2.0,
        device="cuda",
    )
    coarse = None
    if coarsen:
        coarse = CoarseRouter(bg, factor=coarsen, device="cuda")
        ctg.set_coarse(
            GridParams(
                coarse.grid.cells_x,
                coarse.grid.cells_y,
                coarse.grid.cell_size,
                -off_r * cell,
                -off_r * cell,
            )
        )
    have, reach, lens = 0, 0, []
    for i in range(len(meta)):
        f, x, y, yaw = (float(v) for v in meta[i][:4])
        bx, by = float(meta[i][7]), float(meta[i][8])
        r0, s0 = bx + off_r * cell, by + off_r * cell
        crop = np.ascontiguousarray(h[i][off_r : off_r + nr, off_r : off_r + nr])
        kw = {}
        if coarse is not None:
            m = np.ascontiguousarray((seen[i] > 0).astype(np.float32))
            kw["coarse_value"] = coarse.solve(dev(h[i]), dev(m), (goal[0] - bx, goal[1] - by))
        ctg.compute(dev(crop), (goal[0] - r0, goal[1] - s0), **kw)
        p = trace_optimal(ctg, (x - r0, y - s0, yaw), NT, nr, nr, 0.0, 0.0, cell)
        have += 1
        if len(p) >= 2:
            end = p[-1] + np.array([r0, s0])
            gd = float(np.hypot(end[0] - goal[0], end[1] - goal[1]))
            plen = float(np.hypot(*np.diff(p, axis=0).T).sum())
            # "reaches" = the policy walk ends at the goal cell, or at the window edge heading there
            if gd < 1.0 or plen > 0.8 * min(
                float(np.hypot(x - goal[0], y - goal[1])), nr * cell / 2
            ):
                reach += 1
                lens.append(plen)
    return have, reach, (np.median(lens) if lens else 0.0)


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--a", default="studies/closed_loop/out/ab_A", help="run WITHOUT a routing layer")
ap.add_argument("--c", default="studies/closed_loop/out/ab_C", help="run WITH one")
ap.add_argument("--coarsen", type=int, default=1, help="the --coarsen the `--c` run used")
args = ap.parse_args()

print(f"  {'world':<9s} {'no routing layer':>24s} {'with it':>26s}")
for w in ("gap", "slalom", "pillars", "pocket", "ridge", "bumpy"):
    a = judge(f"{args.a}/{w}.npz", 0)
    c = judge(f"{args.c}/{w}.npz", args.coarsen)
    print(
        f"  {w:<9s} {a[1]:>6d}/{a[0]:<4d} ({100*a[1]/max(a[0],1):>3.0f}%) med {a[2]:>5.1f} m"
        f"   {c[1]:>6d}/{c[0]:<4d} ({100*c[1]/max(c[0],1):>3.0f}%) med {c[2]:>5.1f} m"
    )
