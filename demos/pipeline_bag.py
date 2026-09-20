"""The pipeline on a real bag: a rolling belief window, replayed sweep by sweep.

Same chain as `pipeline_panels.py`, but nothing here is staged. The uncertainty is whatever the
dTOF and the fusion produce, the drift is whatever Odin's SLAM accrues, and the seams appear
where the robot's own driving puts them.

The window follows the robot (`recenter`, whole cells only, so nothing is resampled), which
means it also FORGETS: what scrolls off the trailing edge is gone. That is the fine layer's job
description, not a defect -- something else has to remember the far field.

The goal is taken from the robot's own future pose, a fixed distance further along the track it
actually drove, so it is always somewhere it genuinely went.

AS IT STANDS THE PLANNER REFUSES THIS MAP, and that is the honest output rather than a bug to
tune away. The window holds ~54 tall blobs, 78% of them one 0.2 m cell wide and around 1.1 m
high, each re-measured as often as the ground beside it. Dilated by a 1.45 m robot they close
every corridor, so 77.9% of cells are blocked at some heading and the goal is unreachable --
while the real robot drove straight through, over ground its own track shows to be flat to
3.3 cm at p90. Something has to give, and which thing is the open question: see `despike`.

    PYTHONPATH=studies:src .venv/bin/python demos/pipeline_bag.py out_odin0 --shots 4
    PYTHONPATH=studies:src .venv/bin/python demos/pipeline_bag.py out_odin0 --span 10 --cell 0.15
"""

from __future__ import annotations

import argparse
import sys


def fill_from_ground(height, seen, passes=24):
    """Fill unmeasured cells from their measured neighbours, not with a constant.

    This is the whole difference between a usable map and a wall of towers. Odin's ground sits
    around -1.5 m, so filling holes with 0.0 -- which is what nan_to_num does, and what the naive
    version of this demo did -- puts a 1.5 m step at every speckle gap and the settle refuses the
    entire window. Growing the measured surface outward keeps the fill on the ground it is
    standing next to, which is the only value that is not a claim.
    """
    import numpy as np

    out = np.where(seen, height, np.nan)
    for _ in range(passes):
        holes = np.isnan(out)
        if not holes.any():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(out.shape, np.int32)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s = np.roll(np.roll(out, dr, 0), dc, 1)
            ok = ~np.isnan(s)
            acc[ok] += s[ok]
            cnt[ok] += 1
        grow = holes & (cnt > 0)
        out[grow] = acc[grow] / cnt[grow]
    return np.nan_to_num(out, nan=float(np.nanmedian(np.where(seen, height, np.nan))))


def despike(height, k=1):
    """Median filter the height. OFF by default, and it is not a fix -- read this first.

    It looks like one. On out_odin0 at 82% coverage it takes the blocked fraction from 77.9% to
    22.8% and turns an unreachable goal into a 3.43 m route. But what it removes is not noise:

        the three tallest "spikes" stand 1.08-1.11 m above their neighbours
        each is built from 3-7 returns, not one stray point
        they are re-measured as often as the ground (19 returns/cell against 23)
        and as recently (both last seen 0.0 s ago)
        78% of the 54 tall blobs are a SINGLE 0.2 m cell

    Persistently observed, metre-tall, and thinner than the robot. That is the description of a
    post or a sapling -- and of the thin sticks this project has already driven through once
    because the settle straddled them. A 3x3 median flattens all three of those cells straight
    back to ground level. It deletes obstacles.

    So it stays here as an instrument, not a default: `--despike 1` shows how much of the
    blocking those tall thin things account for, which is most of it.
    """
    import numpy as np

    st = np.stack(
        [
            np.roll(np.roll(height, dr, 0), dc, 1)
            for dr in range(-k, k + 1)
            for dc in range(-k, k + 1)
        ]
    )
    return np.median(st, 0)


def replay(bag, span, cell, n_theta, k_sigma, shots, max_frames, outdir, device, despike_k):
    import os

    import numpy as np
    import warp as wp
    from elevation_belief import DriftRates
    from elevation_belief import ElevationBelief
    from elevation_belief import NoiseModel

    sys.path.insert(0, "studies")
    from calib import bagio
    from mcap_ros2.reader import read_ros2_messages

    from helhest.engine import GridParams
    from helhest.engine import RobotParams
    from helhest.engine import SolverParams
    from helhest.planning.costtogo import CostToGo

    path = bagio.bag_path(bag)
    odom = bagio.read_odometry(path)
    # cumulative path length, so "8 m further along" means 8 m DRIVEN rather than 8 m away
    step = np.r_[0.0, np.linalg.norm(np.diff(odom.xyz[:, :2], axis=0), axis=1)]
    arc = np.cumsum(step)

    n = int(round(span / cell))
    belief = ElevationBelief(
        (-span / 2, span / 2, -span / 2, span / 2),
        cell,
        # a + b*range, from studies/calib: the near-field sd lands at ~2 cm
        noise=NoiseModel("linear", a=0.012, b=0.004),
        rates=DriftRates.odin_slam(),  # fitted in studies/calib/fit_drift.py
        device=device,
    )
    # The planner works in WINDOW-LOCAL coordinates: the window slides, so a fixed world origin
    # would need the whole solver rebuilt every time it moved.
    ctg = CostToGo(
        GridParams(cells_x=n, cells_y=n, cell_size=cell, origin_x=0.0, origin_y=0.0),
        RobotParams(),
        SolverParams(),
        n_theta=n_theta,
        k_sigma=k_sigma,
        margin_weight=0.5,
        device=device,
    )

    total = sum(1 for _ in read_ros2_messages(path, topics=[bagio.CLOUD_TOPIC]))
    total = min(total, max_frames)
    marks = {int(total * f) for f in np.linspace(0.3, 0.97, shots)}
    os.makedirs(outdir, exist_ok=True)

    prev_t, k, saved = None, 0, []
    for msg in read_ros2_messages(path, topics=[bagio.CLOUD_TOPIC]):
        c = msg.ros_msg
        t = c.header.stamp.sec + c.header.stamp.nanosec * 1e-9
        i = int(np.argmin(np.abs(odom.t - t)))
        if abs(odom.t[i] - t) > 0.1:
            continue
        pts = bagio._unpack_cloud(c)
        if pts.size == 0:
            continue
        on = (
            (pts[:, 0] > bagio.SELF_X[0])
            & (pts[:, 0] < bagio.SELF_X[1])
            & (pts[:, 1] > bagio.SELF_Y[0])
            & (pts[:, 1] < bagio.SELF_Y[1])
        )
        pts = pts[~on]
        if pts.size == 0:
            continue
        roll, pitch, yaw = odom.rpy[i]
        w = pts.astype(np.float64) @ bagio._rpy_to_mat(roll, pitch, yaw).T + odom.xyz[i]
        # A height crop is a PRECONDITION of keep-the-highest fusion, not an optimisation:
        # without it anything overhead ends up in the map as ground.
        band = (w[:, 2] > odom.xyz[i][2] - 2.0) & (w[:, 2] < odom.xyz[i][2] + 0.6)
        w = w[band]
        if w.shape[0] < 200:
            continue

        rxy = (float(odom.xyz[i][0]), float(odom.xyz[i][1]))
        belief.recenter(rxy)
        if prev_t is not None and t > prev_t:
            belief.motion_update(t - prev_t, rxy)
        prev_t = t
        belief.measure_scan(w, odom.xyz[i])
        k += 1

        if k in marks:
            j = int(np.searchsorted(arc, arc[i] + 8.0))  # 8 m further along the driven track
            j = min(j, len(odom.t) - 1)
            goal_w = odom.xyz[j][:2]
            goal_local = (float(goal_w[0] - belief.xmin), float(goal_w[1] - belief.ymin))
            lo, hi = 1.5 * cell, span - 1.5 * cell
            goal_local = (min(max(goal_local[0], lo), hi), min(max(goal_local[1], lo), hi))

            lay = belief.layers()
            seen = lay["valid"].numpy() != 0
            height = fill_from_ground(lay["raw_h"].numpy(), seen)
            if despike_k:
                height = despike(height, despike_k)
            sd = np.sqrt(np.maximum(lay["meas_var"].numpy(), 0.0))
            a = lambda v: wp.array(  # noqa: E731
                np.ascontiguousarray(v, np.float32), dtype=wp.float32
            )
            ctg.solve_gap(
                a(height),
                goal_local,
                a(sd),
                measured=a(seen.astype(np.float32)),
                drift=a(belief.drift().numpy()),
            )
            robot_local = (rxy[0] - belief.xmin, rxy[1] - belief.ymin)
            saved.append(
                dict(
                    k=k,
                    t=t,
                    span=span,
                    seen=seen,
                    height=height,
                    sd=sd,
                    spread=ctg._spread.numpy(),
                    z=ctg.zmargin.numpy(),
                    doubt=ctg.doubt_pessimistic.numpy(),
                    vp=ctg.V_pessimistic.numpy(),
                    vo=ctg.V_optimistic.numpy(),
                    cap=ctg._vcap,
                    robot=robot_local,
                    goal=goal_local,
                    gap=ctg.gap_at(*robot_local, float(yaw)),
                    driven=float(arc[i]),
                )
            )
            print(
                f"  frame {k:>4d}  t={t - odom.t[0]:6.1f}s  driven {arc[i]:6.1f} m  "
                f"measured {100*seen.mean():4.1f}%  gap {saved[-1]['gap']['gap_m']:7.2f} m"
            )
        if k >= total:
            break
    return saved


def render(shots, outdir):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    for s in shots:
        span, seen, cap = s["span"], s["seen"], s["cap"]
        ext = (0.0, span, 0.0, span)
        mask = lambda v: np.where(seen, v, np.nan)  # noqa: E731

        def best(v):
            m = v.min(2)
            return np.where(seen, np.where(m >= cap * 0.9, -1.0, m), np.nan)

        bp, bo = best(s["vp"]), best(s["vo"])
        gap = np.where((bp < 0) | (bo < 0), -1.0, bp - bo)
        panels = [
            ("height  [m]", mask(s["height"]), "terrain", None),
            ("measurement sd  [m]", mask(s["sd"]), "magma", None),
            ("footprint drift spread  [m^2]", mask(s["spread"]), "magma", None),
            ("z: margin in sigmas", mask(np.clip(s["z"].min(2), 0, 8)), "viridis", None),
            ("doubt (blocked by IGNORANCE)", mask(s["doubt"].max(2)), "cividis", None),
            ("V believing the map  [m]", bp, "viridis", "route"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
        for ax, (title, img, cmap, lim) in zip(axes.ravel(), panels):
            if lim == "route":
                cm = plt.get_cmap(cmap).copy()
                cm.set_under("black")
                kw = dict(cmap=cm, vmin=0.0)
            else:
                kw = dict(cmap=cmap)
            ax.set_facecolor("0.82")
            im = ax.imshow(img, origin="lower", extent=ext, **kw)
            ax.plot(*s["robot"], "o", ms=9, mfc="deepskyblue", mec="k")
            ax.plot(*s["goal"], "*", ms=15, mfc="w", mec="k")
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046)
        g = s["gap"]
        reach = "reachable" if g["reachable_pessimistic"] else "UNREACHABLE believing the map"
        fig.suptitle(
            f"out_odin0, frame {s['k']}, {s['driven']:.0f} m driven   "
            f"|   window {span:.0f} m, rolling   |   {100*seen.mean():.0f}% measured   "
            f"|   gap {g['gap_m']:.2f} m, {reach}\n"
            "blue = robot, star = goal (8 m further along the track it drove)   "
            "grey = never measured   black = no route",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        p = f"{outdir}/bag_{s['k']:04d}.png"
        fig.savefig(p, dpi=105)
        plt.close(fig)
        out.append(p)
        print(f"wrote {p}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", nargs="?", default="out_odin0")
    ap.add_argument("--span", type=float, default=14.0, help="[m] rolling window width")
    ap.add_argument("--cell", type=float, default=0.2)
    ap.add_argument("--n-theta", type=int, default=12)
    ap.add_argument("--k-sigma", type=float, default=2.0)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=100000)
    ap.add_argument("--outdir", default="/tmp/pipeline_bag")
    ap.add_argument(
        "--despike",
        type=int,
        default=0,
        help="median filter half-width. NOT a fix -- it deletes real obstacles; see despike()",
    )
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    shots = replay(
        a.bag,
        a.span,
        a.cell,
        a.n_theta,
        a.k_sigma,
        a.shots,
        a.max_frames,
        a.outdir,
        a.device,
        a.despike,
    )
    render(shots, a.outdir)


if __name__ == "__main__":
    main()
