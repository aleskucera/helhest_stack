"""The probabilistic planning pipeline in one picture, on a scene built to show each stage.

Three repositories, one figure. `elevation_belief` fuses sweeps into a height, a measurement sd
and a pose drift; the settle producer turns those into per-pose margins and the sigmas they are
judged against; `terrain_value_field` value-iterates the result twice, believing the map and then
as if it were certain.

The scene is synthetic on purpose, and staged so the four things the planner must tell apart are
each present and each look different:

  a ROCK        genuinely bad ground. Blocked on any reading, no doubt -- looking will not help.
  a MIST patch  scanned from far away, so its heights are poorly known. Blocked believing the
                map, clear if it turns out fine: pure doubt, and the thing worth driving to.
  a SEAM        the right half re-scanned a minute after the left, so the two halves no longer
                share their pose drift. A footprint straddling it is differencing heights that
                drifted apart, which costs margin even though both halves look well measured.
  a VOID        never scanned at all. Carries no height and no drift -- and must read as absent
                rather than as flat, certain ground.

    python demos/pipeline_panels.py --shot /tmp/panels.png
    python demos/pipeline_panels.py --no-drift --shot /tmp/nodrift.png   # what the seam costs
"""

from __future__ import annotations

import argparse


def build(no_drift: bool, seam_age: float, device: str):
    import numpy as np
    import warp as wp
    from elevation_belief import DriftRates
    from elevation_belief import ElevationBelief
    from elevation_belief import NoiseModel

    from helhest.engine import GridParams
    from helhest.engine import RobotParams
    from helhest.engine import SolverParams
    from helhest.planning.costtogo import CostToGo

    N, CELL = 61, 0.24
    HALF = N * CELL / 2
    rng = np.random.default_rng(7)

    def ground(x, y):
        """Rolling ground, plus a rock the robot genuinely cannot climb."""
        z = 0.12 * np.sin(0.7 * x) + 0.09 * np.cos(0.5 * y)
        rock = (np.abs(x + 2.0) < 0.9) & (np.abs(y - 2.4) < 0.9)
        return z + np.where(rock, 0.75, 0.0)

    MIST = (1.2, 3.6, -3.6, -1.2)  # xlo, xhi, ylo, yhi

    def sweep(n, xlo, xhi, ylo, yhi, noise, skip_mist=False):
        x = rng.uniform(xlo, xhi, n)
        y = rng.uniform(ylo, yhi, n)
        if skip_mist:
            # the good sweep must MISS the mist patch, or fusing it on top averages the
            # uncertainty away and there is nothing left to be uncertain about
            keep = ~((x > MIST[0]) & (x < MIST[1]) & (y > MIST[2]) & (y < MIST[3]))
            x, y = x[keep], y[keep]
        z = ground(x, y) + rng.normal(0.0, noise, len(x))
        return np.c_[x, y, z].astype(np.float64)

    belief = ElevationBelief(
        (-HALF, HALF, -HALF, HALF),
        CELL,
        # a = 0.01 + 0.002/m puts the near-field sd at ~2 cm, which is where
        # studies/calib/fit_drift.py put Odin's measured var_meas (1.75 cm)
        noise=NoiseModel("linear", a=0.01, b=0.002),
        rates=DriftRates.odin_slam(),
    )
    sensor = np.array([0.0, 0.0, 1.2])

    # 1. sweep everything except the VOID (a strip at the top the sensor never reaches)
    belief.measure_scan(sweep(90000, -HALF, HALF, -HALF, HALF * 0.55, 0.008, True), sensor)
    # 2. the MIST patch, and its uncertainty is PHYSICAL rather than stipulated: seen from 45 m
    # away and only a handful of returns, so the noise model's range term inflates it and there
    # is little to average down. This is what the far edge of a real sweep looks like.
    belief.measure_scan(sweep(260, *MIST, 0.05), np.array([-45.0, -45.0, 1.2]))
    # 3. time passes, then only the right half is re-scanned -> a SEAM down the middle
    belief.motion_update(seam_age, (0.0, 0.0))
    belief.measure_scan(sweep(45000, 0.0, HALF, -HALF, HALF * 0.55, 0.008, True), sensor)

    lay = belief.layers()
    seen = lay["valid"].numpy() != 0
    height = np.nan_to_num(lay["raw_h"].numpy(), nan=0.0)
    meas_sd = np.sqrt(np.maximum(lay["meas_var"].numpy(), 0.0))
    drift = belief.drift().numpy()

    a = lambda v: wp.array(np.ascontiguousarray(v, np.float32), dtype=wp.float32)  # noqa: E731
    ctg = CostToGo(
        GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=-HALF, origin_y=-HALF),
        RobotParams(),
        SolverParams(),
        n_theta=12,
        z_veto=2.0,
        charge_per_sigma=0.5,
        device=device,
    )
    goal = (HALF - 2 * CELL, 0.0)
    ctg.solve_gap(
        a(height),
        goal,
        a(meas_sd),
        measured=a(seen.astype(np.float32)),
        drift=None if no_drift else a(drift),
    )
    spread = ctg._spread.numpy()
    return dict(
        N=N,
        CELL=CELL,
        HALF=HALF,
        goal=goal,
        seen=seen,
        height=height,
        meas_sd=meas_sd,
        drift=drift,
        spread=spread,
        ctg=ctg,
        z=ctg.zmargin.numpy(),
        blocked=ctg.blocked.numpy(),
        vp=ctg.V_pessimistic.numpy(),
        vo=ctg.V_optimistic.numpy(),
        doubt=ctg.doubt_pessimistic.numpy(),
        cap=ctg._vcap,
    )


def render(d, out, no_drift):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HALF, cap, seen = d["HALF"], d["cap"], d["seen"]
    ext = (-HALF, HALF, -HALF, HALF)
    mask = lambda v: np.where(seen, v, np.nan)  # noqa: E731

    def best(v):
        """Min over heading. Never-measured goes to NaN (drawn as the grey background);
        unreachable goes BELOW the scale, so the two do not share a colour."""
        m = v.min(2)
        return np.where(seen, np.where(m >= cap * 0.9, -1.0, m), np.nan)

    bp, bo = best(d["vp"]), best(d["vo"])
    unreachable = (bp < 0) | (bo < 0)
    gap = np.where(unreachable, -1.0, bp - bo)

    panels = [
        ("height  [m]", mask(d["height"]), "terrain", None),
        ("measurement sd  [m]", mask(d["meas_sd"]), "magma", None),
        ("pose drift  [m^2]", mask(np.where(d["drift"] < 0, np.nan, d["drift"])), "magma", None),
        ("footprint drift SPREAD  [m^2]", mask(d["spread"]), "magma", None),
        ("z: margin in sigmas", mask(np.clip(d["z"].min(2), 0, 8)), "viridis", None),
        ("blocked  (any heading)", mask(d["blocked"].max(2)), "Reds", (0, 1)),
        ("V believing the map  [m]", best(d["vp"]), "viridis", "route"),
        ("V if the map were certain  [m]", best(d["vo"]), "viridis", "route"),
        ("doubt  (blocked by IGNORANCE)", mask(d["doubt"].max(2)), "cividis", None),
        ("gap = V_believed - V_certain  [m]", gap, "inferno", "route"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(22, 9.2))
    for ax, (title, img, cmap, lim) in zip(axes.ravel(), panels):
        if lim == "route":
            cm = plt.get_cmap(cmap).copy()
            cm.set_under("black")  # no route, as distinct from never measured
            kw = dict(cmap=cm, vmin=0.0)
        elif lim is None:
            kw = dict(cmap=cmap)
        else:
            kw = dict(cmap=cmap, vmin=lim[0], vmax=lim[1])
        ax.set_facecolor("0.82")  # never measured
        im = ax.imshow(img, origin="lower", extent=ext, **kw)
        ax.set_title(title, fontsize=10)
        ax.plot(*d["goal"], "w*", ms=13, mec="k")
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.ravel():
        ax.text(-2.0, 2.4, "rock", color="w", fontsize=7, ha="center")
        ax.text(2.4, -2.4, "mist", color="w", fontsize=7, ha="center")
        ax.text(0.0, -HALF + 0.5, "seam", color="w", fontsize=7, ha="center")
        ax.text(0.0, HALF - 0.8, "void", color="w", fontsize=7, ha="center")
    head = "the pipeline, stage by stage" + ("   [drift NOT passed]" if no_drift else "")
    fig.suptitle(
        f"{head}\nbelief (4 panels) -> settle producer (z, blocked) -> value field (V, doubt, gap)"
        "        grey = never measured     black = no route",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", default="/tmp/pipeline_panels.png")
    ap.add_argument("--seam-age", type=float, default=90.0, help="[s] between the two sweeps")
    ap.add_argument("--no-drift", action="store_true", help="withhold drift: the seam vanishes")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    d = build(args.no_drift, args.seam_age, args.device)

    import numpy as np

    seen = d["seen"]
    print(f"  cells measured      {100*seen.mean():5.1f}%")
    print(f"  blocked (any head)  {100*d['blocked'].max(2)[seen].mean():5.1f}%")
    print(f"  doubted             {100*(d['doubt'].max(2)[seen] > 0).mean():5.1f}%")
    sp = d["spread"][seen]
    print(f"  drift spread        median {np.median(sp):.2e}  max {sp.max():.2e}  [m^2]")
    r, c = d["N"] // 2, 4
    x = -d["HALF"] + c * d["CELL"]
    y = -d["HALF"] + r * d["CELL"]
    print("\n  at the robot's end of the map:")
    for k, v in d["ctg"].gap_at(x, y, 0.0).items():
        print(f"    {k:>26s}: {v}")
    render(d, args.shot, args.no_drift)


if __name__ == "__main__":
    main()
