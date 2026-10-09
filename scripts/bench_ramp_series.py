"""Lattice-planner steepness benchmark: solve every ramp map WITH and WITHOUT the `blocked` field.

    python scripts/make_ramp_series.py          # produce the maps first
    python scripts/bench_ramp_series.py
    python scripts/bench_ramp_series.py --plot /tmp/ramp_bench.png --paths /tmp/ramp_paths

Each map is flat ground crossed by one symmetric full-width ridge whose face angle is the only
variable (see helhest.planning.rampmaps). Start and goal sit on opposite X edges, so the ridge
cannot be driven around: V at the start is a clean feasibility verdict on "can this robot cross a
ramp this steep".

The ABLATION. CostToGo feeds the lattice two per-pose fields: `blocked` (the settle's hard verdict
-- envelope OR residual OR clearance) and `graded_tilt` (a soft preference for flat ground). The
"off" arm replaces `blocked` with zeros and re-runs the SAME value iteration on the SAME
`graded_tilt`, so the only difference is the hard verdict. That isolates what the settle buys, and
graded_tilt cannot stand in for it: tilt is a COST, not a constraint, so every face is crossable
for a price -- and the price is BOUNDED, at arc_len * (1 + flatness_weight * tilt) over the few
cells the face occupies. Worse, that price SHRINKS with steepness, because a steeper ramp is a
shorter one: fewer tilted cells to pay for. The "off" arm therefore does not merely return a longer
path, it returns a confidently cheap one through terrain the robot cannot drive, and rates the
steepest wall in the series as the cheapest crossing of the lot. The audit columns measure that
directly: of the poses the "off" path drove through, how many does the settle call infeasible, and
how far past the envelope do they go.

Note which settle term does the blocking here. On a RAMP the wheels always find a rest pose, so
residual stays ~1e-4 and clearance stays above clear_margin right up to 75 deg; the envelope (tilt
past the robot's own climb/descend/roll limits) is the operative term over essentially the whole
sweep. Residual and clearance only begin to fire at the steep end, and only on poses the envelope
already blocks -- so this series does NOT exercise them, and a family that does needs a vertical
STEP rather than a ramp. The per-term census below reports this rather than assuming it.

A useful built-in check: where the settle blocks nothing at all (5-15 deg) the two arms are
literally the same computation, and their V and path must agree to the last digit.

Solving is device-resident throughout (one CostToGo, one captured graph, replayed per map -- the
series shares a grid by construction). The readbacks here are host-side ANALYSIS of a finished
solve, one per map, not a per-frame round trip.

CUDA-only (graph capture); skips cleanly without a GPU.
"""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import warp as wp
from helhest import dynamics
from helhest.engine import GridParams
from helhest.planning.costtogo import CostToGo

N_THETA = 24


def load_series(map_dir: pathlib.Path) -> list[dict]:
    """Every map in the directory, in sweep order (the filenames sort that way by construction)."""
    paths = sorted(map_dir.glob("bump_a*.npz"))
    if not paths:
        raise SystemExit(f"no bump_a*.npz in {map_dir} -- run scripts/make_ramp_series.py first")
    maps = []
    for path in paths:
        with np.load(path) as data:
            maps.append({k: data[k] for k in data.files} | {"name": path.name})
    return maps


def lattice_state(x: float, y: float, yaw: float, x0: float, y0: float, cell: float) -> tuple:
    """World pose -> lattice (row, col, heading bin), matching _goal_cell_kernel's floor mapping."""
    return (
        int((y - y0) / cell),
        int((x - x0) / cell),
        int(math.floor((yaw % (2.0 * math.pi)) / (2.0 * math.pi / N_THETA))) % N_THETA,
    )


def trace_states(
    ctg: CostToGo,
    V: np.ndarray,
    blocked: np.ndarray,
    tilt: np.ndarray,
    start_rct: tuple[int, int, int],
    max_steps: int = 800,
) -> tuple[list[tuple[int, int, int]], float, float, bool]:
    """Follow the lattice's own policy from the start pose -> (states, billed m, walked m, reached).

    Two lengths, because they disagree. `billed` is what the solver charged (the primitive's
    nominal arc length); `walked` is the euclidean length of the cell-center polyline the lattice
    actually realizes. _build_primitives ROUNDS each arc's endpoint to whole cells, so at cell 0.1
    with step 0.3 a turning primitive lands at (dr, dc) = (-2, 3) -- a 0.36 m move billed as 0.30 m.
    That gap is a property of the lattice, present in both arms, and nothing to do with `blocked`.

    Same selection rule as lattice_solver.trace_optimal (min over feasible arcs of arc_cost +
    V[successor]), but it yields the lattice STATES rather than a polyline: feasibility here is
    heading-dependent, so auditing which poses the "off" arm drove through needs the heading bin,
    which a polyline has already thrown away. V/blocked are passed in rather than read off `ctg`
    because the ablation arm's V is not the one sitting in ctg.V.
    """
    s = ctg.solver
    pdr, pdc = s._prim_dr.numpy(), s._prim_dc.numpy()
    pheading, pcost = s._prim_heading.numpy(), s._prim_cost.numpy()
    sdr, sdc, sn = s._sweep_dr.numpy(), s._sweep_dc.numpy(), s._sweep_n.numpy()
    tilt_weight = float(ctg.flatness_weight)
    gr, gc = (int(v) for v in ctg._goal_rc.numpy())
    ny, nx, _ = V.shape

    cell = ctg.grid.cell_size
    r, c, t = start_rct
    states = [(r, c, t)]
    arc_len, geom_len = 0.0, 0.0
    for _ in range(max_steps):
        if abs(r - gr) <= 1 and abs(c - gc) <= 1:
            return states, arc_len, geom_len, True
        best_p, best_val, best_step = -1, np.inf, 0.0
        for p in range(s.n_prim):
            ns, ok, tsum = int(sn[t, p]), True, 0.0
            for si in range(ns):
                sr, sc = r + int(sdr[t, p, si]), c + int(sdc[t, p, si])
                if not (0 <= sr < ny and 0 <= sc < nx) or blocked[sr, sc, t] > 0.5:
                    ok = False
                    break
                tsum += tilt[sr, sc, t]
            if not ok:
                continue
            nr, nc, nt = r + int(pdr[t, p]), c + int(pdc[t, p]), int(pheading[t, p])
            if not (0 <= nr < ny and 0 <= nc < nx):
                continue
            arc = float(pcost[t, p]) * (1.0 + tilt_weight * tsum / ns if ns > 0 else 1.0)
            val = arc + V[nr, nc, nt]
            if val < best_val:
                best_val, best_p, best_step = val, p, float(pcost[t, p])
        if best_p < 0 or best_val >= ctg._vcap * 0.9:
            # policy dead-ends: no feasible arc, or every successor is +inf
            return states, arc_len, geom_len, False
        pr, pc = r, c
        r, c, t = r + int(pdr[t, best_p]), c + int(pdc[t, best_p]), int(pheading[t, best_p])
        states.append((r, c, t))
        arc_len += best_step  # the arc's own length, not its tilt-weighted cost
        geom_len += cell * math.hypot(r - pr, c - pc)
    return states, arc_len, geom_len, False


def audit_path(states: list, nx: int, settle: dict, blocked: np.ndarray, cell: float) -> dict:
    """What the SETTLE says about the poses a path drove through -- the optimism measure.

    `blocked` here is always the TRUE (settle-derived) field, whichever arm produced the path, so
    the "off" arm's path is graded against the constraint it was not given.
    """
    # pose b = (r*nx + c)*n_theta + t, the C-order flatten _feasibility_kernel uses
    b = np.array([(r * nx + c) * N_THETA + t for r, c, t in states], np.int64)
    rct = np.array(states, np.int64)
    bad = blocked[rct[:, 0], rct[:, 1], rct[:, 2]] > 0.5
    pitch, roll = settle["pitch"][b], settle["roll"][b]
    return {
        # the per-pose arrays the path figures draw; the tables read only the scalars below
        "states": rct,
        "pose_bad": bad,
        "pose_z": settle["z"][b],  # the settled BODY height, so a figure can sit it on the terrain
        "pose_pitch_deg": np.degrees(pitch),
        "n_poses": len(states),
        "n_infeasible": int(bad.sum()),
        # lateral excursion off the start row: the ridge spans the full width, so the ONLY reason
        # to leave the straight line is the heading quantization, not the terrain
        "y_drift_m": float(np.abs(rct[:, 0] - rct[0, 0]).max() * cell),
        # climbing is nose-UP = NEGATIVE pitch, so the two limits are reported separately
        "max_climb_deg": float(np.degrees(np.maximum(-pitch, 0.0).max())),
        "max_descend_deg": float(np.degrees(np.maximum(pitch, 0.0).max())),
        "max_roll_deg": float(np.degrees(np.abs(roll).max())),
        # the term graded_tilt cannot express: a settle that never converged
        "max_residual": float(settle["residual"][b].max()),
        "min_clearance": float(settle["clearance"][b].min()),
    }


def term_census(settle: dict, robot) -> dict:
    """Fraction of the lattice each settle term rejects, so the table shows WHICH one blocks.

    These overlap (a pose can trip several), so they do not sum to blocked_frac -- the point is
    which terms are live on this terrain, not a partition.
    """
    pitch, roll = settle["pitch"], settle["roll"]
    envelope = (
        (np.abs(roll) > robot.max_roll)
        | (pitch < -robot.max_pitch_up)  # climbing is nose-UP = NEGATIVE pitch
        | (pitch > robot.max_pitch_down)
    )
    return {
        "env_frac": float(envelope.mean()),
        "resid_frac": float((settle["residual"] > robot.resid_tol).mean()),
        "clear_frac": float((settle["clearance"] < robot.clear_margin).mean()),
    }


def run_map(ctg: CostToGo, m: dict, robot) -> dict:
    """Solve one map both ways and audit both paths. Returns one row of the results table."""
    cell, (x0, y0) = float(m["cell"]), (float(m["origin"][0]), float(m["origin"][1]))
    heights = np.ascontiguousarray(m["heights"], np.float32)
    elev = wp.array(heights, dtype=wp.float32, device=ctg.device)
    goal = (float(m["goal"][0]), float(m["goal"][1]))
    start_rct = lattice_state(
        float(m["start"][0]), float(m["start"][1]), float(m["start"][2]), x0, y0, cell
    )

    # --- arm ON: the shipped pipeline, settle feasibility + value iteration in one captured graph
    v_on = ctg.compute(elev, goal).numpy().copy()
    blocked = ctg.blocked.numpy()
    tilt = ctg.graded_tilt.numpy()
    der = ctg.settle_sim.derived.numpy()[0]  # (z, pitch, roll) per pose, the static settle
    settle = {
        "z": der[:, 0],
        "pitch": der[:, 1],
        "roll": der[:, 2],
        "residual": ctg.settle_sim.residual.numpy()[0],
        "clearance": ctg.settle_sim.clearance.numpy()[0],
    }

    # --- arm OFF: same graded_tilt, same goal, blocked replaced by zeros. Re-recorded eagerly
    # rather than captured -- it runs 15 times total, and a second graph would need a second solver.
    free = wp.zeros_like(ctg.blocked)
    raw = ctg.solver._record_solve(free, ctg.graded_tilt, ctg._goal_rc, ctg.flatness_weight, False)
    v_off = np.minimum(raw.numpy(), ctg._vcap)  # the clamp compute() applies to its own result

    nx = v_on.shape[1]
    row = {
        "up_deg": float(m["up_deg"]),
        "blocked_frac": float(blocked.mean()),
        "vcap": float(ctg._vcap),
        # the feasibility mask at the heading the crossing is driven at -- blocked is
        # heading-DEPENDENT, so a 2D overlay has to name which heading it is showing
        "blocked_slice": blocked[:, :, start_rct[2]].copy(),
        "start_rct": start_rct,
        **term_census(settle, robot),
    }
    for arm, V, feas in (("on", v_on, blocked), ("off", v_off, np.zeros_like(blocked))):
        states, arc_len, geom_len, reached = trace_states(ctg, V, feas, tilt, start_rct)
        row[arm] = {
            "v_start": float(V[start_rct]),
            "reachable": bool(V[start_rct] < ctg._vcap * 0.9),
            "path_m": arc_len,
            "geom_m": geom_len,
            "reached_goal": reached,
            **audit_path(states, nx, settle, blocked, cell),
        }
    return row


def print_feasibility(rows: list[dict]) -> None:
    """What the settle says about each map's terrain, and which of its three terms said it."""
    print(f"\n{'deg':>5} {'blocked%':>9} {'envelope%':>10} {'residual%':>10} {'clearance%':>11}")
    print("-" * 48)
    for r in rows:
        print(
            f"{r['up_deg']:5.0f} {100 * r['blocked_frac']:9.2f} {100 * r['env_frac']:10.2f} "
            f"{100 * r['resid_frac']:10.2f} {100 * r['clear_frac']:11.2f}"
        )
    print(
        "blocked = the OR of the three terms, so they overlap and do not sum. blocked% FALLS with\n"
        "steepness because a steeper ramp is a shorter one -- fewer cells of terrain to reject, not\n"
        "safer terrain."
    )


def print_ablation(rows: list[dict]) -> None:
    print(
        f"\n{'deg':>5} | {'V*_on':>7} {'goal':>5} | {'V*_off':>7} {'goal':>5} {'billed':>7} "
        f"{'walked':>7} {'drift':>6} {'bad':>5} {'climb':>6} {'desc':>6} {'roll':>6} "
        f"{'clear':>6} {'resid':>8}"
    )
    print("-" * 103)
    for r in rows:
        on, off = r["on"], r["off"]
        v_on = "UNREACH" if not on["reachable"] else f"{on['v_start']:7.2f}"
        v_off = "UNREACH" if not off["reachable"] else f"{off['v_start']:7.2f}"
        print(
            f"{r['up_deg']:5.0f} | {v_on:>7} {'yes' if on['reached_goal'] else 'no':>5} "
            f"| {v_off:>7} {'yes' if off['reached_goal'] else 'no':>5} {off['path_m']:7.2f} "
            f"{off['geom_m']:7.2f} {off['y_drift_m']:6.2f} {off['n_infeasible']:5d} "
            f"{off['max_climb_deg']:6.1f} {off['max_descend_deg']:6.1f} "
            f"{off['max_roll_deg']:6.1f} {off['min_clearance']:6.3f} {off['max_residual']:8.5f}"
        )
    print(
        "V* = cost-to-go at the start pose. Everything right of it describes the OFF arm's path,\n"
        "graded against the settle it was not given: billed = length the solver charged, walked =\n"
        "euclidean length of the realized polyline, drift = furthest it strayed off the start row\n"
        "[m], bad = poses the settle calls infeasible, climb/desc/roll = the extreme body angles it\n"
        "reached [deg], clear/resid = worst clearance [m] and settle residual along the way."
    )


def summarize(rows: list[dict], robot) -> None:
    """The headline: where each arm stops, and what the gap between them costs."""
    ok_on = [r["up_deg"] for r in rows if r["on"]["reached_goal"]]
    ok_off = [r["up_deg"] for r in rows if r["off"]["reached_goal"]]
    optimistic = [r for r in rows if r["off"]["reached_goal"] and not r["on"]["reached_goal"]]
    agree = [r for r in rows if r["blocked_frac"] == 0.0]
    print(
        f"\nenvelope: climb {math.degrees(robot.max_pitch_up):.0f} deg, "
        f"descend {math.degrees(robot.max_pitch_down):.0f} deg, "
        f"roll {math.degrees(robot.max_roll):.0f} deg\n"
        f"  blocked ON : crosses up to {max(ok_on) if ok_on else float('nan'):.0f} deg "
        f"({len(ok_on)}/{len(rows)} maps)\n"
        f"  blocked OFF: crosses up to {max(ok_off) if ok_off else float('nan'):.0f} deg "
        f"({len(ok_off)}/{len(rows)} maps)"
    )
    # The ridge spans the full width, so the shortest crossing is the straight one and any drift
    # is the planner's own. This used to be large: _build_primitives charged every primitive the
    # same nominal `step` while rounding its endpoint to whole cells, so a turning step covered
    # 0.36 m for a 0.30 m fee and weaving was cheaper per metre than driving straight. Primitives
    # are now charged the displacement they realize, which should hold drift at zero here and keep
    # billed == walked. It is not part of the ablation -- both arms price arcs identically -- so
    # what it is really watching for is a regression sneaking in via the benchmark.
    drift = max(r["off"]["y_drift_m"] for r in rows)
    over = max(abs(r["off"]["geom_m"] - r["off"]["path_m"]) for r in rows)
    if drift > 0.5 * rows[0]["off"]["path_m"] / max(len(rows), 1) or over > 0.05:
        print(
            f"  WARNING -- arc pricing looks inconsistent: paths drift up to {drift:.2f} m off the\n"
            f"  straight line and walk up to {over:.2f} m further than billed. See\n"
            f"  scripts/flat_map_detour.py, which isolates this on featureless terrain."
        )
    else:
        print(
            f"  arc pricing consistent: billed == walked to {over:.3f} m, drift {drift:.2f} m "
            f"(straight, as it should be)."
        )
    if agree:  # where the settle blocks nothing the two arms are the same computation
        same = all(r["on"]["v_start"] == r["off"]["v_start"] for r in agree)
        print(
            f"  control: {len(agree)} maps block nothing "
            f"({agree[0]['up_deg']:.0f}-{agree[-1]['up_deg']:.0f} deg); "
            f"the arms agree exactly there: {same}"
        )
    if not optimistic:
        print("  no map where the arms disagree.")
        return
    worst = max(optimistic, key=lambda r: r["off"]["max_descend_deg"])
    cheapest = min(rows, key=lambda r: r["off"]["v_start"])
    print(
        f"  OFF returns a path the robot cannot drive on {len(optimistic)} maps "
        f"({optimistic[0]['up_deg']:.0f}-{optimistic[-1]['up_deg']:.0f} deg).\n"
        f"  worst at {worst['up_deg']:.0f} deg: {worst['off']['n_infeasible']}/"
        f"{worst['off']['n_poses']} poses infeasible, peak descend "
        f"{worst['off']['max_descend_deg']:.1f} deg vs a "
        f"{math.degrees(robot.max_pitch_down):.0f} deg limit.\n"
        f"  and the cost signal points the WRONG way: OFF rates {cheapest['up_deg']:.0f} deg the\n"
        f"  cheapest crossing of the series (V* {cheapest['off']['v_start']:.2f}, against "
        f"{rows[0]['off']['v_start']:.2f} for the {rows[0]['up_deg']:.0f} deg ramp) -- a steeper "
        f"ramp is a\n  shorter one, so graded_tilt bills fewer tilted cells for it."
    )


def _path_xy(states: np.ndarray, x0: float, y0: float, cell: float) -> tuple:
    """Lattice states -> world cell centers, the convention trace_optimal draws paths on."""
    return x0 + (states[:, 1] + 0.5) * cell, y0 + (states[:, 0] + 0.5) * cell


def plot_paths(rows: list[dict], maps: list[dict], out_dir: pathlib.Path, robot) -> None:
    """One figure per map: where each arm actually drove, and what attitude it drove at.

    The lower panel is the point of the exercise -- it draws the robot's own settled body at every
    pose the OFF arm passed through, at the pitch and height the settle put it at, so a pose 55 deg
    nose-down on a wall face is visible as a body standing on a wall rather than as a table cell.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    out_dir.mkdir(parents=True, exist_ok=True)
    body = robot.rear_offset  # [m] wheelbase: the length of the drawn body segment
    for row, m in zip(rows, maps):
        cell = float(m["cell"])
        x0, y0 = float(m["origin"][0]), float(m["origin"][1])
        heights = m["heights"]
        ny, nx = heights.shape
        extent = [x0, x0 + nx * cell, y0, y0 + ny * cell]
        xs = x0 + (np.arange(nx) + 0.5) * cell
        run = float(m["height"]) / np.tan(np.radians(float(m["up_deg"])))
        half_window = run + 0.5 * float(m["plateau"]) + 1.5

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 2])

        top = axes[0]
        top.imshow(
            heights,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="terrain",
            vmin=-float(m["height"]) / 3.0,
            vmax=float(m["height"]),
        )
        # the infeasible set at THIS heading, as a red wash over the terrain it rejects
        top.imshow(
            np.ma.masked_less(row["blocked_slice"], 0.5),
            origin="lower",
            extent=extent,
            aspect="equal",
            alpha=0.45,
            cmap=ListedColormap(["red"]),
            vmin=0.0,
            vmax=1.0,
        )
        for arm, color, style in (("off", "tab:orange", "-"), ("on", "tab:blue", "--")):
            px, py = _path_xy(row[arm]["states"], x0, y0, cell)
            label = f"blocked {arm}: {row[arm]['geom_m']:.1f} m walked"
            if not row[arm]["reached_goal"]:
                label += " (dead end)"
            top.plot(px, py, style, color=color, lw=2.5, label=label)
            bad = row[arm]["pose_bad"]
            if bad.any():  # the poses the settle rejects, wherever this arm drove through them
                top.plot(px[bad], py[bad], "x", color="red", ms=9, mew=2, ls="none")
        top.plot(m["start"][0], m["start"][1], "o", color="lime", ms=9, mec="black")
        top.plot(m["goal"][0], m["goal"][1], "X", color="red", ms=11, mec="black")
        top.set_ylabel("y [m]")
        top.legend(loc="upper left", fontsize=8)
        top.set_title(
            "red wash = poses the settle blocks at this heading; "
            "x = infeasible poses the path drove through"
        )

        prof = axes[1]
        prof.plot(xs, heights[0], color="0.4", lw=1.5)
        prof.fill_between(xs, -0.4, heights[0], color="0.85")
        st = row["off"]
        px, _ = _path_xy(st["states"], x0, y0, cell)
        # nose-DOWN is positive pitch, so the nose end drops; the crossing runs along +x by
        # construction here, which is what lets a single x-z section stand in for the body
        dx = 0.5 * body * np.cos(np.radians(st["pose_pitch_deg"]))
        dz = 0.5 * body * np.sin(np.radians(st["pose_pitch_deg"]))
        for i in range(len(px)):
            if abs(px[i]) > half_window:
                continue  # outside the zoom window; the flat approach carries no information
            color = "red" if st["pose_bad"][i] else "tab:green"
            prof.plot(
                [px[i] - dx[i], px[i] + dx[i]],
                [st["pose_z"][i] + dz[i], st["pose_z"][i] - dz[i]],
                "-",
                color=color,
                lw=2.0,
                alpha=0.9,
                solid_capstyle="round",
            )
        prof.set_xlim(-half_window, half_window)
        prof.set_aspect("equal")  # 1:1, so the drawn body attitude is the real one
        prof.set_xlabel("x [m]")
        prof.set_ylabel("z [m]")
        prof.grid(alpha=0.3)
        prof.set_title(
            f"the OFF arm's body at every pose (green = settle-feasible, red = not), "
            f"1:1 aspect -- peak {st['max_climb_deg']:.0f} deg up / "
            f"{st['max_descend_deg']:.0f} deg down against a "
            f"{math.degrees(robot.max_pitch_up):.0f}/{math.degrees(robot.max_pitch_down):.0f} "
            f"deg envelope"
        )

        v_on = "UNREACHABLE" if not row["on"]["reachable"] else f"{row['on']['v_start']:.2f}"
        fig.suptitle(
            f"{m['name']}   {row['up_deg']:.0f} deg face, {float(m['height']):.2f} m high   "
            f"blocked {100 * row['blocked_frac']:.1f}% of poses   "
            f"V* on {v_on} / off {row['off']['v_start']:.2f}   "
            f"OFF drove {st['n_infeasible']}/{st['n_poses']} poses infeasible"
        )
        out = out_dir / f"path_a{round(row['up_deg'] * 10):04d}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {len(rows)} path figures to {out_dir}")


def plot(rows: list[dict], out: pathlib.Path, robot) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deg = [r["up_deg"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax = axes[0]
    for arm, color in (("on", "tab:blue"), ("off", "tab:orange")):
        # unreachable poses sit exactly at the cap; mark them rather than plotting a flat line
        vals = [r[arm]["v_start"] for r in rows]
        ok = [r[arm]["reachable"] for r in rows]
        ax.plot(deg, vals, "-o", ms=4, color=color, label=f"blocked {arm}")
        ax.plot(
            [d for d, k in zip(deg, ok) if not k],
            [v for v, k in zip(vals, ok) if not k],
            "x",
            ms=10,
            mew=2,
            color=color,
            ls="none",
            label=f"{arm}: unreachable",
        )
    ax.axhline(rows[0]["vcap"], color="k", ls=":", lw=1, label="V cap")
    ax.set_ylabel("V at start [m-equiv]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("cost-to-go at the start pose (OFF never rises: every angle costs it ~20 m)")
    # the OFF curve's own variation (20.75 -> 20.32, i.e. steeper reading as CHEAPER) is a 2%
    # effect next to the cap, so it needs its own axes or it is just a flat line
    inset = ax.inset_axes((0.08, 0.45, 0.34, 0.45))
    inset.plot(deg, [r["off"]["v_start"] for r in rows], "-o", ms=3, color="tab:orange")
    inset.set_title("OFF, zoomed", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(alpha=0.3)

    ax = axes[1]
    # blocked is the OR, so it sits exactly under whichever term is live -- draw it as a fat
    # underlay so the term on top stays visible and the coincidence is the readable result
    ax.plot(
        deg,
        [100 * r["blocked_frac"] for r in rows],
        "-",
        lw=5,
        alpha=0.3,
        color="tab:red",
        label="blocked (the OR)",
    )
    for key, label, color in (
        ("env_frac", "envelope", "tab:purple"),
        ("resid_frac", "residual", "tab:green"),
        ("clear_frac", "clearance", "tab:brown"),
    ):
        ax.plot(deg, [100 * r[key] for r in rows], "-o", ms=4, color=color, label=label)
    ax.set_ylabel("poses rejected [%]")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
    ax.set_title("which settle term rejects the terrain (overlapping, so they do not sum)")

    ax = axes[2]
    ax.plot(
        deg,
        [r["off"]["n_infeasible"] for r in rows],
        "-o",
        ms=4,
        color="tab:orange",
        label="poses on the OFF path the settle blocks",
    )
    ax.plot(
        deg,
        [r["on"]["n_infeasible"] for r in rows],
        "-o",
        ms=4,
        color="tab:blue",
        label="same, ON path (must stay 0)",
    )
    ax.set_ylabel("infeasible poses on path")
    ax.set_xlabel("ramp face angle [deg]")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("optimism: infeasible poses in the returned path")

    twin = ax.twinx()
    twin.plot(deg, [r["off"]["max_descend_deg"] for r in rows], "--", color="gray", lw=1)
    twin.axhline(math.degrees(robot.max_pitch_down), color="gray", ls=":", lw=1)
    twin.set_ylabel("peak descend on OFF path [deg] (dashed)", color="gray")

    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dir", type=pathlib.Path, default=pathlib.Path("/tmp/ramp_series"))
    ap.add_argument("--plot", type=pathlib.Path, default=None, help="also write a summary PNG")
    ap.add_argument(
        "--paths",
        type=pathlib.Path,
        default=None,
        help="write one per-map figure of the paths each arm drove, into this directory",
    )
    args = ap.parse_args()

    wp.init()
    if not wp.is_cuda_available():
        print("CUDA not available -- the cost-to-go solve is GPU-only (graph capture). Skipping.")
        return

    maps = load_series(args.dir)
    m0 = maps[0]
    ny, nx = m0["heights"].shape
    cell = float(m0["cell"])
    grid = GridParams(nx, ny, cell, float(m0["origin"][0]), float(m0["origin"][1]))
    for m in maps:  # the series is built to share one grid; a mismatch would invalidate the compare
        assert m["heights"].shape == (ny, nx) and float(m["cell"]) == cell, f"{m['name']} off-grid"

    robot_params = dynamics.robot_params()
    ctg = CostToGo(grid, robot_params, dynamics.planning_solver(), n_theta=N_THETA, device="cuda")
    print(
        f"=== ramp-series ablation: {len(maps)} maps, {ny}x{nx} @ {cell} m x {N_THETA} headings "
        f"= {ny * nx * N_THETA} poses each ==="
    )

    rows = [run_map(ctg, m, robot_params) for m in maps]
    print_feasibility(rows)
    print_ablation(rows)
    summarize(rows, robot_params)
    if args.plot is not None:
        plot(rows, args.plot, robot_params)
    if args.paths is not None:
        plot_paths(rows, maps, args.paths, robot_params)


if __name__ == "__main__":
    main()
