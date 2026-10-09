"""Timing benchmark for the control/ stack: one MPPI frame, as the robot runs it.

Per frame the node puts the MPPI window's terrain into the rollout simulator (`set_terrain`, which
dilates one wheel envelope per yaw bin for the cylinder wheel), hands MPPI the routing field and
the wall veto, rebuilds the clearance map, and runs `replan` -- sample, roll out, cost, reweight,
`plan_n_refine` times, CUDA-graph-captured. Everything is the robot's configuration from
`ros/config/odin.params.yaml` through `helhest.planner_config` (see `_common`): batch, horizon,
friction replicas, cost weights, sampler, cylinder wheel, window sizes.

The first row of each sweep is the robot's point. Compare with the node's `plan:replan` /
`plan:clear_map` stages (`profile_stages`, over a bag replay). ctrl_RTF = dt / replan: how much
faster than real time the controller plans. CUDA-only; skips without a GPU.

Run from the repo root:  python -m benchmarks.control [--world slalom]
"""

from __future__ import annotations

import argparse

import warp as wp
from helhest import dynamics
from helhest import worlds as W
from helhest.control.mppi import MppiGpu

from ._common import build_costtogo
from ._common import build_rollout_sim
from ._common import robot_scene
from ._common import route_inputs
from ._common import RobotScene
from ._common import time_fn

DT = dynamics.DT  # control timestep [s]


def _planner(rs: RobotScene, batch: int, device: str) -> tuple[MppiGpu, object]:
    """The node's planner, armed with a solved routing field; returns (planner, rollout sim)."""
    sim = build_rollout_sim(rs, device, batch, rs.cfg.horizon)
    planner = MppiGpu(sim, rs.cfg.cost, sampling=rs.cfg.sampling, n_theta=rs.cfg.n_theta)
    planner.reset_nominal(rs.cfg.nominal_reset)
    planner.set_mu_band(1.0, rs.cfg.mu_span)
    ctg = build_costtogo(rs, device)
    ctg.compute(goal_xy=rs.goal_route, **route_inputs(rs, device))
    planner.cw.lattice_cap = ctg._vcap
    sgrid = rs.sgrid.build()
    planner.set_lattice(ctg.V_escape, sgrid)
    if planner.cw.veto > 0.0:
        planner.set_veto(ctg.hazard, sgrid)
    if planner.cw.clear_time > 0.0:
        planner.update_clearance()
    return planner, sim


def _row(label: str, batch: int, n_refine: int, t: float, robot: bool) -> None:
    mark = "  <- robot" if robot else ""
    print(
        f"    {label:>10} {batch:>6} {n_refine:>6} {t * 1e3:>10.2f} {1.0 / t:>7.0f} "
        f"{DT / t:>8.1f}x{mark}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="slalom", choices=list(W.WORLDS))
    args = ap.parse_args()

    wp.init()
    if not wp.is_cuda_available():
        print("CUDA not available -- MPPI is GPU-only (CUDA graph capture). Skipping.")
        return
    device = "cuda"
    rs = robot_scene(args.world)
    reps = 10
    batch0, refine0, n_mu = rs.cfg.batch, int(rs.params["plan_n_refine"]), rs.cfg.sampling.n_mu
    g = rs.win_grid
    print(
        f"\n=== MPPI frame  world={args.world}  window {g.cells_y}x{g.cells_x} at "
        f"{g.cell_size:.2f} m  T={rs.cfg.horizon}  n_theta={rs.cfg.n_theta}  n_mu={n_mu} ==="
    )
    header = f"    {'':>10} {'B':>6} {'n_ref':>6} {'ms':>10} {'Hz':>7} {'ctrl_RTF':>9}"

    planner, sim = _planner(rs, batch0, device)
    terrain = wp.array(rs.win, dtype=wp.float32, device=device)
    print("  per-frame inputs (robot point):")
    print(header)
    _row("terrain", batch0, 0, time_fn(lambda: sim.set_terrain(terrain), reps, device), True)
    if planner.cw.clear_time > 0.0:
        t = time_fn(planner.update_clearance, reps, device)
        _row("clear_map", batch0, 0, t, True)

    print("  replan, n_refine sweep:")
    print(header)
    for n_refine in sorted({refine0, 1, 5}):
        t = time_fn(lambda: planner.replan(rs.state, rs.goal, n_refine), reps, device)
        _row("replan", batch0, n_refine, t, n_refine == refine0)

    print("  replan, batch sweep (multiples of the friction replicas):")
    print(header)
    for batch in sorted({batch0, batch0 // 2 - (batch0 // 2) % n_mu, 2 * batch0}):
        p, _ = (planner, sim) if batch == batch0 else _planner(rs, batch, device)
        t = time_fn(lambda: p.replan(rs.state, rs.goal, refine0), reps, device)
        _row("replan", batch, refine0, t, batch == batch0)


if __name__ == "__main__":
    main()
