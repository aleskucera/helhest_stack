"""MPPI's clearance-time cost (plan_clear_mppi_weight > 0) through a real replan.

Nothing else runs a replan with the cost on. From 2026-09-30 (when the keep-away term joined it in
the cost kernel) until 2026-10-08, every replan with it on read out of bounds and crashed the
process: the kernel computed the footprint clearance twice per step, and the second inlined copy
read the `robot` struct's array handle past its local copy (compute-sanitizer). The cost was off
on the robot, so nothing noticed.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest import dynamics
from helhest.control.mppi import MppiGpu
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.planner_config import planner_config

N, CELL = 100, 0.08  # 8 m square, origin at the min corner


def _replan(mppi_weight: float) -> np.ndarray:
    cfg = planner_config(
        {"plan_clear_t_react": 0.08, "plan_clear_mppi_weight": mppi_weight, "plan_keep_away": 10.0}
    )
    sim = ForwardSimulator(
        dynamics.robot_params(cfg.wheel_width),
        dynamics.planning_solver(command_delay=0.0),
        GridParams(N, N, CELL, 0.0, 0.0),
        cfg.batch,
        cfg.horizon,
    )
    sim.set_uniform_friction(0.8)
    h = np.zeros((N, N), np.float32)
    h[60:, :] = 1.0  # a 1 m wall from y = 4.8 m: the robot drives along it 0.6 m away
    sim.set_terrain(wp.array(h))
    planner = MppiGpu(sim, cfg.cost, sampling=cfg.sampling, n_theta=cfg.n_theta)
    planner.set_lattice(wp.zeros((N, N, cfg.n_theta), dtype=float))
    planner.update_clearance()
    planner.replan(np.array([1.0, 4.2, 0.0], np.float32), (7.0, 4.2), 2)
    wp.synchronize()
    return planner.nominal()


def test_a_replan_with_the_clearance_time_cost_on_runs_beside_a_wall():
    u = _replan(mppi_weight=1.0)
    assert np.isfinite(u).all()


def test_the_cost_off_still_plans():
    assert np.isfinite(_replan(mppi_weight=0.0)).all()
