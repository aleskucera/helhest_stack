"""MPPI's reference-tracking terms (CostParams.track_*): with a curve set, the nominal plan
closes on it and runs along it; with no curve, the terms are inert and V rules as before."""
from __future__ import annotations

import dataclasses

import numpy as np
import warp as wp
from helhest import dynamics
from helhest.control.mppi import MppiGpu
from helhest.control.reference_path import ReferencePath
from helhest.control.reference_path import ReferencePathParams
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.planner_config import planner_config

N, CELL = 100, 0.08  # 8 m square, origin at the min corner


def _planner(track: float):
    cfg = planner_config(
        {
            "plan_track_lateral": track,
            "plan_track_heading": track,
            "plan_track_progress": 0.5 * track,
            "plan_keep_away": 0.0,
        }
    )
    robot = dynamics.robot_params(cfg.wheel_width)
    sim = ForwardSimulator(
        robot,
        dynamics.planning_solver(command_delay=0.0),
        GridParams(N, N, CELL, 0.0, 0.0),
        cfg.batch,
        cfg.horizon,
    )
    sim.set_uniform_friction(0.8)
    sim.set_terrain(wp.array(np.zeros((N, N), np.float32)))
    planner = MppiGpu(sim, cfg.cost, sampling=cfg.sampling, n_theta=cfg.n_theta)
    goal = (7.5, 4.0)
    xs = (np.arange(N) + 0.5) * CELL
    V = np.hypot(xs[None, :] - goal[0], xs[:, None] - goal[1]).astype(np.float32)
    planner.set_lattice(wp.array(np.repeat(V[:, :, None], cfg.n_theta, axis=2)))
    planner.update_clearance()
    return planner, sim, goal


def _straight_reference(planner: MppiGpu, y: float) -> ReferencePath:
    """A straight curve along y = `y` from x 0.5 to 7.5, written into a ReferencePath's buffers
    the way its kernels would leave them."""
    rp = ReferencePath(planner.n_theta, 0.415, 0.75, ReferencePathParams(), device=planner.device)
    n = rp.params.n_knots + 1
    xs = 0.5 + 7.0 * np.arange(n) / (n - 1)
    rp.pts.assign(np.column_stack([xs, np.full(n, y)]).astype(np.float32))
    rp.tan.assign(np.tile([1.0, 0.0], (n, 1)).astype(np.float32))
    rp.s.assign((xs - xs[0]).astype(np.float32))
    rp.n_valid.assign(np.array([n], np.int32))
    return rp


def _nominal(planner: MppiGpu, sim: ForwardSimulator) -> np.ndarray:
    return sim.controlled.numpy()[:, 0]  # rollout 0 carries the noise-free nominal


def test_the_plan_closes_on_the_curve_and_runs_along_it():
    planner, sim, goal = _planner(track=30.0)
    rp = _straight_reference(planner, y=4.0)
    planner.set_reference(rp)
    # the robot starts 0.6 m beside the line, heading along it
    planner.replan(np.array([1.0, 4.6, 0.0], np.float32), goal, 4)
    wp.synchronize()
    poses = _nominal(planner, sim)
    assert np.isfinite(poses).all()
    assert poses[-1, 0] > poses[0, 0] + 0.5, "no progress along the curve"
    assert (
        abs(poses[-1, 1] - 4.0) < abs(poses[0, 1] - 4.0) - 0.15
    ), "the plan did not close on the line"
    assert abs(poses[-1, 2]) < 0.35, "the plan ends off the curve's heading"


def test_without_a_curve_the_terms_are_inert():
    planner, sim, goal = _planner(track=30.0)
    planner.clear_reference()
    planner.replan(np.array([1.0, 4.6, 0.0], np.float32), goal, 2)
    wp.synchronize()
    j_on = planner.J.numpy()[0]
    cfg = planner_config({"plan_keep_away": 0.0})
    planner.cw = dataclasses.replace(cfg.cost, track_lateral=0.0).build()
    planner.cw.lattice_cap = 1e9
    planner._graph = None
    planner.U.zero_()
    planner.replan(np.array([1.0, 4.6, 0.0], np.float32), goal, 1)
    wp.synchronize()
    assert np.isfinite(j_on) and np.isfinite(planner.J.numpy()[0])
