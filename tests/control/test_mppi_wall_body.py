"""MPPI's body-through-wall term (CostParams.wall_body).

The rollout's hard terms come from the settle, which knows wheels and belly. A wall at the tail's
corner is neither: with only keep_away pricing it, MPPI's nominal plan ran through a belief wall in
17 % of drive_sim's trunks frames and every wall contact began with the plan inside one
(2026-10-27). The case here is the one the settle cannot see: the robot parked along a wall with
its goal behind it, so the cheapest turn sweeps the tail's corner through the wall.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import warp as wp
from helhest import dynamics
from helhest.control.mppi import MppiGpu
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.planner_config import planner_config

N, CELL = 100, 0.08  # 8 m square, origin at the min corner
WALL_Y = 4.8  # [m] the wall's face: a 1 m step from this row on
GAP = 0.2  # [m] between the robot's flank and the face


def _turn_around_beside_a_wall(wall_body: float, keep_away: float) -> tuple[np.ndarray, float]:
    """The nominal rollout's poses and the deepest the body's rectangle reaches past the face."""
    cfg = planner_config({"plan_keep_away": keep_away})
    cfg = dataclasses.replace(cfg, cost=dataclasses.replace(cfg.cost, wall_body=wall_body))
    robot = dynamics.robot_params(cfg.wheel_width)
    sim = ForwardSimulator(
        robot,
        dynamics.planning_solver(command_delay=0.0),
        GridParams(N, N, CELL, 0.0, 0.0),
        cfg.batch,
        cfg.horizon,
    )
    sim.set_uniform_friction(0.8)
    h = np.zeros((N, N), np.float32)
    h[int(round(WALL_Y / CELL)) :, :] = 1.0
    sim.set_terrain(wp.array(h))
    planner = MppiGpu(sim, cfg.cost, sampling=cfg.sampling, n_theta=cfg.n_theta)
    # the goal is straight behind the robot: V falls toward it from every heading alike, so the
    # cheapest plan is to turn around on the spot
    half_w = robot.half_track + 0.5 * (robot.wheel_width or 2.0 * robot.wheel_radius)
    y0 = WALL_Y - GAP - half_w
    goal = (0.5, y0)
    xs = (np.arange(N) + 0.5) * CELL
    V = np.hypot(xs[None, :] - goal[0], xs[:, None] - goal[1]).astype(np.float32)
    planner.set_lattice(wp.array(np.repeat(V[:, :, None], cfg.n_theta, axis=2)))
    planner.update_clearance()
    planner.replan(np.array([4.0, y0, 0.0], np.float32), goal, 2)
    wp.synchronize()
    poses = sim.controlled.numpy()[:, 0]  # rollout 0 is the noise-free nominal
    tail = robot.rear_offset + robot.wheel_radius
    corners = np.array(
        [
            [-tail, -half_w],
            [-tail, half_w],
            [robot.wheel_radius, -half_w],
            [robot.wheel_radius, half_w],
        ]
    )
    ca, sa = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    corner_y = (
        poses[:, 1, None] + sa[:, None] * corners[None, :, 0] + ca[:, None] * corners[None, :, 1]
    )
    return poses, float(corner_y.max() - WALL_Y)


def test_the_body_term_keeps_the_tail_out_of_the_wall_while_turning_around():
    poses, depth = _turn_around_beside_a_wall(wall_body=1.0, keep_away=0.0)
    assert np.isfinite(poses).all()
    # half a cell: the face sits somewhere inside its cell, and the perimeter is sampled
    assert depth <= 0.5 * CELL, f"the body reached {depth:.3f} m past the face"


def test_without_it_the_settle_alone_lets_the_tail_sweep_through():
    """The sensitivity check: the settle has no wheel at the tail's corner, so with the term off
    and keep_away off the same turn is free to go through the wall."""
    _, depth = _turn_around_beside_a_wall(wall_body=0.0, keep_away=0.0)
    assert depth > 0.5 * CELL, f"the turn stayed {-depth:.3f} m clear without the term: not probing"


def test_a_body_already_on_a_face_is_not_charged_for_standing_there():
    """The term is RELATIVE to the start pose: after a wall hug the body often stands inside the
    belief's fringe, and an absolute charge priced every motion the same as standing still, so
    MPPI stood still (false_door, half the run at governor scale 1). Standing with the tail
    0.04 m inside the face must cost the same with the term as without it."""
    costs = []
    for wall_body in (1.0, 0.0):
        cfg = planner_config({"plan_keep_away": 0.0})
        cfg = dataclasses.replace(cfg, cost=dataclasses.replace(cfg.cost, wall_body=wall_body))
        robot = dynamics.robot_params(cfg.wheel_width)
        sim = ForwardSimulator(
            robot,
            dynamics.planning_solver(command_delay=0.0),
            GridParams(N, N, CELL, 0.0, 0.0),
            cfg.batch,
            cfg.horizon,
        )
        sim.set_uniform_friction(0.8)
        h = np.zeros((N, N), np.float32)
        h[:, : int(round(2.0 / CELL))] = 1.0  # the wall BEHIND the robot: a face at x = 2.0
        sim.set_terrain(wp.array(h))
        planner = MppiGpu(sim, cfg.cost, sampling=cfg.sampling, n_theta=cfg.n_theta)
        tail = robot.rear_offset + robot.wheel_radius
        goal = (7.5, 4.0)
        xs = (np.arange(N) + 0.5) * CELL
        V = np.hypot(xs[None, :] - goal[0], xs[:, None] - goal[1]).astype(np.float32)
        planner.set_lattice(wp.array(np.repeat(V[:, :, None], cfg.n_theta, axis=2)))
        planner.update_clearance()
        planner.U.zero_()  # rollout 0 is the noise-free nominal: standing still
        planner.replan(np.array([2.0 + tail - 0.04, 4.0, 0.0], np.float32), goal, 1)
        wp.synchronize()
        costs.append(float(planner.J.numpy()[0]))
    # one infeasible unit is 1e5 per metre of depth per step; 0.04 m over the horizon is ~1e5
    assert abs(costs[0] - costs[1]) < 1.0e3, f"standing still cost {costs[0]:.0f} vs {costs[1]:.0f}"
