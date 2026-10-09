"""Shared helpers for the planning/control benchmarks: the robot's own planner configuration on its
own window sizes, and timing.

Everything the planner is built from comes from `ros/config/odin.params.yaml` through
`helhest.planner_config`, the one function the node and `drive_sim` use -- so a number here is a
number for the controller the robot runs, not for library defaults. The terrain is a stress world,
resampled to the map cell and cropped to the robot's two robot-centred windows: the MPPI window
(`win_m`) at the map cell and the routing window (`route_m`) max-pooled by `plan_lat_coarsen`,
exactly as `navigation_node._build_planner` sizes them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
import yaml
from helhest import dynamics
from helhest import worlds as W
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.heightmap import Heightmap
from helhest.planner_config import planner_config
from helhest.planner_config import PlannerConfig
from helhest.planning.costtogo import CostToGo

ROBOT_PARAMS = Path(__file__).resolve().parents[1] / "ros/config/odin.params.yaml"
# navigation_node's declared defaults for what the params file leaves out (its `d(...)` calls)
NODE_DEFAULTS = {
    "resolution": 0.08,
    "win_m": 12.0,
    "route_m": 16.0,
    "plan_lat_coarsen": 3,
    "plan_n_refine": 3,
    "plan_friction": 0.8,
    "k_turn": -1.0,
    "terrain": "outdoor",
}


def robot_params() -> dict:
    """The robot's parameters: the node's defaults, overlaid with its params file."""
    doc = yaml.safe_load(ROBOT_PARAMS.read_text())
    params = next(v["ros__parameters"] for v in doc.values() if "ros__parameters" in v)
    return {**NODE_DEFAULTS, **params}


@dataclass
class RobotScene:
    """One planning frame as the node sees it, in the MPPI window's frame (origin at its corner)."""

    params: dict
    cfg: PlannerConfig
    k_turn: float
    win: np.ndarray  # [ny, nx] MPPI window terrain at the map cell
    win_grid: GridParams
    route: np.ndarray  # [ny, nx] routing window terrain, max-pooled by plan_lat_coarsen
    route_grid: GridParams  # in its own frame (origin 0), as CostToGo is built
    sgrid: GridParams  # the routing grid placed in the MPPI window's frame
    state: np.ndarray  # (x, y, yaw) of the robot, MPPI frame
    goal: tuple[float, float]  # MPPI frame
    goal_route: tuple[float, float]  # routing frame


def _window(scene: Heightmap, center: np.ndarray, n: int, cell: float) -> np.ndarray:
    """An n x n window at `cell`, centred on `center`, sampled (nearest) from the world raster."""
    offsets = (np.arange(n) - n / 2 + 0.5) * cell
    cols = np.clip(np.rint((center[0] + offsets - scene.x0) / scene.cell), 0, scene.nx - 1)
    rows = np.clip(np.rint((center[1] + offsets - scene.y0) / scene.cell), 0, scene.ny - 1)
    return np.ascontiguousarray(scene.H[rows.astype(int)[:, None], cols.astype(int)[None, :]])


def robot_scene(world: str) -> RobotScene:
    """The robot at the world's start, the goal pulled into the routing window along its bearing."""
    params = robot_params()
    cfg = planner_config(params)
    kt = float(params["k_turn"])
    k_turn = kt if kt >= 0.0 else dynamics.k_turn_for(str(params["terrain"]))
    builder, start, goal = W.WORLDS[world]
    scene = builder()
    cell = float(params["resolution"])
    nw = int(round(float(params["win_m"]) / cell))
    nr = int(round(float(params["route_m"]) / cell))
    k = max(1, int(params["plan_lat_coarsen"]))
    center = np.asarray(start[:2], np.float64)
    win = _window(scene, center, nw, cell)
    fine_route = _window(scene, center, nr, cell)
    nrc = nr // k  # max-pool keeps thin walls, as the node's belief pool does
    route = np.ascontiguousarray(
        fine_route[: nrc * k, : nrc * k].reshape(nrc, k, nrc, k).max(axis=(1, 3))
    )
    # the node's goals may lie far outside the window; here one stays inside so every row routes
    bearing = np.asarray(goal, np.float64) - center
    reach = min(float(np.hypot(*bearing)), 0.4 * nr * cell)
    goal_world = center + bearing / max(float(np.hypot(*bearing)), 1e-9) * reach
    win_corner = center - nw * cell / 2
    route_corner = center - nr * cell / 2
    return RobotScene(
        params=params,
        cfg=cfg,
        k_turn=k_turn,
        win=win,
        win_grid=GridParams(nw, nw, cell, 0.0, 0.0),
        route=route,
        route_grid=GridParams(nrc, nrc, cell * k, 0.0, 0.0),
        sgrid=GridParams(
            nrc, nrc, cell * k, (nw // 2 - nr // 2) * cell, (nw // 2 - nr // 2) * cell
        ),
        state=np.array([*(center - win_corner), start[2]], np.float32),
        goal=tuple(goal_world - win_corner),
        goal_route=tuple(goal_world - route_corner),
    )


def build_costtogo(rs: RobotScene, device: str, n_theta: int | None = None) -> CostToGo:
    """The node's CostToGo: its routing grid, the cylinder wheel, the plan config's settings."""
    kwargs = dict(rs.cfg.costtogo)
    if n_theta is not None:
        kwargs["n_theta"] = n_theta
    return CostToGo(
        rs.route_grid,
        dynamics.robot_params(rs.cfg.wheel_width),
        dynamics.planning_solver(k_turn=rs.k_turn),
        **kwargs,
        device=device,
    )


def route_inputs(rs: RobotScene, device: str) -> dict[str, wp.array]:
    """Device inputs to `CostToGo.compute` shaped like the node's belief crops: every cell
    measured, a flat 2 cm height sd and no pose drift, so the sigma path runs as on the robot."""
    shape = rs.route.shape
    return dict(
        elevation=wp.array(rs.route, dtype=wp.float32, device=device),
        measured=wp.array(np.ones(shape, np.float32), dtype=wp.float32, device=device),
        sigma=wp.array(np.full(shape, 0.02, np.float32), dtype=wp.float32, device=device),
        drift=wp.array(np.zeros(shape, np.float32), dtype=wp.float32, device=device),
    )


def build_rollout_sim(rs: RobotScene, device: str, batch: int, horizon: int) -> ForwardSimulator:
    """The node's MPPI rollout simulator: its window, solver, wheel and uniform friction."""
    sim = ForwardSimulator(
        dynamics.robot_params(rs.cfg.wheel_width),
        dynamics.planning_solver(k_turn=rs.k_turn),
        rs.win_grid,
        batch,
        horizon,
        device,
    )
    sim.set_terrain(wp.array(rs.win, dtype=wp.float32, device=device))
    sim.set_uniform_friction(float(rs.params["plan_friction"]))
    return sim


def time_fn(fn: Callable[[], object], reps: int, device: str) -> float:
    """Mean wall-clock with a warmup (captures the CUDA graph) and syncs around the timed loop."""
    fn()
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    wp.synchronize_device(device)
    return (time.perf_counter() - t0) / reps
