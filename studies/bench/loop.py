"""Closed-loop harness for the section-6 benchmark: plan on belief, drive on truth.

Adapted from `demos/navigate_partial.py`, which already had the hard parts -- occluded
synthetic lidar, an accumulated map, unknown cells inpainted optimistically to flat, a
`WarpDriver` standing in for reality, and the MPPI + cost-to-go split. What this adds is the
thing under test:

  A LOOK. Once per `look_interval` frames the robot may stop and aim its sensor at a chosen
  bearing, trading `look_cost_frames` of time for a long-range, narrow-FOV observation. The
  POLICY chooses the bearing; everything else is identical across policies. Time-to-goal is
  measured in frames, so a look is paid for in exactly the currency the metric counts.

The belief is deliberately optimistic (unknown -> flat), matching the demo and the real
planner. That is what makes an unseen barrier dangerous: the robot happily plans straight
through a wall it has not observed, discovers it at short range, and has to recover.

Nothing here knows which policy it is running. `policy` is a callable
`(belief, pose, plan) -> bearing | None`, so the comparison cannot accidentally give one
policy information another does not get.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np
import warp as wp

from helhest import dynamics
from helhest.control.mppi import CostParams
from helhest.control.mppi import MppiGpu
from helhest.control.terminal import dock_control
from helhest.driver import WarpDriver
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.perception.lidar import crop_window
from helhest.perception.lidar import lidar_scan
from helhest.perception.lidar import MultiScanMap
from helhest.planning.costtogo import CostToGo
from helhest.worlds import matching_friction

from . import world as W

MOUNT = 0.4
GOAL_RADIUS = 0.4  # [m] counts as reached
DOCK_RADIUS = 1.5
LOOK_INTERVAL = 25  # frames between opportunities to look
LOOK_COST_FRAMES = 12  # the price of a look, in the same units as time-to-goal
MAX_FRAMES = 900


@dataclass
class Belief:
    """What the robot currently thinks the world looks like."""

    elev: np.ndarray  # observed heights; unknown cells read 0.0 (optimistic)
    known: np.ndarray
    scene_x0: float
    scene_y0: float
    cell: float

    def sigma(self, unknown: float = 0.25, observed: float = 0.01) -> np.ndarray:
        """Placeholder per-cell uncertainty. Unknown cells dominate, as they should."""
        return np.where(self.known, observed, unknown)


@dataclass
class Trace:
    """Everything needed to score a run and to check it did what it claimed."""

    reached: bool = False
    frames: int = 0
    look_frames: int = 0
    n_looks: int = 0
    contacts: int = 0
    path_len: float = 0.0
    closest: float = 99.0
    look_bearings: list = field(default_factory=list)
    look_at_gap: int = 0  # looks that actually revealed part of the gap
    look_at_decoy: int = 0
    gap_known_frame: int = -1  # first frame the gap was known
    trail: list = field(default_factory=list)

    @property
    def total_time(self) -> int:
        """Frames driven plus frames spent standing still looking."""
        return self.frames + self.look_frames


def _scan(bw, pose, fov, rng_m):
    return lidar_scan(
        bw.scene.H,
        bw.scene.x0,
        bw.scene.y0,
        bw.scene.cell,
        pose,
        fov_deg=fov,
        max_range=rng_m,
        mount_height=MOUNT,
    )


def run(
    bw: W.BenchWorld,
    policy=None,
    device: str = "cuda",
    win_m: float = 9.0,
    route_m: float = 18.0,
    n_theta: int = 24,
    lat_coarsen: int = 4,
    batch: int = 4096,
    horizon: int = 70,
    max_frames: int = MAX_FRAMES,
    omniscient: bool = False,
) -> Trace:
    """One closed-loop episode. `policy=None` is the null baseline (never looks).

    `omniscient=True` hands the robot the true map from frame 0. That is not a policy -- it is
    the ORACLE, the best time-to-goal any amount of sensing could ever buy. Without it the
    baseline's time is a number with nothing to compare against, and there is no way to say
    whether the scenario leaves any headroom for a sensing policy to recover.
    """
    scene = bw.scene
    cell = scene.cell
    goal = np.asarray(bw.goal, np.float64)
    mu = matching_friction(scene)

    drv = WarpDriver(scene, mu, init_pose=tuple(bw.start), device=device)  # REALITY
    ww = wh = int(round(win_m / cell))
    win_grid = GridParams(ww, wh, cell, 0.0, 0.0)
    plan_sim = ForwardSimulator(
        dynamics.robot_params(), dynamics.planning_solver(), win_grid, batch, horizon, device
    )
    plan_sim.set_uniform_friction(0.8)
    planner = MppiGpu(plan_sim, CostParams(), n_theta=n_theta)
    planner.reset_nominal(1.5)

    rww = rwh = int(round(max(route_m, win_m) / cell))
    kr = max(1, int(lat_coarsen))
    rcny, rcnx, rccell = rwh // kr, rww // kr, cell * kr
    route_grid = GridParams(rcnx, rcny, rccell, 0.0, 0.0)
    ctg = CostToGo(
        route_grid,
        dynamics.robot_params(),
        dynamics.planning_solver(),
        n_theta=n_theta,
        device=device,
    )
    planner.cw.lattice_cap = ctg._vcap
    sgrid = GridParams(
        rcnx, rcny, rccell, (ww // 2 - rww // 2) * cell, (wh // 2 - rwh // 2) * cell
    ).build()

    mm = MultiScanMap(scene.ny, scene.nx)
    if omniscient:
        mm.elev[:] = scene.H
        mm.known[:] = True
    tr = Trace()
    last_look = -LOOK_INTERVAL
    prev = None

    for f in range(max_frames):
        st = drv.render_state()
        rx, ry, yaw = st.x, st.y, st.yaw
        tr.trail.append((rx, ry))
        if prev is not None:
            tr.path_len += float(np.hypot(rx - prev[0], ry - prev[1]))
        prev = (rx, ry)
        d = float(np.hypot(rx - goal[0], ry - goal[1]))
        tr.closest = min(tr.closest, d)
        if d < GOAL_RADIUS:
            tr.reached = True
            break

        # --- default sensing: free, every frame, short range -------------------------------
        obs, known = _scan(bw, (rx, ry, yaw), W.DEFAULT_FOV, W.DEFAULT_RANGE)
        mm.integrate(obs, known)

        # --- the look: the decision under test ---------------------------------------------
        if policy is not None and f - last_look >= LOOK_INTERVAL:
            belief = Belief(
                np.where(mm.known, mm.elev, 0.0), mm.known.copy(), scene.x0, scene.y0, cell
            )
            bearing = policy(belief, (rx, ry, yaw), bw)
            if bearing is not None:
                lobs, lknown = _scan(bw, (rx, ry, float(bearing)), W.LOOK_FOV, W.LOOK_RANGE)
                fresh = lknown & ~mm.known
                mm.integrate(lobs, lknown)
                tr.n_looks += 1
                tr.look_frames += LOOK_COST_FRAMES
                tr.look_bearings.append(float(bearing))
                # Did the look do what its policy intends? Recorded, not assumed.
                if (fresh & bw.gap_mask).sum() > 0.05 * bw.gap_mask.sum():
                    tr.look_at_gap += 1
                if (fresh & bw.decoy_mask).sum() > 0.05 * bw.decoy_mask.sum():
                    tr.look_at_decoy += 1
                last_look = f
        if tr.gap_known_frame < 0 and (mm.known & bw.gap_mask).sum() > 0.3 * bw.gap_mask.sum():
            tr.gap_known_frame = f

        # --- plan on the belief -------------------------------------------------------------
        elev, kn, wx0, wy0 = crop_window(mm, scene, rx, ry, ww, wh, cell)
        elev = np.where(kn, elev, 0.0).astype(np.float32)  # unknown -> flat (optimistic)
        goal_l = (goal[0] - wx0, goal[1] - wy0)
        state_l = np.array([rx - wx0, ry - wy0, yaw], np.float32)
        plan_sim.set_terrain(wp.array(np.ascontiguousarray(elev), dtype=wp.float32, device=device))

        relev, rkn, rwx0, rwy0 = crop_window(mm, scene, rx, ry, rww, rwh, cell)
        relev = np.where(rkn, relev, 0.0).astype(np.float32)
        goal_r = (goal[0] - rwx0, goal[1] - rwy0)
        Hc = (
            relev[: rcny * kr, : rcnx * kr].reshape(rcny, kr, rcnx, kr).max(axis=(1, 3))
            if kr > 1
            else relev
        )
        V = ctg.compute(wp.array(np.ascontiguousarray(Hc), dtype=wp.float32, device=device), goal_r)
        planner.set_lattice(V, sgrid)

        if d < DOCK_RADIUS:
            cmd = dock_control(state_l, goal_l)
        else:
            planner.replan(state_l, goal_l, 3)
            u = planner.nominal()
            cmd = np.array([u[0, 0], u[0, 1], 0.5 * (u[0, 0] + u[0, 1])], np.float32)
        drv.step(cmd)
        if drv.clear < 0.05:
            tr.contacts += 1
        tr.frames = f + 1

    tr.frames = max(tr.frames, 1)
    return tr
