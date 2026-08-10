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
# A FIXED, SMALL look budget, identical for every policy. Originally the robot could look
# every 25 frames for as long as the episode ran -- up to 36 looks at 12 frames each, 432
# frames of pure overhead against 20-230 frames of headroom. That does not measure aiming, it
# measures how often a policy chooses to stop, and it swamped the signal completely:
# attribution beat the null baseline on the hard seeds and lost catastrophically on the easy
# ones purely on look cost. Capping the budget makes the overhead a CONSTANT that cancels in
# paired comparisons, so what remains is the only thing the claim is about -- WHERE to aim.
LOOK_INTERVAL = 30  # frames between opportunities
MAX_LOOKS = 4  # hard cap per episode
LOOK_COST_FRAMES = 8  # the price of a look, in the same units as time-to-goal
MAX_FRAMES = 900


@dataclass
class Belief:
    """What the robot currently thinks the world looks like."""

    elev: np.ndarray  # observed heights; unknown cells read 0.0 (optimistic)
    known: np.ndarray
    scene_x0: float
    scene_y0: float
    cell: float

    def sigma(self, observed: float = 0.01, base: float = 0.12, gain: float = 1.4) -> np.ndarray:
        """Per-cell uncertainty, predicted from CONTEXT rather than assumed uniform.

        A uniform "unknown = 0.25" makes every unobserved cell identical, which leaves an
        information-gain objective almost indifferent between directions: the look cone has the
        same area whichever way it points, so an area-counting NBV has no signal and picks
        essentially arbitrarily. That is not a fair baseline -- it is a baseline with nothing to
        go on -- and beating it would say nothing.

        Real uncertainty models (UNRealNet arXiv:2407.08720, the Neural-Processes elevation
        model arXiv:2508.03890) predict HIGHER uncertainty over rough or complex terrain before
        observing it, from context. Emulated here at block resolution: measure roughness where
        the map IS observed, then carry it into neighbouring unobserved blocks. Rough
        neighbourhoods therefore read as high-sigma while smooth ones read as low-sigma, which
        is what gives an entropy-directed sensor a real preference to be wrong about.
        """
        block = max(int(round(2.0 / self.cell)), 1)
        ny, nx = self.known.shape
        by, bx = ny // block, nx // block
        e = self.elev[: by * block, : bx * block].reshape(by, block, bx, block)
        k = self.known[: by * block, : bx * block].reshape(by, block, bx, block)
        seen = k.any(axis=(1, 3))
        hi = np.where(k, e, -np.inf).max(axis=(1, 3))
        lo = np.where(k, e, np.inf).min(axis=(1, 3))
        rough = np.where(seen, np.nan_to_num(hi - lo, neginf=0.0, posinf=0.0), np.nan)
        # carry observed roughness into neighbouring unobserved blocks
        for _ in range(6):
            filled = np.nan_to_num(rough, nan=0.0)
            have = ~np.isnan(rough)
            acc = np.zeros_like(filled)
            cnt = np.zeros_like(filled)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    acc += np.roll(np.roll(filled, dy, 0), dx, 1)
                    cnt += np.roll(np.roll(have.astype(float), dy, 0), dx, 1)
            grown = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
            rough = np.where(have, rough, np.where(cnt > 0, grown, np.nan))
        rough = np.nan_to_num(rough, nan=0.0)
        full = np.kron(rough, np.ones((block, block)))
        pad = np.zeros((ny, nx))
        pad[: full.shape[0], : full.shape[1]] = full
        return np.where(self.known, observed, base + gain * pad)


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


def _trace_route(V, cell, wx0, wy0, pose, n_steps: int = 60):
    """Follow the cost-to-go downhill from the robot -> the route the planner intends to drive.

    V is [ny, nx, n_theta]; minimising over heading gives the heading-free value field. Simple
    steepest descent on it, in world coordinates. Returns [k, 2] or None if the robot is off
    the routing window.
    """
    Vmin = V.min(axis=2)
    ny, nx = Vmin.shape
    # world -> grid uses the CELL-CENTER convention (cell i's center is at wx0 + (i+0.5)*cell,
    # see ranking.py's XX/YY build and helhest.heightmap.Heightmap), not a corner convention --
    # using the corner offset the traced route ~0.17 m diagonally from the cells it claims to
    # pass through.
    c = int(round((pose[0] - wx0) / cell - 0.5))
    r = int(round((pose[1] - wy0) / cell - 0.5))
    if not (0 <= r < ny and 0 <= c < nx):
        return None
    pts = []
    for _ in range(n_steps):
        pts.append((wx0 + (c + 0.5) * cell, wy0 + (r + 0.5) * cell))
        best, br, bc = Vmin[r, c], r, c
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < ny and 0 <= cc < nx and Vmin[rr, cc] < best:
                    best, br, bc = Vmin[rr, cc], rr, cc
        if (br, bc) == (r, c):
            break
        r, c = br, bc
    return np.array(pts) if len(pts) > 1 else None


class _MapView:
    """Duck-typed stand-in for MultiScanMap so `crop_window` can crop a SAMPLED map."""

    def __init__(self, elev, known):
        self.elev = elev
        self.known = known


def _make_route_sampler(scene, ctg, rww, rwh, kr, rcny, rcnx, rccell, goal, device, n=6):
    """Returns `sample_routes(mm, pose, sigma, seed) -> list of world-frame paths`.

    THE ELITE SET, done properly. Tracing several descents of ONE cost-to-go field does not
    work: the field is distance-like, its descent path is essentially a unique geodesic, and
    stochastic descent produced 0.54 m of spread at any temperature. Worse, the belief inpaints
    unknown ground as flat and therefore PASSABLE, so under that belief the planner is not
    uncertain at all -- it is confidently wrong, and there is no disagreement to detect.

    So the alternatives have to come from the UNCERTAINTY: draw maps consistent with the belief
    (mean + sigma noise over the unknown cells), re-solve the cost-to-go on each, and trace the
    route it implies. Where those routes diverge is where the map genuinely has not yet decided
    the plan -- which is the quantity a decision-focused sensor should be maximising.
    """

    def sample_routes(mm, pose, sigma, seed):
        rx, ry = pose[0], pose[1]
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            draw = mm.elev.copy()
            unknown = ~mm.known
            if unknown.any():
                # smooth, correlated draw: unknown ground is wrong in patches, not per cell
                z = rng.normal(size=draw.shape).astype(np.float32)
                for _ in range(3):
                    z = 0.25 * (
                        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)
                    )
                z *= 1.0 / (z.std() + 1e-9)
                draw[unknown] = (sigma * z)[unknown]
            view = _MapView(draw, np.ones_like(mm.known))
            relev, _, wx0, wy0 = crop_window(view, scene, rx, ry, rww, rwh, cell_of(scene))
            Hc = (
                relev[: rcny * kr, : rcnx * kr].reshape(rcny, kr, rcnx, kr).max(axis=(1, 3))
                if kr > 1
                else relev
            )
            V = ctg.compute(
                wp.array(np.ascontiguousarray(Hc), dtype=wp.float32, device=device),
                (goal[0] - wx0, goal[1] - wy0),
            )
            r = _trace_route(V.numpy(), rccell, wx0, wy0, (rx, ry))
            if r is not None:
                out.append(r)
        return out

    return sample_routes


def cell_of(scene):
    return scene.cell


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

    sample_routes = _make_route_sampler(scene, ctg, rww, rwh, kr, rcny, rcnx, rccell, goal, device)
    mm = MultiScanMap(scene.ny, scene.nx)
    if omniscient:
        mm.elev[:] = scene.H
        mm.known[:] = True
    tr = Trace()
    last_look = -LOOK_INTERVAL
    prev = None
    route = None  # the planner's intended path; None until the first cost-to-go solve
    routes = []  # near-optimal alternatives (the elite set)

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
        if policy is not None and f - last_look >= LOOK_INTERVAL and tr.n_looks < MAX_LOOKS:
            belief = Belief(
                np.where(mm.known, mm.elev, 0.0), mm.known.copy(), scene.x0, scene.y0, cell
            )
            routes = sample_routes(mm, (rx, ry), belief.sigma(), seed=1000 + f)
            # Per-look rng, seeded from the episode's own seed and the frame it looks on --
            # NOT global numpy state -- so cvar's Monte-Carlo draws vary across seeds, episodes,
            # and looks within an episode (previously frozen to np.random.default_rng(0) inside
            # the policy, since loop.py called policies positionally and never passed one) while
            # staying reproducible for a repeated run with the same seed.
            rng = np.random.default_rng([bw.seed, f])
            bearing = policy(belief, (rx, ry, yaw), bw, route, routes, rng)
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
        # The planner's INTENDED ROUTE, traced by descending the cost-to-go. This is what a
        # decision-focused policy must be sensitive to. Using the straight line to the goal
        # instead -- the obvious shortcut -- makes attribution keep staring at a wall it has
        # already seen, because the straight line still points through it, while the route it
        # will actually drive bends away along the barrier.
        route = _trace_route(V.numpy(), rccell, rwx0, rwy0, (rx, ry))

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
