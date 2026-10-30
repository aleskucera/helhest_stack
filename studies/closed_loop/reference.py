"""The reference trajectory: what a run SHOULD have driven, computed from the world alone.

The closed-loop harness scores reach, time, contact and held frames. None of those asks for a
good-looking path, so no planner we have produces one except by accident: the route field is a
shortest-time cost-to-go and cuts every corner it may, and MPPI descends it with random samples.
This module defines "good" so it can be scored (quality.py): the shortest path from start to goal
that keeps a chosen clearance from every obstacle, never bends tighter than a chosen curvature,
and changes curvature no faster than a chosen rate. Three numbers, all in metres:

    clearance_m   [m]    the robot's SIDE stays at least this far from a wall
    radius_min_m  [m]    the tightest turn, 1 / kappa_max
    clothoid_m    [m]    the path length over which curvature may go from 0 to kappa_max
                         (kappa_rate_max = kappa_max / clothoid_m)

Built in three steps, each deterministic, so two runs of the same world give the same curve and a
symmetric world gives a symmetric curve:

  1. a shortest path on a grid restricted to cells at least `offset` from an obstacle, where
     offset = clearance + the robot's half-width -- this fixes WHICH WAY ROUND, the homotopy.
     Where the world is tighter than that (the gap's 1.8 m door is narrower than the body plus
     twice 0.5 m) the offset is lowered until a path exists, and the clearance asked of the curve
     is lowered to match only where the taut string had to give it up;
  2. that path pulled taut into the shortest polyline with the clearance kept along every
     straight (a grid path is one of many equal-cost staircases);
  3. the polyline resampled at equal spacing and smoothed by minimising
         length + w_k * sum kappa^2 + w_dk * sum (d kappa / ds)^2
     subject to clearance, |kappa| <= kappa_max and |d kappa / ds| <= kappa_rate_max, all as
     quadratic penalties, by L-BFGS with an analytic gradient, coarse spacing first and refined
     twice (the curvature-rate term stiffens as 1 / spacing^3, so the fine problem alone crawls).
     The curve starts and ends straight: curvature ramps from 0 at the start pose and back to 0
     at the goal within the clothoid bound. The smoothness weights are small: they pick the
     smoothest of the shortest paths, they do not trade length for smoothness.

Obstacles are the world's boxes (helhest.worlds.OBSTACLES) plus the arena's edge, which the
simulator closes with a wall; the distance to them is exact, so the penalty and its gradient
agree and the optimiser converges.

The body is two discs of the robot's half-width (half-track plus half a wheel, 0.415 m): one at
the front axle, which is the base origin the path is the track of, and one at the rear axle,
0.75 m behind it along the path's tangent. `clearance_m` is kept by both, so on a curve the
tail's inner wheel, which runs inside the front axle's track, is what sets the offset at an
inside corner. The nose (0.35 m ahead of the front axle) and the tail's end (a wheel radius
behind the rear axle) are not modelled beyond those discs; the simulator's own clearance
(worlds.obstacle_clearance) uses the full rectangle, so its number can sit a few centimetres
below the reference's at a corner.
"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import os

# the optimiser's vectors are a few hundred long: BLAS threads only add overhead, and with the
# default thread count the box reference took 80 s of wall clock for 11 min of CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from helhest import dynamics  # noqa: E402
from helhest import worlds as W  # noqa: E402


@dataclasses.dataclass(frozen=True)
class ReferenceParams:
    # The defaults are the shape chosen on the box world (2026-10-28): wide and gentle, the
    # wheels 0.8 m off a wall, no turn tighter than 1.5 m, 1.5 m of path to wind the curvature
    # up or down. Doors narrower than the body plus twice the clearance (gap, false_door) are
    # passed through their middle at the clearance they allow.
    clearance_m: float = 0.8  # [m] from the robot's side to the nearest obstacle
    radius_min_m: float = 1.5  # [m] the tightest turn the path may take
    clothoid_m: float = 1.5  # [m] path length to go from straight to the tightest turn
    spacing_m: float = 0.1  # [m] between the path's points
    # smoothness weights: dimensionless multipliers on sum(kappa^2 ds) [1/m] and
    # sum((dkappa/ds)^2 ds) [1/m^3], small so that length still dominates
    w_curvature: float = 0.05
    w_curvature_rate: float = 0.02
    w_penalty: float = 20000.0  # on every violated bound (clearance, kappa, dkappa/ds)

    @property
    def kappa_max(self) -> float:
        return 1.0 / self.radius_min_m

    @property
    def kappa_rate_max(self) -> float:
        return self.kappa_max / self.clothoid_m


@dataclasses.dataclass(frozen=True)
class Scene:
    boxes: tuple[W.Box, ...]
    xlim: tuple[float, float]  # the arena: the simulator closes its extent with a wall
    ylim: tuple[float, float]

    def distance(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[N] signed distance from points [N, 2] to the nearest obstacle (negative inside) and
        its gradient [N, 2], the unit direction away from that obstacle."""
        p = np.atleast_2d(p)
        edges = np.array(
            [
                p[:, 0] - self.xlim[0],
                self.xlim[1] - p[:, 0],
                p[:, 1] - self.ylim[0],
                self.ylim[1] - p[:, 1],
            ]
        )
        best = edges.min(axis=0)
        which = edges.argmin(axis=0)
        grad = np.array([(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)])[which]
        for b in self.boxes:
            if abs(b.yaw) > 1e-9:
                raise NotImplementedError("rotated boxes")
            q = np.abs(p - (b.cx, b.cy)) - (b.hx, b.hy)
            outside = np.maximum(q, 0.0)
            d_out = np.hypot(outside[:, 0], outside[:, 1])
            d_in = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
            d = d_out + d_in
            g = np.where(
                (d_out > 0.0)[:, None],
                outside / np.maximum(d_out, 1e-12)[:, None],
                np.where((q[:, 0] > q[:, 1])[:, None], [[1.0, 0.0]], [[0.0, 1.0]]),
            )
            g = g * np.sign(p - (b.cx, b.cy))
            closer = d < best
            best = np.where(closer, d, best)
            grad[closer] = g[closer]
        return best, grad


def half_width_m() -> float:
    """The robot's half-width: half-track plus half a wheel (the disc the path is offset for)."""
    rp = dynamics.robot_params(0.1)
    return float(rp.half_track + 0.5 * rp.wheel_width)


def rear_axle_m() -> float:
    """[m] the rear axle behind the front one, the base origin whose track the path is."""
    return float(dynamics.robot_params(0.1).rear_offset)


def rear_discs(p: np.ndarray, rear: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rear axle's position for every point of a path: `rear` behind it along the tangent
    taken from the neighbours (one-sided at the ends). Returns the positions [N, 2], the unit
    tangents [N, 2] and the tangent chords' lengths [N]."""
    v = np.empty_like(p)
    v[1:-1] = p[2:] - p[:-2]
    v[0] = p[1] - p[0]
    v[-1] = p[-1] - p[-2]
    length = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)
    t = v / length[:, None]
    return p - rear * t, t, length


def scene_for_world(world: str, cell: float = 0.06) -> Scene:
    builder = W.WORLDS[world][0]
    hm = builder(cell)
    return Scene(
        boxes=tuple(W.OBSTACLES[world]),
        xlim=(float(hm.x0), float(hm.x0 + hm.nx * hm.cell)),
        ylim=(float(hm.y0), float(hm.y0 + hm.ny * hm.cell)),
    )


def grid_path(
    scene: Scene, start: np.ndarray, goal: np.ndarray, offset: float, cell: float
) -> np.ndarray | None:
    """Shortest 8-connected grid path from start to goal through cells at least `offset` from an
    obstacle, as world points [M, 2], or None when there is none. The start and goal cells are
    admitted whatever their clearance, so a start against a wall still has a path."""
    xs = np.arange(scene.xlim[0] + 0.5 * cell, scene.xlim[1], cell)
    ys = np.arange(scene.ylim[0] + 0.5 * cell, scene.ylim[1], cell)
    XX, YY = np.meshgrid(xs, ys)
    dist, _ = scene.distance(np.column_stack([XX.ravel(), YY.ravel()]))
    free = (dist >= offset).reshape(XX.shape)
    ny, nx = free.shape

    def cell_of(p: np.ndarray) -> tuple[int, int]:
        c = int(np.clip((p[0] - scene.xlim[0]) / cell, 0, nx - 1))
        r = int(np.clip((p[1] - scene.ylim[0]) / cell, 0, ny - 1))
        return r, c

    s0, g0 = cell_of(start), cell_of(goal)
    free[s0] = True
    free[g0] = True
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    steps += [(-1, -1, 2**0.5), (-1, 1, 2**0.5), (1, -1, 2**0.5), (1, 1, 2**0.5)]
    best = np.full((ny, nx), np.inf)
    prev = np.full((ny, nx, 2), -1, np.int32)
    best[s0] = 0.0
    heap = [(0.0, s0)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if (r, c) == g0:
            break
        if d > best[r, c]:
            continue
        for dr, dc, w in steps:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ny and 0 <= cc < nx and free[rr, cc]:
                # a diagonal must not cut between two blocked orthogonal neighbours
                if dr and dc and not (free[r + dr, c] and free[r, c + dc]):
                    continue
                nd = d + w
                if nd < best[rr, cc]:
                    best[rr, cc] = nd
                    prev[rr, cc] = (r, c)
                    heapq.heappush(heap, (nd, (rr, cc)))
    if not np.isfinite(best[g0]):
        return None
    cells = [g0]
    while cells[-1] != s0:
        r, c = prev[cells[-1]]
        cells.append((int(r), int(c)))
    cells.reverse()
    pts = np.array([[xs[c], ys[r]] for r, c in cells])
    pts[0], pts[-1] = start[:2], goal[:2]
    return pts


def widest_grid_path(
    scene: Scene, start: np.ndarray, goal: np.ndarray, offset: float, cell: float
) -> tuple[np.ndarray, float]:
    """The grid path at the asked offset, or at the largest smaller offset (to the cell) that
    still has one: the door narrower than the clearance asked is passed through its middle."""
    lo, hi = half_width_m() + cell, offset
    path = grid_path(scene, start, goal, hi, cell)
    if path is not None:
        return path, hi
    if grid_path(scene, start, goal, lo, cell) is None:
        raise ValueError(f"no path even at {lo:.2f} m offset: the robot does not fit")
    while hi - lo > 0.5 * cell:
        mid = 0.5 * (lo + hi)
        if grid_path(scene, start, goal, mid, cell) is not None:
            lo = mid
        else:
            hi = mid
    return grid_path(scene, start, goal, lo, cell), lo


def string_pull(pts: np.ndarray, scene: Scene, offset: float, step: float = 0.03) -> np.ndarray:
    """The taut string: the shortest polyline through the grid path's points whose every straight
    keeps the clearance, found as the shortest path in the visibility graph of those points. A
    grid path is one of many equal-cost staircases (a diagonal costs sqrt 2, so "straight then
    diagonal" and "diagonal then straight" tie); taut, it is the Euclidean shortest polyline,
    which is what the smoother rounds. Greedy furthest-visible pulling is not enough: it keeps
    the staircase's bias and gives an asymmetric polyline on a symmetric world. The straights are
    tested a hair inside the offset: they touch the offset circle around a corner by
    construction, and the smoother enforces the exact bound afterwards."""
    pts = resample(pts, 4.0 * step)  # ~0.12 m: enough vertices to land on every tangent point
    m = len(pts)

    def clear_between(a: np.ndarray, b: np.ndarray) -> bool:
        n = max(int(np.linalg.norm(b - a) / step), 2)
        line = a[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None, :]
        return bool((scene.distance(line)[0] >= offset - 0.5 * step).all())

    best = np.full(m, np.inf)
    prev = np.full(m, -1, np.int64)
    best[0] = 0.0
    heap = [(0.0, 0)]
    while heap:
        d, i = heapq.heappop(heap)
        if i == m - 1:
            break
        if d > best[i]:
            continue
        for j in range(i + 1, m):
            if clear_between(pts[i], pts[j]):
                nd = d + float(np.linalg.norm(pts[j] - pts[i]))
                if nd < best[j]:
                    best[j] = nd
                    prev[j] = i
                    heapq.heappush(heap, (nd, j))
    chain = [m - 1]
    while chain[-1] != 0:
        chain.append(int(prev[chain[-1]]))
    return pts[chain[::-1]]


def chaikin(pts: np.ndarray, rounds: int) -> np.ndarray:
    """Corner cutting, ends fixed: a cheap near-smooth start for the optimiser, which otherwise
    begins at a polyline whose corners have unbounded curvature and crawls out of that."""
    for _ in range(rounds):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        pts = np.vstack([pts[:1], np.column_stack([q, r]).reshape(-1, 2), pts[-1:]])
    return pts


def resample(pts: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(s[-1] / spacing)), 2)
    si = np.linspace(0.0, s[-1], n + 1)
    return np.column_stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])])


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def curvature(pts: np.ndarray) -> np.ndarray:
    """Signed curvature at every point: the turn between the segments either side over their mean
    length, +ve = turning left; 0 at the ends, where the curve is built to run straight."""
    k = np.zeros(len(pts))
    d = np.diff(pts, axis=0)
    theta = np.arctan2(d[:, 1], d[:, 0])
    seg = np.hypot(d[:, 0], d[:, 1])
    k[1:-1] = _wrap(np.diff(theta)) / np.maximum(0.5 * (seg[:-1] + seg[1:]), 1e-9)
    return k


def smooth_objective(
    x: np.ndarray,
    pts: np.ndarray,
    fixed_second: np.ndarray,
    fixed_penult: np.ndarray,
    scene: Scene,
    target: np.ndarray,
    ds: float,
    prm: ReferenceParams,
) -> tuple[float, np.ndarray]:
    """Length + curvature terms + bound penalties, with the analytic gradient.

    Curvature is the turn between consecutive segments over the spacing, which equals the
    geometric curvature when the points are evenly spaced -- and the spacing term keeps them so.
    That makes every term a function of the segment headings, whose gradient is a two-point
    stencil. The curvature is taken as 0 before the first segment and after the last, so the
    rate bound makes the curve start and end straight. `target` [N] is the clearance asked of
    each point (the offset, lowered where the world is too tight for it)."""
    n = len(pts)
    p = np.empty((n, 2))
    p[0], p[1], p[-2], p[-1] = pts[0], fixed_second, fixed_penult, pts[-1]
    p[2:-2] = x.reshape(-1, 2)
    d = np.diff(p, axis=0)  # [n-1, 2] segments
    seg = np.hypot(d[:, 0], d[:, 1])
    seg_safe = np.maximum(seg, 1e-9)
    theta = np.arctan2(d[:, 1], d[:, 0])  # [n-1]
    k = _wrap(np.diff(theta)) / ds  # [n-2], at points 1..n-2
    k_ext = np.concatenate([[0.0], k, [0.0]])  # straight before the start, after the end
    dk = np.diff(k_ext) / ds  # [n-1]
    dist, dgrad = scene.distance(p)
    clear = dist - target
    rear = rear_axle_m()
    q, t, chord = rear_discs(p, rear)
    dist_r, dgrad_r = scene.distance(q)
    clear_r = dist_r - target
    k_over = np.maximum(np.abs(k) - prm.kappa_max, 0.0) * np.sign(k)
    dk_over = np.maximum(np.abs(dk) - prm.kappa_rate_max, 0.0) * np.sign(dk)
    j = seg.sum()
    j += prm.w_curvature * np.sum(k * k) * ds
    j += prm.w_curvature_rate * np.sum(dk * dk) * ds
    j += prm.w_penalty * np.sum(np.minimum(clear, 0.0) ** 2)
    j += prm.w_penalty * np.sum(np.minimum(clear_r, 0.0) ** 2)
    j += prm.w_penalty * np.sum(k_over**2)
    j += prm.w_penalty * np.sum(dk_over**2)
    j += 10.0 * np.sum((seg - ds) ** 2)
    # --- gradient, assembled through dJ/dtheta and dJ/dseg
    dj_dk = 2.0 * prm.w_curvature * k * ds + 2.0 * prm.w_penalty * k_over  # [n-2]
    dj_ddk = 2.0 * prm.w_curvature_rate * dk * ds + 2.0 * prm.w_penalty * dk_over  # [n-1]
    # dk_i = (k_ext[i+1] - k_ext[i]) / ds, and k_ext[1..n-2] are the free k
    dj_dk = dj_dk + (dj_ddk[:-1] - dj_ddk[1:]) / ds
    dj_dtheta = np.zeros(n - 1)
    dj_dtheta[1:] += dj_dk / ds
    dj_dtheta[:-1] -= dj_dk / ds
    dj_dseg = 1.0 + 20.0 * (seg - ds)
    # theta_i = atan2(dy, dx): d theta / d(dx, dy) = (-dy, dx) / |d|^2; seg_i: d / |d|
    g_d = (dj_dtheta / seg_safe**2)[:, None] * np.column_stack([-d[:, 1], d[:, 0]])
    g_d += (dj_dseg / seg_safe)[:, None] * d
    g_p = np.zeros((n, 2))
    g_p[1:] += g_d
    g_p[:-1] -= g_d
    g_p += (2.0 * prm.w_penalty * np.minimum(clear, 0.0))[:, None] * dgrad
    # the rear disc: q_i = p_i - rear * t_i, with t_i = v_i / |v_i| and v_i = p_{i+1} - p_{i-1}
    # (one-sided at the ends); d t / d v = (I - t t^T) / |v|
    g_q = (2.0 * prm.w_penalty * np.minimum(clear_r, 0.0))[:, None] * dgrad_r
    g_p += g_q
    h = rear * (g_q - t * np.sum(t * g_q, axis=1)[:, None]) / chord[:, None]
    g_p[2:] -= h[1:-1]
    g_p[:-2] += h[1:-1]
    g_p[1] -= h[0]
    g_p[0] += h[0]
    g_p[-1] -= h[-1]
    g_p[-2] += h[-1]
    return float(j), g_p[2:-2].reshape(-1)


def smooth(
    init: np.ndarray,
    taut_dense: np.ndarray,
    scene: Scene,
    offset: float,
    prm: ReferenceParams,
    start_heading: float,
    end_heading: float,
) -> np.ndarray:
    """Minimise length + curvature terms under the bounds, at spacings 4x, 2x and 1x the asked
    one in turn, each warm-started from the last. The ends and both headings are fixed: the
    second point stays on the start heading one spacing ahead, the second-to-last on the end
    heading one spacing short of the goal. The clearance asked of a point is the offset, or the
    taut string's own clearance there when that is smaller: a door narrower than the offset is
    passed through its middle, not refused."""
    # the clearance the TAUT string keeps, by arc length: the corner-cut start sits inside the
    # offset at every corner and must not lower the target there
    ref_clear = scene.distance(taut_dense)[0] - 0.01
    ref_s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(taut_dense, axis=0).T))])
    pts = init
    for factor in (4, 2, 1):
        ds = prm.spacing_m * factor
        pts = resample(pts, ds)
        n = len(pts)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        target = np.minimum(offset, np.interp(s, ref_s, ref_clear))
        fixed_second = pts[0] + ds * np.array([np.cos(start_heading), np.sin(start_heading)])
        fixed_penult = pts[-1] - ds * np.array([np.cos(end_heading), np.sin(end_heading)])
        args = (pts, fixed_second, fixed_penult, scene, target, ds, prm)
        res = minimize(
            smooth_objective,
            pts[2:-2].reshape(-1),
            args=args,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 4000, "maxfun": 8000, "ftol": 1e-13, "gtol": 1e-9, "maxcor": 30},
        )
        p = np.empty((n, 2))
        p[0], p[1], p[-2], p[-1] = pts[0], fixed_second, fixed_penult, pts[-1]
        p[2:-2] = res.x.reshape(-1, 2)
        pts = p
    return pts


@dataclasses.dataclass(frozen=True)
class Reference:
    world: str
    params: ReferenceParams
    points: np.ndarray  # [N, 2] world metres, start to goal
    s: np.ndarray  # [N] arc length
    kappa: np.ndarray  # [N] signed curvature, +ve = left
    clearance: np.ndarray  # [N] the body's side to the nearest obstacle, front or rear axle disc
    taut: np.ndarray  # [M, 2] the taut polyline the curve was rounded from
    offset_used: float  # [m] the offset the taut string ran at (lower than asked in a tight world)
    scene: Scene


def reference_for_world(
    world: str, prm: ReferenceParams = ReferenceParams(), cell: float = 0.06
) -> Reference:
    _, start, goal = W.WORLDS[world]
    scene = scene_for_world(world, cell)
    offset = prm.clearance_m + half_width_m()
    start_xy, goal_xy = np.array(start[:2], float), np.array(goal[:2], float)
    raw, used = widest_grid_path(scene, start_xy, goal_xy, offset, cell)
    taut = string_pull(raw, scene, used)
    if np.linalg.norm(taut[0] - start_xy) > 1e-9:
        taut = taut[::-1]
    init = resample(chaikin(resample(taut, 0.5), 3), prm.spacing_m)
    # The goal carries no heading. The curve arrives the way it left -- the start heading --
    # which keeps a symmetric world's curve symmetric, unless the taut string's last leg comes
    # in more than 45 deg off that (pocket: the goal is entered from behind; false_door: from
    # the side), when that leg's own bearing is used rather than a hook at the goal.
    last = taut[-1] - taut[-2]
    last_bearing = float(np.arctan2(last[1], last[0]))
    end_heading = float(start[2])
    if abs(_wrap(np.array([last_bearing - end_heading]))[0]) > 0.25 * np.pi:
        end_heading = last_bearing
    taut_dense = resample(taut, 0.5 * prm.spacing_m)
    pts = smooth(init, taut_dense, scene, offset, prm, float(start[2]), end_heading)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    return Reference(
        world=world,
        params=prm,
        points=pts,
        s=np.concatenate([[0.0], np.cumsum(seg)]),
        kappa=curvature(pts),
        clearance=np.minimum(
            scene.distance(pts)[0], scene.distance(rear_discs(pts, rear_axle_m())[0])[0]
        )
        - half_width_m(),
        taut=taut,
        offset_used=used,
        scene=scene,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--world", default="box")
    p.add_argument("--clearance", type=float, default=0.8)
    p.add_argument("--radius", type=float, default=1.5)
    p.add_argument("--clothoid", type=float, default=1.5)
    p.add_argument("--out", default=None, help="save the reference as .npz")
    a = p.parse_args()
    prm = ReferenceParams(clearance_m=a.clearance, radius_min_m=a.radius, clothoid_m=a.clothoid)
    ref = reference_for_world(a.world, prm)
    print(
        f"{a.world}: {ref.s[-1]:.2f} m, |kappa| max {np.abs(ref.kappa).max():.2f} 1/m "
        f"(bound {prm.kappa_max:.2f}), clearance min {ref.clearance.min():.2f} m "
        f"(asked {prm.clearance_m:.2f}, taut string at {ref.offset_used - half_width_m():.2f})"
    )
    if a.out:
        np.savez_compressed(
            a.out, points=ref.points, s=ref.s, kappa=ref.kappa, clearance=ref.clearance
        )


if __name__ == "__main__":
    main()
