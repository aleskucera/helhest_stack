"""The reference path: the curve MPPI tracks, built every frame on the belief.

MPPI used to descend the route field V directly. V is a shortest-time cost-to-go, so its gradient
runs along the taut string and MPPI, with a 1.7 s horizon, follows it into every corner late and
hard, and never on the same line twice (drive_sim box world, 2026-10-28: 0.3-0.45 m mean offset
and 0.7-1.2 m mirror error against the reference, one yaw reversal per run). This stage turns the
descent of V into the curve the offline oracle (studies/closed_loop/reference.py) would draw on
the belief, and MPPI's goal term becomes "stay on it" (CostParams.track_*).

What it keeps clear of is the settle's verdict, not walls: `forbidden_distance` measures, on the
routing grid, how far each cell is from a cell where the settle forbids the robot at every
heading -- a wall, a slope past the tip-over envelope, a ditch the belly cannot clear. The robot
is meant for off-road ground, where a tall step is the least of what to keep away from, and the
same verdict is what MPPI vetoes with. The taut string may straighten V's descent only where
that does not cost more in V's own terms, so a straight never cuts back over the rough ground V
routed around (`_taut_kernel`).

Every frame, all on the device the fields already live on, captured as one CUDA graph:

  1. `_descent_kernel`   walk V downhill from the robot's cell to the window's edge or the goal:
                         the homotopy, as a staircase of cell centres;
  2. `_taut_kernel`      pull it taut: the shortest polyline through those points whose straights
                         keep `offset` = clearance + half-width from the wall-distance map
                         (consecutive points stay joined whatever their clearance, so a descent
                         that hugs a wall still yields a path, with its clearance lowered there);
                         every straight is tested at once (`_visible_kernel`);
  3. `_init_kernel`      on a cold start only (`_mode_kernel`: a new way out of the window), the
                         curvature profile from the taut string's heading;
  4. `_optimise_kernel`  the curve as a curvature profile kappa[N] over a free spacing ds,
                         integrated from the robot's pose: Adam steps on
                             w_end |p_j* - end|^2 + w_len j* ds + w_clear (clearance deficit of
                             the front and rear axle discs)^2 + w_k sum kappa^2 ds
                             + w_dk sum (dkappa/ds)^2 ds
                         with |kappa| <= 1/R and |dkappa/ds| <= (1/R)/clothoid enforced by
                         projection after every step, so every iterate is a drivable clothoid
                         spline. One thread block, one lane per point, every step in one launch.
                         Warm-started from last frame's profile, re-anchored at the robot, so the
                         optimisation is spread over frames.

The point-based objective the oracle uses needs L-BFGS: its curvature-rate hinge is too stiff for
any gradient step (prototyped 2026-10-28, every step diverged). The curvature parameterisation
makes the bounds free and the rest smooth; its cost is a lever-arm coupling that Adam handles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..grid import Grid
from ..grid import sample_field

# lanes of the optimiser's block (`_optimise_kernel`): one per curve point, so n_knots + 1 at most
BLOCK = wp.constant(128)
_vec_block = wp.types.vector(length=128, dtype=float)


@dataclass(frozen=True)
class ReferencePathParams:
    clearance_m: float = 0.8  # [m] the wheels' outer side to the nearest wall face
    radius_min_m: float = 1.5  # [m] the tightest turn the curve may take
    clothoid_m: float = 1.5  # [m] path length from straight to the tightest turn
    spacing_m: float = 0.25  # [m] between curvature knots at the start; the optimiser scales it
    n_knots: int = 64  # kappa knots; 64 at 0.25 m covers a 10 m window's longest path
    iterations: int = 100  # Adam steps per frame, warm-started from the last
    iterations_cold: int = 400  # Adam steps on a cold start (a new way out of the window)
    step: float = 0.01  # Adam step on kappa [1/m per iteration]
    w_end: float = 20.0
    w_clear: float = 50.0
    w_k: float = 0.02
    w_dk: float = 0.2
    w_len: float = 1.0
    # A TURN IN PLACE AT THE START, priced per rad^2. The curve's radius bound is a preference
    # the robot does not have -- it can pivot -- and from a pose pointing the wrong way beside a
    # wall no curve within the bound fits: the optimiser returned loops through the slab
    # (slalom, 2026-10-28). The start heading is free, at this price; MPPI's spin band turns.
    w_pivot: float = 2.0
    pivot_init_rad: float = 1.05  # a cold start pivots outright when the way on is this far off
    # A CURVE THAT GOES THROUGH A WALL IS NOT USED: MPPI tracks V that frame instead. A point's
    # wheels must clear by this much, or by what the start pose itself has less 5 cm. 0, not a
    # margin: a pivot beside a wall swings the tail close (4 cm on slalom) without touching.
    reject_m: float = 0.0
    taut_step_m: float = 0.25  # [m] the descent is resampled to this for the visibility graph
    n_taut: int = 96  # visibility-graph vertices at most
    rebase_m: float = 1.5  # [m] the taut string's end moving more than this = a cold start
    v_tol: float = 0.3  # [m of V] a straight may rise this far above V's descent (_taut_kernel)
    w_veto: float = 20.0  # per unit of the settle's veto at the curve's own heading (_veto_at)
    # [m] how far the settle's every-heading veto reaches past a real face on flat ground: the
    # wheels and belly cannot stand that close at any heading, so part of the body is already in
    # the forbidden region, and measuring clearance + half-width from it counted that part twice
    # (box: 0.6 m too wide). Measured on drive_sim's box and slalom, 2026-10-28: 0.34 m past the
    # box, up to 0.44 past the slabs. Subtracted from the half-width the stage measures with.
    forbidden_growth_m: float = 0.3
    # STUDY (studies/reference_paths). A cold start when the taut string leaves last frame's
    # curve, whether or not its end moved: with the end fixed (the goal in the window) a curve
    # warm-started on the other side of an obstacle was kept for good (forest, facing_wall).
    cold_on_divergence: bool = False
    # [m of clearance per m of path] 0 = off. The clearance asked at a knot is capped at the END's
    # own clearance plus this times the path length from it (not the start's: see _taper_kernel):
    # a goal 0.6 m from a wall cannot be reached at 1.2 m, and asking for it bent the curve into
    # a hook (goal_at_wall); a robot already beside a wall had every curve rejected (facing_wall).
    clear_taper: float = 0.0
    # [m of V] 0 = off. HYSTERESIS ON THE WAY ROUND: among the strictly lower neighbours within
    # this much of the lowest, the descent takes the one nearest last frame's taut string, scored
    # V + sticky_w * distance. Two ways round an obstacle that cost nearly the same swapped with
    # the map's every update (facing_wall: north, south, north -- into the wall).
    sticky_tol_m: float = 0.0
    sticky_w: float = 0.5  # [m of V per m off last frame's string]
    # [1/m] 0 = off. Stop a frame's Adam steps once a step moves no curvature knot by more than
    # this (nor the spacing or the start pivot by the like): a warm start is mostly converged.
    stop_tol: float = 0.0
    # A cold start when last frame's curve turned more than 270 deg beyond what the taut string
    # turns: a loop. The curvature profile cannot unwind one by small steps, so a warm start kept
    # a full circle beside the robot for seconds (false_door).
    cold_on_loop: bool = False
    # THE WAY ROUND FROM THE ROBOT'S OWN HEADING. False: walk down V's minimum over headings, so
    # the route is the one for the best heading at every cell, and the start's heading is the
    # curve's own business (a pivot priced at w_pivot). True: roll the lattice's own policy out
    # from the robot's cell AND heading bin -- the move each state's V was computed from -- so
    # whether to pivot, which way, or to arc round is what the cost-to-go itself decided. Needs
    # `update(policy=...)` (the solver's move costs); without them the old walk runs.
    policy_descent: bool = False
    # A cold start when the way on (the taut string's first half metre) is more than
    # `pivot_init_rad` off the warm curve's own heading where the robot stands. The cold start
    # pivots outright past that angle, but a warm one never asks again: a robot facing away from
    # its route kept a curve swinging metres out and back instead of turning (patek1 t 41-45 s).
    cold_on_repivot: bool = False
    # 0 = off. A cold start when last frame's curve was longer than this times the taut string
    # plus 1 m: a warm start polishes whatever it was given, and a curve that wandered -- a loop
    # under the 270 deg test, a swing out past the goal and back -- stays a local minimum
    # (patek1 t 41-45 s: 4-5 m loops the loop test and the re-pivot test both let through).
    cold_on_detour: float = 0.0
    # Per m^2 of each point's distance to the taut string, up to the arrival. The length term
    # prices the KNOT COUNT to arrival, an integer, so it has no gradient on the curve's shape:
    # once a curve reaches its target nothing but the tiny curvature weights shapes it, and a
    # gentle bulge drifts freely -- 1.3 m off a straight string through a gap, growing frame by
    # frame while the robot crawled (gap), the sag after a turn (uturn). 0 = off.
    w_taut: float = 0.0

    @property
    def kappa_max(self) -> float:
        return 1.0 / self.radius_min_m

    @property
    def kappa_rate_max(self) -> float:
        return self.kappa_max / self.clothoid_m


@wp.kernel
def _forbidden_mask_kernel(
    blocked: wp.array3d(dtype=float),  # [ny, nx, n_theta] the settle's veto, 1 = forbidden
    n_theta: int,
    mask: wp.array2d(dtype=float),  # [ny, nx] out: 1 = forbidden at every heading
):
    r, c = wp.tid()
    allb = float(1.0)
    for t in range(n_theta):
        if blocked[r, c, t] < 0.5:
            allb = 0.0
    mask[r, c] = allb


@wp.kernel
def _forbidden_distance_kernel(
    mask: wp.array2d(dtype=float),
    cell: float,
    reach: int,  # [cells] distances past this read as reach * cell
    out: wp.array2d(dtype=float),  # [m] centre to the nearest forbidden cell's centre
):
    r, c = wp.tid()
    rows = mask.shape[0]
    cols = mask.shape[1]
    best2 = float(reach * reach)
    for i in range(-reach, reach + 1):
        rr = r + i
        if rr >= 0 and rr < rows:
            for j in range(-reach, reach + 1):
                cc = c + j
                if cc >= 0 and cc < cols:
                    if mask[rr, cc] > 0.5:
                        best2 = wp.min(best2, float(i * i + j * j))
    out[r, c] = wp.sqrt(best2) * cell


def forbidden_distance(
    blocked: wp.array, n_theta: int, cell: float, reach_m: float, mask: wp.array, out: wp.array
) -> wp.array:
    """[m] from every routing cell to the nearest cell the settle forbids at every heading, on
    device; `mask` and `out` [ny, nx] are the caller's buffers. The stage's clearance is
    measured to these cells: a wall's, a slope's past the envelope, a ditch's alike."""
    wp.launch(
        _forbidden_mask_kernel,
        dim=out.shape,
        inputs=[blocked, int(n_theta)],
        outputs=[mask],
        device=out.device,
    )
    wp.launch(
        _forbidden_distance_kernel,
        dim=out.shape,
        inputs=[mask, float(cell), max(1, int(np.ceil(reach_m / cell)))],
        outputs=[out],
        device=out.device,
    )
    return out


@wp.func
def _min_over_theta(V: wp.array3d(dtype=float), r: int, c: int, n_theta: int) -> float:
    best = float(1.0e30)
    for t in range(n_theta):
        best = wp.min(best, V[r, c, t])
    return best


@wp.kernel
def _vmin_kernel(V: wp.array3d(dtype=float), out: wp.array3d(dtype=float)):
    """V's minimum over headings, once a frame, as a one-heading field: the descent and the
    visibility test read it per cell instead of taking 24 loads each time (0.9 ms of a frame was
    the single-thread descent's loads, A500)."""
    r, c = wp.tid()
    out[r, c, 0] = _min_over_theta(V, r, c, V.shape[2])


@wp.func
def _clear_along(
    field: wp.array2d(dtype=float), grid: Grid, a: wp.vec2, b: wp.vec2, step: float
) -> float:
    """The least wall distance along the straight a-b, sampled every `step`."""
    n = wp.max(int(wp.length(b - a) / step), 2)
    best = float(1.0e30)
    for i in range(n + 1):
        p = a + (b - a) * (float(i) / float(n))
        best = wp.min(best, sample_field(field, grid, p[0], p[1]))
    return best


@wp.kernel
def _prev_dist_kernel(
    taut: wp.array(dtype=wp.vec2),  # last frame's taut string (read before this frame's)
    n_taut: wp.array(dtype=int),
    rgrid: Grid,
    cap: float,
    out: wp.array2d(dtype=float),  # [ny, nx] distance of each cell centre to it, capped
):
    r, c = wp.tid()
    p = wp.vec2(
        rgrid.origin_x + float(c) * rgrid.cell_size, rgrid.origin_y + float(r) * rgrid.cell_size
    )
    n = n_taut[0]
    best = cap
    if n < 2:
        best = 0.0  # nothing to stick to
    for i in range(n - 1):
        a = taut[i]
        ab = taut[i + 1] - a
        L2 = wp.max(wp.dot(ab, ab), 1.0e-12)
        t = wp.clamp(wp.dot(p - a, ab) / L2, 0.0, 1.0)
        best = wp.min(best, wp.length(p - (a + t * ab)))
    out[r, c] = best


@wp.kernel
def _descent_kernel(
    V: wp.array3d(dtype=float),  # [ny, nx, n_theta] the route field, in this frame's window
    rgrid: Grid,  # the grid V lives on, expressed in the MPPI window's frame
    n_theta: int,
    start: wp.array(dtype=float),  # [3] x, y, yaw in the MPPI window's frame
    goal: wp.array(dtype=float),  # [3] x, y in the same frame, and 1 when it is in the window
    clear_field: wp.array2d(dtype=float),  # wall distance, covering the routing window
    cgrid: Grid,
    half_width: float,
    max_pts: int,
    prev_dist: wp.array2d(dtype=float),  # [ny, nx] off last frame's string (_prev_dist_kernel)
    sticky_tol: float,  # 0 = off (ReferencePathParams.sticky_tol_m)
    sticky_w: float,
    out: wp.array(dtype=wp.vec2),  # [max_pts] cell centres, the robot's first
    n_out: wp.array(dtype=int),  # [1]
):
    """Greedy descent of min-over-heading V through the 8 neighbours. Never up: a strictly lower
    neighbour is taken when there is one, and an EQUAL one -- the lattice's field comes in flat
    steps, and the box's walk stopped on one 1 m from the robot, leaving MPPI no curve (2026-10-28)
    -- is taken only to cross such a step (within `flat_tol`), straight on as far as the grid
    allows, never back to the cell just left, and for at most `max_flat` cells without a new low.
    A tolerance for real climbing was tried and removed: on slalom it walked along a wall into a
    dead end."""
    cell = rgrid.cell_size
    c = wp.clamp(int(wp.floor((start[0] - rgrid.origin_x) / cell + 0.5)), 0, rgrid.cells_x - 1)
    r = wp.clamp(int(wp.floor((start[1] - rgrid.origin_y) / cell + 0.5)), 0, rgrid.cells_y - 1)
    out[0] = wp.vec2(start[0], start[1])
    n = int(1)
    cur = _min_over_theta(V, r, c, n_theta)
    pr = int(-1)
    pc = int(-1)
    ddr = int(0)  # the last step, for crossing a flat step straight on
    ddc = int(0)
    vmin = cur
    n_at_min = int(1)
    max_flat = int(15)
    # [m of V] the field's steps are not flat to the bit: the box's walk stopped in a dent 0.0006
    # deep. 2 cm a step, 15 steps at most, is 0.3 m; a wall raises V by metres
    flat_tol = float(0.02)
    for _ in range(max_pts - 1):
        best = cur
        br = r
        bc = c
        eq_r = int(-1)
        eq_c = int(-1)
        eq_score = int(-100)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr = r + dr
                cc = c + dc
                if (dr != 0 or dc != 0) and rr >= 0 and rr < rgrid.cells_y:
                    if cc >= 0 and cc < rgrid.cells_x and not (rr == pr and cc == pc):
                        v = _min_over_theta(V, rr, cc, n_theta)
                        if v < best - 1.0e-6:
                            best = v
                            br = rr
                            bc = cc
                        elif wp.abs(v - cur) <= flat_tol:
                            score = dr * ddr + dc * ddc  # straight on scores highest
                            if score > eq_score:
                                eq_score = score
                                eq_r = rr
                                eq_c = cc
        if sticky_tol > 0.0 and (br != r or bc != c):
            # among the lower neighbours near the lowest, the one nearest last frame's way
            sbest = best + sticky_w * prev_dist[br, bc]
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    rr = r + dr
                    cc = c + dc
                    if (dr != 0 or dc != 0) and rr >= 0 and rr < rgrid.cells_y:
                        if cc >= 0 and cc < rgrid.cells_x and not (rr == pr and cc == pc):
                            v = _min_over_theta(V, rr, cc, n_theta)
                            if v < cur - 1.0e-6 and v <= best + sticky_tol:
                                sc = v + sticky_w * prev_dist[rr, cc]
                                if sc < sbest:
                                    sbest = sc
                                    br = rr
                                    bc = cc
        if br == r and bc == c:
            if eq_r < 0 or cur >= 80.0:
                break
            br = eq_r
            bc = eq_c
        if sticky_tol > 0.0:  # the sticky choice may not be the lowest
            best = _min_over_theta(V, br, bc, n_theta)
        if best >= 80.0:  # the cap: a vetoed cell is not on the way
            break
        ddr = br - r
        ddc = bc - c
        pr = r
        pc = c
        r = br
        c = bc
        cur = _min_over_theta(V, r, c, n_theta)
        out[n] = wp.vec2(rgrid.origin_x + float(c) * cell, rgrid.origin_y + float(r) * cell)
        n += 1
        # progress is a new low; `max_flat` cells without one end the walk, and it is cut back to
        # the last low -- a flat region can otherwise be wandered in circles (box, 256 cells)
        if cur < vmin - 1.0e-6:
            vmin = cur
            n_at_min = n
        elif n - n_at_min >= max_flat:
            break
    n = n_at_min
    # The goal itself ends the path when it is in the window and the wheels clear every wall on
    # the straight from where the walk stopped: the field's heading minimum has a basin a few
    # metres short of the goal (the approach costs a heading the lattice prices), and a curve
    # ending in that basin left MPPI a stub for the last metres and a hook at the goal (box), or
    # no curve at all a gate short of it (slalom, 2026-10-28).
    if goal[2] > 0.5 and n < max_pts:
        g = wp.vec2(goal[0], goal[1])
        if wp.length(out[n - 1] - g) >= 0.05:
            if _clear_along(clear_field, cgrid, out[n - 1], g, 0.05) >= half_width + 0.05:
                out[n] = g
                n += 1
    n_out[0] = n


@wp.kernel
def _policy_kernel(
    V: wp.array3d(dtype=float),  # [ny, nx, n_theta] the route field (clamped at its cap)
    edge_cost: wp.array4d(dtype=float),  # [ny, nx, n_theta, n_prim] the solver's move costs
    prim_dr: wp.array2d(dtype=int),
    prim_dc: wp.array2d(dtype=int),
    prim_heading: wp.array2d(dtype=int),
    n_prim: int,
    rgrid: Grid,
    start: wp.array(dtype=float),  # [3]
    goal: wp.array(dtype=float),  # [3]
    clear_field: wp.array2d(dtype=float),
    cgrid: Grid,
    half_width: float,
    max_pts: int,
    out: wp.array(dtype=wp.vec2),  # [max_pts] in/out: kept as the greedy walk left it on failure
    n_out: wp.array(dtype=int),  # [1] in/out
):
    """The lattice policy from the robot's state: at each state the move whose cost plus the
    next state's V is least -- exactly V's own minimiser -- until no move improves on V (a seed:
    the goal, or the coarse layer's ring at the window's edge). Moves that stay in the cell
    (pivots) leave no point. When the robot's own state has no route (V capped), the greedy walk
    the descent kernel already wrote stands."""
    cell = rgrid.cell_size
    nt = V.shape[2]
    c = wp.clamp(int(wp.floor((start[0] - rgrid.origin_x) / cell + 0.5)), 0, rgrid.cells_x - 1)
    r = wp.clamp(int(wp.floor((start[1] - rgrid.origin_y) / cell + 0.5)), 0, rgrid.cells_y - 1)
    yaw = start[2] - wp.floor(start[2] / 6.2831853) * 6.2831853
    t = int(wp.floor(yaw / (6.2831853 / float(nt)) + 0.5)) % nt
    if V[r, c, t] >= 80.0:
        return
    out[0] = wp.vec2(start[0], start[1])
    n = int(1)
    for _ in range(4 * max_pts):
        cur = V[r, c, t]
        best = float(1.0e30)
        bp = int(-1)
        for p in range(n_prim):
            e = edge_cost[r, c, t, p]
            if e < 1.0e29:
                val = e + V[r + prim_dr[t, p], c + prim_dc[t, p], prim_heading[t, p]]
                if val < best:
                    best = val
                    bp = p
        if bp < 0 or best > cur + 1.0e-4 or best >= 80.0:
            break  # a seed: nothing beats standing here
        nr = r + prim_dr[t, bp]
        nc = c + prim_dc[t, bp]
        t = prim_heading[t, bp]
        if nr != r or nc != c:
            r = nr
            c = nc
            if n < max_pts:
                out[n] = wp.vec2(rgrid.origin_x + float(c) * cell, rgrid.origin_y + float(r) * cell)
                n += 1
    if goal[2] > 0.5 and n < max_pts:
        g = wp.vec2(goal[0], goal[1])
        if wp.length(out[n - 1] - g) >= 0.05:
            if _clear_along(clear_field, cgrid, out[n - 1], g, 0.05) >= half_width + 0.05:
                out[n] = g
                n += 1
    if n >= 2:
        n_out[0] = n


@wp.func
def _v_at(V: wp.array3d(dtype=float), grid: Grid, n_theta: int, p: wp.vec2) -> float:
    c = wp.clamp(int(wp.floor((p[0] - grid.origin_x) / grid.cell_size + 0.5)), 0, grid.cells_x - 1)
    r = wp.clamp(int(wp.floor((p[1] - grid.origin_y) / grid.cell_size + 0.5)), 0, grid.cells_y - 1)
    return _min_over_theta(V, r, c, n_theta)


@wp.func
def _v_excess(
    V: wp.array3d(dtype=float), grid: Grid, n_theta: int, a: wp.vec2, b: wp.vec2, step: float
) -> float:
    """The most the heading-minimum of V along the straight a-b rises above the line between
    its ends: 0 on ground V crosses evenly, the extra cost-to-go where the straight crosses
    ground V priced and went around."""
    va = _v_at(V, grid, n_theta, a)
    vb = _v_at(V, grid, n_theta, b)
    n = wp.max(int(wp.length(b - a) / step), 2)
    worst = float(0.0)
    for i in range(n + 1):
        f = float(i) / float(n)
        worst = wp.max(worst, _v_at(V, grid, n_theta, a + (b - a) * f) - (va + (vb - va) * f))
    return worst


@wp.kernel
def _resample_kernel(
    desc: wp.array(dtype=wp.vec2),
    n_desc: wp.array(dtype=int),
    resample: float,  # [m] vertex spacing of the visibility graph
    max_v: int,
    verts: wp.array(dtype=wp.vec2),  # [max_v] out: the resampled descent
    n_verts: wp.array(dtype=int),  # [1] out
):
    """The descent's staircase resampled to `resample` spacing, first and last point kept: the
    vertices of the visibility graph `_taut_kernel` searches."""
    n = n_desc[0]
    m = int(1)
    verts[0] = desc[0]
    acc = float(0.0)
    for i in range(1, n):
        acc += wp.length(desc[i] - desc[i - 1])
        if acc >= resample and m < max_v - 1:
            verts[m] = desc[i]
            m += 1
            acc = 0.0
    if n > 1 and m < max_v and wp.length(desc[n - 1] - verts[m - 1]) > 1.0e-6:
        verts[m] = desc[n - 1]
        m += 1
    n_verts[0] = m


@wp.kernel
def _visible_kernel(
    verts: wp.array(dtype=wp.vec2),
    n_verts: wp.array(dtype=int),
    clear_field: wp.array2d(dtype=float),  # [ny, nx] distance to forbidden on `sgrid`
    sgrid: Grid,
    V: wp.array3d(dtype=float),  # [ny, nx, n_theta] the route field, on `rgrid`
    rgrid: Grid,
    n_theta: int,
    v_tol: float,  # [m of V] how far a straight may run above V's own descent
    offset: float,  # [m] clearance + half-width
    vis: wp.array2d(dtype=int),  # [max_v, max_v] out: 1 = the straight u-j may be taken
):
    """Every straight of the visibility graph at once, one thread a pair: it was the inner loop of
    a single-thread Dijkstra and 6.8 ms of the frame (2026-10-29). A straight must keep `offset`
    from forbidden cells AND cost no more in V's own terms than the descent it replaces;
    consecutive points are always joined."""
    u, j = wp.tid()
    m = n_verts[0]
    ok = int(0)
    if u < m and j < m and j > u:
        if j == u + 1:
            ok = 1
        elif _clear_along(clear_field, sgrid, verts[u], verts[j], 0.05) >= offset:
            if _v_excess(V, rgrid, n_theta, verts[u], verts[j], 0.08) <= v_tol:
                ok = 1
    vis[u, j] = ok


@wp.kernel
def _taut_kernel(
    verts: wp.array(dtype=wp.vec2),
    n_verts: wp.array(dtype=int),
    vis: wp.array2d(dtype=int),  # [max_v, max_v] from `_visible_kernel`
    best: wp.array(dtype=float),  # [max_v] work
    prev: wp.array(dtype=int),  # [max_v] work
    done: wp.array(dtype=int),  # [max_v] work
    taut: wp.array(dtype=wp.vec2),  # [max_v] out, start to end
    n_taut: wp.array(dtype=int),  # [1]
):
    """The shortest polyline through the descent's points over the straights `_visible_kernel`
    allowed, by Dijkstra. On flat ground V falls linearly along a straight and every clear
    straight passes; across a bump V routed around, the cost-to-go of the cells the straight
    crosses rises above the line between its ends."""
    m = n_verts[0]
    for i in range(m):
        best[i] = 1.0e30
        prev[i] = -1
        done[i] = 0
    best[0] = 0.0
    for _ in range(m):
        # the unsettled vertex with the least cost
        u = int(-1)
        ub = float(1.0e30)
        for i in range(m):
            if done[i] == 0 and best[i] < ub:
                ub = best[i]
                u = i
        if u < 0:
            break
        done[u] = 1
        if u == m - 1:
            break
        for j in range(u + 1, m):
            d = wp.length(verts[j] - verts[u])
            if vis[u, j] == 1:
                if ub + d < best[j]:
                    best[j] = ub + d
                    prev[j] = u
    # unwind, then reverse into `taut`
    k = int(0)
    i = m - 1
    for _ in range(m):
        done[k] = i  # reuse `done` as the reversed chain
        k += 1
        if i == 0:
            break
        i = prev[i]
    for q in range(k):
        taut[q] = verts[done[k - 1 - q]]
    n_taut[0] = k


@wp.func
def _wrap(a: float) -> float:
    return a - 6.283185307 * wp.floor((a + 3.14159265) / 6.283185307)


@wp.func
def _field_grad(field: wp.array2d(dtype=float), grid: Grid, p: wp.vec2, h: float) -> wp.vec2:
    gx = (sample_field(field, grid, p[0] + h, p[1]) - sample_field(field, grid, p[0] - h, p[1])) / (
        2.0 * h
    )
    gy = (sample_field(field, grid, p[0], p[1] + h) - sample_field(field, grid, p[0], p[1] - h)) / (
        2.0 * h
    )
    return wp.vec2(gx, gy)


@wp.func
def _room_at(clear_field: wp.array2d(dtype=float), sgrid: Grid, p: wp.vec2, offset: float) -> float:
    """The room: the largest wall distance on three rings around the point."""
    room = sample_field(clear_field, sgrid, p[0], p[1])
    for ring in range(1, 4):
        rad = offset * float(ring) / 3.0
        for a in range(12):
            ang = 0.5235988 * float(a)
            room = wp.max(
                room,
                sample_field(
                    clear_field, sgrid, p[0] + rad * wp.cos(ang), p[1] + rad * wp.sin(ang)
                ),
            )
    return room


@wp.kernel
def _room_kernel(
    samp_p: wp.array(dtype=wp.vec2),
    samp_q: wp.array(dtype=int),
    n_samp: wp.array(dtype=int),
    clear_field: wp.array2d(dtype=float),
    sgrid: Grid,
    offset: float,
    clear_target: wp.array(dtype=float),  # [N] in/out: lowered to each sample's room
):
    """`_targets_kernel`'s room test, one thread a sample: it was 37 field reads a sample in a
    single thread, 0.6 ms of a frame (A500). A min is order-free, so the result is the same."""
    k = wp.tid()
    if k < n_samp[0]:
        room = _room_at(clear_field, sgrid, samp_p[k], offset)
        wp.atomic_min(clear_target, samp_q[k], wp.min(offset, room - 0.01))


@wp.kernel
def _taper_kernel(
    taut: wp.array(dtype=wp.vec2),
    n_taut: wp.array(dtype=int),
    target: wp.array(dtype=wp.vec2),
    t_len: wp.array(dtype=float),  # [1] the taut string's length
    clear_field: wp.array2d(dtype=float),
    sgrid: Grid,
    ds0: float,
    n_knots: int,
    taper: float,
    clear_target: wp.array(dtype=float),  # [N] in/out
):
    """ReferencePathParams.clear_taper: cap the clearance asked by the end's own plus the taper
    times the distance from it -- toward the END only. A goal beside a wall must be reachable;
    from the start the cap let the curve keep the robot against the wall it started beside,
    which with w_taut (a pull to a string that starts there) halved the 2.6 m corridor (9/10 ->
    5/10 reached, robot-faithful drive_sim)."""
    if n_taut[0] < 2:
        return
    end = target[0]
    c1 = sample_field(clear_field, sgrid, end[0], end[1])
    for q in range(n_knots):
        s_q = (float(q) + 0.5) * ds0
        cap = c1 + taper * wp.max(t_len[0] - s_q, 0.0)
        clear_target[q] = wp.min(clear_target[q], cap)


@wp.kernel
def _targets_kernel(
    taut: wp.array(dtype=wp.vec2),
    n_taut: wp.array(dtype=int),
    goal: wp.array(dtype=float),  # [3] x, y, in-window flag
    clear_field: wp.array2d(dtype=float),
    sgrid: Grid,
    offset: float,
    ds0: float,
    n_knots: int,
    target: wp.array(dtype=wp.vec2),  # [1] out: the taut string's end, moved out to the offset
    taut_at: wp.array(dtype=wp.vec2),  # [N] out: the taut point at each knot's arc length
    clear_target: wp.array(dtype=float),  # [N] out: the clearance asked at each knot
    samp_p: wp.array(dtype=wp.vec2),  # [S] out: the points whose room `_room_kernel` tests
    samp_q: wp.array(dtype=int),  # [S] out: the knot each one lowers
    n_samp: wp.array(dtype=int),  # [1] out
    t_len: wp.array(dtype=float),  # [1] out: the taut string's length
):
    """Every frame: the end to reach, the taut point under each knot (the fallback pull where a
    point has strayed inside a wall, where the distance map has no gradient), and the clearance
    to ask. The descent of V hugs walls at the route's own margin, well inside the offset, so the
    taut string's clearance says nothing about how much room there is; the room is the largest
    wall distance within `offset` of the taut point, and the target is the offset or that room,
    whichever is less: a door narrower than the offset is passed through its middle."""
    n = n_taut[0]
    for q in range(n_knots):
        clear_target[q] = offset
        taut_at[q] = taut[wp.max(n - 1, 0)]
    end = taut[wp.max(n - 1, 0)]
    if goal[2] < 0.5:
        # The descent leaves the window where the route hugs a corner at the route's own margin,
        # and the curve, anchored to that point, hugged the corner with it however hard the
        # clearance term pushed the rest (slalom, 2026-10-28). Walk the end up the distance map
        # to the offset; the goal itself, when it is the end, stays where it is.
        h = 0.5 * sgrid.cell_size
        for _ in range(16):
            if sample_field(clear_field, sgrid, end[0], end[1]) >= offset:
                break
            gr = _field_grad(clear_field, sgrid, end, h)
            if wp.length_sq(gr) < 1.0e-6:
                break
            end = end + 0.1 * wp.normalize(gr)
    target[0] = end
    n_samp[0] = 0
    t_len[0] = 0.0
    if n < 2:
        return
    max_s = samp_p.shape[0]
    k = int(0)
    s_c = float(0.0)
    for i in range(n - 1):
        seg = taut[i + 1] - taut[i]
        length = wp.length(seg)
        ns = wp.max(int(length / (0.5 * ds0)), 1)
        for j in range(ns + 1):
            p = taut[i] + seg * (float(j) / float(ns))
            s_here = s_c + length * float(j) / float(ns)
            q = wp.clamp(int(s_here / ds0), 0, n_knots - 1)
            if j == 0 or q != wp.clamp(int((s_here - length / float(ns)) / ds0), 0, n_knots - 1):
                taut_at[q] = p
            if k < max_s:  # the room is tested in parallel (_room_kernel)
                samp_p[k] = p
                samp_q[k] = q
                k += 1
            else:  # past the buffer: here, as before
                room = _room_at(clear_field, sgrid, p, offset)
                clear_target[q] = wp.min(clear_target[q], wp.min(offset, room - 0.01))
        s_c += length
    n_samp[0] = k
    t_len[0] = s_c


@wp.func
def _taut_heading_at(
    taut: wp.array(dtype=wp.vec2), n: int, start_heading: float, s: float
) -> float:
    """The taut string's heading at arc length `s`, unwrapped from the start heading (before the
    string it is the start heading, past its end the last leg's)."""
    heading = start_heading
    acc = float(0.0)
    for i in range(n - 1):
        if s < acc:  # before this leg: the heading so far (the start's, before the string)
            return heading
        seg = taut[i + 1] - taut[i]
        length = wp.length(seg)
        if length > 1.0e-6:  # a repeated vertex has no heading (atan2(0, 0) would read east)
            heading += _wrap(wp.atan2(seg[1], seg[0]) - heading)
        if s < acc + length:
            return heading
        acc += length
    return heading


@wp.kernel
def _mode_kernel(
    taut: wp.array(dtype=wp.vec2),
    n_taut: wp.array(dtype=int),
    pts: wp.array(dtype=wp.vec2),  # [N+1] last frame's curve
    n_valid: wp.array(dtype=int),  # [1] last frame's: 0 = none, or it was rejected
    warm: wp.array(dtype=int),  # [1] the caller allows a warm start
    rebase_m: float,
    taut_step: float,
    iterations: int,
    iterations_cold: int,
    any_end: int,  # 1: test the divergence every frame, not only when the end moved
    th: wp.array(dtype=float),  # [N] last frame's heading at each knot
    loop_check: int,  # 1: a curve turning 270 deg more than the string is a cold start
    start: wp.array(dtype=float),  # [3] the robot now
    repivot: float,  # [rad] > 0: the way on this far off the warm curve's heading is cold
    detour: float,  # > 0: last frame's curve longer than this x the string + 1 m is cold
    last_end: wp.array(dtype=wp.vec2),  # [1] in/out: the taut string's end last frame
    cold: wp.array(dtype=int),  # [1] out: 1 cold, 0 warm, -1 no string (no curve this frame)
    limit: wp.array(dtype=int),  # [1] out: Adam steps this frame
):
    """Warm or cold, decided on the device so a frame is one graph with no host read mid-way.

    A cold start is a NEW WAY OUT, not a moved end: the end runs ahead along the same corridor
    as the window scrolls, and restarting from the taut string then threw away a converged curve
    and paid four times the steps for it (false_door, 2026-10-29: 5 % of frames at 54-60 ms). So
    the end moving past `rebase_m` is cold only when the string has also left last frame's curve
    somewhere along that curve's length."""
    n = n_taut[0]
    if n < 2:
        cold[0] = -1
        limit[0] = 0
        return
    end = taut[n - 1]
    nv = n_valid[0]
    c = int(0)
    if warm[0] == 0 or nv < 2:
        c = 1
    if c == 0 and loop_check == 1:
        turned = float(0.0)
        for j in range(wp.min(nv - 1, th.shape[0]) - 1):
            turned += wp.abs(th[j + 1] - th[j])
        bends = float(0.0)
        h_prev = float(0.0)
        first = int(1)
        for i in range(n - 1):
            seg = taut[i + 1] - taut[i]
            if wp.length(seg) > 1.0e-6:
                h = wp.atan2(seg[1], seg[0])
                if first == 0:
                    bends += wp.abs(_wrap(h - h_prev))
                h_prev = h
                first = 0
        if turned > bends + 4.712389:
            c = 1
    if c == 0 and detour > 0.0:
        lc = float(0.0)
        for j in range(nv - 1):
            lc += wp.length(pts[j + 1] - pts[j])
        lt = float(0.0)
        for i in range(n - 1):
            lt += wp.length(taut[i + 1] - taut[i])
        if lc > detour * lt + 1.0:
            c = 1
    if c == 0 and repivot > 0.0:
        jb = int(0)
        bd = float(1.0e30)
        for j in range(nv):
            dd = wp.length(pts[j] - wp.vec2(start[0], start[1]))
            if dd < bd:
                bd = dd
                jb = j
        here = th[wp.min(jb, th.shape[0] - 1)]
        way = _taut_heading_at(taut, n, here, 0.5)
        if wp.abs(_wrap(way - here)) > repivot:
            c = 1
    if c == 1:
        pass
    elif any_end == 1 or wp.length(end - last_end[0]) > rebase_m:
        prev_len = float(0.0)
        for j in range(nv - 1):
            prev_len += wp.length(pts[j + 1] - pts[j])
        s = float(0.0)
        for i in range(n - 1):
            seg = taut[i + 1] - taut[i]
            length = wp.length(seg)
            ns = wp.max(int(length / taut_step), 1)
            for k in range(ns):
                if s + length * float(k) / float(ns) <= prev_len:
                    p = taut[i] + seg * (float(k) / float(ns))
                    near = float(1.0e30)
                    for j in range(nv):
                        near = wp.min(near, wp.length(pts[j] - p))
                    if near > rebase_m:
                        c = 1
            s += length
    last_end[0] = end
    cold[0] = c
    lim = iterations
    if c == 1:
        lim = iterations_cold
    limit[0] = lim


@wp.kernel
def _init_kernel(
    cold: wp.array(dtype=int),  # [1] runs only on a cold start
    taut: wp.array(dtype=wp.vec2),
    n_taut: wp.array(dtype=int),
    start: wp.array(dtype=float),  # [3]
    kappa_max: float,
    kappa_rate: float,
    clothoid: float,  # [m] the window the taut string's heading is differentiated over
    ds0: float,
    n_knots: int,
    pivot_init: float,
    kappa: wp.array(dtype=float),  # [N] out
    ds: wp.array(dtype=float),  # [1] out
    phi: wp.array(dtype=float),  # [1] out: the turn in place at the start
    m: wp.array(dtype=float),  # [N+2] Adam state, zeroed
    v: wp.array(dtype=float),  # [N+2]
    it: wp.array(dtype=int),  # [1]
):
    """Cold start: when the way on is more than `pivot_init` off the robot's heading, the curve
    starts with that turn in place. Then the taut string's heading by arc length, differentiated over the clothoid
    length, so each corner becomes a turn spread over that length with the sign the string
    actually turns. One profile per corner, placed and summed, cancelled where the string is a
    staircase of small corners near the robot: the first batch's curves set off the wrong way
    and looped (box, slalom, 2026-10-28)."""
    if cold[0] != 1:
        return
    n = n_taut[0]
    for i in range(n_knots):
        kappa[i] = 0.0
    for i in range(n_knots + 2):
        m[i] = 0.0
        v[i] = 0.0
    it[0] = 0
    ds[0] = ds0
    phi[0] = 0.0
    if n < 2:
        return
    first = _taut_heading_at(taut, n, start[2], 0.5)
    h0 = start[2]
    if wp.abs(first - start[2]) > pivot_init:
        phi[0] = first - start[2]
        h0 = first
    w = 0.5 * clothoid
    for q in range(n_knots):
        s_q = (float(q) + 0.5) * ds0
        ahead = _taut_heading_at(taut, n, h0, s_q + w)
        behind = _taut_heading_at(taut, n, h0, s_q - w)
        kappa[q] = (ahead - behind) / (2.0 * w)
    r = kappa_rate * ds0
    for q in range(n_knots):
        kappa[q] = wp.clamp(kappa[q], -kappa_max, kappa_max)
    for q in range(1, n_knots):
        kappa[q] = wp.clamp(kappa[q], kappa[q - 1] - r, kappa[q - 1] + r)
    for q in range(n_knots - 2, -1, -1):
        kappa[q] = wp.clamp(kappa[q], kappa[q + 1] - r, kappa[q + 1] + r)


@wp.kernel
def _rebase_kernel(
    cold: wp.array(dtype=int),  # [1] runs only on a warm start
    pts: wp.array(dtype=wp.vec2),  # [N+1] last frame's curve
    th: wp.array(dtype=float),  # [N] its heading at each knot
    n_valid: wp.array(dtype=int),
    start: wp.array(dtype=float),  # [3] the robot now
    n_knots: int,
    kappa: wp.array(dtype=float),  # [N] in/out: shifted past the robot
    phi: wp.array(dtype=float),  # [1] out: the curve's heading there, less the robot's
    m: wp.array(dtype=float),
    v: wp.array(dtype=float),
):
    """Warm start: drop the knots the robot has passed, so the profile continues from where it
    stands, and start it at last frame's heading there (a pivot still to be made stays one);
    Adam's moments shift with it."""
    if cold[0] != 0:
        return
    n = n_valid[0]
    best = float(1.0e30)
    jb = int(0)
    for j in range(n):
        d = wp.length(pts[j] - wp.vec2(start[0], start[1]))
        if d < best:
            best = d
            jb = j
    phi[0] = _wrap(th[wp.min(jb, n_knots - 1)] - start[2])
    if jb == 0:
        return
    for q in range(n_knots):
        src = q + jb
        if src < n_knots:
            kappa[q] = kappa[src]
            m[q] = m[src]
            v[q] = v[src]
        else:
            kappa[q] = 0.0
            m[q] = 0.0
            v[q] = 0.0


@wp.func
def _veto_at(
    blocked: wp.array3d(dtype=float), grid: Grid, n_theta: int, p: wp.vec2, th: float
) -> float:
    """The settle's veto at a position AND a heading, interpolated: bilinear in the cells, linear
    between the two nearest heading bins. Off-road a cell is often passable only along a few
    headings (bumpy's mound at x 7-8: blocked at 54-71 % of them), and a curve that crosses it
    off its channel high-centred the robot (2026-10-28)."""
    two_pi = 6.2831853
    fx = (p[0] - grid.origin_x) / grid.cell_size
    fy = (p[1] - grid.origin_y) / grid.cell_size
    xi = wp.clamp(int(wp.floor(fx)), 0, grid.cells_x - 2)
    yi = wp.clamp(int(wp.floor(fy)), 0, grid.cells_y - 2)
    ax = wp.clamp(fx - float(xi), 0.0, 1.0)
    ay = wp.clamp(fy - float(yi), 0.0, 1.0)
    m = th - wp.floor(th / two_pi) * two_pi
    ft = m / (two_pi / float(n_theta))
    t0 = int(wp.floor(ft)) % n_theta
    t1 = (t0 + 1) % n_theta
    at = ft - wp.floor(ft)
    v0 = (
        (1.0 - ax) * (1.0 - ay) * blocked[yi, xi, t0]
        + ax * (1.0 - ay) * blocked[yi, xi + 1, t0]
        + (1.0 - ax) * ay * blocked[yi + 1, xi, t0]
        + ax * ay * blocked[yi + 1, xi + 1, t0]
    )
    v1 = (
        (1.0 - ax) * (1.0 - ay) * blocked[yi, xi, t1]
        + ax * (1.0 - ay) * blocked[yi, xi + 1, t1]
        + (1.0 - ax) * ay * blocked[yi + 1, xi, t1]
        + ax * ay * blocked[yi + 1, xi + 1, t1]
    )
    return (1.0 - at) * v0 + at * v1


@wp.func
def _clearance_grad_at(
    i: int,
    p: wp.vec2,
    th_i: float,
    ct: float,  # the clearance asked at this knot
    taut_i: wp.vec2,  # the taut point under it
    cut: int,  # points past it are the unpriced tail
    clear_field: wp.array2d(dtype=float),
    sgrid: Grid,
    blocked: wp.array3d(dtype=float),  # [ny, nx, n_theta] the settle's veto on `rgrid`
    rgrid: Grid,
    n_theta: int,
    w_veto: float,
    rear: float,
    w_clear: float,
    n_knots: int,
) -> wp.vec3:
    """dJ/dp (x, y) and dJ/dtheta of point `i`'s clearance and veto terms: the clearance deficit of
    the front disc at the point and of the rear disc `rear` behind it along the heading, each
    squared and weighted, and the settle's veto at the point's own heading. A point inside a
    wall, where the distance map is flat at zero, is pulled toward the taut string instead."""
    h = 0.5 * sgrid.cell_size
    g = wp.vec2(0.0, 0.0)
    gt = float(0.0)
    c = sample_field(clear_field, sgrid, p[0], p[1]) - ct
    if i > cut:
        c = 0.0  # the tail
    if c < 0.0:
        grad = _field_grad(clear_field, sgrid, p, h)
        if wp.length_sq(grad) < 1.0e-6:
            # the distance map is flat inside a wall: take the direction to the taut string as
            # the direction the distance grows in, so the descent step leaves the wall
            grad = wp.normalize(taut_i - p + wp.vec2(1.0e-6, 0.0))
        g = g + (2.0 * w_clear * c) * grad
    if i < n_knots and i <= cut:
        tv = wp.vec2(wp.cos(th_i), wp.sin(th_i))
        q = p - rear * tv
        cq = sample_field(clear_field, sgrid, q[0], q[1]) - ct
        if cq < 0.0:
            gradq = _field_grad(clear_field, sgrid, q, h)
            if wp.length_sq(gradq) < 1.0e-6:
                gradq = wp.normalize(taut_i - q + wp.vec2(1.0e-6, 0.0))
            gq = (2.0 * w_clear * cq) * gradq
            g = g + gq
            gt = -rear * (-wp.sin(th_i) * gq[0] + wp.cos(th_i) * gq[1])
    if i >= 1 and i < n_knots and i <= cut and w_veto > 0.0:
        # the veto at the point's own heading, by central differences in x, y and heading
        hx = 0.5 * rgrid.cell_size
        hth = 3.14159265 / float(n_theta)
        bx = _veto_at(blocked, rgrid, n_theta, p + wp.vec2(hx, 0.0), th_i)
        bx -= _veto_at(blocked, rgrid, n_theta, p - wp.vec2(hx, 0.0), th_i)
        by = _veto_at(blocked, rgrid, n_theta, p + wp.vec2(0.0, hx), th_i)
        by -= _veto_at(blocked, rgrid, n_theta, p - wp.vec2(0.0, hx), th_i)
        bt = _veto_at(blocked, rgrid, n_theta, p, th_i + hth)
        bt -= _veto_at(blocked, rgrid, n_theta, p, th_i - hth)
        g = g + w_veto * wp.vec2(bx / (2.0 * hx), by / (2.0 * hx))
        gt += w_veto * bt / (2.0 * hth)
    return wp.vec3(g[0], g[1], gt)


@wp.func
def _sync(lane: int) -> float:
    """A barrier for the block: every tile collective synchronises it, so a tiny reduction is
    the one Warp spells. Used where one lane writes what the others read next."""
    v = float(lane)
    return wp.tile_sum(wp.tile(v))[0]


@wp.kernel
def _optimise_kernel(
    start: wp.array(dtype=float),  # [3]
    target: wp.array(dtype=wp.vec2),  # [1]
    clear_target: wp.array(dtype=float),  # [N]
    taut_at: wp.array(dtype=wp.vec2),  # [N]
    clear_field: wp.array2d(dtype=float),
    sgrid: Grid,
    blocked: wp.array3d(dtype=float),  # [ny, nx, n_theta] the settle's veto on `rgrid`
    rgrid: Grid,
    n_theta: int,
    limit: wp.array(dtype=int),  # [1] Adam steps this frame (`_mode_kernel`)
    kappa_max: float,
    kappa_rate: float,
    ds0: float,
    step: float,
    w_end: float,
    w_clear: float,
    w_k: float,
    w_dk: float,
    w_len: float,
    w_pivot: float,
    w_veto: float,
    rear: float,
    n_knots: int,
    taut: wp.array(dtype=wp.vec2),  # the taut string, for `w_taut`
    n_taut: wp.array(dtype=int),
    w_taut: float,
    kappa: wp.array(dtype=float),  # [N] in/out
    ds: wp.array(dtype=float),  # [1] in/out
    phi: wp.array(dtype=float),  # [1] in/out
    m: wp.array(dtype=float),  # [N+2] in/out: Adam state (kappa..., ds, phi)
    v: wp.array(dtype=float),  # [N+2]
    it: wp.array(dtype=int),  # [1]
    th: wp.array(dtype=float),  # [N] out: the final profile's heading at each knot
    scratch: wp.array(dtype=float),  # [BLOCK] work: the rate projection, lane 0 to the rest
    stop_tol: float,  # 0 = run all `limit` steps
    used: wp.array(dtype=int),  # [1] out: the steps this frame took
):
    """Every Adam step of the frame in one launch, one block, one lane per point.

    Each step: integrate the profile (the headings and points are prefix sums), cut it at the
    first arrival at the target, the clearance and veto gradient of every point at once, the
    adjoint through the integration (suffix sums), one Adam step on the profile, the spacing and
    the start pivot, and the projection onto the bounds. The steps used to be three launches
    each, two of them a single thread walking the knots through global memory: 105 us a step,
    up to 54 ms a frame (2026-10-29).

    The curvature gradient reads both neighbours' values from before the step. The single-thread
    version walked the knots down in place and so read the upper neighbour's updated value;
    every knot now steps from the same profile, which is the gradient the objective defines."""
    _b, i = wp.tid()
    N = n_knots
    knot = i < N
    on = i <= N  # a point of the curve: the knots' starts and the end
    q_i = wp.min(i, N - 1)
    k_i = float(0.0)
    m_i = float(0.0)
    v_i = float(0.0)
    if knot:
        k_i = kappa[i]
        m_i = m[i]
        v_i = v[i]
    ct = clear_target[q_i]
    taut_i = taut_at[q_i]
    tgt = target[0]
    p0 = wp.vec2(start[0], start[1])
    d = ds[0]
    ph = phi[0]
    m_ds = m[N]
    v_ds = v[N]
    m_ph = m[N + 1]
    v_ph = v[N + 1]
    n_it = it[0]
    n_used = int(0)
    for _ in range(limit[0]):
        k_old = k_i
        d_old = d
        ph_old = ph
        n_used += 1
        # the curve: theta_i = theta_0 + d sum_{j<i} kappa_j, p_i = p_0 + d sum_{j<i} t(theta_j)
        kk = wp.where(knot, k_i, 0.0)
        cum = wp.tile_scan_exclusive(wp.tile(kk))
        cum_i = cum[i]
        th_i = start[2] + ph + d * cum_i
        cx = wp.where(knot, wp.cos(th_i), 0.0)
        cy = wp.where(knot, wp.sin(th_i), 0.0)
        sx = wp.tile_scan_exclusive(wp.tile(cx))
        sy = wp.tile_scan_exclusive(wp.tile(cy))
        p_i = p0 + d * wp.vec2(sx[i], sy[i])
        # the cut: the FIRST arrival within a knot of the target, else the nearest point. A
        # profile that loops passes the target twice, and the second pass would price the loop
        dd = wp.where(on, wp.length_sq(p_i - tgt), 1.0e30)
        arrival = wp.where(on and dd <= d * d, i, BLOCK)
        cut = wp.tile_min(wp.tile(arrival))[0]
        nearest = wp.tile_argmin(wp.tile(dd))[0]
        if cut >= BLOCK:
            cut = nearest
        g3 = wp.vec3(0.0, 0.0, 0.0)
        if on:
            g3 = _clearance_grad_at(
                i,
                p_i,
                th_i,
                ct,
                taut_i,
                cut,
                clear_field,
                sgrid,
                blocked,
                rgrid,
                n_theta,
                w_veto,
                rear,
                w_clear,
                N,
            )
        gpx = g3[0]
        gpy = g3[1]
        if i == cut:
            gpx += 2.0 * w_end * (p_i[0] - tgt[0])
            gpy += 2.0 * w_end * (p_i[1] - tgt[1])
        # only where the point already has the clearance it is asked: the taut string hugs what
        # it bends round, and pulling to it there undid the clearance term (patek0 clearance p5
        # 0.53 -> 0.37 m, trunks 0.09 m from a trunk); in free space it straightens the curve
        free = sample_field(clear_field, sgrid, p_i[0], p_i[1]) >= ct
        if w_taut > 0.0 and on and i >= 1 and i <= cut and free:
            # toward the nearest point of the taut string
            t_near = p_i
            t_best = float(1.0e30)
            for t_k in range(n_taut[0] - 1):
                t_a = taut[t_k]
                t_ab = taut[t_k + 1] - t_a
                t_f = wp.dot(p_i - t_a, t_ab) / wp.max(wp.dot(t_ab, t_ab), 1.0e-12)
                t_q = t_a + wp.clamp(t_f, 0.0, 1.0) * t_ab
                t_d = wp.length_sq(p_i - t_q)
                if t_d < t_best:
                    t_best = t_d
                    t_near = t_q
            gpx += 2.0 * w_taut * (p_i[0] - t_near[0])
            gpy += 2.0 * w_taut * (p_i[1] - t_near[1])
        # the adjoint: dp_m/dtheta_i = d (-sin, cos)(theta_i) and dp_m/dd = sum_{j<m} t(theta_j)
        # for m > i, so each knot needs the gradient summed over the points after it
        incx = wp.tile_scan_inclusive(wp.tile(gpx))
        incy = wp.tile_scan_inclusive(wp.tile(gpy))
        totx = wp.tile_sum(wp.tile(gpx))[0]
        toty = wp.tile_sum(wp.tile(gpy))[0]
        after_x = totx - incx[i]
        after_y = toty - incy[i]
        gt = float(0.0)
        dds = float(0.0)
        if knot:
            gt = g3[2] + d * (-wp.sin(th_i) * after_x + wp.cos(th_i) * after_y)
            dds = wp.cos(th_i) * after_x + wp.sin(th_i) * after_y + gt * cum_i + w_k * k_i * k_i
        gds = w_len * float(cut) + wp.tile_sum(wp.tile(dds))[0]
        gt_tot = wp.tile_sum(wp.tile(gt))[0]
        gphi = 2.0 * w_pivot * ph + gt_tot
        # dJ/dkappa_j = d sum_{k>j} dJ/dtheta_k + the curvature and curvature-rate terms
        gt_inc = wp.tile_scan_inclusive(wp.tile(gt))
        tk = wp.tile(kk)
        n_it += 1
        bias_m = 1.0 - wp.pow(0.9, float(n_it))
        bias_v = 1.0 - wp.pow(0.999, float(n_it))
        m_ph = 0.9 * m_ph + 0.1 * gphi
        v_ph = 0.999 * v_ph + 0.001 * gphi * gphi
        ph = wp.clamp(
            ph - step * (m_ph / bias_m) / (wp.sqrt(v_ph / bias_v) + 1.0e-8), -3.14159265, 3.14159265
        )
        if knot:
            g_k = d * (gt_tot - gt_inc[i]) + 2.0 * w_k * k_i * d
            if i < N - 1:
                g_k -= 2.0 * w_dk * (tk[i + 1] - k_i) / d
            if i > 0:
                g_k += 2.0 * w_dk * (k_i - tk[i - 1]) / d
            m_i = 0.9 * m_i + 0.1 * g_k
            v_i = 0.999 * v_i + 0.001 * g_k * g_k
            k_i = k_i - step * (m_i / bias_m) / (wp.sqrt(v_i / bias_v) + 1.0e-8)
        m_ds = 0.9 * m_ds + 0.1 * gds
        v_ds = 0.999 * v_ds + 0.001 * gds * gds
        d = wp.clamp(
            d - 0.5 * step * (m_ds / bias_m) / (wp.sqrt(v_ds / bias_v) + 1.0e-8),
            0.6 * ds0,
            1.4 * ds0,
        )
        # the projection onto the bounds; the rate bound is a forward and a backward pass, so
        # one lane runs it from the block's tile and hands the result back through `scratch`
        k_i = wp.clamp(k_i, -kappa_max, kappa_max)
        kp = wp.tile(wp.where(knot, k_i, 0.0))
        if i == 0:
            r = kappa_rate * d
            buf = _vec_block()
            buf[0] = kp[0]
            for q in range(1, N):
                buf[q] = wp.clamp(kp[q], buf[q - 1] - r, buf[q - 1] + r)
            for q in range(N - 2, -1, -1):
                buf[q] = wp.clamp(buf[q], buf[q + 1] - r, buf[q + 1] + r)
            for q in range(N):
                scratch[q] = buf[q]
        _sync(i)
        if knot:
            k_i = scratch[i]
        _sync(i)  # every lane has read before lane 0 writes the next step's
        if stop_tol > 0.0:
            moved = wp.tile_max(wp.tile(wp.where(knot, wp.abs(k_i - k_old), 0.0)))[0]
            moved = wp.max(moved, wp.max(wp.abs(d - d_old), wp.abs(ph - ph_old)))
            if moved < stop_tol:
                break
    if knot:
        kappa[i] = k_i
        m[i] = m_i
        v[i] = v_i
    # the final profile's headings, for next frame's warm start (`_rebase_kernel`)
    kk = wp.where(knot, k_i, 0.0)
    cum = wp.tile_scan_exclusive(wp.tile(kk))
    if knot:
        th[i] = start[2] + ph + d * cum[i]
    if i == 0:
        used[0] = n_used
        ds[0] = d
        phi[0] = ph
        m[N] = m_ds
        v[N] = v_ds
        m[N + 1] = m_ph
        v[N + 1] = v_ph
        it[0] = n_it


@wp.kernel
def _finish_kernel(
    n_taut: wp.array(dtype=int),  # [1] < 2: no string, so no curve
    start: wp.array(dtype=float),
    target: wp.array(dtype=wp.vec2),
    kappa: wp.array(dtype=float),
    ds: wp.array(dtype=float),
    phi: wp.array(dtype=float),
    clear_field: wp.array2d(dtype=float),
    sgrid: Grid,
    half_width: float,
    rear: float,
    reject_m: float,
    taper: float,  # ReferencePathParams.clear_taper: the floor eases toward each end too
    blocked: wp.array3d(dtype=float),
    rgrid: Grid,
    n_theta: int,
    n_knots: int,
    pts: wp.array(dtype=wp.vec2),  # [N+1] out
    tan: wp.array(dtype=wp.vec2),  # [N+1] out
    s_out: wp.array(dtype=float),  # [N+1] out
    n_valid: wp.array(dtype=int),  # [1] out: points up to the one nearest the target
):
    if n_taut[0] < 2:
        n_valid[0] = 0
        return
    N = n_knots
    d = ds[0]
    heading = start[2] + phi[0]
    p = wp.vec2(start[0], start[1])
    pts[0] = p
    s_out[0] = 0.0
    for i in range(N):
        tan[i] = wp.vec2(wp.cos(heading), wp.sin(heading))
        p = p + d * tan[i]
        pts[i + 1] = p
        s_out[i + 1] = s_out[i] + d
        heading += d * kappa[i]
    tan[N] = wp.vec2(wp.cos(heading), wp.sin(heading))
    jb = int(0)
    bd = float(1.0e30)
    for i in range(N + 1):
        dd = wp.length_sq(pts[i] - target[0])
        if dd < bd:
            bd = dd
            jb = i
    n_valid[0] = jb + 1
    # REJECT a curve that takes the wheels through a wall (ReferencePathParams.reject_m): front
    # and rear axle discs along it, against the floor or what the start pose already has
    c0 = sample_field(clear_field, sgrid, start[0], start[1]) - half_width
    floor_m = wp.min(reject_m, c0 - 0.05)
    c1 = sample_field(clear_field, sgrid, pts[jb][0], pts[jb][1]) - half_width
    reject = int(0)
    for i in range(1, jb + 1):
        here = sample_field(clear_field, sgrid, pts[i][0], pts[i][1]) - half_width
        if i < N:
            q = pts[i] - rear * tan[i]
            here = wp.min(here, sample_field(clear_field, sgrid, q[0], q[1]) - half_width)
        f = floor_m
        if taper > 0.0:
            # eased toward the END too (a goal beside a wall), never made stricter than the
            # start's rule: a robot pressed to a wall lost every curve to a tighter floor
            f = wp.min(f, c1 + taper * (s_out[jb] - s_out[i]) - 0.05)
        if here < f:
            reject = 1
    if reject == 1:
        n_valid[0] = 0
    # and a pose the settle forbids at the curve's own heading (past the first two: the robot may
    # already stand on one)
    for i in range(2, wp.min(jb + 1, N)):
        heading_i = wp.atan2(tan[i][1], tan[i][0])
        if _veto_at(blocked, rgrid, n_theta, pts[i], heading_i) > 0.5:
            n_valid[0] = 0


def _grid_key(g: Grid) -> tuple:
    return (
        int(g.cells_x),
        int(g.cells_y),
        float(g.cell_size),
        float(g.origin_x),
        float(g.origin_y),
    )


class ReferencePath:
    """The per-frame reference curve, device-resident; `update` runs the kernels and the results
    are read by `MppiGpu.set_reference`."""

    def __init__(
        self,
        n_theta: int,
        half_width_m: float,
        rear_axle_m: float,
        params: ReferencePathParams = ReferencePathParams(),
        device: wp.Device | str | None = None,
    ) -> None:
        self.device = wp.get_device(device)
        # the optimiser is one thread block, one lane per curve point, and a CPU block is 1 lane
        if not self.device.is_cuda:
            raise ValueError(f"ReferencePath runs on a CUDA device, got {self.device}")
        if params.n_knots + 1 > BLOCK:
            raise ValueError(f"n_knots {params.n_knots} needs a block of n_knots + 1 <= {BLOCK}")
        self.params = params
        self.n_theta = int(n_theta)
        # measured from the settle's forbidden region, which already holds part of the body
        self.half_width = float(half_width_m) - params.forbidden_growth_m
        self.rear = float(rear_axle_m)
        N = params.n_knots
        with wp.ScopedDevice(self.device):
            self.start = wp.zeros(3, dtype=wp.float32)
            self.goal = wp.zeros(3, dtype=wp.float32)
            self.desc = wp.zeros(256, dtype=wp.vec2)
            self.vmin = wp.zeros((1, 1, 1), dtype=wp.float32)  # sized to V on the first frame
            self.prev_dist = wp.zeros((1, 1), dtype=wp.float32)  # sized to V on the first frame
            self.n_desc = wp.zeros(1, dtype=wp.int32)
            mv = params.n_taut
            self.verts = wp.zeros(mv, dtype=wp.vec2)
            self.n_verts = wp.zeros(1, dtype=wp.int32)
            self.vis = wp.zeros((mv, mv), dtype=wp.int32)
            self.best = wp.zeros(mv, dtype=wp.float32)
            self.prev = wp.zeros(mv, dtype=wp.int32)
            self.done = wp.zeros(mv, dtype=wp.int32)
            self.taut = wp.zeros(mv, dtype=wp.vec2)
            self.n_taut = wp.zeros(1, dtype=wp.int32)
            self.kappa = wp.zeros(N, dtype=wp.float32)
            self.ds = wp.full(1, params.spacing_m, dtype=wp.float32)
            self.target = wp.zeros(1, dtype=wp.vec2)
            self.taut_at = wp.zeros(N, dtype=wp.vec2)
            self.clear_target = wp.zeros(N, dtype=wp.float32)
            self.samp_p = wp.zeros(1024, dtype=wp.vec2)
            self.samp_q = wp.zeros(1024, dtype=wp.int32)
            self.n_samp = wp.zeros(1, dtype=wp.int32)
            self.t_len = wp.zeros(1, dtype=wp.float32)
            self.m = wp.zeros(N + 2, dtype=wp.float32)
            self.v = wp.zeros(N + 2, dtype=wp.float32)
            self.phi = wp.zeros(1, dtype=wp.float32)
            self.it = wp.zeros(1, dtype=wp.int32)
            self.th = wp.zeros(N, dtype=wp.float32)
            self.pts = wp.zeros(N + 1, dtype=wp.vec2)
            self.tan = wp.zeros(N + 1, dtype=wp.vec2)
            self.s = wp.zeros(N + 1, dtype=wp.float32)
            self.n_valid = wp.zeros(1, dtype=wp.int32)
            self.warm = wp.zeros(1, dtype=wp.int32)
            self.last_end = wp.zeros(1, dtype=wp.vec2)
            self.cold = wp.zeros(1, dtype=wp.int32)
            self.limit = wp.zeros(1, dtype=wp.int32)
            self.scratch = wp.zeros(BLOCK, dtype=wp.float32)
            self.used = wp.zeros(1, dtype=wp.int32)
        # The frame is ~300-1200 tiny single-thread launches, so on CUDA it runs as one captured
        # graph: launched eagerly it cost 17.6 ms median and 54 ms on a cold start (false_door on
        # dasenka, 2026-10-29), as much as the whole cost-to-go solve. The graph bakes in the
        # input arrays and grids, so it is rebuilt when they change.
        self._graph = None
        self._graph_key: tuple | None = None
        self._policy = None

    @property
    def offset(self) -> float:
        return self.params.clearance_m + self.half_width

    def _record(
        self,
        V: wp.array,
        route_grid: Grid,
        clear_field: wp.array,
        sim_grid: Grid,
        blocked: wp.array,
    ) -> None:
        """The whole frame as launches, every decision on the device, so it is captured once."""
        prm = self.params
        N = prm.n_knots
        wp.launch(
            _vmin_kernel,
            (V.shape[0], V.shape[1]),
            inputs=[V],
            outputs=[self.vmin],
            device=self.device,
        )
        if prm.sticky_tol_m > 0.0:
            wp.launch(
                _prev_dist_kernel,
                (route_grid.cells_y, route_grid.cells_x),
                inputs=[self.taut, self.n_taut, route_grid, 5.0],
                outputs=[self.prev_dist],
                device=self.device,
            )
        wp.launch(
            _descent_kernel,
            1,
            inputs=[
                self.vmin,
                route_grid,
                1,
                self.start,
                self.goal,
                clear_field,
                sim_grid,
                self.half_width,
                256,
                self.prev_dist,
                prm.sticky_tol_m,
                prm.sticky_w,
            ],
            outputs=[self.desc, self.n_desc],
            device=self.device,
        )
        if prm.policy_descent and self._policy is not None:
            ec, pdr, pdc, ph, n_prim = self._policy
            wp.launch(
                _policy_kernel,
                1,
                inputs=[
                    V,
                    ec,
                    pdr,
                    pdc,
                    ph,
                    n_prim,
                    route_grid,
                    self.start,
                    self.goal,
                    clear_field,
                    sim_grid,
                    self.half_width,
                    256,
                ],
                outputs=[self.desc, self.n_desc],
                device=self.device,
            )
        wp.launch(
            _resample_kernel,
            1,
            inputs=[self.desc, self.n_desc, prm.taut_step_m, prm.n_taut],
            outputs=[self.verts, self.n_verts],
            device=self.device,
        )
        wp.launch(
            _visible_kernel,
            (prm.n_taut, prm.n_taut),
            inputs=[
                self.verts,
                self.n_verts,
                clear_field,
                sim_grid,
                self.vmin,
                route_grid,
                1,
                prm.v_tol,
                self.offset,
            ],
            outputs=[self.vis],
            device=self.device,
        )
        wp.launch(
            _taut_kernel,
            1,
            inputs=[self.verts, self.n_verts, self.vis, self.best, self.prev, self.done],
            outputs=[self.taut, self.n_taut],
            device=self.device,
        )
        wp.launch(
            _mode_kernel,
            1,
            inputs=[
                self.taut,
                self.n_taut,
                self.pts,
                self.n_valid,
                self.warm,
                prm.rebase_m,
                prm.taut_step_m,
                prm.iterations,
                prm.iterations_cold,
                int(prm.cold_on_divergence),
                self.th,
                int(prm.cold_on_loop),
                self.start,
                prm.pivot_init_rad if prm.cold_on_repivot else 0.0,
                prm.cold_on_detour,
            ],
            outputs=[self.last_end, self.cold, self.limit],
            device=self.device,
        )
        wp.launch(
            _targets_kernel,
            1,
            inputs=[
                self.taut,
                self.n_taut,
                self.goal,
                clear_field,
                sim_grid,
                self.offset,
                prm.spacing_m,
                N,
            ],
            outputs=[
                self.target,
                self.taut_at,
                self.clear_target,
                self.samp_p,
                self.samp_q,
                self.n_samp,
                self.t_len,
            ],
            device=self.device,
        )
        wp.launch(
            _room_kernel,
            self.samp_p.shape[0],
            inputs=[self.samp_p, self.samp_q, self.n_samp, clear_field, sim_grid, self.offset],
            outputs=[self.clear_target],
            device=self.device,
        )
        if prm.clear_taper > 0.0:
            wp.launch(
                _taper_kernel,
                1,
                inputs=[
                    self.taut,
                    self.n_taut,
                    self.target,
                    self.t_len,
                    clear_field,
                    sim_grid,
                    prm.spacing_m,
                    N,
                    prm.clear_taper,
                ],
                outputs=[self.clear_target],
                device=self.device,
            )
        wp.launch(
            _init_kernel,
            1,
            inputs=[
                self.cold,
                self.taut,
                self.n_taut,
                self.start,
                prm.kappa_max,
                prm.kappa_rate_max,
                prm.clothoid_m,
                prm.spacing_m,
                N,
                prm.pivot_init_rad,
            ],
            outputs=[self.kappa, self.ds, self.phi, self.m, self.v, self.it],
            device=self.device,
        )
        wp.launch(
            _rebase_kernel,
            1,
            inputs=[self.cold, self.pts, self.th, self.n_valid, self.start, N],
            outputs=[self.kappa, self.phi, self.m, self.v],
            device=self.device,
        )

        wp.launch_tiled(
            _optimise_kernel,
            dim=[1],
            inputs=[
                self.start,
                self.target,
                self.clear_target,
                self.taut_at,
                clear_field,
                sim_grid,
                blocked,
                route_grid,
                self.n_theta,
                self.limit,
                prm.kappa_max,
                prm.kappa_rate_max,
                prm.spacing_m,
                prm.step,
                prm.w_end,
                prm.w_clear,
                prm.w_k,
                prm.w_dk,
                prm.w_len,
                prm.w_pivot,
                prm.w_veto,
                self.rear,
                N,
                self.taut,
                self.n_taut,
                prm.w_taut,
            ],
            outputs=[
                self.kappa,
                self.ds,
                self.phi,
                self.m,
                self.v,
                self.it,
                self.th,
                self.scratch,
                prm.stop_tol,
                self.used,
            ],
            block_dim=BLOCK,
            device=self.device,
        )
        wp.launch(
            _finish_kernel,
            1,
            inputs=[
                self.n_taut,
                self.start,
                self.target,
                self.kappa,
                self.ds,
                self.phi,
                clear_field,
                sim_grid,
                self.half_width,
                self.rear,
                prm.reject_m,
                prm.clear_taper,
                blocked,
                route_grid,
                self.n_theta,
                N,
            ],
            outputs=[self.pts, self.tan, self.s, self.n_valid],
            device=self.device,
        )

    def update(
        self,
        V: wp.array,
        route_grid: Grid,
        clear_field: wp.array,
        sim_grid: Grid,
        start_xyz: np.ndarray,
        warm: bool = True,
        goal_xy: np.ndarray | None = None,
        blocked: wp.array | None = None,
        policy=None,
    ) -> int:
        """One frame: descend `V` (on `route_grid`, both in the MPPI window's frame) from
        `start_xyz`, pull taut against `clear_field` (on `sim_grid`), then refine the curvature
        profile. Returns the number of valid curve points (< 3 = no usable path this frame).

        `clear_field` is the distance to forbidden poses (`forbidden_distance`) and must cover
        the routing window: the curve reaches its edge, and a map that ends sooner reads its
        border value past it, so obstacles there were invisible to both the optimiser and the
        rejection (slalom, loops through the third gate's slab).

        The homotopy is fixed by the descent each frame, but the profile is kept from last frame
        (`warm`) unless the taut string left last frame's curve (`_mode_kernel`), which is a new
        way out of the window and a cold start with `iterations_cold` steps instead of
        `iterations`. The returned count is the frame's one read back from the device."""
        # the solver whose move costs the policy walk follows (params.policy_descent)
        pol = None
        if policy is not None:
            pol = (
                policy._edge_cost,
                policy._prim_dr,
                policy._prim_dc,
                policy._prim_heading,
                int(policy.n_prim),
            )
        if (pol is None) != (getattr(self, "_policy", None) is None) or (
            pol is not None and pol[0].ptr != self._policy[0].ptr
        ):
            self._graph = None
        self._policy = pol
        if blocked is None:  # no settle verdict given: a field of zeros, the heading terms inert
            if getattr(self, "_no_veto", None) is None or self._no_veto.shape != V.shape:
                self._no_veto = wp.zeros(V.shape, dtype=wp.float32, device=self.device)
            blocked = self._no_veto
        self.start.assign(np.asarray(start_xyz[:3], np.float32))
        g = np.zeros(3, np.float32)
        if goal_xy is not None:
            g[:2] = goal_xy[:2]
            ext = route_grid.cells_x * route_grid.cell_size
            g[2] = float(
                route_grid.origin_x <= g[0] <= route_grid.origin_x + ext
                and route_grid.origin_y <= g[1] <= route_grid.origin_y + ext
            )
        self.goal.assign(g)
        self.warm.assign(np.array([int(warm)], np.int32))
        key = (V.ptr, clear_field.ptr, blocked.ptr, _grid_key(route_grid), _grid_key(sim_grid))
        if self.params.sticky_tol_m > 0.0 and self.prev_dist.shape != (
            route_grid.cells_y,
            route_grid.cells_x,
        ):
            self.prev_dist = wp.zeros(
                (route_grid.cells_y, route_grid.cells_x), dtype=wp.float32, device=self.device
            )
            self._graph = None
        if self.vmin.shape != (V.shape[0], V.shape[1], 1):
            self.vmin = wp.zeros((V.shape[0], V.shape[1], 1), dtype=wp.float32, device=self.device)
            self._graph = None
        if self._graph is None or key != self._graph_key:
            with wp.ScopedCapture(device=self.device) as cap:
                self._record(V, route_grid, clear_field, sim_grid, blocked)
            self._graph, self._graph_key = cap.graph, key
        wp.capture_launch(self._graph)
        return int(self.n_valid.numpy()[0])

    def curve(self) -> np.ndarray:
        """[n_valid, 2] the curve's points, host side (recording and tests)."""
        n = int(self.n_valid.numpy()[0])
        return self.pts.numpy()[:n].copy()
