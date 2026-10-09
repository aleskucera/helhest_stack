"""Path QP: smooth the route's path inside a clearance corridor, plan the speed along it, follow it.

The study/mpc-qp prototype. MPPI averages random samples, so its turning is noisy, and it plans speed
it does not get to keep -- the clearance governor caps it afterwards. Here the route field's own
path (which already chose the side of every obstacle, globally) is turned into a corridor -- how
far the robot may shift left or right of it at every point before its body comes within `margin`
of a wall face -- and a quadratic program picks the smoothest path inside it:

    minimise   w_curv * sum(kappa^2) + w_dcurv * sum(dkappa/ds ^2) + w_off * sum(e^2)
    subject to lo <= e <= hi,  |kappa| <= kappa_max,
               e and its slope at s = 0 match the robot's pose

with e the lateral offset from the reference and kappa linearised in e (small offsets). The limits
are hard: OSQP returns a path inside them or reports the problem infeasible, in which case the
caller falls back to MPPI. The speed is planned along the result before it is driven -- curvature,
the clearance law, then forward and backward acceleration passes -- so the robot slows before a
tight spot, not at it.

Device/host split: the corridor reads the wall-distance map on the GPU (one thread per reference
point, `2N` floats back); the QP is a few dozen variables, solved on the host (approved for the
prototype, see the 2026-10-09 discussion).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp
import warp as wp

from ..engine.terrain import Grid
from .qp_gpu import bands_to_csc
from .qp_gpu import rows_to_csc
from ..engine.terrain import sample_field


@dataclass(frozen=True)
class QpPathParams:
    ds: float = 0.15  # [m] spacing of the samples the corridor and the cost are evaluated at
    # [m] spacing of the spline knots the offset is built from (see spline_terms: one unknown per
    # sample made the problem too stiff for any iterative solver to finish)
    knot_m: float = 0.45
    # [m] how far ahead the path is optimised. Bending round an obstacle early and gently, tail
    # cleared, takes ~7 m (box: 4.5 m swerved late to a 2.7 m shelf). Past the routing window the
    # caller continues the route on the coarse layer (extend_route), or it stops at ~5 m.
    horizon_m: float = 8.0
    # [m] body-to-wall clearance the corridor keeps (beyond the half width). The clearance law needs
    # ~c0 + t_react x speed of it to pass at speed: 0.25 held the box pass to 0.44 m/s, 0.45 to 0.83.
    margin: float = 0.45
    max_offset: float = 3.0  # [m] how far the corridor search looks to each side
    w_curv: float = 1.0
    w_dcurv: float = 4.0
    # cost of being off the route -- the only term that says where to go. With w_curv it sets how
    # far the path takes to return after being pushed off, ~(w_curv / w_off)^(1/4): 2.1 m at 0.05,
    # so the return outlasted the plan and was deferred every frame (box shelf, trunks drift to
    # the map edge); 1 m at 1.0, which fits.
    w_off: float = 1.0
    # end the path on the route, heading along it, when the corridor there allows. Without it the
    # cheapest plan defers its way back into the tail, which is never driven (replanned first).
    pin_end: bool = False
    # [1/m] tightest turn the path may ask for: the route lattice's own turn radius (0.5 m). A hard
    # limit. How fast the curvature may CHANGE is only a cost (w_dcurv): what the robot can follow
    # depends on its speed, and the speed plan below slows for it.
    kappa_max: float = 2.0
    v_max: float = 2.0  # [m/s]
    a_lat: float = 1.0  # [m/s^2] lateral acceleration
    a_acc: float = 1.0  # [m/s^2] speeding up
    a_dec: float = 1.0  # [m/s^2] braking -- the command chain's plan_max_decel x wheel radius
    t_react: float = 0.08  # [s] the clearance law, as plan_clear_t_react
    c0: float = 0.15  # [m] the clearance law's fixed margin, as plan_clear_c0
    v_min: float = 0.3  # [m/s] the speed plan never asks for less while it has a path
    look_min: float = 0.8  # [m] pure-pursuit lookahead at rest
    look_gain: float = 0.5  # [s] lookahead grows with speed
    spin_deg: float = 60.0  # heading error past which the caller should turn on the spot instead


@wp.func
def _offset_free(
    clear: wp.array2d(dtype=wp.float32),
    grid: Grid,
    p: wp.vec3f,
    body: wp.array(dtype=wp.float32),
    need: wp.float32,
    e: wp.float32,
) -> bool:
    """Every body sample, laid along heading p[2] and shifted e to the left, keeps `need`."""
    ca = wp.cos(p[2])
    sa = wp.sin(p[2])
    free = bool(True)
    for b in range(body.shape[0]):
        u = body[b]
        if sample_field(clear, grid, p[0] + ca * u - sa * e, p[1] + sa * u + ca * e) < need:
            free = False
    return free


@wp.kernel
def _corridor_kernel(
    clear: wp.array2d(dtype=wp.float32),  # [ny, nx] distance to the nearest wall face [m]
    grid: Grid,
    ref: wp.array(dtype=wp.vec3f),  # [N] (x, y, heading) on `grid`
    body: wp.array(dtype=wp.float32),  # [K] sample points along the body axis [m], axle = 0
    need: wp.float32,  # [m] clearance each body sample must have
    step: wp.float32,  # [m] lateral search resolution
    n_side: int,  # lateral search steps to each side
    lo: wp.array(dtype=wp.float32),  # [N] right bound of the offset (<= 0 when the ref is free)
    hi: wp.array(dtype=wp.float32),  # [N] left bound
):
    """For reference point i, the contiguous range of lateral offsets e (left = +) at which every
    body sample, laid along the reference heading, keeps `need` clearance. A reference point that
    is itself blocked gets the free range nearest to it; none at all gives lo > hi."""
    i = wp.tid()
    p = ref[i]
    # the free offset nearest to the reference: search outward, left first at equal distance
    found = bool(False)
    j0 = int(0)
    for k in range(n_side + 1):
        for sgn in range(2):
            j = wp.where(sgn == 0, k, -k)
            if not found and _offset_free(clear, grid, p, body, need, float(j) * step):
                found = True
                j0 = j
    if not found:
        lo[i] = 1.0
        hi[i] = -1.0
        return
    # grow the free interval around it in both directions
    up = int(j0)
    going = bool(True)
    for k in range(1, n_side + 1):
        j = j0 + k
        if going and j <= n_side and _offset_free(clear, grid, p, body, need, float(j) * step):
            up = j
        else:
            going = False
    dn = int(j0)
    going = bool(True)
    for k in range(1, n_side + 1):
        j = j0 - k
        if going and j >= -n_side and _offset_free(clear, grid, p, body, need, float(j) * step):
            dn = j
        else:
            going = False
    lo[i] = float(dn) * step
    hi[i] = float(up) * step


def resample_reference(pts: np.ndarray, ds: float, length: float) -> np.ndarray | None:
    """The route's path (a lattice staircase) -> evenly spaced (x, y, heading, curvature), lightly
    smoothed, cut to `length`. None when there is less than three points of path."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return None
    seg = np.hypot(*np.diff(pts, axis=0).T)
    keep = np.r_[True, seg > 1e-6]
    pts = pts[keep]
    if len(pts) < 2:
        return None
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    n = int(min(length, s[-1]) / ds) + 1
    if n < 3:
        return None
    si = np.arange(n) * ds
    x, y = np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])
    # the staircase's corners are not the robot's: average them out before reading heading.
    # Padded by odd reflection, which keeps the ends where they are: "edge" padding dragged the
    # first points together (0.00, 0.15, 0.30 m became 0.09, 0.18, 0.30), and the QP, which takes
    # its samples to be ds apart, then started every path ~10 deg past the robot's heading -- the
    # tail swung into a trunk the governor then braked for
    k = 5
    if n > k:
        pad = k // 2
        x = np.convolve(np.pad(x, pad, mode="reflect", reflect_type="odd"), np.ones(k) / k, "valid")
        y = np.convolve(np.pad(y, pad, mode="reflect", reflect_type="odd"), np.ones(k) / k, "valid")
        # smoothing cuts corners, so the spacing is no longer ds: resample by arc length again
        s2 = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        n = int(s2[-1] / ds) + 1
        if n < 3:
            return None
        si = np.arange(n) * ds
        x, y = np.interp(si, s2, x), np.interp(si, s2, y)
    th = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    kappa = np.gradient(th) / ds
    return np.stack([x, y, th, kappa], 1)


def consistent_length(lo: np.ndarray, hi: np.ndarray, overlap: float = 0.05) -> int:
    """How many leading corridor points form one passage: stops before a point with no room
    (lo > hi) or one whose free interval does not overlap the previous one by `overlap` [m] -- a
    path cannot get from one to the next."""
    for i in range(len(lo)):
        if lo[i] > hi[i]:
            return i
        if i and min(hi[i], hi[i - 1]) - max(lo[i], lo[i - 1]) < overlap:
            return i
    return len(lo)


def trim_at_edge(
    pts: np.ndarray, lo_xy: tuple[float, float], hi_xy: tuple[float, float], gap: float
) -> np.ndarray:
    """Cut the fine route at its first point within `gap` [m] of the routing window's edge.

    The lattice route cannot step out of its window, so at the edge it wanders along it (spins,
    short arcs) and piles up metres of path that go nowhere -- which then read as horizon already
    covered. Past the edge the coarse layer's value is the route (extend_route)."""
    pts = np.asarray(pts, np.float64)
    inside = (
        (pts[:, 0] > lo_xy[0] + gap)
        & (pts[:, 0] < hi_xy[0] - gap)
        & (pts[:, 1] > lo_xy[1] + gap)
        & (pts[:, 1] < hi_xy[1] - gap)
    )
    out = np.flatnonzero(~inside[1:])  # the start is where the robot is, never cut
    return pts if out.size == 0 else pts[: out[0] + 2]


def _cut_corners(pts: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Chaikin corner cutting: each round replaces every interior corner by two points 1/4 and
    3/4 along its segments. The ends stay where they are; the curve stays inside the polygon."""
    for _ in range(rounds):
        if len(pts) < 3:
            return pts
        a, b = pts[:-1], pts[1:]
        cut = np.stack([0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b], 1).reshape(-1, 2)
        pts = np.concatenate([pts[:1], cut[1:-1], pts[-1:]], 0)
    return pts


def extend_route(
    pts: np.ndarray,
    coarse_v: np.ndarray,
    origin: tuple[float, float],
    cell: float,
    length: float,
    goal: tuple[float, float] | None = None,
) -> np.ndarray:
    """Continue the fine route `pts` [k, 2] downhill on the coarse value grid `coarse_v` [cy, cx]
    (cell centres at origin + (i + 0.5) * cell, same frame as pts) until it is `length` long.

    The fine route stops at the routing window's edge, ~5 m out, which is shorter than the
    horizon the path needs to bend early round an obstacle. The coarse layer already priced that
    window's border, so walking down its value continues the same route, only coarser. Steps go to
    the lowest of the 8 neighbours and stop when none is lower (a closed pocket) or at `goal`,
    which ends the route exactly -- not at its coarse cell's centre, up to 0.4 m past or beside it.
    The continuation is corner-cut: a staircase of 0.6 m cells has corners sharper than the
    sideways shift the QP may put on it, and shifting a corner by more than its radius loops."""
    pts = np.asarray(pts, np.float64)
    if goal is not None and len(pts) and np.hypot(*(pts[-1] - goal)) < cell:
        return np.concatenate([pts, np.asarray(goal, np.float64)[None]], 0)
    done = float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0
    if done >= length or len(pts) == 0:
        return pts
    ny, nx = coarse_v.shape
    r = int((pts[-1, 1] - origin[1]) / cell)
    c = int((pts[-1, 0] - origin[0]) / cell)
    if not (0 <= r < ny and 0 <= c < nx) or not np.isfinite(coarse_v[r, c]):
        return pts
    ext = [pts[-1]]
    last = pts[-1]
    while done < length:
        best, br, bc = coarse_v[r, c], r, c
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < ny and 0 <= cc < nx and coarse_v[rr, cc] < best:
                    best, br, bc = coarse_v[rr, cc], rr, cc
        if (br, bc) == (r, c):
            break
        r, c = br, bc
        p = np.array([origin[0] + (c + 0.5) * cell, origin[1] + (r + 0.5) * cell])
        if goal is not None and np.hypot(*(p - goal)) < cell:
            ext.append(np.asarray(goal, np.float64))
            break
        # the first coarse centres can lie beside or behind the fine route's end: skip to the
        # first one a whole cell on, or the joint folds back on itself
        if np.hypot(*(p - last)) < cell:
            continue
        done += float(np.hypot(*(p - last)))
        ext.append(p)
        last = p
    if len(ext) == 1:
        return pts
    # cut the corners from the fine route's last ~1 m on, so the joint is rounded too
    keep = max(0, len(pts) - 5)
    return np.concatenate(
        [pts[:keep], _cut_corners(np.concatenate([pts[keep:-1], np.array(ext)], 0))], 0
    )


def _bspline(u: np.ndarray, order: int) -> np.ndarray:
    """Uniform cubic B-spline basis [len(u), 4] at span parameter u in [0, 1], or its `order`-th
    derivative with respect to u."""
    if order == 0:
        return (
            np.stack(
                [(1 - u) ** 3, 3 * u**3 - 6 * u**2 + 4, -3 * u**3 + 3 * u**2 + 3 * u + 1, u**3], 1
            )
            / 6.0
        )
    if order == 1:
        return np.stack([-((1 - u) ** 2), 3 * u**2 - 4 * u, -3 * u**2 + 2 * u + 1, u**2], 1) / 2.0
    if order == 2:
        return np.stack([1 - u, 3 * u - 2, -3 * u + 1, u], 1)
    return np.tile([-1.0, 3.0, -3.0, 1.0], (len(u), 1))


@dataclass
class SplineQp:
    """The path QP over a cubic B-spline's control points, in the form both solvers take
    (control/qp_gpu.py): P's upper bands, and constraint rows that each touch 4 neighbouring
    control points (`a_col` .. `a_col` + 3). Scaled: cost by its largest diagonal, each row by its
    largest coefficient -- neither changes the minimiser."""

    p_band: np.ndarray  # [N, 4] P[i, i + d]
    q: np.ndarray  # [N]
    a_coef: np.ndarray  # [M, 4]
    a_col: np.ndarray  # [M] int
    lo: np.ndarray  # [M]
    hi: np.ndarray  # [M]
    span: np.ndarray  # [n] first control point of each fine sample
    value: np.ndarray  # [n, 4] basis at each fine sample: offset
    second: np.ndarray  # [n, 4] and its second derivative along s [1/m]

    def offsets(self, c: np.ndarray) -> np.ndarray:
        """[m] the path's offset from the reference at each fine sample."""
        return (self.value * c[self.span[:, None] + np.arange(4)]).sum(1)

    def curvature(self, c: np.ndarray) -> np.ndarray:
        """[1/m] d2e/ds2 at each fine sample: the path's curvature minus the reference's."""
        return (self.second * c[self.span[:, None] + np.arange(4)]).sum(1)


def spline_terms(
    kr: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    e0: float,
    slope0: float,
    ds: float,
    p: QpPathParams,
    pin_end: bool = False,
) -> SplineQp:
    """The path QP with the offset a cubic B-spline (knots ~`p.knot_m` apart) evaluated at the
    reference's samples (`ds` apart): the cost and the corridor still live on every sample, only
    the unknowns are fewer and coupled smoothly.

    Why not one unknown per sample: the curvature-rate term scales as 1/ds^6, which at 0.15 m
    made the cost ~1e8 stiffer sideways-wiggling than sideways-shifting. Every iterative solver
    then stopped early along the shift -- OSQP at 1e-4 was within 5 mm of the optimum on only half
    of a test set, up to ~1 m off on the rest. Knots 0.45 m apart cut that ratio ~700x."""
    n = len(kr)
    length = (n - 1) * ds
    spans = max(1, int(math.ceil(length / p.knot_m - 1e-9)))
    h = length / spans  # the knots stretched to end exactly on the last sample
    nv = spans + 3
    s = np.arange(n) * ds
    span = np.minimum((s / h).astype(int), spans - 1)
    u = s / h - span
    b0, b1, b2, b3 = (_bspline(u, k) / h**k for k in range(4))
    dkr = np.gradient(kr) / ds
    # cost: the same per-sample sums as before -- curvature, its rate, the offset
    P = np.zeros((nv, 4))
    q = np.zeros(nv)
    for basis, weight, target in ((b2, p.w_curv, kr), (b3, p.w_dcurv, dkr), (b0, p.w_off, None)):
        for t1 in range(4):
            if target is not None:
                np.add.at(q, span + t1, 2.0 * weight * target * basis[:, t1])
            for t2 in range(t1, 4):
                np.add.at(P[:, t2 - t1], span + t1, 2.0 * weight * basis[:, t1] * basis[:, t2])
    scale = 1.0 / P[:, 0].max()
    # rows: the corridor at every sample, the curvature limit at every sample, then the start
    # (offset and slope where the robot is) and, if asked, the end on the route along it
    coef = [b0, b2]
    col = [span, span]
    low = [lo, -p.kappa_max - kr]
    high = [hi, p.kappa_max - kr]
    pins = [(b0[0], span[0], e0), (b1[0], span[0], slope0)]
    if pin_end and lo[-2] <= 0.0 <= hi[-2] and lo[-1] <= 0.0 <= hi[-1]:
        pins += [(b0[-1], span[-1], 0.0), (b1[-1], span[-1], 0.0)]
    for c, k, v in pins:
        coef.append(c[None])
        col.append(np.array([k]))
        low.append(np.array([v]))
        high.append(np.array([v]))
    a_coef = np.concatenate(coef, 0)
    norm = np.abs(a_coef).max(1)
    return SplineQp(
        p_band=P * scale,
        q=q * scale,
        a_coef=a_coef / norm[:, None],
        a_col=np.concatenate(col).astype(np.int32),
        lo=np.concatenate(low) / norm,
        hi=np.concatenate(high) / norm,
        span=span,
        value=b0,
        second=b2,
    )


@dataclass
class QpPath:
    """One optimised path, its speed plan, and what the solver said."""

    xy: np.ndarray  # [N, 2]
    heading: np.ndarray  # [N]
    kappa: np.ndarray  # [N] curvature of the optimised path [1/m]
    v: np.ndarray  # [N] planned speed [m/s]
    lo: np.ndarray  # [N] corridor bounds used
    hi: np.ndarray
    offset: np.ndarray  # [N] lateral offset from the reference
    status: str


class PathQp:
    """Corridor (GPU) + QP (OSQP, host) + speed plan + pure pursuit, for one robot."""

    def __init__(
        self,
        params: QpPathParams,
        wheel_radius: float,
        half_width: float,
        tail: float,
        head: float,
        device: str | None = None,
    ) -> None:
        self.p = params
        self.r = float(wheel_radius)
        self.half_width = float(half_width)
        self.device = wp.get_device(device)
        n_body = 9
        self._body = wp.array(
            np.linspace(-tail, head, n_body).astype(np.float32), device=self.device
        )
        self._n_max = int(params.horizon_m / params.ds) + 2
        self._ref = wp.zeros(self._n_max, dtype=wp.vec3f, device=self.device)
        self._lo = wp.zeros(self._n_max, dtype=wp.float32, device=self.device)
        self._hi = wp.zeros(self._n_max, dtype=wp.float32, device=self.device)

    def corridor(
        self, ref: np.ndarray, clear: wp.array, grid: Grid, margin: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(ref)
        self._ref.assign(
            np.ascontiguousarray(np.pad(ref[:, :3], ((0, self._n_max - n), (0, 0))), np.float32)
        )
        step = 0.05
        wp.launch(
            _corridor_kernel,
            dim=n,
            inputs=[
                clear,
                grid,
                self._ref,
                self._body,
                self.half_width + (self.p.margin if margin is None else margin),
                step,
                int(self.p.max_offset / step),
            ],
            outputs=[self._lo, self._hi],
            device=self.device,
        )
        return self._lo.numpy()[:n].astype(np.float64), self._hi.numpy()[:n].astype(np.float64)

    def solve(
        self,
        ref: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        e0: float,
        psi0: float,
        v0: float,
        stop_at_end: bool,
    ) -> QpPath | None:
        """The smoothest path inside [lo, hi] that starts at the robot (offset e0, heading error
        psi0 from the reference), and the speed planned along it. None if infeasible."""
        p, ds = self.p, self.p.ds
        # the robot is where it is: never let the first points' bounds exclude it
        lo, hi = lo.copy(), hi.copy()
        for i in range(min(3, len(ref))):
            lo[i], hi[i] = min(lo[i], e0 - 0.05), max(hi[i], e0 + 0.05)
        # plan only as far as the corridor is one passage: up to a point with no room, or where it
        # jumps sides (a route through an obstacle -- the coarse layer's, at 0.6 m -- is free to
        # the left at one point and to the right at the next). Stop before the cut.
        # on the inside of a bend the path's radius is the bend's minus the shift, so keep the
        # shift to (radius - the tightest radius the robot turns, 1 / kappa_max): a path shifted
        # past the bend's radius loops (an offset curve's swallowtail), and short of it already
        # turns 1 / (1 - kappa e) times sharper than the bend -- which the curvature limit, being
        # linearised about the reference, cannot see. A bend tighter than the robot's turn makes
        # the limit negative: the path must pass it on the outside
        kr_all = ref[:, 3]
        inside = np.abs(kr_all) > 1e-6
        radius = 1.0 / np.where(inside, np.abs(kr_all), 1.0)
        room = np.minimum(0.8 * radius, radius - 1.0 / p.kappa_max)
        lim = np.where(inside, np.sign(kr_all) * room, 0.0)
        hi = np.where(kr_all > 1e-6, np.minimum(hi, lim), hi)
        lo = np.where(kr_all < -1e-6, np.maximum(lo, lim), lo)
        for i in range(min(3, len(ref))):  # ... but the robot is still where it is
            lo[i], hi[i] = min(lo[i], e0 - 0.05), max(hi[i], e0 + 0.05)
        n = consistent_length(lo, hi)
        if n < len(ref):
            ref, lo, hi, stop_at_end = ref[:n], lo[:n], hi[:n], True
        if n < 8:
            return None
        kr = ref[:, 3]
        slope0 = math.tan(max(-1.2, min(1.2, psi0)))
        terms = spline_terms(kr, lo, hi, e0, slope0, ds, p, p.pin_end)
        nv = len(terms.q)
        solver = osqp.OSQP()
        solver.setup(
            sp.triu(bands_to_csc(terms.p_band, nv), format="csc"),
            terms.q,
            rows_to_csc(terms.a_coef, terms.a_col, nv),
            terms.lo,
            terms.hi,
            verbose=False,
            # 1e-6, not OSQP's 1e-4: at 1e-4 ADMM has not settled which corridor sample the path
            # touches, the polish guesses the wrong set and is dropped, and the rough iterate it
            # returns was >10 cm off the optimum on 10 % of QPs recorded in drive_sim (0 % at
            # 1e-6, 99.9 % within 5 mm; 0.26 ms median per QP)
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=20000,
            polishing=True,
        )
        res = solver.solve()
        status = str(res.info.status)
        self.last_status = status
        if res.x is None or "solved" not in status or "inaccurate" in status:
            return None
        c = np.asarray(res.x, np.float64)
        e = terms.offsets(c)
        nx_, ny_ = -np.sin(ref[:, 2]), np.cos(ref[:, 2])
        xy = np.stack([ref[:, 0] + e * nx_, ref[:, 1] + e * ny_], 1)
        heading = np.unwrap(np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])))
        kappa = kr + terms.curvature(c)
        # [m] body clearance: the corridor's own margin plus the room left inside it. A bound at the
        # search's reach is not a wall -- nothing was found that far out -- so it limits nothing.
        reach = p.max_offset - 1e-3
        room_hi = np.where(hi >= reach, np.inf, hi - e)
        room_lo = np.where(lo <= -reach, np.inf, e - lo)
        slack = p.margin + np.minimum(room_hi, room_lo)
        v = self.speed_plan(kappa, slack, v0, stop_at_end)
        return QpPath(xy, heading, kappa, v, lo, hi, e, status)

    def speed_plan(
        self,
        kappa: np.ndarray,
        slack: np.ndarray,
        v0: float,
        stop_at_end: bool,
        tail_swing: float = 0.0,
    ) -> np.ndarray:
        """Speed along the path, planned ahead: curvature and clearance limits, then the
        acceleration pass from the current speed and the braking pass from the end. `slack` is
        the body's clearance to the nearest wall face along the path [m]. `tail_swing` is the
        governor's turn_ratio x tail [m]: its law caps the FASTEST body point, v + that x |yaw
        rate| = v (1 + tail_swing |kappa|), so the plan divides the allowed speed by that."""
        p, ds = self.p, self.p.ds
        v = np.minimum(p.v_max, np.sqrt(p.a_lat / np.maximum(np.abs(kappa), 1e-6)))
        allowed = np.maximum(p.v_min, (slack - p.c0) / p.t_react)
        v = np.minimum(v, allowed / (1.0 + tail_swing * np.abs(kappa)))
        v[0] = max(0.0, v0)
        for i in range(1, len(v)):
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * p.a_acc * ds))
        if stop_at_end:
            v[-1] = 0.0
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * p.a_dec * ds))
        return v

    def timed_poses(self, path: QpPath, dt: float, steps: int) -> np.ndarray:
        """The path as the robot will drive it: (x, y, heading) every `dt` seconds along the speed
        plan, `steps + 1` poses -- the shape the clearance governor checks a plan in."""
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path.xy, axis=0).T))]
        v = np.maximum(path.v, 0.05)
        t = np.r_[0.0, np.cumsum(np.diff(s) / (0.5 * (v[1:] + v[:-1])))]
        ts = np.minimum(np.arange(steps + 1) * dt, t[-1])
        x, y = np.interp(ts, t, path.xy[:, 0]), np.interp(ts, t, path.xy[:, 1])
        th = np.interp(ts, t, path.heading)
        return np.stack([x, y, th], 1)

    def track(
        self, path: QpPath, x: float, y: float, yaw: float, alpha: float, half_track: float
    ) -> tuple[float, float]:
        """Pure pursuit on the optimised path at the planned speed -> (wl, wr) [rad/s]. `alpha` is
        the turn model's resistance (yaw = R (wr - wl) / (2 b alpha)), so the asked-for yaw rate
        is what the model expects the wheels to produce."""
        p = self.p
        d = np.hypot(path.xy[:, 0] - x, path.xy[:, 1] - y)
        i0 = int(np.argmin(d))
        v = float(path.v[min(i0 + 1, len(path.v) - 1)])
        look = p.look_min + p.look_gain * v
        far = np.flatnonzero((np.arange(len(d)) >= i0) & (d >= look))
        tx, ty = path.xy[far[0]] if len(far) else path.xy[-1]
        ang = math.atan2(ty - y, tx - x) - yaw
        ang = (ang + math.pi) % (2.0 * math.pi) - math.pi
        reach = max(math.hypot(tx - x, ty - y), 1e-3)
        kappa = 2.0 * math.sin(ang) / reach
        wz = kappa * v
        mean = v / self.r
        half_diff = half_track * alpha * wz / self.r
        return mean - half_diff, mean + half_diff
