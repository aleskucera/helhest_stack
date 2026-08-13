"""GPU-resident MPPI inner loop (sample -> rollout -> cost -> reweight) + CUDA graph.

The numpy MPPI loop is host-bound: per refine ~15 ms of numpy (cost, noise, target_wheel_omega)
vs ~2.7 ms of GPU work. These Warp kernels move the whole refine onto the device so
nothing but the executed pose ever comes back, and the refine is captured as a CUDA
graph and replayed. The cost kernel is validated by CONTRACT (analytic checks in
tests/control/test_mppi.py), not against a numpy twin.

Kernels (all suffixed _kernel):
  _sample_target_wheel_omega  spline-knot noise -> Ub -> writes the rollout's target_wheel_omega buffer (rear -> 0)
  _cost          per-rollout scalar cost J[B] (cost-to-go goal V^2 + graded-infeasible + effort/smooth)
  _minmax/_bisect_*/_count_below/_elite_u   CEM reweight (top-k elite mean) of U, on device
  _bump_seed/_reset_minmax     device-side RNG counter + reduction resets (graph-safe)
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..engine.robot import Robot
from ..engine.simulator import ForwardSimulator
from ..engine.terrain import _locate
from ..engine.terrain import Grid
from ..engine.terrain import sample_field
from ..profiling import StageProfiler


def _n_bisect(n_cand: int) -> int:
    """Device-side bisection steps for the CEM elite threshold tau. Each step halves the search
    interval, so n steps resolve tau to (jmax - jmin) / 2^n of the cost range. We want that finer
    than the spacing between candidate costs (~(jmax - jmin) / n_cand) so #{J <= tau} lands on
    target_k -> ceil(log2(n_cand)) bits + 5 (~32x margin). A FIXED count (not data-dependent) is what
    keeps the refine CUDA-graph-capturable -- a host sort/partition would need a readback + sync."""
    return int(np.ceil(np.log2(max(2, n_cand)))) + 5


@dataclass
class SamplingConfig:
    """MppiGpu config: how candidates are sampled + the elite selected each refine."""

    sigma: float = 0.5  # per-step Gaussian jitter on the wheel speeds
    sigma_knot: float = 1.0  # spline-knot noise -> smooth committed maneuvers
    n_knots: int = 4  # control knots spread over the horizon
    # wheel-speed box [wmin, wmax]. wmin < 0 enables reverse/point turns; the EFFECTIVE lower
    # bound is a device scalar (MppiGpu.set_wmin) so the node can gate reverse per frame on map
    # coverage behind the robot without recapturing the CUDA graph. wmin here is the initial value.
    wmin: float = 0.0
    wmax: float = 4.0
    wide_frac: float = 0.25  # fraction of candidates drawn from the WIDE global-search prior
    # fraction drawn from the STRAIGHT prior (zero differential, wl == wr): straight-ahead is usually
    # near-optimal, so seeding it explicitly stops the sampling-noise wobble on a clear shot. 0 = off.
    straight_frac: float = 0.0
    # Fraction drawn from the SPIN prior: zero mean, pure differential (wl = -v, wr = +v), i.e.
    # turning on the spot. Nothing else in the sampler can express this, because wmin >= 0 forbids
    # a negative wheel speed and the tightest sampleable turn is one wheel STOPPED -- a 0.54 m
    # radius, not zero. Without this a goal behind the robot can only be reached by driving a
    # forward loop. 0 = off.
    spin_frac: float = 0.0
    # Minimum |wheel speed| for a spin candidate. MEASURED on the robot 2026-08-10: below about
    # 2 rad/s it will not break loose on the spot at all, so a smaller command just strains.
    spin_min: float = 2.0
    # fraction drawn from the PIVOT prior (wl == -wr, turn in place). Only useful with reverse
    # enabled (effective wmin < 0); with wmin >= 0 the clamp degrades these to sharp arcs. 0 = off.
    pivot_frac: float = 0.0
    elite_frac: float = 0.02  # CEM top-k elite fraction
    # robust-mu replicas: each candidate is rolled out n_mu times under different friction scales
    # (MppiGpu.set_mu_band) and ranked by its WORST replica cost -- a candidate must survive the
    # whole mu band to win. n_mu must divide the batch; 1 = nominal (no robustness).
    n_mu: int = 1
    # turn-mode deadband [rad/s per step]: a candidate whose |mean net differential| stays below
    # this is "straight" (neutral); above it, it is a committed left/right maneuver, and the
    # elite mean only mixes candidates on the best candidate's side (obstacle dead ahead ->
    # commit early instead of averaging left- and right-passers into a straight-at-it mean).
    turn_mode_th: float = 0.5


@wp.struct
class CostWeights:
    """Device-side per-rollout cost weights, passed into _cost_kernel. Built from CostParams (below).
    The robot's envelope + feasibility thresholds (roll/pitch limits, roll/pitch cost shape,
    clear_margin, resid_tol) are NOT here -- they come from the Robot struct, one shared source."""

    goal_terminal: float
    goal_running: float
    explore_fallback: float
    # the cost-to-go's unreachable cap (planner.cw.lattice_cap = ctg._vcap); V >= it means the goal
    # is unreachable in-window (the field is flat) -> arm the explore_fallback straight-line pull
    lattice_cap: float
    out_of_bounds: float
    effort: float
    smoothness: float
    infeasible: float
    # penalize the turn differential (wr - wl)^2 -> a gradient toward STRAIGHT where the goal cost is
    # flat w.r.t. heading (the free-heading goal). Distinct from effort (which penalizes total speed).
    turn: float
    # friction-saturation certificate: penalize demand/grip past 1 (dimensionless overshoot).
    # This is what slows the robot where grip is short: demand grows with v*wz and accel.
    saturation: float
    # tip-over margin: penalize negative wheel loads (fraction of the robot weight).
    tip: float
    # penalize occupying UNMEASURED map cells while reversing (no sensor coverage backward).
    unknown: float
    # mild per-meter preference against reverse motion (forward keeps the sensor looking ahead).
    reverse: float
    # (alpha - 1) -> grip recovery: total_grip = (alpha-1)*m*g/k_turn. 0 disables saturation.
    inv_k_turn: float
    dt: float  # rollout timestep [s] for the accel term of the saturation demand


@dataclass(frozen=True)
class CostParams:  # host-side cost weights -- what you tune; build() -> the device CostWeights
    """The MPPI cost weights. The defaults are the lattice-routing tuning the demos run (routing +
    feasibility; the terminal dock handles reach+stop). A weight set to 0 disables its term in the
    kernel. The robot's envelope + feasibility thresholds are NOT here -- they live on the Robot
    struct (one shared source)."""

    goal_terminal: float = 3.0  # cost-to-go V^2 at the horizon end -> end the plan at the goal
    # cost-to-go V^2 averaged over the horizon -> make progress every step
    goal_running: float = 0.3
    # straight-line pull where V saturates (goal unreachable in-window)
    explore_fallback: float = 1.0
    out_of_bounds: float = 50.0  # soft wall just inside the world edge (V is clamped off-grid)
    effort: float = 2e-3  # penalize wheel-speed^2
    smoothness: float = 2e-3  # penalize wheel-speed CHANGES (jerk)
    # penalize clearance/residual/tip-over violations (does obstacle avoidance)
    infeasible: float = 1e5
    # penalize turning (wr - wl)^2 -> prefer straight when the goal cost doesn't care about heading;
    # small enough that a real need to turn (obstacle/offset goal) still wins. 0 = off.
    turn: float = 0.0
    # friction-saturation certificate weight (per unit demand/grip overshoot, early-weighted sum).
    # ~300 makes a sustained 20% overshoot compete with real routing differences and a 2x overshoot
    # dominate; the certificate is exact at tan(pitch) = mu for station-holding (see test).
    saturation: float = 300.0
    # tip-over weight (per unit min-load deficit as a fraction of weight). A lifting wheel also
    # invalidates the settle, so this must dominate like the infeasible terms do.
    tip: float = 2e4
    # unmeasured-cell occupancy while REVERSING (forward motion into unknown stays allowed -- the
    # sensor sees it before arrival; backward there is no sensor, so unknown must hard-lose).
    unknown: float = 1e4
    # per-meter shaping against reverse -- sized so reverse is an ESCAPE, not a route. A pivot's
    # V-surcharge is small (the router blends turning into arcs) and a pi pivot eats most of the
    # horizon, so myopic backward progress outbids pivot-then-forward at low weights: measured, at
    # 5 the robot backs up 11 m and creeps through a gap backward; at 25 it still reverses whole
    # routes. ~75 makes any forward-capable route win while a genuinely stuck robot (forward
    # progress impossible, V flat ahead) still backs out over remembered ground.
    reverse: float = 75.0

    def build(self) -> CostWeights:
        cw = CostWeights()
        cw.goal_terminal = self.goal_terminal
        cw.goal_running = self.goal_running
        cw.explore_fallback = self.explore_fallback
        cw.lattice_cap = 1e9  # off until armed: planner.cw.lattice_cap = ctg._vcap
        cw.out_of_bounds = self.out_of_bounds
        cw.effort = self.effort
        cw.smoothness = self.smoothness
        cw.infeasible = self.infeasible
        cw.turn = self.turn
        cw.saturation = self.saturation
        cw.tip = self.tip
        cw.unknown = self.unknown
        cw.reverse = self.reverse
        cw.inv_k_turn = 0.0  # armed by MppiGpu from the sim's solver (0 = saturation off)
        cw.dt = 0.1  # overwritten by MppiGpu from the sim's solver
        return cw


@wp.func
def _bilinear(
    field: wp.array3d(dtype=float),
    yi: int,
    xi: int,
    fx: float,
    fy: float,
    t: int,
):
    """Bilinear read of the heading-t slice field[:, :, t] at the fractional cell (yi + fy, xi + fx)."""
    return (
        (1.0 - fx) * (1.0 - fy) * field[yi, xi, t]
        + fx * (1.0 - fy) * field[yi, xi + 1, t]
        + (1.0 - fx) * fy * field[yi + 1, xi, t]
        + fx * fy * field[yi + 1, xi + 1, t]
    )


@wp.func
def sample_lattice(
    field: wp.array3d(dtype=float),
    grid: Grid,
    n_theta: int,
    x: float,
    y: float,
    yaw: float,
):
    """Trilinear sample of the orientation-aware cost-to-go V(x, y, theta): bilinear in (x, y),
    linear in the (wrapped) heading. Misaligned poses read high/inf, so MPPI's rollouts prefer an
    approach the forward-only robot can actually complete."""
    c = _locate(grid, x, y)
    two_pi = 6.2831853
    dth = two_pi / float(n_theta)
    m = yaw - wp.floor(yaw / two_pi) * two_pi  # yaw mod 2pi in [0, 2pi)
    ft = m / dth
    ftf = wp.floor(ft)
    t0 = int(ftf) % n_theta
    t1 = (t0 + 1) % n_theta
    fth = ft - ftf
    fx = c[2]
    fy = c[3]
    xi = int(c[0])
    yi = int(c[1])
    va = _bilinear(field, yi, xi, fx, fy, t0)
    vb = _bilinear(field, yi, xi, fx, fy, t1)
    return (1.0 - fth) * va + fth * vb


@wp.func
def _knot_bracket(
    t: int,
    horizon: int,
    n_knots: int,
):
    """n_knots control knots evenly spaced over the horizon: the two bracketing step t and the
    interpolation fraction between them. Deterministic -> threads sharing a knot agree."""
    knot_spacing = float(horizon - 1) / float(n_knots - 1)
    knot_pos = float(t) / knot_spacing
    knot_lo = int(knot_pos)
    knot_hi = wp.min(knot_lo + 1, n_knots - 1)
    frac = knot_pos - float(knot_lo)
    return knot_lo, knot_hi, frac


@wp.kernel
def _sample_target_wheel_omega_kernel(
    U: wp.array2d(dtype=float),
    sigma: float,
    sigma_knot: float,
    wlo: wp.array(
        dtype=float
    ),  # [1] effective lower wheel-speed bound (device -> live reverse gate)
    wmax: float,
    n_cand: int,  # candidates; rollouts r = k*n_cand + c are mu replicas SHARING candidate c's controls
    n_wide: int,
    n_straight: int,
    n_spin: int,
    spin_min: float,
    n_pivot: int,
    n_knots: int,
    seed: wp.array(dtype=int),
    target_wheel_omega: wp.array2d(dtype=wp.vec3),
):
    # Candidate index c keys ALL randomness, so the n_mu replicas of a candidate get identical
    # controls (their costs differ only through mu_scale). Candidate 0 keeps the nominal; the next
    # n_wide draw from the WIDE global-search prior; then n_spin from the SPIN prior (wl = -wr,
    # magnitude floored); then n_straight from the STRAIGHT prior (wl == wr); then n_pivot from the
    # PIVOT prior (wl == -wr, one signed rate per knot); the rest jitter around the nominal (NARROW
    # local refine).
    t, r = wp.tid()
    b = r % n_cand  # candidate this rollout replicates
    wmin = wlo[0]
    horizon = target_wheel_omega.shape[0]
    knot_lo, knot_hi, frac = _knot_bracket(t, horizon, n_knots)
    wheel_l = U[t, 0]
    wheel_r = U[t, 1]
    if b == 0:
        pass  # candidate 0 keeps the nominal (no noise)
    elif b < n_wide:
        # WIDE prior (global search): knots sampled UNIFORMLY over the full [wmin, wmax] box,
        # independent of the nominal -> a broad variety of maneuvers (the whole control space,
        # including reverse arcs when wmin < 0), so the elite can escape local minima.
        span = wmax - wmin
        left_lo = wmin + span * wp.randf(wp.rand_init(seed[0] + 1234, (b * n_knots + knot_lo) * 2))
        left_hi = wmin + span * wp.randf(wp.rand_init(seed[0] + 1234, (b * n_knots + knot_hi) * 2))
        right_lo = wmin + span * wp.randf(
            wp.rand_init(seed[0] + 1234, (b * n_knots + knot_lo) * 2 + 1)
        )
        right_hi = wmin + span * wp.randf(
            wp.rand_init(seed[0] + 1234, (b * n_knots + knot_hi) * 2 + 1)
        )
        wheel_l = (1.0 - frac) * left_lo + frac * left_hi
        wheel_r = (1.0 - frac) * right_lo + frac * right_hi
    elif b < n_wide + n_spin:
        # SPIN prior: zero mean, pure differential -- turn on the spot. One speed per candidate,
        # held across the horizon, because a spin that changes its mind mid-rollout is not a spin.
        # Magnitude is floored at spin_min: the real robot will not break loose below ~2 rad/s.
        u_spin = wp.randf(wp.rand_init(seed[0] + 5150, b))
        mag = spin_min + (wmax - spin_min) * u_spin
        if wp.randf(wp.rand_init(seed[0] + 6271, b)) < 0.5:
            mag = -mag
        wheel_l = -mag
        wheel_r = mag
    elif b < n_wide + n_spin + n_straight:
        # STRAIGHT prior: zero differential (wl == wr -> drives straight ahead). One common forward
        # speed per knot (so it can ramp/decelerate along the horizon while staying straight). Straight
        # is usually the near-optimal path, so seeding it explicitly lets the elite collapse onto a
        # clean straight command instead of averaging noisy turns; when a turn is actually needed these
        # cost more and simply don't win.
        span = wmax - wmin
        v_lo = wmin + span * wp.randf(wp.rand_init(seed[0] + 4321, b * n_knots + knot_lo))
        v_hi = wmin + span * wp.randf(wp.rand_init(seed[0] + 4321, b * n_knots + knot_hi))
        v = (1.0 - frac) * v_lo + frac * v_hi
        wheel_l = v
        wheel_r = v
    elif b < n_wide + n_spin + n_straight + n_pivot:
        # PIVOT prior: opposite wheels (wl == -wr -> turn in place), one signed rate per knot so a
        # pivot can ease in/out. With reverse disabled (wmin >= 0) the clamp below degrades these
        # to sharp forward arcs -- harmless, just redundant with WIDE.
        s = wmax * (2.0 * wp.randf(wp.rand_init(seed[0] + 7551, b)) - 1.0)  # signed rate scale
        m_lo = wp.randf(wp.rand_init(seed[0] + 7551, b * n_knots + knot_lo + 1))
        m_hi = wp.randf(wp.rand_init(seed[0] + 7551, b * n_knots + knot_hi + 1))
        v = s * ((1.0 - frac) * m_lo + frac * m_hi)
        wheel_l = -v
        wheel_r = v
    else:
        # NARROW (local refine): SPLINE bias around the nominal + per-step jitter (option A).
        # Knots keyed on (b, knot) -> shared across t -> a smooth committed maneuver. Recompute
        # the bracketing knots on the fly (rand_init is deterministic) -- no shared storage.
        eps_l_lo = wp.randn(wp.rand_init(seed[0], (b * n_knots + knot_lo) * 2))
        eps_l_hi = wp.randn(wp.rand_init(seed[0], (b * n_knots + knot_hi) * 2))
        eps_r_lo = wp.randn(wp.rand_init(seed[0], (b * n_knots + knot_lo) * 2 + 1))
        eps_r_hi = wp.randn(wp.rand_init(seed[0], (b * n_knots + knot_hi) * 2 + 1))
        wheel_l += sigma_knot * ((1.0 - frac) * eps_l_lo + frac * eps_l_hi)
        wheel_r += sigma_knot * ((1.0 - frac) * eps_r_lo + frac * eps_r_hi)
        # light per-step jitter (distinct stream)
        jitter = wp.rand_init(seed[0] + 9176, t * n_cand + b)
        wheel_l += sigma * wp.randn(jitter)
        wheel_r += sigma * wp.randn(jitter)
    # Clamp to the effective wheel-speed box (wmin >= 0 -> no reverse) -- EXCEPT the spin band,
    # whose whole point is one reversed wheel. Clamping it to wmin would silently zero that wheel
    # and turn every spin candidate into a one-wheel-stopped arc, which is the thing the prior
    # exists to get past. The pivot band gets no such escape: it is symmetric (wl == -wr), so with
    # reverse disabled it degrades harmlessly to a forward arc instead of collapsing to zero.
    lo = wmin
    if b >= n_wide and b < n_wide + n_spin:
        lo = -wmax
    target_wheel_omega[t, r] = wp.vec3(wp.clamp(wheel_l, lo, wmax), wp.clamp(wheel_r, lo, wmax), 0.0)


@wp.kernel
def _cost_kernel(
    controlled: wp.array2d(dtype=wp.vec3),
    derived: wp.array2d(dtype=wp.vec3),
    clearance: wp.array2d(dtype=float),
    residual: wp.array2d(dtype=float),
    target_wheel_omega: wp.array2d(dtype=wp.vec3),  # Ub in components [0], [1]
    current_wheel_omega: wp.array2d(dtype=wp.vec3),  # [T+1, B] realized omega; row t+1 drove step t
    twist: wp.array2d(dtype=wp.vec3),  # [T+1, B] solved body twist (vx, vy, yaw_rate); row t+1 drove step t
    turning: wp.array2d(dtype=wp.vec2),  # [T, B] (alpha, x_icr) used at step t
    loads: wp.array2d(dtype=wp.vec3),  # [T, B] wheel normal loads at the post-step pose
    measured: wp.array2d(dtype=wp.float32),  # [ny, nx] 1 = real data (sim grid); gates reverse
    sgrid: Grid,  # the sim grid `measured` lives on
    goal: wp.array(dtype=float),  # [2] world goal (device -> graph-safe, changes per replan)
    grid: Grid,  # geometry for sampling lattice_field at the rollout pose
    lattice_field: wp.array3d(
        dtype=float
    ),  # [ny, nx, n_theta] cost-to-go V(x,y,theta); the goal cost
    n_theta: int,
    cw: CostWeights,
    robot: Robot,  # envelope + feasibility thresholds (shared with the cost-to-go feasibility)
    horizon: int,
    Jout: wp.array(dtype=float),
    Jsafe: wp.array(dtype=float),  # the SAFETY share of Jout (see _robust_j_kernel)
):
    r = wp.tid()  # rollout
    run_sum = float(0.0)
    oob_sum = float(0.0)
    terminal_cost = float(0.0)
    effort_sum = float(0.0)
    smooth_sum = float(0.0)
    turn_sum = float(0.0)
    penalty_sum = float(0.0)
    sat_sum = float(0.0)
    tip_sum = float(0.0)
    unk_sum = float(0.0)
    rev_sum = float(0.0)
    edge = float(0.4)  # soft-wall margin inside the grid border
    x_lo = grid.origin_x + edge
    x_hi = grid.origin_x + float(grid.cells_x) * grid.cell_size - edge
    y_lo = grid.origin_y + edge
    y_hi = grid.origin_y + float(grid.cells_y) * grid.cell_size - edge
    mg_w = robot.mass * robot.gravity  # robot weight [N]
    prev_l = float(0.0)
    prev_r = float(0.0)
    for t in range(horizon):
        pose = controlled[t + 1, r]  # (x, y, yaw) after step t (pose 0 is shared by all candidates)
        om = current_wheel_omega[t + 1, r]  # realized (lagged) omega that drove step t
        v = twist[t + 1, r][0]  # realized body forward speed [m/s]; < 0 = reversing
        alpha = turning[t, r][0]
        wz = robot.wheel_radius * (om[1] - om[0]) / (2.0 * robot.half_track * alpha)
        if cw.out_of_bounds > 0.0:
            # soft wall at the world edge: depth past the margin (V is clamped off-grid, so the
            # goal term alone doesn't stop the robot driving off the map -- this does).
            oob_sum += wp.max(x_lo - pose[0], 0.0) + wp.max(pose[0] - x_hi, 0.0)
            oob_sum += wp.max(y_lo - pose[1], 0.0) + wp.max(pose[1] - y_hi, 0.0)
        # goal cost = the orientation-aware cost-to-go V(x,y,theta)^2 (routes around walls; misaligned
        # poses read high). A REVERSING step progresses along -x_body, i.e. like a forward-only robot
        # facing yaw+pi -- sample V there, so backing toward the goal is actually rewarded. Where V is
        # SATURATED (>= cap, goal unreachable in-window so the field is flat) fall back to a
        # straight-line pull -> the robot EXPLORES toward the goal, not creeps.
        yaw_eff = pose[2]
        if v < 0.0:
            yaw_eff += 3.14159265
        vl = sample_lattice(lattice_field, grid, n_theta, pose[0], pose[1], yaw_eff)
        if cw.explore_fallback > 0.0 and vl >= cw.lattice_cap * 0.9:
            dx = pose[0] - goal[0]
            dy = pose[1] - goal[1]
            goal_cost = cw.lattice_cap * cw.lattice_cap + cw.explore_fallback * (dx * dx + dy * dy)
        else:
            goal_cost = vl * vl
        run_sum += goal_cost
        terminal_cost = goal_cost  # last iter sticks -> terminal goal cost
        wheels = target_wheel_omega[t, r]  # (wL, wR) commanded
        effort_sum += wheels[0] * wheels[0] + wheels[1] * wheels[1]
        diff = wheels[1] - wheels[0]  # turn differential -> penalize (prefer straight)
        turn_sum += diff * diff
        if t > 0:
            dl = wheels[0] - prev_l
            dr = wheels[1] - prev_r
            smooth_sum += dl * dl + dr * dr
        prev_l = wheels[0]
        prev_r = wheels[1]
        # GRADED validity (option C): penalize HOW FAR past the margin/tol and HOW EARLY, not a
        # binary flag. De-saturates the cost (it still ranks when every sample violates), and
        # eating into the safety margin costs little while a real penetration costs a lot.
        clear_viol = wp.max(robot.clear_margin - clearance[t, r], 0.0)
        resid_viol = wp.max(residual[t, r] - robot.resid_tol, 0.0)
        # roll/pitch stability envelope (same limits as the cost-to-go feasibility): tipping is
        # invalid. The pitch limits are MOTION-relative: driving forward, climbing is nose-up =
        # NEGATIVE pitch; reversing up a slope the nose points downhill, so the roles mirror.
        pitch = derived[t + 1, r][1]
        roll = derived[t + 1, r][2]
        roll_viol = wp.max(wp.abs(roll) - robot.max_roll, 0.0)
        if v >= 0.0:
            climb_viol = wp.max(-pitch - robot.max_pitch_up, 0.0)
            descend_viol = wp.max(pitch - robot.max_pitch_down, 0.0)
        else:
            climb_viol = wp.max(pitch - robot.max_pitch_up, 0.0)
            descend_viol = wp.max(-pitch - robot.max_pitch_down, 0.0)
        early = float(horizon - t) / float(horizon)  # earlier violations hurt more (imminent)
        penalty_sum += early * (clear_viol + resid_viol + roll_viol + climb_viol + descend_viol)
        # FRICTION SATURATION certificate: demand the maneuver places on friction vs the grip
        # budget the model computed for this step. total_grip is recovered from alpha
        # (alpha = 1 + k_turn*total_grip/(m*g)); demand = slope hold (longitudinal) and
        # centripetal + side slope (lateral), combined as a friction ellipse. Past 1.0 the
        # commanded motion is not achievable -- the model itself never limits it (it gets MORE
        # optimistic as mu drops), so this term is what makes low-grip terrain slow/avoided.
        # NO accel term here: with tau_motor = 0 the model jumps wheel speed instantaneously, so
        # |dv|/dt would tax every speed change with a fictitious spike -- accel-vs-grip belongs in
        # the dynamics (grip-limited momentum), not the certificate.
        if cw.saturation > 0.0 and cw.inv_k_turn > 0.0:
            grip = wp.max((alpha - 1.0) * mg_w * cw.inv_k_turn, 1.0)  # [N], floored
            d_long = mg_w * wp.abs(wp.sin(pitch))
            d_lat = robot.mass * (wp.abs(v * wz) + robot.gravity * wp.abs(wp.sin(roll)))
            sat = wp.sqrt(d_long * d_long + d_lat * d_lat) / grip
            sat_sum += early * wp.max(sat - 1.0, 0.0)
        # TIP-OVER margin: a negative wheel load = CoM outside the support triangle (and the settle
        # pose is no longer trustworthy from here on) -- penalize the deficit as a weight fraction.
        if cw.tip > 0.0:
            ld = loads[t, r]
            min_n = wp.min(wp.min(ld[0], ld[1]), ld[2])
            tip_sum += early * wp.max(-min_n, 0.0) / mg_w
        # REVERSE: unmeasured cells behind the robot must hard-lose (no backward sensor coverage);
        # measured ones cost only the mild per-meter shaping.
        if v < -0.01:
            unk_sum += early * (1.0 - sample_field(measured, sgrid, pose[0], pose[1]))
            rev_sum += -v * cw.dt
    # goal_running is a mean over the horizon; effort/smoothness are raw sums (so they scale with the
    # horizon) -- the weights are tuned to that, mind it if the horizon changes. (Reaching + stopping at
    # the goal, and the right approach heading, are the cost-to-go + dock controller's job -- no
    # heading/endgame term.)
    # SAFETY share: the terms that must hold under EVERY mu hypothesis (worst-case reduce).
    # The goal/effort/shaping terms are averaged instead -- worst-casing the goal punishes
    # trajectory divergence per se and degenerates toward "slow is safest".
    safe = (
        cw.out_of_bounds * oob_sum
        + penalty_sum * cw.infeasible
        + cw.saturation * sat_sum
        + cw.tip * tip_sum
        + cw.unknown * unk_sum
    )
    Jsafe[r] = safe
    Jout[r] = (
        cw.goal_terminal * terminal_cost
        + cw.goal_running * (run_sum / float(horizon))
        + cw.effort * effort_sum
        + cw.smoothness * smooth_sum
        + cw.turn * turn_sum
        + cw.reverse * rev_sum
        + safe
    )


@wp.kernel
def _robust_j_kernel(
    J: wp.array(dtype=float),  # [B] per-rollout TOTAL cost (rollouts k*n_cand + c share controls)
    Jsafe: wp.array(dtype=float),  # [B] the safety share of J
    n_cand: int,
    n_mu: int,
    Jc: wp.array(dtype=float),  # [n_cand] robust per-candidate cost -> written
):
    """Per-candidate robust cost = WORST-replica safety + MEAN-replica everything else.

    Worst-casing only the safety terms (collision/tip/saturation/unknown/bounds) keeps the
    guarantee -- no mu hypothesis may crash -- while the averaged goal term stops the CVaR
    degeneracy where trajectory divergence across hypotheses makes "slow (or still)" look
    safest: measured, pure worst-case cost +53% traversal time on turny ground at a +-40% band
    and +232% at +-80%; the split removes the tax without touching the safety semantics.
    n_mu = 1 reduces to a copy."""
    c = wp.tid()
    worst_safe = Jsafe[c]
    mean_rest = J[c] - Jsafe[c]
    for k in range(1, n_mu):
        r = k * n_cand + c
        worst_safe = wp.max(worst_safe, Jsafe[r])
        mean_rest += J[r] - Jsafe[r]
    Jc[c] = worst_safe + mean_rest / float(n_mu)


# --- CEM reweight (option B): elite = top-k lowest-cost candidates; U = their mean. Rank-based,
# so the validity penalty can't blow up the weighting (invalid samples just don't make the
# elite). The top-k threshold tau is found by device-side BISECTION (a host partition would
# break the CUDA graph): bisect tau until #{J <= tau} ~= target_k. ---
@wp.kernel
def _reset_minmax_kernel(
    jmin: wp.array(dtype=float), jmax: wp.array(dtype=float), count: wp.array(dtype=float)
):
    jmin[0] = 1.0e30
    jmax[0] = -1.0e30
    count[0] = 0.0


@wp.kernel
def _minmax_kernel(
    J: wp.array(dtype=float), jmin: wp.array(dtype=float), jmax: wp.array(dtype=float)
):
    cost = J[wp.tid()]
    wp.atomic_min(jmin, 0, cost)
    wp.atomic_max(jmax, 0, cost)


@wp.kernel
def _bisect_init_kernel(
    jmin: wp.array(dtype=float),
    jmax: wp.array(dtype=float),
    tau_lo: wp.array(dtype=float),
    tau_hi: wp.array(dtype=float),
    tau: wp.array(dtype=float),
    count: wp.array(dtype=float),
):
    tau_lo[0] = jmin[0]
    tau_hi[0] = jmax[0]
    tau[0] = 0.5 * (jmin[0] + jmax[0])
    count[0] = 0.0


@wp.kernel
def _count_below_kernel(
    J: wp.array(dtype=float),
    tau: wp.array(dtype=float),
    count: wp.array(dtype=float),
):
    if J[wp.tid()] <= tau[0]:
        wp.atomic_add(count, 0, 1.0)


@wp.kernel
def _bisect_step_kernel(
    count: wp.array(dtype=float),
    target_k: float,
    tau_lo: wp.array(dtype=float),
    tau_hi: wp.array(dtype=float),
    tau: wp.array(dtype=float),
):
    if count[0] > target_k:
        tau_hi[0] = tau[0]  # too many below tau -> lower it
    else:
        tau_lo[0] = tau[0]  # too few -> raise it
    tau[0] = 0.5 * (tau_lo[0] + tau_hi[0])
    count[0] = 0.0  # reset for the next count pass


@wp.kernel
def _cand_dir_kernel(
    target_wheel_omega: wp.array2d(dtype=wp.vec3),
    horizon: int,
    turn_th: float,  # net-differential deadband; below it a candidate is "straight" (neutral)
    dir_out: wp.array(dtype=float),  # [n_cand] net direction: +1 forward-ish, -1 reverse-ish
    turn_out: wp.array(dtype=float),  # [n_cand] net turn mode: -1 right / 0 neutral / +1 left
):
    """Per-candidate maneuver mode keys. The elite is MULTIMODAL in two ways and a plain mean
    averages the modes into the worst of both worlds:
      - direction (reverse enabled): back-up vs pivot-and-drive -> mean is ~zero velocity;
      - turn side (obstacle dead ahead): pass-left vs pass-right -> mean aims AT the obstacle,
        and under slew-limited (smooth) actuation the robot then cannot dodge late.
    dir = sign of the summed mean wheel speed; turn = sign of the summed differential with a
    deadband (cruise noise stays neutral). The elite mean is restricted to candidates compatible
    with the best candidate's keys (see _elite_u_kernel)."""
    b = wp.tid()
    s = float(0.0)
    dsum = float(0.0)
    for t in range(horizon):
        w = target_wheel_omega[t, b]
        s += w[0] + w[1]
        dsum += w[1] - w[0]
    dir_out[b] = wp.where(s < 0.0, -1.0, 1.0)
    tk = float(0.0)
    if dsum > turn_th:
        tk = 1.0
    elif dsum < -turn_th:
        tk = -1.0
    turn_out[b] = tk


@wp.kernel
def _best_dir_kernel(
    J: wp.array(dtype=float),  # [n_cand] robust per-candidate cost
    jmin: wp.array(dtype=float),
    dirs: wp.array(dtype=float),
    turns: wp.array(dtype=float),
    n_cand: int,
    best_dir: wp.array(dtype=float),  # [1] direction of the lowest-cost candidate
    best_turn: wp.array(dtype=float),  # [1] turn mode of the lowest-cost candidate
):
    best_dir[0] = 1.0
    best_turn[0] = 0.0
    for b in range(n_cand):
        if J[b] <= jmin[0]:
            best_dir[0] = dirs[b]
            best_turn[0] = turns[b]
            return


@wp.kernel
def _elite_u_kernel(
    J: wp.array(dtype=float),  # [n_cand] robust per-candidate cost
    tau: wp.array(dtype=float),
    dirs: wp.array(dtype=float),  # [n_cand] net direction per candidate
    turns: wp.array(dtype=float),  # [n_cand] net turn mode per candidate
    best_dir: wp.array(dtype=float),  # [1] direction of the best candidate
    best_turn: wp.array(dtype=float),  # [1] turn mode of the best candidate
    target_wheel_omega: wp.array2d(
        dtype=wp.vec3
    ),  # replicas share controls -> read columns < n_cand
    wlo: wp.array(dtype=float),  # [1] effective lower wheel-speed bound
    wmax: float,
    n_cand: int,
    U: wp.array2d(dtype=float),
):
    t, wheel = wp.tid()  # (timestep, wheel: 0=L, 1=R)
    # MODE-COHERENT elite mean: average only elites compatible with the best candidate's
    # maneuver mode -- same direction, and same turn side (a NEUTRAL/straight candidate is
    # compatible with either side; if the best is neutral, sided candidates are excluded so a
    # left/right split can't pull the mean off the straight line). Forward-only cruising makes
    # every key (+1, 0), which reduces to the plain elite mean.
    elite_sum = float(0.0)
    elite_n = float(0.0)
    for b in range(n_cand):
        if J[b] <= tau[0] and dirs[b] == best_dir[0]:
            if turns[b] == best_turn[0] or turns[b] == 0.0:
                elite_sum += target_wheel_omega[t, b][wheel]
                elite_n += 1.0
    U[t, wheel] = wp.clamp(elite_sum / wp.max(elite_n, 1.0), wlo[0], wmax)


@wp.kernel
def _bump_seed_kernel(seed: wp.array(dtype=int)):
    seed[0] = seed[0] + 1


class MppiGpu:
    """GPU-resident MPPI: owns the nominal control `U` + scratch on device and runs the
    refine (sample -> rollout -> cost -> CEM reweight) entirely on the GPU. On CUDA
    the refine is captured once and replayed as a graph (the RNG counter is bumped
    in-graph, so each replay draws fresh noise); on CPU it runs eager. Wraps a ForwardSimulator.

    `goal` and `start_pose` are device arrays set per replan, so the captured graph picks
    up new values; the weights/sigma/wmax/elite_frac are baked at capture (fixed per planner)."""

    def __init__(
        self,
        sim: ForwardSimulator,
        cost: CostParams,  # cost weights (host) -> built into the device CostWeights struct
        sampling: SamplingConfig = SamplingConfig(),  # noise / wheel-speed box / elite fraction
        n_theta: int = 16,
        seed: int = 0,
        profile: bool = False,
    ):
        sampling = sampling or SamplingConfig()

        self.sim = sim
        self.device = sim.device

        self.n_rollouts, self.horizon = sim.batch_size, sim.n_steps

        # rollouts = n_mu friction replicas per candidate (n_mu = 1 -> plain MPPI)
        self.n_mu = max(1, int(sampling.n_mu))
        if self.n_rollouts % self.n_mu != 0:
            raise ValueError(f"n_mu={self.n_mu} must divide the batch size {self.n_rollouts}")
        self.n_cand = self.n_rollouts // self.n_mu
        self.n_bisect = _n_bisect(self.n_cand)  # CEM threshold bisection steps (scales with n_cand)
        self.sampling = sampling
        self.n_wide = int(sampling.wide_frac * self.n_cand)  # candidates drawn from the WIDE prior
        self.n_straight = int(
            sampling.straight_frac * self.n_cand
        )  # candidates from the STRAIGHT prior
        self.n_spin = int(sampling.spin_frac * self.n_cand)  # candidates from the SPIN prior
        self.n_pivot = int(sampling.pivot_frac * self.n_cand)  # candidates from the PIVOT prior

        # CEM elite count (over candidates)
        self.target_k = float(int(sampling.elite_frac * self.n_cand))
        self.cw = cost.build()  # host CostParams -> device CostWeights struct (weights only)
        # arm the saturation certificate from the sim's solver: total_grip is recovered from alpha
        # via k_turn, and the accel demand needs the rollout dt. k_turn <= 0 leaves it off.
        self.cw.inv_k_turn = 1.0 / sim.solver.k_turn if sim.solver.k_turn > 0.0 else 0.0
        self.cw.dt = sim.solver.dt
        # the robot's envelope/shape + feasibility thresholds (max_roll/pitch, roll/pitch cost shape,
        # clear_margin, resid_tol) are read straight from sim.robot in the cost kernel -- not copied here.

        self.robot = sim.robot
        with wp.ScopedDevice(self.device):
            self.U = wp.zeros((self.horizon, 2), dtype=wp.float32)
            self.J = wp.zeros(self.n_rollouts, dtype=wp.float32)  # cost per rollout (all replicas)
            self.Jsafe = wp.zeros(self.n_rollouts, dtype=wp.float32)  # safety share of J
            self.Jc = wp.zeros(self.n_cand, dtype=wp.float32)  # robust per-candidate cost
            self.jmin = wp.zeros(1, dtype=wp.float32)  # CEM bisection scalars
            self.jmax = wp.zeros(1, dtype=wp.float32)
            self.tau_lo = wp.zeros(1, dtype=wp.float32)
            self.tau_hi = wp.zeros(1, dtype=wp.float32)
            self.tau = wp.zeros(1, dtype=wp.float32)
            self.count = wp.zeros(1, dtype=wp.float32)
            self.dirs = wp.zeros(self.n_cand, dtype=wp.float32)  # per-candidate net direction
            self.turns = wp.zeros(self.n_cand, dtype=wp.float32)  # per-candidate net turn mode
            self.best_dir = wp.zeros(1, dtype=wp.float32)  # best candidate's direction
            self.best_turn = wp.zeros(1, dtype=wp.float32)  # best candidate's turn mode
            self.seed = wp.array([int(seed)], dtype=wp.int32)
            self.goal = wp.zeros(2, dtype=wp.float32)
            # effective lower wheel-speed bound, device-side so the node can gate reverse per frame
            # (set_wmin) without recapturing the CUDA graph
            self.wlo = wp.array([float(sampling.wmin)], dtype=wp.float32)
            ny, nx = sim.elevation.shape
            self.n_theta = int(n_theta)
            self.lattice_field = wp.zeros((ny, nx, n_theta), dtype=wp.float32)  # V(x,y,theta)
            # observed-cell mask on the sim grid (1 = real data); all-measured by default so the
            # unknown-cell penalty is inert until a perception mask is supplied (set_measured)
            self.measured = wp.full((ny, nx), 1.0, dtype=wp.float32)
        self.set_mu_band()  # nominal mu (fills sim.mu_scale for the replica layout)

        # the grid the cost kernel samples the lattice field on: defaults to the sim grid, but a COARSER
        # grid can be set (set_lattice(V, grid)) so the routing field is solved at low resolution --
        # the rollouts do fine obstacle avoidance, so the global router needn't be sim-resolution.
        self.lattice_grid = sim.grid
        self._graph = None

        # opt-in per-stage profiling of the captured refine loop (CUDA-event timing; off = no overhead)
        self._prof = StageProfiler(self.device, ("sample", "rollout", "cost", "reweight"), profile)
        self._n_refine_done = 0

    def reset_timing(self):
        """Clear the accumulated per-stage refine timings (e.g. after a warmup replan)."""
        self._prof.reset()

    def timing_stats(self):
        """Per-stage refine timing over profiled replans (CUDA + profile=True), first refine excluded:
        {stage: {"mean_ms", "std_ms", "n"}} for sample / rollout / cost / reweight. Use the means (the
        event reads sync, so a profiling run is serialized -- its wall-clock isn't the real rate).
        """
        return self._prof.stats()

    def reset_nominal(self, value=1.5):
        self.U.fill_(float(value))

    def nominal(self):
        """The current nominal control U [T, 2], on host."""
        return self.U.numpy()

    def set_nominal(self, U_host):
        self.U.assign(np.ascontiguousarray(U_host, np.float32))

    def set_mu_band(self, center=1.0, span=0.0):
        """Friction-uncertainty band for the robust replicas: replica k of every candidate rolls
        out under mu_scale evenly spaced in [center - span, center + span] (n_mu = 1 -> just
        `center`). Feed `center` from an online turn-gain estimate and `span` from its residual
        uncertainty. Cheap (one [B] upload) -- call whenever the estimate moves."""
        if self.n_mu == 1:
            scales = np.array([center], np.float32)
        else:
            scales = np.linspace(center - span, center + span, self.n_mu).astype(np.float32)
        self.mu_scales = np.maximum(scales, 0.05)  # keep every hypothesis physical
        self.sim.set_mu_scale(np.repeat(self.mu_scales, self.n_cand))

    def set_wmin(self, wmin):
        """Effective lower wheel-speed bound (device scalar, graph-safe). The node gates reverse on
        map coverage: sampling.wmin when the cells behind are measured, 0.0 otherwise."""
        self.wlo.assign(np.array([float(wmin)], np.float32))

    def set_measured(self, mask):
        """Observed-cell mask on the SIM grid (1 = cell has real data, 0 = blind). Reversing over
        blind cells is penalized by cw.unknown; forward motion is unaffected. Accepts a numpy bool/
        float array or a device array of the sim's [ny, nx] shape."""
        if isinstance(mask, wp.array):
            wp.copy(self.measured, mask)
        else:
            self.measured.assign(np.ascontiguousarray(mask, np.float32))

    def set_lattice(self, V, grid=None):
        """Copy the orientation-aware cost-to-go V[ny', nx', n_theta] into the stable buffer the cost
        kernel reads. `grid` is the Grid V was solved on (a coarse grid for a low-res routing field);
        defaults to the sim grid. Call before the first replan; on re-solve (moving goal) call again
        with the SAME shape -- it copies into the stable buffer the captured graph reads."""
        if tuple(V.shape) != tuple(self.lattice_field.shape):
            self.lattice_field = wp.zeros(V.shape, dtype=float, device=self.device)
        wp.copy(self.lattice_field, V)
        if grid is not None:
            self.lattice_grid = grid

    def _refine(self):
        """One MPPI iteration: sample -> rollout -> cost -> CEM reweight, all on device."""
        self._prof.mark(0)
        wp.launch(_bump_seed_kernel, 1, inputs=[self.seed], device=self.device)
        wp.launch(
            _sample_target_wheel_omega_kernel,
            (self.horizon, self.n_rollouts),
            inputs=[
                self.U,
                self.sampling.sigma,
                self.sampling.sigma_knot,
                self.wlo,
                self.sampling.wmax,
                self.n_cand,
                self.n_wide,
                self.n_straight,
                self.n_spin,
                self.sampling.spin_min,
                self.n_pivot,
                self.sampling.n_knots,
                self.seed,
            ],
            outputs=[self.sim.target_wheel_omega],
            device=self.device,
        )
        self._prof.mark(1)  # sample done
        self.sim.rollout_launch()
        self._prof.mark(2)  # rollout done
        wp.launch(
            _cost_kernel,
            self.n_rollouts,
            inputs=[
                self.sim.controlled,
                self.sim.derived,
                self.sim.clearance,
                self.sim.residual,
                self.sim.target_wheel_omega,
                self.sim.current_wheel_omega,
                self.sim.twist,
                self.sim.turning,
                self.sim.loads,
                self.measured,
                self.sim.grid,
                self.goal,
                self.lattice_grid,
                self.lattice_field,
                self.n_theta,
                self.cw,
                self.robot,
                self.horizon,
            ],
            outputs=[self.J, self.Jsafe],
            device=self.device,
        )
        # collapse the mu replicas: worst-replica safety + mean-replica goal/shaping
        wp.launch(
            _robust_j_kernel,
            self.n_cand,
            inputs=[self.J, self.Jsafe, self.n_cand, self.n_mu],
            outputs=[self.Jc],
            device=self.device,
        )
        self._prof.mark(3)  # cost done
        self._cem_reweight()
        self._prof.mark(4)  # reweight (CEM) done

    def _cem_reweight(self):
        """Top-k elite mean -> U over candidates (cost = Jc, the robust per-candidate cost): find
        the threshold tau by device-side bisection (#{Jc <= tau} ~= target_k), then average the
        elite candidates' controls."""

        wp.launch(
            _reset_minmax_kernel,
            1,
            inputs=[self.jmin, self.jmax, self.count],
            device=self.device,
        )
        wp.launch(
            _minmax_kernel,
            self.n_cand,
            inputs=[self.Jc, self.jmin, self.jmax],
            device=self.device,
        )
        wp.launch(
            _bisect_init_kernel,
            1,
            inputs=[
                self.jmin,
                self.jmax,
                self.tau_lo,
                self.tau_hi,
                self.tau,
                self.count,
            ],
            device=self.device,
        )
        for _ in range(self.n_bisect):
            wp.launch(
                _count_below_kernel,
                self.n_cand,
                inputs=[self.Jc, self.tau, self.count],
                device=self.device,
            )
            wp.launch(
                _bisect_step_kernel,
                1,
                inputs=[self.count, self.target_k, self.tau_lo, self.tau_hi, self.tau],
                device=self.device,
            )
        wp.launch(
            _cand_dir_kernel,
            self.n_cand,
            inputs=[
                self.sim.target_wheel_omega,
                self.horizon,
                self.sampling.turn_mode_th * float(self.horizon),
            ],
            outputs=[self.dirs, self.turns],
            device=self.device,
        )
        wp.launch(
            _best_dir_kernel,
            1,
            inputs=[self.Jc, self.jmin, self.dirs, self.turns, self.n_cand],
            outputs=[self.best_dir, self.best_turn],
            device=self.device,
        )
        wp.launch(
            _elite_u_kernel,
            (self.horizon, 2),
            inputs=[
                self.Jc,
                self.tau,
                self.dirs,
                self.turns,
                self.best_dir,
                self.best_turn,
                self.sim.target_wheel_omega,
                self.wlo,
                self.sampling.wmax,
                self.n_cand,
                self.U,
            ],
            device=self.device,
        )

    def replan(self, state, goal_xy, n_refine):
        """Run n_refine GPU refines from `state` toward world `goal_xy`; updates U in place."""
        self.goal.assign(np.asarray(goal_xy[:2], np.float32))
        self.sim.start_pose.assign(
            np.ascontiguousarray(
                np.tile(np.asarray(state, np.float32), (self.n_rollouts, 1)), np.float32
            )
        )
        if self.device.is_cuda:
            if self._graph is None:
                with wp.ScopedCapture(device=self.device) as cap:
                    self._refine()
                self._graph = cap.graph
            for _ in range(n_refine):
                wp.capture_launch(self._graph)
                self._n_refine_done += 1
                if self._prof.enabled and self._n_refine_done > 1:  # skip the first (cold) refine
                    self._prof.accumulate()
        else:
            for _ in range(n_refine):
                self._refine()
        return self.U
