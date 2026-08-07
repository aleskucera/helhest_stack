"""Device kinematics: the quasi-static settle + the monolithic forward step.

One Warp thread = one rollout. The 3x3 Newton settle runs in registers (numerical
Jacobian, fixed iters). Mirrors the numpy `placement`/`state` reference so that
stays the finite-diff oracle. Orientation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

Arg convention (Option 2): the differentiated grids (height, friction) are plain
`wp.array` kernel args; everything else rides in three structs --
`Grid` (terrain.py), `Robot` (robot.py), `Solver` (this module).

State across timesteps is two vec3s (avoids the length-6 spatial_vector type):
controlled = (x, y, yaw) the wheel-driven DOF; derived = (z, pitch, roll) the terrain-settled DOF.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from .linalg import solve3
from .robot import Robot
from .rotations import drot_x
from .rotations import drot_y
from .rotations import drot_z
from .rotations import euler_zyx
from .rotations import rot_x
from .rotations import rot_y
from .rotations import rot_z
from .terrain import _locate
from .terrain import Grid
from .terrain import sample_field
from .terrain import sample_height_grad
from .terrain import sample_normal

# Warp 1.13/1.14 ptxas MISCOMPILES this module's large combined `step` kernel at
# -O3 on CUDA: a register spill produces an invalid __local__ read (illegal
# memory access at runtime). Verified via compute-sanitizer + an -O level sweep
# (-O3 crashes; -O2/-O1/-O0 are correct). -O2 is correct and ~as fast as -O3, so
# pin this module to it. CPU is unaffected (defaults to -O2).
wp.set_module_options({"optimization_level": 2})

# Certificate denominators are floored at this fraction of the robot's weight: a near-unloaded
# contact would otherwise report an enormous ratio while transmitting almost nothing.
_GRIP_FLOOR_FRAC = wp.constant(1.0e-3)

_TWO_PI = wp.constant(2.0 * float(np.pi))


# --- settle/integration numerics: host params + the device-side `Solver` struct ---
@wp.struct
class Solver:
    """Device-side settle/integration numerics — the built form of SolverParams.
    All scalars -> safe as a struct.

    `k_turn` (the friction->alpha turning gain) and `tau_motor` ride along here
    as non-diff scalars; promote either to a plain length-1 array only if
    d(loss)/d(param) is ever needed.
    """

    newton_iters: wp.int32  # max Newton iterations (cap)
    atol: wp.float32  # stop early once |residual| < atol
    max_step: wp.vec3  # per-iter Newton cap (z[m], pitch[rad], roll[rad])
    tilt_clamp: wp.float32
    dt: wp.float32
    k_turn: wp.float32
    tau_motor: wp.float32  # first-order actuator lag time constant [s]; 0 = no lag


@dataclass
class SolverParams:  # settle/integration numerics — tuning, separate from the robot
    dt: float = 0.1  # integration / control timestep [s]
    newton_iters: int = 12  # settle Newton cap; DEEP by default (the IFT adjoint needs a
    # converged root); forward-only planning caps at 6 (warm-started settle needs ~2).
    atol: float = 1e-6  # settle early-exit tol; TIGHT by default because the IFT settle
    # adjoint assumes residual~=0 at the root. Forward-only planning loosens it to ~1e-4
    # (0.1mm, 100x under the resid_tol=1e-2 validity gate) to save ~1 Newton iter/settle.
    max_step: tuple = (0.1, 0.2, 0.2)  # per-iter Newton cap (z[m], pitch[rad], roll[rad])
    tilt_clamp: float = 1.05  # clamp |pitch|, |roll| to ~60 deg
    k_turn: float = 2.0
    tau_motor: float = 0.0  # actuator lag [s]; 0 = instantaneous (no lag)

    def build(self) -> Solver:
        s = Solver()
        s.newton_iters = self.newton_iters
        s.atol = self.atol
        s.max_step = wp.vec3(self.max_step[0], self.max_step[1], self.max_step[2])
        s.tilt_clamp = self.tilt_clamp
        s.dt = self.dt
        s.k_turn = self.k_turn
        s.tau_motor = self.tau_motor
        return s


@wp.func
def motor_lag_step(
    current_wheel_omega: wp.vec3, target_wheel_omega: wp.vec3, dt: float, tau: float
) -> wp.vec3:
    """First-order actuator lag: advance current_wheel_omega one timestep toward target_wheel_omega.

    alpha = min(dt / tau, 1.0) so the update never overshoots. When tau == 0
    (the default) alpha clamps to 1.0 and current_wheel_omega = target_wheel_omega immediately,
    reproducing the original instantaneous-tracking behavior.
    """
    alpha = wp.min(dt / wp.max(tau, 1e-6), 1.0)
    return current_wheel_omega + alpha * (target_wheel_omega - current_wheel_omega)


@wp.func
def clearances(
    envelope: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    x: float,
    y: float,
    yaw: float,
    z: float,
    pitch: float,
    roll: float,
):
    """Signed wheel clearances c_i = wc_z - H_env(wc_xy) - r_wheel for the 3 wheels (wc = wheel center)."""
    R = euler_zyx(yaw, pitch, roll)
    p = wp.vec3(x, y, z)

    c = wp.vec3()
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_center = p + R * robot.wheel_pos[st_i]
        height = sample_field(envelope, grid, wheel_center[0], wheel_center[1])
        c[st_i] = wheel_center[2] - height - robot.wheel_radius

    return c


@wp.func
def settle(
    envelope: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    controlled: wp.vec3,
    derived_init: wp.vec3,
):
    """Solve (z, pitch, roll) with an ANALYTIC 3x3 Newton Jacobian.

    `controlled` = (x, y, yaw) the fixed planar pose; solve `derived` = (z, pitch, roll).
    c_i = wc_iz - envelope(wc_ixy) - r_wheel, wc_i = (x,y,z) + Rz Ry Rx wheel_i.
    dc_i/dz = 1; dc_i/dpitch and dc_i/droll come from dR/dpitch, dR/droll applied
    to wheel_i, combined with the terrain gradient (gx,gy) at the wheel center. One
    euler + one value+grad sample per wheel per iter (vs 4 evals numerically).
    """
    x = controlled[0]
    y = controlled[1]
    yaw = controlled[2]
    Rz = rot_z(yaw)
    derived = derived_init
    for _ in range(solver.newton_iters):
        Ry = rot_y(derived[1])
        Rx = rot_x(derived[2])
        Rot = Rz * Ry * Rx
        dRp = Rz * drot_y(derived[1]) * Rx  # d(Rot)/dpitch
        dRr = Rz * Ry * drot_x(derived[2])  # d(Rot)/droll
        p = wp.vec3(x, y, derived[0])

        res = wp.vec3()  # residual: per-wheel clearance c_i
        J = wp.mat33()  # Jacobian: row i = dc_i/d(z, pitch, roll)
        for i in range(wp.static(3)):
            st_i = wp.static(i)

            wheel_pos = robot.wheel_pos[st_i]
            wheel_center = p + Rot * wheel_pos

            s = sample_height_grad(envelope, grid, wheel_center[0], wheel_center[1])
            height, gx, gy = s[0], s[1], s[2]  # surface height + slope under the wheel center

            res[st_i] = wheel_center[2] - height - robot.wheel_radius

            # z lifts the wheel center (dc/dz = 1); pitch/roll swing it (dRp/dRr * wheel)
            # across the terrain slope (gx, gy).
            dp = dRp * wheel_pos
            dr = dRr * wheel_pos
            J[st_i, 0] = 1.0
            J[st_i, 1] = dp[2] - gx * dp[0] - gy * dp[1]
            J[st_i, 2] = dr[2] - gx * dr[0] - gy * dr[1]

        if wp.dot(res, res) < solver.atol * solver.atol:
            break  # converged: the current derived is the root (skip the rest)
        delta = solve3(J, res)  # Newton step: J @ delta = res
        # damped Newton step on every DOF...
        for i in range(wp.static(3)):
            st_i = wp.static(i)
            derived[st_i] = derived[st_i] - wp.clamp(
                delta[st_i], -solver.max_step[st_i], solver.max_step[st_i]
            )
        # ...then clamp the tilt angles; z (height) is left free
        derived[1] = wp.clamp(derived[1], -solver.tilt_clamp, solver.tilt_clamp)
        derived[2] = wp.clamp(derived[2], -solver.tilt_clamp, solver.tilt_clamp)
    return derived


@wp.func
def _scatter_h(
    adj_envelope: wp.array2d(dtype=wp.float32), grid: Grid, x: float, y: float, coef: float
):
    """Accumulate coef * (bilinear weights of (x,y)) into the envelope adjoint array.

    This is d(sample_field)/dH at (x,y): the same 4-node stencil sample_field
    reads (via the shared `_locate`), scattered with atomics (many output cells may
    hit the same node).
    """
    c = _locate(grid, x, y)
    xi = int(c[0])
    yi = int(c[1])
    frac_x = c[2]
    frac_y = c[3]
    wp.atomic_add(adj_envelope, yi, xi, coef * (1.0 - frac_x) * (1.0 - frac_y))
    wp.atomic_add(adj_envelope, yi, xi + 1, coef * frac_x * (1.0 - frac_y))
    wp.atomic_add(adj_envelope, yi + 1, xi, coef * (1.0 - frac_x) * frac_y)
    wp.atomic_add(adj_envelope, yi + 1, xi + 1, coef * frac_x * frac_y)


@wp.func_grad(settle)
def adj_settle(
    envelope: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    controlled: wp.vec3,
    derived_init: wp.vec3,
    adj_ret: wp.vec3,
):
    """Implicit (IFT) adjoint of the settle. adj_ret = cotangent on the settled `derived`.

    With residual c(derived, controlled, envelope) = 0 and J = dc/d(derived) at the root:
        lambda = J^-T adj_ret
        adj_theta = -(dc/dtheta)^T lambda   for theta in {controlled, envelope}
    Everything analytic (same J as the forward + closed-form pose derivatives from
    the rotation derivatives and the terrain gradient). d/d(derived_init) = 0 (root
    independent of warm start).
    """
    x = controlled[0]
    y = controlled[1]
    yaw = controlled[2]
    # recompute the converged derived
    derived = settle(envelope, grid, robot, solver, controlled, derived_init)
    Rz = rot_z(yaw)
    Ry = rot_y(derived[1])
    Rx = rot_x(derived[2])
    Rot = Rz * Ry * Rx
    dRp = Rz * drot_y(derived[1]) * Rx
    dRr = Rz * Ry * drot_x(derived[2])
    dRyaw = drot_z(yaw) * Ry * Rx
    p = wp.vec3(x, y, derived[0])

    # build J (same formula as the forward) and stash the terrain slopes for phase 2
    J = wp.mat33()
    gx = wp.vec3()
    gy = wp.vec3()
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_pos = robot.wheel_pos[st_i]
        wheel_center = p + Rot * wheel_pos
        s = sample_height_grad(envelope, grid, wheel_center[0], wheel_center[1])
        gx[st_i] = s[1]
        gy[st_i] = s[2]
        dp = dRp * wheel_pos
        dr = dRr * wheel_pos
        J[st_i, 0] = 1.0
        J[st_i, 1] = dp[2] - s[1] * dp[0] - s[2] * dp[1]
        J[st_i, 2] = dr[2] - s[1] * dr[0] - s[2] * dr[1]
    lam = solve3(wp.transpose(J), adj_ret)

    # adj_controlled = -(dc/dcontrolled)^T lambda  (dc_i/dx = -gx_i, dc_i/dy = -gy_i, yaw via dRyaw);
    # adj_envelope: scatter lambda_i into wheel-center i's bilinear stencil.
    adj_pose = wp.vec3()
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_pos = robot.wheel_pos[st_i]
        wheel_center = p + Rot * wheel_pos  # cheap recompute (no re-sample)
        dy = dRyaw * wheel_pos
        cw = dy[2] - gx[st_i] * dy[0] - gy[st_i] * dy[1]
        adj_pose[0] = adj_pose[0] + gx[st_i] * lam[st_i]
        adj_pose[1] = adj_pose[1] + gy[st_i] * lam[st_i]
        adj_pose[2] = adj_pose[2] - cw * lam[st_i]
        _scatter_h(wp.adjoint[envelope], grid, wheel_center[0], wheel_center[1], lam[st_i])
    wp.adjoint[controlled] += adj_pose


# Batched-terrain settle: the custom-grad function can NOT take a per-rollout slice envelope[b]
# (Warp scatters wp.adjoint of a view into freed/None storage -> segfault), so the FULL [B,ny,nx]
# array + the index b are threaded through; the forward delegates to the 2D settle on the slice and
# the adjoint scatters into wp.adjoint[envelope] at [b, ...] via _scatter_h_bt.
@wp.func
def _scatter_h_bt(
    adj_env: wp.array3d(dtype=wp.float32), b: int, grid: Grid, x: float, y: float, coef: float
):
    """_scatter_h for one slice b of a [B, ny, nx] envelope adjoint."""
    c = _locate(grid, x, y)
    xi = int(c[0])
    yi = int(c[1])
    frac_x = c[2]
    frac_y = c[3]
    wp.atomic_add(adj_env, b, yi, xi, coef * (1.0 - frac_x) * (1.0 - frac_y))
    wp.atomic_add(adj_env, b, yi, xi + 1, coef * frac_x * (1.0 - frac_y))
    wp.atomic_add(adj_env, b, yi + 1, xi, coef * (1.0 - frac_x) * frac_y)
    wp.atomic_add(adj_env, b, yi + 1, xi + 1, coef * frac_x * frac_y)


@wp.func
def settle_bt(
    envelope: wp.array3d(dtype=wp.float32),
    b: int,
    grid: Grid,
    robot: Robot,
    solver: Solver,
    controlled: wp.vec3,
    derived_init: wp.vec3,
) -> wp.vec3:
    """Settle rollout b on its own terrain slice. Forward only -- the custom grad replaces the
    body, so the inner 2D settle(envelope[b]) is never differentiated (no view-adjoint)."""
    return settle(envelope[b], grid, robot, solver, controlled, derived_init)


@wp.func_grad(settle_bt)
def adj_settle_bt(
    envelope: wp.array3d(dtype=wp.float32),
    b: int,
    grid: Grid,
    robot: Robot,
    solver: Solver,
    controlled: wp.vec3,
    derived_init: wp.vec3,
    adj_ret: wp.vec3,
):
    """adj_settle for slice b: identical IFT math, reading envelope[b] (forward) and scattering
    into wp.adjoint[envelope] at [b, ...]."""
    x = controlled[0]
    y = controlled[1]
    yaw = controlled[2]
    derived = settle(envelope[b], grid, robot, solver, controlled, derived_init)
    Rz = rot_z(yaw)
    Ry = rot_y(derived[1])
    Rx = rot_x(derived[2])
    Rot = Rz * Ry * Rx
    dRp = Rz * drot_y(derived[1]) * Rx
    dRr = Rz * Ry * drot_x(derived[2])
    dRyaw = drot_z(yaw) * Ry * Rx
    p = wp.vec3(x, y, derived[0])

    J = wp.mat33()
    gx = wp.vec3()
    gy = wp.vec3()
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_pos = robot.wheel_pos[st_i]
        wheel_center = p + Rot * wheel_pos
        s = sample_height_grad(envelope[b], grid, wheel_center[0], wheel_center[1])
        gx[st_i] = s[1]
        gy[st_i] = s[2]
        dp = dRp * wheel_pos
        dr = dRr * wheel_pos
        J[st_i, 0] = 1.0
        J[st_i, 1] = dp[2] - s[1] * dp[0] - s[2] * dp[1]
        J[st_i, 2] = dr[2] - s[1] * dr[0] - s[2] * dr[1]
    lam = solve3(wp.transpose(J), adj_ret)

    adj_pose = wp.vec3()
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_pos = robot.wheel_pos[st_i]
        wheel_center = p + Rot * wheel_pos
        dy = dRyaw * wheel_pos
        cw = dy[2] - gx[st_i] * dy[0] - gy[st_i] * dy[1]
        adj_pose[0] = adj_pose[0] + gx[st_i] * lam[st_i]
        adj_pose[1] = adj_pose[1] + gy[st_i] * lam[st_i]
        adj_pose[2] = adj_pose[2] - cw * lam[st_i]
        _scatter_h_bt(wp.adjoint[envelope], b, grid, wheel_center[0], wheel_center[1], lam[st_i])
    wp.adjoint[controlled] += adj_pose


@wp.func
def normal_loads(
    envelope: wp.array2d(dtype=wp.float32), grid: Grid, robot: Robot, R: wp.mat33, p: wp.vec3
):
    """Quasi-static contact normal loads N_i from gravity (3x3 force/torque solve).

    The body pose is (R, p): R is the orientation (body->world, from euler_zyx of
    yaw/pitch/roll) and p is the body-origin position (x, y, z). Body-frame points
    map to world as q_world = p + R * q_body, so the CoM and each wheel center are
    placed at this pose; the contacts then sit on `envelope` (the wheel-envelope
    grid). `robot` carries mass/gravity/com/wheel_pos/wheel_radius.

    Row 0: vertical force balance Sum N_i n_iz = m g.
    Rows 1-2: horizontal torque balance about the CoM. Returns N = vec3(N0,N1,N2).
    """
    com_world = p + R * robot.com

    A = wp.mat33()  # row 0: n_iz (vertical force); rows 1-2: (r_i x n_i)_xy (torque about CoM)
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_pos = robot.wheel_pos[st_i]
        wheel_center = p + R * wheel_pos
        n = sample_normal(envelope, grid, wheel_center[0], wheel_center[1])
        ct = wheel_center - robot.wheel_radius * n  # contact point
        r = ct - com_world  # moment arm about the CoM
        m = wp.cross(r, n)
        A[0, st_i] = n[2]
        A[1, st_i] = m[0]
        A[2, st_i] = m[1]

    b = wp.vec3(robot.mass * robot.gravity, 0.0, 0.0)
    return solve3(A, b)


@wp.func
def stability_margin(robot: Robot, loads: wp.vec3) -> float:
    """Static tip-over margin: min_i(N_i) / (m g) -- the least-loaded contact as a weight fraction.

    Zero means one wheel carries nothing, i.e. the CoM has reached an edge of the support
    triangle; negative means `normal_loads` is pressing down a wheel that should have lifted, and
    from that step on z, pitch AND roll are all untrustworthy, not just the flagged wheel.

    MEASURED CAVEAT (tests/engine/certificates.py, `selftest_ramp_margin`): on this robot the
    margin barely moves. `normal_loads` balances vertical force and horizontal torque with the
    contact NORMALS only -- the tangential (friction) reaction that actually holds the body on a
    slope, and its moment about the CoM, are absent -- so on any uniform plane the solve reduces
    to the body-frame barycentric weights of the CoM, giving

        min_i(N_i) / (m g) = |com_x| / rear_offset / (cos(pitch) cos(roll)),

    which RISES with tilt instead of falling. Terrain SHAPE does move it -- load transfer enters
    only through the wheel-radius contact offsets, since the wheel positions are body-fixed -- but
    only weakly: over the 12 shape worlds in `selftest_shape_margin` (rocks and 0.6 m spikes under
    each wheel, crests, valleys, saddles, roofs, tilts to 57 deg) it stays inside [0.23, 0.33].
    Read it as a load-transfer diagnostic, not as the slope tip-over test; the geometric tip
    angles (29.5 deg over the front axle, 34.6 deg about a rear edge) are NOT what it reports.
    """
    return wp.min(wp.min(loads[0], loads[1]), loads[2]) / (robot.mass * robot.gravity)


@wp.func
def contact_grip(
    envelope: wp.array2d(dtype=wp.float32),
    friction: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    R: wp.mat33,
    p: wp.vec3,
    loads: wp.vec3,
) -> float:
    """Coulomb budget Sum_i mu_i N_i of the three contacts at the pose (R, p).

    `loads` are that pose's normal loads; mu is sampled at each contact point, exactly as in
    `step_predict`'s turning solve (same wheel centers, same normal offset).
    """
    total = float(0.0)
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_center = p + R * robot.wheel_pos[st_i]
        n = sample_normal(envelope, grid, wheel_center[0], wheel_center[1])
        ct = wheel_center - robot.wheel_radius * n  # contact point
        total += sample_field(friction, grid, ct[0], ct[1]) * loads[st_i]
    return total


@wp.func
def friction_saturation(
    robot: Robot,
    loads: wp.vec3,
    grip: float,
    pitch: float,
    roll: float,
    forward_speed: float,
    yaw_rate: float,
) -> float:
    """Tangential demand / Coulomb budget at this pose: > 1 means the twist is unachievable.

    Demand is the friction force the contacts must supply to hold the quasi-static twist,
    combined through the friction ellipse:

        demand_long = m g sin(pitch)                              (gravity along the slope)
        demand_lat  = m v psi_dot + m g cos(pitch) sin(roll)      (centripetal + cross-slope)

    Budget is `mu_bar * m g cos(pitch) cos(roll)`, where mu_bar = Sum_i mu_i N_i / Sum_i N_i is
    the load-weighted friction coefficient. The budget is deliberately rebuilt from the weight
    rather than taken as `Sum_i mu_i N_i` directly: `normal_loads` is a normal-only balance, so
    its loads sum to m g / (cos pitch cos roll) instead of the true m g cos(tilt) (see
    `stability_margin`). Using them raw would make the certificate cross 1.0 at
    sin(theta) cos(theta) = mu, which peaks at 0.5 -- so it could never fire at all on mu > 0.5
    terrain. Only the load RATIOS are taken from the solve, and that bias cancels in mu_bar.

    Both denominators are floored at 1e-3 of the robot's weight: an unloaded contact otherwise
    reports enormous saturation while transmitting nothing.

    Quasi-static, so there is no `m a` term (no body-velocity state; IMPROVEMENTS.md section 5).
    """
    weight = robot.mass * robot.gravity
    floor = _GRIP_FLOOR_FRAC * weight
    cp = wp.cos(pitch)
    cr = wp.cos(roll)
    load_sum = loads[0] + loads[1] + loads[2]
    mu_bar = grip / wp.max(load_sum, floor)
    budget = mu_bar * weight * cp * cr

    demand_long = weight * wp.sin(pitch)
    demand_lat = robot.mass * forward_speed * yaw_rate + weight * cp * wp.sin(roll)
    demand = wp.sqrt(demand_long * demand_long + demand_lat * demand_lat)
    return demand / wp.max(budget, floor)


@wp.func
def torque_saturation(robot: Robot, pitch: float) -> float:
    """Required drive torque / motor limit at this pose: > 1 means the grade stalls the drivetrain.

    Holding or climbing a grade needs m g sin(pitch) of tractive force, shared by the three
    wheels, so each must deliver `m g |sin(pitch)| * wheel_radius / 3` at the wheel. This is the
    second reason code next to `friction_saturation`, and the planner responses differ: slip means
    pick another route, stall means pick another speed.

    Gravity only -- turn resistance and rolling resistance are NOT included, which is what makes
    the boundary independent of mu (and hence separable from the friction certificate). With the
    default `motor_torque_limit = inf` this is identically 0; see RobotParams for the measurement
    that is still missing.
    """
    torque = robot.mass * robot.gravity * wp.abs(wp.sin(pitch)) * robot.wheel_radius / 3.0
    return torque / robot.motor_torque_limit


@wp.func
def chassis_clearance(
    elevation: wp.array2d(dtype=wp.float32), grid: Grid, robot: Robot, R: wp.mat33, p: wp.vec3
):
    """Min signed clearance of the chassis bottom-face points above RAW terrain.

    Negative == high-centered (belly penetrates). `elevation` is the raw heightmap.
    """
    cmin = float(1.0e9)
    for i in range(robot.n_chassis):
        w = p + R * robot.chassis_pts[i]
        c = w[2] - sample_field(elevation, grid, w[0], w[1])
        cmin = wp.min(cmin, c)
    return cmin


@wp.func
def yaw_bin(yaw: float, n_yaw: int) -> int:
    """Index of the yaw-binned envelope slice nearest to heading `yaw` (wrapped into [0, n_yaw)).

    The spherical wheel envelope is yaw-invariant, so the stack is a single slice and this is a
    constant 0 -- the default path never touches the float math below. A CYLINDER wheel
    (RobotParams.wheel_width) is not yaw-invariant and gets one dilated slice per bin.

    COUPLED CONSTRAINT (IMPROVEMENTS.md section 7): with a yaw-dependent envelope, `psi_dot * dt`
    must stay inside one bin or the rollout aliases across slices. At 32 bins (11.25 deg) and
    dt = 0.1 s that holds up to psi_dot ~ 2 rad/s, which spin-in-place reaches at roughly
    omega_max = 4.5 rad/s. omega_max is not recorded anywhere in this repo; if it is near
    8 rad/s the cylinder and a finer step have to land together. Not solved here -- documented.
    """
    if n_yaw == 1:
        return 0
    bins = float(n_yaw)
    k = int(wp.floor(yaw / (_TWO_PI / bins) + 0.5))
    return ((k % n_yaw) + n_yaw) % n_yaw


# ----------------------------------------------------------------------------
# forward step + rollout
# ----------------------------------------------------------------------------
# step_predict/step_finalize hold the per-thread physics shared by the shared-terrain and
# batched-terrain kernels (Warp inlines @wp.func, so each kernel's codegen is identical to a
# hand-written one -- no cost for the feature it doesn't use). Terrain reads go through 2D views:
# the shared kernel passes the whole [ny,nx] field, the batched kernel passes its slice terrain[tid].
# The ONE op that can't take a view is the custom-grad settle, so it's called in the kernel between
# predict and finalize (settle on 2D, settle_bt on the full 3D array + index). No @wp.struct return
# (struct returns don't differentiate cleanly here): predict returns the pose, finalize writes the
# rest into the output arrays at tid.


@wp.func
def body_twist(robot: Robot, om: wp.vec3, alpha: float) -> wp.vec2:
    """Body-frame (forward speed, yaw rate) from the wheel speeds and the turn resistance alpha.

    Differential drive on the front pair; `alpha` (from the grip solve) widens the effective
    track, so it damps yaw only. The ONE place this mapping lives -- the integration in
    `step_predict` and the certificates in `step_finalize` must not drift apart.
    """
    vx = robot.wheel_radius * (om[0] + om[1]) / 2.0
    wz = robot.wheel_radius * (om[1] - om[0]) / (2.0 * robot.half_track * alpha)
    return wp.vec2(vx, wz)


@wp.func
def step_predict(
    env_i: wp.array2d(dtype=wp.float32),
    fric_i: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    om: wp.vec3,  # (wL, wR, w_rear) this step
    pc: wp.vec3,  # (x, y, yaw) current state
    tc: wp.vec3,  # (z, pitch, roll) current state
    tid: int,
    turn_out: wp.array(dtype=wp.vec2),  # [B] (alpha, x_icr) -> written at tid
) -> wp.vec4:
    """Grip-weighted ICR + turn resistance from the CURRENT pose, then Euler integrate. Write
    turn_out[tid]=(alpha, x_icr); return the predicted (pre-settle) pose and alpha as
    (xn, yn, yawn, alpha) -- `step_finalize` needs alpha to rebuild the twist, and returning it
    beats re-reading the output array inside the kernel (that would put a read-after-write on a
    grad-tracked buffer in the taped path)."""
    x = pc[0]
    y = pc[1]
    yaw = pc[2]
    R = euler_zyx(yaw, tc[1], tc[2])
    p = wp.vec3(x, y, tc[0])

    loads = normal_loads(env_i, grid, robot, R, p)  # per-wheel normal load N_i
    total_grip = float(0.0)  # Sum_i grip_i
    grip_x = float(0.0)  # Sum_i grip_i * wheel_x  (x_icr = grip_x / total_grip)
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_pos = robot.wheel_pos[st_i]
        wheel_center = p + R * wheel_pos
        n = sample_normal(env_i, grid, wheel_center[0], wheel_center[1])
        ct = wheel_center - robot.wheel_radius * n  # contact point
        grip = sample_field(fric_i, grid, ct[0], ct[1]) * loads[st_i]  # grip_i = mu_i * N_i
        total_grip += grip
        grip_x += grip * wheel_pos[0]
    x_icr = grip_x / total_grip  # grip-weighted ICR offset
    alpha = 1.0 + solver.k_turn * total_grip / (robot.gravity * robot.mass)  # turn resistance

    twist = body_twist(robot, om, alpha)
    vx = twist[0]
    wz = twist[1]
    vy = -x_icr * wz
    vw = R * wp.vec3(vx, vy, 0.0)
    turn_out[tid] = wp.vec2(alpha, x_icr)
    return wp.vec4(x + vw[0] * solver.dt, y + vw[1] * solver.dt, yaw + wz * solver.dt, alpha)


@wp.func
def step_finalize(
    env_i: wp.array2d(dtype=wp.float32),
    elev_i: wp.array2d(dtype=wp.float32),
    fric_i: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    pose_next: wp.vec3,  # predicted (xn, yn, yawn)
    settled: wp.vec3,  # settled (z, pitch, roll) of the new pose
    om: wp.vec3,  # (wL, wR, w_rear) realized this step
    alpha: float,  # turn resistance used this step (from step_predict)
    tid: int,
    controlled_next: wp.array(dtype=wp.vec3),  # [B] -> written at tid
    derived_next: wp.array(dtype=wp.vec3),
    loads_out: wp.array(dtype=wp.vec3),
    clear_out: wp.array(dtype=float),
    resid_out: wp.array(dtype=float),
    stability_out: wp.array(dtype=float),
    saturation_out: wp.array(dtype=float),
    stall_out: wp.array(dtype=float),
):
    """Write the NEW state + diagnostics at tid from the predicted pose and its settled tilt."""
    controlled_next[tid] = pose_next
    derived_next[tid] = settled
    xn = pose_next[0]
    yn = pose_next[1]
    yawn = pose_next[2]
    Rn = euler_zyx(yawn, settled[1], settled[2])
    pn = wp.vec3(xn, yn, settled[0])
    loads = normal_loads(env_i, grid, robot, Rn, pn)
    loads_out[tid] = loads
    stability_out[tid] = stability_margin(robot, loads)
    grip = contact_grip(env_i, fric_i, grid, robot, Rn, pn, loads)
    twist = body_twist(robot, om, alpha)
    saturation_out[tid] = friction_saturation(
        robot, loads, grip, settled[1], settled[2], twist[0], twist[1]
    )
    stall_out[tid] = torque_saturation(robot, settled[1])
    clear_out[tid] = chassis_clearance(elev_i, grid, robot, Rn, pn)
    cres = clearances(env_i, grid, robot, xn, yn, yawn, settled[0], settled[1], settled[2])
    resid_out[tid] = wp.max(wp.max(wp.abs(cres[0]), wp.abs(cres[1])), wp.abs(cres[2]))


@wp.kernel
def init_state_kernel(
    envelope: wp.array2d(dtype=wp.float32),  # [ny, nx] shared across the batch
    grid: Grid,
    robot: Robot,
    solver: Solver,
    start_pose: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw)
    controlled: wp.array2d(dtype=wp.vec3),  # [T+1, B] (x, y, yaw)      -> writes row 0
    derived: wp.array2d(dtype=wp.vec3),  # [T+1, B] (z, pitch, roll) -> writes row 0
):
    """Seed row 0 of a rollout (shared terrain): settle the start pose onto the terrain. Each
    thread is one rollout (tid = batch index), warm-started with z = envelope + wheel_radius."""
    tid = wp.tid()
    pc = start_pose[tid]
    z0 = sample_field(envelope, grid, pc[0], pc[1]) + robot.wheel_radius
    settled = settle(envelope, grid, robot, solver, pc, wp.vec3(z0, 0.0, 0.0))
    controlled[0, tid] = pc
    derived[0, tid] = settled


@wp.kernel
def init_state_kernel_bt(
    envelope: wp.array3d(dtype=wp.float32),  # [B, ny, nx] per-rollout terrain
    grid: Grid,
    robot: Robot,
    solver: Solver,
    start_pose: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw)
    controlled: wp.array2d(dtype=wp.vec3),  # [T+1, B] -> writes row 0
    derived: wp.array2d(dtype=wp.vec3),  # [T+1, B] -> writes row 0
):
    """Batched-terrain init: rollout tid settles its start pose on its own slice envelope[tid]."""
    tid = wp.tid()
    pc = start_pose[tid]
    z0 = sample_field(envelope[tid], grid, pc[0], pc[1]) + robot.wheel_radius
    settled = settle_bt(envelope, tid, grid, robot, solver, pc, wp.vec3(z0, 0.0, 0.0))
    controlled[0, tid] = pc
    derived[0, tid] = settled


@wp.kernel
def step_kernel(
    envelope: wp.array2d(dtype=wp.float32),  # [ny, nx] shared across the batch
    elevation: wp.array2d(dtype=wp.float32),
    friction: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    target_wheel_omega: wp.array(dtype=wp.vec3),  # [B] commanded (wL, wR, w_rear) this step
    current_wheel_omega_in: wp.array(dtype=wp.vec3),  # [B] lagged omega entering this step
    controlled: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw) current state
    derived: wp.array(dtype=wp.vec3),  # [B] (z, pitch, roll) current state
    current_wheel_omega_out: wp.array(dtype=wp.vec3),  # [B] lagged omega after this step -> written
    controlled_next: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw) settled NEW state -> written
    derived_next: wp.array(dtype=wp.vec3),  # [B] (z, pitch, roll) NEW state -> written
    loads_out: wp.array(dtype=wp.vec3),  # [B] N_i of the NEW state
    turn_out: wp.array(dtype=wp.vec2),  # [B] (alpha, x_icr) used this step
    clear_out: wp.array(dtype=float),  # [B] belly clearance of the NEW state
    resid_out: wp.array(dtype=float),  # [B] settle residual (max|c|) of the NEW state
    stability_out: wp.array(dtype=float),  # [B] min N_i / (m g) of the NEW state
    saturation_out: wp.array(dtype=float),  # [B] friction demand / budget of the NEW state
    stall_out: wp.array(dtype=float),  # [B] required torque / motor limit of the NEW state
):
    tid = wp.tid()
    tc = derived[tid]
    omega = motor_lag_step(
        current_wheel_omega_in[tid], target_wheel_omega[tid], solver.dt, solver.tau_motor
    )
    current_wheel_omega_out[tid] = omega
    pred = step_predict(
        envelope,
        friction,
        grid,
        robot,
        solver,
        omega,
        controlled[tid],
        tc,
        tid,
        turn_out,
    )
    pose_next = wp.vec3(pred[0], pred[1], pred[2])
    settled = settle(envelope, grid, robot, solver, pose_next, tc)
    step_finalize(
        envelope,
        elevation,
        friction,
        grid,
        robot,
        pose_next,
        settled,
        omega,
        pred[3],
        tid,
        controlled_next,
        derived_next,
        loads_out,
        clear_out,
        resid_out,
        stability_out,
        saturation_out,
        stall_out,
    )


@wp.kernel
def step_kernel_bt(
    envelope: wp.array3d(dtype=wp.float32),  # [B, ny, nx] per-rollout terrain
    elevation: wp.array3d(dtype=wp.float32),
    friction: wp.array3d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    target_wheel_omega: wp.array(dtype=wp.vec3),  # [B] commanded (wL, wR, w_rear) this step
    current_wheel_omega_in: wp.array(dtype=wp.vec3),  # [B] lagged omega entering this step
    controlled: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw) current state
    derived: wp.array(dtype=wp.vec3),  # [B] (z, pitch, roll) current state
    current_wheel_omega_out: wp.array(dtype=wp.vec3),  # [B] lagged omega after this step -> written
    controlled_next: wp.array(dtype=wp.vec3),  # [B] -> written
    derived_next: wp.array(dtype=wp.vec3),
    loads_out: wp.array(dtype=wp.vec3),
    turn_out: wp.array(dtype=wp.vec2),
    clear_out: wp.array(dtype=float),
    resid_out: wp.array(dtype=float),
    stability_out: wp.array(dtype=float),
    saturation_out: wp.array(dtype=float),
    stall_out: wp.array(dtype=float),
):
    """Batched-terrain step: rollout tid steps on its own slices; settle uses the full 3D array."""
    tid = wp.tid()
    tc = derived[tid]
    omega = motor_lag_step(
        current_wheel_omega_in[tid], target_wheel_omega[tid], solver.dt, solver.tau_motor
    )
    current_wheel_omega_out[tid] = omega
    pred = step_predict(
        envelope[tid],
        friction[tid],
        grid,
        robot,
        solver,
        omega,
        controlled[tid],
        tc,
        tid,
        turn_out,
    )
    pose_next = wp.vec3(pred[0], pred[1], pred[2])
    settled = settle_bt(envelope, tid, grid, robot, solver, pose_next, tc)
    step_finalize(
        envelope[tid],
        elevation[tid],
        friction[tid],
        grid,
        robot,
        pose_next,
        settled,
        omega,
        pred[3],
        tid,
        controlled_next,
        derived_next,
        loads_out,
        clear_out,
        resid_out,
        stability_out,
        saturation_out,
        stall_out,
    )


@wp.kernel
def rollout_kernel(
    n_steps: int,
    envelope: wp.array3d(dtype=wp.float32),  # [n_yaw, ny, nx] wheel envelope per yaw bin
    elevation: wp.array2d(dtype=wp.float32),
    friction: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    start_pose: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw)
    init_current_wheel_omega: wp.array(
        dtype=wp.vec3
    ),  # [B] initial lagged omega (e.g. encoder reading)
    target_wheel_omega: wp.array2d(dtype=wp.vec3),  # [T, B] commanded (wL, wR, w_rear)
    controlled: wp.array2d(dtype=wp.vec3),  # [T+1, B] (x, y, yaw)
    derived: wp.array2d(dtype=wp.vec3),  # [T+1, B] (z, pitch, roll)
    current_wheel_omega_out: wp.array2d(dtype=wp.vec3),  # [T+1, B] realized omega after lag
    loads_out: wp.array2d(dtype=wp.vec3),  # [T, B]
    turn_out: wp.array2d(dtype=wp.vec2),  # [T, B]
    clear_out: wp.array2d(dtype=float),  # [T, B]
    resid_out: wp.array2d(dtype=float),  # [T, B]
    stability_out: wp.array2d(dtype=float),  # [T, B] min N_i / (m g)
    saturation_out: wp.array2d(dtype=float),  # [T, B] friction demand / budget
    stall_out: wp.array2d(dtype=float),  # [T, B] required torque / motor limit
):
    """FORWARD-ONLY whole-rollout fusion: one thread per rollout walks all n_steps steps,
    carrying the state (pc, tc, current) in registers instead of round-tripping it through
    global memory between per-step launches (~1.2x faster than init_state_kernel +
    n_steps*step_kernel). This is the hot planning path; the differentiable/calibration
    path keeps the per-step step_kernel (the register carry is NOT auto-diffable --
    backprop needs the intermediate states this kernel overwrites).

    `envelope` is the yaw-binned stack; with the default spherical wheel it is one slice and
    `yaw_bin` is a constant 0, so this reads exactly the grid the 2D kernels read.

    MUST stay bit-identical to init_state_kernel + n_steps*step_kernel (guarded by
    tests/engine/step.selftest_rollout_kernel). Edit the physics in both.
    """
    b = wp.tid()
    n_yaw = envelope.shape[0]
    # init_state: settle the start pose -> row 0
    pc = start_pose[b]
    env_0 = envelope[yaw_bin(pc[2], n_yaw)]
    z0 = sample_field(env_0, grid, pc[0], pc[1]) + robot.wheel_radius
    tc = settle(env_0, grid, robot, solver, pc, wp.vec3(z0, 0.0, 0.0))
    current = init_current_wheel_omega[b]  # initial lagged omega carried in registers
    controlled[0, b] = pc
    derived[0, b] = tc
    current_wheel_omega_out[0, b] = current

    for t in range(n_steps):
        x = pc[0]
        y = pc[1]
        yaw = pc[2]
        R = euler_zyx(yaw, tc[1], tc[2])
        p = wp.vec3(x, y, tc[0])
        env_c = envelope[yaw_bin(yaw, n_yaw)]  # envelope slice of the CURRENT heading

        loads = normal_loads(env_c, grid, robot, R, p)  # per-wheel normal load N_i
        total_grip = float(0.0)  # Sum_i grip_i
        grip_x = float(0.0)  # Sum_i grip_i * wheel_x  (x_icr = grip_x / total_grip)
        for i in range(wp.static(3)):
            st_i = wp.static(i)
            wheel_pos = robot.wheel_pos[st_i]
            wheel_center = p + R * wheel_pos
            n = sample_normal(env_c, grid, wheel_center[0], wheel_center[1])
            ct = wheel_center - robot.wheel_radius * n  # contact point
            grip = sample_field(friction, grid, ct[0], ct[1]) * loads[st_i]  # grip_i = mu_i * N_i
            total_grip += grip
            grip_x += grip * wheel_pos[0]
        x_icr = grip_x / total_grip  # grip-weighted ICR offset
        alpha = 1.0 + solver.k_turn * total_grip / (robot.gravity * robot.mass)  # turn resistance

        # Apply lag first (update-then-use): tau_motor=0 gives current = target_wheel_omega[t] exactly.
        current = motor_lag_step(current, target_wheel_omega[t, b], solver.dt, solver.tau_motor)
        current_wheel_omega_out[t + 1, b] = current
        twist = body_twist(robot, current, alpha)
        vx = twist[0]
        wz = twist[1]
        vy = -x_icr * wz
        vw = R * wp.vec3(vx, vy, 0.0)
        xn = x + vw[0] * solver.dt
        yn = y + vw[1] * solver.dt
        yawn = yaw + wz * solver.dt

        pose_next = wp.vec3(xn, yn, yawn)
        env_n = envelope[yaw_bin(yawn, n_yaw)]  # envelope slice of the NEW heading
        settled = settle(env_n, grid, robot, solver, pose_next, tc)
        controlled[t + 1, b] = pose_next
        derived[t + 1, b] = settled

        Rn = euler_zyx(yawn, settled[1], settled[2])
        pn = wp.vec3(xn, yn, settled[0])
        loads = normal_loads(env_n, grid, robot, Rn, pn)
        loads_out[t, b] = loads
        stability_out[t, b] = stability_margin(robot, loads)
        grip_n = contact_grip(env_n, friction, grid, robot, Rn, pn, loads)
        saturation_out[t, b] = friction_saturation(
            robot, loads, grip_n, settled[1], settled[2], vx, wz
        )
        stall_out[t, b] = torque_saturation(robot, settled[1])
        turn_out[t, b] = wp.vec2(alpha, x_icr)
        clear_out[t, b] = chassis_clearance(elevation, grid, robot, Rn, pn)
        cres = clearances(env_n, grid, robot, xn, yn, yawn, settled[0], settled[1], settled[2])
        resid_out[t, b] = wp.max(wp.max(wp.abs(cres[0]), wp.abs(cres[1])), wp.abs(cres[2]))

        pc = pose_next  # carry state in registers (no global round-trip)
        tc = settled
