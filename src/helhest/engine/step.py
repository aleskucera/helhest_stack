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

_TWO_PI = wp.constant(2.0 * float(np.pi))

# Finite-difference step for the shear twist Jacobian [m/s and rad/s]. Small enough that the
# secant tracks the tangent, large enough to stay clear of float32 cancellation.
_SHEAR_FD_STEP = wp.constant(1.0e-4)

# Speed floor for the rolling-resistance direction [m/s]: the force must fall to zero at a
# standstill (a parked robot rolls nowhere), and this keeps the unit vector finite there.
_ROLL_FLOOR = wp.constant(1.0e-2)


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
    command_delay_steps: wp.int32  # whole-step transport delay on the wheel command; 0 = none
    shear_lk: wp.float32  # contact length / shear modulus; <= 0 = legacy kinematic traction
    yaw_tau: wp.float32  # [s] first-order yaw-rate lag on the legacy twist; 0 = off
    yaw_relax_len: wp.float32  # [m] relaxation LENGTH form of the same lag; 0 = off
    contact_patch: wp.float32  # contact patch radius [m], torsional term of the shear model
    shear_iters: wp.int32  # Newton iterations for the shear twist solve
    inertia_gain: wp.float32  # 1/dt when body momentum is on, 0 = quasi-static
    rolling_resistance: wp.float32  # resistance to rolling as a fraction of normal load


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
    # Transport delay [s] between issuing a wheel command and the wheels acting on it. MEASURED at
    # 189-249 ms on out_experiment_goal_unreachable0/1 (scripts/fit_actuator_lag.py), where the
    # response is essentially pure delay: the first-order `tau_motor` fits at 0.02-0.05 s, which is
    # a no-op at dt = 0.1 since its blend saturates. Quantised to whole steps at build(), so it is
    # only representable in multiples of dt. Default 0 = no delay, the pre-existing behaviour.
    command_delay: float = 0.0
    # --- traction model. <= 0 keeps the legacy kinematic twist (commanded speed always achieved,
    # friction only bends the turn). Above 0 switches to the quasi-static shear force balance, in
    # which the twist is solved: `shear_lk` is contact length / shear deformation modulus L/K
    # (soils put K at 0.01-0.06 m and L near 0.15 m, so L/K ~ 3-15), `contact_patch` is the patch
    # radius for the torsional term (0.075 m is the polar radius of a 0.10 x 0.24 m contact).
    # FITTED from the bags (scripts/fit_traction.py), against measured wheel speeds and a gyro --
    # no command in the loop, so no delay and no steady-state manoeuvre needed. On QUASI-STATIC
    # samples L/K lands at 8-15 (RMS 0.0393-0.0396 rad/s vs the legacy model's 0.0456, and 21%
    # better on median), which is also where the soil literature puts it. That fit predates
    # rolling_resistance and the two are coupled: with the measured mu_roll = 0.09 the yaw channel
    # wants L/K nearer 12. Refit both together if either is changed.
    # OPT-IN, and the reason is in that same fit: pooled over ALL turning samples both models sit
    # at RMS ~0.16 and are indistinguishable, because yaw-inertia transients carry roughly 4x the
    # variance of the traction difference. This model is real but it is not the dominant error.
    shear_lk: float = 0.0
    # Yaw-rate lag [s] for the LEGACY twist: the body's rotational inertia, which the kinematic
    # map omits by returning a steady-state yaw rate instantly. 0 = off (the pre-existing
    # behaviour). Fitted against Chrono at a converged step, not measured on the robot -- see
    # scripts/engine_ranking.py and PREREG_chrono_vehicle.md.
    yaw_tau: float = 0.0
    # Relaxation LENGTH [m]: the same yaw lag, but keyed to distance travelled instead of time,
    # tau_eff = yaw_relax_len / |v|. Rigid-body yaw inertia cannot be what the lag represents --
    # mu m g b / I_zz is 30 rad/s^2, so inertia settles in 0.033 s, a third of one planner step --
    # whereas a tyre's lateral force builds over DISTANCE, which lands in the right range at these
    # speeds. The two forms are distinguishable because only this one is speed-dependent. 0 = off.
    yaw_relax_len: float = 0.0
    contact_patch: float = 0.075
    shear_iters: int = 6
    # Body momentum, IMPLICITLY integrated inside the same twist solve (requires shear_lk > 0).
    # Explicit integration is not an option here: the shear curve makes the contacts stiff, with
    # time constants near 20 ms in translation and 8 ms in yaw, so an explicit step would force
    # dt ~ 5 ms and a 20x longer horizon -- the wall IMPROVEMENTS.md section 9(a) warns about.
    # Implicit Euler is unconditionally stable on this dissipative system and costs three extra
    # terms in a residual that is already being evaluated. False keeps the quasi-static solve,
    # which stays the reference case the numpy oracle in tests/engine/traction.py validates.
    body_momentum: bool = False
    # Rolling resistance as a fraction of normal load, opposing each contact's motion over the
    # ground (not its slip). MEASURED, not fitted: the torque calibration's fit offset is 36-38 Nm
    # total = ~106 N = 0.09 of this robot's weight (scripts/wheel_torque_from_bags.py). Without it
    # the shear model needs NO tractive force to drive straight, so it develops no longitudinal
    # slip and tracks the ground exactly -- against a measured forward gain of 0.906-0.925 on
    # Odin's SLAM odometry. No value of shear_lk can fix that; only this term can. Also what makes
    # the robot coast to a stop instead of drifting on when commands go to zero.
    rolling_resistance: float = 0.09

    def build(self) -> Solver:
        s = Solver()
        s.newton_iters = self.newton_iters
        s.atol = self.atol
        s.max_step = wp.vec3(self.max_step[0], self.max_step[1], self.max_step[2])
        s.tilt_clamp = self.tilt_clamp
        s.dt = self.dt
        s.k_turn = self.k_turn
        s.tau_motor = self.tau_motor
        s.command_delay_steps = int(round(self.command_delay / self.dt))
        s.shear_lk = self.shear_lk
        s.yaw_tau = self.yaw_tau
        s.yaw_relax_len = self.yaw_relax_len
        s.contact_patch = self.contact_patch
        s.shear_iters = self.shear_iters
        s.inertia_gain = (1.0 / self.dt) if self.body_momentum else 0.0
        s.rolling_resistance = self.rolling_resistance
        return s


@wp.func
def motor_lag_step(
    current_wheel_omega: wp.vec3, target_wheel_omega: wp.vec3, dt: float, tau: float
) -> wp.vec3:
    """First-order actuator lag: advance current_wheel_omega one timestep toward target_wheel_omega.

    alpha = 1 - exp(-dt/tau) is the EXACT step of a first-order lag held over dt, not the linear
    dt/tau. The two agree only while dt << tau: at the measured tau = 0.19 s they differ by 3% at
    dt = 0.01 but 29% at the planner's dt = 0.1, so the linear form would make the modelled wheels
    a third more responsive than the ones that were fitted. tau -> 0 still gives alpha = 1 and
    instantaneous tracking, so the default path is unchanged.
    """
    alpha = 1.0 - wp.exp(-dt / wp.max(tau, 1e-6))
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

    Row 0: force balance ALONG THE SURFACE NORMAL, Sum N_i = m g (n_bar . z).
    Rows 1-2: horizontal torque balance about the CoM, including the moment of the tangential
    (friction) reaction. Returns N = vec3(N0,N1,N2).

    THE TANGENTIAL REACTION. What holds the robot on a slope is not the contact normals alone:
    the contacts also supply an in-plane friction force, and it acts at the ground, BELOW the CoM,
    so it carries a moment about the CoM. Balancing normals only -- which this function used to do
    -- gets the load split wrong as soon as the ground tilts. Measured against Project Chrono
    (PREREG_chrono.md, scripts/chrono_compare.py): at 25 deg of pitch the least-loaded contact
    came out at 0.291 m g against Chrono's 0.042, an error of 0.249 m g = 259 N. Worse, on a side
    slope the old balance produced NO left/right transfer at all, and could not have: with
    parallel normals the split collapses to the CoM's barycentric weight, and the CoM sits on the
    centreline. Chrono transfers 0.405 m g by 25 deg of bank.

    The closure is that friction is shared in proportion to normal load, f_i = (N_i / S) F_t with
    S = Sum N_i. The total tangential force then follows from force balance, F_t = m g z - S n_bar,
    and the moment it contributes is LINEAR in N_i, so this stays a 3x3 solve at the same cost.
    Resolving the force balance along n_bar instead of vertically is what makes S right: friction
    has a vertical component on a slope, so the normals alone do not carry the full weight. The
    old row gave Sum N_i = m g / (cos pitch cos roll), a 10% overshoot at 25 deg, where the truth
    is m g cos(tilt) -- which Chrono confirms to 1e-4.

    On FLAT ground n_bar = z, S = m g and F_t = 0, so every coefficient reduces to the previous
    one and the result is bit-identical. Only sloped terrain moves.
    """
    com_world = p + R * robot.com
    weight = robot.mass * robot.gravity

    normals = wp.mat33()  # rows: the three contact normals
    arms = wp.mat33()  # rows: the three moment arms about the CoM
    n_sum = wp.vec3()
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_center = p + R * robot.wheel_pos[st_i]
        n = sample_normal(envelope, grid, wheel_center[0], wheel_center[1])
        r = (wheel_center - robot.wheel_radius * n) - com_world  # contact point, then moment arm
        for k in range(wp.static(3)):
            st_k = wp.static(k)
            normals[st_i, st_k] = n[st_k]
            arms[st_i, st_k] = r[st_k]
        n_sum += n

    n_bar = wp.normalize(n_sum)
    # total normal load and the in-plane reaction that holds the robot on the slope
    load_sum = weight * n_bar[2]
    tangential = wp.vec3(0.0, 0.0, weight) - load_sum * n_bar
    # guard: on a near-vertical face load_sum collapses and the 1/S weighting would blow up
    inv_sum = 1.0 / wp.max(load_sum, 1.0e-3 * weight)

    A = wp.mat33()
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        n = wp.vec3(normals[st_i, 0], normals[st_i, 1], normals[st_i, 2])
        r = wp.vec3(arms[st_i, 0], arms[st_i, 1], arms[st_i, 2])
        m = wp.cross(r, n) + inv_sum * wp.cross(r, tangential)
        A[0, st_i] = 1.0
        A[1, st_i] = m[0]
        A[2, st_i] = m[1]

    return solve3(A, wp.vec3(load_sum, 0.0, 0.0))


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
def integrate_pose(pose: wp.vec3, world_vel: wp.vec3, yaw_rate: float, dt: float) -> wp.vec3:
    """Advance (x, y, yaw) one step along the EXACT arc of a constant twist.

    Within a step the body twist is constant, so the true path is a circular arc, not the straight
    chord forward Euler takes. Euler holds the heading fixed across the step and is first order in
    dt: on flat ground at 2.1 m/s it lands 10-19 cm off the analytic arc over a 2.5 s horizon, and
    halving dt only halves that (measured in tests/engine/integrator.py). Integrating the arc in
    closed form is EXACT for a constant twist at any dt, for about ten extra flops.

    Rotating the world velocity back by yaw gives the (constant) velocity in the yaw-free frame;
    integrating Rz(yaw + psi_dot t) across the step then contributes

        sin(theta) / psi_dot        and       (1 - cos(theta)) / psi_dot,   theta = psi_dot dt

    which tend to (dt, 0) as psi_dot -> 0 -- the Euler update, recovered exactly. The small-angle
    branch uses those limits directly to avoid 0/0; it matches the series to first order, so the
    derivative stays continuous for the taped path.

    `world_vel` already carries the body's pitch/roll projection, and both stay fixed across the
    step (the settle updates them afterwards), so only the yaw rotation has to be integrated.
    """
    yaw = pose[2]
    cos_yaw = wp.cos(yaw)
    sin_yaw = wp.sin(yaw)
    # world velocity -> yaw-free frame, where it is constant over the step
    u_x = cos_yaw * world_vel[0] + sin_yaw * world_vel[1]
    u_y = -sin_yaw * world_vel[0] + cos_yaw * world_vel[1]

    theta = yaw_rate * dt
    integral_cos = dt  # int_0^dt cos(psi_dot t) dt
    integral_sin = 0.5 * theta * dt  # int_0^dt sin(psi_dot t) dt
    if wp.abs(theta) > 1.0e-6:
        integral_cos = wp.sin(theta) / yaw_rate
        integral_sin = (1.0 - wp.cos(theta)) / yaw_rate

    local_x = integral_cos * u_x - integral_sin * u_y
    local_y = integral_sin * u_x + integral_cos * u_y
    return wp.vec3(
        pose[0] + cos_yaw * local_x - sin_yaw * local_y,
        pose[1] + sin_yaw * local_x + cos_yaw * local_y,
        yaw + theta,
    )


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
def _shear_residual(
    robot: Robot,
    mu: wp.vec3,  # per-wheel friction coefficient at the contact
    loads: wp.vec3,  # per-wheel normal load N_i
    om: wp.vec3,  # per-wheel driven speed (wL, wR, w_rear)
    twist: wp.vec3,  # candidate body twist (vx, vy, yaw_rate)
    shear_lk: float,  # contact length / shear deformation modulus, L/K
    patch: float,  # contact patch radius [m], for the torsional term
    gravity_tangential: wp.vec2,  # body-frame in-plane weight component
    previous: wp.vec3,  # last step's twist; unused when inertia_gain is 0
    inertia_gain: float,  # 1/dt for implicit momentum, 0 for the quasi-static solve
    rolling_resistance: float,  # fraction of normal load resisting travel over the ground
) -> wp.vec3:
    """Net (Fx, Fy, Mz) on the body for a candidate twist. Zero at equilibrium.

    Each contact slips at s = (body velocity there) - (driven rim speed); the ground opposes it
    with a force whose MAGNITUDE follows the Janosi-Hanamoto shear curve rather than jumping to
    mu*N at infinitesimal slip:

        lambda = (|s| / |R w|) (L/K)          mobilised = 1 - (1 - exp(-lambda)) / lambda

    lambda is a slip RATIO -- slip velocity over rolling speed. That is what makes this model
    sensitive to forward speed at fixed differential (rigid Coulomb is not: adding a common
    speed leaves every slip velocity unchanged, so it cannot produce any speed dependence).
    It also removes the need for a slip regulariser: the force goes to zero smoothly with |s|.

    The spin channel (patch * yaw_rate) carries the torsional resistance of a finite contact
    patch, which a point contact has none of.
    """
    forward = twist[0]
    lateral = twist[1]
    yaw_rate = twist[2]
    total_x = gravity_tangential[0]
    total_y = gravity_tangential[1]
    total_mz = float(0.0)
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel = robot.wheel_pos[st_i]
        x = wheel[0]
        y = wheel[1]
        slip_x = forward - yaw_rate * y - robot.wheel_radius * om[st_i]
        slip_y = lateral + yaw_rate * x
        slip_spin = patch * yaw_rate
        norm = wp.sqrt(slip_x * slip_x + slip_y * slip_y + slip_spin * slip_spin)

        rolling = wp.abs(robot.wheel_radius * om[st_i])
        lam = float(1.0e6)  # a wheel that is not rolling simply slides: fully mobilised
        if rolling > 1.0e-6:
            lam = wp.max(norm / rolling * shear_lk, 1.0e-9)
        mobilised = 1.0 - (1.0 - wp.exp(-lam)) / lam

        scale = float(0.0)
        if norm > 1.0e-9:
            scale = mu[st_i] * loads[st_i] * mobilised / norm
        force_x = -scale * slip_x
        force_y = -scale * slip_y
        # rolling resistance opposes the contact's TRAVEL over the ground, not its slip against
        # the rim -- so it is present even in perfect rolling, and vanishes at a standstill.
        travel_x = forward - yaw_rate * y
        travel_y = lateral + yaw_rate * x
        travel = wp.sqrt(travel_x * travel_x + travel_y * travel_y)
        roll_scale = rolling_resistance * loads[st_i] / (travel + _ROLL_FLOOR)
        force_x -= roll_scale * travel_x
        force_y -= roll_scale * travel_y

        total_x += force_x
        total_y += force_y
        total_mz += x * force_y - y * force_x - scale * patch * slip_spin
    # a steady turn is not equilibrium: the body accelerates centripetally (a = omega x v)
    total_x -= robot.mass * (-lateral * yaw_rate)
    total_y -= robot.mass * (forward * yaw_rate)
    # implicit body momentum: the unknown twist is the one at the END of the step, so a backward
    # difference against the previous twist appears in the residual. inertia_gain = 0 removes it
    # and recovers the massless quasi-static balance exactly.
    total_x -= robot.mass * inertia_gain * (forward - previous[0])
    total_y -= robot.mass * inertia_gain * (lateral - previous[1])
    total_mz -= robot.yaw_inertia * inertia_gain * (yaw_rate - previous[2])
    return wp.vec3(total_x, total_y, total_mz)


@wp.func
def shear_twist(
    robot: Robot,
    solver: Solver,
    mu: wp.vec3,
    loads: wp.vec3,
    om: wp.vec3,
    gravity_tangential: wp.vec2,
    previous: wp.vec3,
    guess: wp.vec3,
) -> wp.vec3:
    """Solve the 3x3 force/moment balance for the body twist (vx, vy, yaw_rate).

    Same shape as the settle: fixed iteration count, no data-dependent branching, runs in
    registers. The Jacobian is finite-differenced -- the analytic form of d(mobilised * slip
    direction)/d(twist) is available but was not worth the risk while this is opt-in. Warm-started
    from the kinematic twist, and each step is clamped so a bad Jacobian cannot throw the solve.
    """
    twist = guess
    for _ in range(solver.shear_iters):
        residual = _shear_residual(
            robot,
            mu,
            loads,
            om,
            twist,
            solver.shear_lk,
            solver.contact_patch,
            gravity_tangential,
            previous,
            solver.inertia_gain,
            solver.rolling_resistance,
        )
        jac = wp.mat33()
        for k in range(wp.static(3)):
            st_k = wp.static(k)
            probe = twist
            probe[st_k] = probe[st_k] + _SHEAR_FD_STEP
            shifted = _shear_residual(
                robot,
                mu,
                loads,
                om,
                probe,
                solver.shear_lk,
                solver.contact_patch,
                gravity_tangential,
                previous,
                solver.inertia_gain,
                solver.rolling_resistance,
            )
            jac[0, st_k] = (shifted[0] - residual[0]) / _SHEAR_FD_STEP
            jac[1, st_k] = (shifted[1] - residual[1]) / _SHEAR_FD_STEP
            jac[2, st_k] = (shifted[2] - residual[2]) / _SHEAR_FD_STEP
        delta = solve3(jac, residual)
        step = wp.vec3(
            wp.clamp(delta[0], -1.0, 1.0),
            wp.clamp(delta[1], -1.0, 1.0),
            wp.clamp(delta[2], -1.0, 1.0),
        )
        # Backtracking with a FIXED trial count: evaluate the full step and three halvings, keep
        # whichever lowers the residual most. Uniform work per thread (no data-dependent loop, so
        # graph capture and warp coherence are unaffected), and necessary rather than decorative:
        # a raw Newton step diverges once the shear curve saturates, because the force magnitude
        # then stops depending on the twist and only its DIRECTION does, leaving the Jacobian
        # nearly singular. That stiffness is the rigid-Coulomb limit reasserting itself.
        best = twist
        best_norm = wp.length(residual)
        scale = float(1.0)
        for _trial in range(wp.static(4)):
            candidate = twist - scale * step
            trial_norm = wp.length(
                _shear_residual(
                    robot,
                    mu,
                    loads,
                    om,
                    candidate,
                    solver.shear_lk,
                    solver.contact_patch,
                    gravity_tangential,
                    previous,
                    solver.inertia_gain,
                    solver.rolling_resistance,
                )
            )
            if trial_norm < best_norm:
                best_norm = trial_norm
                best = candidate
            scale = scale * 0.5
        twist = best
    return twist


@wp.func
def traction_twist(
    robot: Robot,
    solver: Solver,
    mu: wp.vec3,
    loads: wp.vec3,
    om: wp.vec3,
    tilt: wp.vec2,  # (pitch, roll) of the current pose
    alpha: float,  # legacy turn resistance, computed by the caller
    x_icr: float,  # legacy grip-weighted ICR offset, computed by the caller
    previous: wp.vec3,  # last step's twist; only read when body momentum is on
) -> wp.vec3:
    """Body twist (vx, vy, yaw_rate) from the wheel speeds, by whichever traction model is on.

    solver.shear_lk <= 0 (the default) keeps the LEGACY kinematic model: the commanded forward
    speed is always achieved and friction only bends the turn through alpha = 1 + k_turn * grip
    and the grip-weighted ICR. Above 0 it solves the shear force balance instead, in which the
    twist -- forward speed included -- is an OUTPUT, gravity enters directly (so a slope produces
    drift), and alpha/x_icr become emergent rather than parameters.
    """
    if solver.shear_lk <= 0.0:
        legacy = body_twist(robot, om, alpha)
        yaw_rate = legacy[1]
        # YAW LAG. The kinematic map is a STEADY-STATE relation: it returns the yaw rate the robot
        # would eventually hold, and returns it instantly. Over a planning horizon the command
        # changes every step and the body never gets there, so the quasi-static model over-rotates
        # on exactly the manoeuvres MPPI samples -- measured against Chrono, the endpoint error
        # grows from 0.08 m on near-straight candidates to 0.31 m on hard turns.
        #
        # It is NOT rigid-body yaw inertia, despite looking like it. mu m g b / I_zz is 30 rad/s^2,
        # so inertia settles in 0.033 s -- a third of one planner step, invisible at dt = 0.1 --
        # while what fits is nearer 0.25 s. Keying the lag to DISTANCE instead of time
        # (`yaw_relax_len`, a tyre relaxation length) fits better and at the bag-measured alpha:
        # 0.051 m against 0.073 m unlagged and 0.059 m for the constant-time form, and the two are
        # separable because only the distance form matches the fast candidates (0.059 m against
        # 0.074 m) while both match the slow ones. Implicit, so stable at any dt.
        tau = solver.yaw_tau
        if solver.yaw_relax_len > 0.0:
            # distance-keyed: tau = sigma / |v|, floored so a stationary robot cannot divide by 0
            tau = solver.yaw_relax_len / wp.max(wp.abs(legacy[0]), 0.05)
        if tau > 0.0:
            blend = solver.dt / (solver.dt + tau)
            yaw_rate = previous[2] + blend * (yaw_rate - previous[2])
        return wp.vec3(legacy[0], -x_icr * yaw_rate, yaw_rate)
    kinematic = body_twist(robot, om, 1.0)  # warm start: the ideal differential-drive twist
    weight = robot.mass * robot.gravity
    cos_pitch = wp.cos(tilt[0])
    gravity_tangential = wp.vec2(weight * wp.sin(tilt[0]), -weight * cos_pitch * wp.sin(tilt[1]))
    # under momentum the previous twist is much the better warm start -- the body cannot have
    # moved far in one step, which is the entire point of carrying it
    guess = wp.vec3(kinematic[0], 0.0, kinematic[1] * 0.5)
    if solver.inertia_gain > 0.0:
        guess = previous
    return shear_twist(robot, solver, mu, loads, om, gravity_tangential, previous, guess)


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
    prev_twist: wp.vec3,  # body twist entering this step (only read under body momentum)
    tid: int,
    turn_out: wp.array(dtype=wp.vec2),  # [B] (alpha, x_icr) -> written at tid
    twist_out: wp.array(dtype=wp.vec3),  # [B] solved (vx, vy, yaw_rate) -> written at tid
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
    mu = wp.vec3()  # per-wheel friction at the contact (the shear model needs them separately)
    total_grip = float(0.0)  # Sum_i grip_i
    grip_x = float(0.0)  # Sum_i grip_i * wheel_x  (x_icr = grip_x / total_grip)
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_pos = robot.wheel_pos[st_i]
        wheel_center = p + R * wheel_pos
        n = sample_normal(env_i, grid, wheel_center[0], wheel_center[1])
        ct = wheel_center - robot.wheel_radius * n  # contact point
        mu[st_i] = sample_field(fric_i, grid, ct[0], ct[1])
        grip = mu[st_i] * loads[st_i]  # grip_i = mu_i * N_i
        total_grip += grip
        grip_x += grip * wheel_pos[0]

    x_icr = grip_x / total_grip  # grip-weighted ICR offset
    alpha = 1.0 + solver.k_turn * total_grip / (robot.gravity * robot.mass)  # turn resistance

    twist = traction_twist(
        robot, solver, mu, loads, om, wp.vec2(tc[1], tc[2]), alpha, x_icr, prev_twist
    )
    twist_out[tid] = twist
    vx = twist[0]
    vy = twist[1]
    wz = twist[2]
    vw = R * wp.vec3(vx, vy, 0.0)
    # `turning` stays (alpha, x_icr) under BOTH models. Under the shear model they are emergent --
    # what the solved twist implies -- rather than inputs to it.
    if solver.shear_lk > 0.0 and wp.abs(wz) > 1.0e-9:
        alpha = body_twist(robot, om, 1.0)[1] / wz
        x_icr = -vy / wz
    turn_out[tid] = wp.vec2(alpha, x_icr)
    next_pose = integrate_pose(pc, vw, wz, solver.dt)
    return wp.vec4(next_pose[0], next_pose[1], next_pose[2], alpha)


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
    twist_in: wp.array(dtype=wp.vec3),  # [B] body twist entering this step
    current_wheel_omega_out: wp.array(dtype=wp.vec3),  # [B] lagged omega after this step -> written
    controlled_next: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw) settled NEW state -> written
    derived_next: wp.array(dtype=wp.vec3),  # [B] (z, pitch, roll) NEW state -> written
    loads_out: wp.array(dtype=wp.vec3),  # [B] N_i of the NEW state
    turn_out: wp.array(dtype=wp.vec2),  # [B] (alpha, x_icr) used this step
    clear_out: wp.array(dtype=float),  # [B] belly clearance of the NEW state
    resid_out: wp.array(dtype=float),  # [B] settle residual (max|c|) of the NEW state
    twist_out: wp.array(dtype=wp.vec3),  # [B] solved body twist -> written (momentum state)
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
        twist_in[tid],
        tid,
        turn_out,
        twist_out,
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
    twist_in: wp.array(dtype=wp.vec3),  # [B] body twist entering this step
    current_wheel_omega_out: wp.array(dtype=wp.vec3),  # [B] lagged omega after this step -> written
    controlled_next: wp.array(dtype=wp.vec3),  # [B] -> written
    derived_next: wp.array(dtype=wp.vec3),
    loads_out: wp.array(dtype=wp.vec3),
    turn_out: wp.array(dtype=wp.vec2),
    clear_out: wp.array(dtype=float),
    resid_out: wp.array(dtype=float),
    twist_out: wp.array(dtype=wp.vec3),
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
        twist_in[tid],
        tid,
        turn_out,
        twist_out,
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
    command_history: wp.array2d(dtype=wp.vec3),  # [>=1, B] commands already in flight, oldest first
    init_twist: wp.array(dtype=wp.vec3),  # [B] body twist at rollout start (momentum only)
    controlled: wp.array2d(dtype=wp.vec3),  # [T+1, B] (x, y, yaw)
    derived: wp.array2d(dtype=wp.vec3),  # [T+1, B] (z, pitch, roll)
    current_wheel_omega_out: wp.array2d(dtype=wp.vec3),  # [T+1, B] realized omega after lag
    loads_out: wp.array2d(dtype=wp.vec3),  # [T, B]
    turn_out: wp.array2d(dtype=wp.vec2),  # [T, B]
    clear_out: wp.array2d(dtype=float),  # [T, B]
    resid_out: wp.array2d(dtype=float),  # [T, B]
    twist_out: wp.array2d(dtype=wp.vec3),  # [T+1, B] solved body twist (momentum state)
):
    """FORWARD-ONLY whole-rollout fusion: one thread per rollout walks all n_steps steps,
    carrying the state (pc, tc, current) in registers instead of round-tripping it through
    global memory between per-step launches (~1.2x faster than init_state_kernel +
    n_steps*step_kernel). This is the hot planning path; the differentiable/calibration
    path keeps the per-step step_kernel (the register carry is NOT auto-diffable --
    backprop needs the intermediate states this kernel overwrites).

    `envelope` is the yaw-binned stack; with the default spherical wheel it is one slice and
    `yaw_bin` is a constant 0, so this reads exactly the grid the 2D kernels read.

    With `solver.command_delay_steps = n > 0` the wheels act on the command issued n steps ago:
    step t applies `command_history[t]` while t < n (the commands already in flight when the
    rollout started, oldest first) and `target_wheel_omega[t - n]` afterwards. At n = 0 the branch
    always takes target_wheel_omega[t] and `command_history` is never read, so the default path is
    unchanged. The per-step `step_kernel` does NOT do this -- there the caller supplies whichever
    command should act on that step.

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
    twist_state = init_twist[b]  # body twist carried in registers alongside it
    twist_out[0, b] = twist_state
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
        mu = wp.vec3()  # per-wheel friction at the contact
        total_grip = float(0.0)  # Sum_i grip_i
        grip_x = float(0.0)  # Sum_i grip_i * wheel_x  (x_icr = grip_x / total_grip)
        for i in range(wp.static(3)):
            st_i = wp.static(i)
            wheel_pos = robot.wheel_pos[st_i]
            wheel_center = p + R * wheel_pos
            n = sample_normal(env_c, grid, wheel_center[0], wheel_center[1])
            ct = wheel_center - robot.wheel_radius * n  # contact point
            mu[st_i] = sample_field(friction, grid, ct[0], ct[1])
            grip = mu[st_i] * loads[st_i]  # grip_i = mu_i * N_i
            total_grip += grip
            grip_x += grip * wheel_pos[0]

        # Transport delay: the wheels act on a command issued solver.command_delay_steps ago.
        commanded = target_wheel_omega[t, b]
        if t < solver.command_delay_steps:
            commanded = command_history[t, b]  # already in flight when this rollout started
        elif solver.command_delay_steps > 0:
            commanded = target_wheel_omega[t - solver.command_delay_steps, b]
        # Apply lag first (update-then-use): tau_motor=0 gives current = commanded exactly.
        current = motor_lag_step(current, commanded, solver.dt, solver.tau_motor)
        current_wheel_omega_out[t + 1, b] = current
        x_icr = grip_x / total_grip  # grip-weighted ICR offset
        alpha = 1.0 + solver.k_turn * total_grip / (robot.gravity * robot.mass)  # turn resistance

        twist = traction_twist(
            robot, solver, mu, loads, current, wp.vec2(tc[1], tc[2]), alpha, x_icr, twist_state
        )
        twist_state = twist
        twist_out[t + 1, b] = twist
        vx = twist[0]
        vy = twist[1]
        wz = twist[2]
        # `turning` stays (alpha, x_icr) under BOTH models; under the shear model they are
        # emergent -- what the solved twist implies -- rather than inputs to it.
        if solver.shear_lk > 0.0 and wp.abs(wz) > 1.0e-9:
            alpha = body_twist(robot, current, 1.0)[1] / wz
            x_icr = -vy / wz
        vw = R * wp.vec3(vx, vy, 0.0)
        pose_next = integrate_pose(pc, vw, wz, solver.dt)
        xn = pose_next[0]
        yn = pose_next[1]
        yawn = pose_next[2]
        env_n = envelope[yaw_bin(yawn, n_yaw)]  # envelope slice of the NEW heading
        settled = settle(env_n, grid, robot, solver, pose_next, tc)
        controlled[t + 1, b] = pose_next
        derived[t + 1, b] = settled

        Rn = euler_zyx(yawn, settled[1], settled[2])
        pn = wp.vec3(xn, yn, settled[0])
        loads = normal_loads(env_n, grid, robot, Rn, pn)
        loads_out[t, b] = loads
        turn_out[t, b] = wp.vec2(alpha, x_icr)
        clear_out[t, b] = chassis_clearance(elevation, grid, robot, Rn, pn)
        cres = clearances(env_n, grid, robot, xn, yn, yawn, settled[0], settled[1], settled[2])
        resid_out[t, b] = wp.max(wp.max(wp.abs(cres[0]), wp.abs(cres[1])), wp.abs(cres[2]))

        pc = pose_next  # carry state in registers (no global round-trip)
        tc = settled
