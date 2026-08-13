"""Robot model: geometry + mass, host params and the device-side `Robot` struct.

`RobotParams` (host dataclass, what you tune) builds into `Robot` (a `@wp.struct`
of read-only constants passed into the kernels). The numpy geometry/mass also live
in the top-level `model.py` for the reference/viz paths; these are the device twin.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

# --- provenance: where the default mass/com come from (common.py). Not used at runtime. ---
_MASSES = np.array(
    [
        [-0.13, 0.0, 0.0, 78.8375],  # front box
        [-0.61, 0.0, 0.0, 10.8625],  # rear box
        [0.0, 0.365, 0.0, 5.5],  # left wheel
        [0.0, -0.365, 0.0, 5.5],  # right wheel
        [-0.75, 0.0, 0.0, 5.5],  # rear wheel
    ]
)
DEFAULT_MASS = float(_MASSES[:, 3].sum())  # 106.2 kg
DEFAULT_COM = (_MASSES[:, :3] * _MASSES[:, 3:4]).sum(0) / DEFAULT_MASS  # x≈-0.198
# Own yaw inertia of each body about its OWN centre: the two chassis boxes as uniform slabs
# m(a^2+b^2)/12 at 0.48x0.56 and 0.48x0.24, and each wheel as a cylinder about a vertical axis
# (the transverse inertia, m(3r^2+h^2)/12 = 0.173 for r=0.35, h=0.10, m=5.5).
_OWN_YAW_INERTIA = np.array(
    [
        78.8375 * (0.48**2 + 0.56**2) / 12.0,
        10.8625 * (0.48**2 + 0.24**2) / 12.0,
        0.173021,
        0.173021,
        0.173021,
    ]
)
# I_zz about the CoM: each body's own inertia plus its parallel-axis term. ~10.1 kg m^2. Derived
# from the same table the mass and CoM come from, so it cannot drift away from them.
DEFAULT_YAW_INERTIA = float(
    (
        _OWN_YAW_INERTIA
        + _MASSES[:, 3]
        * ((_MASSES[:, 0] - DEFAULT_COM[0]) ** 2 + (_MASSES[:, 1] - DEFAULT_COM[1]) ** 2)
    ).sum()
)


@wp.struct
class Robot:
    """Device-side robot constants, passed into kernels as one struct.

    Safe as a wp.struct: `wheel_pos`/`chassis_pts` are read-only (not
    differentiated), so the struct-autodiff limitation does not apply. Only the
    differentiated grids (height, friction) stay plain top-level kernel args.
    """

    wheel_pos: wp.array(dtype=wp.vec3)  # [3] left/right/rear
    chassis_pts: wp.array(dtype=wp.vec3)  # [Np] belly non-penetration samples
    n_chassis: wp.int32  # len(chassis_pts); struct-member .shape is unreliable on CUDA
    wheel_radius: wp.float32
    half_track: wp.float32
    com: wp.vec3
    mass: wp.float32
    yaw_inertia: wp.float32  # [kg m^2] about the CoM; only the momentum traction model reads it
    gravity: wp.float32
    # --- planning capabilities (mirror of the RobotParams fields; the dynamics kernels don't read
    # these, but carrying them on the built struct lets the planner read one object). ---
    min_turn_radius: wp.float32
    max_roll: wp.float32  # radians
    max_pitch_up: wp.float32
    max_pitch_down: wp.float32
    clear_margin: wp.float32
    resid_tol: wp.float32
    roll_cost_weight: wp.float32
    pitch_cost_weight: wp.float32


@dataclass(frozen=True)
class RobotParams:  # host-side robot knobs — what you nudge
    wheel_radius: float = 0.35
    # Wheel WIDTH [m]. None keeps the spherical wheel envelope, which reaches a full wheel_radius
    # sideways and so lifts the robot over rocks it would really straddle; a float switches
    # ForwardSimulator to the yaw-binned cylinder envelope. The REAL wheel is 0.10 m wide
    # (ruler-measured; ostrich examples/helhest_junior/robot_parameters.md section 6, collision
    # cylinder r = 0.35, half-height 0.05) -- so the sphere over-reaches sideways by 7x. Left at
    # DEFAULT since 2026-08-10: the CYLINDER envelope at the ruler-measured 0.10 m tread. The
    # sphere reaches the full 0.35 m radius sideways, so a rock 0.3 m beside the wheel lifts and
    # tilts the robot when in reality it is straddled -- systematically pessimistic in tight and
    # rocky places. Set to None for the sphere.
    #
    # This half-width is also the SAFETY-MARGIN dial, and a far better one than the sphere: the
    # sphere's 0.35 m of lateral margin is fixed and unavoidable, whereas 0.10 is honest, 0.15
    # keeps 5 cm of margin per side and 0.20 keeps 10 cm -- all still far tighter than the sphere.
    # Prefer widening this, or clear_margin / max_roll, over going back to the sphere.
    #
    # NOT usable with DifferentiableSimulator: the taped settle would need a yaw index threaded
    # through it, so the gradient paths pass wheel_width=None explicitly.
    wheel_width: float | None = 0.10
    half_track: float = 0.365
    rear_offset: float = 0.75
    gravity: float = 9.81
    mass: float = DEFAULT_MASS
    yaw_inertia: float = DEFAULT_YAW_INERTIA  # [kg m^2] derived from the mass table, not measured
    com: tuple = (float(DEFAULT_COM[0]), 0.0, 0.0)  # full vec3, independent of mass
    chassis_nx: int = 3
    chassis_ny: int = 3
    # --- planning capabilities: the robot's own limits. build() copies these into the device Robot
    # struct, so the cost-to-go feasibility AND the MPPI cost kernels read one shared source. ---
    # tightest forward arc the planner assumes (skid-steer maneuverability)
    min_turn_radius: float = 0.5
    # [rad] lateral tip-over limit (symmetric; narrow track -> strict)
    max_roll: float = np.radians(15.0)
    max_pitch_up: float = np.radians(25.0)  # [rad] climbing limit (nose UP, pitch < 0)
    # [rad] descending limit (nose DOWN, pitch > 0; front-heavy)
    max_pitch_down: float = np.radians(15.0)
    # min belly-terrain gap [m]; below it the pose is infeasible (high-centers)
    clear_margin: float = 0.05
    # settle residual above which the pose is infeasible (can't find a resting pose)
    resid_tol: float = 1e-2
    # graded cost-to-go penalty per radian of tilt -- roll weighted MORE than pitch (roll is the
    # dangerous axis), so among feasible poses the router prefers low-roll lines (attack slopes head-on).
    roll_cost_weight: float = 1.0
    pitch_cost_weight: float = 0.5

    def build(self, device="cuda") -> Robot:
        b, l = self.half_track, self.rear_offset
        wheel_pos = np.array([[0, b, 0], [0, -b, 0], [-l, 0, 0]], np.float32)
        cpts = self._chassis_pts()
        r = Robot()
        r.wheel_pos = wp.array(wheel_pos, dtype=wp.vec3, device=device)
        r.chassis_pts = wp.array(cpts, dtype=wp.vec3, device=device)
        r.n_chassis = int(cpts.shape[0])
        r.wheel_radius = self.wheel_radius
        r.half_track = self.half_track
        r.com = wp.vec3(*self.com)
        r.mass = self.mass
        r.yaw_inertia = self.yaw_inertia
        r.gravity = self.gravity
        r.min_turn_radius = self.min_turn_radius
        r.max_roll = self.max_roll
        r.max_pitch_up = self.max_pitch_up
        r.max_pitch_down = self.max_pitch_down
        r.clear_margin = self.clear_margin
        r.resid_tol = self.resid_tol
        r.roll_cost_weight = self.roll_cost_weight
        r.pitch_cost_weight = self.pitch_cost_weight
        return r

    def _chassis_pts(self):
        boxes = [(-0.13, 0.0, 0.0, 0.24, 0.28, 0.10), (-0.61, 0.0, 0.0, 0.24, 0.12, 0.10)]
        pts = [
            [cx + sx * hx, cy + sy * hy, cz - hz]
            for cx, cy, cz, hx, hy, hz in boxes
            for sx in np.linspace(-1, 1, self.chassis_nx)
            for sy in np.linspace(-1, 1, self.chassis_ny)
        ]
        return np.array(pts, np.float32)
