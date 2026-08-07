"""Implicit body momentum inside the shear traction solve.

Run:  python -m tests.engine.momentum

The quasi-static solve answers "what twist balances the forces". With `body_momentum` it answers
"what twist at the END of this step is consistent with the forces AND with where the body already
was" -- implicit Euler, so the inertia terms sit inside the same 3x3 Newton rather than needing a
separate integration.

Implicit is not a stylistic choice. Near zero slip the shear curve is stiff -- force rises over a
slip scale of R*omega/(L/K), so dF/dv reaches ~1700 N per m/s per contact and m / sum(dF/dv) is
around 20 ms. An explicit step would need dt well under that, i.e. a 20x longer horizon for the
same 2.5 s, which is the timestep wall in IMPROVEMENTS.md section 9(a). Implicit Euler is
unconditionally stable on a dissipative system and costs three extra terms in a residual that is
evaluated anyway; `selftest_stability` confirms it holds at dt up to 0.5 s, 25x that scale.

Note that the OBSERVED transient is much longer than 20 ms (a few hundred ms -- see
`selftest_transient_is_gradual`). That is not damping: it is the net yaw moment starting near zero
from a standing start. Stiffness sets what the integrator must survive; the moment sets how fast
the body actually turns.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams

CELL, CELLS = 0.1, 241
ORIGIN = (-8.0, -12.0)
SHEAR_LK = 8.0
MU = 0.8
COMMAND = (3.0, 5.0)  # a turning command, so both the forward and yaw channels are exercised


def _device() -> str:
    return "cuda" if wp.get_cuda_device_count() > 0 else "cpu"


def _rollout(steps: int, dt: float, momentum: bool, command=COMMAND, init=None) -> np.ndarray:
    """Return the body twist per step, [steps+1, 3]."""
    device = _device()
    sim = ForwardSimulator(
        RobotParams(),
        SolverParams(dt=dt, shear_lk=SHEAR_LK, shear_iters=8, body_momentum=momentum),
        GridParams(CELLS, CELLS, CELL, *ORIGIN),
        1,
        steps,
        device=device,
    )
    sim.set_uniform_friction(MU)
    sim.set_terrain(wp.zeros((CELLS, CELLS), dtype=wp.float32, device=device))
    if init is not None:
        sim.init_twist.assign(np.asarray([init], np.float32))
    left, right = command
    omega = np.tile(np.float32([left, right, 0.5 * (left + right)]), (steps, 1, 1))
    sim.rollout(omega, (0.0, 0.0, 0.0))
    return sim.twist.numpy()[:, 0, :]


def selftest_disabled_is_quasistatic() -> None:
    """body_momentum off must leave every step at the quasi-static solution."""
    twist = _rollout(6, 0.1, momentum=False)
    spread = float(np.abs(twist[1:] - twist[1]).max())
    print(f"  quasi-static: twist constant across steps to {spread:.2e}")
    assert spread < 1e-5, "the quasi-static solve should not depend on the previous step"
    print("disabled  OK")


def selftest_settles_to_quasistatic() -> None:
    """From rest, the dynamic twist must converge to the quasi-static one -- same equilibrium."""
    steady = _rollout(6, 0.1, momentum=False)[1]
    dynamic = _rollout(40, 0.1, momentum=True)
    print(f"  {'step':>6} {'vx':>8} {'yaw_rate':>10}")
    for k in (1, 2, 3, 5, 10, 40):
        print(f"  {k:6d} {dynamic[k, 0]:8.4f} {dynamic[k, 2]:10.4f}")
    print(f"  quasi-static target      {steady[0]:8.4f} {steady[2]:10.4f}")
    error = float(np.abs(dynamic[-1] - steady).max())
    print(f"  |dynamic(40) - quasi-static| = {error:.2e}")
    assert error < 5e-3, "momentum changed the equilibrium, it should only change the approach"
    print("settling  OK")


def selftest_transient_is_gradual() -> None:
    """Starting from rest the body must ACCELERATE into the turn, not arrive instantly.

    The approach is slower than the contact time constants alone suggest, and that is a real
    consequence of a slip-RATIO force law rather than a damping artifact. At rest every wheel has
    100% slip, so lambda = L/K identically for all three and each mobilises the same fraction of
    mu*N; the left and right longitudinal forces are then proportional to their (equal) normal
    loads and their yaw moments cancel. Yaw can only develop once the body has picked up forward
    speed and the two slip ratios separate -- so the model predicts a turn that builds over a few
    hundred ms from a standing start, not one that snaps to its steady rate.
    """
    steady = _rollout(6, 0.1, momentum=False)[1]
    dynamic = _rollout(20, 0.1, momentum=True)
    first = abs(dynamic[1, 2] / steady[2])
    print(f"  yaw rate after one step is {first:.1%} of steady state")
    assert first < 0.98, "the first step already reached steady state -- inertia is not acting"
    assert first > 0.02, "the first step did not move at all"
    rising = np.abs(dynamic[1:12, 2])
    assert np.all(np.diff(rising) > -1e-6), "the approach should be monotone, not oscillatory"
    print("transient  OK (monotone approach, no overshoot)")


def selftest_stability() -> None:
    """The implicit step must stay stable at timesteps far beyond the contact time constants."""
    steady = _rollout(6, 0.1, momentum=False)[1]
    print(f"  {'dt [s]':>8} {'final yaw_rate':>15} {'monotone':>10}")
    for dt in (0.02, 0.05, 0.1, 0.25, 0.5):
        twist = _rollout(max(int(3.0 / dt), 8), dt, momentum=True)
        series = np.abs(twist[1:, 2])
        monotone = bool(np.all(np.diff(series) > -1e-4))
        print(f"  {dt:8.3f} {twist[-1, 2]:15.4f} {str(monotone):>10}")
        assert np.isfinite(twist).all(), f"dt={dt} diverged"
        assert monotone, f"dt={dt} oscillates -- the step is behaving explicitly"
        assert abs(twist[-1, 2] - steady[2]) < 0.02, f"dt={dt} settled somewhere else"
    print("stability  OK (same equilibrium, no oscillation, dt up to 0.5 s)")


def selftest_initial_twist_is_used() -> None:
    """A rollout starting from a moving robot must begin from that twist, not from rest."""
    steady = _rollout(6, 0.1, momentum=False)[1]
    from_rest = _rollout(20, 0.1, momentum=True)
    from_steady = _rollout(20, 0.1, momentum=True, init=steady)
    print(f"  from rest   step1 yaw_rate {from_rest[1, 2]:.4f}")
    print(f"  from steady step1 yaw_rate {from_steady[1, 2]:.4f}  (target {steady[2]:.4f})")
    assert abs(from_steady[1, 2] - steady[2]) < 5e-3, "starting at equilibrium should stay there"
    assert (
        abs(from_rest[1, 2] - steady[2]) > 5e-3
    ), "starting from rest should NOT be at equilibrium"
    print("initial twist  OK")


if __name__ == "__main__":
    wp.init()
    selftest_disabled_is_quasistatic()
    print()
    selftest_settles_to_quasistatic()
    print()
    selftest_transient_is_gradual()
    print()
    selftest_stability()
    print()
    selftest_initial_twist_is_used()
