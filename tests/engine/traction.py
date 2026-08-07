"""Shear-compliance traction model vs an independent numpy solve of the same balance.

Run:  python -m tests.engine.traction

The legacy traction model assumes the commanded forward speed and bends only the turn, through
`alpha = 1 + k_turn * grip`. The shear model (SolverParams.shear_lk > 0) instead SOLVES the body
twist from a force balance in which each contact's force follows the Janosi-Hanamoto shear curve

    lambda = (|slip| / |R omega|) * (L/K)      mobilised = 1 - (1 - exp(-lambda)) / lambda

so force develops over a finite shear displacement rather than jumping to mu*N at infinitesimal
slip. That distinction is the whole point: with rigid Coulomb the front wheels supply any needed
yaw moment at essentially zero longitudinal slip, so the robot turns at the ideal kinematic rate.
Finite compliance is what makes alpha exceed 1, which `selftest_compliance_creates_alpha` pins by
stiffening the ground and watching alpha fall back toward 1.

Because lambda is a slip RATIO -- slip velocity over ROLLING speed -- the model is also sensitive
to forward speed at a fixed differential, which no rate-independent Coulomb model can be (adding a
common speed leaves every slip velocity unchanged). `selftest_speed_dependence` pins that
behaviour. Whether the real robot has it is NOT established: the archive contains almost no
steady-state turning, so it awaits a calibration drive.

VALIDATED ENVELOPE. With a physical contact patch the device solve tracks the host solve to ~1e-3
for L/K anywhere from 5 to 1e7. With patch = 0 AND L/K above ~1e3 it does not converge at all --
it returns its warm start. That corner is genuinely ill-posed rather than a defect: a point
contact under rigid Coulomb has a force whose magnitude no longer depends on the twist, only its
direction, so the Jacobian goes singular. It is also the corner nobody should run, and the reason
finite compliance is the better-conditioned model.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams

CELL, CELLS = 0.1, 161
ORIGIN = (-6.0, -8.0)
DT = 0.1
SHEAR_LK = 8.0
PATCH = 0.075
MU = 0.8


def _device() -> str:
    return "cuda" if wp.get_cuda_device_count() > 0 else "cpu"


def _reference_twist(
    omega: np.ndarray, lk: float, patch: float, rolling: float = 0.09, iters: int = 200
) -> np.ndarray:
    """Host solve of the same force balance -- the oracle for the device version.

    Deliberately written from the equations rather than ported from the kernel: a damped Newton
    with a line search, run to convergence, so agreement is evidence about the physics and not a
    shared transcription.
    """
    rp = RobotParams()
    radius, half_track, rear = rp.wheel_radius, rp.half_track, rp.rear_offset
    wheels = np.array([[0.0, half_track], [0.0, -half_track], [-rear, 0.0]])
    # the engine's own flat-ground normal loads: the CoM's barycentric weights times m g
    loads = rp.mass * rp.gravity * np.array([0.368, 0.368, 0.264])

    def residual(twist: np.ndarray) -> np.ndarray:
        forward, lateral, yaw_rate = twist
        out = np.zeros(3)
        for (x, y), w, load in zip(wheels, omega, loads):
            slip = np.array([forward - yaw_rate * y - radius * w, lateral + yaw_rate * x])
            gen = np.array([slip[0], slip[1], patch * yaw_rate])
            norm = np.linalg.norm(gen)
            rolling_speed = abs(radius * w)  # NOT `rolling`: that is the resistance coefficient
            lam = max(norm / rolling_speed * lk, 1e-9) if rolling_speed > 1e-6 else 1e6
            mobilised = 1.0 - (1.0 - np.exp(-lam)) / lam
            scale = MU * load * mobilised / norm if norm > 1e-9 else 0.0
            fx, fy = -scale * gen[0], -scale * gen[1]
            travel = np.array([forward - yaw_rate * y, lateral + yaw_rate * x])
            roll_scale = rolling * load / (np.linalg.norm(travel) + 1e-2)
            fx -= roll_scale * travel[0]
            fy -= roll_scale * travel[1]
            out[0] += fx
            out[1] += fy
            out[2] += x * fy - y * fx - scale * patch * gen[2]
        out[0] -= rp.mass * (-lateral * yaw_rate)
        out[1] -= rp.mass * (forward * yaw_rate)
        return out

    twist = np.array(
        [
            radius * (omega[0] + omega[1]) / 2.0,
            0.0,
            radius * (omega[1] - omega[0]) / (2.0 * half_track * 2.0),
        ]
    )
    for _ in range(iters):
        r = residual(twist)
        if np.linalg.norm(r) < 1e-10:
            break
        jac = np.zeros((3, 3))
        for k in range(3):
            probe = twist.copy()
            probe[k] += 1e-7
            jac[:, k] = (residual(probe) - r) / 1e-7
        try:
            step = np.clip(np.linalg.solve(jac, r), -1.0, 1.0)  # clamped, as the kernel does
        except np.linalg.LinAlgError:
            break
        scale = 1.0
        for _ in range(60):
            if np.linalg.norm(residual(twist - scale * step)) < np.linalg.norm(r):
                break
            scale *= 0.5
        twist = twist - scale * step
    # A silently non-converged oracle is worse than none: it would turn a device bug into a
    # "disagreement" and, worse, a device FIX into a regression. Unclamped Newton walked off to a
    # 691 N residual once rolling resistance was added, which is how this check earned its place.
    final = float(np.linalg.norm(residual(twist)))
    assert final < 1.0, f"reference solve did not converge: |residual| = {final:.1f} N"
    return twist


def _run(omega: tuple[float, float], shear_lk: float, patch: float = PATCH, rolling: float = 0.09):
    """One step on flat ground; returns (yaw_rate, alpha, x_icr) as the engine reports them."""
    device = _device()
    sim = ForwardSimulator(
        RobotParams(),
        SolverParams(
            dt=DT,
            shear_lk=shear_lk,
            contact_patch=patch,
            shear_iters=12,
            rolling_resistance=rolling,
        ),
        GridParams(CELLS, CELLS, CELL, *ORIGIN),
        1,
        1,
        device=device,
    )
    sim.set_uniform_friction(MU)
    sim.set_terrain(wp.zeros((CELLS, CELLS), dtype=wp.float32, device=device))
    left, right = omega
    cmd = np.array([[[left, right, 0.5 * (left + right)]]], np.float32)
    controlled, _, _, _ = sim.rollout(cmd, (0.0, 0.0, 0.0))
    yaw_rate = float(controlled[1, 0, 2] - controlled[0, 0, 2]) / DT
    turning = sim.turning.numpy()[0, 0]
    return yaw_rate, float(turning[0]), float(turning[1])


def selftest_compliance_creates_alpha() -> None:
    """alpha > 1 comes from shear COMPLIANCE: stiffen the ground and it collapses toward 1.

    Point contact (patch = 0) isolates the effect, since the torsional term is switched off. As
    L/K grows the curve approaches rigid Coulomb, where full mu*N develops at infinitesimal slip
    -- the front wheels then supply the yaw moment with no lost differential and alpha -> 1.
    Checked against the host solve at each step, so this is the physics and not a plateau of the
    solver.
    """
    print(f"{'L/K':>8} {'engine alpha':>13} {'numpy alpha':>12}")
    previous = None
    for lk in (5.0, 20.0, 30.0, 100.0):  # rolling resistance off here: isolate the shear curve
        _, alpha, _ = _run((4.0, 6.0), lk, patch=0.0, rolling=0.0)
        reference = _reference_twist(np.array([4.0, 6.0, 5.0]), lk, 0.0, rolling=0.0)
        rp = RobotParams()
        expected = (rp.wheel_radius * 2.0 / (2.0 * rp.half_track)) / reference[2]
        print(f"{lk:8.0f} {alpha:13.4f} {expected:12.4f}")
        assert abs(alpha - expected) / abs(expected) < 0.01, "device disagrees with the host solve"
        if previous is not None:
            assert alpha < previous, "stiffer ground must turn MORE freely, not less"
        previous = alpha
    assert previous < 1.35, "alpha should be heading to 1 as the shear curve stiffens"
    print("compliance  OK (alpha falls toward 1 as the ground stiffens)")


def selftest_matches_reference() -> None:
    """The device solve must reproduce an independent host solve of the same balance."""
    print(f"{'(wL, wR)':>14} {'engine psi_dot':>15} {'numpy psi_dot':>14} {'rel err':>9}")
    worst = 0.0
    for left, right in ((4.0, 6.0), (2.0, 8.0), (5.0, 5.5), (1.0, 3.0), (6.0, 6.0)):
        yaw_rate, _, _ = _run((left, right), SHEAR_LK)
        reference = _reference_twist(np.array([left, right, 0.5 * (left + right)]), SHEAR_LK, PATCH)
        if abs(reference[2]) < 1e-6:
            continue
        rel = abs(yaw_rate - reference[2]) / abs(reference[2])
        worst = max(worst, rel)
        print(f"{f'({left}, {right})':>14} {yaw_rate:15.5f} {reference[2]:14.5f} {rel:9.2e}")
    print(f"worst relative disagreement {worst:.2e}")
    assert worst < 5e-3, "device shear solve disagrees with the host solve of the same equations"
    print("reference agreement  OK")


def selftest_speed_dependence() -> None:
    """alpha must RISE with forward speed at a fixed differential -- the slip-ratio signature."""
    print(f"{'forward':>9} {'alpha':>8}")
    alphas = []
    for forward in (0.75, 2.0, 3.0, 4.75):
        _, alpha, _ = _run((forward - 1.0, forward + 1.0), SHEAR_LK)
        alphas.append(alpha)
        print(f"{forward:9.2f} {alpha:8.3f}")
    # Not monotone once rolling resistance is on: at low speed the resistance eats a larger share
    # of the friction budget, which lifts alpha there and partly cancels the slip-ratio trend. The
    # net rise across the range drops from 1.24 without resistance to about 1.08 with it.
    assert alphas[-1] > alphas[0], "alpha should still be higher at speed than at a crawl"
    assert alphas[-1] / alphas[0] > 1.03, "the speed dependence has vanished entirely"
    print(f"speed dependence  OK (ratio {alphas[-1] / alphas[0]:.2f} over the range)")
    print("  NOTE: a model prediction, not a validated fact -- the bags have no steady-state turns")


def selftest_legacy_untouched() -> None:
    """shear_lk = 0 must reproduce the legacy kinematic twist exactly."""
    rp = RobotParams()
    for left, right in ((4.0, 6.0), (2.0, 8.0)):
        yaw_rate, alpha, x_icr = _run((left, right), 0.0)
        expected = rp.wheel_radius * (right - left) / (2.0 * rp.half_track * alpha)
        print(f"  ({left}, {right}) alpha={alpha:.4f} psi_dot={yaw_rate:.5f} vs {expected:.5f}")
        assert abs(yaw_rate - expected) < 1e-6, "legacy twist is no longer the kinematic one"
        assert abs(alpha - (1.0 + 2.0 * MU)) < 1e-3, "legacy alpha is not 1 + k_turn * mu"
    print("legacy path  OK")


if __name__ == "__main__":
    wp.init()
    selftest_legacy_untouched()
    print()
    selftest_compliance_creates_alpha()
    print()
    selftest_matches_reference()
    print()
    selftest_speed_dependence()
