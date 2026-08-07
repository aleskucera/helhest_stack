"""Analytic verification of the engine's quasi-static certificates.

Run:  python -m tests.engine.certificates

Every check here has a closed-form answer, so it is asserted, not eyeballed:

  `selftest_ramp_margin`  -- the tip-over margin `min_i(N_i)/(m g)` on tilted planes, against
                             the support-triangle geometry AND against the `max_roll` gate.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams

CELL = 0.1
CELLS = 161
ORIGIN = (-8.0, -8.0)


def _device() -> str:
    return "cuda" if wp.get_cuda_device_count() > 0 else "cpu"


def _plane(pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """A plane tilted by `pitch_deg` about +y (rising with x) and `roll_deg` about +x."""
    axis = ORIGIN[0] + CELL * np.arange(CELLS)
    X, Y = np.meshgrid(axis, axis)
    H = np.tan(np.radians(pitch_deg)) * X + np.tan(np.radians(roll_deg)) * Y
    return np.ascontiguousarray(H, np.float32)


class _Static:
    """One-step rollout on a given terrain, wheels stationary unless `omega` says otherwise."""

    def __init__(self, robot_params: RobotParams | None = None, mu: float = 0.6):
        self.robot_params = robot_params or RobotParams()
        self.device = _device()
        self.sim = ForwardSimulator(
            self.robot_params,
            SolverParams(),
            GridParams(CELLS, CELLS, CELL, *ORIGIN),
            1,
            1,
            device=self.device,
        )
        self.sim.set_uniform_friction(mu)

    def run(
        self, heights: np.ndarray, omega: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> dict[str, np.ndarray | float]:
        self.sim.set_terrain(wp.array(heights, dtype=wp.float32, device=self.device))
        cmd = np.asarray(omega, np.float32).reshape(1, 1, 3)
        self.sim.rollout(cmd, (0.0, 0.0, 0.0))
        derived = self.sim.derived.numpy()[1, 0]
        return {
            "loads": self.sim.loads.numpy()[0, 0],
            "pitch": float(derived[1]),
            "roll": float(derived[2]),
            "stability": float(self.sim.stability.numpy()[0, 0]),
            "saturation": float(self.sim.saturation.numpy()[0, 0]),
            "alpha": float(self.sim.turning.numpy()[0, 0, 0]),
            "residual": float(self.sim.residual.numpy()[0, 0]),
        }


def _tip_angles(rp: RobotParams) -> tuple[float, float]:
    """Static tip-over angles [deg] from the support TRIANGLE, by hand.

    Contacts project to (0, +b), (0, -b), (-l, 0) with the CoM at (com_x, 0), all a wheel radius
    below the CoM -- so each edge tips at atan(distance / wheel_radius) for a tilt perpendicular
    to it. Front axle = the x = 0 edge (nose-over, descending); rear edges = the two lines from a
    front wheel back to the rear wheel.
    """
    b, l, h = rp.half_track, rp.rear_offset, rp.wheel_radius
    com_x = abs(rp.com[0])
    d_front = com_x
    d_rear = b * (l - com_x) / np.hypot(b, l)
    return float(np.degrees(np.arctan(d_front / h))), float(np.degrees(np.arctan(d_rear / h)))


def selftest_ramp_margin() -> None:
    """Tip-over margin on tilted planes: what it reports, and what it provably cannot report.

    Two analytic identities are asserted, and they are what makes the disagreement with the
    geometric tip angle a FINDING rather than a bug:

      sum_i N_i / (m g)      = 1 / (cos pitch cos roll)
      min_i N_i / (m g)      = (|com_x| / rear_offset) / (cos pitch cos roll)

    i.e. on a uniform plane `normal_loads` returns the body-frame barycentric weights of the CoM,
    scaled by 1/cos. It is a normal-only balance: the tangential (friction) reaction that holds
    the robot on the slope, and the overturning moment it exerts about the CoM, are simply not in
    the equations, so the margin RISES with tilt and never reaches 0.
    """
    rp = RobotParams()
    static = _Static(rp)
    tip_front, tip_rear = _tip_angles(rp)
    weight_frac = abs(rp.com[0]) / rp.rear_offset  # rear-wheel share of the weight on the flat
    max_roll_deg = np.degrees(rp.max_roll)

    print(
        f"support-triangle tip angles: front axle {tip_front:.1f} deg, rear edge {tip_rear:.1f} deg"
    )
    print(f"planner gate: max_roll {max_roll_deg:.1f} deg")
    print(f"{'tilt[deg]':>9} {'axis':>5} {'pitch':>8} {'roll':>8} {'minN/mg':>9} {'sumN/mg':>9}")

    worst_sum, worst_min, margin_at = 0.0, 0.0, {}
    for axis in ("roll", "pitch"):
        for tilt in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 29.5, 34.6, 40.0, 50.0):
            # the pitch sweep DESCENDS (height falls with x -> nose-down, positive pitch): that is
            # the direction that tips the CoM over the front axle
            r = static.run(
                _plane(
                    pitch_deg=-tilt if axis == "pitch" else 0.0,
                    roll_deg=tilt if axis == "roll" else 0.0,
                )
            )
            scale = 1.0 / (np.cos(r["pitch"]) * np.cos(r["roll"]))
            loads_sum = float(np.sum(r["loads"])) / (rp.mass * rp.gravity)
            worst_sum = max(worst_sum, abs(loads_sum - scale))
            worst_min = max(worst_min, abs(r["stability"] - weight_frac * scale))
            margin_at[(axis, tilt)] = r["stability"]
            print(
                f"{tilt:9.1f} {axis:>5} {np.degrees(r['pitch']):8.2f} {np.degrees(r['roll']):8.2f} "
                f"{r['stability']:9.4f} {loads_sum:9.4f}"
            )

    print(
        f"vertical balance sum N = m g / (cp cr): worst dev {worst_sum:.2e}; "
        f"min N = {weight_frac:.4f} m g / (cp cr): worst dev {worst_min:.2e}"
    )
    assert worst_sum < 2e-3, "normal loads no longer sum to m g / (cos pitch cos roll)"
    assert worst_min < 2e-3, "min N is no longer the CoM's barycentric weight over 1/cos"

    # the disagreement, asserted: the margin is POSITIVE and RISING past both tip angles
    for axis, tilt in (("pitch", 29.5), ("roll", 34.6), ("roll", 50.0)):
        assert margin_at[(axis, tilt)] > margin_at[(axis, 0.0)], "margin should rise with tilt"
    print(
        f"DISAGREEMENT: at the {tip_front:.1f} deg front-axle tip angle the margin reads "
        f"{margin_at[('pitch', 29.5)]:.4f} (flat: {margin_at[('pitch', 0.0)]:.4f}), and at "
        f"{margin_at[('roll', 50.0)]:.4f} by 50 deg of bank it is still rising -- it never "
        f"reaches 0 on a uniform slope. The max_roll={max_roll_deg:.0f} deg gate fires where the "
        f"margin reads {margin_at[('roll', 15.0)]:.4f}, i.e. the two never agree; only the gate "
        f"protects against slope tip-over."
    )
    print("ramp margin  OK (identities hold; margin is slope-blind by construction)")


def _shape_worlds() -> list[tuple[str, np.ndarray]]:
    """Terrain SHAPES (not slopes) that redistribute load between the three contacts."""
    axis = ORIGIN[0] + CELL * np.arange(CELLS)
    X, Y = np.meshgrid(axis, axis)

    def bump(cx: float, cy: float, h: float, s: float) -> np.ndarray:
        return h * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * s**2))

    return [
        ("rock under left front h=0.2", bump(0.0, 0.365, 0.2, 0.15)),
        ("rock under left front h=0.5", bump(0.0, 0.365, 0.5, 0.15)),
        ("spike under left front h=0.6", bump(0.0, 0.365, 0.6, 0.06)),
        ("rock under rear h=0.3", bump(-0.75, 0.0, 0.3, 0.15)),
        ("rock under rear h=0.5", bump(-0.75, 0.0, 0.5, 0.15)),
        ("spike under rear h=0.6", bump(-0.75, 0.0, 0.6, 0.06)),
        ("crest -x^2", -(X**2)),
        ("valley +x^2", X**2),
        ("lateral crest -2y^2", -2.0 * Y**2),
        ("saddle 0.5(x^2-y^2)", 0.5 * (X**2 - Y**2)),
        ("roof -3|y|", -3.0 * np.abs(Y)),
        ("roof -3|x|", -3.0 * np.abs(X)),
    ]


def selftest_shape_margin() -> None:
    """The margin DOES move with terrain shape -- but nowhere near zero, on any of these.

    This is the evidence behind `step.stability_margin`'s caveat: load transfer in a normal-only
    balance is bounded by the wheel-radius contact offsets, because the wheel positions themselves
    are body-fixed. Nothing here (up to 60 deg of tilt and a wheel on a 0.6 m spike) unloads a
    contact.
    """
    static = _Static()
    print(f"{'world':32s} {'minN/mg':>9} {'pitch':>8} {'roll':>8} {'resid':>9}")
    margins = []
    for label, heights in _shape_worlds():
        r = static.run(np.ascontiguousarray(heights, np.float32))
        margins.append(r["stability"])
        print(
            f"{label:32s} {r['stability']:9.4f} {np.degrees(r['pitch']):8.2f} "
            f"{np.degrees(r['roll']):8.2f} {r['residual']:9.1e}"
        )
    lo, hi = min(margins), max(margins)
    print(f"shape margin range [{lo:.4f}, {hi:.4f}] over {len(margins)} worlds")
    assert lo > 0.15, "a shape world finally unloaded a contact -- update the caveat, it is stale"
    assert hi - lo > 0.02, "the margin no longer responds to terrain shape at all"
    print("shape margin  OK (responds to shape, never approaches tip-over)")


def selftest_friction_saturation() -> None:
    """The friction certificate on constant slopes: it must cross 1.0 exactly at tan(theta) = mu.

    A stationary robot on a plane of tilt theta needs m g sin(theta) of tangential force and has
    mu m g cos(theta) of budget, so `saturation` = tan(theta) / mu -- along the slope (pitch) and
    across it (roll) alike. The crossing is the whole point: below it the robot can hold station,
    above it it cannot.
    """
    print(f"{'tilt[deg]':>9} {'axis':>5} {'mu':>6} {'saturation':>11} {'tan/mu':>8} {'rel err':>9}")
    worst = 0.0
    for tilt in (10.0, 20.0, 30.0):
        tan_theta = float(np.tan(np.radians(tilt)))
        for axis in ("pitch", "roll"):
            for mu in (0.2, 0.4, 0.6, 0.9, tan_theta):
                static = _Static(mu=mu)
                r = static.run(
                    _plane(
                        pitch_deg=-tilt if axis == "pitch" else 0.0,
                        roll_deg=tilt if axis == "roll" else 0.0,
                    )
                )
                expected = tan_theta / mu
                rel = abs(r["saturation"] - expected) / expected
                worst = max(worst, rel)
                print(
                    f"{tilt:9.1f} {axis:>5} {mu:6.3f} {r['saturation']:11.4f} "
                    f"{expected:8.4f} {rel:9.2e}"
                )
            # mu == tan(theta) is the crossing itself
            assert abs(r["saturation"] - 1.0) < 5e-3, "certificate does not cross 1 at tan = mu"
    print(f"slope saturation = tan(theta)/mu: worst relative error {worst:.2e}")
    assert worst < 5e-3, "friction saturation no longer matches the analytic slope value"

    # the centripetal term, on flat ground: demand = m v psi_dot, budget = mu m g
    mu = 0.6
    static = _Static(mu=mu)
    flat = _plane()
    print(f"{'wL':>6} {'wR':>6} {'saturation':>11} {'v psi/(mu g)':>13} {'rel err':>9}")
    worst_turn = 0.0
    for wl, wr in ((1.0, 3.0), (2.0, 4.0), (0.5, 5.0)):
        r = static.run(flat, omega=(wl, wr, 0.0))
        rp = static.robot_params
        v = rp.wheel_radius * (wl + wr) / 2.0
        yaw_rate = rp.wheel_radius * (wr - wl) / (2.0 * rp.half_track * r["alpha"])
        expected = v * yaw_rate / (mu * rp.gravity)
        rel = abs(r["saturation"] - expected) / expected
        worst_turn = max(worst_turn, rel)
        print(f"{wl:6.1f} {wr:6.1f} {r['saturation']:11.4f} {expected:13.4f} {rel:9.2e}")
    print(f"turn saturation = v psi_dot / (mu g): worst relative error {worst_turn:.2e}")
    assert worst_turn < 5e-3, "centripetal demand no longer matches m v psi_dot"
    print("friction saturation  OK")


if __name__ == "__main__":
    wp.init()
    selftest_ramp_margin()
    print()
    selftest_shape_margin()
    print()
    selftest_friction_saturation()
