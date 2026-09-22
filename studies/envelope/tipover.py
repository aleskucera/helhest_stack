"""Measure the robot's stability envelope by driving it over, instead of asserting it.

`RobotParams` carries max_roll = 15 deg, max_pitch_up = 25 and max_pitch_down = 15, and those
numbers were chosen rather than measured. Everything downstream rests on them: the settle vetoes
a pose by comparing against them, and `k_sigma` then demands a margin in sigmas ON TOP of them --
so tuning k_sigma against an envelope nobody measured is fitting a margin to a guess.

Static geometry says they are about half. The support triangle is the two front wheels at
(0, +/-0.365) and the rear at (-0.75, 0); the CoM sits 0.198 m behind the front axle, and tipping
about an edge begins at atan(lever / h_com):

    tipping about               lever    static limit   currently set
    the front axle (nose-down)  0.198 m       29.5 deg        15 deg
    a side edge (roll)          0.242 m       34.6 deg        15 deg
    the rear wheel (nose-up)    0.552 m       57.6 deg        25 deg

That rests on a number which is itself assumed: every entry in the mass table sits at z = 0, i.e.
axle height, so h_com is taken to be the wheel radius. A real chassis sits above its axles, and
the limits go as atan(lever / h_com) -- at h_com = 0.5 m they become 21.6 and 25.8. Measuring the
angle the robot actually goes over at therefore also measures h_com, by inference.

THE EXPERIMENT IS A DRIVE, NOT A STAND. Standing the robot on a tilted plane does not work: it
slides, which pollutes the tip signal and eventually runs it off the terrain, and seating it
means dropping it, whose impact tumbles it far below any static limit. The first version of this
file did both and reported a tip at 10 deg of roll and a 2.69 m slide at mu = 3.0, where sliding
needs 71.6.

So the robot drives onto a ramp instead, which is how it meets terrain anyway -- with load
transfer, momentum, and the wheels engaging one at a time.

  roll        the ramp covers only y > Y_SPLIT, between the rear wheel and the left one, so a
              SINGLE wheel climbs. That is how a skid-steer actually rolls over: the support
              triangle collapses toward the line through the right and rear wheels.
  nose-up     a full-width ramp, driven forward.
  nose-down   the same ramp driven BACKWARD up it, so the nose points downhill. Reversing is what
              makes a nose-down attitude reachable without first descending something.

The ramp is CURVED, H = k x^2 / 2, so its local slope k*x grows with distance and one drive
sweeps the whole range. A constant-angle ramp only ever answers pass or fail at its own angle.
At k = 0.15 the three predicted limits all fall inside the first 5 m of travel.

Friction is swept because a slope has two ways to fail and they trade off: sliding begins at
atan(mu), 38.7 deg at mu = 0.8, while tipping begins where the geometry says. At high mu the
robot cannot slide, so what is left is the tip-over angle alone; at realistic mu the measurement
is whichever binds first, which is the number a planner actually wants.

WHAT IT FOUND (k = 0.15, 0.45 m/s; `departure.py` reads the traces):

    axis        geometry   measured                       currently set
    roll           34.6d   TIPPED at 48.2 / 48.4 deg              15 deg
    nose-down      29.5d   held to 33.3 (mu 3.0), 25.4 (mu 0.8)   15 deg
    nose-up        57.6d   held to 36.1, ran out of ramp          25 deg

The two roll runs agree to 0.2 deg across a 4x friction range, so that is a tip, not a slide --
and it is 14 deg ABOVE the static prediction, because driving into the wedge the climbing wheel
carries load a static support triangle does not. The consequence for anyone setting a limit from
this: the GEOMETRY is the conservative source, not the simulator. Measured does not beat
calculated here, it exceeds it.

nose-down at mu = 0.8 stops at 25.4 deg where mu = 3.0 reaches 33.3 -- reversing up a ramp at
realistic friction, traction gives out before stability does, so that axis has a limit reality
enforces independently of tipping.

nose-up is unresolved: 15 s of driving only reaches x ~ 5 m, where the ramp is 36.9 deg, and the
robot held. Longer runs would settle it.

One caveat on all of it: the simulator's CoM sits at AXLE height because every entry in the mass
table has z = 0. A real chassis is higher and every limit scales as atan(lever / h_com), so these
numbers are an upper bound on a real robot's, not an estimate of them.

Runs on dasenka -- see studies/closed_loop/README.md for the container invocation.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from examples.helhest_junior.odin_sim import world as world_mod
from examples.helhest_junior.odin_sim.sim import build_sim
from examples.helhest_junior.odin_sim.sim import HelhestJuniorConfig
from helhest.heightmap import Heightmap

WHEEL_RADIUS = HelhestJuniorConfig.WHEEL_RADIUS
# between the rear wheel on the centreline and the left wheel at +0.365, so exactly one wheel
# climbs and the rear stays on the flat
Y_SPLIT = 0.18


def ramp_loader(kind: str, k: float, span: float = 30.0, cell: float = 0.2):
    """`load_world` returning a curved ramp rising from x = 0, and the start pose for `kind`."""

    def load(_name: str):
        n = int(span / cell)
        xs = (np.arange(n) + 0.5) * cell - span / 3  # more room ahead than behind
        ys = (np.arange(n) + 0.5) * cell - span / 2
        X, Y = np.meshgrid(xs, ys)
        H = np.where(X > 0.0, 0.5 * k * X * X, 0.0)
        if kind == "roll":
            H = np.where(Y > Y_SPLIT, H, 0.0)  # only the left wheel climbs
        yaw = np.pi if kind == "nose-down" else 0.0  # reverse up it, nose pointing downhill
        return (
            Heightmap(H, (xs[0] - cell / 2, ys[0] - cell / 2), cell),
            (-1.5, 0.0, float(yaw)),
            (9.0, 0.0),
        )

    return load


def quat_rp(q: np.ndarray) -> tuple[float, float]:
    """(roll, pitch) from an xyzw quaternion, in the settle's convention: nose-up = NEGATIVE."""
    x, y, z, w = q
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return float(np.arctan2(r21, r22)), float(np.arcsin(-np.clip(r20, -1.0, 1.0)))


def run(
    kind: str, mu: float, k: float, omega: float, steps: int, dt: float, kp: float, every: int
) -> dict:
    """Drive onto the ramp and record the whole trace; classify afterwards, not here."""
    world_mod.load_world = ramp_loader(kind, k)
    sim = build_sim(
        world="ramp",
        dt=dt,
        viewer=False,
        terrain_mu=mu,
        solid_obstacles=False,
        bounding_walls=False,  # nothing to stop it going over
    )
    back = kind == "nose-down"
    want_yaw = np.pi if back else 0.0
    # Reading body_q back costs a device->host sync, so do it every `every` steps rather than
    # every one: at 100 Hz that is still 20 samples a second, far finer than a tumble needs, and
    # the first version's per-step readback is what made a six-drive sweep take over half an hour.
    trace = []
    w = -omega if back else omega
    cmd = np.array([w, w, w], np.float32)
    for i in range(steps):
        if i % every == 0:
            b = sim.current_state.body_q.numpy()[0]
            roll, pitch = quat_rp(b[3:7])
            yaw = float(
                np.arctan2(2 * (b[6] * b[5] + b[3] * b[4]), 1 - 2 * (b[4] ** 2 + b[5] ** 2))
            )
            trace.append([float(b[0]), float(b[1]), float(b[2]), roll, pitch, yaw])
            if abs(roll) > np.radians(85.0) or abs(pitch) > np.radians(85.0):
                break  # it is over; the rest of the tumble carries no information
            # hold the heading, or a lifted wheel steers the robot off its own ramp
            err = (yaw - want_yaw + np.pi) % (2.0 * np.pi) - np.pi
            c = float(np.clip(-kp * err, -0.5, 0.5))
            cmd[:] = (w * (1.0 - c), w * (1.0 + c), w)
        sim.set_wheel_command(cmd)
        sim.step()
    return dict(kind=kind, mu=mu, trace=np.asarray(trace, np.float64))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mu", type=float, nargs="+", default=[0.8, 3.0])
    p.add_argument("--curve", type=float, default=0.15, help="ramp k: local slope = k*x")
    p.add_argument("--omega", type=float, default=1.5, help="[rad/s] wheel speed while climbing")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--rate", type=float, default=100.0)
    p.add_argument("--kp", type=float, default=1.5, help="heading-hold gain")
    p.add_argument("--every", type=int, default=5, help="steps between state readbacks")
    p.add_argument("--out", default="/local/kuceral4/tmp/tipover.npz")
    a = p.parse_args()

    out, deg = {}, np.degrees
    for mu in a.mu:
        print(f"\n=== mu = {mu} ===", flush=True)
        print(
            f"  {'test':>10s} {'travel':>8s} {'max |roll|':>11s} {'max |pitch|':>12s} {'ended':>10s}"
        )
        for kind in ("roll", "nose-up", "nose-down"):
            print(f"  {kind:>10s} running...", flush=True)
            t0 = time.perf_counter()
            r = run(kind, mu, a.curve, a.omega, a.steps, 1.0 / a.rate, a.kp, a.every)
            t = r["trace"]
            out[f"{kind}_mu{mu}"] = t
            over = abs(t[-1, 3]) > np.radians(85.0) or abs(t[-1, 4]) > np.radians(85.0)
            print(
                f"  {kind:>10s} {t[-1,0]-t[0,0]:>7.2f}m {deg(np.abs(t[:,3]).max()):>10.1f}d"
                f" {deg(np.abs(t[:,4]).max()):>11.1f}d {('WENT OVER' if over else 'held'):>10s}"
                f"  [{len(t)} steps, {time.perf_counter() - t0:.0f} s]",
                flush=True,
            )
    np.savez_compressed(a.out, **out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
