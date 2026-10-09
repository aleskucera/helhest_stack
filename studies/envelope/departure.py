"""Where did the body's attitude stop following the terrain? That is the limit.

On the ramp the terrain ASKS for a specific attitude at each x -- atan(h/track) where one wheel
climbs, atan(k*x) where both do. A stable robot tracks that demand. The moment its own attitude
outruns the demand and keeps going, it is committed; that departure angle is the measurement.
The peak attitude in the summary line is not -- that is the tumble afterwards, and it only tells
you where the recording threshold was.

Measured (studies/envelope/tipover.py, k = 0.15, 0.45 m/s):

    run                departed at   terrain asked   verdict
    roll mu=0.8              48.4d           40.2d   WENT OVER
    roll mu=3.0              48.2d           40.1d   WENT OVER
    nose-down mu=3.0            --           31.9d   held to 33.3d
    nose-down mu=0.8            --           26.1d   held to 25.4d
    nose-up mu=0.8/3.0          --           36.3d   held to ~36d

The two roll runs agree to 0.2 deg across a 4x friction range, so that is a tip and not a slide.
It is also 14 deg ABOVE the static geometric prediction of 34.6 -- driving into the wedge, the
climbing wheel carries load a static support triangle does not. So for setting a limit the
GEOMETRY is the conservative source here, not the simulator, which is the opposite of what one
would assume from "measured beats calculated".

nose-down at mu = 0.8 stops at 25.4 deg where mu = 3.0 reaches 33.3: reversing up a ramp at
realistic friction, traction gives out before stability does.
"""

import numpy as np

K = 0.15  # ramp curvature: H = 0.5*k*x^2, local slope = k*x
TRACK = 0.73  # 2 * half_track
Y_SPLIT = 0.18
d = np.load("studies/closed_loop/out/tipover.npz")
deg = np.degrees


def terrain_roll(x):
    """One wheel on the ramp: the roll the terrain is ASKING for at this x."""
    h = np.where(x > 0, 0.5 * K * x * x, 0.0)
    return np.arctan2(h, TRACK)


def terrain_pitch(x):
    """Full-width ramp: the pitch the terrain is asking for is its local slope."""
    return np.arctan(np.where(x > 0, K * x, 0.0))


print(
    f"  {'run':>16s} {'samples':>8s} {'departed at':>12s} {'terrain asked':>14s} {'verdict':>10s}"
)
for key in sorted(d.files):
    t = d[key]
    x, roll, pitch = t[:, 0], t[:, 3], t[:, 4]
    if key.startswith("roll"):
        want, got = terrain_roll(x), np.abs(roll)
    else:
        want, got = terrain_pitch(x), np.abs(pitch)
    # departure = the body's attitude outruns what the terrain demands, and keeps going
    excess = got - want
    bad = np.flatnonzero(excess > np.radians(8.0))
    if len(bad) and got.max() > np.radians(60.0):
        i = int(bad[0])
        print(
            f"  {key:>16s} {len(t):>8d} {deg(got[i]):>11.1f}d {deg(want[i]):>13.1f}d"
            f" {'WENT OVER':>10s}"
        )
    else:
        print(
            f"  {key:>16s} {len(t):>8d} {'--':>11s} {deg(want.max()):>13.1f}d"
            f" {'held to '+f'{deg(got.max()):.1f}d':>10s}"
        )
