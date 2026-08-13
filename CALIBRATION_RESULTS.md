# First calibration trip — 2026-08-10

Two bags, `calibrate` (125 s, driven manually) and `compact` (183 s, scripted spins). Both carry
`/joint_states` and `/joint_setpoints` at 100 Hz and `/odin1/imu` at 397 Hz, so every fit below
scores MEASURED wheel speeds against the gyro. Yaw is `/odin1/imu` +z (response 1.786 rad/s against
0.009 and 0.004 on x and y — unambiguous).

## What the robot said

| quantity | measured here | previously held | verdict |
|---|---|---|---|
| turn gain α | **≈ 1.5** | 2.20 | the old value does not reproduce |
| forward gain | **0.932** | 0.906–0.925 | confirmed, marginally higher |
| yaw lag τ | **0.000 s** | 0.25 s / σ 0.15 m | **refuted** |
| spin breakaway | **~2 rad/s** | not known | new |

### α ≈ 1.5, and it is consistent everywhere

Three independent estimates from two bags and two regimes:

- steady driving arcs, 5 segments: median **1.52**, weighted mean 1.53
- spins in place at w = 2/3/4: **1.50 / 1.60 / 1.70**
- the lag fit's steady gain, pooled over 6 driving onsets: **1.55**

Spanning mean speeds 0.37–3.73 rad/s and both directions. The shipped `k_turn = 0.6` gives
α = 1 + k_turn·μ = **1.48**. So the shipped value is right for this surface, and the "planner
under-turns by 1.5x" concern in TIER1_REPORT §6 rests on the 2.20 figure, which this trip does not
reproduce.

Two caveats before anyone changes `k_turn`. This is ONE surface, and α = 1 + k_turn·μ makes α a
function of the terrain — which is why `K_TURN_INDOOR` and `K_TURN_OUTDOOR` already differ. The
2.20 may be a genuinely different surface rather than an error. And converged Chrono independently
predicted α ≈ 1.6, which agrees with the robot and not with 2.20.

### The yaw lag is refuted

Today's `yaw_relax_len` was fitted against Chrono at σ = 0.15 m, on the argument that a tyre's
lateral force builds over distance. The robot disagrees. Fitting (α, τ) to 6 driving-arc onsets
over a **7x** contact-speed range (0.23–1.59 m/s), τ = 0 is preferred and the cost rises
monotonically away from it:

    tau [s]   0.00    0.05    0.10    0.15    0.25    0.40    0.60
    RMS       0.1845  0.1890  0.1969  0.2041  0.2137  0.2211  0.2268
                      +2.5%   +6.7%  +10.7%  +15.8%  +19.9%  +23.0%

Once the measured wheel speeds are fed in — which already contain the 0.19 s actuator lag — the
body's yaw follows with no further lag worth modelling. So **Chrono has a yaw lag the robot does
not**, and `yaw_tau` / `yaw_relax_len` were compensating for a simulator artifact. That is also why
3aec2da found `yaw_tau` redundant with `k_turn`: both were absorbing the same non-physical thing.

Both knobs default to 0, so nothing shipped is affected. They stay in the tree as a documented
negative rather than being deleted, because the measurement is one surface and one afternoon.

The `compact` spin bag could NOT settle this on its own: τ came out 0.080 / 0.200 / 0.040 at
w = 2/3/4, non-monotone, fitting neither hypothesis. Its speed range was only 2x, which is the
next item.

### The spin breakaway, and what it cost

Below about 2 rad/s the robot will not spin in place — it either does not break loose or struggles.
That is a real property neither our engine nor Chrono predicts (with speed-controlled wheels both
assume the commanded spin simply happens), and it forced `compact` from the designed
0.5/1.0/2.0/4.0 down to 2/3/4, collapsing the contact-speed range from 8x to 2x. The σ question was
then settled by the manual driving bag instead, whose arcs happen to span 7x.

Lesson for the next trip: **the wide speed range has to come from driving, not spinning.**

## Forward gain

4 straight segments, SLAM odometry against measured wheels: 0.937 / 0.928 / 0.937 / 0.907, median
**0.932**. Consistent with the 0.906–0.925 on record. The legacy model gives 1.000 by construction,
so the ~7% loss is real and unmodelled.

## The rear wheel is fixed — and that closes the loop

Confirmed by the operator on 2026-08-10. With the rear modelled correctly as a fixed axle, all
three independent numbers agree:

| | α |
|---|---|
| the robot, this trip | **1.50** |
| Project Chrono, fixed rear, converged dt | **1.60** |
| our engine at the shipped `k_turn = 0.6` | **1.48** |

Within 7% across a real robot, a full contact solver and a quasi-static kinematic model. The
caster branch that was explored in `chrono_vehicle_model.py` gives α 1.1–1.3 and is contradicted
by the robot; it is retained only as documentation of a ruled-out alternative, and `fixed` is now
the default there.

Worth being explicit that the earlier framing was wrong in both directions: a fixed axle was once
claimed here to be "kinematically unable to turn the vehicle" (it was Chrono's rolling-friction
constraint locking the yaw, 99e850e), and the docs' description of the wheel as "trailing" and
"kinematically redundant" was read as implying a swivel. Neither survived contact with the
hardware.

## What is still open

- ~~The rear-wheel mounting.~~ **CLOSED**: the operator confirms the rear axle is FIXED, not a
  caster. That was the largest unknown in the yaw model and it resolves in the model's favour —
  see below.
- **Slopes.** No `slope` bag yet, so the `normal_loads` load-transfer fix remains validated against
  Chrono only. The archive still tops out at 5.4 deg of tilt.
- **α on other surfaces.** One surface is not a terrain model.
