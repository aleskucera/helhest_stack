# Pre-registration — helhest engine vs Project Chrono, rigid-contact statics

Written before the first comparison ran. Chrono 9.0.1 (`pychrono` from the projectchrono conda
channel), `ChSystemNSC` — complementarity contact with a real friction cone.

## Why statics first

The engine's settle is a 3x3 solve for `(z, pitch, roll)` that puts all three wheels on the
envelope, and `normal_loads` then solves force + torque balance using contact NORMALS only. Both
are testable against a full rigid-body contact solver on terrain with a closed-form answer. If
they disagree on a flat plane, nothing downstream — traction, momentum, the ranking ablation —
means anything, so this is the gate before any dynamics comparison.

## Setup

One rigid body (no suspension or articulation — the engine has none) at m = 106.2 kg, CoM at
(-0.198, 0, 0) in body frame. Three wheels: front pair at `(0, +/-0.365, 0)`, rear at
`(-0.75, 0, 0)`, radius 0.35 m. Cylinder collision shapes of half-tread 0.05 m about body +y, so
the geometry matches the contact point committed in 621a4a0. Rigid plane, mu = 0.8, tilted in
pitch (about body y) and in roll (about body x), 0 to 25 deg.

Compared: resting `(z, pitch, roll)`, and the three normal loads as a fraction of `m g`.

## Predictions

Stated as falsifiable claims, with the reason each is expected.

1. **Geometry agrees at every tilt.** The settled `(z, pitch, roll)` is pure geometry — a rigid
   tripod resting on a plane — and contains no force reasoning at all. Bar: pitch and roll within
   **0.5 deg**, z within **1 cm**, at every tilt tested. A failure here is a bug in the settle or
   in the envelope, not a modelling difference, and it invalidates everything else.

2. **Loads agree on the flat.** With no tangential force required, the omitted friction reaction
   carries no moment, so the normals-only balance should be exact. Bar: each `N_i / (m g)` within
   **2%** of Chrono's at zero tilt.

3. **Loads diverge with tilt, and in a specific direction.** On a slope the robot is held by a
   tangential friction reaction acting at the contacts, BELOW the CoM, which produces an
   overturning moment the engine does not model. So the engine should under-report load transfer.
   Concretely, the engine's own identity says its least-loaded contact RISES with tilt,

       min_i N_i / (m g) = (|com_x| / rear_offset) / (cos pitch cos roll) = 0.2637 / (cp cr)

   while a real solve should show it FALL. Bar: at 25 deg of pitch the two must differ by more
   than **5%** of `m g`, and Chrono's `min N` must be the smaller. If instead they agree, the
   `stability_margin` finding in TIER1_REPORT.md section 1 is wrong and the margin was a usable
   tip-over test after all.

4. **The sum is the diagnostic.** The engine's loads sum to `m g / (cos p cos r)` rather than the
   true `m g cos(tilt)`; at 25 deg that is a 21% overshoot. Chrono's normal loads should sum to
   `m g cos(tilt)`. Bar: Chrono within **2%** of `m g cos(tilt)`, and the engine's overshoot
   visible and matching its identity to **1%**.

## What would change as a result

- 1 fails -> a settle/envelope bug; stop and fix before anything else.
- 3 fails -> `stability_margin` was fine and its removal in 5b42664 was wrong on the merits
  (though still right on cost, since nothing consumed it).
- 3 and 4 pass -> the size of the normals-only error is quantified for the first time, which is
  what decides whether `normal_loads` should gain the tangential moment. That was declared out of
  scope in the Tier-1 session precisely because nobody had measured it.

## What this does NOT test

Traction, slip, momentum, deformable soil. Chrono's rigid contact says nothing about Janosi
Hanamoto shear or L/K, and this conda build has no `pychrono.vehicle`, so SCM is unavailable
without a source build. The slope-drive question (the 169 cm divergence at 15 deg) is a DYNAMICS
comparison and comes after this gate passes.

---

## Addendum, written after the run: the cylinder contact is NOT resolved by this setup

The sphere result above stands (it is the engine's default contact and every prediction resolved).
The cylinder was attempted as well, to settle the modelling question left open in 621a4a0 -- whether
a real cylinder rides its RIM on a side slope, as that commit assumes, or bridges with the load
resultant near the mid-plane. Three attempts, none conclusive:

- **`ChCollisionShapeCylinder` + Bullet.** Reports ONE contact point per wheel. A cylinder on a
  plane is a LINE contact, so a single point is degenerate -- its position along the tread is
  arbitrary. Symptom: a flat plane came out asymmetric (left 0.3803 vs right 0.3551) and roll 10
  transferred load the WRONG WAY. Not usable.
- **`Type_MULTICORE`.** Not available: this conda build reports "Chrono was not built with Thrust
  support" and falls back to Bullet SILENTLY -- identical numbers, no error. Worth knowing, because
  nothing in the API tells you the backend request was ignored.
- **A 48-gon convex-hull prism.** Fixes the LOADS: flat ground returns to 0.3680 / 0.3680 / 0.2640,
  matching the sphere exactly, and every load sum equals cos(tilt). Load distribution under roll
  then tracks the sphere to within 0.015 of m g, which is a real (if secondary) result: on a
  uniform plane the contact SHAPE barely moves the load split. But the contact POINT still will
  not localise -- the load-weighted axial offset gives left ~0.1 cm while right jumps +4.00, -3.47,
  -3.94 cm across adjacent tilts. Inconsistent signs on neighbouring cases is a facet artifact.

So 621a4a0's rim-vs-bridge question stays open. Resolving it needs either a Chrono build with a
true cylinder-plane manifold, or a different instrument entirely -- and it is worth remembering
that the ablation already found the cylinder envelope decision-neutral (elite overlap 99-100%),
so the question is about physical honesty rather than planner behaviour.
