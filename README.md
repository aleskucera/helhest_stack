# Helhest Junior — kinematic differentiable simulator

A fast, differentiable, **purely kinematic** twin of the Helhest Junior for
planning. No Newton/Axion, no dynamics. The robot is a rigid tripod:

- **Controlled DOF** `(x, y, yaw)` — driven by the 3 wheels via no-slip
  differential-drive kinematics, with friction-dependent turning captured by two
  ICR parameters `(alpha, x_ICR)` derived from a per-cell friction field.
- **Derived DOF** `(z, roll, pitch)` — from a quasi-static settling solve
  against a heightmap (FB-NCP; wheels bilateral, chassis unilateral).

Differentiable w.r.t. the heightmap `h`, the friction field `mu`, and the
turning coefficient `k` (implicit gradients through the settling solve).
Batched over many rollouts for sampling-based planning.

See the design discussion for the full math. This package is built in phases,
each independently verifiable:

| Phase | Content | Verify | Status |
|-------|---------|--------|--------|
| 0 | scaffold, heightmap rasterizer + bilinear sampler, rosbag loader | height under wheel matches scene; run loads | ✅ |
| 1 | flat-ground forward twist, scalar `(alpha, x_ICR)`, SE(2) integration, batched | reproduces ~0.40 m/s cruise + displacement on run 18_04_51 | ✅ |
| 2 | heightmap placement (settle), wheels bilateral, normal loads `N_i`, sphere-wheel envelope | flat→level; ramp→pitched; loads=scale meas.; box climbs | ✅ |
| 3 | chassis non-penetration → high-center feasibility signal (wheels stay grounded; pose-resolving active-set deferred to Phase 5) | benign→clear; tall block→high-center w/ depth | ✅ |
| 4 | per-cell `mu` field + moment-centroid turning map | uniform→`1+k·mu`/CoM_x; slippery rear turns more; signs correct | ✅ |
| 5 | implicit gradients (`d/dh`, `d/dmu`, `d/dk`), BPTT | finite-diff check < 1e-2 | ⬜ |
| 6 | calibration vs rosbags | RMSE ≤ full-physics bar; cross-run | ⬜ |
| 7 | speed benchmark | orders faster than Axion replay | ⬜ |
| 8 | planning demo (MPPI / gradient) | reaches goal, avoids high-center | ⬜ |

Geometry/masses are pulled from `examples/helhest_junior/common.py`
(`HelhestJuniorConfig`); the wheel order/sign remap from
`examples/helhest_junior/replay_real.py` is reproduced in `data.py`.
