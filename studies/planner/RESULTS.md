# Planner gating measurements

Step 2 of `PROBABILISTIC_PLANNING_PLAN.md` section 7: the two cheap measurements that the plan
flags as each able to invalidate a design choice in Part A. Run 2026-09-18, RTX A500.
Scripts: `studies/planner/{settle_sensitivity,lattice_timing}.py`; JSON in `studies/out/planner/`.

---

## 6.2 Does the settle's clamping mask the sensitivity? **No.**

The z-margin design (plan section 3.1) rests on the settle's analytic Jacobian carrying map
uncertainty into attitude uncertainty. The worry was that `settle()`'s damped Newton --
`max_step` per iteration, `tilt_clamp` every pass -- clips the response.

Finite difference against the closed form. With wheels at `(0, +b)`, `(0, -b)`, `(-l, 0)` and
small angles, `roll = (e1 - e2)/2b` and `pitch = (e3 - (e1+e2)/2)/l`, so
`d(roll)/de1 = 1/2b = 1.370` and `d(pitch)/de3 = 1/l = 1.333` rad/m.

| perturbation | measured `d(roll)/dh` | measured `d(pitch)/dh` | residual |
|---|---|---|---|
| front-L, 2 mm … 50 mm | **1.370 … 1.372** | −0.667 | ≤3e-7 |
| front-R, 2 mm … 50 mm | **−1.370 … −1.372** | −0.667 | ≤3e-7 |
| rear, 2 mm … 50 mm | 0.000 | **1.333 … 1.334** | ≤3e-7 |

Exact, over a 25× range of perturbation size. The cross term `d(pitch)/de1 = −1/2l = −0.667` is
also exact. Cross-slopes to 30° settle with residuals ≤2e-7, four orders under the `resid_tol`
of 1e-2, and `tilt_clamp` (60°) never comes near binding on a robot that tips at 15°.

**Section 3.1 stands: `J^-1` is the right carrier for sigma.**

## …but the path from a map cell to a support is not a plain max

Sweeping ONE raised cell along the front-left wheel's axis gives a response that is symmetric
about the wheel centre, decays with offset, and reaches zero at |dx| ≈ 0.20 m — not the 0.35 m
a flat max over the wheel footprint would give:

| offset from wheel centre | ±0.02 | ±0.06 | ±0.10 | ±0.14 | ±0.18 | ±0.22 m |
|---|---|---|---|---|---|---|
| support lift from a 5 cm cell | 0.028 | 0.023 | 0.023 | 0.006 | 0.006 | **0.000** m |

Two mechanisms, both real:

- **The envelope is a cylinder dilation, not a flat max.** A bump of height `h` touches a wheel
  of radius `R` only within `sqrt(R² − (R−h)²)` of the centre — 0.18 m for `h` = 5 cm, `R` =
  0.35 m, which is the measured cut-off. The dilation is offset-corrected.
- **The support is then bilinearly sampled** at the wheel centre across four envelope cells, so
  a single raised cell contributes only its bilinear weight. Raising a 5×5 patch instead
  recovers the full 1.370 sensitivity (table above).

**This corrects plan section 3.3.** The maximum the Clark fold addresses is over
*offset-corrected* heights `h_i − c(dx_i)`, not raw cell heights, and its result is then blended
across four envelope cells. A fold written against raw heights would be wrong in both the
support's mean and its sigma. The correction `c` is fixed robot geometry, so it costs nothing —
but it has to be there.

A second consequence worth noting: the cylinder envelope is `wheel_width` = 0.10 m wide
laterally against 0.70 m long. Contests are overwhelmingly LONGITUDINAL. A first attempt at the
contested-contact test placed the competing cells ±0.10 m apart in `y`, outside the 0.10 m
width, and measured exactly nothing.

---

## 6.7 Lattice solve time, and whether two solves fit. **7.9 ms; yes.**

Frame budget 69 ms at 14.5 Hz. Deployed settings: 16 m window, 0.24 m routing cell
(`plan_lat_coarsen` 3), `plan_n_theta` 24.

| config | poses | median | p90 | % frame |
|---|---|---|---|---|
| **deployed (coarsen 3)** | 107,736 | **7.86 ms** | 7.90 | **11.4%** |
| coarsen 4 | 60,000 | 3.53 | 3.86 | 5.1% |
| coarsen 2 | 240,000 | 22.42 | 22.82 | 32.5% |
| optimistic: 12 headings | 53,868 | 3.69 | 3.85 | 5.4% |
| optimistic: coarse + 12 headings | 30,000 | **2.00** | 2.02 | 2.9% |

(The params file records 10.1 ms for the deployed config on another machine; 7.9 here.)

**The two-solve gap of section 4.3 is cheap.** Pessimistic plus a coarse 12-heading optimistic
solve is 9.86 ms, 14.3% of a frame — 2 ms more than the single solve already costs. The
exploration trigger, the decision-relevant target selection, and the automatic diagnosis of an
ignorance-blocked goal all come for about 3% of a frame.

**And replan latency is not the carrot's risk.** At 7.9 ms and 0.5 m/s the robot commits 4 mm
blind. What bounds the carrot is the map update rate (69 ms) and its own tracking error, not the
planner. The concern raised when MPPI was dropped is resolved; plan section 6.6 (carrot and
pivot primitives do not compose) is the one that still matters.

---

## Net effect on the plan

- **6.2 closes.** No clamping, no masking; section 3.1 is sound as written.
- **6.7 closes.** Both the single solve and the two-solve scheme fit comfortably.
- **3.3 needs a correction** before implementation: fold over offset-corrected heights, and
  account for the bilinear blend across four envelope cells.
- Part A is otherwise unblocked.
