# Session prompt: implement IMPROVEMENTS.md Tier 1 — WITHOUT disturbing the research line

Paste or reference this at the start of a fresh session. It encodes the non-interference
rules agreed on 2026-08-07.

---

You are implementing motion-model improvements in ~/projects/helhest_stack. Read, in this
order, before writing any code: `CLAUDE.md` (behavioral + style rules — they are binding),
`IMPROVEMENTS.md` (the spec for everything below), and skim `studies/HANDOFF.md` §0-§1
only (context: a paper's numbers are pinned to the CURRENT engine behavior on branch
`study/adjoint-sensitivity`; your job must not change what existing code computes by
default).

## Hard constraints (violating any of these ruins concurrent research work)

1. **Branch:** create `engine/tier1-certificates` off `main`. Never commit to
   `study/adjoint-sensitivity`. Never modify anything under `studies/`.
2. **Bit-identity by default:** with default parameters, every EXISTING simulator output
   (envelope, controlled, derived, loads, clearance, residual, turning) must remain
   bit-identical to pre-change behavior. New functionality = new output arrays or
   opt-in parameters only.
3. **Write the golden test FIRST:** before touching the engine, add a test that runs a
   fixed seeded batch (e.g. the studies/adjoint/scene.py rollout set, B=8, T=16) through
   `ForwardSimulator` and `DifferentiableSimulator` and records every output array's
   hash/values to a fixture; every commit thereafter must pass it with defaults. This test
   IS the non-interference proof.
4. **Do NOT wire anything into the default MPPI cost.** Certificates are computed and
   exposed; consuming them in planning is a separate, later decision (a paper's cost
   definition is frozen). Opt-in cost hooks are fine if default-off.
5. Surgical changes only (CLAUDE.md §3): no adjacent refactors, no formatting sweeps.
   Warp constraints (CLAUDE.md §6 + IMPROVEMENTS.md §9): everything stays inside the fused
   kernels where it lives, no host sync per step, no data-dependent iteration counts, keep
   graph capture working, mind warp divergence (one thread per rollout).

## Work order (each item: implement -> verify as specified -> commit separately)

### 1. IMPROVEMENTS.md §3 — tip-over margin (pure wiring, zero new math)
`stability_margin = min_i(N_i) / (m*g)` from the already-computed `loads_out`; expose as a
new per-step output array. Verify (spec'd in §3): a bank-angle ramp world — margin must hit
0 at the geometric tip angle computable by hand from the support TRIANGLE (note from the
research line: on this tripod the front-axle edge governs, ~29.5°, NOT atan(half_track/h);
see studies/CLAIMS.md §2.2 if curious). Cross-check against the existing max_roll=15° gate
and report if they disagree (that disagreement is itself a finding, not a bug to hide).

### 2. §1 — friction saturation certificate (additive output)
In the same kernel, after `normal_loads`: demand_long/lat per §1's formulas, friction
ellipse, `saturation = demand / max(total_grip, floor)` — GUARD the denominator (§1 warns:
near-unloaded contacts otherwise report huge saturation while transmitting nothing). New
output array; no dynamics change. Verify: constant-slope worlds at 10/20/30° with a μ
sweep — the certificate must cross 1.0 exactly at tan(θ)=μ. This is an analytic test;
assert it, don't eyeball it.

### 3. §2 — torque/stall certificate (additive; parameterized placeholder)
Same machinery as §1, second reason code. The real motor torque envelope is UNKNOWN
hardware data (IMPROVEMENTS.md open question #2): implement with a `motor_torque_limit`
RobotParams field, default `inf` (= certificate never fires = bit-identity preserved), and
leave a loud TODO for the measured value. Verify with a finite test value: stall boundary
appears at the grade where required torque crosses it, independent of μ.

### 4. §4 — cylinder wheel envelope (BEHIND A FLAG, sphere stays default)
The yaw-binned rectangular envelope stack per §4's spec (32 bins; `wheel_width` new
RobotParams field, default `None` -> current spherical path exactly, satisfying constraint
2). Respect §4's coupled consequence note (§7 yaw-rate binning constraint — document it in
the docstring, don't solve it). Verify: the §4 world (single rock offset laterally from the
wheel track): sphere lifts the robot, cylinder must not; compare predicted roll to hand
geometry. Also confirm the golden test still passes with `wheel_width=None`.

### Skip for now
§5-§8, §10, §11 (conditional/deferred per IMPROVEMENTS.md); §9 bracketing (already built
and validated in the research line — `studies/bench/` — do not duplicate).

## Deliverables

- Commits on `engine/tier1-certificates`, one per item, messages in the repo's style (see
  `git log` — lowercase topic prefix, sentence-style what-and-why). No Claude co-author
  line (user rule).
- The golden bit-identity test, passing at every commit.
- Verification outputs (the analytic crossings, ramp angles, rock-world roll comparison)
  printed in the test suite or a short runner script — reproducible, not screenshots.
- A short session report listing: what changed file-by-file, the three open hardware
  numbers still needed from the user (ω_max, motor torque envelope, real wheel width —
  IMPROVEMENTS.md open questions 1/2/4), and any max_roll-vs-margin disagreement found.
- Do NOT merge to main in this session; leave the branch for the user's review.

## After this lands (separate session, not yours)

The research line will rerun its scripted study chain against the new engine as a
robustness check (studies/HANDOFF.md §8 item; conclusions are expected to transfer, numbers
to shift). Nothing you do should anticipate or depend on that.
