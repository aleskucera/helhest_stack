# planning/ refactor: terrain_value_field is the only cost-to-go machinery

Decisions (2026-09-26):

1. **One feasibility model.** Margins in sigmas (`terrain_value_field.margin`) decide what is
   blocked. The deployed hard thresholds are the `z_veto = 0` case of it, not a separate path.
   Preferences (flatness, robust-tube charge, clearance time price) are **costs**, added to the
   classified pose cost. The field runs in stages: classify -> the robot's post-processing ->
   iterate.
2. **Gap / doubt:** keep tvf's `solve_pair` / `value_of_looking` / `at`. Delete `CostToGo`'s
   `solve_gap` / `gap_at` / `doubt_targets`, and port the two demos that use them.
3. **`GeometricProducer`** becomes a test fixture.

Out of scope: turning the sigma model on for the robot (the node passing sigma), archiving the
standalone terrain_value_field repo, anything in MPPI / governor / clearance law.

Each step is its own commit, and every step must leave the golden check (step 0) green. Steps
that are meant to change behaviour are held back to step 8 and measured on their own.

---

## 0. Freeze the current behaviour (golden fields)

`studies/planning_refactor/golden.py` records the fields of the CURRENT `CostToGo` and
`CoarseRouter`, and `tests/planning/test_golden_costtogo.py` replays them.

- **Configs:**
  - A. the deployed params file: clearance route on, robust tube 0.20 m / 15 deg, coarse layer
    with world-anchored origin, pivot 0.3;
  - B. A with the clearance route off, so the tube vetoes;
  - C. the sim z-margin path: `z_veto 2`, `charge_per_sigma > 0`, with sigma and drift supplied;
  - D. step gate on (`obstacle_step_m 0.3`) with a measured mask containing blind cells;
  - E. the goal outside the window, so the coarse ring is seeded.
- **Inputs:** the 8 stress worlds at 3 robot poses each, plus 3 real-bag frames from the
  `plan_debug_record` dumps (in_speed_new corridor, outdoor).
- **Recorded:** `V`, `V_escape`, `blocked`, `hazard`, the solver's pose cost, the seeds,
  `zmargin`, `doubt`, `descent_bearing`, and the coarse `V`.
- **Also record** the closed-loop baseline at HEAD on dasenka (8 worlds x 3 seeds, both arms
  there) and the `plan:ctg` / `plan:coarse` steady-state ms from a bag replay.

**Verify:** two recordings are bit-identical (determinism). The replay test passes at HEAD.

## 1. One `Grid`

- Move `terrain_value_field/grid.py` to a neutral `helhest/grid.py`, so `engine` does not import
  `planning`.
- `engine.terrain.Grid` becomes that struct, and `GridParams.build()` returns it. Both samplers
  already use the vec4 locate and cell-centre origins (518876a), so merge `sample_field` and
  `sample` into one.

**Verify:** full test suite, golden bit-identical, the engine gradient tests (the FD check on
non-uniform fields).

## 2. Library fixes that grew up in helhest go into the library

- a. The diagonal corner rule (`coarse._omni_no_corner_cutting`) becomes the behaviour of
  `omni_control_set`, and the helhest copy is deleted.
- b. `hierarchical` gets helhest's seeding, replacing `boundary_seeds_kernel`:
  - the goal resolved on device;
  - the coarse origin as a device array, so a captured graph follows the world-anchored memory;
  - the "goal in the window means no ring" rule, with its measured justification.

  `_goal_cell_kernel`, `_seed_goal_kernel` and `_seed_goal_and_boundary_kernel` leave
  `costtogo.py`.

**Verify:** `test_coarse_sealing`, `test_hierarchical` (updated for the new kernel),
`test_grid_conventions`, golden bit-identical.

## 3. `TerrainValueField` in stages

- `classify(constraints)` writes `z`, `z_certain`, `pose_cost` and `doubt`.
  - `certain` becomes a device scalar (helhest's `sigma_scale`), so one graph serves both
    readings.
  - `z_veto` is set outside the capture.
- **Hard constraints:** `floor_i = 0` means "veto iff margin < 0; no charge, no doubt". This is
  what the settle residual and the step gate are: tests with no sigma, where
  margin/(tiny sigma) would be a hack.
- `add_cost(extra)`: `pen += extra` on the classified pose cost with the sign kept, clamped at
  0. This is exactly what `_pose_cost_kernel` does today.
- `seed_goal(goal_xy)` / `seed_goal_and_ring(...)` on device, taken from step 2.
- `iterate()`; `set_turn_price` passed through to the solver.
- `solve`, `solve_pair` and `value_of_looking` stay, rewritten as compositions of the stages.

**Verify:** tvf's existing tests unchanged and green. New tests:
- staged == `solve`, bitwise;
- a hard constraint vetoes exactly at margin < 0 and adds no doubt;
- one captured graph replays with a new goal / `z_veto` / coarse origin.

## 4. The settle producer (`planning/settle_producer.py`)

Odin's physics, and nothing else:

- The settle writes six constraints: roll, climb, descend, belly (soft, with sigma) and residual,
  step gate (hard).
- Sigma is floored per cell and then carried through the wheel geometry, as `_margin_kernel` does
  now. `floor_i` is the floor carried the same way, so tvf's `max(sigma_i, floor_i)` never
  changes anything and `z_certain` equals today's `z_opt`. With no sigma supplied, sigma = floor.
- Side channel, for the tube: `hazard` (unresolved, faces via `_step_hazard_kernel`),
  `violation`, and the flatness `tilt`.
- The half-bin yaw offset in the sigma sampling is kept here on purpose (fixed in 8a), so the
  equivalence stays exact.

**Verify:** golden bit-identical on A, B, D and E, where `z_veto = 0` makes this a pure sign test.
On C, bit-identical is the target; any difference must be float-reorder sized (<= a few ulp in
`z`, and zero flips of `blocked`), and gets written down rather than tolerated silently.

## 5. `CostToGo` becomes orchestration

- One captured frame: producer -> `field.classify` -> robust tube / clearance route charge (from
  the side channel) -> `field.add_cost` -> seed -> `field.iterate` -> clamp -> escape.
- The public surface the node, drive_sim, MPPI and the demos use stays:
  - `compute`, `set_coarse`, `descent_bearing`, `timing_stats`;
  - `V`, `V_escape`, `blocked`, `hazard`, `zmargin`, `doubt`;
  - `_vcap`, `solver`, `step`, `robot`, `grid`.
- Profiling marks are kept. `_feasibility_kernel`, `_margin_kernel`, `_pose_cost_kernel` and the
  seeding kernels are deleted from `costtogo.py`. Expected size: under half of today's 1464
  lines.

**Verify:** golden (as in step 4), full suite, the node and drive_sim import and run one frame.

## 6. `CoarseRouter` on the field

It keeps its pooling (the producer) and hands its pose cost to a heading-free
`TerrainValueField` (`add_cost` + `iterate`) instead of building its own `ValueSolver`.

**Verify:** golden coarse `V` bit-identical, `test_coarse_sealing`.

## 7. Gap / doubt and the fixture

- Delete `solve_gap`, `gap_at`, `doubt_targets` (the numpy policy walk), `V_pessimistic` /
  `V_optimistic` / `doubt_pessimistic` and `tests/planning/test_gap.py`. Its distinct cases (the
  sigma sweep of the gap) move to tvf's `test_value_of_looking` if they aren't already covered.
- Port `demos/pipeline_bag.py` and `demos/pipeline_panels.py` to `at()` / `value_of_looking()`.
- Move `GeometricProducer` to `tests/planning/terrain_value_field/`, and point
  `studies/terrain_value_field/repl.py` at it.

**Verify:** full suite; both demos run on one bag frame.

## 8. Behaviour changes, one at a time, each measured

- a. Sample sigma at the settle's own heading (`t`, not `t + 0.5`) in the settle producer. This
  affects only the z-margin path (sim only). Re-record golden C and run the drive_sim z-margin
  arm before and after.
- b. The coarse-ring lookup (`hierarchical.seed_goal_and_ring_kernel`, moved verbatim from
  `CostToGo` in step 2b) rounds `min_corner + c*cell` against the coarse grid's MIN CORNER. That
  picks the nearest coarse cell only when the two cell sizes match; at a coarse factor k it reads
  0.5*(1 - 1/k) coarse cells off (0.44 at the deployed ~8). tvf's old `boundary_seeds_kernel` used
  centres and was right. Since 2b `seed_from_coarse` inherits the offset too. Fix: centres, then
  re-record golden A/E and run the closed loop.
- c. Anything else found in steps 1-7 goes here, not into the step it was found in.

## 9. Final validation

- Full test suite.
- Bag replay (in_speed_new, outdoor): audit table identical to 2026-09-26, and `plan:ctg` /
  `plan:coarse` ms within session noise of the step-0 baseline. The expected cost is the 4-D
  margin/sigma buffers, about 6 constraints x 2 floats per pose. If that shows up, the fix is a
  fused producer+reduce kernel, not reverting the design.
- dasenka: 8 worlds x 3 seeds, both arms there. Frames, clearance and turning must be the
  same as the step-0 baseline (the fields are bit-identical, so any difference is a bug).
- Update the docstrings that describe the old split (`costtogo.py` intro, `lattice_solver.py`,
  the tvf README) and the memory.

Push only when asked.
