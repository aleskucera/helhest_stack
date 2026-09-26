# The robot plans on the belief map

Today the robot and the simulator plan on different maps. `elevation_node` rasterises the
accumulated point cloud (`DeviceMapAccumulator` -> `HeightMapBuilder`, max per cell, host-side
numpy inpaint) and passes the planner no sigma. `drive_sim` plans on `elevation_belief` (a Kalman
filter per cell: mean, measurement sd, pose drift). Every sim result so far therefore tested a
mapper the robot does not run, and the planner's sigma path is dead on the robot.

Goal: the node builds all three planning maps (routing, MPPI, coarse) from an `ElevationBelief`
fed with the ICP-corrected scan, through the same helper `drive_sim` uses.

Decisions (2026-09-26):

1. **All three planning maps come from the belief**, the MPPI map included. That replaces the
   current single-scan MPPI raster, which reacts instantly to moving objects but forgets
   everything behind and beside the robot.
2. **Carving: start with elevation_belief's own** (visibility carve, `max_range` 6 m, tuned in
   sim). If the dynamic bags show it over- or under-carving, port the robot's rules
   (consecutive-free persistence, frontier, the near-and-in-front gap gate) into elevation_belief.
3. **One `charge_per_sigma` for sim and robot**, set in `planner_config`, default 0 until measured.
   The node currently runs CostToGo's default 0.5 (a charge on floor-only sigma) and drive_sim
   passes 0.

Unchanged: ICP and localisation (the accumulator stays as ICP's target and the RViz point map),
`z_veto` (stays 0 on the robot), the planner, the MPPI cost. elevation_belief stays a separate,
git-pinned repo (clark_paper uses it); the node depends on it the way drive_sim does.

Each step is its own commit; nothing is pushed without asking.

---

## 0. Baselines

- Replay `bags/in_speed_odin0` and `bags/out_odin0` through the current node
  (`studies/bag_replay/run_bag.sh`, `-p profile_stages:=true`) twice each; keep the recordings,
  the audit table (`audit.py`) and steady-state stage times.
- Find a bag with moving people or objects among the Odin bags (`ostrich*`, `fast_experiment*`,
  `compact`) -- by eye on the recorded clouds -- and replay it the same way. Without one, the
  carving decision cannot be checked on real data; say so rather than skip it.

**Verify:** the two replays of each bag agree on the audit table (within the frame-count scatter
seen on 2026-09-26: 56 vs 79 planning frames).

## 1. One belief-frame helper, shared by drive_sim and the node

`helhest/perception/belief_frame.py`, class `BeliefFrame`, wrapping `ElevationBelief`:

- `update(points_world: wp.array, sensor_origin, robot_xy, dt)`: recenter, motion_update, carve,
  measure_scan -- exactly drive_sim's sequence and arguments today.
- `layers()`: device-resident, preallocated `height` (mean, inpainted on the device), `measured`
  (0/1), `meas_sd`, `drift`, on the belief window.
- The noise model and drift rates drive_sim uses (`NoiseModel("linear", 0.012, 0.004)`,
  `DriftRates.odin_slam()`) become the helper's defaults, stated once.

drive_sim switches to it.

**Verify:** a test feeds the same recorded scans (dumped from one drive_sim run) through the
helper and through drive_sim's current inline code; all four layers are bit-identical. Then one
dasenka sweep (9 worlds x 3) against the 2026-09-26 `ring` arm: same outcomes within scatter.

## 2. The node builds its planning maps from the belief

Behind a node param `map_source` (`accumulator` | `belief`, default `accumulator` for now), so one
tree replays both and step 4 is a same-session A/B.

- **The belief window** covers the coarse raster (`plan_coarse_win_m`, 20 m) at the map resolution
  (0.08 m): 250 x 250 cells. It is fed every frame with the ICP-corrected world-frame scan (the
  `world_scan` the accumulator gets), the sensor origin from `world_T_sensor`, and dt from the
  cloud stamps.
- **Routing map** (0.24 m, 3 x 3 pooled): height = max of the inpainted mean (as the accumulator
  path pools today), measured = any cell measured, sd = max over the block, drift = max over the
  block. Taking the max of sd and drift is the conservative pooling, but it is NEW (drive_sim routes
  at its map cell and never pools), so it gets its own unit test and is named in the commit.
- **MPPI map** (0.08 m, 12 m): a centred crop of the inpainted belief mean, replacing the
  single-scan raster.
- **The flat patch under the robot** (`FlatGroundFootprint`): the sensor never sees the ground under
  the chassis, and the node stamps it flat in the MPPI map today. Check what the belief reads
  there; if it is unmeasured, add the footprint to the belief as a pseudo-measurement at the
  wheels' contact height (one kernel, no host copy), so the MPPI map keeps the same guarantee.
- **Coarse map**: pooled from the belief window (drive_sim does this), replacing the
  `HeightMapBuilder` raster of the accumulated cloud.
- `CostToGo.compute` receives `sigma=meas_sd` and `drift`. With `z_veto = 0` and the charge at 0
  (step 3), sigma changes nothing yet; it is there so step 3's setting means what it says.
- The whole path stays on the device: no `.numpy()` and no host inpaint, which also removes the
  GPU->CPU->GPU round trips the accumulator path makes today (the routing and MPPI rasters are
  read back, inpainted in numpy and uploaded again).

**Verify:** unit tests for the pooling (max height, any-measured, max sd, max drift on a known
block); the node replays a bag with `map_source:=belief` and writes a recording; profile stages
`plan:belief_update` and `plan:belief_layers` appear.

## 3. One charge_per_sigma

`plan_charge_per_sigma` (default 0) and `plan_z_veto` (default 0) in `planner_config`, passed by
both the node and drive_sim through `cfg.costtogo`; drive_sim's `--charge-per-sigma` / `--z-veto`
stay as explicit overrides. This CHANGES the robot (0.5 -> 0), so it is its own commit, measured
on the baseline bags with the accumulator map (charge 0.5 vs 0), before the map switch.

**Verify:** `test_planner_config` golden updated; the bag A/B shows what the 0.5 was doing.

## 4. Bag A/B: accumulator vs belief

Same session, interleaved, twice each, on in_speed, outdoor and the dynamic bag:

- audit table: no-route frames at the robot's heading, frames with no coarse route, sealed blocks,
  path samples in sealed / unseen;
- frame time: steady-state `total`, `plan:*` stages, and the new belief stages;
- carving: a side-by-side replay page of the two maps on the dynamic bag (the scrub page takes bag
  recordings already).

**Pass:** no-route and sealed counts not worse; frame time within a few ms of the accumulator path
(the belief update is added, the host round trips are removed); moving objects leave no trail
that blocks routing.

## 5. Carving, only if step 4 fails it

Port the robot's rules into elevation_belief (its own repo, a commit there, then bump the pin
here), and re-run step 4's dynamic bag. Also re-run the sim dynamic study
(`studies/dynamic/person_carve.py`) so the sim-tuned behaviour does not regress.

## 6. Switch and delete

- `map_source: belief` in `ros/odin/odin_elevation.params.yaml`, then delete the accumulator's
  planning rasters, the single-scan MPPI raster, the host inpaint and the `map_source` param.
  The accumulator keeps ICP's target and the RViz point map.
- One more dasenka sweep and one bag replay on the final tree.
- Update `ros/README.md`, the node's docstrings, and memory.

## Risks

- **Frame time.** A per-cell Kalman update on a 250 x 250 window every frame. drive_sim runs it at
  0.2 m cells; at 0.08 m it is 6x the cells.
- **Moving objects.** The belief's carve has never seen real data.
- **The pooling rule** (step 2) is new; max sd and max drift may over-veto once the sigma path is
  switched on. It changes nothing while `z_veto` and the charge are 0.
- **Noise model.** The linear model (1.2 cm + 0.4 cm/m) was fitted to Odin bags
  (studies/calib/fit_drift.py). Check the cell size it was fitted at: at 0.08 m a cell collects
  about 6x fewer returns than at 0.2 m, so the fused sd per cell is larger.
