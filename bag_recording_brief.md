# Manual-drive bag recording — session brief

## What we're doing and why

We're about to close the loop on the MPPI + terminal-dock controller (it currently runs in
`elevation_node` in **visualization-only** mode — it plans and publishes `planned_path`/`mppi_fan`
but emits **no motor commands**). Before we ever energize motors under the controller, we want to
characterize the **real robot** from **manually driven** rosbags, at zero motor risk.

Three questions these bags must answer:

1. **Turn response / understeer-oversteer.** Our planner model uses a skid-steer turn gain
   `K_TURN=2.0` and an assumed min turn radius. We need the *real* yaw rate produced by a given
   left/right wheel-speed difference, and whether it's speed- or surface-dependent. This is the
   single most important calibration.
2. **Stopping distance / momentum.** The engine settles quasi-statically (no momentum). The dock
   controller assumes it can decelerate to a stop *at* the goal. Real stopping distance from speed
   tells us how badly it will overshoot, and how to tune the dock.
3. **Localization + map sanity.** Replayed through the node, do we get a good yaw estimate and a map
   that doesn't paint walls as clear? If localization/yaw is bad, the controller can't work no matter
   what — we want to know that from a bag, not from a crash.

**Do NOT run the controller in the loop this session.** Motors are driven by a human on the remote
only. This is pure open-loop data collection.

## Recording setup

- **Record everything:** `ros2 bag record -a`. A characterization session must capture the
  **commanded wheel omegas together with the gyro and the localization/odom output** in the same bag
  so we can pair command → response. Recording all topics is the safe way to guarantee nothing is
  missed. (If `-a` is too heavy, at minimum: the wheel-omega command/setpoint topic that
  `helhest_llc` uses, `/imu/data` **and** `/ouster/imu`, the Ouster point cloud, wheel/2D odom
  `/odom_2d`, and `/tf` + `/tf_static`.)
- **DDS + isolation:** set the SHM Fast-DDS profile (`FASTRTPS_DEFAULT_PROFILES_FILE=ros/fastdds_shm.xml`)
  and a private `ROS_DOMAIN_ID` so the 6 MB Ouster clouds aren't silently dropped and we don't collide
  with anything else on the network.
- **One bag per maneuver**, named clearly (e.g. `arc_diff0.5_slow`, `stop_from_fast`,
  `turn_in_place`). Separate bags >> one long bag — replay/analysis is per-maneuver.
- Note the **surface** and conditions in the bag name or a scratch note (concrete / grass / gravel /
  wet). Friction differences are exactly what we're trying to see.
- Before driving each maneuver, **hold still for ~2–3 s** (gives a clean zero baseline), then execute,
  then **hold still ~2–3 s** at the end.

## Maneuvers to record

Drive these deliberately and cleanly — constant commands held steady beat fancy driving. Keep speeds
modest; we care about the response, not covering ground.

1. **Fixed-differential arcs** — the key one. Hold a *constant* left/right wheel-speed difference and
   let the robot sweep a steady circle for several seconds. Repeat for a few differentials (gentle →
   sharp) and at ~2 speeds each. → real turn gain, understeer/oversteer, speed dependence.
2. **Straight line** at a few constant speeds, several seconds each. → forward wheel-omega → ground
   speed, and wheel slip.
3. **Hard stop from speed** — get up to a representative speed on a straight, then command stop.
   Repeat a couple times. → stopping distance = the dock's overshoot budget (momentum the model
   ignores).
4. **Turn in place** — spin left, spin right, steady. → the skid regime, where our quasi-static model
   is least faithful.
5. **One representative traverse** — drive a natural line near the kind of terrain/obstacles we'll
   actually navigate. → replay and eyeball the planner's fan/path vs. the line a human chose
   (feasibility-optimism check).
6. If the site has **different surfaces**, repeat maneuvers 1 and 3 on each.

~15–20 minutes of driving total. That covers everything we can learn open-loop; the closed-loop
behavior (dock park, feedback vs. understeer) comes in a later, conservative, motors-on run.

## Sanity checks before leaving the site

- Play one arc bag back and confirm the **wheel-command topic, an IMU with gyro, and `/odom_2d` are
  all present and non-empty** in the same bag (`ros2 bag info`). If command↔response can't be paired,
  the bag is useless — better to catch it on site.
- Confirm the Ouster cloud is actually in the bag and at full rate (no DDS drops).
