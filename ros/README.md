# helhest_stack ROS

The `navigation_node` (localization + belief mapping + cost-to-go / MPPI planning) and the
`odin-demo` tmuxinator. Run via the apptainer container + `dev-shell.sh` (see the tmuxinator
header). This file records **deployment gotchas** that are easy to lose hours to.

## KNOWN ISSUE: large LiDAR clouds silently dropped by DDS

**RMW note (read first):** this applies only when the RMW is **Fast DDS** (`rmw_fastrtps_cpp`)
— e.g. bag replay on a dev box. The **live robot runs Zenoh** (`rmw_zenoh_cpp`), whose
reliable-TCP transport has no such failure mode: measured same-host, a bare subscriber
received **201/201** `/ouster/points` at 10 Hz, zero loss. So `fastdds_shm.xml` below is
**inert under Zenoh** — don't chase it on the robot; check `RMW_IMPLEMENTATION` first.

**Symptom:** the node processes only a fraction of `/ouster/points` (e.g. ~40%); the
accumulated map is sparse/streaky and localization sees big per-frame rotations (ICP
rejects). Slowing the bag rate does **not** help. A bare do-nothing subscriber also only
receives a fraction — so it is **not** compute, ICP, or the node; it is the **transport**.

**Cause:** an Ouster 1024×128 cloud is **~6 MB**. With Fast DDS (the RMW in that case) over
best-effort UDP, each cloud is fragmented into thousands of packets; if the OS socket
receive buffer can't hold a whole cloud, reassembly fails and the **entire message is
dropped** — per message, regardless of playback rate. Defaults are far too small:
`net.core.rmem_max` is typically 4 MB (< one cloud) and Fast DDS's default shared-memory
segment is ~512 KB, so it silently falls back to the broken UDP path.

**Fix (same machine — lidar driver + node + rviz on one host):** use the shared-memory
transport profile `ros/fastdds_shm.xml` (64 MB SHM segment; SHM has no fragmentation and
ignores `rmem_max`). Point **every** participant at it:

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE="$REPO/ros/fastdds_shm.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="$REPO/ros/fastdds_shm.xml"
```

The `odin-demo` tmuxinator already sets this in every pane. For any other launcher
(launch files, systemd, a robot bringup script) you must set it too, or clouds drop.

**Fix (multi-host — lidar and node on different machines over the network):** SHM is
same-host only. Instead raise the kernel socket buffer above one cloud (needs root):

```bash
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.rmem_default=8388608
# persist:
echo -e "net.core.rmem_max=134217728\nnet.core.rmem_default=8388608" \
  | sudo tee /etc/sysctl.d/60-ros2-pointcloud.conf
```

**Verify the fix:** a bare subscriber should receive ~all clouds. Measured on `rotate`
(325 clouds): before → 136/325 processed (58% dropped, ICP rejects); after SHM →
318/325 (2%, 0 rejects). A do-nothing subscriber went 109/325 → 325/325.

## Other defaults worth knowing

- **Odin only** (since 2026-09-28): the pose is Odin's on-device SLAM (`/odin1/odometry`), trusted
  as is -- there is no ICP, deskew, gyro rotation prior or accumulated point cloud any more. Every
  planning map (MPPI terrain, routing grid with sigma and drift, coarse layer) is a crop of one
  `elevation_belief` window, through the same helper the closed-loop sim uses
  (`helhest/perception/belief_frame.py`); RViz shows it as `elevation_local` / `elevation_global`.
  Moving objects are handled by the belief's own carve (`belief_carve_m`, 6 m), which has not yet
  been checked on a real bag with people in it (studies/belief_mapping/STEP0.md). The Ouster-era
  notes on ICP, the gyro prior, `/imu/data` vs `/ouster/imu` and the accumulator's carve rules are
  in this file's git history before that date.
- **Gyro glitch guard (`max_gyro_rate_dps`, 600):** the Ouster-era `/imu/data` spiked to 1000-8000
  deg/s for a single sample (real motion peaks ~300). The yaw-rate loop, the turn adapter and the
  mu estimate read the gyro, so `_imu_callback` still drops any sample above the gate.
- **Node broadcasts `odom→base_link`** (`publish_odom_tf`, default on) at the full odom rate:
  the odometry arrives as a message with no TF, and without it `odin1_base_link` is disconnected
  from `map` and RViz can't place/follow the robot. Set false only if the odom source starts
  broadcasting it. RViz: world-up view = Fixed Frame `map` + view Target Frame
  `base_link`; robot-up view = Fixed Frame `base_link` (needs the dense odom-rate TF above).
