# helhest_stack ROS

The `elevation_node` (localization + accumulated mapping + MPPI planning viz) and the
`elevation-demo` tmuxinator. Run via the apptainer container + `dev-shell.sh` (see the
tmuxinator header). This file records **deployment gotchas** that are easy to lose hours to.

## KNOWN ISSUE: large LiDAR clouds silently dropped by DDS

**Symptom:** the node processes only a fraction of `/ouster/points` (e.g. ~40%); the
accumulated map is sparse/streaky and localization sees big per-frame rotations (ICP
rejects). Slowing the bag rate does **not** help. A bare do-nothing subscriber also only
receives a fraction — so it is **not** compute, ICP, or the node; it is the **transport**.

**Cause:** an Ouster 1024×128 cloud is **~6 MB**. With Fast DDS (the default RMW) over
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

The `elevation-demo` tmuxinator already sets this in every pane. For any other launcher
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

- **Rotation prior = integrated gyro**, not the fused `/imu/data` orientation (its yaw is
  wrong-sign on this hardware — AHRS ENU/NED bug). See `elevation_node._gyro_orientation_base`.
- **Accumulator voxel grid is world-snapped** so the map does not erode under translation
  (`DeviceMapAccumulator._min_corner`).
- **Map maintenance:** visibility ray-carve of dynamic obstacles ON, time-based recency
  age-out OFF (erased static structure), reset-on-tracking-loss ON. `NO_FORGET=1` disables
  all three; `RECENCY=1` re-enables the age-out.
