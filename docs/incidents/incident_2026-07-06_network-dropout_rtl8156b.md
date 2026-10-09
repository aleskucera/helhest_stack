# Incident Report — Network Dropout & Manual Reboot (helhest-jr-jetson)

| | |
|---|---|
| **Date** | 2026-07-06 |
| **Host** | `helhest-jr-jetson` (NVIDIA Jetson Orin, aarch64) |
| **Reported by** | Operator ("lost connection, had to restart the robot") |
| **Investigated by** | Claude Code session `8c7af722` |
| **Severity** | Medium — full loss of remote access (VNC/SSH), required a manual power-cycle. No data loss. |
| **Root cause** | USB 2.5GbE adapter (Realtek RTL8156B, `r8152` driver) wedged under load while running **without its firmware patch** (`rtl8156b-2.fw` missing). |
| **Attributable to the ROS/TF work in progress?** | **No.** See §6 and §8. |

---

## 1. TL;DR

The robot's primary network uplink is a **USB Realtek RTL8156B 2.5GbE dongle** (interface `eth_switch`, `192.168.18.5/24`). Its firmware patch file `/lib/firmware/rtl_nic/rtl8156b-2.fw` is **not present on the system**, so on every boot the `r8152` driver logs `Direct firmware load ... failed with error -2` and runs the adapter in a degraded state.

During the session, under sustained traffic (a TurboVNC desktop streaming a live Ouster point cloud), the adapter entered a **`Tx status -71` (USB `-EPROTO`) error storm** — **616 occurrences between 14:19:34 and 14:27:04** — and the link went down. Remote access was lost and the operator power-cycled the robot at ~14:27:06.

The concurrent ROS work (two `static_transform_publisher` nodes + read-only `ros2` queries) generates negligible network load and cannot power-cycle a Jetson. Kernel logs show **no OOM, no GPU/`Xid` fault, and no thermal shutdown**. The failure is a hardware/firmware fault on the USB NIC, and the adapter was **already logging firmware-load failures from 14:11**, before any TF work.

---

## 2. Impact

- Total loss of remote connectivity (VNC on `eth_switch`, SSH) to the Jetson.
- Operator forced to physically restart the robot.
- All in-flight, non-persistent state was lost: the two background `static_transform_publisher` nodes, the running `tmuxinator helhest-stack` session (Zenoh router + Ouster driver), and any open RViz.
- **No persistent data lost.** On-disk artifacts survived the reboot (repo, in-repo `.venv`, `~/set_ouster_tf.sh`, tmuxinator config).

---

## 3. System context

**Compute / OS**
- Jetson Orin, `aarch64`; kernel `6.8.12-1021-tegra`; L4T `R39 (release), REVISION: 2.0`.

**Network interfaces**
| Interface | Role | Driver | Address | State |
|---|---|---|---|---|
| `eth_switch` | **Primary uplink** (robot LAN / NUC) | `r8152` (USB) | `192.168.18.5/24` | UP (the failing device) |
| `eth_ouster` | Dedicated Ouster LiDAR port | onboard | link-local | UP (PTP-synced, unaffected) |
| `wlP1p1s0` | Wi-Fi | — | — | DOWN |
| `can0`/`can1` | CAN bus | — | — | DOWN |

**The failing adapter**
- USB ID `0bda:8156` — *Realtek Semiconductor Corp. USB 10/100/1G/2.5G LAN* (RTL8156B).
- Attached via a Realtek USB 3.0 hub (`0bda:0420`) on `usb-3610000.usb-1`.
- Driver `r8152` version `v1.12.13`; **`ethtool -i` firmware-version is blank** (the patch never loaded).
- MAC `00:e0:4c:68:00:20`; currently negotiated at 1000 Mb/s Full.

---

## 4. Timeline (2026-07-06, local time)

| Time | Event |
|---|---|
| 14:11–14:16 | `r8152` logs repeated `Direct firmware load for rtl_nic/rtl8156b-2.fw failed with error -2` (adapter running degraded). **Predates all TF work.** |
| ~14:1x–14:2x | Session work: TurboVNC desktop up (GPU/VirtualGL), `tmuxinator helhest-stack` (Zenoh + Ouster driver @ 10 Hz), RViz viewing `/ouster/points`; then two `static_transform_publisher` nodes + read-only `ros2`/`tf2_echo` queries. |
| **14:19:34** | **First `r8152 ... eth_switch: Tx status -71`** — TX error storm begins. |
| 14:19:34 → 14:27:04 | **616** `Tx status -71` events over ~7.5 min; link deteriorating; VNC/SSH becomes unusable. |
| ~14:27:06 | Previous boot ends — operator power-cycles the robot after losing the connection. |
| ~14:28 onward | Robot back up. Same `rtl8156b-2.fw` firmware-load failure recurs on the new boot; `eth_switch` links (carrier on) but remains unpatched. |

---

## 5. What was running (load context)

The traffic profile at the time was heavy for a USB NIC:
- **TurboVNC** desktop(s) rendering via VirtualGL, with **RViz streaming a live Ouster `PointCloud2`** (`/ouster/points`, 1024×128 @ 10 Hz, plus second-return `/ouster/points2`) back to the operator's client over `eth_switch`.
- This sustained, bursty throughput is exactly the condition under which an unpatched `r8152` adapter is known to throw `-EPROTO` (`-71`) TX errors and stall.

The ROS control-plane additions during the session were trivial by comparison:
- 2× `static_transform_publisher` — each republishes a single 7-float transform at a low fixed rate (kB/s-class, if that).
- Read-only `ros2 topic` / `tf2_echo` queries — transient.

---

## 6. What the Claude Code session did (complete list)

For the record, the actions taken immediately before the dropout were:

1. Started **two background `static_transform_publisher` nodes**:
   - `base_link → os_sensor` (`xyz 0.12 0 0.1`, `rpy 0 1.5708 0`)
   - `base_link → imu` (identity)
2. Ran **read-only** verification queries (`ros2 run tf2_ros tf2_echo`, `ros2 topic echo/hz/info/list`).
3. **Wrote** the helper file `~/set_ouster_tf.sh` (a convenience script) — **not executed**.

None of these write to the network beyond negligible ROS pub/sub metadata, and none can cause a kernel-level device failure or reboot.

---

## 7. Evidence (from `journalctl`)

**Firmware never loads (both the crashed boot and the current one):**
```
kernel: r8152 2-1:1.0: Direct firmware load for rtl_nic/rtl8156b-2.fw failed with error -2
kernel: r8152 2-1:1.0: unable to load firmware patch rtl_nic/rtl8156b-2.fw (-2)
```
- Crashed boot (`-b -1`): the firmware-failure message appears **26** times.

**TX error storm on the crashed boot (`-b -1`):**
```
Jul 06 14:19:34 helhest-jr-jetson kernel: r8152 2-1:1.0 eth_switch: Tx status -71   <-- first
...
Jul 06 14:27:04 helhest-jr-jetson kernel: r8152 2-1:1.0 eth_switch: Tx status -71   <-- last
```
- Count of `Tx status -71`: **616**.
- `-71` = Linux `-EPROTO` (USB protocol error) on the bulk TX path.

**Firmware file is absent:**
```
$ ls -l /lib/firmware/rtl_nic/rtl8156b-2.fw
(MISSING)
```

**`ethtool -i eth_switch`:**
```
driver: r8152
version: v1.12.13
firmware-version:            <-- blank: patch not loaded
bus-info: usb-3610000.usb-1
```

---

## 8. Root-cause analysis

**Primary cause.** The RTL8156B is operating **without its `rtl8156b-2.fw` firmware patch** because that file is not installed under `/lib/firmware/rtl_nic/`. The `r8152` driver falls back to the chip's on-board firmware, a configuration known to be unstable under sustained/bursty TX load and to surface as `Tx status -71` stalls. Once the TX path wedges, the link stops carrying traffic and remote access is lost.

**Trigger.** Sustained high-throughput traffic over `eth_switch` — specifically RViz streaming a live LiDAR point cloud to the operator's viewer across the VNC/remote link.

**Ruled out (no supporting evidence in the logs):**
- **Out-of-memory** — no `oom-kill` / `Killed process` entries.
- **GPU / CUDA fault** — no `nvgpu` / `gk20a` / `Xid` / GPU-fault entries (Warp was not even running at the time; earlier terrain-node tests had been terminated).
- **Thermal shutdown** — no critical-temp / throttle trip; the only "thermal" log lines are `jtop` enumerating sensors.
- **The ROS/TF work** — negligible network load; started around/after the adapter was already erroring; cannot reset the device or reboot the host.

**Conclusion.** Hardware/firmware fault on the USB network dongle. Independent of the ROS work in progress.

---

## 9. Recommended remediation (NOT applied — operator declined firmware changes)

Listed for future action; **nothing in this section was performed.**

1. **Install the missing firmware (system-level).** Place `rtl8156b-2.fw` into `/lib/firmware/rtl_nic/` (from the upstream `linux-firmware` tree), then reload the module (`modprobe -r r8152 && modprobe r8152`) or reboot. This removes the firmware-load failure and is the direct fix for the instability. Requires `sudo` and touches a system directory (outside the repo-contained convention used elsewhere on this machine).
2. **Reduce load on the USB NIC (no system change).** Avoid pushing heavy sustained traffic over `eth_switch`:
   - Run RViz **on the robot's local VNC desktop** (GPU-local rendering — already the setup) instead of streaming raw clouds to a remote client.
   - Or use a **wired/direct** link for viewing, separate from the operational uplink.
3. **Driver-level mitigations (system-level, if firmware install is undesirable).** e.g. disabling USB autosuspend for the device, or toggling TX/RX offloads via `ethtool -K eth_switch ...` — these sometimes reduce `r8152` stalls but are workarounds, not fixes.

---

## 10. Detecting recurrence

Quick health checks:
```bash
# live count of TX protocol errors (should stay ~0)
journalctl -k -f | grep --line-buffered "Tx status -71"

# firmware state (blank firmware-version == still unpatched)
ethtool -i eth_switch | grep firmware-version

# link state
ip -brief addr show eth_switch
```
A rising `Tx status -71` count is the early-warning sign of an imminent stall.

---

## 11. Resume checklist (return to where we were)

Everything below survived the reboot on disk; only the running processes need restarting.
1. Bring the sensors back up: `tmuxinator start helhest-stack` (Zenoh router + Ouster driver).
2. Re-publish the LiDAR mount TF for tuning: `~/set_ouster_tf.sh <roll> <pitch> <yaw>` (base_link→os_sensor; the last try was `rpy 0 1.5708 0`, which the operator found wrong — needs the "other-way + roll about x" adjustment).
3. Re-publish `base_link → imu` (identity) if IMU visualization is needed.
4. Reconnect the viewer — preferably RViz on the robot's local VNC desktop (see §9.2) to keep load off `eth_switch`.

---

## Appendix A — Key identifiers

- Adapter: `0bda:8156` Realtek RTL8156B, driver `r8152 v1.12.13`, bus `usb-3610000.usb-1`, MAC `00:e0:4c:68:00:20`, iface `eth_switch`, IP `192.168.18.5/24`.
- Missing file: `/lib/firmware/rtl_nic/rtl8156b-2.fw`.
- Kernel: `6.8.12-1021-tegra`; L4T `R39 rev 2.0`.
- Crashed boot: 616× `Tx status -71` (14:19:34–14:27:04); 26× firmware-load-failed.
