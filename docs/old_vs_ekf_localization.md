# Localization loop: `elevation_node` vs `elevation_node_ekf`

Both nodes run the same per-frame pipeline (scan → ICP → accumulate → maps → plan).
The key difference is **how they derive the pose(s)** used to place scans in the map
and broadcast the robot's location. `elevation_node` has one pose — the raw ICP result.
`elevation_node_ekf` produces two: **`map_T_base`** (raw ICP, used for map writing and
carving) and **`world_T_base`** (EKF-blended, used for TF, planning, and the next ICP
seed). This separation is what makes the EKF filter's smoothing safe: it never contaminates
the accumulated point cloud.

---

## `elevation_node` — odom + ICP, direct

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Per-frame inputs: odom_msg, cloud_msg, imu_buffer                          │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  localizer.predict()  │
          │                       │
          │  translation: odom Δ  │
          │  rotation:    gyro Δ  │   (gyro rotation delta integrated from the
          │                       │    buffered IMU; replaces wheel-odom yaw
          └──────────┬────────────┘    which is unreliable under skid)
                     │
                     │  world_T_base_pred  (odom-predicted SE(3) pose)
                     │  sweep_delta        (used to deskew the scan)
                     ▼
          ┌───────────────────────┐
          │  Scan pre-processing  │
          │  z-crop, self-filter, │
          │  range-crop, deskew,  │
          │  outlier removal      │
          └──────────┬────────────┘
                     │  scan_wp  (denoised base-frame scan, on GPU)
                     ▼
          ┌───────────────────────────────────────┐
          │  localizer.update()  — ICP             │
          │                                       │
          │  seed:      world_T_base_pred          │
          │  target:    accumulated map (map_wp)   │
          │  prior:     gravity_up (IMU tilt)      │
          │                                       │
          │  → outcome.pose   (accepted SE(3))     │
          │    or world_T_base_pred (on reject)    │
          └──────────┬────────────────────────────┘
                     │
                     │  world_T_base = outcome.pose
                     │           ▲
                     │           └─ raw ICP result; no further modification
                     ▼
          ┌───────────────────────┐
          │  world_scan =         │
          │  transform_points(    │
          │    scan_wp,           │
          │    world_T_base)      │   ← THE only pose that enters the map
          └──────────┬────────────┘
                     │
                     ▼
          ┌───────────────────────┐
          │  acc.step()           │
          │  (carve + merge +     │
          │   voxelise + crop)    │   accumulated_map = self.map_wp
          └───────────────────────┘
```

### Key property
`world_T_base` **is** the ICP result. There is no second estimator; the scan is
placed into the map exactly where ICP put it.

---

## `elevation_node_ekf` — odom + ICP + EKF physics filter

Steps are sequential in code order. "State carried across frames" shows what
each step reads from / writes to persistent fields.

**Pipeline rate:** the entire per-frame pipeline runs once per incoming lidar
cloud — **10 Hz** (DT = 0.1 s, confirmed in `dynamics.py` and referenced in
comments at lines 503 and 536 of `elevation_node_ekf.py`). IMU samples arrive
at ~100 Hz and are buffered; they are consumed in bulk inside each 10 Hz frame
(gyro integration for deskew + yaw-rate mean for EKF predict). The odom TF
broadcast runs at the full wheel-odom rate (higher than 10 Hz) so `base_link`
stays dense for TF lookups between frames.

```
════════════════════════════════════════════════════════════════════════════════
  PER-FRAME INPUTS
  odom_T_base   (current wheel-odom SE(3))
  cloud_msg     (raw lidar sweep + per-point timestamps)
  imu_buffer    (recent IMU samples covering this sweep)
  _prev_meas_wheel  (joint_states velocities from the previous frame)
════════════════════════════════════════════════════════════════════════════════

  PERSISTENT STATE ENTERING THIS FRAME
  localizer._world_T_base_prev   ← EKF-blended pose from frame N-1
                                   (set by set_corrected_pose at end of N-1)
  localizer._odom_T_base_prev    ← odom pose stored at end of frame N-1
  localizer._imu_R_base_prev     ← IMU orientation stored at end of N-1
  ekf.x  = [x, y, ψ, ẋ, ẏ, ψ̇]  ← EKF state after measurement update of N-1
  ekf.P  = 6×6 covariance        ← EKF covariance after measurement update of N-1

────────────────────────────────────────────────────────────────────────────────
  STEP 1 — DESKEW SEED: localizer.predict()
────────────────────────────────────────────────────────────────────────────────
  reads:  localizer._world_T_base_prev   (EKF-blended pose, frame N-1)
          localizer._odom_T_base_prev    (odom at frame N-1)
          odom_T_base                    (odom at frame N)
          imu_R_base                     (IMU world_R_base at frame N)
          localizer._imu_R_base_prev     (IMU world_R_base at frame N-1)

  computes:
    sweep_delta[:3,:3] = imu_R_base_prev.T @ imu_R_base    ← rotation from IMU
    sweep_delta[:3, 3] = odom translation delta             ← translation from wheels
    world_T_base_pred  = _world_T_base_prev @ sweep_delta   ← dead-reckoned ICP seed

  produces:
    world_T_base_pred   SE(3) — ICP seed for this frame; used in STEP 3
    sweep_delta         SE(3) — odom-only motion delta; used for deskew in STEP 2

  NOTE: does NOT touch ekf.x / ekf.P

────────────────────────────────────────────────────────────────────────────────
  STEP 2 — SCAN PREPROCESSING (deskew + denoise)
────────────────────────────────────────────────────────────────────────────────
  uses sweep_delta to motion-compensate the lidar sweep → scan_wp  (device array)

────────────────────────────────────────────────────────────────────────────────
  STEP 3 — EKF PREDICT  (physics model advance)
────────────────────────────────────────────────────────────────────────────────
  reads:  ekf.x, ekf.P  (state from end of frame N-1)
          _prev_meas_wheel  u = [v_l, v_r]
          _gyro_wz_mean(t_prev, t_curr)  ωz  (slip-immune heading rate; or None)
          dt_ratio = clamp((t_cloud − t_prev) / DT, 0.5, 3.0)

  computes (robot-local frame to avoid float overflow at large world coords):
    q_local    = ekf.x with xy zeroed (shift to local origin)
    x_pred     = predict_q6d(q_local, u, omega_z=ωz)    ← physics rollout
    F          = jacobian_F_6d_analytical(q_local, x_pred, DT)
    x_pred[0:2] += [off_x, off_y]   ← shift back to world coords
    x_pred[2]  adjusted by dt_ratio  ← scale yaw delta
    F[0,2], F[1,2] *= dt_ratio       ← scale dt-proportional Jacobian columns

    ekf.predict(F, x_pred, q_scale=dt_ratio)
      ekf.x  ← x_pred                (physics-predicted state)
      ekf.P  ← F @ P @ Fᵀ + Q*r     (predicted covariance; P⁻ for step 5)

  produces:
    ekf.x  updated to physics-predicted state  → consumed by STEP 5
    ekf.P  = P⁻ (prior covariance)             → consumed by STEP 5 Kalman gain

  NOTE: does NOT produce a pose matrix; world_T_base_pred is still from STEP 1

────────────────────────────────────────────────────────────────────────────────
  STEP 4 — ICP: localizer.update()
────────────────────────────────────────────────────────────────────────────────
  reads:  scan_wp           (preprocessed device cloud from STEP 2)
          world_T_base_pred (ICP seed from STEP 1)
          map_wp            (accumulated reference cloud, device)
          gravity_up        (gravity direction in base frame from IMU)

  runs point-to-plane ICP seeded at world_T_base_pred

  produces:
    outcome.pose    SE(3) — refined robot pose (or fallback = seed if rejected)
    outcome.status  "ok" | "rejected" | "sparse"
    outcome.rms_residual_m, outcome.num_inliers  (used in STEP 5 for R_adaptive)

────────────────────────────────────────────────────────────────────────────────
  STEP 5 — EKF MEASUREMENT UPDATE  (only when outcome.status == "ok")
────────────────────────────────────────────────────────────────────────────────
  reads:  outcome.pose       → z = [x_icp, y_icp, ψ_icp]
          outcome.rms_residual_m, outcome.num_inliers
          ekf.x, ekf.P = P⁻  (from STEP 3)

  computes:
    scale      = max((rms/rms_nom)² × (N_nom/N_inl), 0.25)  ← ICP quality weight
    R_adaptive = R_ICP × scale

    ekf.update_icp(z, R=R_adaptive):
      y = z − H·ekf.x            innovation [Δx, Δy, Δψ]
      S = H·P⁻·Hᵀ + R_adaptive
      K = P⁻·Hᵀ·S⁻¹             Kalman gain  (K < I: soft blend)
      ekf.x += K @ y             blended state
      ekf.P  = (I − K·H)·P⁻

  produces:
    ekf.x  = EKF-blended [x, y, ψ, ẋ, ẏ, ψ̇]  → consumed by STEP 6
    ekf.P  = posterior covariance               → carried to next frame

────────────────────────────────────────────────────────────────────────────────
  STEP 6 — POSE ASSEMBLY  (_splice_planar)
────────────────────────────────────────────────────────────────────────────────
  reads:  outcome.pose        → z, roll, pitch  (gravity-anchored from ICP)
          ekf.x[0:3]          → x, y, ψ         (physics+ICP blended)

  world_T_base = _splice_planar(outcome.pose, ekf.x[0], ekf.x[1], ekf.x[2])
    z / roll / pitch  ←  ICP  (terrain tilt, not modelled by planar EKF)
    x / y / yaw       ←  EKF  (physics + ICP blended; smoother than raw ICP)

  map_T_base = outcome.pose   ← raw ICP only; EKF bias never enters the map

  localizer.set_corrected_pose(world_T_base)
    → writes world_T_base into localizer._world_T_base_prev
    → NEXT frame's STEP 1 will propagate this EKF-fused pose forward

────────────────────────────────────────────────────────────────────────────────
  STEP 7 — MAP WRITING  (transform + accumulate)
────────────────────────────────────────────────────────────────────────────────
  world_scan = transform_points(scan_wp, map_T_base)   ← raw ICP pose only
  acc.step(world_scan, ...)   → map_wp updated

  On accepted ICP frames: map_T_base == outcome.pose  (no EKF bias in the map).
  On rejected frames:     map_T_base == world_T_base_pred (odom fallback seed);
                          EKF-derived seed enters the map only in this fallback
                          case (see "Map-bias caveat" below).

────────────────────────────────────────────────────────────────────────────────
  OUTPUTS OF THIS FRAME
────────────────────────────────────────────────────────────────────────────────
  → TF / planning:  world_T_base  (EKF-blended; smooth)
  → map_wp:         accumulated cloud built from map_T_base  (raw ICP; unbiased)

  PERSISTENT STATE WRITTEN FOR NEXT FRAME
  localizer._world_T_base_prev  ← world_T_base  (EKF-blended)
  localizer._odom_T_base_prev   ← odom_T_base   (current odom tick)
  localizer._imu_R_base_prev    ← imu_R_base    (current IMU orientation)
  ekf.x                         ← posterior state  [x, y, ψ, ẋ, ẏ, ψ̇]
  ekf.P                         ← posterior covariance
════════════════════════════════════════════════════════════════════════════════
```

### Key property
`elevation_node_ekf` produces **two poses per frame**:

- **`map_T_base = outcome.pose`** — the raw ICP result (or odom fallback on reject).
  This is what places the scan in the accumulated cloud and what the carving
  rays are cast from. On accepted ICP frames it is identical to what the plain
  node uses, so the map is immune to EKF drift on those frames.

- **`world_T_base = _splice_planar(outcome.pose, ekf.x[0:3])`** — the Kalman-blended
  pose: z/roll/pitch from ICP (gravity-anchored terrain tilt), x/y/yaw from the EKF
  state (physics prediction + ICP update, K < I). This is exported to TF, planning,
  and feeds the next ICP seed via `localizer.set_corrected_pose()`.

The design is the **raw-map / filtered-export separation**: the map is built from
raw sensor data only, so the ICP measurement cannot be biased by the filter's own
history. The EKF measurement update itself is a standard **absolute-pose** loosely-
coupled fusion (`z = [x_icp, y_icp, ψ_icp]`, `H = [I₃|0₃]`). The update uses an
**adaptive measurement noise** `R_adaptive = R_ICP × scale` where
`scale = max((rms/rms_nom)² × (N_nom/N_inl), 0.25)` — scaled by ICP fitness
(RMS residual and inlier count) so noisy alignments get less weight and clean ones
get more, with a floor at 0.25 preventing R from collapsing to zero on exceptionally
clean scans.

### Map-bias caveat

The "map is immune to EKF drift" guarantee holds on **accepted ICP frames** only. On
a localizer **reject** (`outcome.status == "rejected"`), `outcome.pose` falls back to
`world_T_base_pred` — the odom/gyro delta applied to the previous fused (`world_T_base`)
pose, which does include EKF influence. In that case `map_T_base = outcome.pose` is
effectively EKF-derived, and if the filter is diverging the map will reflect it. The
sustained-reject reset machinery (`reset_after_rejects`) bounds this scenario by
wiping and re-seeding the map before the ICP seed drifts far enough to cause permanent
localizer rejects.

---

## Side-by-side summary

| | `elevation_node` | `elevation_node_ekf` |
|---|---|---|
| **ICP seed** | odom Δ applied to previous ICP pose | odom Δ applied to previous **EKF-fused** pose |
| **Map writing pose** | `outcome.pose` (raw ICP) | `outcome.pose` (raw ICP — identical on accepted frames) |
| **Post-ICP export pose** | `outcome.pose` directly | `_splice_planar(outcome.pose, ekf.x)` |
| **x/y/yaw source (export)** | raw ICP | EKF-blended (physics predict + ICP update) |
| **z/roll/pitch source** | raw ICP | raw ICP (same) |
| **Fallback when ICP rejects** | odom-predicted pose | EKF-predicted pose (physics model) |
| **Physics model input** | n/a | `_prev_meas_wheel` (wheel speeds) + `_gyro_wz_mean` (IMU ωz, slip-immune yaw) |
| **Localizer seed override** | n/a | `set_corrected_pose(world_T_base)` each frame |
| **ICP measurement noise** | n/a | adaptive: `R_ICP × scale`, `scale = max((rms/rms_nom)² × (N_nom/N_inl), 0.25)` |
| **Accumulator code** | identical | identical |

---

## Bag replay behaviour

The EKF predict step is driven by two inputs:

- **Translation** (`_prev_meas_wheel`): measured wheel velocities from `/joint_states`.
  The same signal in live operation and bag replay (`ekf-demo` play_topics must
  include `/joint_states`; without it predict gets `u = [0,0,0]`), so the prediction
  tracks the actual motion in both cases.
- **Yaw** (`_gyro_wz_mean`): the base-frame IMU gyro rate averaged over the
  inter-cloud window, replacing the wheel-differential yaw estimate. The gyro is
  immune to wheel slip and lateral-dynamics model error, which caused the EKF heading
  to lag ICP by up to 9° during turns — the χ² gate would then reject valid ICP
  corrections for the duration of the turn. Falls back to the wheel differential when
  no IMU samples are available.

---

## EKF noise matrices and tunable parameters

All values live in `elevation_node_ekf.py` — the matrices near the top of the file,
the ROS parameters in `_declare_parameters()`.

### State vector

```
x = [x,  y,  ψ,  ẋᵂ,  ẏᵂ,  ψ̇]   (6-DOF)
     pos  pos  yaw  world-frame velocities
```

### Initial state covariance P₀  (diagonal, `[6×6]`)

```python
_SIG_P0 = [0.10 m,  0.10 m,  2.0°,  0.30 m/s,  0.30 m/s,  0.20 rad/s]
```

| State | 1-σ | Rationale |
|---|---|---|
| x, y | 0.10 m | first ICP fix typically within 10 cm of the odom seed |
| ψ | 2.0° | gyro heading is accurate at boot; small initial heading uncertainty |
| ẋᵂ, ẏᵂ | 0.30 m/s | velocities not directly measured; generous to let the predict step dominate early |
| ψ̇ | 0.20 rad/s | same rationale |

### Process noise Q  (diagonal, `[6×6]`)

```python
_SIG_Q = [0.02 m,  0.02 m,  0.5°,  0.15 m/s,  0.15 m/s,  0.10 rad/s]
```

Represents uncertainty added per predict step (one LiDAR frame ≈ 0.1 s).
Velocity rows are generous because F[:,3:6] = 0 — the simulator re-derives
velocities from the wheel-speed input `u` at each step rather than integrating them.

| State | 1-σ/step | Rationale |
|---|---|---|
| x, y | 0.02 m | ~2 cm/step model error for straight-line sliding terrain |
| ψ | 0.5° | small heading model error; gyro covers most of it |
| ẋᵂ, ẏᵂ | 0.15 m/s | unobservable states; kept ≈ P₀ so P_vv stays near its initial value |
| ψ̇ | 0.10 rad/s | same |

### ICP measurement noise R_ICP  (diagonal, `[3×3]`)

```python
_SIG_R_ICP = [0.05 m,  0.05 m,  1.0°]
```

Nominal (scale = 1) uncertainty of an ICP pose measurement. Used as the base
matrix that `R_adaptive = R_ICP × scale` scales.

| Component | 1-σ | Rationale |
|---|---|---|
| x, y | 0.05 m | typical lateral + longitudinal ICP position spread under normal conditions |
| ψ | 1.0° | heading from ICP is usually better than position; 1° is conservative |

### Adaptive measurement noise parameters (ROS params)

| Parameter | Default | Units | Description |
|---|---|---|---|
| `icp_r_rms_nom` | `0.018` | m | Nominal ICP RMS residual. Calibrated as geometric mean of observed median RMS across 3 bags (0.019 m, 0.015 m, 0.021 m) — minimax-optimal at 0.018 m (max cross-bag scale deviation 0.30). Set to the typical median RMS for your sensor and scene. |
| `icp_r_inl_nom` | `4800` | # points | Nominal inlier count. Calibrated to observed mean inlier count. |

`scale = max((rms / rms_nom)² × (N_nom / N_inl),  0.25)`

- `scale = 1.0` at nominal operating conditions → `R_adaptive = R_ICP`
- `scale > 1` for a noisy/sparse scan → larger R → smaller K → less ICP weight
- `scale < 1` for a clean/dense scan → smaller R → larger K → more ICP weight
- Floor at **0.25** prevents R from collapsing below ¼ R_ICP even on perfect scans

Effective Kalman gain K_xy observed across two bags: **0.36–0.42** (takes roughly
one-third of the ICP position innovation, discards two-thirds as predicted by the
physics model). This was verified by bag replay in July 2026.

---

## Debug topics (`publish_ekf_debug`)

`elevation_node_ekf` publishes four informational topics when the ROS parameter
`publish_ekf_debug` is `true` (default).  Nothing in the stack subscribes to them;
they exist purely for monitoring and tuning.

| Topic | Type | Rate | Contents |
|---|---|---|---|
| `ekf/pose_pred` | `geometry_msgs/PoseStamped` | every frame | Planar x/y/ψ immediately after `ekf.predict()`, before the ICP measurement update. Same stamp as `ekf/odom` so a logger can pair the predict-only pose against the posterior. |
| `ekf/odom` | `nav_msgs/Odometry` | every frame | Fused pose (EKF x/y/yaw, ICP z/roll/pitch), world-frame velocity rotated into base_frame, and the full 6×6 covariance blocks (z/roll/pitch diagonal = 1e6 sentinel — not filtered). |
| `ekf/nis_icp` | `std_msgs/Float32` | accepted ICP frames only | Normalised Innovation Squared (NIS) of the ICP measurement update. |
| `ekf/diagnostics` | `diagnostic_msgs/DiagnosticArray` | every frame | Per-frame scalar summary: `status`, `nis`, `innov_x_m`, `innov_y_m`, `innov_yaw_deg`, `r_scale`, `rms_residual_m`, `num_inliers`, `dt_ratio`, `consecutive_rejects`. Level `WARN` on rejected/sparse frames or when NIS exceeds the χ²(3) 99th percentile. Once a predict step has run this session, the message also carries covariance fields: `cov_pred_{x,y,psi,vx,vy,psidot}` and `cov_upd_{x,y,psi,vx,vy,psidot}` (per-state diagonal of P⁻ and P⁺), `cov_pred_trace`, `cov_pred_logdet`, `cov_upd_trace`, `cov_upd_logdet` (PSD-size scalars of the full matrix), and `cov_updated` (`"1"` when an ICP update was applied this frame, `"0"` for predict-only). |

### Reading NIS

The ICP update observes three DOF (`[x, y, ψ]`), so NIS = yᵀ S⁻¹ y is **χ²(3)** distributed
when the filter is consistent (Q and R match the true noise):

| NIS value | Interpretation |
|---|---|
| mean ≈ 3 | consistent — Q/R are well tuned |
| sustained > 7.81 (95th percentile) | innovation is too large → R_ICP too small, or Q too small (filter over-confident), or predict is biased |
| sustained < 1 | innovation is too small → R_ICP too large (filter ignores ICP) |
| > 11.34 (99th percentile) | single-frame: flags in `ekf/diagnostics` as WARN |

Watch `ekf/nis_icp` in PlotJuggler or `rqt_plot`; a rolling mean well above 7 during
normal driving is a signal to raise `_SIG_R_ICP` or lower `_SIG_Q` (or vice-versa).
`ekf/odom` covariance gives a sanity check that P does not collapse or diverge over time.
