# Helhest Localization Stack

Helhest localization is scan-to-map registration with an odometry prediction.
The implementation lives in `src/helhest/localization/`.

## Simple (TL-DR) summary
*STEP 1: Every frame, the localizer always produces an odom+IMU prediction first, blending odometry (for translation) and IMU (for rotation, if available). This predicted pose is later used as a fallback.*

*STEP 2: ICP from lidar scans runs against a submap and its result is accepted only if it passes some gates (enough inliers, not too large a correction, low enough RMS). If it passes, the ICP pose replaces the prediction from STEP 1. If it fails, the odom/IMU prediction is kept as-is.*

*So the priority is simply:*
1. *ICP result — used when the scan matches the map well enough.*
2. *Odom + IMU prediction — used when ICP is skipped (sparse submap) or rejected (bad gate).*

*There is no blending or weighted fusion: it's a hard accept/reject switch. The accepted pose — whichever it is — then becomes the starting point for the next frame's prediction, so errors from a rejected frame don't accumulate beyond one step of odom drift.*

## Pose convention

All poses are 4×4 SE(3) matrices written $T_{B \to A}$, meaning "the pose of
frame B expressed in frame A". A point $p^B$ in frame B maps to frame A as

$$
p^A = T_{B \to A} \, \begin{pmatrix}p^B \\ 1\end{pmatrix}.
$$

The two frames that matter throughout are `world` (a fixed global frame) and
`base` (the robot base link). Odometry and IMU each provide their own estimates
of $T_{B \to W}$ or $R_{B \to W}$.

Shorthand used in equations below: $W$ = world, $O$ = odom, $B_k$ = base at the
previous frame, $B_{k+1}$ = base at the current frame, $\hat{B}$ = predicted base.

## State

The localizer stores three values between frames:

- $T_{B_k \to W}$ — the last accepted (or fallback) world pose.
- $T_{B_k \to O}$ — the odometry pose at that same frame, used to difference the
  next incoming odom reading.
- $R_{B_k \to W}^\text{IMU}$ — the IMU world rotation at that frame (3×3), or `None`.

## Step 1 — Prediction

`Localizer.predict(odom_T_base_curr, imu_R_base_curr)` computes a predicted
world pose for the new frame.

**Frame-to-frame odom delta.** The raw odometry origin drifts globally, so the
localizer works with the relative motion between consecutive frames:

$$
\Delta T = T_{B_k \to O}^{-1} \cdot T_{B_{k+1} \to O}.
$$

This delta is immune to slow drift in the odom origin.

**Rotation source selection.** Wheel odometry is unreliable for yaw during fast
skid-steer turns: the robot spins without translating, so the wheel encoders
underestimate the yaw rate. The IMU gyro integrates the true angular rate and is
used for rotation when available. The two sensors supply what they are good at:

$$
\Delta R =
\begin{cases}
\bigl(R_{B_k \to W}^\text{IMU}\bigr)^\top R_{B_{k+1} \to W}^\text{IMU}
& \text{if IMU available,} \\
\Delta R_\text{odom} & \text{otherwise.}
\end{cases}
$$

The translation column of $\Delta T$ is always kept from odometry; only the
rotation block is replaced.

**Predicted world pose.**

$$
T_{\hat{B} \to W} = T_{B_k \to W} \cdot \Delta T.
$$

The same $\Delta T$ is returned as the **sweep delta** for LiDAR deskewing (see
below).

## Step 2 — LiDAR deskewing

A spinning LiDAR accumulates a sweep over a full rotation. Points measured at
different instants experience different robot poses, smearing the cloud if the
robot is moving.

`pose_math.deskew_scan(points, alphas, sweep_delta)` corrects this using a
constant-velocity model. Each point has a sweep fraction $\alpha \in [0,1]$
where $\alpha = 0$ is the sweep start and $\alpha = 1$ is the sweep end (the
reference instant). With the inter-frame rotation decomposed on its screw axis,

$$
R(\alpha) = \exp(\alpha\,\log \Delta R),
$$

the corrected point is

$$
p'_i = \Delta R^\top \bigl(R(\alpha_i)\,p_i + (\alpha_i - 1)\,t_\Delta\bigr).
$$

This re-expresses every point in the sweep-end base frame, removing the
within-sweep smear before ICP.

## Step 3 — Submap crop

`Localizer.update()` crops the device-resident reference cloud to a square
xy-box of half-extent `submap_radius_m` (default 15 m) around the predicted
robot position. This avoids running ICP against the entire accumulated map and
keeps the nearest geometry as the target. If the cropped submap has fewer than
`min_submap_points` (default 2000) points, ICP is skipped and the odom
prediction is used directly (`status = "sparse"`).

## Step 4 — ICP registration

The deskewed scan is registered against the submap. Two modes:

**Single init.** `IcpAligner.align(scan, submap, init_pose)` runs Gauss-Newton
point-to-plane ICP from the predicted pose. The residual minimised is the sum
of squared point-to-plane distances over inlier correspondences:

$$
\mathcal{L} = \sum_{i \in \text{inliers}}
\bigl(\hat{n}_i \cdot (T\,p_i - q_i)\bigr)^2,
$$

where $\hat{n}_i$ is the target normal at the matched reference point $q_i$.

**Yaw multi-start.** When `yaw_restarts > 1`, $H$ initial poses are seeded by
rotating the predicted heading by angles uniformly spaced over
$\pm\,\texttt{yaw\_search\_deg}/2$. All $H$ inits are aligned in one batched
GN pass and the result with the lowest RMS residual (subject to the inlier gate)
is kept. This escapes the wrong rotational basin that a single init falls into
after a fast skid-steer in-place turn.

An optional IMU gravity vector can be passed as `gravity_up` to anchor the ICP
roll and pitch to the measured vertical, preventing roll/pitch drift in
geometrically degenerate scenes.

## Step 5 — Acceptance gate

The ICP result is accepted only if **all** of the following hold:

$$
N_\text{inliers} \ge N_\text{min},
\qquad
\|\Delta t\|_2 \le d_\text{max},
\qquad
\angle(\Delta R) \le \theta_\text{max},
$$

and, when `max_rms_residual_m > 0`:

$$
\text{RMS} = \sqrt{\frac{\mathcal{L}}{N_\text{inliers}}} \le r_\text{max}.
$$

The RMS residual is the primary fitness gate. The ICP `converged` flag is
deliberately **not** used: a good alignment often plateaus just above the GN
step tolerance within the iteration cap, and gating on convergence would reject
those valid results.

Default thresholds (`LocalizerConfig`):

| Parameter | Default | Meaning |
|---|---|---|
| `min_inliers` | 500 | Minimum matched points for acceptance |
| `max_correction_trans_m` | 1.0 m | Max translation shift from prediction |
| `max_correction_rot_rad` | 15° | Max rotation shift from prediction |
| `max_rms_residual_m` | 0 (off) | Max point-to-plane RMS; 0 disables |

## Step 6 — Commit

If accepted, the ICP-refined pose $T_{B_{k+1} \to W}^\text{ICP}$ becomes the
new $T_{B_k \to W}$ for the next frame. If rejected or fallen back, the odom
prediction is committed instead. Either way the current odom and IMU readings
are stored so the next `predict()` can difference against them.

$$
T_{B_k \to W} \leftarrow
\begin{cases}
T_{B_{k+1} \to W}^\text{ICP} & \text{accepted,} \\
T_{\hat{B} \to W} & \text{rejected / fallback.}
\end{cases}
$$

Corrections therefore compound forward: the ICP result of one frame is the
starting point for the next frame's prediction, so a good lock stays locked and
a diverged frame costs only one step's worth of odom drift before the next scan
can recover.

## Data residency

The scan, reference cloud, and submap crop remain on device as `wp.array(vec3)`
throughout. Only 4×4 poses (64 floats), inlier counts, and scalar diagnostics
cross to the host.
