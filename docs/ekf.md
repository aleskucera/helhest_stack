# Extended Kalman Filter — Helhest localization

## 1. State and sensors

Two filter variants live in `helhest.filtering.ekf`:

| Class | State \(\mathbf{x}\) | Dimension |
|-------|----------------------|-----------|
| `EKF` | \([x,\; y,\; \psi]\) | 3 |
| `EKF6D` | \([x,\; y,\; \psi,\; \dot{x}^W,\; \dot{y}^W,\; \dot{\psi}]\) | 6 |

Both observe the same two sensor sources:

* **ICP** — LiDAR scan-to-map registration returns an absolute world-frame pose \([x, y, \psi]\). Applied only when the registration succeeds.
* **Odom + IMU** — wheel-encoder dead-reckoning (xy) fused with gyro integration (\(\psi\)). Applied every frame regardless of ICP status, providing a fallback when ICP is rejected.

`EKF6D` is used in the closed-loop pipeline demo; `EKF` exists as a simpler 3-DOF alternative.

---

## 2. Process model and Jacobian

The predict step is deliberately kept **outside** the filter: the caller computes both the nonlinear prediction \(f(\mathbf{x}_t, \mathbf{u})\) and its linearisation \(F = \partial f/\partial \mathbf{x}\), then passes them in. The filter itself only propagates the covariance:

\[
\mathbf{x}^- = f(\mathbf{x}_t,\, \mathbf{u}_t), \qquad P^- = F P F^\top + Q
\]

The kinematic model \(f\) and its Jacobian \(F\) are described in `docs/state_model_proper.md`. The key structural fact that shapes the implementation is that the `ForwardSimulator` derives velocity from the **input** \(\mathbf{u}\) and the heading \(\psi\) at each step — it never reads the stored velocity states \([\dot{x}^W, \dot{y}^W, \dot{\psi}]\) from \(\mathbf{x}\). Consequently, columns 3–5 of \(F\) are analytically zero:

\[
F = \begin{pmatrix} F_{3\times3} & \mathbf{0}_{3\times3} \\ F_{3\times3}^{vel} & \mathbf{0}_{3\times3} \end{pmatrix}
\]

This means perturbing the velocity sub-state has no effect on the next state, and the Jacobian computation only needs \(3 \times 2 = 6\) forward-simulation rollouts (one central-difference pair per position/heading dimension), run in a single batched GPU launch.

On flat ground \(F\) can be written analytically (see `state_model_proper.md`). On real terrain it is computed numerically by `jacobian_F_6d` using central differences with perturbation \(\delta = 10^{-4}\).

---

## 3. Measurement updates

Both sensors observe the **same** three states — position and heading — through the observation matrix:

\[
H = \begin{bmatrix} I_3 \mid \mathbf{0}_3 \end{bmatrix} \quad \in \mathbb{R}^{3 \times 6}
\]

(For `EKF`, \(H = I_3\) trivially.) The update equations are standard EKF:

\[
S = H P^- H^\top + R, \qquad K = P^- H^\top S^{-1}
\]
\[
\mathbf{y} = \mathbf{z} - H\mathbf{x}^-, \qquad \mathbf{y}[2] \leftarrow \operatorname{wrap}(\mathbf{y}[2])
\]
\[
\mathbf{x}^+ = \mathbf{x}^- + K\mathbf{y}, \qquad P^+ = (I - KH)\,P^-
\]

The heading innovation \(\mathbf{y}[2]\) is always wrapped to \((-\pi, \pi]\) before the update so that, e.g., a 1° overshoot past north is not interpreted as a 359° error.

The two update paths differ only in their noise covariance:

| Path | Noise matrix | Typical \(\sigma\) |
|------|--------------|--------------------|
| `update_icp` | \(R_\text{ICP}\) | 5 cm, 5 cm, 1° |
| `update_odom_imu` | \(R_\text{odom}\) | 30 cm, 30 cm, 10° |

The larger odom noise reflects its role as a continuous-coverage fallback, not a precision sensor. When ICP succeeds, both updates are applied in sequence (ICP first, then odom); the odom update slightly widens the distribution again where the dead-reckoning and ICP disagree, which is the correct Bayesian behaviour.

---

## 4. The EKF in `pipeline_ekf.py` — SE(2)/SE(3) interface

The closed-loop pipeline operates in two representations simultaneously:

* **SE(3)** — 4×4 homogeneous transforms, used by the `Localizer`, the map accumulator, and the point-cloud transform chain.
* **EKF state vector** — flat \(\mathbb{R}^6\), used by the filter.

The conversion between them happens at two points each frame.

### 4.1 EKF state → SE(3) (for ICP seeding and downstream use)

After the predict step the EKF state \([x, y, \psi]\) is lifted to a 4×4 SE(3) matrix to seed the ICP registration and to transform point clouds:

\[
T_\text{pred} = \begin{pmatrix} \cos\psi & -\sin\psi & 0 & x \\ \sin\psi & \phantom{-}\cos\psi & 0 & y \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
\]

This is a **flat planar** embedding: \(z = 0\), pitch \(\theta = 0\), roll \(\phi = 0\). It is valid in the demo because the simulated worlds are flat-ground environments and the process model is also flat-ground (`_build_flat_sims` loads a zero-elevation terrain). Roll and pitch are never excited.

After both updates, the fused state is turned back into a pose matrix the same way:

\[
T_{wb} = \operatorname{se2\_to\_mat}(\mathbf{x}[0],\, \mathbf{x}[1],\, \mathbf{x}[2])
\]

`T_wb` is then used for all downstream stages — map accumulation, MPPI planning, cost-to-go — unchanged from the non-EKF pipeline.

### 4.2 SE(3) → EKF measurement (reading the ICP result)

The ICP registration returns a full SE(3) pose `outcome.pose`. The scalar triple \([x, y, \psi]\) is extracted by reading the translation and computing the yaw from the rotation columns:

\[
x = T[0,3], \quad y = T[1,3], \quad \psi = \operatorname{atan2}(T[1,0],\; T[0,0])
\]

Only these three numbers enter the EKF as the measurement \(\mathbf{z}_\text{ICP}\). The z-translation, pitch, and roll returned by ICP are discarded in the demo (they are zero on flat ground anyway).

### 4.3 Per-frame flow summary

```
frame t:
  1. PREDICT
     x_pred  = predict_q6d(ekf.x, u_{t-1}, sim_pred)   # nonlinear f, GPU
     F       = jacobian_F_6d(ekf.x, u_{t-1}, sim_jac)  # numerical ∂f/∂x, GPU
     ekf.predict(F, x_pred)                             # P⁻ = F P Fᵀ + Q

  2. ICP SEED & UPDATE  (if map available)
     T_pred  = se2_to_mat(ekf.x[0:3])                  # EKF → SE(3)
     outcome = localizer.update(scan, T_pred, map, ...)
     if outcome.status == "ok":
         z_icp = mat_to_se2(outcome.pose)[:3]           # SE(3) → R³
         ekf.update_icp(z_icp)

  3. ODOM UPDATE  (every frame)
     T_odom  = T_odom @ odom_step(...)                  # advance dead-reckoning
     z_odom  = mat_to_se2(T_odom)[:3]                   # SE(3) → R³
     ekf.update_odom_imu(z_odom)

  4. RECONSTRUCT POSE
     T_wb    = se2_to_mat(ekf.x[0], ekf.x[1], ekf.x[2])  # EKF → SE(3)
     # T_wb drives map accumulation, MPPI, and cost-to-go
```

### 4.4 Why the flat embedding is a demo limitation

On real hardware with the gravity-anchored ICP (`gravity_weight > 0`), `outcome.pose` carries genuine roll and pitch from the IMU. Replacing `world_T_base` with `se2_to_mat(ekf.x[0:3])` would zero those angles out, misaligning the footprint stamp and breaking the z-accuracy of the accumulated map on sloped terrain. A production integration would preserve the ICP pose's z-column and z-translation, overwriting only the xy-position and the planar rotation block with the EKF result.
