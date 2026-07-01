# NumPy implementation (`src/kinematic_helhest/reference/`)

Single-threaded, pure NumPy. Entry points: `state.make_state` (init) and `state.step` (advance).

Each step follows **predict → project**:

1. **Build envelope** (once): morphologically dilate the raw heightmap by a spherical cap of radius `R` → wheel placement surface `surf`.
2. **Turning params**: sample friction `mu_i` at the three contact points, weight by normal loads `N_i` → friction-weighted ICR offset `x_icr` and turn-resistance factor `alpha`.
3. **Twist (predict)**: skid-steer formula from wheel speeds `omega = [wL, wR, w_rear]` → body-frame `(vx, vy, wz)`. `vy = −x_icr · wz`; forward speed divided by `alpha`.
4. **World velocity + Euler**: rotate `(vx, vy, 0)` by the current orientation matrix `R` (so climbing on a slope reduces horizontal progress), integrate `(x, y, yaw)` by `dt`.
5. **Settle (project)**: finite-difference Newton solve for `(z, pitch, roll)` at the new planar pose — drives wheel hub clearances to zero against `surf`. Warm-started from previous solution.
6. **Normal loads**: quasi-static 3×3 linear solve (vertical force balance + torque about CoM) → `N_i` [N] per wheel.
7. **Chassis clearance**: sample raw terrain under the body grid → minimum belly gap; negative → `valid = False` (high-centred).

# Warp implementation (`src/kinematic_helhest/engine/`)

GPU-parallel, batched. Runs thousands of rollouts simultaneously (MPPI). The same physics as the reference pipeline is split across a set of Warp kernels and inlined device functions.

## Threading model

The fundamental mapping is **one Warp thread = one batch rollout** for all simulation kernels (`init_state_kernel*`, `step_kernel*`, `rollout_kernel`). `wp.tid()` in those kernels returns a single integer `b ∈ [0, B)` that indexes the rollout. Terrain dilation kernels use a different mapping: **one thread per grid cell** `(iy, ix)` for the shared-terrain path, or `(b, iy, ix)` for the per-batch differentiable path.

---

## Terrain dilation pipeline

Run once before simulation (off the autodiff tape in the forward path; only `gather_bt` is on-tape in the differentiable path).

### `_contact_kernel` — compute-bound
`wp.tid()` → `(iy, ix)`, one thread per output heightmap cell.

| | |
|---|---|
| **Inputs** | raw elevation `[ny, nx]`, cell size, wheel radius `R`, envelope radius in cells |
| **Outputs** | arg-max neighbor indices `contact_iy`, `contact_ix [ny, nx]`; spherical-cap offset `contact_cap [ny, nx]` |

For each cell the thread loops over the disk of radius `env_radius` cells, computes the spherical cap height $\sqrt{R^2 - d^2} - R$ for each neighbor, and records the neighbor that maximises `elevation + cap`. The winner's cell index and cap value are stored for the subsequent gather. The work per thread scales as $O((2r+1)^2)$ where $r$ = `env_radius`, making this **compute-bound** with scattered global reads.

### `_gather_kernel` — memory-bound
`wp.tid()` → `(iy, ix)`, one thread per output cell.

| | |
|---|---|
| **Inputs** | elevation `[ny, nx]`, `contact_iy`, `contact_ix`, `contact_cap` (all `[ny, nx]`) |
| **Output** | `envelope[iy, ix] = elevation[contact_iy[iy,ix], contact_ix[iy,ix]] + contact_cap[iy,ix]` |

Trivial arithmetic; dominated by 4 indexed reads and 1 write → **memory-bound**.

### `pad_edge` — memory-bound (differentiable path only)
`wp.tid()` → `(b, py, px)` over the padded shape `[B, pny, pnx]`.

| | |
|---|---|
| **Inputs** | raw elevation `[B, ny, nx]`, pad radius |
| **Output** | edge-replicated padded elevation `[B, pny, pnx]` |

Each thread copies one element with clamped-index addressing → **memory-bound**.

### `contact_tiled` — compute-bound with shared-memory reuse (differentiable path only)
`wp.launch_tiled(dim=(B, n_tiles_y, n_tiles_x), block_dim=128)`. One warp-tile per `(batch, tile_y, tile_x)`.

| | |
|---|---|
| **Inputs** | padded elevation `[B, pny, pnx]`, disk offset table `off_dy`, `off_dx`, `off_cap [K]` |
| **Output** | arg-max index table `best_k [B, ny, nx]` |

The tile block loads a `(T + 2R) × (T + 2R)` halo from padded elevation into shared memory, then iterates over all `K` disk offsets using `tile_map`. For each offset it adds `off_cap[k]` to the shifted view and updates a running arg-max index `bk` and running max `acc`. The final index is stored to `best_k`. Shared memory amortizes global reads across all `K` offset comparisons → **compute-bound**.

### `gather_bt` — memory-bound, on autodiff tape
`wp.tid()` → `(b, iy, ix)`, one thread per `(batch, cell)`.

| | |
|---|---|
| **Inputs** | elevation `[B, ny, nx]`, `best_k [B, ny, nx]`, disk offset table |
| **Output** | `envelope[b, iy, ix] = elevation[b, qy, qx] + off_cap[k]` where `(qy, qx, k)` come from `best_k` |

Recorded on `wp.Tape`; its custom adjoint scatters cotangents back to `elevation` via bilinear atomic scatter → **memory-bound**.

---

## Simulation kernels

### `init_state_kernel` / `init_state_kernel_bt` — compute-bound
`wp.tid()` → `b ∈ [0, B)`.

| | |
|---|---|
| **Inputs** | envelope (`[ny,nx]` shared / `[B,ny,nx]` per-batch), Grid/Robot/Solver structs, start poses `[B] vec3(x,y,yaw)` |
| **Outputs** | `controlled[0, b] = (x,y,yaw)`, `derived[0, b] = (z,pitch,roll)` |

Each thread bilinear-samples the envelope at `(x, y)` for an initial `z₀ = H_\text{eff}(x,y) + R`, then calls `settle` to find the converged `(z, pitch, roll)`. Cost is dominated by the Newton loop → **compute-bound**. The `_bt` variant indexes `envelope[b]` for per-rollout terrain; used by `DifferentiableSimulator` on tape.

### `step_kernel` / `step_kernel_bt` — compute-bound
`wp.tid()` → `b ∈ [0, B)`.

| | |
|---|---|
| **Inputs** | envelope, elevation, friction grids; Robot/Solver/Grid structs; wheel speeds `[B] vec3(ωL, ωR, ωrear)`; `controlled [B] vec3`, `derived [B] vec3` |
| **Outputs** | `controlled_next [B] vec3`, `derived_next [B] vec3`, `loads_out [B] vec3(N₀,N₁,N₂)`, `turn_out [B] vec2(alpha, x_icr)`, `clear_out [B] float`, `resid_out [B] float` |

Each thread executes three inlined physics functions in sequence (see below): `step_predict` → `settle` → `step_finalize`. Cost is dominated by two `normal_loads` solves and one `settle` Newton loop → **compute-bound**. The `_bt` variant reads `envelope[b]` / `elevation[b]` / `friction[b]` and is recorded on `wp.Tape` for autodiff. Used by `DifferentiableSimulator` in a loop over `T` timesteps.

### `rollout_kernel` — compute-bound, register-heavy (forward path only)
`wp.tid()` → `b ∈ [0, B)`.

| | |
|---|---|
| **Inputs** | `n_steps` scalar; shared terrain grids `[ny,nx]`; start poses `[B] vec3`; control sequence `[T, B] vec3` |
| **Outputs** | `controlled [T+1, B] vec3`, `derived [T+1, B] vec3`, `loads_out [T, B] vec3`, `turn_out [T, B] vec2`, `clear_out [T, B] float`, `resid_out [T, B] float` |

The fused rollout. Each thread inlines the full init (settle from start pose) and then loops `t = 0..T-1`, inlining `step_predict` + `settle` + diagnostics each iteration. The intermediate planar and derived states `(pc, tc)` are carried in **registers** between steps — there is no global memory round-trip between iterations. Only the final states and diagnostics are written out. This is the performance-critical path for MPPI planning. **Not autodiffable** (register carry breaks tape). **Compute-bound** with high register pressure.

---

## Inlined device functions (`@wp.func`)

These are not independent kernels; they are inlined into the kernels above at compile time. Each runs in the context of a batch thread `b`.

| Function | Inputs | Output | Role |
|----------|--------|--------|------|
| `settle` | envelope, structs, `(x,y,yaw)`, `(z,pitch,roll)_init` | converged `(z,pitch,roll)` | Analytic-Jacobian Newton: drives $c_i = z_{\text{hub},i} - H_\text{eff} - R = 0$ for all 3 wheels. 3 equations / 3 unknowns. Has a custom `@wp.func_grad` implementing the implicit function theorem adjoint for backprop through the converged root. |
| `normal_loads` | envelope, structs, `R`, body origin `p` | `vec3(N₀, N₁, N₂)` | Builds a 3×3 system from vertical-force balance and two torque-balance equations about the CoM; solves for contact normal loads via a 3×3 analytic solver. |
| `step_predict` | envelope/friction slices, structs, wheel speeds `ω`, current `(pc, tc)`, `turn_out` buffer | predicted `(xn, yn, yawn)` | Calls `normal_loads` → computes per-wheel grip $w_i = \mu_i N_i$ → derives `x_icr`, `alpha` → applies skid-steer twist → rotates to world frame → Euler-integrates `(x,y,yaw)`. Writes `turn_out[b]`. |
| `step_finalize` | predicted planar pose, settled derived, output buffers, `b` | — (writes to output arrays) | Commits `controlled_next[b]` and `derived_next[b]`; calls `normal_loads` at new pose for `loads_out[b]`; calls `chassis_clearance` for `clear_out[b]`; evaluates settle residual for `resid_out[b]`. |
| `chassis_clearance` | raw elevation, structs, `R`, body origin `p` | min clearance `float` | Loops over all `n_chassis` belly sample points, transforms each to world frame, bilinear-samples the raw elevation (not the envelope), returns the minimum signed gap. Negative → high-centred. |

---

## Summary

| Kernel | `wp.tid()` domain | Bound |
|--------|-------------------|-------|
| `_contact_kernel` | `(iy, ix)` — grid cell | Compute |
| `_gather_kernel` | `(iy, ix)` — grid cell | Memory |
| `pad_edge` | `(b, py, px)` — padded cell | Memory |
| `contact_tiled` | `(b, ti, tj)` — batch × tile | Compute (shared mem) |
| `gather_bt` | `(b, iy, ix)` — batch × cell | Memory |
| `init_state_kernel(_bt)` | `b` — rollout | Compute |
| `step_kernel(_bt)` | `b` — rollout | Compute |
| `rollout_kernel` | `b` — rollout | Compute (register-heavy) |
