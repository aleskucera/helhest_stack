"""Differentiable wheel-envelope dilation (Warp).

Grayscale morphological dilation of the raw elevation by the spherical wheel cap:

    envelope[i,j] = max_{|d|<=R} ( elevation[i+dy, j+dx] + sqrt(R^2 - d^2) - R ),  d = |off|*cell_size

The dilation = arg-max of (elevation[neighbor] + cap) over the disk. The gradient is analytical: the
adjoint scatters to the arg-max (contact) cell. Implementations:
  - 2D `_contact_kernel` + `_gather_kernel` (one thread per output cell): the forward planner's
    shared [ny, nx] terrain, and the FD oracle. CPU or CUDA.
  - batched [B, ny, nx] for DifferentiableSimulator (CUDA-only) -- split into `make_tiled_contact`
    (off-tape: a shared-memory tiled arg-max picking the offset `best_k`; needs an edge-padded input
    via `pad_edge`) + `gather_bt` (on-tape: envelope = elevation[contact] + cap, whose cheap scatter
    adjoint IS the analytical gradient -- no autodiff through the convolution).
  - local PATCHES for DifferentiableSimulator's CYLINDER wheel -- `contact_patch` (off-tape) +
    `gather_patch` (on-tape), the same split on a robot-sized square of envelope per (rollout,
    timestep) instead of the whole map. The cylinder element is yaw-dependent, so there is no grid
    to share across headings; see the section at the bottom of this file.

The cylinder element itself is `cylinder_offset_table` (memoised as `cyl_table`, padded per yaw bin
by `cyl_bin_tables`), binned over [0, PI) by `step.yaw_bin`. ONE definition, shared by
ForwardSimulator, the taped patches and `helhest.risk.contact`.

The batched contact also emits a `margin` = winner - runner-up. Freezing the arg-max makes the
gradient exact ONLY while the arg-max cannot move, and `margin` is exactly how far the terrain
would have to move for it to flip -- i.e. the radius within which d(envelope)/d(elevation) is
the true derivative. Study A measured that radius controlling the breakdown (it is
`R - sqrt(R^2 - cell^2)` = 3.6 mm on flat ground at 0.05 m cells, far below realistic map
sigma), so `margin` is the linearisation-validity flag for anything propagating uncertainty
through this dilation. Diagnostic only: it never enters the envelope or any gradient.
"""

import functools
import math

import numpy as np
import warp as wp


@wp.kernel
def _contact_kernel(
    elevation: wp.array2d(dtype=wp.float32),
    cell_size: float,
    wheel_radius: float,
    env_radius: int,
    contact_iy: wp.array2d(dtype=wp.int32),
    contact_ix: wp.array2d(dtype=wp.int32),
    contact_cap: wp.array2d(dtype=wp.float32),
):
    """Non-diff pass: pick the contact cell (arg-max of elevation[neighbor] + cap)."""
    iy, ix = wp.tid()
    ny = elevation.shape[0]
    nx = elevation.shape[1]
    best_lift = float(-1.0e9)
    best_iy = iy
    best_ix = ix
    best_cap = float(0.0)
    for dy in range(-env_radius, env_radius + 1):
        for dx in range(-env_radius, env_radius + 1):
            dist = wp.sqrt(float(dy * dy + dx * dx)) * cell_size
            if dist <= wheel_radius:
                cap = wp.sqrt(wheel_radius * wheel_radius - dist * dist) - wheel_radius
                qy = wp.clamp(iy + dy, 0, ny - 1)
                qx = wp.clamp(ix + dx, 0, nx - 1)
                lift = elevation[qy, qx] + cap
                if lift > best_lift:
                    best_lift = lift
                    best_iy = qy
                    best_ix = qx
                    best_cap = cap
    contact_iy[iy, ix] = best_iy
    contact_ix[iy, ix] = best_ix
    contact_cap[iy, ix] = best_cap


@wp.kernel
def _gather_kernel(
    elevation: wp.array2d(dtype=wp.float32),
    contact_iy: wp.array2d(dtype=wp.int32),
    contact_ix: wp.array2d(dtype=wp.int32),
    contact_cap: wp.array2d(dtype=wp.float32),
    envelope: wp.array2d(dtype=wp.float32),
):
    """Diff pass: envelope = elevation[contact cell] + cap. Adjoint scatters to it."""
    iy, ix = wp.tid()
    envelope[iy, ix] = elevation[contact_iy[iy, ix], contact_ix[iy, ix]] + contact_cap[iy, ix]


def cylinder_offset_table(
    cell_size: float, wheel_radius: float, half_width: float, yaw: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Structuring element of a CYLINDER wheel heading `yaw`, as (dy, dx, cap) offset lists.

    A real wheel is a cylinder with its axis across the body, so along the direction of travel it
    presents exactly the sphere's circle -- the cap is unchanged there -- while laterally it
    reaches only `half_width` instead of the full radius. The element is the rotated rectangle
    2*wheel_radius (along travel) x 2*half_width (across), with

        cap = sqrt(wheel_radius^2 - u^2) - wheel_radius,   u = the ALONG-travel offset only.

    Unlike the sphere's disk this is NOT yaw-invariant, which is why the cylinder needs one
    dilated envelope per yaw bin instead of one grid shared by every rollout and step.
    """
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    # own search radius: the rotated rectangle's corner reaches past wheel_radius, so the disk's
    # env_radius would clip it at 45 deg
    env_radius = int(math.ceil(math.hypot(wheel_radius, half_width) / cell_size))
    dy_l, dx_l, cap_l = [], [], []
    for dy in range(-env_radius, env_radius + 1):
        for dx in range(-env_radius, env_radius + 1):
            wx, wy = dx * cell_size, dy * cell_size
            along = wx * cos_yaw + wy * sin_yaw
            across = -wx * sin_yaw + wy * cos_yaw
            if abs(along) <= wheel_radius and abs(across) <= half_width:
                dy_l.append(dy)
                dx_l.append(dx)
                cap_l.append(math.sqrt(wheel_radius**2 - along**2) - wheel_radius)
    return np.array(dy_l, np.int32), np.array(dx_l, np.int32), np.array(cap_l, np.float32)


# Heading quantization the cylinder table is shared and gathered at. The cylinder element is not
# yaw-invariant (see `cylinder_offset_table`), so every consumer that reuses a table across yaws
# -- the dilation, and the risk estimator's structuring-element upload -- must bin at the SAME
# resolution or they disagree about which element a pose sees.
N_YAW_BINS = 32


@functools.lru_cache(maxsize=None)
def cyl_table(cell_size: float, wheel_radius: float, half_width: float, b: int, n_bins: int):
    """`cylinder_offset_table` for yaw bin `b` of `n_bins` over [0, PI), memoised.

    The table is a deterministic function of its arguments, and profiling the contact estimator
    showed it being rebuilt on every call -- 336 rebuilds across 48 plan evaluations, ~11% of the
    deployed cylinder path. Bins span [0, PI) because the element is symmetric under yaw -> yaw+PI.
    """
    return cylinder_offset_table(cell_size, wheel_radius, half_width, math.pi * b / n_bins)


def wheel_half_width(rp) -> float:
    """[m] half the wheel tread, from the robot's own `wheel_width`.

    ONE definition. The tread is a ruler measurement that lives on `RobotParams`; the structuring
    element needs its half. Carrying a second literal (`0.05`) alongside `wheel_width = 0.10` is
    the same number stored twice, and they only agree until someone widens one of them.
    `wheel_width = None` is the spherical envelope, which has no tread at all.
    """
    if rp.wheel_width is None:
        raise ValueError("wheel_width is None (spherical envelope): no tread half-width exists")
    return 0.5 * float(rp.wheel_width)


@wp.kernel
def _contact_table_kernel(
    elevation: wp.array2d(dtype=wp.float32),
    off_dy: wp.array(dtype=wp.int32),
    off_dx: wp.array(dtype=wp.int32),
    off_cap: wp.array(dtype=wp.float32),
    contact_iy: wp.array2d(dtype=wp.int32),
    contact_ix: wp.array2d(dtype=wp.int32),
    contact_cap: wp.array2d(dtype=wp.float32),
):
    """`_contact_kernel` for an arbitrary structuring element supplied as an offset table.

    The disk version bakes in the in-circle test and computes its cap on device; this one walks a
    host-built (dy, dx, cap) table, so any element shape works -- the cylinder's rotated rectangle
    in practice. Feeds the same `_gather_kernel`.
    """
    iy, ix = wp.tid()
    ny = elevation.shape[0]
    nx = elevation.shape[1]
    best_lift = float(-1.0e9)
    best_iy = iy
    best_ix = ix
    best_cap = float(0.0)
    for k in range(off_dy.shape[0]):
        qy = wp.clamp(iy + off_dy[k], 0, ny - 1)
        qx = wp.clamp(ix + off_dx[k], 0, nx - 1)
        lift = elevation[qy, qx] + off_cap[k]
        if lift > best_lift:
            best_lift = lift
            best_iy = qy
            best_ix = qx
            best_cap = off_cap[k]
    contact_iy[iy, ix] = best_iy
    contact_ix[iy, ix] = best_ix
    contact_cap[iy, ix] = best_cap


# --- batched dilation for DifferentiableSimulator (CUDA): tiled arg-max contact (off-tape, picks
# offset k) + gather (on-tape, the analytical gradient). Splitting them keeps the expensive arg-max
# off the tape and makes the backward a cheap scatter (vs. autodiffing the whole convolution). The
# offset table feeds both the tiled contact (the disk loop) and the gather (best_k -> cell).
def wheel_offset_table(
    env_radius: int, cell_size: float, wheel_radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-circle spherical-cap offsets (dy, dx, cap), precomputed once on the host. Shared by the
    offset-table contact and the gather (both index into it via `best_k`)."""
    dy_l, dx_l, cap_l = [], [], []
    for dy in range(-env_radius, env_radius + 1):
        for dx in range(-env_radius, env_radius + 1):
            dist = math.sqrt(float(dy * dy + dx * dx)) * cell_size
            if dist <= wheel_radius:
                dy_l.append(dy)
                dx_l.append(dx)
                cap_l.append(math.sqrt(wheel_radius * wheel_radius - dist * dist) - wheel_radius)
    return np.array(dy_l, np.int32), np.array(dx_l, np.int32), np.array(cap_l, np.float32)


@wp.kernel
def gather_bt(
    elevation: wp.array3d(dtype=wp.float32),
    best_k: wp.array3d(dtype=wp.float32),
    off_dy: wp.array(dtype=wp.int32),
    off_dx: wp.array(dtype=wp.int32),
    off_cap: wp.array(dtype=wp.float32),
    envelope: wp.array3d(dtype=wp.float32),
):
    """Gather (on-tape, the analytical gradient): envelope = elevation[contact] + cap, where the
    contact offset is `best_k` (fixed = the subgradient). The adjoint is a cheap scatter of
    envelope.grad to the contact cell -- no autodiff through the arg-max."""
    b, iy, ix = wp.tid()
    ny = elevation.shape[1]
    nx = elevation.shape[2]
    k = int(best_k[b, iy, ix])
    qy = wp.clamp(iy + off_dy[k], 0, ny - 1)
    qx = wp.clamp(ix + off_dx[k], 0, nx - 1)
    envelope[b, iy, ix] = elevation[b, qy, qx] + off_cap[k]


# --- shared-memory tiled arg-max contact (B, CUDA): ~2x faster contact, off-tape (non-diff) ---
@wp.func
def _addf(v: wp.float32, c: wp.float32):
    return v + c


@wp.func
def _maxf(a: wp.float32, b: wp.float32):
    return wp.max(a, b)


@wp.func
def _subf(a: wp.float32, b: wp.float32):
    return a - b


@wp.func
def _sel_k(lift: wp.float32, acc: wp.float32, bk: wp.float32, kf: wp.float32):
    """Index update: if this offset's lifted value beats the running max, take its index k."""
    if lift > acc:
        return kf
    return bk


@wp.func
def _sel_second(lift: wp.float32, acc: wp.float32, sec: wp.float32):
    """Runner-up update -- call BEFORE `acc` absorbs `lift`, so `acc` is still the old leader.

    Tracks the SECOND-largest lifted value so the contact can report how close the decision
    was. This is a pure diagnostic: it never enters the envelope or any gradient.
    """
    if lift > acc:
        return acc  # the old leader is demoted to runner-up
    if lift > sec:
        return lift
    return sec


@wp.kernel
def pad_edge(raw: wp.array3d(dtype=wp.float32), pad_r: int, padded: wp.array3d(dtype=wp.float32)):
    """Edge-replicate pad each [ny, nx] slice into a tile-aligned [ny+slack, nx+slack] so the tiled
    halo loads never go out of bounds (used by the tiled contact)."""
    b, py, px = wp.tid()
    ny = raw.shape[1]
    nx = raw.shape[2]
    padded[b, py, px] = raw[b, wp.clamp(py - pad_r, 0, ny - 1), wp.clamp(px - pad_r, 0, nx - 1)]


def make_tiled_contact(env_radius: int, tile: int = 16):
    """Build a batched tiled arg-max contact kernel specialized to `env_radius` (only the tile/halo
    dims are baked; the offset table is passed at launch, so the disk is a RUNTIME loop -- not
    unrolled. Unrolling the disk OR carrying the index in a vec2 both blow up Warp's compile;
    runtime loop + two float tiles (running max + running index) keeps it fast to compile). Maps
    over (B, ny_tiles, nx_tiles) via `launch_tiled`: each block loads its (tile+2R)^2 halo into a
    shared tile once, then arg-max-accumulates the cap-shifted views. Writes `best_k` (the contact
    offset per cell). Input must be edge-padded (`pad_edge`). Off-tape (the `gather` supplies grad).
    """
    R = env_radius
    T = tile
    HALO = T + 2 * R

    @wp.kernel
    def contact_tiled(
        elev_pad: wp.array3d(dtype=wp.float32),  # [B, ny+slack, nx+slack] edge-padded
        off_dy: wp.array(dtype=wp.int32),
        off_dx: wp.array(dtype=wp.int32),
        off_cap: wp.array(dtype=wp.float32),
        best_k: wp.array3d(dtype=wp.float32),  # [B, ny, nx] arg-max offset index
        margin: wp.array3d(dtype=wp.float32),  # [B, ny, nx] winner - runner-up [m]
    ):
        b, ti, tj = wp.tid()
        halo = wp.tile_load(
            elev_pad[b],
            shape=(HALO, HALO),
            offset=(ti * T, tj * T),
            storage="shared",
            bounds_check=True,
        )
        acc = wp.tile_full((T, T), -1.0e9, dtype=wp.float32, storage="register")
        sec = wp.tile_full((T, T), -1.0e9, dtype=wp.float32, storage="register")
        bk = wp.tile_full((T, T), 0.0, dtype=wp.float32, storage="register")
        for k in range(off_dy.shape[0]):
            win = wp.tile_view(halo, offset=(R + off_dy[k], R + off_dx[k]), shape=(T, T))
            capt = wp.tile_full((T, T), off_cap[k], dtype=wp.float32, storage="register")
            lifted = wp.tile_map(_addf, win, capt)
            kt = wp.tile_full((T, T), float(k), dtype=wp.float32, storage="register")
            # both the index and the runner-up must be updated BEFORE acc absorbs `lifted`,
            # because both compare against the OLD running max
            bk = wp.tile_map(_sel_k, lifted, acc, bk, kt)
            sec = wp.tile_map(_sel_second, lifted, acc, sec)
            acc = wp.tile_map(_maxf, acc, lifted)
        wp.tile_store(best_k[b], bk, offset=(ti * T, tj * T), bounds_check=True)
        wp.tile_store(
            margin[b], wp.tile_map(_subf, acc, sec), offset=(ti * T, tj * T), bounds_check=True
        )

    return contact_tiled


# --- LOCAL envelope patches for the taped cylinder path (DifferentiableSimulator) -------------
# The cylinder element is yaw-dependent, so the forward planner's trick -- dilate the whole grid
# once per yaw bin and share it across every rollout and step -- does not carry over to the taped
# path, whose terrain is per-rollout: that would be a [B, n_yaw, ny, nx] stack, MEASURED at 11.8 GB
# on the deployed node's shape (B=4096, 150x150, 32 bins) against a 3.7 GB card.
#
# It is also unnecessary. The envelope is only ever READ under the wheels (`step.py`: `clearances`,
# `settle`'s `sample_height_grad`, `normal_loads`' `sample_normal`), never away from the robot, so
# the taped path materialises only a small square PATCH of the envelope per (rollout, timestep) --
# large enough to cover every wheel-centre stencil of that step, in that step's yaw bin. Memory
# becomes O(B x T x patch) instead of O(B x n_yaw x ny x nx), independent of the map size.
#
# The split is exactly the one the full-grid path uses and the one Section III studies: the arg-max
# is frozen OFF the tape (`contact_patch` -> `best_k`) and only the gather runs ON it
# (`gather_patch`), so the derivative is still the scatter adjoint of a frozen linear gather. What
# changes is WHICH candidates the arg-max ranges over (the yaw bin's cylinder element instead of
# the yaw-invariant disk) and WHERE it is evaluated (a patch per step instead of the whole map).


def cyl_bin_tables(
    cell_size: float, wheel_radius: float, half_width: float, n_bins: int = N_YAW_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every yaw bin's cylinder element, padded into one `[n_bins, K]` table + `k_real`.

    A kernel that picks its element per THREAD cannot hold a ragged list of per-bin tables, so the
    `cyl_table` outputs are padded to the longest bin and the real count carried alongside. Padding
    is never read -- every loop is bounded by `k_real[bin]`. (`helhest.risk.contact` builds the same
    padded shape from the same `cyl_table`; it pads to its own compile-time `MAX_K` instead, because
    its fold carries the candidates in fixed-length registers.)
    """
    tables = [cyl_table(cell_size, wheel_radius, half_width, b, n_bins) for b in range(n_bins)]
    k_real = np.array([len(t[0]) for t in tables], np.int32)
    K = int(k_real.max())
    dy = np.zeros((n_bins, K), np.int32)
    dx = np.zeros((n_bins, K), np.int32)
    cap = np.zeros((n_bins, K), np.float32)
    for b, (t_dy, t_dx, t_cap) in enumerate(tables):
        k = len(t_dy)
        dy[b, :k], dx[b, :k], cap[b, :k] = t_dy, t_dx, t_cap
    return dy, dx, cap, k_real


@wp.func
def patch_source(org: wp.vec2i, i: int, j: int, dy: int, dx: int, ny: int, nx: int) -> wp.vec2i:
    """Cell of the full `[ny, nx]` map that candidate offset (dy, dx) reads for patch cell (i, j).

    Patch cell (i, j) IS map cell (org.y + i, org.x + j); out-of-map reads clamp, which
    edge-replicates exactly as `_locate`'s clamp does when sampling the full grid, so a patch
    hanging off the map border still reproduces the full-grid envelope there. The ONE place this
    mapping lives -- the off-tape arg-max and the on-tape gather must agree on it cell for cell,
    or the frozen index would point somewhere the forward never looked.
    """
    return wp.vec2i(
        wp.clamp(org[0] + j + dx, 0, nx - 1),
        wp.clamp(org[1] + i + dy, 0, ny - 1),
    )


@wp.kernel
def contact_patch(
    elevation: wp.array3d(dtype=wp.float32),  # [B, ny, nx] raw per-rollout terrain
    org: wp.array(dtype=wp.vec2i),  # [B] patch lower-left cell (x, y) in map indices
    bin_idx: wp.array(dtype=wp.int32),  # [B] yaw bin of this rollout at this step
    off_dy: wp.array2d(dtype=wp.int32),  # [n_bins, K] padded element tables
    off_dx: wp.array2d(dtype=wp.int32),
    off_cap: wp.array2d(dtype=wp.float32),
    k_real: wp.array(dtype=wp.int32),  # [n_bins] real candidates before the padding
    best_k: wp.array3d(dtype=wp.int32),  # [B, P, P] arg-max candidate -> written
    margin: wp.array3d(dtype=wp.float32),  # [B, P, P] winner - runner-up [m] -> written
):
    """Off-tape arg-max over one patch: the frozen contact of rollout `b`'s wheel element.

    One thread per patch cell, walking that rollout's yaw-bin element in a runtime loop. Not
    differentiable by construction -- `gather_patch` supplies the gradient -- so the comparison
    chain costs no backward. `margin` is the winner-runner-up gap: how far the terrain must move
    before this cell's arg-max flips, i.e. the radius inside which the frozen gradient is the true
    derivative (Section III). Diagnostic only; it enters neither the envelope nor any gradient.
    """
    b, i, j = wp.tid()
    ny = elevation.shape[1]
    nx = elevation.shape[2]
    bn = bin_idx[b]
    best = float(-1.0e9)
    second = float(-1.0e9)
    bk = int(0)
    for k in range(k_real[bn]):
        q = patch_source(org[b], i, j, off_dy[bn, k], off_dx[bn, k], ny, nx)
        lift = elevation[b, q[1], q[0]] + off_cap[bn, k]
        if lift > best:
            second = best
            best = lift
            bk = k
        elif lift > second:
            second = lift
    best_k[b, i, j] = bk
    margin[b, i, j] = best - second


@wp.kernel
def gather_patch(
    elevation: wp.array3d(dtype=wp.float32),  # [B, ny, nx] raw per-rollout terrain
    org: wp.array(dtype=wp.vec2i),  # [B] patch lower-left cell
    bin_idx: wp.array(dtype=wp.int32),  # [B] yaw bin
    best_k: wp.array3d(dtype=wp.int32),  # [B, P, P] frozen arg-max from `contact_patch`
    off_dy: wp.array2d(dtype=wp.int32),
    off_dx: wp.array2d(dtype=wp.int32),
    off_cap: wp.array2d(dtype=wp.float32),
    patch: wp.array3d(dtype=wp.float32),  # [B, P, P] envelope patch -> written
):
    """On-tape gather: patch = elevation[frozen contact] + cap. Its scatter adjoint IS
    d(envelope)/d(elevation) -- the same analytical gradient the full-grid `gather_bt` gives, with
    the cylinder's candidate set and evaluated only where the wheels read."""
    b, i, j = wp.tid()
    ny = elevation.shape[1]
    nx = elevation.shape[2]
    bn = bin_idx[b]
    k = best_k[b, i, j]
    q = patch_source(org[b], i, j, off_dy[bn, k], off_dx[bn, k], ny, nx)
    patch[b, i, j] = elevation[b, q[1], q[0]] + off_cap[bn, k]
