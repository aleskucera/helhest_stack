from __future__ import annotations

import warp as wp

wp.init()


@wp.func
def _accumulate_row(
    JtJ: wp.array(dtype=wp.float32, ndim=2),
    Jtr: wp.array(dtype=wp.float32),
    j0: wp.float32,
    j1: wp.float32,
    j2: wp.float32,
    j3: wp.float32,
    j4: wp.float32,
    j5: wp.float32,
    r: wp.float32,
    w: wp.float32,
):
    """Atomically add w·J^T J (6x6) and w·J^T r (6x1) into accumulators."""
    wp.atomic_add(Jtr, 0, w * j0 * r)
    wp.atomic_add(Jtr, 1, w * j1 * r)
    wp.atomic_add(Jtr, 2, w * j2 * r)
    wp.atomic_add(Jtr, 3, w * j3 * r)
    wp.atomic_add(Jtr, 4, w * j4 * r)
    wp.atomic_add(Jtr, 5, w * j5 * r)

    wp.atomic_add(JtJ, 0, 0, w * j0 * j0)
    wp.atomic_add(JtJ, 0, 1, w * j0 * j1)
    wp.atomic_add(JtJ, 0, 2, w * j0 * j2)
    wp.atomic_add(JtJ, 0, 3, w * j0 * j3)
    wp.atomic_add(JtJ, 0, 4, w * j0 * j4)
    wp.atomic_add(JtJ, 0, 5, w * j0 * j5)
    wp.atomic_add(JtJ, 1, 1, w * j1 * j1)
    wp.atomic_add(JtJ, 1, 2, w * j1 * j2)
    wp.atomic_add(JtJ, 1, 3, w * j1 * j3)
    wp.atomic_add(JtJ, 1, 4, w * j1 * j4)
    wp.atomic_add(JtJ, 1, 5, w * j1 * j5)
    wp.atomic_add(JtJ, 2, 2, w * j2 * j2)
    wp.atomic_add(JtJ, 2, 3, w * j2 * j3)
    wp.atomic_add(JtJ, 2, 4, w * j2 * j4)
    wp.atomic_add(JtJ, 2, 5, w * j2 * j5)
    wp.atomic_add(JtJ, 3, 3, w * j3 * j3)
    wp.atomic_add(JtJ, 3, 4, w * j3 * j4)
    wp.atomic_add(JtJ, 3, 5, w * j3 * j5)
    wp.atomic_add(JtJ, 4, 4, w * j4 * j4)
    wp.atomic_add(JtJ, 4, 5, w * j4 * j5)
    wp.atomic_add(JtJ, 5, 5, w * j5 * j5)


@wp.func
def _power_iterate(C: wp.mat33, v0: wp.vec3, iters: int) -> wp.vec3:
    """Largest-eigenvector power iteration on a 3x3 symmetric matrix."""
    v = wp.normalize(v0)
    for _ in range(iters):
        v = C @ v
        n = wp.length(v)
        if n > 1.0e-20:
            v = v / n
    return v


@wp.kernel
def estimate_normals_kernel(
    grid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    radius: wp.float32,
    min_neighbors: wp.int32,
    power_iters: wp.int32,
    normals: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
):
    """Per-point PCA normal via power iteration on covariance."""
    i = wp.tid()
    p = points[i]

    # Gather neighbor statistics.
    mean = wp.vec3(0.0, 0.0, 0.0)
    count = int(0)
    neighbors = wp.hash_grid_query(grid, p, radius)
    for index in neighbors:
        q = points[index]
        d = wp.length(q - p)
        if d <= radius:
            mean = mean + q
            count += 1

    if count < min_neighbors:
        normals[i] = wp.vec3(0.0, 0.0, 0.0)
        valid[i] = 0
        return

    mean = mean / float(count)

    # Covariance.
    c00 = float(0.0)
    c01 = float(0.0)
    c02 = float(0.0)
    c11 = float(0.0)
    c12 = float(0.0)
    c22 = float(0.0)
    neighbors2 = wp.hash_grid_query(grid, p, radius)
    for index in neighbors2:
        q = points[index]
        d = wp.length(q - p)
        if d <= radius:
            dq = q - mean
            c00 += dq[0] * dq[0]
            c01 += dq[0] * dq[1]
            c02 += dq[0] * dq[2]
            c11 += dq[1] * dq[1]
            c12 += dq[1] * dq[2]
            c22 += dq[2] * dq[2]

    inv_n = 1.0 / float(count)
    C = wp.mat33(
        c00 * inv_n,
        c01 * inv_n,
        c02 * inv_n,
        c01 * inv_n,
        c11 * inv_n,
        c12 * inv_n,
        c02 * inv_n,
        c12 * inv_n,
        c22 * inv_n,
    )

    # Largest eigenvector of C.
    v1 = _power_iterate(C, wp.vec3(1.0, 0.0, 0.0), power_iters)
    l1 = wp.dot(v1, C @ v1)

    # Deflate and find second-largest.
    v1v1 = wp.outer(v1, v1)
    D = C - l1 * v1v1
    # Seed orthogonal to v1 for the second iteration.
    seed = wp.vec3(0.0, 1.0, 0.0)
    if wp.abs(v1[1]) > 0.9:
        seed = wp.vec3(1.0, 0.0, 0.0)
    v2 = _power_iterate(D, seed, power_iters)

    # Smallest eigenvector is orthogonal to the top two.
    n = wp.normalize(wp.cross(v1, v2))

    normals[i] = n
    valid[i] = 1


@wp.kernel
def voxel_accumulate_kernel(
    points: wp.array(dtype=wp.vec3),
    origin: wp.vec3,
    inv_voxel: wp.float32,
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    occupied: wp.array(dtype=wp.int32),
    occ_counter: wp.array(dtype=wp.int32),
):
    """Assign each point to its voxel bucket and accumulate sum/count.

    Records each bucket the moment its count goes 0->1 (exactly one thread wins
    that transition), so the compact pass can visit only occupied buckets instead
    of scanning the whole grid.
    """
    i = wp.tid()
    p = points[i]
    fx = (p[0] - origin[0]) * inv_voxel
    fy = (p[1] - origin[1]) * inv_voxel
    fz = (p[2] - origin[2]) * inv_voxel
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    if ix < 0 or ix >= nx:
        return
    if iy < 0 or iy >= ny:
        return
    if iz < 0 or iz >= nz:
        return
    bucket = ix + iy * nx + iz * nx * ny
    if wp.atomic_add(counts, bucket, 1) == 0:
        occupied[wp.atomic_add(occ_counter, 0, 1)] = bucket
    wp.atomic_add(sums, bucket, p)


@wp.kernel
def voxel_compact_kernel(
    occupied: wp.array(dtype=wp.int32),
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    out_points: wp.array(dtype=wp.vec3),
):
    """Emit one centroid per occupied bucket (indexed off `occupied`) and reset
    that bucket to empty, leaving `sums`/`counts` zeroed for the next call."""
    idx = occupied[wp.tid()]
    out_points[wp.tid()] = sums[idx] / float(counts[idx])
    sums[idx] = wp.vec3(0.0, 0.0, 0.0)
    counts[idx] = 0


@wp.kernel
def transform_points_kernel(
    src: wp.array(dtype=wp.vec3),
    R: wp.mat33,
    t: wp.vec3,
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    out[i] = R @ src[i] + t


@wp.kernel
def accumulate_system_kernel(
    grid: wp.uint64,
    target: wp.array(dtype=wp.vec3),
    target_normals: wp.array(dtype=wp.vec3),
    target_valid: wp.array(dtype=wp.int32),
    transformed_src: wp.array(dtype=wp.vec3),
    max_dist: wp.float32,
    huber_delta: wp.float32,
    JtJ: wp.array(dtype=wp.float32, ndim=2),
    Jtr: wp.array(dtype=wp.float32),
    cost: wp.array(dtype=wp.float32),
    num_inliers: wp.array(dtype=wp.int32),
):
    """For each source point: find nearest target, build point-to-plane Jacobian row."""
    i = wp.tid()
    p = transformed_src[i]

    best = max_dist * max_dist
    best_idx = int(-1)
    neighbors = wp.hash_grid_query(grid, p, max_dist)
    for index in neighbors:
        if target_valid[index] == 0:
            continue
        diff = target[index] - p
        d2 = wp.dot(diff, diff)
        if d2 < best:
            best = d2
            best_idx = index

    if best_idx < 0:
        return

    n = target_normals[best_idx]
    q = target[best_idx]
    r = wp.dot(p - q, n)

    # Huber weight.
    ar = wp.abs(r)
    w = float(1.0)
    if ar > huber_delta:
        w = huber_delta / ar

    # Jacobian row J = [p × n, n] (6-vector: rotation then translation).
    j0 = p[1] * n[2] - p[2] * n[1]
    j1 = p[2] * n[0] - p[0] * n[2]
    j2 = p[0] * n[1] - p[1] * n[0]
    j3 = n[0]
    j4 = n[1]
    j5 = n[2]

    _accumulate_row(JtJ, Jtr, j0, j1, j2, j3, j4, j5, r, w)
    wp.atomic_add(cost, 0, w * r * r)
    wp.atomic_add(num_inliers, 0, 1)
