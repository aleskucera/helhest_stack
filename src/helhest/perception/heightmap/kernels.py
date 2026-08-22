from __future__ import annotations

import warp as wp

wp.init()


@wp.kernel
def rasterize_all_kernel(
    points: wp.array(dtype=wp.vec3),
    xmin: float,
    ymin: float,
    inv_res: float,
    width: int,
    height: int,
    max_map: wp.array2d(dtype=wp.float32),
    min_map: wp.array2d(dtype=wp.float32),
    sum_map: wp.array2d(dtype=wp.float32),
    count_map: wp.array2d(dtype=wp.int32),
    sum64_map: wp.array2d(dtype=wp.float64),
    sumsq64_map: wp.array2d(dtype=wp.float64),
):
    """Scatter (N, 3) points into max/min/sum/count per cell via atomics.

    sum64/sumsq64 accumulate in float64: cell heights are O(10) m but per-cell
    sigma is O(0.02) m, so var = E[z^2] - E[z]^2 cancels ~7 significant digits --
    float32 accumulation would destroy the std estimate.
    """
    tid = wp.tid()
    p = points[tid]
    j = int((p[0] - xmin) * inv_res)
    i = int((p[1] - ymin) * inv_res)
    if i < 0 or i >= height or j < 0 or j >= width:
        return
    wp.atomic_max(max_map, i, j, p[2])
    wp.atomic_min(min_map, i, j, p[2])
    wp.atomic_add(sum_map, i, j, p[2])
    wp.atomic_add(count_map, i, j, 1)
    z64 = wp.float64(p[2])
    wp.atomic_add(sum64_map, i, j, z64)
    wp.atomic_add(sumsq64_map, i, j, z64 * z64)


@wp.kernel
def finalize_kernel(
    sum_map: wp.array2d(dtype=wp.float32),
    count_map: wp.array2d(dtype=wp.int32),
    max_map: wp.array2d(dtype=wp.float32),
    min_map: wp.array2d(dtype=wp.float32),
    mean_map: wp.array2d(dtype=wp.float32),
    sum64_map: wp.array2d(dtype=wp.float64),
    sumsq64_map: wp.array2d(dtype=wp.float64),
    std_map: wp.array2d(dtype=wp.float32),
):
    """Finalize per-cell reductions: mean = sum / count; NaN for empty cells.

    std needs >= 2 samples to be defined (a single point has zero sample variance
    but no meaningful spread estimate); float64 var is clamped at 0 before sqrt to
    guard against a tiny negative from float64 cancellation.
    """
    i, j = wp.tid()
    c = count_map[i, j]
    nan = wp.float32(wp.nan)
    if c > 0:
        mean_map[i, j] = sum_map[i, j] / float(c)
    else:
        mean_map[i, j] = nan
        max_map[i, j] = nan
        min_map[i, j] = nan
    if c >= 2:
        c64 = wp.float64(c)
        m64 = sum64_map[i, j] / c64
        var64 = sumsq64_map[i, j] / c64 - m64 * m64
        var64 = wp.max(var64, wp.float64(0.0))
        std_map[i, j] = wp.float32(wp.sqrt(var64))
    else:
        std_map[i, j] = nan


@wp.kernel
def stamp_footprint_kernel(
    primary: wp.array2d(dtype=wp.float32),
    i0: int,
    j0: int,
    xmin: float,
    ymin: float,
    res: float,
    a: float,
    b: float,
    c: float,
    fill_only: int,  # 1 = only write cells currently NaN (no real measurement)
):
    """Force a flat ground plane z = a*x + b*y + c over the footprint rectangle.

    Launched over a (di, dj) block anchored at (i0, j0). The plane is constant in
    the robot body frame (the robot's footprint is flat there); expressed in the
    gravity-aligned grid frame it tilts with the robot's roll/pitch, so a/b are
    generally non-zero. x/y are cell-center coordinates in the grid frame.
    """
    di, dj = wp.tid()
    i = i0 + di
    j = j0 + dj
    if fill_only == 1 and not wp.isnan(primary[i, j]):
        return
    x = xmin + (float(j) + 0.5) * res
    y = ymin + (float(i) + 0.5) * res
    primary[i, j] = a * x + b * y + c


@wp.kernel
def blur_axis_kernel(
    src: wp.array2d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    radius: int,
    axis: int,  # 0 = vertical (along i), 1 = horizontal (along j)
    dst: wp.array2d(dtype=wp.float32),
):
    """NaN-aware separable Gaussian blur along one axis."""
    i, j = wp.tid()
    h = src.shape[0]
    w = src.shape[1]
    acc = float(0.0)
    wsum = float(0.0)
    for k in range(-radius, radius + 1):
        ii = i
        jj = j
        if axis == 0:
            ii = i + k
        else:
            jj = j + k
        if ii < 0 or ii >= h or jj < 0 or jj >= w:
            continue
        v = src[ii, jj]
        if wp.isnan(v):
            continue
        wgt = weights[k + radius]
        acc += v * wgt
        wsum += wgt
    if wsum > 0.0:
        dst[i, j] = acc / wsum
    else:
        dst[i, j] = wp.float32(wp.nan)


@wp.kernel
def diffuse_step_kernel(
    src: wp.array2d(dtype=wp.float32),
    fixed: wp.array2d(dtype=wp.int32),
    dst: wp.array2d(dtype=wp.float32),
):
    """One Jacobi diffusion step: non-fixed cells ← mean of non-NaN neighbors."""
    i, j = wp.tid()
    if fixed[i, j] == 1:
        dst[i, j] = src[i, j]
        return
    h = src.shape[0]
    w = src.shape[1]
    acc = float(0.0)
    count = float(0.0)
    if i > 0:
        v = src[i - 1, j]
        if not wp.isnan(v):
            acc += v
            count += 1.0
    if i < h - 1:
        v = src[i + 1, j]
        if not wp.isnan(v):
            acc += v
            count += 1.0
    if j > 0:
        v = src[i, j - 1]
        if not wp.isnan(v):
            acc += v
            count += 1.0
    if j < w - 1:
        v = src[i, j + 1]
        if not wp.isnan(v):
            acc += v
            count += 1.0
    if count > 0.0:
        dst[i, j] = acc / count
    else:
        dst[i, j] = wp.float32(wp.nan)


@wp.kernel
def downsample_kernel(
    src: wp.array2d(dtype=wp.float32),
    src_fixed: wp.array2d(dtype=wp.int32),
    dst: wp.array2d(dtype=wp.float32),
    dst_fixed: wp.array2d(dtype=wp.int32),
):
    """2x2 average downsample, NaN-aware. A cell is fixed if any source cell was fixed."""
    i, j = wp.tid()
    si = i * 2
    sj = j * 2
    sh = src.shape[0]
    sw = src.shape[1]
    acc = float(0.0)
    count = float(0.0)
    any_fixed = int(0)
    for di in range(2):
        for dj in range(2):
            ii = si + di
            jj = sj + dj
            if ii < sh and jj < sw:
                v = src[ii, jj]
                if not wp.isnan(v):
                    acc += v
                    count += 1.0
                if src_fixed[ii, jj] == 1:
                    any_fixed = 1
    if count > 0.0:
        dst[i, j] = acc / count
    else:
        dst[i, j] = wp.float32(wp.nan)
    dst_fixed[i, j] = any_fixed


@wp.kernel
def upsample_inject_kernel(
    coarse: wp.array2d(dtype=wp.float32),
    fine: wp.array2d(dtype=wp.float32),
    fine_fixed: wp.array2d(dtype=wp.int32),
):
    """Upsample coarse solution to fine grid; only write into non-fixed cells."""
    i, j = wp.tid()
    if fine_fixed[i, j] == 1:
        return
    ci = i / 2
    cj = j / 2
    if ci < coarse.shape[0] and cj < coarse.shape[1]:
        v = coarse[ci, cj]
        if not wp.isnan(v):
            fine[i, j] = v
