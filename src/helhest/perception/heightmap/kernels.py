from __future__ import annotations

import warp as wp

wp.init()


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


@wp.kernel
def finite_mask_kernel(
    src: wp.array2d(dtype=wp.float32),
    fixed: wp.array2d(dtype=wp.int32),
):
    """1 where `src` holds a real height, 0 where it is NaN -- the inpaint's Dirichlet set."""
    i, j = wp.tid()
    v = src[i, j]
    fixed[i, j] = wp.where(wp.isnan(v), 0, 1)
