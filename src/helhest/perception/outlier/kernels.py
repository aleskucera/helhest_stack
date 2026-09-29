from __future__ import annotations

import warp as wp

wp.init()


@wp.kernel
def neighbor_count_kernel(
    grid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    search_radius: wp.float32,
    min_neighbors: wp.int32,
    valid: wp.array(dtype=wp.int32),
):
    """`valid[i] = 1` when at least `min_neighbors` other points lie within `search_radius`.

    Stops counting at `min_neighbors`: the near field holds over a thousand neighbours per ball,
    and only whether the count clears the gate matters.
    """
    i = wp.tid()
    p = points[i]
    count = int(0)
    r2 = search_radius * search_radius
    neighbors = wp.hash_grid_query(grid, p, search_radius)
    for index in neighbors:
        if index == i:
            continue
        diff = points[index] - p
        if wp.dot(diff, diff) > r2:
            continue
        count += 1
        if count >= min_neighbors:
            break
    valid[i] = wp.where(count >= min_neighbors, 1, 0)


@wp.kernel
def compact_inliers_kernel(
    points: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
    out_counter: wp.array(dtype=wp.int32),
    out_points: wp.array(dtype=wp.vec3),
):
    """Write the surviving points to a compact buffer."""
    i = wp.tid()
    if valid[i] == 1:
        slot = wp.atomic_add(out_counter, 0, 1)
        out_points[slot] = points[i]
