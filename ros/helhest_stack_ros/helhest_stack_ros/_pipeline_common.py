"""ROS <-> numpy plumbing for navigation_node: a quaternion, a timed cloud, a grid as a cloud.

What is left of the helpers the terrain_toolkit nodes shared (they were removed on 2026-09-27; they
live on in the standalone terrain_toolkit repo).
"""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header

from helhest.perception import TerrainMap


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    s = 0.0 if n == 0.0 else 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy, 0.0],
            [xy + wz, 1.0 - (xx + zz), yz - wx, 0.0],
            [xz - wy, yz + wx, 1.0 - (xx + yy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pointcloud2_to_xyz_time_array(
    msg: PointCloud2, time_field: str = "t"
) -> tuple[np.ndarray, np.ndarray | None]:
    """Parse xyz plus a per-point time field, kept index-aligned.

    Reads `time_field` in the *same* `read_points` call as x/y/z so the shared
    `skip_nans` mask drops the same rows from both — a separate read would
    desync the two. Returns `(xyz (N, 3) float32, times (N,) float64)`, with
    `times = None` when the cloud has no such field (e.g. an unorganized sensor).
    Ouster organized clouds carry `t` = ns since the scan start, per point.
    """
    has_time = any(f.name == time_field for f in msg.fields)
    names = ("x", "y", "z", time_field) if has_time else ("x", "y", "z")
    pc = pc2.read_points(msg, field_names=names, skip_nans=True, reshape_organized_cloud=False)
    if isinstance(pc, np.ndarray) and pc.dtype.names is not None:
        xyz = np.stack([pc["x"], pc["y"], pc["z"]], axis=-1).astype(np.float32)
        times = pc[time_field].astype(np.float64) if has_time else None
    else:  # generator fallback (older sensor_msgs_py)
        arr = np.array(list(pc), dtype=np.float64)
        xyz = arr[:, :3].astype(np.float32)
        times = arr[:, 3] if has_time else None
    return xyz, times


def grid_to_cloud(
    terrain_map: TerrainMap,
    x_min: float,
    y_min: float,
    resolution: float,
    stamp,
    frame_id: str,
    *,
    z_offset: float = 0.0,
    logger=None,
) -> PointCloud2 | None:
    """Convert a TerrainMap into a PointCloud2 with one float32 field per layer.

    `x_min`/`y_min` place the grid origin (min corner) in `frame_id`. `z_offset`
    is added to every elevation so a grid built in a robot-shifted frame can be
    republished at its true world height (see the accumulator node).
    """
    if terrain_map.elevation is None:
        if logger is not None:
            logger.warning("TerrainMap.elevation is None — skipping publish.")
        return None

    rows, cols = terrain_map.elevation.shape  # (ny, nx)

    row_idx = np.arange(rows, dtype=np.float32)
    col_idx = np.arange(cols, dtype=np.float32)
    row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing="ij")

    x_coords = (x_min + (col_grid + 0.5) * resolution).astype(np.float32)
    y_coords = (y_min + (row_grid + 0.5) * resolution).astype(np.float32)

    # as_dict() already skips layers that were not downloaded (None).
    layer_dict = terrain_map.as_dict()
    layer_names = sorted(layer_dict.keys())

    # Drop cells the SupportRatioMask flagged as too far from any real
    # measurement: those have NaN traversability (and NaN slope/step/roughness)
    # even though inpaint filled their elevation. Publishing them would make
    # the heightmap look complete in regions where we actually have no data.
    # When the filter chain is disabled (no traversability layer at all),
    # fall back to elevation finiteness — there's no support signal to use.
    valid = np.isfinite(terrain_map.elevation)
    if terrain_map.traversability is not None:
        valid &= np.isfinite(terrain_map.traversability)

    x_valid = x_coords[valid]
    y_valid = y_coords[valid]
    z_valid = (terrain_map.elevation[valid] + z_offset).astype(np.float32)
    layers_valid = [layer_dict[k][valid].astype(np.float32) for k in layer_names]

    n_pts = x_valid.shape[0]
    point_data = np.column_stack([x_valid, y_valid, z_valid] + layers_valid)

    fields: list[PointField] = []
    offset = 0
    for name in ("x", "y", "z"):
        fields.append(PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1))
        offset += 4
    for name in layer_names:
        fields.append(PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1))
        offset += 4

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id

    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = n_pts
    cloud_msg.fields = fields
    cloud_msg.is_bigendian = False
    cloud_msg.point_step = offset
    cloud_msg.row_step = offset * n_pts
    cloud_msg.is_dense = False
    cloud_msg.data = point_data.astype(np.float32).tobytes()
    return cloud_msg
