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


def elevation_to_cloud(
    elevation: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    stamp,
    frame_id: str,
) -> PointCloud2:
    """An elevation grid [rows, cols] as a PointCloud2: one point per finite cell, at the cell
    centre, with fields x, y, z and `elevation` (= z). `x_min`/`y_min` are the grid's min corner
    in `frame_id`. NaN cells -- never measured -- are left out."""
    rows, cols = elevation.shape
    row_grid, col_grid = np.meshgrid(
        np.arange(rows, dtype=np.float32), np.arange(cols, dtype=np.float32), indexing="ij"
    )
    valid = np.isfinite(elevation)
    x = (x_min + (col_grid + 0.5) * resolution).astype(np.float32)[valid]
    y = (y_min + (row_grid + 0.5) * resolution).astype(np.float32)[valid]
    z = elevation[valid].astype(np.float32)
    points = np.column_stack([x, y, z, z])

    fields = [
        PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, name in enumerate(("x", "y", "z", "elevation"))
    ]
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    cloud = PointCloud2()
    cloud.header = header
    cloud.height = 1
    cloud.width = int(points.shape[0])
    cloud.fields = fields
    cloud.is_bigendian = False
    cloud.point_step = 16
    cloud.row_step = 16 * cloud.width
    cloud.is_dense = False
    cloud.data = points.astype(np.float32).tobytes()
    return cloud
