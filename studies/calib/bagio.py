"""Read Odin mcap bags into plain arrays: SLAM poses and dTOF clouds.

The only host-side numpy in this pipeline, and it is the boundary CLAUDE.md section 6
allows: parsing an incoming ROS payload plus small host-side control values (4x4 poses).
Clouds go to the device in `mapbuild.py` and stay there.

Odin specifics (ros/odin/odin_elevation.params.yaml): the cloud is already in
`odin1_base_link`, so the sensor TF is identity; `/odin1/odometry` IS the on-device SLAM
pose, so there is no ICP in this path.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
from mcap_ros2.reader import read_ros2_messages

CLOUD_TOPIC = "/odin1/cloud_raw"
ODOM_TOPIC = "/odin1/odometry"

# Robot self-filter box in base_link [m], measured for the Odin mount (odin_elevation.params.yaml).
SELF_X = (-0.05, 0.55)
SELF_Y = (-0.75, 0.75)


@dataclass
class Frame:
    """One dTOF sweep, already in world coordinates."""

    t: float
    points: np.ndarray  # [N, 3] world xyz, self-filtered
    sensor_xyz: np.ndarray  # [3] world sensor origin: range binning + the z-window reference


@dataclass
class Odometry:
    """The SLAM pose track. `rpy` is roll/pitch/yaw [rad], ZYX convention."""

    t: np.ndarray  # [M]
    xyz: np.ndarray  # [M, 3]
    rpy: np.ndarray  # [M, 3]


def quat_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Quaternion -> (roll, pitch, yaw) [rad], ZYX (yaw about world z, then pitch, then roll)."""
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def quat_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion -> 3x3 rotation matrix."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def bag_path(name: str, root: str = "bags") -> str:
    """Directory name (e.g. "ostrich4") -> its single .mcap file."""
    hits = glob.glob(os.path.join(root, name, "*.mcap"))
    if not hits:
        raise FileNotFoundError(f"no .mcap under {os.path.join(root, name)}")
    return hits[0]


def read_odometry(path: str) -> Odometry:
    """The full SLAM pose track, in bag order."""
    t: list[float] = []
    xyz: list[tuple[float, float, float]] = []
    rpy: list[tuple[float, float, float]] = []
    for msg in read_ros2_messages(path, topics=[ODOM_TOPIC]):
        o = msg.ros_msg
        p = o.pose.pose
        q = p.orientation
        t.append(o.header.stamp.sec + o.header.stamp.nanosec * 1e-9)
        xyz.append((p.position.x, p.position.y, p.position.z))
        rpy.append(quat_to_rpy(q.x, q.y, q.z, q.w))
    return Odometry(np.asarray(t), np.asarray(xyz), np.asarray(rpy))


def _unpack_cloud(msg) -> np.ndarray:
    """PointCloud2 -> [N, 3] float32 xyz in the sensor frame, NaN/zero returns dropped.

    The Odin dTOF layout is x,y,z float32 at offsets 0/4/8 in a 19-byte point; the
    remaining fields (intensity, confidence, offset_time) are not read here.
    """
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    raw = raw.reshape(-1, msg.point_step)
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(-1, 3)
    good = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 1e-6)
    return xyz[good]


def read_frames(
    path: str,
    odom: Odometry,
    sensor_roll: float = 0.0,
    sensor_pitch: float = 0.0,
) -> list[Frame]:
    """Every cloud, self-filtered and transformed into the odometry (map) frame.

    The pose is looked up by nearest odometry stamp; clouds and odometry are both published
    at ~14.5 Hz off the same device, so no interpolation is warranted.

    `sensor_roll` / `sensor_pitch` [rad] are an extrinsic correction applied to the cloud in the
    sensor frame before the pose. The bag's `tf_static` carries only an identity
    odin1_base_link -> imu_link, so any physical mount tilt is unmodelled and shows up as a
    constant BODY-frame attitude bias in the map -- which is how it is fitted here.
    """
    sensor_rot = _rpy_to_mat(sensor_roll, sensor_pitch, 0.0)
    frames: list[Frame] = []
    for msg in read_ros2_messages(path, topics=[CLOUD_TOPIC]):
        c = msg.ros_msg
        t = c.header.stamp.sec + c.header.stamp.nanosec * 1e-9
        i = int(np.argmin(np.abs(odom.t - t)))
        if abs(odom.t[i] - t) > 0.1:  # no pose within one frame period -> unusable
            continue
        pts = _unpack_cloud(c)
        if pts.size == 0:
            continue
        on_robot = (
            (pts[:, 0] > SELF_X[0])
            & (pts[:, 0] < SELF_X[1])
            & (pts[:, 1] > SELF_Y[0])
            & (pts[:, 1] < SELF_Y[1])
        )
        pts = pts[~on_robot]
        if pts.size == 0:
            continue
        roll, pitch, yaw = odom.rpy[i]
        rot = _rpy_to_mat(roll, pitch, yaw) @ sensor_rot
        world = pts.astype(np.float64) @ rot.T + odom.xyz[i]
        frames.append(Frame(t=t, points=world.astype(np.float32), sensor_xyz=odom.xyz[i].copy()))
    return frames


def _rpy_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(roll, pitch, yaw) -> 3x3, ZYX: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx
