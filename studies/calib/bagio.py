"""Read Odin mcap bags into plain arrays: SLAM poses and dTOF clouds.

The only host-side numpy in this pipeline, and it is the boundary CLAUDE.md section 6
allows: parsing an incoming ROS payload plus small host-side control values (4x4 poses).
Clouds go to the device in `mapbuild.py` and stay there.

Odin specifics (ros/config/odin.params.yaml): `/odin1/odometry` IS the on-device SLAM pose, so
there is no ICP in this path. The cloud's frame depends on when the bag was recorded: vendor driver
<= 0.13.0 put it in `odin1_base_link` already (identity), while from 0.13.1 (cras_odin_driver
27c11d5, 2026-09-13) it is in `odin1_lidar` and the bag's own /tf_static carries the lidar
calibration. `read_frames` reads the frame_id and applies that static chain, so both kinds of bag
land in the same frame.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages

CLOUD_TOPIC = "/odin1/cloud_raw"
# helhest-nav records the masked cloud only (ros/tools/record_nav.sh); older bags have cloud_raw
CLOUD_TOPICS = (CLOUD_TOPIC, "/odin1/cloud_filtered")
ODOM_TOPIC = "/odin1/odometry"
BASE_FRAME = "odin1_base_link"

# Robot self-filter box in base_link [m], measured for the Odin mount (odin.params.yaml).
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


def cloud_topic(path: str) -> str:
    """The Odin cloud topic this bag carries, raw first."""
    with open(path, "rb") as fh:
        present = {ch.topic for ch in make_reader(fh).get_summary().channels.values()}
    for topic in CLOUD_TOPICS:
        if topic in present:
            return topic
    raise ValueError(f"{path}: none of {CLOUD_TOPICS} recorded")


def static_transform(path: str, child: str, parent: str = BASE_FRAME) -> np.ndarray:
    """parent_T_child [4x4] composed from the bag's /tf_static.

    The driver republishes its calibration on every start, and it has been seen to differ between
    starts (up to ~5 cm, ~3 deg on 2026-09-29), so the LAST value of each edge wins: a bag
    spanning a driver restart gets the later calibration throughout.
    """
    edges: dict[str, tuple[str, np.ndarray]] = {}
    for msg in read_ros2_messages(path, topics=["/tf_static"]):
        for tf in msg.ros_msg.transforms:
            r, t = tf.transform.rotation, tf.transform.translation
            m = np.eye(4)
            m[:3, :3] = quat_to_mat(r.x, r.y, r.z, r.w)
            m[:3, 3] = (t.x, t.y, t.z)
            edges[tf.child_frame_id] = (tf.header.frame_id, m)
    out, frame = np.eye(4), child
    while frame != parent:
        if frame not in edges:
            raise ValueError(
                f"{path}: /tf_static has no chain {parent} <- {child} (stuck at {frame})"
            )
        frame, m = edges[frame]
        out = m @ out
    return out


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

    A cloud not already in `odin1_base_link` is first taken there through the bag's static TF
    (`static_transform`), so the self-filter box and the pose apply in the frame they are defined
    in, and `sensor_xyz` is the lidar's own origin rather than the base's.

    `sensor_roll` / `sensor_pitch` [rad] are an extrinsic correction applied in the base frame
    before the pose. The mount between the robot and the Odin (base_link -> odin1_base_link) is not
    in the Odin's calibration, so a physical mount tilt shows up as a constant BODY-frame attitude
    bias in the map -- which is how it is fitted here.
    """
    sensor_rot = _rpy_to_mat(sensor_roll, sensor_pitch, 0.0)
    base_T: dict[str, np.ndarray] = {BASE_FRAME: np.eye(4)}
    frames: list[Frame] = []
    for msg in read_ros2_messages(path, topics=[cloud_topic(path)]):
        c = msg.ros_msg
        t = c.header.stamp.sec + c.header.stamp.nanosec * 1e-9
        i = int(np.argmin(np.abs(odom.t - t)))
        if abs(odom.t[i] - t) > 0.1:  # no pose within one frame period -> unusable
            continue
        pts = _unpack_cloud(c)
        if pts.size == 0:
            continue
        if c.header.frame_id not in base_T:
            base_T[c.header.frame_id] = static_transform(path, c.header.frame_id)
        m = base_T[c.header.frame_id]
        pts = (pts.astype(np.float64) @ m[:3, :3].T + m[:3, 3]).astype(np.float32)
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
        sensor = rot @ m[:3, 3] + odom.xyz[i]
        frames.append(Frame(t=t, points=world.astype(np.float32), sensor_xyz=sensor))
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
