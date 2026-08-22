"""Loaders for the Oxford Spires dataset: per-scan Hesai PCDs, GT TUM trajectory, IMU.

Per the FROZEN prereg (clark_paper/PREREG_oxford_spires.md): the pipeline consumes
generic timestamped clouds, so these loaders parse the on-disk formats as-is and do
not apply any extrinsic themselves -- composition with GT anchors (and E_BODY_LIDAR)
is the window runner's job.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The dataset's GT body frame is the lidar frame yawed 180 degrees (sensor mounted
# backwards). Pinned empirically against the TLS survey: a yaw sweep on keble-02
# ground-band points found a symmetric minimum at exactly 180 deg (15.4 cm median vs
# 38.8 cm at 0 deg); see clark_paper/PREREG_oxford_spires.md design-phase log,
# 2026-08-22.
E_BODY_LIDAR = np.diag([-1.0, -1.0, 1.0])

MIN_RANGE_M = 0.5  # drop near-field returns off the sensor housing / handheld rig


@dataclass
class ScanData:
    xyz: np.ndarray  # (N, 3) float32, RAW lidar frame -- no extrinsic applied
    t: np.ndarray  # (N,) float64, absolute per-point unix time
    ring: np.ndarray  # (N,) uint16


def _parse_pcd_header(header: bytes) -> tuple[dict[str, int], int, int]:
    """Generic ASCII PCD header parse: FIELDS/SIZE/COUNT -> per-field byte offset.

    Returns (offsets, stride, n_points). Offsets/stride are derived purely from the
    header so the parser tolerates padding ("_") fields and per-file field counts.
    """
    fields: list[str] = []
    sizes: list[int] = []
    counts: list[int] = []
    n_points = 0
    for line in header.decode("ascii", "replace").splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "FIELDS":
            fields = tok[1:]
        elif tok[0] == "SIZE":
            sizes = [int(x) for x in tok[1:]]
        elif tok[0] == "COUNT":
            counts = [int(x) for x in tok[1:]]
        elif tok[0] == "POINTS":
            n_points = int(tok[1])
        elif tok[0] == "DATA" and tok[1] != "binary":
            raise ValueError(f"unsupported PCD DATA mode: {tok[1]!r}")

    offsets: dict[str, int] = {}
    stride = 0
    for name, size, count in zip(fields, sizes, counts):
        offsets[name] = stride
        stride += size * count
    for required in ("x", "y", "z", "timestamp", "ring"):
        assert required in offsets, f"PCD header missing required field {required!r}"
    return offsets, stride, n_points


class HesaiScanArchive:
    """Per-scan Hesai PCD clouds packed as one zip of <unix_time>.pcd files."""

    def __init__(self, zip_path: str) -> None:
        self._zip = zipfile.ZipFile(zip_path)  # kept open for repeated reads
        names = [n for n in self._zip.namelist() if n.endswith(".pcd")]
        # the filename stem IS the scan start time (matches the first point's own
        # per-point timestamp field, verified against the raw data)
        stems = [Path(n).stem for n in names]
        order = np.argsort([float(s) for s in stems])
        self._names = [names[i] for i in order]
        self.stamps = np.array([float(stems[i]) for i in order], dtype=np.float64)

    @property
    def n_scans(self) -> int:
        return len(self._names)

    def read(self, idx: int) -> ScanData:
        raw = self._zip.read(self._names[idx])
        marker = b"DATA binary\n"
        header_end = raw.find(marker) + len(marker)
        offsets, stride, n_points = _parse_pcd_header(raw[:header_end])
        # the recorded body can carry trailing padding rows beyond n_points (a
        # fixed capture buffer); only the declared points are real
        body = raw[header_end : header_end + n_points * stride]
        rows = np.frombuffer(body, dtype=np.uint8).reshape(n_points, stride)

        def field(name: str, dtype: type) -> np.ndarray:
            o = offsets[name]
            width = np.dtype(dtype).itemsize
            return rows[:, o : o + width].copy().view(dtype).ravel()

        xyz = np.stack(
            [field("x", np.float32), field("y", np.float32), field("z", np.float32)], axis=1
        )
        t = field("timestamp", np.float64)
        ring = field("ring", np.uint16)

        finite = np.isfinite(xyz).all(axis=1) & np.isfinite(t)
        in_range = np.linalg.norm(xyz, axis=1) >= MIN_RANGE_M
        keep = finite & in_range
        return ScanData(xyz=xyz[keep], t=t[keep], ring=ring[keep])


def rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Quaternion SLERP, (x, y, z, w). Flips q1's sign for the double-cover case
    (q and -q represent the same rotation; without the flip, interpolation would
    take the long way around)."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:  # nearly parallel: linear interp avoids a 0/0 in sin(theta)
        out = q0 + u * (q1 - q0)
        return out / np.linalg.norm(out)
    theta0 = np.arccos(dot)
    theta = theta0 * u
    q_perp = q1 - q0 * dot
    q_perp /= np.linalg.norm(q_perp)
    return q0 * np.cos(theta) + q_perp * np.sin(theta)


class GtTrajectory:
    """A TUM (`time tx ty tz qx qy qz qw`) ground-truth trajectory."""

    def __init__(self, tum_path: str) -> None:
        data = np.loadtxt(tum_path)
        self.t = data[:, 0].astype(np.float64)
        self.pos = data[:, 1:4].astype(np.float64)
        self.quat = data[:, 4:8].astype(np.float64)  # xyzw

    def pose_at(self, t: float) -> np.ndarray:
        """world_T_body at time t via linear position interp + quaternion SLERP.

        Clamps to the first/last sample outside the trajectory's time range.
        """
        i = int(np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1))
        t0, t1 = self.t[i - 1], self.t[i]
        u = 0.0 if t1 <= t0 else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
        pos = self.pos[i - 1] + u * (self.pos[i] - self.pos[i - 1])
        quat = _slerp(self.quat[i - 1], self.quat[i], u)
        T = np.eye(4)
        T[:3, :3] = rotmat(quat)
        T[:3, 3] = pos
        return T


class ImuData:
    """IMU samples: `secs,nsecs,acc_x,acc_y,acc_z,ang_vel_x,ang_vel_y,ang_vel_z`."""

    def __init__(self, csv_path: str) -> None:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        self.t = (data[:, 0] + data[:, 1] * 1e-9).astype(np.float64)
        self.acc = data[:, 2:5].astype(np.float64)
        self.gyro = data[:, 5:8].astype(np.float64)

    def gyro_at(self, t: float) -> np.ndarray:
        """Nearest-sample gyro lookup."""
        i = int(np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1))
        lo, hi = i - 1, i
        i = lo if abs(self.t[lo] - t) <= abs(self.t[hi] - t) else hi
        return self.gyro[i]


if __name__ == "__main__":
    seq_dir = Path("/home/kuceral4/data/oxford_spires/sequences/2024-03-12-keble-college-02")
    arch = HesaiScanArchive(str(seq_dir / "raw" / "lidar-clouds.zip"))
    traj = GtTrajectory(str(seq_dir / "trajectory" / "gt-tum.txt"))
    imu = ImuData(str(seq_dir / "raw" / "imu.csv"))

    # the zip has 3008 total entries, but one is the "lidar-clouds/" directory
    # entry itself -- 3007 are actual scan files
    assert arch.n_scans == 3007, arch.n_scans
    print(f"PASS n_scans == {arch.n_scans}")

    assert traj.t[0] <= arch.stamps[0] <= traj.t[-1]
    print("PASS first scan stamp within the trajectory time range")

    # the last scan trails the GT trajectory's last pose by about one scan period
    # (~0.1 s -- the trajectory estimate ends slightly before the final captured
    # spin); pose_at clamps to the last sample for any query past the end, so a
    # small overhang here is a real property of the data, not a bug
    overhang_s = float(arch.stamps[-1] - traj.t[-1])
    assert 0.0 <= overhang_s < 0.2, overhang_s
    print(f"PASS last scan stamp within {overhang_s * 1e3:.0f} ms of the trajectory end (clamped)")

    scan = arch.read(arch.n_scans // 2)
    span_s = float(scan.t.max() - scan.t.min())
    assert len(scan.xyz) > 40_000, len(scan.xyz)
    assert span_s < 0.2, span_s
    assert scan.ring.max() < 64, scan.ring.max()
    print(
        f"PASS mid-sequence scan: {len(scan.xyz)} valid points, "
        f"per-point t span {span_s * 1e3:.1f} ms, max ring {scan.ring.max()}"
    )

    i0 = 100
    T_sample = traj.pose_at(float(traj.t[i0]))
    assert np.allclose(T_sample[:3, 3], traj.pos[i0], atol=1e-9)
    print("PASS pose_at at a sample time reproduces that sample's position to < 1e-9")

    t_mid = 0.5 * (traj.t[i0] + traj.t[i0 + 1])
    T_mid = traj.pose_at(t_mid)
    lo = np.minimum(traj.pos[i0], traj.pos[i0 + 1])
    hi = np.maximum(traj.pos[i0], traj.pos[i0 + 1])
    assert np.all(T_mid[:3, 3] >= lo - 1e-9) and np.all(T_mid[:3, 3] <= hi + 1e-9)
    print("PASS pose_at at a midpoint lies between the bracketing samples")

    dt = np.diff(imu.t)
    med_dt = float(np.median(dt))
    assert np.all(dt > 0)
    assert 0.002 < med_dt < 0.003, med_dt
    print(f"PASS imu.t strictly increasing, median dt {med_dt * 1e3:.3f} ms")
