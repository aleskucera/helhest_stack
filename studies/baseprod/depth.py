"""D435i depth deprojection and the registered-truth sampler.

Registered truth: ground_true(x, y) = DSM(x, y) + f(x, y) - r_wheel, with f the correction
fitted in register.py (f = r_wheel - e, e the DSM's vertical error) and r_wheel = 0.075 m
(Gerdes et al., arXiv:2411.04700 Fig. 10 caption: "a wheel diameter of 15 cm").

The intrinsics are the frozen CameraInfo values in constants.py -- never the factory nominal
the spike started from, and never refitted.
"""
from __future__ import annotations

import glob
import math
import os

import numpy as np
from PIL import Image

from .constants import (CX, CY, DEPTH_SCALE, FX, FY, IMG_H, IMG_W, NOMINAL_PITCH_DEG,
                        R_MAX, R_MIN, R_WHEEL, STRIDE)
from .geometry import load_tf, quat_R
from .paths import traverse_dir
from .register import body_R, npz_path

Z_MIN, Z_MAX = 0.35, 6.0   # the wider band the calibration fits use (belief uses R_MIN/R_MAX)


def frame_paths(name: str) -> list[str]:
    d = traverse_dir(name)
    ps = sorted(glob.glob(f"{d}/RS_DEPTH_16bit/*.png"))
    if not ps:                       # archives that extract one level deeper
        ps = sorted(glob.glob(f"{d}/*/RS_DEPTH_16bit/*.png"))
    return ps


def frame_time(p: str) -> int:
    return int(os.path.basename(p).split("_")[0])


class Poses:
    """Rover pose at an arbitrary time, from the registration run's saved RTK track."""

    def __init__(self, name: str):
        z = np.load(npz_path(name))
        self.t = z["t"].astype(np.float64)
        self.base = z["base"]
        self.yaw, self.pitch, self.roll = z["yaw"], z["pitch"], z["roll"]
        self.beta, self.order = z["beta"], int(z["order"])
        self.x, self.y = z["x"], z["y"]

    def at(self, ts: np.ndarray):
        ts = np.asarray(ts, np.float64)
        pos = np.stack([np.interp(ts, self.t, self.base[:, i]) for i in range(3)], axis=1)
        yaw = np.interp(ts, self.t, self.yaw)
        pit = np.interp(ts, self.t, self.pitch)
        rol = np.interp(ts, self.t, self.roll)
        return pos, body_R(yaw, pit, rol)

    def correction(self, x, y):
        """f(x, y). design() normalises by the FIT SET's mean and range, reproduced here."""
        xs = (np.asarray(x) - self.x.mean()) / max(np.ptp(self.x), 1e-9)
        ys = (np.asarray(y) - self.y.mean()) / max(np.ptp(self.y), 1e-9)
        cols = [np.ones_like(xs)]
        if self.order >= 1:
            cols += [xs, ys]
        if self.order >= 2:
            cols += [xs * xs, xs * ys, ys * ys]
        return np.stack(cols, 1) @ self.beta


_UV = None


def uv_grid():
    global _UV
    if _UV is None:
        v, u = np.mgrid[0:IMG_H:STRIDE, 0:IMG_W:STRIDE]
        _UV = (u.ravel().astype(np.float32), v.ravel().astype(np.float32))
    return _UV


def cam_transform(name: str, dpitch_deg: float) -> np.ndarray:
    """base_link -> camera_depth_optical_frame with the mount pitch set to 20 deg + dpitch.

    The rotation is applied at the real `link_mast` pivot, 0.44 m from the optical centre, so
    the 8 mm/deg camera translation is carried too -- not just the ray directions.
    """
    tf = load_tf(traverse_dir(name))
    th = math.radians(NOMINAL_PITCH_DEG + dpitch_deg)
    q = np.array([0.0, math.sin(th / 2), 0.0, math.cos(th / 2)])
    M = np.eye(4)
    M[:3, :3] = quat_R(q)
    M[:3, 3] = tf[("link_mast", "camera_bottom_screw_frame")][:3, 3]
    T = tf[("base_link", "link_mast")] @ M
    for a, b in (("camera_bottom_screw_frame", "camera_link"),
                 ("camera_link", "camera_depth_frame"),
                 ("camera_depth_frame", "camera_depth_optical_frame")):
        T = T @ tf[(a, b)]
    return T


def deproject_frame(path: str, lo: float = Z_MIN, hi: float = Z_MAX):
    """One depth PNG -> (optical-frame points [M, 3], depth [M]) at the frozen intrinsics."""
    d = np.asarray(Image.open(path), np.uint16)[::STRIDE, ::STRIDE].ravel() * DEPTH_SCALE
    u, v = uv_grid()
    ok = (d > lo) & (d < hi)
    if not ok.any():
        return None
    dd, uu, vv = d[ok], u[ok], v[ok]
    return np.stack([(uu - CX) * dd / FX, (vv - CY) * dd / FY, dd], 1), dd


def sample_frames(name: str, n: int) -> list[str]:
    ps = frame_paths(name)
    idx = np.linspace(0, len(ps) - 1, min(n, len(ps))).astype(int)
    return [ps[i] for i in idx]


def residuals(paths, poses: Poses, T_cam, cor, fill):
    """z(point) - registered truth at the point's own (x, y), with slant range."""
    ts = np.array([frame_time(p) for p in paths], np.float64)
    pos, R = poses.at(ts)
    dz_all, rng_all = [], []
    for i, p in enumerate(paths):
        out = deproject_frame(p)
        if out is None:
            continue
        po, dd = out
        pw = (po @ T_cam[:3, :3].T + T_cam[:3, 3]) @ R[i].T + pos[i]
        dsm, _ = cor.disc_mean(pw[:, 0], pw[:, 1], 0.02, mask=fill)
        e = pw[:, 2] - (dsm + poses.correction(pw[:, 0], pw[:, 1]) - R_WHEEL)
        m = np.isfinite(e)
        dz_all.append(e[m])
        rng_all.append(dd[m])
    return np.concatenate(dz_all), np.concatenate(rng_all)


__all__ = ["Poses", "cam_transform", "deproject_frame", "frame_paths", "frame_time",
           "residuals", "sample_frames", "uv_grid", "R_MIN", "R_MAX"]
