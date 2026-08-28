"""Rover geometry from the dataset's own TF tree. No hand-typed numbers except frame names.

Ported unchanged from the spike (`~/data/baseprod_audit/spike/geometry.py`) except that the
traverse directory now comes from `paths.traverse_dir` instead of a two-entry dict, so the
same code serves design and held-out traverses.

Two traps in the delivery are handled here and must not be undone:
  1. TF.csv's base_link -> link_bogie_MRB rotation is the dataset's known-bad rear-bogie
     branch; TF_CORRECTED_REAR_BOGIE.csv is used instead (composing the raw one puts the two
     rear wheels 0.35-0.62 m off in z).
  2. link_mast -> imu_link is a 180 deg rotation about Z; taking the IMU's raw roll/pitch as
     the body's inverts both signs and DOUBLES the tilt (register.rover_poses handles it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import traverse_dir

WHEELS = ["LF", "LM", "LR", "RF", "RM", "RR"]
TF_COLS = ["Timestamp", "Frame_ID", "Child_Frame_ID",
           "TX", "TY", "TZ", "QX", "QY", "QZ", "QW"]

BOGIE = {"LF": "link_bogie_LFB", "LM": "link_bogie_LFB", "RF": "link_bogie_RFB",
         "RM": "link_bogie_RFB", "LR": "link_bogie_MRB", "RR": "link_bogie_MRB"}


def quat_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def T(row):
    M = np.eye(4)
    q = np.array([row.QX, row.QY, row.QZ, row.QW], float)
    M[:3, :3] = quat_R(q / np.linalg.norm(q))
    M[:3, 3] = [row.TX, row.TY, row.TZ]
    return M


_TF_CACHE: dict[str, dict] = {}


def load_tf(tdir, dyn="TF_CORRECTED_REAR_BOGIE.csv"):
    """dict (parent, child) -> 4x4: TF_STATIC + the per-link MEDIAN of the dynamic TF."""
    tdir = str(tdir)
    ck = f"{tdir}|{dyn}"
    if ck in _TF_CACHE:
        return _TF_CACHE[ck]
    st = pd.read_csv(f"{tdir}/TF_STATIC.csv")
    with open(f"{tdir}/{dyn}") as fh:
        hdr = 0 if fh.readline().startswith("Timestamp") else None
    dy = pd.read_csv(f"{tdir}/{dyn}", header=hdr, names=None if hdr == 0 else TF_COLS)
    out = {}
    for _, r in st.iterrows():
        out[(r.Frame_ID, r.Child_Frame_ID)] = T(r)
    g = dy.groupby(["Frame_ID", "Child_Frame_ID"])[
        ["TX", "TY", "TZ", "QX", "QY", "QZ", "QW"]].median()
    for (p, c), r in g.iterrows():
        out[(p, c)] = T(r)
    _TF_CACHE[ck] = out
    return out


def tf_of(name: str):
    return load_tf(traverse_dir(name))


def chain(tf, path):
    M = np.eye(4)
    for p, c in zip(path[:-1], path[1:]):
        M = M @ tf[(p, c)]
    return M


def wheel_xyz(tf):
    """base_link -> each DRV (wheel-axle) frame origin. [6, 3] in body coords."""
    out = {}
    for w in WHEELS:
        p = ["base_link", BOGIE[w], f"link_DEP_{w}", f"link_STR_{w}", f"link_DRV_{w}"]
        out[w] = chain(tf, p)[:3, 3]
    return out


def gnss_xyz(tf):
    return chain(tf, ["base_link", "link_mast", "gnss_link"])[:3, 3]


def depth_cam(tf):
    """base_link -> camera_depth_optical_frame, as delivered (nominal 20 deg mount)."""
    return chain(tf, ["base_link", "link_mast", "camera_bottom_screw_frame",
                      "camera_link", "camera_depth_frame", "camera_depth_optical_frame"])
