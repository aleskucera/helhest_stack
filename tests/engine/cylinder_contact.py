"""The cylinder contact point in `normal_loads`, against a numpy oracle.

Run:  python -m tests.engine.cylinder_contact

`normal_loads` puts each wheel's contact at the wheel's support point in direction -n, n being
the terrain normal. A sphere's is one radius down the normal. A cylinder of half-tread w touches
at its RIM:

    n_perp = n - (n . a) a,    ct = c - R n_perp/|n_perp| - w sgn(n . a) a

with a the spin axis (body +y). w is derived from RobotParams.wheel_width (the same knob that
picks the wheel-envelope shape) rather than a separate field -- w = wheel_width/2, and w = 0
(wheel_width=None, the sphere envelope) must reproduce the sphere contact exactly.

Four checks, on exact planes where the engine's bilinear central difference reproduces the slope
exactly, so the oracle can compute the normal analytically and stays independent of the sampler:

  1. wheel_width=None reproduces the sphere contact and the old loads (also covered bit-exactly
     by the golden fixture's diff_cuda path, which pins wheel_width=None).
  2. no LATERAL tilt -> the axial term vanishes and the cylinder must equal the sphere. This is
     what catches a wrong axis or a wrong sign, because both are invisible until the ground
     tilts sideways. It also pins sgn(0) = 0: the contact is then a LINE across the tread whose
     load resultant is centred, and wp.sign would put it on a rim instead.
  3. lateral tilt -> the contact sits on the UPHILL rim, exactly w off the mid-plane, because
     that is the side where the ground comes up to meet the wheel first.
  4. the loads the kernel returns match the oracle's own 3x3 solve -- INCLUDING the tangential
     (friction) reaction the current `normal_loads` folds into the torque balance (the model
     this file's contact-point fix was ported onto is not the simple normal-only balance the
     original commit shipped against; the oracle below is that model's numpy twin, not a
     transcription of the older one).
"""

from __future__ import annotations

import numpy as np
import warp as wp
from helhest.engine import Grid
from helhest.engine import GridParams
from helhest.engine import Robot
from helhest.engine import RobotParams
from helhest.engine.rotations import euler_zyx
from helhest.engine.step import normal_loads


@wp.kernel
def loads_probe(
    envelope: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    pose: wp.array(dtype=wp.vec3),  # (x, y, yaw)
    zpr: wp.array(dtype=wp.vec3),  # (z, pitch, roll), held fixed so only the contact varies
    out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    R = euler_zyx(pose[tid][2], zpr[tid][1], zpr[tid][2])
    q = wp.vec3(pose[tid][0], pose[tid][1], zpr[tid][0])
    out[tid] = normal_loads(envelope, grid, robot, R, q)


def rot_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def tilted_plane(ny: int, nx: int, cell: float, slope_x: float, slope_y: float) -> np.ndarray:
    iy, ix = np.mgrid[0:ny, 0:nx]
    return (ix * cell * slope_x + iy * cell * slope_y).astype(np.float32)


def oracle(
    slope_x: float,
    slope_y: float,
    rp: RobotParams,
    pose: tuple[float, float, float],
    zpr: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """The same quasi-static solve `normal_loads` does, from the definition -- including the
    tangential (friction) reaction's contribution to the torque balance (fda57bf). Returns
    (loads [3], contacts [3,3])."""
    x, y, yaw = pose
    z, pitch, roll = zpr
    R = rot_zyx(yaw, pitch, roll)
    p = np.array([x, y, z])
    com = p + R @ np.asarray(rp.com)
    weight = rp.mass * rp.gravity
    b, l = rp.half_track, rp.rear_offset
    wheel_pos = np.array([[0.0, b, 0.0], [0.0, -b, 0.0], [-l, 0.0, 0.0]])
    # the plane is exact, so its normal is constant and needs no sampling
    n = np.array([-slope_x, -slope_y, 1.0])
    n /= np.linalg.norm(n)
    axis = R @ np.array([0.0, 1.0, 0.0])
    half_width = 0.0 if rp.wheel_width is None else rp.wheel_width / 2.0

    contacts = np.zeros((3, 3))
    normals = np.zeros((3, 3))
    arms = np.zeros((3, 3))
    for i in range(3):
        c = p + R @ wheel_pos[i]
        if half_width > 0.0:
            n_ax = float(n @ axis)
            n_perp = n - n_ax * axis
            ct = (
                c
                - rp.wheel_radius * (n_perp / np.linalg.norm(n_perp))
                - half_width * np.sign(n_ax) * axis
            )
        else:
            ct = c - rp.wheel_radius * n
        contacts[i] = ct
        normals[i] = n
        arms[i] = ct - com

    n_bar = normals.sum(0)
    n_bar /= np.linalg.norm(n_bar)
    load_sum = weight * n_bar[2]
    tangential = np.array([0.0, 0.0, weight]) - load_sum * n_bar
    inv_sum = 1.0 / max(load_sum, 1.0e-3 * weight)

    A = np.zeros((3, 3))
    for i in range(3):
        m = np.cross(arms[i], normals[i]) + inv_sum * np.cross(arms[i], tangential)
        A[0, i] = 1.0
        A[1, i] = m[0]
        A[2, i] = m[1]
    loads = np.linalg.solve(A, np.array([load_sum, 0.0, 0.0]))
    return loads, contacts


def device_loads(
    env: np.ndarray, grid: Grid, rp: RobotParams, pose: tuple, zpr: tuple, device: str
) -> np.ndarray:
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        loads_probe,
        dim=1,
        inputs=[
            wp.array(env, dtype=wp.float32, device=device),
            grid,
            rp.build(device),
            wp.array(np.array([pose], np.float32), dtype=wp.vec3, device=device),
            wp.array(np.array([zpr], np.float32), dtype=wp.vec3, device=device),
        ],
        outputs=[out],
        device=device,
    )
    return np.asarray(out.numpy()[0], np.float64)


def main() -> None:
    wp.init()
    device = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
    cell, ny, nx = 0.05, 260, 260
    x0 = y0 = -6.5
    grid = GridParams(nx, ny, cell, x0, y0).build()
    pose = (0.0, 0.0, 0.0)  # heading +x, so the spin axis is world +y
    zpr = (0.40, 0.0, 0.0)  # z/pitch/roll pinned: this isolates the contact from the settle
    wheel_width = 0.10  # [m] the ruler-measured tread (half-width 0.05, the merged default)

    cases = [
        ("flat", 0.0, 0.0),
        ("longitudinal 11 deg", 0.20, 0.0),
        ("lateral 11 deg", 0.0, 0.20),
        ("lateral -11 deg", 0.0, -0.20),
        ("both", 0.15, 0.20),
    ]
    print("               case   err sph   err cyl   axial: sph    cyl   |  dN cyl-sph   res bal")
    ok = True
    for name, sx, sy in cases:
        env = tilted_plane(ny, nx, cell, sx, sy)
        got = {}
        for ww in (None, wheel_width):
            rp = RobotParams(wheel_width=ww)
            dev = device_loads(env, grid, rp, pose, zpr, device)
            ref, contacts = oracle(sx, sy, rp, pose, zpr)
            got[ww] = (dev, ref, contacts)

        half_width = wheel_width / 2.0
        err_sph = np.abs(got[None][0] - got[None][1]).max() / got[None][1].max()
        err_cyl = (
            np.abs(got[wheel_width][0] - got[wheel_width][1]).max() / got[wheel_width][1].max()
        )
        d_loads = np.abs(got[wheel_width][0] - got[None][0]).max()
        # how far off the wheel's mid-plane each contact sits (the axis is world +y at yaw 0)
        centres = np.array([[0.0, 0.365, 0.0], [0.0, -0.365, 0.0], [-0.75, 0.0, 0.0]])
        centres[:, 2] += zpr[0]
        ax_sph = (got[None][2] - centres)[:, 1]
        ax_cyl = (got[wheel_width][2] - centres)[:, 1]
        # the balance the solve is supposed to enforce -- Sum N_i = m g (n . z), not m g outright
        # (fda57bf: the row is resolved ALONG THE SURFACE NORMAL, not vertically) -- re-checked on
        # the loads returned
        n = np.array([-sx, -sy, 1.0])
        n /= np.linalg.norm(n)
        rp0 = RobotParams()
        resid = abs(got[wheel_width][0].sum() - rp0.mass * rp0.gravity * n[2])

        print(
            f"{name:>19}{err_sph:>10.1e}{err_cyl:>10.1e}"
            f"{ax_sph[0] * 100:>+11.2f}{ax_cyl[0] * 100:>+7.2f} cm{d_loads:>12.2f} N{resid:>10.1e}"
        )
        ok &= err_sph < 1e-5 and err_cyl < 1e-5 and resid < 1e-3

        # the cylinder's axial offset is the rim, +-w exactly, and 0 only for a LINE contact
        want_ax = 0.0 if sy == 0.0 else -half_width * np.sign(-sy)
        if not np.allclose(ax_cyl, want_ax, atol=1e-9):
            print(f"    FAIL: cylinder axial offset {ax_cyl} expected {want_ax:+.3f} m")
            ok = False
        # and it must be INSIDE the tread, which is the whole point: the sphere's is not
        if np.abs(ax_cyl).max() > half_width + 1e-9:
            print(f"    FAIL: contact {np.abs(ax_cyl).max():.3f} m outside the tread")
            ok = False
        if sy == 0.0 and d_loads > 1e-4:
            print(f"    FAIL: no lateral tilt, yet cylinder != sphere by {d_loads:.3e} N")
            ok = False
        if sy != 0.0 and d_loads < 1e-3:
            print(f"    FAIL: lateral tilt, yet the contact shift changed nothing ({d_loads:.1e})")
            ok = False

    verdict = "PASS" if ok else "FAIL"
    print(f"\n  {verdict} -- kernel matches the oracle; the cylinder rides its own rim")
    assert ok, "cylinder contact point disagrees with the oracle"


if __name__ == "__main__":
    main()
