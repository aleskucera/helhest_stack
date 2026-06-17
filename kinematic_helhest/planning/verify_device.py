"""Phase-1 verification: device terrain ingest + on-device wheel envelope.

Three checks (CPU by default; pass --device cuda for the GPU path):
  1. half-cell alignment of `bounds_to_origin` + `terrain_from_device` — a
     cell-center raster sampled at known world points matches the analytic plane,
     and the naive `x0=xmin` origin shows the expected half-cell bias;
  2. on-device `engine.envelope.wheel_envelope` == numpy `heightmap.wheel_envelope`
     oracle on the real scenes;
  3. a device-fed `Simulator` reproduces the host `Simulator.for_scene` path
     rollout-for-rollout on the same node grid.

Run:  python -m kinematic_helhest.planning.verify_device [--device cuda]
"""
import argparse

import numpy as np
import warp as wp

from .. import friction
from .. import heightmap as hmmod
from ..engine import bounds_to_origin
from ..engine import GridParams
from ..engine import RobotParams
from ..engine import Simulator
from ..engine import SolverParams
from ..engine import terrain_from_device
from ..engine.envelope import wheel_envelope
from ..engine.terrain import _probe
from .mppi import _to_omega


def _sample(terrain, xs, ys, device):
    """Bilinear height at world (xs, ys) via the engine's device sampler."""
    wx = wp.array(xs.astype(np.float32), dtype=wp.float32, device=device)
    wy = wp.array(ys.astype(np.float32), dtype=wp.float32, device=device)
    oh = wp.zeros(len(xs), dtype=wp.float32, device=device)
    on = wp.zeros(len(xs), dtype=wp.vec3, device=device)
    wp.launch(_probe, len(xs), inputs=[terrain.elevation, terrain.g, wx, wy],
              outputs=[oh, on], device=device)
    return oh.numpy()


def check_alignment(device):
    """Cell-center raster of a tilted plane f = a*x + b*y; bilinear is exact for a
    plane, so the correct origin recovers f and the xmin origin is off by ~half a cell."""
    a, b, res = 0.3, 0.2, 0.05
    bounds = (-1.0, 1.0, -1.0, 1.0)  # (xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = bounds
    nx = int(round((xmax - xmin) / res))
    ny = int(round((ymax - ymin) / res))
    xc = xmin + (np.arange(nx) + 0.5) * res
    yc = ymin + (np.arange(ny) + 0.5) * res
    XX, YY = np.meshgrid(xc, yc)  # [ny, nx], values at cell centers
    H = (a * XX + b * YY).astype(np.float32)
    H_wp = wp.array(np.ascontiguousarray(H), dtype=wp.float32, device=device)

    x0, y0 = bounds_to_origin(bounds, res)
    good = terrain_from_device(H_wp, x0, y0, res)
    bad = terrain_from_device(H_wp, xmin, ymin, res)  # naive: node origin = corner

    xs = np.array([-0.3, 0.1, 0.42], np.float64)
    ys = np.array([0.2, -0.15, 0.05], np.float64)
    f = a * xs + b * ys
    err_good = np.abs(_sample(good, xs, ys, device) - f).max()
    err_bad = np.abs(_sample(bad, xs, ys, device) - f).max()
    bias = (abs(a) + abs(b)) * 0.5 * res
    print(f"  alignment: correct-origin err={err_good:.2e}  "
          f"xmin-origin err={err_bad:.2e}  (expected half-cell bias ~{bias:.3f})")
    assert err_good < 1e-5, err_good
    assert err_bad > 0.4 * bias, err_bad
    print("  alignment OK")


def check_envelope(device, R=0.35):
    worst = 0.0
    for name, scene in [("flat", hmmod.flat()), ("box", hmmod.box_scene()),
                        ("ramp", hmmod.ramp_scene())]:
        ref = hmmod.wheel_envelope(scene, R).H
        H_wp = wp.array(np.ascontiguousarray(scene.H, np.float32),
                        dtype=wp.float32, device=device)
        got = wheel_envelope(H_wp, scene.cell, R, device).numpy()
        d = float(np.abs(got - ref).max())
        worst = max(worst, d)
        print(f"  envelope[{name}] max|dHenv|={d:.2e}")
    assert worst < 1e-4, worst
    print(f"  envelope parity OK (worst={worst:.2e})")


def check_end_to_end(device, B=16, T=25):
    """Same node grid both ways: device path must match the host path exactly."""
    scene = hmmod.box_scene()
    mu = friction.uniform(0.8)  # default extent matches box_scene
    params = SolverParams(dt=0.05, k_turn=2.0, newton_iters=12)
    start = (-1.0, 0.0, 0.0)
    omega = _to_omega(np.full((B, T, 2), 2.0, np.float32))

    host = Simulator.for_scene(RobotParams(), params, scene, mu, B, T, device=device)
    ph, _, ch, rh = host.rollout(omega, start)

    H_wp = wp.array(np.ascontiguousarray(scene.H, np.float32),
                    dtype=wp.float32, device=device)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0,
                    R=RobotParams().wheel_radius)
    dev = Simulator(RobotParams(), params, grid, B, T, device)
    dev.set_terrain(H_wp)
    dev.set_uniform_friction(0.8)
    pd, _, cd, rd = dev.rollout(omega, start)

    dp = float(np.abs(ph - pd).max())
    dc = float(np.abs(ch - cd).max())
    dr = float(np.abs(rh - rd).max())
    print(f"  end-to-end device-vs-host  dplanar={dp:.2e} dclear={dc:.2e} dresid={dr:.2e}")
    assert max(dp, dc, dr) < 1e-4, (dp, dc, dr)
    print("  end-to-end OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu", help="warp device: cpu or cuda")
    args = ap.parse_args()
    wp.init()
    print(f"[1/3] alignment ({args.device})");    check_alignment(args.device)
    print(f"[2/3] envelope parity ({args.device})"); check_envelope(args.device)
    print(f"[3/3] end-to-end ({args.device})");    check_end_to_end(args.device)
    print("Phase-1 device path: ALL OK")


if __name__ == "__main__":
    main()
