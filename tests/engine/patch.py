"""The taped CYLINDER path: local envelope patches, against the forward engine and the geometry.

Run:  python -m tests.engine.patch

`DifferentiableSimulator` used to refuse `RobotParams.wheel_width` outright, so every benchmark
scored its gradient baselines on a SPHERE while the estimator and the truth ran on a cylinder. It
now supports the cylinder by materialising a small square envelope PATCH per (rollout, timestep) in
that step's yaw bin, instead of the [B, n_yaw, ny, nx] dilated stack the old error message proposed
(11.8 GB on the deployed node's shape). Four things have to hold for that to be a real replacement:

  parity   -- the patched rollout must reproduce `ForwardSimulator`, whose cylinder path is already
              verified against hand geometry (tests/engine/cylinder.py) and is the oracle here. Not
              bit-identity: the patch grid subtracts its own origin, so `_locate`'s in-cell fraction
              rounds differently in float32 than it does on the full map. The SPHERE pair, which
              shares one grid, is run alongside as the reference for what that path difference costs
              when the envelope is identical.
  yaw bins -- the element is binned over [0, PI) because a cylinder at yaw and at yaw+PI is the same
              shape. Same pose, yaw and yaw+PI: the patch must come out bit-identical.
  coverage -- the patch is sized from the robot (`pad = ceil(reach/cell) + 3`). If a wheel ever
              samples outside it, `_locate` clamps and returns a WRONG height silently, so the bound
              is checked against the reads the kernels make, over the full range of tilts the settle
              can reach.
  memory   -- the working set must not scale with `n_yaw x ny x nx`. That is the whole point.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import DifferentiableSimulator
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.engine.simulator import YAW_BINS

CELL = 0.1
NX, NY = 91, 91
ORIGIN = (-4.5, -4.5)
WHEEL_WIDTH = 0.10  # the ruler-measured tread; `RobotParams`' own default
SEED = 20260826


def _device() -> str:
    return "cuda:0"


def _scene() -> np.ndarray:
    """A seeded bumpy slope: the settle tilts, and the envelope arg-max moves between bins."""
    rng = np.random.default_rng(SEED)
    xs = ORIGIN[0] + CELL * np.arange(NX)
    ys = ORIGIN[1] + CELL * np.arange(NY)
    X, Y = np.meshgrid(xs, ys)
    h = 0.06 * X
    for _ in range(30):
        h += rng.uniform(-0.12, 0.25) * np.exp(
            -((X - rng.uniform(xs[0], xs[-1])) ** 2 + (Y - rng.uniform(ys[0], ys[-1])) ** 2)
            / (2.0 * rng.uniform(0.25, 0.8) ** 2)
        )
    return np.ascontiguousarray(h, np.float32)


def _controls(batch: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Start poses spread over the yaw circle (several bins) and turning commands."""
    yaws = np.linspace(-np.pi, np.pi, batch, endpoint=False)
    start = np.stack(
        [np.full(batch, -1.5), np.linspace(-1.0, 1.0, batch), yaws], axis=1
    ).astype(np.float32)
    rng = np.random.default_rng(SEED + 1)
    omega = rng.uniform(0.6, 2.2, (steps, batch, 3)).astype(np.float32)
    return np.ascontiguousarray(start), np.ascontiguousarray(omega)


def _run_pair(wheel_width: float | None, batch: int, steps: int, device: str):
    """One scene, controls and poses through `ForwardSimulator` and `DifferentiableSimulator`."""
    elevation = _scene()
    start, omega = _controls(batch, steps)
    rp = RobotParams(wheel_width=wheel_width)
    gp = GridParams(NX, NY, CELL, *ORIGIN)
    sp = SolverParams()

    fwd = ForwardSimulator(rp, sp, gp, batch, steps, device=device)
    fwd.set_terrain(wp.array(elevation, dtype=wp.float32, device=device))
    fwd.set_uniform_friction(0.7)
    fwd.start_pose.assign(start)
    fwd.target_wheel_omega.assign(omega)
    fwd.init_current_wheel_omega.zero_()
    fwd.rollout_launch()

    dif = DifferentiableSimulator(rp, sp, gp, batch, steps, device=device)
    with wp.ScopedDevice(device):
        dif.set_terrain(wp.array(np.repeat(elevation[None], batch, 0), dtype=wp.float32))
        dif.set_friction(wp.array(np.full((batch, NY, NX), 0.7, np.float32), dtype=wp.float32))
    dif.start_pose.assign(start)
    dif.target_wheel_omega.assign(omega)
    dif.rollout_taped(loss_fn=None)
    return fwd, dif


def _worst(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def selftest_forward_parity(batch: int = 6, steps: int = 12) -> None:
    """The patched cylinder rollout reproduces `ForwardSimulator`'s cylinder rollout."""
    device = _device()
    out = {}
    for name, width in (("sphere", None), ("cylinder", WHEEL_WIDTH)):
        fwd, dif = _run_pair(width, batch, steps, device)
        d_ctrl = _worst(dif.controlled.numpy(), fwd.controlled.numpy())
        d_der = _worst(dif.derived.numpy(), fwd.derived.numpy())
        d_clear = _worst(dif.clearance.numpy(), fwd.clearance.numpy())
        out[name] = max(d_ctrl, d_der, d_clear)
        print(
            f"  {name:9s} taped vs forward: controlled {d_ctrl:.2e} m  derived {d_der:.2e}  "
            f"clearance {d_clear:.2e}"
        )
        del fwd, dif
    # The sphere pair shares one envelope grid and differs only in fused-vs-per-step launch order;
    # the cylinder pair adds the patch grid's own rounding. Both must stay at float32 noise.
    worst = max(out.values())
    print(f"cylinder forward parity with ForwardSimulator  worst={worst:.2e}  "
          f"{'OK' if worst < 1e-4 else 'REVIEW'}")
    assert worst < 1e-4, f"taped cylinder rollout disagrees with ForwardSimulator by {worst}"


def selftest_yaw_period(batch: int = 8, steps: int = 1) -> None:
    """yaw and yaw+PI are the same cylinder, so they must dilate to the same patch.

    The stack spans [0, PI) for exactly this reason (`step.yaw_bin`). Pairs of rollouts share a
    pose and differ by PI in heading; both the bin index and every cell of the gathered patch must
    come out identical. Run on the ENVELOPE rather than on a settled pose, because the ROBOT is not
    PI-periodic -- its rear wheel swaps ends -- while its wheel element is.
    """
    device = _device()
    elevation = _scene()
    rp = RobotParams(wheel_width=WHEEL_WIDTH)
    gp = GridParams(NX, NY, CELL, *ORIGIN)
    sim = DifferentiableSimulator(rp, SolverParams(), gp, batch, steps, device=device)
    with wp.ScopedDevice(device):
        sim.set_terrain(wp.array(np.repeat(elevation[None], batch, 0), dtype=wp.float32))
    half = batch // 2
    yaws = np.linspace(0.0, np.pi, half, endpoint=False) + 0.13
    pose = np.zeros((batch, 3), np.float32)
    pose[:half, 0] = pose[half:, 0] = np.linspace(-1.0, 1.0, half)
    pose[:half, 1] = pose[half:, 1] = np.linspace(0.6, -0.6, half)
    pose[:half, 2] = yaws
    pose[half:, 2] = yaws + np.pi
    with wp.ScopedDevice(device):
        sim.start_pose.assign(pose)
        sim._contact_patch(0, sim.start_pose)
        sim._gather_patch(0)
    patch = sim.env_patch.numpy()[0]
    bins = sim._patch_bin.numpy()[0]
    org = sim._patch_org.numpy()[0]
    d_bin = int(np.abs(bins[:half] - bins[half:]).max())
    d_org = int(np.abs(org[:half] - org[half:]).max())
    d_patch = _worst(patch[:half], patch[half:])
    print(f"  {half} pose pairs, bins {bins[:half]}  d_bin={d_bin}  d_org={d_org}  "
          f"max|d patch|={d_patch:.2e}")
    ok = d_bin == 0 and d_org == 0 and d_patch == 0.0
    print(f"yaw and yaw+PI dilate identically  {'OK' if ok else 'REVIEW'}")
    assert ok, "the cylinder element is not PI-periodic in this build"


def selftest_patch_covers_reads(cell: float = CELL) -> None:
    """Every envelope read of a step lands strictly inside that step's patch.

    A read that leaves the patch does not fail loudly: `_locate` clamps it to the patch border and
    returns a plausible wrong height. So the bound is checked rather than trusted. The reads are the
    three wheel centres `p + Rot(yaw, pitch, roll) @ wheel_i` -- searched over the whole tilt range
    the settle can reach, since the Newton iterates sample at every intermediate tilt too -- each
    widened by one cell for `sample_normal`'s central difference and by the bilinear stencil's
    second cell, with the pose sitting anywhere inside its own cell.
    """
    rp = RobotParams(wheel_width=WHEEL_WIDTH)
    sp = SolverParams()
    reach = max(rp.half_track, rp.rear_offset)
    pad = int(np.ceil(reach / cell)) + 3
    npatch = 2 * pad + 1

    wheel = np.array([[0, rp.half_track, 0], [0, -rp.half_track, 0], [-rp.rear_offset, 0, 0]])
    # |Rot @ w| = |w| for any rotation, so the horizontal reach is bounded by max|w| regardless of
    # tilt; the sweep below is the check on that argument, not a substitute for it.
    assert np.isclose(np.linalg.norm(wheel, axis=1).max(), reach)

    tilts = np.linspace(-sp.tilt_clamp, sp.tilt_clamp, 9)
    yaws = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    sub = np.linspace(0.0, 1.0, 5)  # the pose's offset inside its own cell
    lo, hi = npatch, -1
    for yaw in yaws:
        for pitch in tilts:
            for roll in tilts:
                cy, sy = np.cos(yaw), np.sin(yaw)
                cp, spi = np.cos(pitch), np.sin(pitch)
                cr, sr = np.cos(roll), np.sin(roll)
                rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                ry = np.array([[cp, 0, spi], [0, 1, 0], [-spi, 0, cp]])
                rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
                w = (rz @ ry @ rx @ wheel.T).T[:, :2]  # wheel-centre offsets from the pose
                for du in sub:
                    for dv in sub:
                        # pose at cell-centre + (du, dv) cells; the patch anchors on floor(), so the
                        # pose's own cell is patch cell `pad`
                        q = (w + np.array([du, dv]) * cell) / cell + pad
                        # sample_normal reads +-1 cell around each centre; _locate then takes the
                        # floor cell and its +1 neighbour
                        lo = min(lo, int(np.floor(q.min() - 1.0 - 0.5)))
                        hi = max(hi, int(np.floor(q.max() + 1.0 - 0.5)) + 1)
    print(f"  cell={cell} pad={pad} patch={npatch}x{npatch}  read cells span [{lo}, {hi}]")
    ok = lo >= 0 and hi <= npatch - 1
    print(f"patch covers every envelope read  {'OK' if ok else 'REVIEW'}")
    assert ok, f"reads span [{lo}, {hi}] outside a {npatch}-cell patch: raise the pad"


def selftest_memory(batch: int = 8, steps: int = 16) -> None:
    """The patch working set is O(B x T x patch): flat in the map size and in the bin count."""
    device = _device()
    rp = RobotParams(wheel_width=WHEEL_WIDTH)
    sp = SolverParams()
    sizes = [(91, 91), (150, 150)]
    got = []
    for nx, ny in sizes:
        sim = DifferentiableSimulator(
            rp, sp, GridParams(nx, ny, CELL, *ORIGIN), batch, steps, device=device
        )
        stack = batch * YAW_BINS * ny * nx * 4  # the [B, n_yaw, ny, nx] envelope stack, not built
        got.append((nx, ny, sim.patch_bytes(), stack))
        print(f"  map {ny:4d}x{nx:<4d} patch stack {sim.patch_bytes()/1e6:7.2f} MB   "
              f"[B, n_yaw, ny, nx] envelope would be {stack/1e6:8.1f} MB")
        del sim
    flat = got[0][2] == got[1][2]
    smaller = all(b < s for _, _, b, s in got)
    print(f"patch memory independent of map size  {'OK' if flat and smaller else 'REVIEW'}")
    assert flat, "patch working set scales with the map -- it must not"
    assert smaller, "patch working set is not smaller than the stack it replaces"


if __name__ == "__main__":
    wp.init()
    selftest_forward_parity()
    selftest_yaw_period()
    selftest_patch_covers_reads()
    selftest_patch_covers_reads(cell=0.05)
    selftest_memory()
