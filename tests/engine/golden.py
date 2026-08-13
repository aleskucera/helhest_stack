"""Golden bit-identity fixture for every simulator output on a fixed seeded batch.

Run:  python -m tests.engine.golden           # check against the committed fixture
      python -m tests.engine.golden --write   # REGENERATE the fixture (deliberate act)

This is the non-interference proof for engine work: a change that adds new outputs or
opt-in parameters must leave every PRE-EXISTING output bit-identical with default
parameters. The fixture pins one `ForwardSimulator` rollout (CPU and CUDA) and one
`DifferentiableSimulator` forward+backward, B=8, T=16, on a seeded terrain/friction/
control set.

Forward arrays are compared EXACTLY (bit-identical). The two gradient arrays are compared
with a tolerance instead: the envelope-adjoint scatter uses atomics, so summation order --
and therefore the last bits -- varies between runs on the same device.

The fixture is device- and Warp-version-specific (float32 CUDA arithmetic is not portable
across architectures). Regenerate it, and say so in the commit, when the device or the Warp
version changes.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import warp as wp

from helhest.engine import DifferentiableSimulator
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams

FIXTURE = Path(__file__).with_name("golden_fixture.npz")

SEED = 20260807
BATCH = 8
STEPS = 16
CELL = 0.1
CELLS_X, CELLS_Y = 121, 81
ORIGIN = (-2.0, -4.0)
GRAD_ATOL = 1e-8  # atomics reorder the adjoint scatter; forward arrays stay exact


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """Seeded elevation + friction fields: a mild slope under a fixed set of gaussian bumps.

    Bumpy enough that the settle tilts, the belly clearance varies and the dilation's arg-max
    moves around -- i.e. every output array actually carries signal.
    """
    rng = np.random.default_rng(SEED)
    xs = ORIGIN[0] + CELL * np.arange(CELLS_X)
    ys = ORIGIN[1] + CELL * np.arange(CELLS_Y)
    X, Y = np.meshgrid(xs, ys)

    elevation = 0.08 * X  # ~4.6 deg base slope
    for _ in range(24):
        cx = rng.uniform(xs[0], xs[-1])
        cy = rng.uniform(ys[0], ys[-1])
        amp = rng.uniform(-0.12, 0.25)
        sigma = rng.uniform(0.25, 0.8)
        elevation += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2))

    friction = 0.55 + 0.15 * np.sin(1.7 * X) * np.cos(2.3 * Y) + 0.05 * rng.random(X.shape)
    return np.ascontiguousarray(elevation, np.float32), np.ascontiguousarray(friction, np.float32)


def _controls() -> tuple[np.ndarray, np.ndarray]:
    """Seeded per-rollout start poses and wheel-speed commands."""
    rng = np.random.default_rng(SEED + 1)
    start = np.stack(
        [
            rng.uniform(-0.5, 0.5, BATCH),
            rng.uniform(-1.5, 1.5, BATCH),
            rng.uniform(-0.6, 0.6, BATCH),
        ],
        axis=1,
    )
    omega = rng.uniform(0.4, 2.4, (STEPS, BATCH, 3))
    return np.ascontiguousarray(start, np.float32), np.ascontiguousarray(omega, np.float32)


def _forward_outputs(sim: ForwardSimulator | DifferentiableSimulator) -> dict[str, np.ndarray]:
    """Every pre-existing output array of a simulator, as numpy."""
    names = ["envelope", "controlled", "derived", "current_wheel_omega", "loads", "turning"]
    names += ["clearance", "clear_soft", "residual"]
    # `clear_soft` only exists on branches carrying the tie-free belly hinge -- pin whatever the
    # simulator actually exposes, so the fixture stays a complete snapshot after a merge.
    return {n: getattr(sim, n).numpy() for n in names if hasattr(sim, n)}


def _run_forward(device: str) -> dict[str, np.ndarray]:
    """One `ForwardSimulator` rollout of the seeded batch on `device`."""
    elevation, friction = _scene()
    start, omega = _controls()
    sim = ForwardSimulator(
        RobotParams(),
        SolverParams(),
        GridParams(CELLS_X, CELLS_Y, CELL, *ORIGIN),
        BATCH,
        STEPS,
        device=device,
    )
    sim.set_terrain(wp.array(elevation, dtype=wp.float32, device=device))
    sim.friction.assign(friction)
    sim.start_pose.assign(start)
    sim.target_wheel_omega.assign(omega)
    sim.init_current_wheel_omega.zero_()
    sim.rollout_launch()
    return _forward_outputs(sim)


def _run_differentiable(device: str) -> dict[str, np.ndarray]:
    """One taped `DifferentiableSimulator` rollout + backward of the seeded batch."""
    elevation, friction = _scene()
    start, omega = _controls()
    rng = np.random.default_rng(SEED + 2)
    # per-rollout terrain: the shared scene plus a small deterministic per-rollout offset, so the
    # batched path is not just B copies of one problem
    elev_stack = elevation[None] + np.float32(0.02) * rng.standard_normal(
        (BATCH, CELLS_Y, CELLS_X), np.float32
    )
    fric_stack = np.repeat(friction[None], BATCH, axis=0)

    sim = DifferentiableSimulator(
        RobotParams(wheel_width=None),  # sphere: DifferentiableSimulator requires it
        SolverParams(),
        GridParams(CELLS_X, CELLS_Y, CELL, *ORIGIN),
        BATCH,
        STEPS,
        device=device,
    )
    sim.set_terrain(wp.array(np.ascontiguousarray(elev_stack, np.float32), dtype=wp.float32))
    sim.set_friction(wp.array(np.ascontiguousarray(fric_stack, np.float32), dtype=wp.float32))
    sim.start_pose.assign(start)
    sim.target_wheel_omega.assign(omega)
    sim.rollout_taped()
    sim.backward()

    out = _forward_outputs(sim)
    out["grad_elevation"] = sim.elevation.grad.numpy()
    out["grad_friction"] = sim.friction.grad.numpy()
    return out


def collect(device: str = "cuda") -> dict[str, np.ndarray]:
    """All golden arrays, keyed `<path>.<array>`."""
    out = {}
    for key, arrays in (
        ("fwd_cpu", _run_forward("cpu")),
        ("fwd_cuda", _run_forward(device)),
        ("diff_cuda", _run_differentiable(device)),
    ):
        for name, value in arrays.items():
            out[f"{key}.{name}"] = value
    return out


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def check(write: bool = False, device: str = "cuda") -> bool:
    got = collect(device)
    if write:
        np.savez_compressed(FIXTURE, **got)
        print(f"wrote {FIXTURE} ({len(got)} arrays)")
        for name in sorted(got):
            print(f"  {name:34s} {_digest(got[name])}  {got[name].shape}")
        return True

    ref = np.load(FIXTURE)
    ok = True
    for name in sorted(got):
        if name not in ref.files:
            print(f"  {name:34s} MISSING from fixture -- regenerate with --write")
            ok = False
            continue
        a, b = got[name], ref[name]
        is_grad = name.split(".")[-1].startswith("grad_")
        if a.shape != b.shape:
            print(f"  {name:34s} SHAPE {a.shape} != {b.shape}")
            ok = False
            continue
        exact = bool(np.array_equal(a, b))
        # grads are atomic-summed: compare to a tolerance scaled by the fixture's own magnitude
        tol = GRAD_ATOL * max(float(np.abs(b).max()), 1.0) if is_grad else 0.0
        close = exact or (is_grad and bool(np.allclose(a, b, rtol=0.0, atol=tol)))
        worst = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0
        status = "OK" if exact else ("OK (atomics, within tol)" if close else "CHANGED")
        print(f"  {name:34s} {_digest(a)}  max|d|={worst:.3e}  {status}")
        ok = ok and close
    return ok


def selftest_golden() -> None:
    ok = check(write=False)
    print(f"golden bit-identity  {'OK' if ok else 'REVIEW -- outputs changed'}")
    assert ok, "simulator outputs differ from the golden fixture"


if __name__ == "__main__":
    if "--write" in sys.argv:
        wp.init()
        check(write=True)
    else:
        selftest_golden()
