"""Matched-element trajectory generation + Monte-Carlo truth for the Clark/SSTA benchmarks.

Stage A (`element.py`) gave every benchmark a `--element cylinder` candidate-footprint table for
the ESTIMATOR (Clark/SSTA's per-node mean/variance). It did not touch where the benchmarks get
their controlled trajectory or their Monte-Carlo ground truth: both always ran through
`studies.adjoint.harness.Harness`, which wraps `DifferentiableSimulator` -- and that simulator is
sphere-only (`engine/simulator.py` raises on `wheel_width is not None`; the taped path would need
a `[B, n_yaw, ny, nx]` envelope stack and a yaw index through the custom-grad settle, out of
scope). So a `--element cylinder` run priced the cylinder estimator against a robot that was
still, physically, contacting the ground as a sphere -- trajectory and truth disagreed with the
estimator about which wheel geometry produced them.

This module runs the SAME rollout through `ForwardSimulator` instead, with the real cylinder
envelope (`RobotParams.wheel_width`), for exactly the two places that need no gradient: the
controlled trajectory, and every Monte-Carlo draw. Callers keep their existing sphere path
(`Harness`/`DifferentiableSimulator`) untouched and branch to this module only under
`element == "cylinder"` -- sphere runs are bit-for-bit what they always were.

`ForwardSimulator.elevation` is ONE shared 2D grid per instance (unlike `DifferentiableSimulator`,
which is `[B, ny, nx]` per-rollout terrain -- that per-rollout capability is exactly what
Monte-Carlo calibration needs and `ForwardSimulator` was never built for; building it would be an
engine change, not a study-side wiring one). `cylinder_mc_truth_terms` works around that the same
way the estimator's own MC truth (`clark.py: _mc_env_stats`) does NOT need to -- by looping one
real terrain draw at a time through a single persistent `ForwardSimulator`, batched over every
plan sharing that draw (so it's `n_draws` small rollouts, not one `n_draws`-wide batch). Measured
at this module's scene scale (90x90 cells, B=16 plans/draw): ~1.7 ms/draw on a laptop GPU, so a
`--seeds 10` reduced-scale run (Step 3) completes in low single-digit seconds per seed.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams

from ..adjoint.harness import _terms_kernel
from ..adjoint.harness import N_TERMS
from ..adjoint.harness import STUDY_CLEAR_MARGIN
from ..adjoint.sigma import NoiseDraws
from .element import WHEEL_HALF_WIDTH

CYLINDER_WHEEL_WIDTH = 2.0 * WHEEL_HALF_WIDTH  # 0.10 m ruler-measured tread, element.py's radius


def cylinder_robot_params(clear_margin: float = STUDY_CLEAR_MARGIN) -> RobotParams:
    """RobotParams for the matched-element physics path: the same half-tread `element.py`'s
    candidate table uses, so the estimator and the ground truth share one geometry."""
    return RobotParams(wheel_width=CYLINDER_WHEEL_WIDTH, clear_margin=clear_margin)


def _build_forward_sim(
    scene,
    batch_size: int,
    n_steps: int,
    device: str,
    clear_margin: float,
    dt: float,
    newton_iters: int,
) -> ForwardSimulator:
    ny, nx = scene.shape
    grid = GridParams(nx, ny, scene.cell, scene.origin_x, scene.origin_y)
    solver = SolverParams(dt=dt, newton_iters=newton_iters, atol=0.0)
    sim = ForwardSimulator(
        cylinder_robot_params(clear_margin), solver, grid, batch_size, n_steps, device=device
    )
    sim.friction.assign(np.ascontiguousarray(scene.friction, np.float32))
    sim.init_current_wheel_omega.zero_()
    return sim


def cylinder_controlled_trajectory(
    scene,
    poses: np.ndarray,
    omega: np.ndarray,
    device: str,
    clear_margin: float = STUDY_CLEAR_MARGIN,
    dt: float = 0.1,
    newton_iters: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """(controlled [T+1,B,3] (x,y,yaw), derived [T+1,B,3] (z,pitch,roll)) belief-map rollout
    under the CYLINDER contact -- the matched-element replacement for
    `Harness(...).sim.controlled.numpy()` / `.sim.derived.numpy()` when `--element cylinder`."""
    sim = _build_forward_sim(
        scene, poses.shape[0], omega.shape[0], device, clear_margin, dt, newton_iters
    )
    sim.set_terrain(wp.array(np.ascontiguousarray(scene.elevation, np.float32), device=device))
    sim.start_pose.assign(np.ascontiguousarray(poses, np.float32))
    sim.target_wheel_omega.assign(np.ascontiguousarray(omega, np.float32))
    sim.rollout_launch()
    return sim.controlled.numpy(), sim.derived.numpy()


def cylinder_mc_truth_terms_from_fields(
    scene,
    fields: np.ndarray | wp.array,
    poses: np.ndarray,
    omega: np.ndarray,
    device: str,
    clear_margin: float = STUDY_CLEAR_MARGIN,
    dt: float = 0.1,
    newton_iters: int = 20,
) -> np.ndarray:
    """[N_TERMS, n_draws, B] Monte-Carlo settle terms under the CYLINDER contact, for a caller-
    supplied `[n_draws, ny, nx]` terrain field stack (already-perturbed elevation -- whatever
    generator the caller used) -- the matched-element replacement for
    `Harness(...).forward(dilate=True)` batched over `n_draws` when `--element cylinder`. The
    SAME `poses`/`omega` (B plans) are rolled out on every draw, one draw's terrain at a time
    (`ForwardSimulator.elevation` is a single shared 2D grid, see module docstring)."""
    n_draws = fields.shape[0]
    batch = poses.shape[0]
    sim = _build_forward_sim(scene, batch, omega.shape[0], device, clear_margin, dt, newton_iters)
    sim.start_pose.assign(np.ascontiguousarray(poses, np.float32))
    sim.target_wheel_omega.assign(np.ascontiguousarray(omega, np.float32))
    if not isinstance(fields, wp.array):
        with wp.ScopedDevice(device):
            fields = wp.array(np.ascontiguousarray(fields, np.float32))

    terms = np.empty((N_TERMS, n_draws, batch), np.float32)
    with wp.ScopedDevice(device):
        terms_dev = wp.zeros((N_TERMS, batch), dtype=wp.float32)
    for d in range(n_draws):
        sim.set_terrain(fields[d])  # device-to-device slice, no host round trip
        sim.rollout_launch()
        terms_dev.zero_()
        wp.launch(
            _terms_kernel,
            batch,
            inputs=[sim.controlled, sim.derived, sim.clearance, sim.clear_soft, sim.n_steps],
            outputs=[terms_dev],
            device=device,
        )
        terms[:, d, :] = terms_dev.numpy()
    return terms


def cylinder_mc_truth_terms(
    scene,
    belief: np.ndarray,
    sigma: np.ndarray,
    poses: np.ndarray,
    omega: np.ndarray,
    device: str,
    seed: int,
    n_draws: int,
    corr_len: float,
    cell: float,
    clear_margin: float = STUDY_CLEAR_MARGIN,
    dt: float = 0.1,
    newton_iters: int = 20,
) -> np.ndarray:
    """`cylinder_mc_truth_terms_from_fields`, generating the `n_draws` terrain realizations here
    via `NoiseDraws` (the same GPU generator `Harness`-based MC truth uses). `seed` matches the
    sphere path's `NoiseDraws.perturb` convention (same draws, different contact physics)."""
    ny, nx = scene.shape
    with wp.ScopedDevice(device):
        base = wp.array(np.ascontiguousarray(np.tile(belief, (n_draws, 1, 1)), np.float32))
        sigma_dev = wp.array(np.ascontiguousarray(sigma, np.float32))
        fields = wp.zeros((n_draws, ny, nx), dtype=wp.float32)
    draws = NoiseDraws((n_draws, ny, nx), cell, corr_len, device)
    draws.perturb(base, sigma_dev, 1.0, fields, seed)
    return cylinder_mc_truth_terms_from_fields(
        scene, fields, poses, omega, device, clear_margin, dt, newton_iters
    )
