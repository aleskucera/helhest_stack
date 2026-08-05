"""Study A0 -- forward parity. Nothing downstream is interpretable without it.

Three checks, in order of how much they would invalidate:

  1. harness replica == `DifferentiableSimulator.rollout_taped`. The replica exists only to
     swap the dilation for the identity; if it drifts from the production recording, every
     envelope-leaf result below is measuring the wrong forward. Must be BIT-identical.
  2. per-rollout-terrain path == shared-terrain fused `rollout_kernel`. Confirms the [B, ny, nx]
     path under test agrees with the planner's hot path on the same terrain.
  3. Warp (float32) == numpy reference (float64). Model parity. This is the ONLY place the
     numpy oracle is used -- it is a cross-check on the model, not the FD oracle.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.heightmap import Heightmap
from helhest.reference.rollout import rollout_terrain

from .harness import Harness
from .scene import DT


def run(harness: Harness, verbose: bool = True) -> dict[str, float]:
    scene = harness.scene
    out: dict[str, float] = {}

    # --- 1. replica vs production rollout_taped ------------------------------------------
    harness._reset_terrain(dilate=True)
    harness._rollout(dilate=True)
    rep_c = harness.sim.controlled.numpy().copy()
    rep_d = harness.sim.derived.numpy().copy()

    harness.sim.set_terrain(harness._raw0)
    wp.copy(harness.sim.friction, harness._fric0)
    harness.sim.rollout_taped(loss_fn=None)
    prod_c = harness.sim.controlled.numpy()
    prod_d = harness.sim.derived.numpy()
    out["replica_vs_rollout_taped"] = float(
        max(np.abs(rep_c - prod_c).max(), np.abs(rep_d - prod_d).max())
    )

    # --- 2. per-rollout terrain vs shared-terrain fused kernel ----------------------------
    ny, nx = scene.shape
    grid = GridParams(nx, ny, scene.cell, scene.origin_x, scene.origin_y)
    fwd = ForwardSimulator(
        RobotParams(),
        SolverParams(dt=DT, newton_iters=harness.newton_iters, atol=0.0),
        grid,
        harness.batch_size,
        harness.n_steps,
        harness.device,
    )
    with wp.ScopedDevice(harness.device):
        fwd.set_terrain(wp.array(np.ascontiguousarray(scene.elevation, np.float32)))
        fwd.friction.assign(np.ascontiguousarray(scene.friction, np.float32))
    fwd.target_wheel_omega.assign(harness.sim.target_wheel_omega.numpy())
    fwd.start_pose.assign(harness.sim.start_pose.numpy())
    fwd.init_current_wheel_omega.zero_()
    fwd.rollout_launch()
    out["batched_vs_fused"] = float(
        max(
            np.abs(fwd.controlled.numpy() - prod_c).max(),
            np.abs(fwd.derived.numpy() - prod_d).max(),
        )
    )

    # --- 3. Warp vs the numpy reference ---------------------------------------------------
    hm = Heightmap(scene.elevation, (scene.origin_x, scene.origin_y), scene.cell)
    mu = Heightmap(scene.friction, (scene.origin_x, scene.origin_y), scene.cell)
    omega = harness.sim.target_wheel_omega.numpy()  # [T, B, 3]
    poses = harness.sim.start_pose.numpy()
    dxy, dtilt = 0.0, 0.0
    for b in range(harness.batch_size):
        ref = rollout_terrain(omega[:, b, :], DT, hm, init_pose=tuple(poses[b]), mu_field=mu, k=2.0)
        dxy = max(dxy, float(np.abs(ref["pose2"] - prod_c[1:, b, :]).max()))
        ref_tilt = np.stack([ref["pitch"], ref["roll"]], axis=1)
        dtilt = max(dtilt, float(np.abs(ref_tilt - prod_d[1:, b, 1:]).max()))
    out["warp_vs_numpy_pose_m"] = dxy
    out["warp_vs_numpy_tilt_rad"] = dtilt

    if verbose:
        print(
            f"  A0.1 replica == rollout_taped         max|d| = {out['replica_vs_rollout_taped']:.3e} (want 0)"
        )
        print(f"  A0.2 batched == fused rollout_kernel  max|d| = {out['batched_vs_fused']:.3e}")
        print(
            f"  A0.3 warp vs numpy reference          pose   = {out['warp_vs_numpy_pose_m']:.3e} m"
        )
        print(
            f"       (float32 vs float64)             tilt   = {out['warp_vs_numpy_tilt_rad']:.3e} rad"
        )
    return out
