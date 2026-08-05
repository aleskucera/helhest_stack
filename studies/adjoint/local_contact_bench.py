"""Localized vs full contact refresh: correctness first, then speed.

    .venv/bin/python -m studies.adjoint.local_contact_bench

Decides whether a second-order attribution pass fits inside a control tick. The curvature
diagonal needs J(+delta) and J(-delta) per cell; per-rollout terrain turns N cells into 2N
slices of ONE launch, and two optimisations remove the wasted work inside that launch:

  +local   refresh the arg-max only inside each perturbation's wheel-radius window, because a
           single perturbed cell cannot change the contact of anything further than R away
  +cheap   restore by writing back one height per slice instead of copying the whole stack
           (which also stopped re-copying friction, which a curvature probe never touches)

Both must be EXACT rather than approximate, so the run checks bit-identical costs and zero
arg-max discrepancies against a full recompute and prints them next to the timings. A faster
wrong answer would be worthless.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import warp as wp

from . import sigma as sigma_mod
from .harness import Harness
from .local_contact import LocalContact
from .scene import build_scene
from .scene import LANE_Y
from .scene import rollouts
from .study_b import _cost
from .study_b import _cost_gradient

SIZES = (32, 128, 256)  # probe cells; the batch is 2N slices
ROLLOUT = 4  # curb head-on


@wp.kernel
def perturb_pairs(
    arr: wp.array3d(dtype=wp.float32),
    cell_y: wp.array(dtype=wp.int32),
    cell_x: wp.array(dtype=wp.int32),
    delta: wp.array(dtype=wp.float32),
):
    """Slice 2k raises cell k by delta_k, slice 2k+1 lowers it. The whole probe in one launch."""
    j = wp.tid()
    pair = j / 2
    sign = 1.0
    if j - 2 * pair == 1:
        sign = -1.0
    arr[j, cell_y[pair], cell_x[pair]] = arr[j, cell_y[pair], cell_x[pair]] + sign * delta[pair]


class Probe:
    """One batched curvature probe, in three variants that must all agree exactly."""

    def __init__(self, scene, pose, omega, cells, sigma_np):
        self.n = len(cells)
        self.batch = 2 * self.n
        poses = np.tile(pose, (self.batch, 1)).astype(np.float32)
        omegas = np.tile(omega[:, None, :], (1, self.batch, 1)).astype(np.float32)
        self.h = Harness(scene, poses, omegas)
        self.lc = LocalContact(self.h.sim, cells)
        with wp.ScopedDevice(self.h.device):
            self._cy = wp.array(np.ascontiguousarray(cells[:, 0], np.int32), dtype=wp.int32)
            self._cx = wp.array(np.ascontiguousarray(cells[:, 1], np.int32), dtype=wp.int32)
            self._d = wp.array(
                np.ascontiguousarray(sigma_np[cells[:, 0], cells[:, 1]], np.float32),
                dtype=wp.float32,
            )

    def _perturb(self) -> None:
        wp.launch(
            perturb_pairs,
            self.batch,
            inputs=[self.h.sim.elevation, self._cy, self._cx, self._d],
            device=self.h.device,
        )

    def _run(self) -> np.ndarray:
        wp.copy(self.h.sim.current_wheel_omega[0], self.h.sim.init_current_wheel_omega)
        self.h.terms.zero_()
        self.h._launches()
        return _cost(self.h.terms.numpy())

    def full(self) -> np.ndarray:
        """Baseline: full terrain reset, full arg-max recompute."""
        self.h._reset_terrain(dilate=True)
        self._perturb()
        self.h.sim._contact()
        return self._run()

    def local(self) -> np.ndarray:
        """Full reset, windowed arg-max."""
        self.h._reset_terrain(dilate=True)
        self._perturb()
        self.lc.refresh()
        return self._run()

    def cheap(self) -> np.ndarray:
        """Per-cell restore, windowed arg-max."""
        self.lc.restore()
        self._perturb()
        self.lc.refresh()
        return self._run()

    def prime(self) -> None:
        """Clean-terrain setup the localized variants need once."""
        self.h._reset_terrain(dilate=True)
        self.lc.freeze()
        self.lc.capture_heights()

    def close(self) -> None:
        del self.lc, self.h, self._cy, self._cx, self._d


def _time(fn, n: int = 5) -> float:
    fn()
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    wp.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main() -> None:
    wp.init()
    scene = build_scene()
    poses, omega, labels = rollouts()
    sigma_np = sigma_mod.sigma_field(scene, inpainted=sigma_mod.decoy_mask(scene, LANE_Y[0] + 1.4))
    ny, nx = scene.shape

    ref = Probe(scene, poses[ROLLOUT], omega[:, ROLLOUT, :], np.zeros((1, 2), int), sigma_np)
    grads, _ = ref.h.adjoint(dilate=True, leaf="elevation")
    strength = np.abs(_cost_gradient(grads)[0])
    idx = np.argwhere(strength > 0.02 * strength.max())
    ranked = idx[np.argsort(-strength[idx[:, 0], idx[:, 1]])]
    ref.close()
    gc.collect()

    print(f"grid {ny}x{nx} = {ny * nx} cells, rollout '{labels[ROLLOUT]}'")
    print("the real 0.1 m perception grid is ~81x158, about 2.3x smaller\n")
    print(
        f"{'N':>5}{'B':>6}{'full ms':>10}{'+local':>9}{'+cheap':>9}{'speedup':>9}"
        f"{'ms/cell':>9}{'cost':>8}{'argmax':>8}"
    )
    for n in SIZES:
        p = Probe(scene, poses[ROLLOUT], omega[:, ROLLOUT, :], ranked[:n], sigma_np)
        c_full = p.full().copy()
        p.prime()
        n_bad = p.lc.verify()
        c_cheap = p.cheap().copy()
        exact = (
            "exact" if np.array_equal(c_full, c_cheap) else f"{np.abs(c_full - c_cheap).max():.1e}"
        )
        p.prime()
        t_full, t_local, t_cheap = _time(p.full), _time(p.local), _time(p.cheap)
        print(
            f"{n:>5}{p.batch:>6}{t_full:>10.1f}{t_local:>9.1f}{t_cheap:>9.1f}"
            f"{t_full / t_cheap:>8.1f}x{t_cheap / n:>9.3f}{exact:>8}{n_bad:>8}"
        )
        p.close()
        gc.collect()
        wp.synchronize()


if __name__ == "__main__":
    main()
