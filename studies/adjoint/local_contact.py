"""Localized arg-max refresh for single-cell perturbations.

Measuring the batched curvature probe showed the arg-max contact pass is 73-82% of its cost,
and almost all of that work is wasted: perturbing ONE cell q can only change the contact of
output cells within one wheel radius of q, because envelope[p] maxes over h[p + d] for
|d| <= R. So p is affected iff |p - q| <= R -- a (2R+1)^2 window, about 225 cells against the
29161 the full pass rescans, roughly a 130x redundancy on the dominant term.

Usage, for a probe stack where slice b perturbs cell `b // 2`:

    broadcast(pristine_best_k, sim._best_k)   # once: every slice starts from the clean arg-max
    <perturb one cell per slice>
    local_contact(...)                        # refresh only each slice's window

Outside its window a slice's `best_k` is untouched and therefore still the clean arg-max,
which is correct precisely because the perturbation cannot reach there. `verify` checks that
against a full `_contact()` on the perturbed terrain -- the claim is exactness, not
approximation, so the test is equality.

Tie-breaking must match the production kernels or the results diverge on ties: both take the
FIRST strictly-maximal offset, so this uses `if lift > best` with k ascending, exactly as
`_contact_kernel` and `make_tiled_contact`'s `_sel_k` do.

STUDY-SIDE prototype. It exists to decide whether second-order attribution fits inside a
control tick; promoting it means integrating with the tiled contact in `envelope.py`.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def restore_cells(
    arr: wp.array3d(dtype=wp.float32),
    cell_y: wp.array(dtype=wp.int32),
    cell_x: wp.array(dtype=wp.int32),
    h0: wp.array(dtype=wp.float32),
):
    """Undo the perturbation by writing back the stored height, one cell per slice.

    Resetting the whole [B, ny, nx] stack instead is a full copy when exactly one cell per
    slice differs -- and it also re-copies friction, which a curvature probe never touches.
    Writing the pristine value back (rather than subtracting delta) keeps it exact in float32.
    """
    j = wp.tid()
    pair = j / 2
    arr[j, cell_y[pair], cell_x[pair]] = h0[pair]


@wp.kernel
def gather_cells(
    arr: wp.array3d(dtype=wp.float32),
    cell_y: wp.array(dtype=wp.int32),
    cell_x: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    """Read the probed cells' heights straight into a device array -- no host round trip."""
    k = wp.tid()
    out[k] = arr[0, cell_y[k], cell_x[k]]


@wp.kernel
def broadcast_best_k(src: wp.array2d(dtype=wp.float32), dst: wp.array3d(dtype=wp.float32)):
    """Seed every slice with the unperturbed arg-max."""
    b, iy, ix = wp.tid()
    dst[b, iy, ix] = src[iy, ix]


def make_local_contact(env_radius: int):
    """Build a windowed arg-max kernel specialised to the wheel's cell radius."""
    R = env_radius

    @wp.kernel
    def local_contact(
        elevation: wp.array3d(dtype=wp.float32),
        cell_y: wp.array(dtype=wp.int32),  # [N] perturbed cell per PAIR of slices
        cell_x: wp.array(dtype=wp.int32),
        off_dy: wp.array(dtype=wp.int32),
        off_dx: wp.array(dtype=wp.int32),
        off_cap: wp.array(dtype=wp.float32),
        best_k: wp.array3d(dtype=wp.float32),
    ):
        """dim = (B, 2R+1, 2R+1): one thread per output cell inside one slice's window."""
        b, wy, wx = wp.tid()
        ny = elevation.shape[1]
        nx = elevation.shape[2]
        pair = b / 2  # slices 2k and 2k+1 both perturb cell k
        iy = cell_y[pair] + wy - R
        ix = cell_x[pair] + wx - R
        if iy >= 0 and iy < ny and ix >= 0 and ix < nx:
            best_lift = float(-1.0e9)  # noqa: UP018
            best = int(0)  # noqa: UP018 -- Warp needs the cast to declare a mutable local
            for k in range(off_dy.shape[0]):
                qy = wp.clamp(iy + off_dy[k], 0, ny - 1)
                qx = wp.clamp(ix + off_dx[k], 0, nx - 1)
                lift = elevation[b, qy, qx] + off_cap[k]
                if lift > best_lift:  # strict: first maximal k, as the production kernels do
                    best_lift = lift
                    best = k
            best_k[b, iy, ix] = float(best)

    return local_contact


class LocalContact:
    """Windowed contact refresh bound to one simulator's buffers."""

    def __init__(self, sim, cells: np.ndarray):
        """`cells` is [N, 2] of (iy, ix); the probe stack must have B = 2N slices."""
        self.sim = sim
        self.radius = sim.env_radius
        self.width = 2 * sim.env_radius + 1
        self._kernel = make_local_contact(sim.env_radius)
        with wp.ScopedDevice(sim.device):
            self._cy = wp.array(np.ascontiguousarray(cells[:, 0], np.int32), dtype=wp.int32)
            self._cx = wp.array(np.ascontiguousarray(cells[:, 1], np.int32), dtype=wp.int32)
            self._pristine = wp.zeros(sim.elevation.shape[1:], dtype=wp.float32)
            self._h0 = wp.zeros(len(cells), dtype=wp.float32)
        self._cells = cells

    def freeze(self) -> None:
        """Capture the clean arg-max and seed every slice with it. Call once, on clean terrain."""
        self.sim._contact()
        wp.copy(self._pristine, self.sim._best_k[0])
        wp.launch(
            broadcast_best_k,
            dim=self.sim._best_k.shape,
            inputs=[self._pristine, self.sim._best_k],
            device=self.sim.device,
        )

    def capture_heights(self) -> None:
        """Record the clean height of each probed cell, for `restore`. Call on clean terrain."""
        wp.launch(
            gather_cells,
            len(self._cells),
            inputs=[self.sim.elevation, self._cy, self._cx, self._h0],
            device=self.sim.device,
        )

    def restore(self) -> None:
        """Undo the perturbation in place -- no full-stack copy, no friction traffic."""
        wp.launch(
            restore_cells,
            self.sim.elevation.shape[0],
            inputs=[self.sim.elevation, self._cy, self._cx, self._h0],
            device=self.sim.device,
        )

    def refresh(self) -> None:
        """Recompute the arg-max only inside each slice's window."""
        wp.launch(
            self._kernel,
            dim=(self.sim._best_k.shape[0], self.width, self.width),
            inputs=[
                self.sim.elevation,
                self._cy,
                self._cx,
                self.sim._off_dy,
                self.sim._off_dx,
                self.sim._off_cap,
                self.sim._best_k,
            ],
            device=self.sim.device,
        )

    def verify(self) -> int:
        """Number of cells where the windowed result differs from a full recompute.

        Must be 0 -- the windowed update is meant to be exact, not approximate.
        """
        mine = self.sim._best_k.numpy().copy()
        self.sim._contact()
        full = self.sim._best_k.numpy()
        n_bad = int((mine != full).sum())
        self.sim._best_k.assign(mine)  # leave the buffer as the caller had it
        return n_bad
