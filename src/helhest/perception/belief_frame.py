"""One frame of the elevation belief, and the layers the planner reads from it.

The single place the robot (`elevation_node`) and the simulator (`drive_sim`) turn a scan into the
map they plan on, so the two cannot drift apart. Needs the `belief` extra (`elevation_belief`),
which is why nothing imports this module unless it plans on the belief.

Per frame, in this order: `recenter` (whole cells, so the grid never leaves the world lattice),
`motion_update` from the second frame on (drift accrues with the time since the last one),
`carve` (the visibility carve that retires what moved), `measure_scan`. Then `layers()`:

    height     the belief's height, inpainted on the device wherever nothing was ever measured.
               The fill is referenced to the surrounding GROUND rather than to zero: a zero fill
               reads as flat terrain wherever the robot has not looked, and where the ground sits
               below zero that phantom plateau closes a ring around the routing window. The
               belief stores 0, not NaN, in a cell it never measured (only `valid` says so), so
               the unmeasured cells are set to NaN here first -- without that the inpaint has
               nothing to fill and the plateau is back. drive_sim ran without it until 2026-09-26
               and never noticed: its worlds' ground sits at 0.
    measured   1 where the belief holds a measurement, else 0 -- the mask the planner takes
    sd         the MEASUREMENT sd, sqrt(meas_var): what the settle's attitude differences see,
               since pose drift is common-mode and cancels (planning/settle_producer.py)
    drift      the pose-drift variance, var_h - var_meas, negative where nothing was measured

All buffers are preallocated and device-resident; nothing here touches the host per frame.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from elevation_belief import DriftRates
from elevation_belief import ElevationBelief
from elevation_belief import NoiseModel

from .heightmap.postprocess import multigrid_inpaint


def odin_noise() -> NoiseModel:
    """Odin's dToF: sd = 0.012 + 0.004 * range [m]. studies/calib/fit_drift.py put the measured
    near-field var_meas at 1.75 cm; the range term is the dToF's own growth."""
    return NoiseModel("linear", a=0.012, b=0.004)


@wp.kernel
def _measured_kernel(
    valid: wp.array2d(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    """The belief's validity flag as the float mask the planner takes."""
    i, j = wp.tid()
    out[i, j] = wp.where(valid[i, j] != 0, 1.0, 0.0)


@wp.kernel
def _unknown_nan_kernel(
    h: wp.array2d(dtype=wp.float32),
    valid: wp.array2d(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    """The belief's height with NaN where it never measured: the inpaint's unknown set."""
    i, j = wp.tid()
    out[i, j] = wp.where(valid[i, j] != 0, h[i, j], wp.nan)


@wp.kernel
def _sd_kernel(
    var: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    """Measurement variance -> sd. Clamped at 0: a fused variance can land a hair below it."""
    i, j = wp.tid()
    out[i, j] = wp.sqrt(wp.max(var[i, j], 0.0))


@wp.kernel
def _crop_kernel(
    src: wp.array2d(dtype=wp.float32),
    r0: wp.int32,
    c0: wp.int32,
    out: wp.array2d(dtype=wp.float32),
):
    """A sub-window of a layer, on the belief's own lattice."""
    i, j = wp.tid()
    out[i, j] = src[i + r0, j + c0]


@wp.kernel
def _pool_kernel(
    height: wp.array2d(dtype=wp.float32),
    measured: wp.array2d(dtype=wp.float32),
    sd: wp.array2d(dtype=wp.float32),
    drift: wp.array2d(dtype=wp.float32),
    r0: wp.int32,
    c0: wp.int32,
    k: wp.int32,
    out_h: wp.array2d(dtype=wp.float32),
    out_m: wp.array2d(dtype=wp.float32),
    out_sd: wp.array2d(dtype=wp.float32),
    out_drift: wp.array2d(dtype=wp.float32),
):
    """k x k blocks of the window at (r0, c0) -> one coarser cell each.

    Height is the max over MEASURED cells only: the inpainted fill must not outvote real ground,
    or one unobserved fine cell would speak for the whole block. A block with nothing measured
    falls back to the max of the inpainted surface, so blind ground reads as terrain rather than
    as a plateau at the map origin. A block counts as measured if any of its cells is. sd is the
    largest over the measured cells (over all of them when none is), the conservative reading of
    the block; drift is the largest, and its negative sentinel for unmeasured cells means a
    measured cell always wins.
    """
    i, j = wp.tid()
    h_meas = float(-1.0e30)
    h_all = float(-1.0e30)
    s_meas = float(0.0)
    s_all = float(0.0)
    d_max = float(-1.0e30)
    any_m = float(0.0)
    for a in range(k):
        for b in range(k):
            r = r0 + i * k + a
            c = c0 + j * k + b
            h = height[r, c]
            s = sd[r, c]
            h_all = wp.max(h_all, h)
            s_all = wp.max(s_all, s)
            d_max = wp.max(d_max, drift[r, c])
            if measured[r, c] > 0.5:
                any_m = 1.0
                h_meas = wp.max(h_meas, h)
                s_meas = wp.max(s_meas, s)
    out_m[i, j] = any_m
    out_h[i, j] = wp.where(any_m > 0.5, h_meas, h_all)
    out_sd[i, j] = wp.where(any_m > 0.5, s_meas, s_all)
    out_drift[i, j] = d_max


class BeliefFrame:
    """An `ElevationBelief` window updated one scan at a time, with its planner layers."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        cell: float,
        carve_range: float = 6.0,  # [m] 0 disables the visibility carve
        noise: NoiseModel | None = None,
        # on-device SLAM, 100x below the dead-reckoning default
        rates: DriftRates | None = None,
        device: wp.Device | str | None = None,
    ) -> None:
        self.device = wp.get_device(device)
        self.belief = ElevationBelief(
            bounds,
            cell,
            noise=noise if noise is not None else odin_noise(),
            rates=rates if rates is not None else DriftRates.odin_slam(),
            device=self.device,
        )
        self.carve_range = float(carve_range)
        n = (self.belief.ny, self.belief.nx)
        # multigrid_inpaint fills IN PLACE, and the array it would fill is the belief's own height
        self._scratch = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.measured = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.sd = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.height = self._scratch
        self.drift = self.belief.drift()
        self._started = False

    @property
    def xmin(self) -> float:
        return self.belief.xmin

    @property
    def ymin(self) -> float:
        return self.belief.ymin

    def update(
        self,
        points_world: wp.array,
        sensor_origin: np.ndarray,
        robot_xy: tuple[float, float],
        dt: float,
    ) -> None:
        """Fold one world-frame scan in. `dt` [s] is the time since the previous update."""
        self.belief.recenter(robot_xy)
        if self._started:
            self.belief.motion_update(dt, robot_xy)
        if self.carve_range > 0.0:
            self.belief.carve(points_world, sensor_origin, max_range=self.carve_range)
        self.belief.measure_scan(points_world, sensor_origin)
        self._started = True

    def layers(self) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        """(height, measured, sd, drift) on the belief window. Owned buffers, overwritten by the
        next call; `height` comes back from the inpaint (see its docstring)."""
        lay = self.belief.layers()
        shape = self.measured.shape
        wp.launch(
            _unknown_nan_kernel,
            dim=shape,
            inputs=[lay["raw_h"], lay["valid"]],
            outputs=[self._scratch],
            device=self.device,
        )
        self.height = multigrid_inpaint(self._scratch)
        wp.launch(
            _measured_kernel,
            dim=shape,
            inputs=[lay["valid"]],
            outputs=[self.measured],
            device=self.device,
        )
        wp.launch(
            _sd_kernel, dim=shape, inputs=[lay["meas_var"]], outputs=[self.sd], device=self.device
        )
        self.drift = self.belief.drift()
        return self.height, self.measured, self.sd, self.drift

    def crop(self, layer: wp.array, r0: int, c0: int, out: wp.array) -> wp.array:
        """`out`-sized sub-window of a belief-window layer, starting at cell (r0, c0)."""
        wp.launch(
            _crop_kernel, dim=out.shape, inputs=[layer, r0, c0], outputs=[out], device=self.device
        )
        return out

    def pool(
        self,
        r0: int,
        c0: int,
        k: int,
        out_h: wp.array,
        out_m: wp.array,
        out_sd: wp.array,
        out_drift: wp.array,
    ) -> None:
        """The last `layers()` pooled k x k from cell (r0, c0) into the four `out_*` (see
        `_pool_kernel` for the rule). For a planner that routes coarser than the map."""
        wp.launch(
            _pool_kernel,
            dim=out_h.shape,
            inputs=[self.height, self.measured, self.sd, self.drift, r0, c0, int(k)],
            outputs=[out_h, out_m, out_sd, out_drift],
            device=self.device,
        )
