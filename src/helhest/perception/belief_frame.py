"""One frame of the elevation belief, and the layers the planner reads from it.

The single place the robot (`elevation_node`) and the simulator (`drive_sim`) turn a scan into the
map they plan on, so the two cannot drift apart. Needs the `belief` extra (`elevation_belief`),
which is why nothing imports this module unless it plans on the belief.

Per frame, in this order: `recenter` (whole cells, so the grid never leaves the world lattice),
`motion_update` from the second frame on (drift accrues with the time since the last one),
`carve` (the visibility carve that retires what moved), `measure_scan`. Then `layers()`:

    height     the belief's height, NaN where nothing was ever measured, inpainted on the device.
               The fill is referenced to the surrounding GROUND rather than to zero: a zero fill
               reads as flat terrain wherever the robot has not looked, and where the ground sits
               below zero that phantom plateau closes a ring around the routing window.
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
def _sd_kernel(
    var: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    """Measurement variance -> sd. Clamped at 0: a fused variance can land a hair below it."""
    i, j = wp.tid()
    out[i, j] = wp.sqrt(wp.max(var[i, j], 0.0))


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
        wp.copy(self._scratch, lay["raw_h"])
        self.height = multigrid_inpaint(self._scratch)
        shape = self.measured.shape
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
        return self.height, self.measured, self.sd, self.belief.drift()
