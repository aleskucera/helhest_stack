"""Sub-cell contact refinement for the wheel-envelope dilation.

Study A measured the frozen-arg-max gradient as exact only within `R - sqrt(R^2 - cell^2)`
= 3.6 mm, and a follow-up showed the cause is QUANTIZATION: restricting the arg-max to cell
centres makes the contact teleport between cells, so a ramp in `d(env)/dh` becomes a step.
Study B then showed that radius is what gates the whole method -- at realistic sigma only 3
of 846 probes stayed inside it.

This refines the contact to sub-cell position and nothing else:

    1. discrete arg-max (the existing `_contact`, unchanged) gives offset k*, which localises
       the true maximiser to within about one cell
    2. a local search over a +-1 cell window at 1/SUB cell steps maximises the BILINEAR lifted
       surface L(d) = bilinear(h, p + d) + cap(|d|)
    3. envelope(p) = bilinear(h, p + d*) + cap(|d*| * cell)

    A separable parabolic fit was tried first and is WRONG here: at a curb the lifted surface
    is kinked, so a parabola through three samples invents a vertex that does not exist. It
    improved the slope lane 15x and made curb and rock 4-5 mm WORSE -- i.e. it failed exactly
    where refinement is supposed to help. A direct search has no smoothness assumption.

Why this is the RIGHT fix rather than smoothing the max. For a true maximiser the envelope
theorem says the derivative equals the partial derivative holding the maximiser FIXED -- so
freezing a *correct* maximiser is exact, and today's error comes entirely from freezing a
maximiser that is not the true one. Refinement therefore makes the frozen-arg-max gradient
asymptotically exact, and it makes the FORWARD more accurate at the same time (the discrete
max systematically under-estimates the wheel rest height). A softmax envelope would do the
opposite: bias the forward, and to smooth over a sigma-sized perturbation it would have to
blur terrain at the sigma scale, i.e. stop seeing the curb.

The gather is written as plain arithmetic on array reads, so Warp's own autodiff produces the
bilinear scatter -- no custom grad needed, unlike the settle.

STUDY-SIDE. Promoting this into `envelope.py`'s tiled path is a change to the graph-captured
hot kernel and should happen only on the evidence this module is here to produce.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine.envelope import wheel_offset_table


@wp.func
def _lift(
    elevation: wp.array3d(dtype=wp.float32),
    b: int,
    iy: int,
    ix: int,
    dy: int,
    dx: int,
    cell_size: float,
    wheel_radius: float,
) -> wp.float32:
    """The dilation's lifted value h[p+d] + cap(|d|), edge-clamped. -1e9 outside the disk."""
    ny = elevation.shape[1]
    nx = elevation.shape[2]
    dist = wp.sqrt(float(dy * dy + dx * dx)) * cell_size
    if dist > wheel_radius:
        return -1.0e9
    qy = wp.clamp(iy + dy, 0, ny - 1)
    qx = wp.clamp(ix + dx, 0, nx - 1)
    return elevation[b, qy, qx] + wp.sqrt(wheel_radius * wheel_radius - dist * dist) - wheel_radius


@wp.func
def _lift_bilinear(
    elevation: wp.array3d(dtype=wp.float32),
    b: int,
    iy: int,
    ix: int,
    fy: float,
    fx: float,
    cell_size: float,
    wheel_radius: float,
) -> wp.float32:
    """L(d) at a FRACTIONAL offset: bilinear terrain plus the spherical cap. -1e9 outside."""
    ny = elevation.shape[1]
    nx = elevation.shape[2]
    dist = wp.sqrt(fy * fy + fx * fx) * cell_size
    if dist > wheel_radius:
        return -1.0e9
    qy = float(iy) + fy
    qx = float(ix) + fx
    y0 = wp.clamp(int(wp.floor(qy)), 0, ny - 2)
    x0 = wp.clamp(int(wp.floor(qx)), 0, nx - 2)
    ty = wp.clamp(qy - float(y0), 0.0, 1.0)
    tx = wp.clamp(qx - float(x0), 0.0, 1.0)
    h = (
        (1.0 - ty) * (1.0 - tx) * elevation[b, y0, x0]
        + (1.0 - ty) * tx * elevation[b, y0, x0 + 1]
        + ty * (1.0 - tx) * elevation[b, y0 + 1, x0]
        + ty * tx * elevation[b, y0 + 1, x0 + 1]
    )
    return h + wp.sqrt(wheel_radius * wheel_radius - dist * dist) - wheel_radius


def make_refine(sub: int, window: int = 1):
    """Build a refinement kernel specialised to a sub-cell step of 1/`sub`.

    The whole +-`window` cell search reads terrain only from a (2*window+2)^2 patch, so the
    patch is loaded ONCE into a mat44 and every candidate interpolates from registers. The
    naive version re-reads 4 global values per candidate -- 1156 reads per output cell against
    16 -- and measured 480 ms for 8 slices, far too slow to sit inside a Monte-Carlo loop.
    """
    N = 2 * sub * window + 1

    @wp.kernel
    def refine_contact(
        elevation: wp.array3d(dtype=wp.float32),
        best_k: wp.array3d(dtype=wp.float32),
        off_dy: wp.array(dtype=wp.int32),
        off_dx: wp.array(dtype=wp.int32),
        cell_size: float,
        wheel_radius: float,
        out_dy: wp.array3d(dtype=wp.float32),
        out_dx: wp.array3d(dtype=wp.float32),
        out_cap: wp.array3d(dtype=wp.float32),
    ):
        """Off-tape: local fine search turning the discrete arg-max into a continuous contact."""
        b, iy, ix = wp.tid()
        ny = elevation.shape[1]
        nx = elevation.shape[2]
        k = int(best_k[b, iy, ix])
        dy0 = off_dy[k]
        dx0 = off_dx[k]

        # 4x4 terrain patch covering every bilinear stencil the search can touch.
        base_y = iy + dy0 - 1
        base_x = ix + dx0 - 1
        patch = wp.mat44()
        for r in range(4):
            for c in range(4):
                patch[r, c] = elevation[
                    b, wp.clamp(base_y + r, 0, ny - 1), wp.clamp(base_x + c, 0, nx - 1)
                ]

        best = float(-1.0e9)  # noqa: UP018
        best_fy = float(dy0)
        best_fx = float(dx0)
        for a in range(N):
            for c in range(N):
                fy = float(dy0) + float(a - sub * window) / float(sub)
                fx = float(dx0) + float(c - sub * window) / float(sub)
                dist = wp.sqrt(fy * fy + fx * fx) * cell_size
                if dist <= wheel_radius:
                    # stencil corner in patch-local coordinates
                    ly = int(wp.floor(fy)) - dy0 + 1
                    lx = int(wp.floor(fx)) - dx0 + 1
                    ty = fy - wp.floor(fy)
                    tx = fx - wp.floor(fx)
                    h = (
                        (1.0 - ty) * (1.0 - tx) * patch[ly, lx]
                        + (1.0 - ty) * tx * patch[ly, lx + 1]
                        + ty * (1.0 - tx) * patch[ly + 1, lx]
                        + ty * tx * patch[ly + 1, lx + 1]
                    )
                    v = h + wp.sqrt(wheel_radius * wheel_radius - dist * dist) - wheel_radius
                    if v > best:
                        best = v
                        best_fy = fy
                        best_fx = fx
        dist = wp.min(wp.sqrt(best_fy * best_fy + best_fx * best_fx) * cell_size, wheel_radius)
        out_dy[b, iy, ix] = best_fy
        out_dx[b, iy, ix] = best_fx
        out_cap[b, iy, ix] = wp.sqrt(wheel_radius * wheel_radius - dist * dist) - wheel_radius

    return refine_contact


@wp.kernel
def gather_subcell(
    elevation: wp.array3d(dtype=wp.float32),
    off_dy: wp.array3d(dtype=wp.float32),
    off_dx: wp.array3d(dtype=wp.float32),
    off_cap: wp.array3d(dtype=wp.float32),
    envelope: wp.array3d(dtype=wp.float32),
):
    """On-tape: envelope = bilinear(elevation, refined contact) + cap.

    Plain arithmetic on array reads, so Warp's autodiff yields the four-node bilinear scatter
    -- the analytical gradient, and now a CONTINUOUS function of the contact position.
    """
    b, iy, ix = wp.tid()
    ny = elevation.shape[1]
    nx = elevation.shape[2]
    qy = float(iy) + off_dy[b, iy, ix]
    qx = float(ix) + off_dx[b, iy, ix]
    y0 = wp.clamp(int(wp.floor(qy)), 0, ny - 2)
    x0 = wp.clamp(int(wp.floor(qx)), 0, nx - 2)
    ty = wp.clamp(qy - float(y0), 0.0, 1.0)
    tx = wp.clamp(qx - float(x0), 0.0, 1.0)
    envelope[b, iy, ix] = (
        (1.0 - ty) * (1.0 - tx) * elevation[b, y0, x0]
        + (1.0 - ty) * tx * elevation[b, y0, x0 + 1]
        + ty * (1.0 - tx) * elevation[b, y0 + 1, x0]
        + ty * tx * elevation[b, y0 + 1, x0 + 1]
    ) + off_cap[b, iy, ix]


class SubcellDilation:
    """Preallocated refined-contact buffers for a [B, ny, nx] terrain stack."""

    def __init__(self, sim, cell_size: float, wheel_radius: float, sub: int = 8):
        self.sim = sim
        self.sub = sub
        self._refine = make_refine(sub)
        self.cell_size = cell_size
        self.wheel_radius = wheel_radius
        dy, dx, _ = wheel_offset_table(sim.env_radius, cell_size, wheel_radius)
        with wp.ScopedDevice(sim.device):
            self.off_dy = wp.zeros(sim.elevation.shape, dtype=wp.float32)
            self.off_dx = wp.zeros(sim.elevation.shape, dtype=wp.float32)
            self.off_cap = wp.zeros(sim.elevation.shape, dtype=wp.float32)
            self._tab_dy = wp.array(dy, dtype=wp.int32)
            self._tab_dx = wp.array(dx, dtype=wp.int32)

    def contact(self) -> None:
        """Off-tape: discrete arg-max (production kernel) then parabolic refinement."""
        self.sim._contact()
        wp.launch(
            self._refine,
            dim=self.sim.elevation.shape,
            inputs=[
                self.sim.elevation,
                self.sim._best_k,
                self._tab_dy,
                self._tab_dx,
                self.cell_size,
                self.wheel_radius,
                self.off_dy,
                self.off_dx,
                self.off_cap,
            ],
            device=self.sim.device,
        )

    def gather(self) -> None:
        """On-tape: build the envelope from the refined contact."""
        wp.launch(
            gather_subcell,
            dim=self.sim.elevation.shape,
            inputs=[self.sim.elevation, self.off_dy, self.off_dx, self.off_cap],
            outputs=[self.sim.envelope],
            device=self.sim.device,
        )


def continuum_envelope(h: np.ndarray, cell: float, wheel_radius: float, sub: int = 8) -> np.ndarray:
    """Brute-force reference: the max over a `sub`-times finer sampling of the BILINEAR terrain.

    The quantity both the discrete and the refined dilation are approximating. Slow numpy --
    a verification oracle, not a runtime path.
    """
    ny, nx = h.shape
    rad = int(np.ceil(wheel_radius / cell))
    out = np.full_like(h, -np.inf)
    for iy in range(-rad * sub, rad * sub + 1):
        for ix in range(-rad * sub, rad * sub + 1):
            fy, fx = iy / sub, ix / sub
            dist = np.hypot(fy, fx) * cell
            if dist > wheel_radius:
                continue
            cap = np.sqrt(wheel_radius**2 - dist**2) - wheel_radius
            y0, x0 = int(np.floor(fy)), int(np.floor(fx))
            ty, tx = fy - y0, fx - x0
            shifted = (
                (1 - ty) * (1 - tx) * _shift(h, y0, x0)
                + (1 - ty) * tx * _shift(h, y0, x0 + 1)
                + ty * (1 - tx) * _shift(h, y0 + 1, x0)
                + ty * tx * _shift(h, y0 + 1, x0 + 1)
            )
            out = np.maximum(out, shifted + cap)
    return out


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    ny, nx = a.shape
    i = np.clip(np.arange(ny)[:, None] + dy, 0, ny - 1)
    j = np.clip(np.arange(nx)[None, :] + dx, 0, nx - 1)
    return a[i, j]
