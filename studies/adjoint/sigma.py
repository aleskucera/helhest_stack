"""Placeholder per-cell map uncertainty, and correlated terrain-noise draws on device.

SENSITIVITY_PLAN.md section 2 is explicit that a CALIBRATED uncertainty layer is not this
project's contribution -- UNRealNet (arXiv:2407.08720) and the Neural-Processes elevation
model (arXiv:2508.03890) already produce and held-out validate one. Study B consumes an
uncertainty field; it does not claim it. So this is a deliberate placeholder shaped like what
the perception heightmap builder already exports (within-cell max-min spread, and a mask of
cells that were inpainted rather than measured), not a model of anything.

Two structures matter for the result and both are here:

  ROUGHNESS-DRIVEN sigma -- large where the terrain within a cell varies, i.e. at the curb
  edge and on the rocks. This is the case where high sigma and high curvature coincide, which
  is where SENSITIVITY_PLAN.md section 4 predicts first order fails.

  An INPAINTED PATCH -- large sigma over a contiguous region that was never observed. Placed
  deliberately OFF the driven line, so it is high-uncertainty but decision-IRRELEVANT: the
  decoy the plan's section 6 benchmark needs, and a direct test of whether attribution beats
  entropy at picking where to look.

Noise draws are correlated, not per-cell independent. Independent per-cell noise is not
conservative here -- it is unrealistically benign, because the wheel envelope is a max over
~37 cells and averages i.i.d. noise away, while a real map is wrong in patches. The draw is a
white field blurred by a separable Gaussian of correlation length `corr_len`, renormalised to
unit marginal variance and then scaled by sigma.

Because the blur is a linear operator W, the FOSM variance under the SAME correlation is
exact and cheap -- no covariance matrix is ever formed:

    Var_FOSM = g^T (D C D) g,   C = W W^T / sum(w^2),   D = diag(sigma)
             = || W^T (sigma * g) ||^2 / sum(w^2)

i.e. blur the sigma-weighted gradient with the same kernel and take its squared norm. That is
`fosm_variance` below; the independent case is the corr_len -> 0 limit, sum_i (g_i sigma_i)^2.
"""

from __future__ import annotations

import numpy as np
import warp as wp

SIGMA_FLOOR = 0.010  # [m] best-case map noise on smooth measured ground
SIGMA_ROUGH_GAIN = 0.6  # [m per m] sigma added per metre of local within-cell spread
SIGMA_CEIL = 0.10  # [m] cap on the roughness-driven part
SIGMA_INPAINTED = 0.12  # [m] cells that were never observed
CORR_LEN = 0.15  # [m] default spatial correlation length of map error


def sigma_field(scene, inpainted: np.ndarray | None = None) -> np.ndarray:
    """[ny, nx] per-cell height uncertainty [m]. Shaped like the perception builder's
    (max - min) spread layer, plus a flat high value wherever `inpainted` is set."""
    h = scene.elevation
    lo = np.full_like(h, np.inf)
    hi = np.full_like(h, -np.inf)
    for dy in (-1, 0, 1):  # 3x3 local spread == the builder's within-cell max - min
        for dx in (-1, 0, 1):
            s = np.roll(np.roll(h, dy, axis=0), dx, axis=1)
            lo, hi = np.minimum(lo, s), np.maximum(hi, s)
    sigma = np.clip(SIGMA_FLOOR + SIGMA_ROUGH_GAIN * (hi - lo), SIGMA_FLOOR, SIGMA_CEIL)
    if inpainted is not None:
        sigma = np.where(inpainted, SIGMA_INPAINTED, sigma)
    return sigma


def decoy_mask(scene, lane_y: float, x_from: float = 0.2, half_width: float = 0.55) -> np.ndarray:
    """A contiguous 'never observed' patch offset laterally from a lane's driven line.

    High sigma, but outside the wheel envelope's reach -- so an entropy-directed sensor would
    go here and an attribution-directed one would not. Study B checks that the FOSM mass
    actually is negligible here; the plan's section 6 benchmark then builds on it.
    """
    ny, nx = scene.shape
    xs = scene.origin_x + (np.arange(nx) + 0.5) * scene.cell
    ys = scene.origin_y + (np.arange(ny) + 0.5) * scene.cell
    XX, YY = np.meshgrid(xs, ys)
    return (XX > x_from) & (np.abs(YY - lane_y) < half_width)


# --- correlated noise on device ----------------------------------------------------------
@wp.kernel
def _white(seed: int, out: wp.array3d(dtype=wp.float32)):
    b, iy, ix = wp.tid()
    nx = out.shape[2]
    ny = out.shape[1]
    state = wp.rand_init(seed, b * ny * nx + iy * nx + ix)
    out[b, iy, ix] = wp.randn(state)


@wp.kernel
def _blur_axis(
    src: wp.array3d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    radius: int,
    along_x: int,
    dst: wp.array3d(dtype=wp.float32),
):
    """One pass of a separable Gaussian blur, edge-clamped."""
    b, iy, ix = wp.tid()
    ny = src.shape[1]
    nx = src.shape[2]
    acc = float(0.0)  # noqa: UP018
    for k in range(-radius, radius + 1):
        qy = iy
        qx = ix
        if along_x == 1:
            qx = wp.clamp(ix + k, 0, nx - 1)
        else:
            qy = wp.clamp(iy + k, 0, ny - 1)
        acc += weights[k + radius] * src[b, qy, qx]
    dst[b, iy, ix] = acc


@wp.kernel
def _scale_add(
    base: wp.array3d(dtype=wp.float32),
    noise: wp.array3d(dtype=wp.float32),
    sigma: wp.array2d(dtype=wp.float32),
    gain: float,
    out: wp.array3d(dtype=wp.float32),
):
    """out = base + gain * sigma * unit-variance noise. Terrain stays device-resident."""
    b, iy, ix = wp.tid()
    out[b, iy, ix] = base[b, iy, ix] + gain * sigma[iy, ix] * noise[b, iy, ix]


class NoiseDraws:
    """Preallocated correlated-noise generator over a [B, ny, nx] terrain stack."""

    def __init__(self, shape: tuple[int, int, int], cell: float, corr_len: float, device):
        self.device = device
        self.corr_len = corr_len
        w, self.radius = _gauss_kernel(corr_len, cell)
        # Blurring white noise by w scales its variance by sum(w^2) per axis; dividing by
        # sqrt of the 2D total restores unit marginal variance so `sigma` means what it says.
        self.renorm = 1.0 / float(np.sqrt((w**2).sum() ** 2))
        with wp.ScopedDevice(device):
            self._w = wp.array(w, dtype=wp.float32)
            self._a = wp.zeros(shape, dtype=wp.float32)
            self._b = wp.zeros(shape, dtype=wp.float32)

    def perturb(
        self, base: wp.array, sigma: wp.array, gain: float, out: wp.array, seed: int
    ) -> None:
        """out = base + gain * sigma * (fresh unit-variance correlated field). All on device --
        a [B, ny, nx] stack is bulk data and never round-trips to the host."""
        wp.launch(_white, dim=self._a.shape, inputs=[seed, self._a], device=self.device)
        if self.radius > 0:
            wp.launch(
                _blur_axis,
                dim=self._a.shape,
                inputs=[self._a, self._w, self.radius, 1, self._b],
                device=self.device,
            )
            wp.launch(
                _blur_axis,
                dim=self._a.shape,
                inputs=[self._b, self._w, self.radius, 0, self._a],
                device=self.device,
            )
        wp.launch(
            _scale_add,
            dim=self._a.shape,
            inputs=[base, self._a, sigma, gain * self.renorm, out],
            device=self.device,
        )


def _gauss_kernel(corr_len: float, cell: float) -> tuple[np.ndarray, int]:
    """Normalised 1D Gaussian taps for a correlation length in metres."""
    if corr_len <= 0.0:
        return np.ones(1, np.float32), 0
    sd = corr_len / cell
    radius = max(int(np.ceil(3.0 * sd)), 1)
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (k / sd) ** 2)
    return (w / w.sum()).astype(np.float32), radius


def fosm_variance(grad: np.ndarray, sigma: np.ndarray, cell: float, corr_len: float) -> float:
    """Var_FOSM under the SAME correlation the Monte-Carlo draws use.

    corr_len = 0 gives the independent-cell formula sum_i (g_i sigma_i)^2. Otherwise it is
    ||W^T (sigma g)||^2 / sum(w^2) with W the separable blur -- exact for this noise model,
    and computed without ever forming a covariance matrix.
    """
    sg = sigma * grad
    if corr_len <= 0.0:
        return float((sg**2).sum())
    w, radius = _gauss_kernel(corr_len, cell)
    pad = np.pad(sg, radius, mode="edge")
    blurred = np.apply_along_axis(lambda m: np.convolve(m, w, mode="valid"), 1, pad)
    blurred = np.apply_along_axis(lambda m: np.convolve(m, w, mode="valid"), 0, blurred)
    return float((blurred**2).sum() / (w**2).sum() ** 2)
