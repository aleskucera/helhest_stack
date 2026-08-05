"""Per-SOURCE-cell validity radius of the frozen-arg-max dilation gradient.

`contact_margin` (engine, `envelope.py`) is indexed by OUTPUT cell: for envelope cell p it is
winner - runner-up, i.e. how far terrain must move before p's contact changes. That is the
right flag when you ask about the envelope.

Study B perturbs a SOURCE cell q of the raw elevation, which is a different question. The
gradient dJ/dh_q sums `denv_p/dh_q` over every output cell p that currently selects q, so it
stays exact only while that SET is unchanged. Two things can change it:

    q loses an output it currently wins    -- needs q to fall by more than margin[p]
    q gains an output it currently loses   -- needs q to rise by more than its deficit there

so the radius is the smaller of the two, minimised over the disk:

    slack(q) = min over offsets d of   margin[p]                       if q wins at p = q - d
                                       envelope[p] - (h[q] + cap_d)    otherwise

Reading `margin[q]` instead -- q's own contest as an output cell -- answers a question nobody
asked, and on this scene the two differ enough to change which cells look risky.

Study-side for now: it is a diagnostic built from arrays the engine already produces
(`elevation`, `envelope`, `_best_k`, `contact_margin`), and it belongs in the engine only if
Study B shows it earns its place.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _source_slack(
    elevation: wp.array2d(dtype=wp.float32),
    envelope: wp.array2d(dtype=wp.float32),
    best_k: wp.array2d(dtype=wp.float32),
    margin: wp.array2d(dtype=wp.float32),
    off_dy: wp.array(dtype=wp.int32),
    off_dx: wp.array(dtype=wp.int32),
    off_cap: wp.array(dtype=wp.float32),
    slack: wp.array2d(dtype=wp.float32),
):
    qy, qx = wp.tid()
    ny = elevation.shape[0]
    nx = elevation.shape[1]
    hq = elevation[qy, qx]
    best = float(1.0e9)  # noqa: UP018
    for k in range(off_dy.shape[0]):
        # q supplies output cell p = q - d, because envelope[p] maxes over h[p + d].
        py = qy - off_dy[k]
        px = qx - off_dx[k]
        if py >= 0 and py < ny and px >= 0 and px < nx:
            if int(best_k[py, px]) == k:
                best = wp.min(best, margin[py, px])  # q wins here: how far before it loses
            else:
                best = wp.min(best, envelope[py, px] - (hq + off_cap[k]))  # how far to win
    slack[qy, qx] = best


def source_slack(harness) -> np.ndarray:
    """[ny, nx] per-source-cell validity radius [m] for the harness's unperturbed terrain."""
    sim = harness.sim
    harness._reset_terrain(dilate=True)
    sim._contact()
    sim._gather()
    with wp.ScopedDevice(harness.device):
        out = wp.zeros(harness.scene.shape, dtype=wp.float32)
    wp.launch(
        _source_slack,
        dim=harness.scene.shape,
        inputs=[
            sim.elevation[0],
            sim.envelope[0],
            sim._best_k[0],
            sim.contact_margin[0],
            sim._off_dy,
            sim._off_dx,
            sim._off_cap,
            out,
        ],
        device=harness.device,
    )
    return out.numpy()
