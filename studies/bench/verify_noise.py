"""Gates on the noise model. Must pass before any result computed on top of it is believed.

    .venv/bin/python -m studies.bench.verify_noise

A noise model is a claim about structure, and a plausible-looking field can easily be the wrong
structure -- which for this study would be fatal, because Study B already showed structure is
what decides the answer. Each gate below checks a property the model is SUPPOSED to have, and
the one that matters most is gate 5: it verifies that the localisation error really is a
rank-3 rigid-body error and not per-cell noise wearing a disguise.
"""

from __future__ import annotations

import numpy as np

from ..adjoint.generalise import fractal_terrain
from . import noise as N

CELL = 0.10


def _grid():
    xs = -1.0 + (np.arange(90) + 0.5) * CELL
    ys = -4.5 + (np.arange(90) + 0.5) * CELL
    return np.meshgrid(xs, ys)


def main() -> None:
    XX, YY = _grid()
    rng_field = np.hypot(XX, YY)
    near = rng_field <= N.MAX_RANGE
    fails = []

    def gate(ok: bool, name: str, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            fails.append(name)

    print("occlusion -- is the ray cast a real line-of-sight test?")
    flat_vis = N.visibility(np.zeros((90, 90)), XX, YY, CELL)[near].mean()
    gate(flat_vis > 0.999, "flat ground is fully visible", f"{flat_vis:.1%}")

    wall = np.where(np.abs(XX - 2.0) < 0.15, 0.8, 0.0)
    vw = N.visibility(wall, XX, YY, CELL)
    behind = near & (XX > 2.3) & (np.abs(YY) < 1.0)
    front = near & (XX < 1.7) & (np.abs(YY) < 1.0)
    gate(vw[behind].mean() < 0.05, "a wall casts a shadow", f"{1 - vw[behind].mean():.1%} hidden")
    gate(vw[front].mean() > 0.95, "and only behind it", f"{vw[front].mean():.1%} visible in front")

    print("\nsensor -- correlated, range-growing, and the right magnitude?")
    truth = fractal_terrain(90, 90, CELL, seed=3)
    _, meas, obs, sig, _ = N.build_belief(truth, XX, YY, CELL, 3, "sensor")
    err = meas - truth
    expect = N.SENSOR_BASE + N.SENSOR_PER_M * rng_field
    ratio = err.std() / expect.mean()
    gate(0.8 < ratio < 1.25, "injected std matches the configured sd", f"ratio {ratio:.2f}")
    ac = np.corrcoef(err[:, :-1].ravel(), err[:, 1:].ravel())[0, 1]
    gate(ac > 0.6, "error is spatially CORRELATED, not per-cell white", f"autocorr@1cell {ac:.2f}")
    near_far = np.abs(err[rng_field > 5]).mean() / np.abs(err[rng_field < 2]).mean()
    gate(near_far > 1.5, "and grows with range", f"far/near |error| {near_far:.2f}x")

    print("\nlocalisation -- is it RANK-3 rigid-body, or per-cell noise in disguise?")
    _, meas, obs, sig, info = N.build_belief(truth, XX, YY, CELL, 3, "localisation")
    d = (meas - truth).ravel()
    gy, gx = np.gradient(truth, CELL)
    # A rigid (dx, dy, dyaw) displaces (x, y) by (dx - y*dyaw, dy + x*dyaw), so to first order
    # the height error is dx*gx + dy*gy + dyaw*(x*gy - y*gx). Three columns, 8100 rows: if a
    # 3-parameter fit explains the field, the error IS a pose error.
    # The DIRECT proof of rank-3: re-apply the three injected numbers and see if the whole
    # field comes back. Regressing against a first-order gradient model would not prove it --
    # and, as reported below, would fail for a reason that is about the model, not the field.
    dx, dy, dyaw = info["pose_error"]
    recon = N.apply_pose_error(truth, XX, YY, CELL, dx, dy, dyaw)
    err_recon = np.abs(recon - meas).max()
    gate(
        err_recon < 1e-6,
        "the WHOLE error field is reproduced from 3 numbers",
        f"max |reconstruction - measured| {err_recon:.2e}, "
        f"pose error ({dx * 100:+.1f}, {dy * 100:+.1f}) cm, {np.degrees(dyaw):+.2f} deg",
    )
    A = np.stack([gx.ravel(), gy.ravel(), (XX * gy - YY * gx).ravel()], axis=1)
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    r2 = 1.0 - ((d - A @ coef) ** 2).sum() / ((d - d.mean()) ** 2).sum()
    disp = np.hypot(N.LOC_SIGMA_XY, 6.0 * N.LOC_SIGMA_YAW) / CELL
    ac1 = np.corrcoef(truth[:, :-1].ravel(), truth[:, 1:].ravel())[0, 1]
    print(
        f"   FINDING, not a gate: the FIRST-ORDER marginal grad(h).d explains only R^2 {r2:.2f}\n"
        f"      of this field. A pose error is not a small perturbation here -- it displaces the\n"
        f"      map by ~{disp:.1f} cells while the terrain decorrelates in ~1 (autocorr {ac1:.2f}\n"
        f"      at one cell), so the displaced map is nearly INDEPENDENT of the truth. Same\n"
        f"      lesson as Study B, now for pose rather than map noise."
    )
    # ...and it is NOT reducible by looking: the same pose error corrupts every measurement.
    print(
        "         (this is why sensing cannot fix it -- revealing more cells delivers more\n"
        "          measurements carrying the SAME three wrong numbers)"
    )

    print("\nsigma -- is what the policies are told consistent with what was injected?")
    for src in ("sensor", "localisation", "all"):
        _, meas, obs, sig, _ = N.build_belief(truth, XX, YY, CELL, 3, src)
        actual = np.abs(meas - truth)[obs].std()
        told = np.median(sig[obs])
        gate(
            0.3 < told / max(actual, 1e-9) < 3.0,
            f"{src:<13} reported sigma within 3x of the true error",
            f"told {told:.4f} vs actual {actual:.4f} ({told / max(actual, 1e-9):.2f}x)",
        )

    print("\nclean -- the section-7 baseline is unchanged")
    _, meas, obs, sig, _ = N.build_belief(truth, XX, YY, CELL, 3, "clean")
    worst = float(np.abs(meas - truth).max())
    gate(worst < 1e-6, "reveals deliver exact truth", f"max error {worst:.2e} (float32 only)")
    gate(
        np.abs(obs.mean() - (rng_field <= 1.5).mean()) < 1e-9,
        "observed set is the 1.5 m disc",
        f"{obs.mean():.1%} observed",
    )

    print("\nhow much of the map each configuration observes, over 8 seeds:")
    for src in N.SOURCES:
        fr = [
            N.build_belief(fractal_terrain(90, 90, CELL, seed=s), XX, YY, CELL, s, src)[2].mean()
            for s in range(8)
        ]
        print(f"  {src:<13} {np.mean(fr):5.1%}")

    print(f"\n{'ALL GATES PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")


if __name__ == "__main__":
    main()
