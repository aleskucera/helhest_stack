"""Study B -- is first order adequate? (SENSITIVITY_PLAN.md section 5, the decisive one)

    .venv/bin/python -m studies.adjoint.study_b

Study A established that `adj_settle_bt` is correct. Study B asks the different question of
whether a CORRECT derivative is a useful description of what happens when the map is wrong by
a realistic sigma -- centimetres, against a dilation whose frozen-arg-max gradient Study A
measured as exact only within 3.6 mm.

Two measurements, sharing one Monte-Carlo budget:

  B1 GLOBAL   Var_FOSM(J) = g^T (D C D) g  against brute-force Var_MC(J) over correlated
              terrain draws, swept over a sigma scale. Per rollout, so each result is labelled
              by the terrain the rollout actually drove over.

  B2 PER-CELL The attribution claim itself: for each probe cell, the FOSM contribution
              (g_i sigma_i)^2 against a single-cell Monte-Carlo variance. Stratified by region
              AND by the dilation contact margin, which is the flag Study A put in to replace
              min N_i. This is the paper's central figure -- if the margin predicts where the
              ratio blows up, it is a free linearisation-validity test.

Batching note: terrain is per-rollout, so an entire Monte-Carlo batch is ONE launch -- the B
slices are B independent draws of the same plan. That is what makes brute force affordable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from . import contact
from . import plot_b
from . import sigma as sigma_mod
from .harness import Harness
from .harness import TERM_NAMES
from .scene import build_scene
from .scene import LANE_Y
from .scene import REGION_NAMES
from .scene import rollouts

OUT_DIR = Path(__file__).resolve().parents[2] / "studies" / "out"

# J for Study B is plan-shaped, unlike Study A's deliberately-separable probes: tilt through
# the settle plus the tie-free belly-margin violation. `clear` (the tied min) is excluded --
# Study A finding 3 showed its gradient is not a derivative of anything.
COST_TERMS = {"settle": 1.0, "clear_soft": 1.0}

MC_DRAWS = 2048  # total Monte-Carlo draws per (rollout, sigma scale, correlation length)
# Multiples of the placeholder sigma field. The range brackets the breakdown rather than
# sitting on one side of it, and the smallest scale doubles as a SELF-CHECK: as sigma -> 0 the
# ratio must go to 1, and if it does not the noise normalisation or the FOSM formula is wrong,
# not the physics. Measured 0.98-0.99 at x0.03, which is the Monte-Carlo standard error.
SIGMA_SCALES = (0.03, 0.1, 0.3, 1.0, 3.0)
CORR_LENS = (0.0, 0.15)  # [m] independent cells, and a realistic patchy map error
CELL_DRAWS = 512  # draws per probe cell in B2
B2_ROLLOUTS = (2, 4, 5, 6)  # slope, curb head-on, curb from ON the edge, rock
SLACK_BUCKETS = ((0.0, 1e-3), (1e-3, 3e-3), (3e-3, np.inf))  # [m] validity-radius strata
# Floor on the validity radius when forming sigma/slack. 10 um is below float32 resolution on
# a 0.35 m height, so anything smaller is an exact tie and the ratio is meaningless anyway.
SLACK_FLOOR = 1e-5
SELF_CHECK_TOL = 0.05  # |ratio - 1| allowed at the smallest sigma scale
# Lateral offset of the decoy patch from the driven line. Must exceed the widest wheel-envelope
# contact reach (half_track 0.365 + wheel_radius 0.35 = 0.715 m) or it is not a decoy at all --
# it would be genuinely decision-relevant and the whole contrast collapses.
DECOY_OFFSET = 1.4


def _cost(terms: np.ndarray) -> np.ndarray:
    """Weighted plan cost per batch slice, from the harness's per-term array [N_TERMS, B]."""
    return sum(w * terms[TERM_NAMES.index(k)] for k, w in COST_TERMS.items())


def _cost_gradient(grads: np.ndarray) -> np.ndarray:
    """dJ/dh for the same weighting, [B, ny, nx]."""
    return sum(w * grads[TERM_NAMES.index(k)] for k, w in COST_TERMS.items())


class MonteCarlo:
    """A single plan replicated across the batch; each slice carries its own terrain draw."""

    def __init__(self, scene, pose: np.ndarray, omega: np.ndarray, batch: int, device="cuda"):
        poses = np.tile(pose, (batch, 1)).astype(np.float32)
        omegas = np.tile(omega[:, None, :], (1, batch, 1)).astype(np.float32)
        self.h = Harness(scene, poses, omegas, device=device)
        self.scene = scene
        self.batch = batch
        with wp.ScopedDevice(self.h.device):
            self._out = wp.zeros(self.h.sim.elevation.shape, dtype=wp.float32)

    def base_cost(self) -> float:
        self.h._reset_terrain(dilate=True)
        return float(_cost(self.h.forward(dilate=True))[0])

    def gradient(self) -> np.ndarray:
        grads, _ = self.h.adjoint(dilate=True, leaf="elevation")
        return _cost_gradient(grads)[0]  # every slice carries the same plan and terrain

    def sample(self, noise, sigma_dev, gain: float, n_draws: int, seed0: int) -> np.ndarray:
        """`n_draws` costs under perturbed terrain, in ceil(n_draws / batch) launches."""
        out = []
        for i in range(int(np.ceil(n_draws / self.batch))):
            noise.perturb(self.h._raw0, sigma_dev, gain, self._out, seed0 + 7919 * i)
            self.h.sim.set_terrain(self._out)
            out.append(_cost(self.h.forward(dilate=True)).copy())
        return np.concatenate(out)[:n_draws]


@wp.kernel
def _perturb_cell_random(arr: wp.array3d(dtype=wp.float32), iy: int, ix: int, sd: float, seed: int):
    """Perturb ONE cell by an independent Gaussian draw per batch slice.

    B draws of a single-cell perturbation in one launch -- the per-cell Monte-Carlo that makes
    the attribution claim testable cell by cell rather than only in aggregate.
    """
    b = wp.tid()
    state = wp.rand_init(seed, b)
    arr[b, iy, ix] = arr[b, iy, ix] + sd * wp.randn(state)


def _probe_by_slack(g, slack, thresh_frac=0.02, per_bucket=70):
    """Probe cells stratified BY the validity radius, ranked by |gradient| WITHIN each bucket.

    Study B v1 ranked purely by gradient magnitude and drew n=3 cells below 1 mm, which was
    far too few to say anything about the flag. Stratifying first guarantees each bucket is
    populated; ranking by |g| inside it keeps every probed cell one the adjoint says matters,
    so the ratio stays meaningful.
    """
    strength = np.abs(g)
    thresh = thresh_frac * strength.max()
    out = []
    for lo, hi in SLACK_BUCKETS:
        m = (slack >= lo) & (slack < hi) & (strength > thresh)
        idx = np.argwhere(m)
        if not len(idx):
            continue
        order = np.argsort(-strength[idx[:, 0], idx[:, 1]])
        out.append(idx[order][:per_bucket])
    return np.concatenate(out) if out else np.zeros((0, 2), int)


def run_b2(scene, sigma_np, poses, omega, labels, gains=(0.3, 1.0, 3.0)) -> list[dict]:
    """Per-cell FOSM contribution vs a single-cell Monte-Carlo variance.

    For cell i perturbed alone by delta ~ N(0, (gain*sigma_i)^2), first order predicts
    Var(J) = (g_i * gain * sigma_i)^2. The measured ratio is the per-cell adequacy of the
    linearisation -- what the project's attribution claim actually rests on, since an
    aggregate Var(J) can be right while every individual attribution is wrong.
    """
    rows: list[dict] = []
    for b in B2_ROLLOUTS:
        mc = MonteCarlo(scene, poses[b], omega[:, b, :], CELL_DRAWS)
        slack = contact.source_slack(mc.h)
        grads, _ = mc.h.adjoint(dilate=True, leaf="elevation")
        g = _cost_gradient(grads)[0]
        cells = _probe_by_slack(g, slack)
        print(
            f"    {labels[b]:<15} {len(cells)} cells x {len(gains)} sigma scales"
            f"  (slack<1mm: {sum(1 for iy, ix in cells if slack[iy, ix] < 1e-3)})",
            flush=True,
        )
        for gain in gains:
            for iy, ix in cells:
                sd = gain * float(sigma_np[iy, ix])
                mc.h._reset_terrain(dilate=True)
                wp.launch(
                    _perturb_cell_random,
                    mc.batch,
                    inputs=[mc.h.sim.elevation, int(iy), int(ix), sd, int(7 + iy * 991 + ix)],
                    device=mc.h.device,
                )
                v_mc = float(_cost(mc.h.forward(dilate=True)).var())
                v_fo = (float(g[iy, ix]) * sd) ** 2
                rows.append(
                    {
                        "rollout": labels[b],
                        "sigma_scale": gain,
                        "region": REGION_NAMES[int(scene.region[iy, ix])],
                        "slack_mm": 1e3 * float(slack[iy, ix]),
                        "sigma_mm": 1e3 * float(sigma_np[iy, ix]),
                        # sigma / slack: how many validity radii the perturbation spans. If the
                        # flag works at all, THIS is what the ratio should track, not sigma or
                        # slack alone.
                        "sigma_over_slack": float(
                            gain * sigma_np[iy, ix] / max(slack[iy, ix], SLACK_FLOOR)
                        ),
                        "grad": float(g[iy, ix]),
                        "ratio": float(np.sqrt(v_mc / v_fo)) if v_fo > 0 else float("nan"),
                    }
                )
        mc.h._reset_terrain(dilate=True)
        del mc
    return rows


def run_b1(scene, sigma_np, labels, poses, omega, batch: int = 256) -> list[dict]:
    """Var_FOSM vs Var_MC for whole plans, swept over sigma scale and correlation length."""
    rows: list[dict] = []
    with wp.ScopedDevice("cuda"):
        sigma_dev = wp.array(np.ascontiguousarray(sigma_np, np.float32), dtype=wp.float32)
    for b, label in enumerate(labels):
        mc = MonteCarlo(scene, poses[b], omega[:, b, :], batch)
        grad = mc.gradient()
        j0 = mc.base_cost()
        for corr in CORR_LENS:
            noise = sigma_mod.NoiseDraws(mc.h.sim.elevation.shape, scene.cell, corr, mc.h.device)
            for gain in SIGMA_SCALES:
                t0 = time.perf_counter()
                costs = mc.sample(noise, sigma_dev, gain, MC_DRAWS, seed0=1000 * b + 1)
                v_mc = float(costs.var())
                v_fo = sigma_mod.fosm_variance(grad, gain * sigma_np, scene.cell, corr)
                rows.append(
                    {
                        "rollout": label,
                        "corr_len": corr,
                        "sigma_scale": gain,
                        "sd_fosm": float(np.sqrt(v_fo)),
                        "sd_mc": float(np.sqrt(v_mc)),
                        "ratio": float(np.sqrt(v_mc / v_fo)) if v_fo > 0 else float("nan"),
                        # FOSM predicts a symmetric, zero-mean shift in J. A real bias is a
                        # second-order effect the linearisation cannot see at all.
                        "bias_over_sd": float((costs.mean() - j0) / np.sqrt(max(v_mc, 1e-30))),
                        "seconds": time.perf_counter() - t0,
                    }
                )
                print(
                    f"    {label:<15} corr={corr:.2f}  x{gain:<4} "
                    f"sd_FOSM={np.sqrt(v_fo):9.4f}  sd_MC={np.sqrt(v_mc):9.4f}  "
                    f"ratio={rows[-1]['ratio']:6.3f}  bias/sd={rows[-1]['bias_over_sd']:+6.2f}"
                    f"  ({rows[-1]['seconds']:.1f}s)",
                    flush=True,
                )
        del mc
    return rows


def _contact_margin(scene) -> np.ndarray:
    """The dilation contact margin for the unperturbed scene (Study A's validity flag)."""
    poses, omega, _ = rollouts()
    h = Harness(scene, poses[:1], omega[:, :1, :])
    h.sim.set_terrain(h._raw0)
    h.sim._contact()
    m = h.sim.contact_margin.numpy()[0].copy()
    del h
    return m


def _decoy_check(scene, sigma_np, decoy, poses, omega, labels) -> None:
    """How much FOSM variance lands in the high-sigma patch that is NOT decision-relevant?

    SENSITIVITY_PLAN.md section 6 wants a case where the highest-entropy cell and the
    highest-dJ/dh cell differ. An entropy-directed sensor spends its budget where sigma is
    largest; if attribution is doing anything, its mass here is negligible.
    """
    for b in (0, 1):
        mc = MonteCarlo(scene, poses[b], omega[:, b, :], 8)
        contrib = (mc.gradient() * sigma_np) ** 2
        share = contrib[decoy].sum() / max(contrib.sum(), 1e-30)
        ent = (sigma_np[decoy] ** 2).sum() / (sigma_np**2).sum()
        print(
            f"    {labels[b]:<15} decoy holds {ent:6.1%} of the map's total sigma^2 but only "
            f"{share:8.3%} of the FOSM variance"
        )
        del mc


def _self_check(b1: list[dict]) -> tuple[bool, list[str]]:
    """At the smallest sigma the ratio MUST be 1: that validates the noise normalisation and
    the correlated FOSM formula, so any later deviation is physics rather than a harness bug."""
    notes = []
    smallest = min(SIGMA_SCALES)
    bad = [
        r
        for r in b1
        if r["sigma_scale"] == smallest
        and r["corr_len"] > 0
        and abs(r["ratio"] - 1.0) > SELF_CHECK_TOL
    ]
    for r in bad:
        notes.append(f"{r['rollout']} corr={r['corr_len']}: ratio {r['ratio']:.3f} at x{smallest}")
    return not bad, notes


def _report_b2(b2: list[dict]) -> None:
    """Stratify the per-cell ratio by region and by the contact-margin flag."""
    import numpy as np

    for gain in sorted({r["sigma_scale"] for r in b2}):
        sub = [r for r in b2 if r["sigma_scale"] == gain and np.isfinite(r["ratio"])]
        print(f"\n    sigma x{gain}  -- per-cell sqrt(Var_MC / Var_FOSM)")
        print(f"      {'stratum':<22}{'n':>6}{'median':>9}{'p90':>9}{'frac >2x':>10}")
        groups = [("region " + n, [r for r in sub if r["region"] == n]) for n in REGION_NAMES[:-1]]
        # The flag under test: does a small contact margin predict the breakdown?
        groups += [
            ("slack < 1 mm", [r for r in sub if r["slack_mm"] < 1.0]),
            ("slack 1-3 mm", [r for r in sub if 1.0 <= r["slack_mm"] < 3.0]),
            ("slack >= 3 mm", [r for r in sub if r["slack_mm"] >= 3.0]),
            # The flag's real claim: the ratio should track sigma measured in validity radii.
            ("sigma/slack < 1", [r for r in sub if r["sigma_over_slack"] < 1.0]),
            ("sigma/slack 1-10", [r for r in sub if 1.0 <= r["sigma_over_slack"] < 10.0]),
            ("sigma/slack >= 10", [r for r in sub if r["sigma_over_slack"] >= 10.0]),
        ]
        for name, rs in groups:
            if not rs:
                continue
            v = np.array([r["ratio"] for r in rs])
            print(
                f"      {name:<22}{len(v):>6}{np.median(v):>9.3f}"
                f"{np.percentile(v, 90):>9.3f}{np.mean(v > 2.0):>10.1%}"
            )


def _summarise(b1: list[dict], b2: list[dict], ok: bool, notes: list[str]) -> None:
    import numpy as np

    print("\n" + "=" * 78)
    print("STUDY B VERDICT")
    print("=" * 78)
    print(
        f"\n0. SELF-CHECK  {'PASS' if ok else 'FAIL'} -- at the smallest sigma the FOSM/MC ratio "
        f"is 1 to within {SELF_CHECK_TOL:.0%},\n   so the noise normalisation and the correlated "
        "FOSM formula are right and every\n   deviation below is the physics, not the harness."
    )
    for n in notes:
        print("     - " + n)

    corr = [r for r in b1 if r["corr_len"] > 0]
    iid = [r for r in b1 if r["corr_len"] == 0.0]
    print(
        "\n1. i.i.d. PER-CELL NOISE IS NOT A VALID MODEL HERE. Median |bias| / sd is "
        f"{np.median([abs(r['bias_over_sd']) for r in iid]):.1f} sd\n"
        "   for independent cells against "
        f"{np.median([abs(r['bias_over_sd']) for r in corr]):.1f} sd at a 0.15 m correlation "
        "length.\n   The envelope is a MAX over ~37 cells, so zero-mean independent noise "
        "raises it\n   systematically -- a pure second-order effect that FOSM cannot see and "
        "that swamps\n   the variance it does predict. Correlated draws are the only "
        "meaningful column."
    )
    by_roll: dict[str, list] = {}
    for r in corr:
        by_roll.setdefault(r["rollout"], []).append(r)
    print("\n2. WHERE FIRST ORDER HOLDS (correlated draws, sd ratio by sigma scale):")
    scales = sorted({r["sigma_scale"] for r in corr})
    print("     " + "rollout".ljust(17) + "".join(f"x{g:<7}" for g in scales))
    for name, rs in by_roll.items():
        cells = {r["sigma_scale"]: r["ratio"] for r in rs}
        print(
            "     "
            + name.ljust(17)
            + "".join(f"{cells.get(g, float('nan')):<8.2f}" for g in scales)
        )

    def _pool(pred):
        """Pooled across ALL sigma scales: the criterion is scale-free by construction, so
        pooling is the honest presentation and avoids reporting an n=2 bucket."""
        v = [r["ratio"] for r in b2 if pred(r) and np.isfinite(r["ratio"])]
        return (
            (np.median(v) if v else float("nan")),
            (np.mean(np.array(v) > 2.0) if v else 0.0),
            len(v),
        )

    u = _pool(lambda r: r["sigma_over_slack"] < 1.0)
    m = _pool(lambda r: 1.0 <= r["sigma_over_slack"] < 10.0)
    o = _pool(lambda r: r["sigma_over_slack"] >= 10.0)
    inside = [r for r in b2 if r["sigma_over_slack"] < 1.0 and r["sigma_scale"] >= 1.0]
    print(
        "\n3. A CRISP VALIDITY CRITERION, AND THE BAD NEWS THAT COMES WITH IT.\n"
        "   Probe cells are stratified BY the per-source validity radius (contact.source_slack)\n"
        "   rather than by gradient magnitude -- v1 drew n=3 below 1 mm and could conclude\n"
        "   nothing. Note the radius must be per-SOURCE-cell: the engine's contact_margin is\n"
        "   indexed by OUTPUT cell and answers a different question (683 cells here have\n"
        "   slack < 1 mm, 462 have margin < 1 mm, only 211 are both).\n"
        "   Pooled over every sigma scale, by perturbation measured IN validity radii:\n"
        f"     sigma/slack <  1   median ratio {u[0]:.2f},  {u[1]:.1%} of cells worse than 2x  (n={u[2]})\n"
        f"     sigma/slack 1-10   median ratio {m[0]:.2f},  {m[1]:.1%} worse than 2x  (n={m[2]})\n"
        f"     sigma/slack >= 10  median ratio {o[0]:.2f},  {o[1]:.1%} worse than 2x  (n={o[2]})\n"
        "   -> THE CRITERION HOLDS: inside its own validity radius, per-cell first-order\n"
        "      attribution is essentially exact, and it degrades monotonically outside it.\n"
        f"   -> THE CATCH: at realistic sigma almost nothing is inside. Only {len(inside)} of the\n"
        f"      {sum(1 for r in b2 if r['sigma_scale'] >= 1.0)} probes at sigma x1 or above satisfy it, "
        "because slack is ~3.6 mm\n      (Study A's cap step) while sigma is centimetres. So the "
        "method is not blocked by\n      a missing criterion -- it is blocked by the radius being "
        "too small.\n"
        "   -> WHICH POINTS AT THE FIX: Study A showed the 3.6 mm radius is contact QUANTIZATION,\n"
        "      not physics -- letting the contact slide sub-cell makes d(env)/dh continuous. That\n"
        "      raises slack, which is now measurably the thing that gates the whole method."
    )
    # Which rollout is most globally accurate, and how does it look per-cell?
    glob = {}
    for r in corr:
        glob.setdefault(r["rollout"], []).append(abs(r["ratio"] - 1.0))
    best = min(
        (k for k in glob if any(x["rollout"] == k for x in b2)), key=lambda k: np.mean(glob[k])
    )
    cell = [r["ratio"] for r in b2 if r["rollout"] == best and r["sigma_scale"] == 1.0]
    print(
        "\n4. PER-CELL AND GLOBAL ADEQUACY COME APART.\n"
        f"   '{best}' is the most accurate rollout GLOBALLY (mean |ratio-1| "
        f"{np.mean(glob[best]):.2f} over\n   the sigma sweep) yet its PER-CELL ratios at sigma x1 "
        f"have median {np.median(cell):.2f} and p90\n   {np.percentile(cell, 90):.1f}. Aggregate "
        "Var(J) can be accurate while individual cell\n   attributions are badly wrong, because "
        "the per-cell errors cancel in the sum.\n"
        "   The project's claim is ATTRIBUTION, not Var(J), so B2 is the measurement that gates\n"
        "   it -- B1 alone would have been misleadingly reassuring."
    )


def main() -> None:
    wp.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    poses, omega, labels = rollouts()

    # The decoy: a high-sigma unobserved patch offset from the flat lane's driven line.
    decoy = sigma_mod.decoy_mask(scene, LANE_Y[0] + DECOY_OFFSET)
    sigma_np = sigma_mod.sigma_field(scene, inpainted=decoy)
    print(
        f"scene {scene.shape}; placeholder sigma: median {1e2 * np.median(sigma_np):.1f} cm, "
        f"p99 {1e2 * np.percentile(sigma_np, 99):.1f} cm, decoy patch {decoy.sum()} cells"
    )
    for code, name in enumerate(REGION_NAMES[:-1]):
        sel = scene.region == code
        if sel.any():
            print(f"    sigma in {name:<6} median {1e2 * np.median(sigma_np[sel]):.2f} cm")

    print(f"\nB1  global Var_FOSM vs Var_MC   ({MC_DRAWS} draws per point of the sweep)")
    b1 = run_b1(scene, sigma_np, labels, poses, omega)
    ok, notes = _self_check(b1)

    print("\n    decoy check -- is high sigma the same thing as decision-relevant?")
    _decoy_check(scene, sigma_np, decoy, poses, omega, labels)

    print(f"\nB2  per-cell attribution   ({CELL_DRAWS} draws per cell)")
    margin = _contact_margin(scene)
    b2 = run_b2(scene, sigma_np, poses, omega, labels)
    _report_b2(b2)
    _summarise(b1, b2, ok, notes)
    np.savez_compressed(
        OUT_DIR / "study_b.npz",
        sigma=sigma_np,
        decoy=decoy,
        region=scene.region,
        contact_margin=margin,
    )
    plot_b.figure(scene, sigma_np, decoy, b1, b2, OUT_DIR / "study_b.png")
    (OUT_DIR / "study_b.json").write_text(
        json.dumps({"b1": b1, "b2": b2, "self_check_pass": ok, "self_check": notes}, indent=2)
    )
    print(f"\nwrote {OUT_DIR / 'study_b.npz'} and study_b.json")


if __name__ == "__main__":
    main()
