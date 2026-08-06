"""Evaluation-only ablation: does decision-focused TRAINING beat the standard modular
architecture "MSE-trained map + uncertainty-aware PLANNING"? No retraining -- this reuses the
mse/decision weights already fit by `dfl.py`'s full run (studies/out/bench/dfl_full.json).

    .venv/bin/python -m studies.bench.dfl_ablation

Four new choice rules, evaluated on the SAME 80 held-out seeds `dfl.py` used:

  mse_step        argmin_k [ J_k(mse map) + KAPPA * sum_t sigma_t ]  -- STEP (Fan et al. 2021)
                   risk-aware planning (risk.py's own formula) stacked on the MSE-trained map.
  mse_cvar         argmin_k of the empirical CVaR_0.9 over M correlated terrain draws around the
                   MSE-trained map -- the strongest point-estimate-plus-planner-side-risk
                   baseline: same noise model, same CVaR level, same common-random-numbers-
                   across-plans protocol risk.py uses for its own Monte-Carlo truth.
  decision_step    the identical STEP choice rule, applied to the DECISION-trained map.
  decision_cvar    the identical CVaR choice rule, applied to the DECISION-trained map -- the
                   composition test: does risk-aware planning stack with decision training?

PRE-REGISTERED READING: if mse_cvar's regret is at or below decision-plain's, the modular
architecture (mean map + uncertainty-aware planner) captures what decision training bought, and
decision training's remaining value is compute (no online sampling/CVaR at plan time). If
decision-plain beats mse_cvar, the (mean, sigma) factorization has thrown away decision-relevant
information that planner-side risk handling, working only from the mean and sigma, cannot
reconstruct. `decision_cvar` vs `decision` (plain) checks whether the two are complementary
(composition beats both parents) or redundant.
"""

from __future__ import annotations

import json
import time

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from ..adjoint.sigma import NoiseDraws
from .dfl import _assemble
from .dfl import build_seed
from .dfl import FAMILY
from .dfl import FEATURE_NAMES
from .dfl import model_fill
from .dfl import NOISE
from .dfl import SeedData
from .dfl import TEST_SEEDS
from .ranking import _cost
from .ranking import _evaluate
from .ranking import build_case
from .ranking import CELL
from .ranking import kendall_tau
from .ranking import N_PLANS
from .ranking import OUT
from .ranking import sign_test
from .risk import _footprint_sigma
from .risk import ALPHA
from .risk import CORR_LEN
from .risk import empirical_cvar
from .risk import KAPPA

M_DRAWS = 128  # correlated terrain draws for the CVaR arms; halved by the runtime guard if slow


def _sigma_field(s: SeedData) -> np.ndarray:
    """The sigma feature is stored unmodified in `phi`; reuse it rather than recompute it."""
    return s.phi[..., FEATURE_NAMES.index("sigma")]


def choice_step(s: SeedData, fill: np.ndarray, grid: tuple[np.ndarray, np.ndarray]) -> tuple[int, np.ndarray]:
    """argmin_k [ J_k(filled map) + KAPPA * sum_t sigma_t ], STEP's Gaussian-CVaR risk term
    (risk.py), read off the trajectory the SAME forward pass on `fill`'s map produced."""
    elev = _assemble(s, fill)
    j_hat = _evaluate(s.harness, elev)
    traj = s.harness.sim.controlled.numpy()[:, :, :2].copy()
    sig_t = _footprint_sigma(traj, _sigma_field(s), grid)
    est = j_hat + KAPPA * sig_t.sum(axis=0)
    return int(np.argmin(est)), est


def choice_cvar(
    s: SeedData,
    fill: np.ndarray,
    hd: Harness,
    draws: NoiseDraws,
    poses: np.ndarray,
    omega: np.ndarray,
    cvar_seed: int,
) -> tuple[int, np.ndarray]:
    """argmin_k CVaR_ALPHA over `hd.batch_size` correlated draws around the filled map, with
    COMMON RANDOM NUMBERS across plans (the same `cvar_seed` every k) -- risk.py's own
    Monte-Carlo-truth pattern (bundled.py/risk.py), reused here as a CHOICE RULE. Draws are
    batched on `hd`'s batch axis, entirely on-device (`NoiseDraws.perturb`); only the K
    per-plan cost vectors round-trip to the host.
    """
    elev = _assemble(s, fill)
    m = hd.batch_size
    with wp.ScopedDevice(hd.device):
        base = wp.array(np.ascontiguousarray(np.tile(elev, (m, 1, 1)), np.float32))
        sig_dev = wp.array(np.ascontiguousarray(_sigma_field(s), np.float32), dtype=wp.float32)
    samples = np.empty((m, N_PLANS), np.float32)
    for k in range(N_PLANS):
        hd.sim.start_pose.assign(np.tile(poses[k], (m, 1)).astype(np.float32))
        hd.sim.target_wheel_omega.assign(
            np.ascontiguousarray(np.repeat(omega[:, k : k + 1, :], m, axis=1), np.float32)
        )
        draws.perturb(base, sig_dev, 1.0, hd.sim.elevation, cvar_seed)
        samples[:, k] = _cost(hd.forward(dilate=True))
    cvar = empirical_cvar(samples, ALPHA)
    return int(np.argmin(cvar)), cvar


def main(m_draws: int = M_DRAWS) -> None:
    t0 = time.time()
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)

    full = json.loads((OUT / "dfl_full.json").read_text())
    w_mse = np.asarray(full["training"]["mse"]["weights"])
    w_decision = np.asarray(full["training"]["decision"]["weights"])
    print(f"loaded trained weights from dfl_full.json  (mse={w_mse}, decision={w_decision})")

    print(f"\nbuilding {len(TEST_SEEDS)} held-out seeds ...")
    test_data = [build_seed(s) for s in TEST_SEEDS]
    print(f"  done in {time.time() - t0:.1f}s")

    # `poses`/`omega` come out of `build_case` identical for every seed in the hybrid family:
    # `_plans_hybrid` (ranking.py) never reads its `rng` argument, so the K plans' wheel-speed
    # profiles are a pure function of the family's constants, not of the seed. Verified by
    # reading the source rather than assumed -- confirmed once here, reused for all 80 seeds.
    scene0, _, _, _, _, poses, omega, grid = build_case(TEST_SEEDS[0], FAMILY, NOISE)

    # Verify the premise behind "perturb using sigma; observed cells stay put": sigma is the
    # OBSERVATION uncertainty (sensor + localisation) on observed cells, which under noise="all"
    # is small but not exactly zero -- report the real numbers rather than assume.
    sig_obs = np.concatenate([_sigma_field(s)[s.observed] for s in test_data])
    sig_unobs = np.concatenate([_sigma_field(s)[s.unobs] for s in test_data])
    sigma_check = {
        "observed_mean": float(sig_obs.mean()),
        "observed_max": float(sig_obs.max()),
        "unobserved_mean": float(sig_unobs.mean()),
        "unobserved_max": float(sig_unobs.max()),
    }
    print(
        f"\nsigma check: observed cells mean {sigma_check['observed_mean']:.4f} "
        f"max {sigma_check['observed_max']:.4f}  |  unobserved mean "
        f"{sigma_check['unobserved_mean']:.4f} max {sigma_check['unobserved_max']:.4f}"
    )

    methods_point = {"mse": lambda s: model_fill(s, w_mse), "decision": lambda s: model_fill(s, w_decision)}

    print("\nSTEP arms (reusing each seed's own cached batch=16 harness) ...")
    step_out: dict[str, dict[str, list[float]]] = {
        name: {"regret": [], "tau": []} for name in ("mse_step", "decision_step")
    }
    for s in test_data:
        for tag, fill_fn in methods_point.items():
            pick, est = choice_step(s, fill_fn(s), grid)
            step_out[f"{tag}_step"]["regret"].append(float(s.j_true[pick] - s.j_true.min()))
            step_out[f"{tag}_step"]["tau"].append(kendall_tau(est, s.j_true))

    print(f"\nCVaR arms: building ONE shared batch={m_draws} harness (geometry/friction are "
          "seed-independent; elevation is overwritten by every draw) ...")
    poses_d = np.tile(poses[0], (m_draws, 1)).astype(np.float32)
    omega_d = np.zeros((omega.shape[0], m_draws, 3), np.float32)
    hd = Harness(scene0, poses_d, omega_d, device="cuda")
    draws = NoiseDraws((m_draws, *scene0.shape), CELL, CORR_LEN, hd.device)
    print(f"  built in {time.time() - t0:.1f}s elapsed; running {len(test_data)} seeds x "
          f"{N_PLANS} plans x 2 maps ...")

    cvar_out: dict[str, dict[str, list[float]]] = {
        name: {"regret": [], "tau": []} for name in ("mse_cvar", "decision_cvar")
    }
    for i, (seed, s) in enumerate(zip(TEST_SEEDS, test_data)):
        # ONE draws-seed shared by mse_cvar and decision_cvar per test seed: the two conditions
        # see IDENTICAL noise realisations, so their comparison is paired on the draws too.
        cvar_seed = 950_000 + seed
        for tag, fill_fn in methods_point.items():
            pick, cvar = choice_cvar(s, fill_fn(s), hd, draws, poses, omega, cvar_seed)
            cvar_out[f"{tag}_cvar"]["regret"].append(float(s.j_true[pick] - s.j_true.min()))
            cvar_out[f"{tag}_cvar"]["tau"].append(kendall_tau(cvar, s.j_true))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(test_data)} seeds  ({time.time() - t0:.1f}s elapsed)", flush=True)
    del hd, draws

    eval_out = {**step_out, **cvar_out}
    plain_mse_regret = np.asarray(full["eval"]["mse"]["regret"])
    plain_decision_regret = np.asarray(full["eval"]["decision"]["regret"])

    print(f"\n{'method':<16}{'regret':>10}{'tau':>10}   (dfl_full.json plain arms, for reference)")
    print(f"{'mse (plain)':<16}{plain_mse_regret.mean():>10.4f}{np.mean(full['eval']['mse']['tau']):>+10.3f}")
    print(
        f"{'decision (plain)':<16}{plain_decision_regret.mean():>10.4f}"
        f"{np.mean(full['eval']['decision']['tau']):>+10.3f}"
    )
    for name in ("mse_step", "mse_cvar", "decision_step", "decision_cvar"):
        r = eval_out[name]
        print(f"{name:<16}{np.mean(r['regret']):>10.4f}{np.mean(r['tau']):>+10.3f}")

    # Paired sign tests, per the coordinator's request. Convention (matching dfl.py's own):
    # d = regret(reference) - regret(candidate); positive => candidate has LOWER regret, i.e.
    # the candidate wins. Stated in prose alongside each number to avoid sign ambiguity.
    pairs = (
        ("mse_cvar", "decision (plain)", np.asarray(eval_out["mse_cvar"]["regret"]), plain_decision_regret),
        ("mse_step", "decision (plain)", np.asarray(eval_out["mse_step"]["regret"]), plain_decision_regret),
        (
            "decision_cvar",
            "decision (plain)",
            np.asarray(eval_out["decision_cvar"]["regret"]),
            plain_decision_regret,
        ),
        ("mse_cvar", "mse (plain)", np.asarray(eval_out["mse_cvar"]["regret"]), plain_mse_regret),
    )
    print("\npaired sign tests on regret (candidate vs reference; positive = candidate wins):")
    paired_report = {}
    for candidate_name, reference_name, candidate_regret, reference_regret in pairs:
        d = reference_regret - candidate_regret
        n, k, p = sign_test(d)
        key = f"{candidate_name}_vs_{reference_name.split()[0]}"
        paired_report[key] = {
            "candidate": candidate_name,
            "reference": reference_name,
            "mean_improvement": float(d.mean()),
            "median_improvement": float(np.median(d)),
            "n_nonzero": n,
            "wins": k,
            "p": p,
        }
        print(
            f"  {candidate_name:<14} vs {reference_name:<17} mean {d.mean():+.4f}  "
            f"median {np.median(d):+.4f}  wins {k}/{n} (ties dropped)  p={p:.3g}"
        )

    out = {
        "family": FAMILY,
        "noise": NOISE,
        "m_draws": m_draws,
        "n_test": len(test_data),
        "sigma_check": sigma_check,
        "eval": eval_out,
        "plain_reference": {
            "mse": {"regret": plain_mse_regret.tolist(), "tau": full["eval"]["mse"]["tau"]},
            "decision": {"regret": plain_decision_regret.tolist(), "tau": full["eval"]["decision"]["tau"]},
        },
        "paired": paired_report,
        "seconds_total": time.time() - t0,
    }
    (OUT / "dfl_ablation.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'dfl_ablation.json'}  ({out['seconds_total']:.1f}s total)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m-draws", type=int, default=M_DRAWS)
    a = ap.parse_args()
    main(a.m_draws)
