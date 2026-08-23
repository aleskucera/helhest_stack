"""Diagnosis: is clark_cylinder.json's negative (mc_mean - j_bel_settle) a cross-element
artifact? Decompose with a MATCHED cylinder belief settle via the same _terms_kernel path."""

import json
import numpy as np
import warp as wp
from studies.bench import matched_truth as mt
from studies.bench.clark import SETTLE_IDX
from studies.bench.ranking import build_case

wp.init()
d = json.load(open("studies/out/bench/clark_cylinder.json"))
rows = d["rows"]
per_seed = []
for r in rows:
    seed = r["seed"]
    scene, _t, _m, _o, sigma, poses, omega, grid = build_case(seed, "hybrid", "all")
    belief = scene.elevation.astype(np.float32)
    terms = mt.cylinder_mc_truth_terms_from_fields(
        scene, belief[None], poses, omega, device="cuda:0"
    )
    j_cyl = terms[SETTLE_IDX, 0, :]  # matched cylinder belief settle
    j_sph = np.array(r["calib"]["j_bel_settle"])  # stored: sphere DifferentiableSimulator
    mc = np.array(r["calib"]["mc_mean"])  # stored: cylinder MC truth mean
    ec = np.array(r["calib"]["e_clark"])  # stored: cylinder analytic Clark mean
    per_seed.append(
        {
            "seed": seed,
            "mc_minus_sph": float(np.mean(mc - j_sph)),
            "mc_minus_cyl": float(np.mean(mc - j_cyl)),
            "sph_minus_cyl": float(np.mean(j_sph - j_cyl)),
            "ec_minus_cyl": float(np.mean(ec - j_cyl)),
            "mc_minus_ec": float(np.mean(mc - ec)),
            "spread_cyl": float(np.max(j_cyl) - np.min(j_cyl)),
        }
    )
    if seed % 20 == 0:
        print(
            f"seed {seed:3d}: mc-sph {per_seed[-1]['mc_minus_sph']:+.3f}  "
            f"mc-cyl {per_seed[-1]['mc_minus_cyl']:+.3f}  "
            f"sph-cyl {per_seed[-1]['sph_minus_cyl']:+.3f}"
        )


def agg(k):
    v = np.array([p[k] for p in per_seed])
    return f"mean {v.mean():+.4f}  median {np.median(v):+.4f}"


print("\n=== pooled over 100 seeds x 16 plans ===")
for k in ["mc_minus_sph", "mc_minus_cyl", "sph_minus_cyl", "ec_minus_cyl", "mc_minus_ec"]:
    print(f"{k:14s} {agg(k)}")
sp = np.array([p["spread_cyl"] for p in per_seed])
mcyl = np.array([p["mc_minus_cyl"] for p in per_seed])
print(f"spread_cyl     median {np.median(sp):.4f}")
print(f"Jensen share of spread (matched): {np.mean(mcyl)/np.median(sp):+.1%}")
neg = int((mcyl <= 0).sum())
print(f"seeds with matched Jensen gap <= 0: {neg}/100")
json.dump(per_seed, open("studies/out/bench/diag_cyl_jensen.json", "w"), indent=1)
print("wrote studies/out/bench/diag_cyl_jensen.json")
