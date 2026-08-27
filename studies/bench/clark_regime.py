"""Where the analytic estimator's DECISION advantage lives, as a function of the terrain.

The estimator's calibration advantage is unconditional (Section III/IV). Its RANKING advantage is
not, and this benchmark measures the condition rather than asserting it.

MECHANISM. A decision is an arg-min over plans, so only the part of the Jensen uplift that
DIFFERS between plans can change it. The uplift per contact is a*phi(alpha) with
alpha = (mean gap between footprint candidates) / (sd of their difference). Terrain with
footprint-scale structure gives the candidates different means, so which one wins varies along a
route and between routes; flat ground under a large sigma makes every candidate contested in the
same way everywhere, and a uniform uplift cancels exactly in an arg-min.

So the governing quantity is sigma against FOOTPRINT-SCALE terrain relief, not sigma alone.

ARMS. All arms run on the deployed cylinder, including `fosm`/`bracket` and the `j_bel` behind
`none`/`step` (harness.py's wheel_width). A sphere-locked gradient arm is scored against a truth
its own geometry never produced and reads as much weaker than it is.

STATISTIC. Mean regret ratio with a bootstrap CI, not a paired sign test: with matched arms the
methods agree on most scenes (measured: 4-29 non-tied of 40), and the advantage is in margin size
on rare bad scenes rather than in win frequency, which is what a CVaR estimator should do and
what a sign test is close to blind to.

    python -m studies.bench.clark_regime --seeds 60
"""
from __future__ import annotations
import argparse, json
import numpy as np
import warp as wp
from helhest.engine import RobotParams
from .ranking import build_case, CELL, N_PLANS, OUT

def _cyl_harness():
    import studies.adjoint.harness as H
    HW = RobotParams().wheel_width
    class CylHarness(H.Harness):
        def __init__(self, *a, **kw):
            kw.setdefault("wheel_width", HW); super().__init__(*a, **kw)
    return CylHarness

def footprint_relief(b, n=7):
    """Terrain relief at the scale of one footprint: the sd of what a 0.7 m smooth removes."""
    k = np.ones(n) / n
    sm = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, b)
    sm = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, sm)
    return float((b - sm)[n:-n, n:-n].std())

def _boot(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed); m = len(a); o = np.empty(n)
    for i in range(n):
        j = rng.integers(0, m, m); o[i] = b[j].mean() / max(a[j].mean(), 1e-12)
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))

def run_regime(label, sigma_fn, terrain_fn, n_seeds, device):
    import studies.bench.ranking as R, studies.bench.clark as C
    if not hasattr(R, "_orig_build_case"):
        R._orig_build_case = R.build_case
    orig = R._orig_build_case
    def patched(seed, family="fan", noise="clean", flat_sigma=False):
        sc, t, m, o, sg, po, om, g = orig(seed, family, noise, flat_sigma)
        XX, YY = g
        if terrain_fn is not None:
            sc.elevation[:] = terrain_fn(np.asarray(sc.elevation), XX, YY, seed)
        if sigma_fn is not None:
            sg = sigma_fn(np.asarray(sg), XX, YY, seed)
        return sc, t, m, o, sg, po, om, g
    R.build_case = patched; C.build_case = patched; C.Harness = _cyl_harness()

    rows, relief, sig = [], [], []
    for s in range(n_seeds):
        sc, _t, _m, _o, sg, _p, _o2, _g = patched(s, "hybrid", "all")
        relief.append(footprint_relief(np.asarray(sc.elevation))); sig.append(float(np.median(sg)))
        rows.append(C.run_seed(s, "hybrid", "all", device, "cylinder"))
    g = lambda a: np.array([r["arms"][a]["regret"] for r in rows])
    cv = g("clark_cvar")
    rec = {"regime": label, "n_seeds": n_seeds,
           "footprint_relief": float(np.median(relief)), "sigma_median": float(np.median(sig)),
           "sigma_over_relief": float(np.median(sig) / max(np.median(relief), 1e-12)),
           "clark_cvar_regret": float(cv.mean())}
    for b in ("fosm", "step", "bracket", "none"):
        x = g(b); lo, hi = _boot(cv, x)
        rec[b] = {"mean_regret": float(x.mean()),
                  "ratio": float(x.mean() / max(cv.mean(), 1e-12)), "ci95": [lo, hi]}
    return rec

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    wp.init()

    def flatten(m):
        def fn(b, XX, YY, seed):
            k = np.ones(7)/7
            sm = np.apply_along_axis(lambda r: np.convolve(r,k,"same"),1,b)
            sm = np.apply_along_axis(lambda c: np.convolve(c,k,"same"),0,sm)
            return sm + m*(b - sm)
        return fn

    # A sweep in the governing quantity ONLY: sigma is untouched and the terrain's broad shape is
    # untouched; only footprint-scale relief is scaled, so nothing else can explain the trend.
    regimes = [("structured x2.0", None, flatten(2.0)),
               ("deployed benchmark", None, None),
               ("flattened x0.25", None, flatten(0.25)),
               ("flattened x0.06 (Oxford-like)", None, flatten(0.06))]
    out = []
    for label, sfn, tfn in regimes:
        r = run_regime(label, sfn, tfn, a.seeds, a.device)
        out.append(r)
        print("  %-30s relief %.4f  sigma/relief %6.1f | clark %.4f | fosm %.2fx [%.2f,%.2f]%s"
              % (label, r["footprint_relief"], r["sigma_over_relief"], r["clark_cvar_regret"],
                 r["fosm"]["ratio"], *r["fosm"]["ci95"],
                 " *" if r["fosm"]["ci95"][0] > 1 else ""), flush=True)
        (OUT / "clark_regime.json").write_text(json.dumps({"regimes": out}, indent=1))
    print(f"\nwrote {OUT / 'clark_regime.json'}")

if __name__ == "__main__":
    main()
