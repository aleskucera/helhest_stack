"""Path-selection fans rerun with the C-FOSM arm (post hoc, 2026-09-14).

The frozen held-out artifacts store only each arm's pick outcome per fan
(regret, top-1), not the candidates' moments, so C-FOSM (the fold's mean with
linearization's sd; the "hybrid" of v2_posthoc_review) cannot be ranked from
them. This reruns `v2_fan.fan` on the same windows with the same seeds
(candidates seed 17, referee seed 18, 8,000 shared terrain draws) and one arm
added; every pre-registered arm's regret and top-1 must reproduce the
artifact to float precision, which the script asserts per window.

    PYTHONPATH=studies_v2 BASEPROD_ROOT=... python -m baseprod.v2_fan_cfosm <windows_root> <artifact_dir> <out_dir> [--workers N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from math import comb
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from spires.risk_calibration import SIGMA_FLOOR, Window, build_nodes
from .score import cell_covariance, cell_variance_split
from .v2_arms import arm_clark, arm_clark_diag, arm_fosm, arm_mean_map, arm_step_form
from .v2_fan import ATT_GATE_RAD, rollout, sample_actions
from .v2_moments import gaussian_cvar
from .v2_score import _attitude_G

FAN = {"K": 32, "T": 3.0, "v_max": 1.0, "w_max": 1.2, "n_ref": 8000}   # v2_runner.FAN
SEED = 17
ARMS_FROZEN = ("clark-corr", "clark", "clark-diag", "fosm", "step-form", "mean-map", "mc-2", "mc-32")


def fan_with_cfosm(win, K, T, v_max, w_max, seed, n_ref):
    """`v2_fan.fan` with `correction=None` (as the runner called it) plus the C-FOSM arm.
    The candidate, gate, referee and arm code is copied verbatim so the seeds line up."""
    rng = np.random.default_rng(seed)
    entry = win.gt_poses[0]
    yaw0 = np.arctan2(entry[1, 0], entry[0, 0])
    pose0 = (float(entry[0, 3]), float(entry[1, 3]), float(yaw0))
    vs, ws_ = sample_actions(rng, K, T, v_max, w_max)
    mu_f, sg_f, cnt_f = win.mu.ravel(), win.sigma.ravel(), win.count.ravel()
    usable = (np.isfinite(mu_f) & np.isfinite(sg_f) & (sg_f > SIGMA_FLOOR) & (cnt_f > 0))
    vi, vg, vl = cell_variance_split(win)
    cands = []
    n_att_gated = n_obs_gated = 0
    for k in range(K):
        track = rollout(pose0, vs[k], ws_[k], T)
        if track is None:
            n_obs_gated += 1
            continue
        try:
            cell_flat, caps, c_eff = build_nodes(win, track)
        except (IndexError, ValueError):
            n_obs_gated += 1
            continue
        t = len(track)
        node_ok = usable[cell_flat].all(axis=1).reshape(3 * t, 4).all(axis=1)
        if not node_ok.all():
            n_obs_gated += 1
            continue
        u_cells, inv = np.unique(cell_flat.ravel(), return_inverse=True)
        u_idx = inv.reshape(cell_flat.shape)
        means = mu_f[u_cells][u_idx] + caps
        G = _attitude_G(c_eff, t)
        e0 = means.max(axis=1)
        x0 = G @ e0
        if np.abs(x0).max() > ATT_GATE_RAD:
            n_att_gated += 1
            continue
        cands.append(dict(track=track, cell_flat=cell_flat, caps=caps, c_eff=c_eff,
                          u_cells=u_cells, u_idx=u_idx, means=means, G=G, t=t))
    rec = {"K": K, "n_att_gated": n_att_gated, "n_obs_gated": n_obs_gated, "n_valid": len(cands)}
    if len(cands) < 4:
        rec["scoreable"] = False
        return rec
    rec["scoreable"] = True
    all_cells = np.unique(np.concatenate([c["u_cells"] for c in cands]))
    pos = {c: i for i, c in enumerate(all_cells)}
    C_un = cell_covariance(all_cells, win, vi, vg, vl)
    L = np.linalg.cholesky(C_un + 1e-12 * np.eye(len(C_un)))
    mu_un = mu_f[all_cells]
    cvar_mc = np.zeros(len(cands))
    chunk = 2048
    samples = [np.empty(n_ref) for _ in cands]
    done = 0
    rng2 = np.random.default_rng(seed + 1)
    while done < n_ref:
        b = min(chunk, n_ref - done)
        h = mu_un + rng2.standard_normal((b, len(mu_un))) @ L.T
        for ci, c in enumerate(cands):
            hc = h[:, [pos[x] for x in c["u_cells"]]]
            sup = (hc[:, c["u_idx"]] + c["caps"]).max(axis=2)
            x = sup @ c["G"].T
            samples[ci][done:done + b] = np.einsum("ij,ij->i", x, x)
        done += b
    q = 0.9
    for ci in range(len(cands)):
        srt = np.sort(samples[ci])
        cvar_mc[ci] = srt[int(np.ceil(q * n_ref)):].mean()
    best = float(cvar_mc.min())
    arm_cvar = {a: np.zeros(len(cands)) for a in ARMS_FROZEN + ("c-fosm",)}
    for ci, c in enumerate(cands):
        mu_all = c["means"].ravel()
        Cc = C_un[np.ix_([pos[x] for x in c["u_cells"]], [pos[x] for x in c["u_cells"]])]
        Cflat = Cc[c["u_idx"].ravel()][:, c["u_idx"].ravel()]
        nc = c["means"].shape[1]
        sl = [np.arange(nc * i, nc * (i + 1)) for i in range(c["means"].shape[0])]
        G = c["G"]
        E_c, sd_c = arm_clark(mu_all, Cflat, sl, G)
        arm_cvar["clark"][ci] = gaussian_cvar(E_c, sd_c)
        arm_cvar["clark-corr"][ci] = gaussian_cvar(E_c, sd_c * 1.0)     # correction=None in the runner's fan call
        E, sd = arm_clark_diag(mu_all, Cflat, sl, G)
        arm_cvar["clark-diag"][ci] = gaussian_cvar(E, sd)
        E_f, sd_f = arm_fosm(mu_all, Cflat, sl, G)
        arm_cvar["fosm"][ci] = gaussian_cvar(E_f, sd_f)
        arm_cvar["c-fosm"][ci] = gaussian_cvar(E_c, sd_f)                # the added arm
        E, sd = arm_step_form(mu_all, Cflat, sl, G)
        arm_cvar["step-form"][ci] = gaussian_cvar(E, sd)
        E, _ = arm_mean_map(mu_all, sl, G)
        arm_cvar["mean-map"][ci] = E
        for nN, nm in ((2, "mc-2"), (32, "mc-32")):
            sub = samples[ci][rng2.integers(0, n_ref, nN)]
            arm_cvar[nm][ci] = gaussian_cvar(float(sub.mean()), float(sub.std(ddof=1)) if nN > 1 else 0.0)
    rec["regret"] = {}
    rec["top1"] = {}
    for a, cv in arm_cvar.items():
        pick = int(np.argmin(cv))
        rec["regret"][a] = float(cvar_mc[pick] - best)
        rec["top1"][a] = bool(cvar_mc[pick] == best)
    rec["cvar_mc_spread"] = float(cvar_mc.max() - best)
    return rec


def load_window(wp: Path) -> Window:
    d = np.load(wp)
    return Window(path=wp, mu=d["mu"].astype(np.float64), sigma=d["sigma"].astype(np.float64),
                  raw_sd=d["raw_sd"].astype(np.float64), meas_sd=d["meas_sd"].astype(np.float64),
                  count=d["count"], x0=float(d["xmin"]), y0=float(d["ymin"]), cell=float(d["cell"]),
                  sx=float(d["sx"]), sy=float(d["sy"]), gt_poses=d["gt_poses"])


def _job(args):
    wp, cond, name, frozen = args
    rec = fan_with_cfosm(load_window(wp), FAN["K"], FAN["T"], FAN["v_max"], FAN["w_max"], SEED, FAN["n_ref"])
    ok = True
    if rec.get("scoreable") and frozen.get("scoreable"):
        for a in ARMS_FROZEN:
            if rec["top1"][a] != frozen["top1"][a] or abs(rec["regret"][a] - frozen["regret"][a]) > 1e-9:
                ok = False
    print(f"[{time.strftime('%T')}] {cond} {name} " + ("reproduces" if ok else "MISMATCH") +
          (f" c-fosm top1={rec['top1']['c-fosm']}" if rec.get("scoreable") else " unscoreable"), flush=True)
    return {"condition": cond, "name": name, "fan": rec, "reproduces_frozen": ok}


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    pk = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pk if p <= pk[k] * (1.0 + 1e-9)))


def summarize(rows):
    out = {}
    for cond in ("foresight", "hindsight"):
        rc = [r for r in rows if r["condition"] == cond and r["fan"].get("scoreable")]
        arms = ARMS_FROZEN + ("c-fosm",)
        d = {"n_fans": len(rc), "n_reproduce_frozen": int(sum(r["reproduces_frozen"] for r in rc)),
             "top1": {a: int(sum(r["fan"]["top1"][a] for r in rc)) for a in arms},
             "regret_median": {a: float(np.median([r["fan"]["regret"][a] for r in rc])) for a in arms},
             "regret_p95": {a: float(np.quantile([r["fan"]["regret"][a] for r in rc], 0.95, method="lower")) for a in arms},
             "paired_mcnemar": {}}
        ours = [r["fan"]["top1"]["c-fosm"] for r in rc]
        for other in ("fosm", "mc-32", "mean-map", "clark-corr"):
            th = [r["fan"]["top1"][other] for r in rc]
            n10 = sum(a and not b for a, b in zip(ours, th)); n01 = sum(b and not a for a, b in zip(ours, th))
            d["paired_mcnemar"][f"c-fosm_vs_{other}"] = {"ours_only": n10, "theirs_only": n01,
                                                         "mcnemar_exact_p": binom_two_sided(min(n10, n01), n10 + n01)}
        out[cond] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows_root")
    ap.add_argument("artifact_dir", help="directory holding heldout_score_{foresight,hindsight}.json")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    root, art, out = Path(a.windows_root), Path(a.artifact_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for cond in ("foresight", "hindsight"):
        rows = json.loads((art / f"heldout_score_{cond}.json").read_text())
        for r in rows:
            if not (r.get("scoreable") and not r.get("flagged") and r.get("fan", {}).get("scoreable")):
                continue
            trav, wname = r["name"].split("/")
            wp = root / f"{cond}_v2" / trav / f"{wname}.npz"
            if not wp.exists():
                print("missing", wp, flush=True); continue
            jobs.append((wp, cond, r["name"], r["fan"]))
    rows_path = out / "fan_cfosm_rows.jsonl"
    done = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[(r["condition"], r["name"])] = r
    jobs = [j for j in jobs if (j[1], j[2]) not in done]
    print(f"{len(jobs)} fans to score ({len(done)} already done), {a.workers} workers", flush=True)
    rows = list(done.values())
    with Pool(a.workers, maxtasksperchild=8) as pool, rows_path.open("a") as fh:
        for r in pool.imap_unordered(_job, jobs, chunksize=1):
            fh.write(json.dumps(r) + "\n"); fh.flush(); rows.append(r)
    s = summarize(rows)
    s["_meta"] = {"written": time.strftime("%F %T"), "fan": FAN, "seed": SEED, "windows_root": str(root),
                  "note": "post hoc; v2 fans rerun with the C-FOSM arm (fold mean, linearization sd); same seeds as the frozen pass, pre-registered arms asserted to reproduce"}
    (out / "fan_cfosm_summary.json").write_text(json.dumps(s, indent=1))
    for cond, d in s.items():
        if cond.startswith("_"):
            continue
        print(f"\n== {cond}: {d['n_fans']} fans, {d['n_reproduce_frozen']} reproduce the frozen arms")
        print("  top1:", d["top1"]); print("  paired:", d["paired_mcnemar"])
    print("wrote", out)


if __name__ == "__main__":
    main()
