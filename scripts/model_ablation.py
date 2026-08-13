"""Does the extra model fidelity change the DECISION, and what does it cost?

IMPROVEMENTS.md section 10, run on the branch that implemented sections 1-5.

The Tier-1/2 work was verified against closed forms (each certificate crosses where algebra says
it must) and against bags (yaw-rate RMS). Neither answers the question a planner cares about:
MPPI does not consume a trajectory, it consumes an ORDERING over candidates and executes the mean
of the elite. A model change that moves every trajectory by 10 cm but reorders nothing is free
fidelity with no decision content; one that reorders the elite changes what the robot does.

So: fix a scene, fix ONE candidate population, roll it out under each model level, cost it with
the real MPPI cost kernel and the real cost-to-go field, and compare

    tau_b            rank agreement over the whole population
    elite overlap    the top elite_frac that CEM actually averages
    d|u0|            the resulting first executed wheel command -- what reaches the motors
    endpoint         mean displacement of the same control sequence, for scale
    ms               rollout cost against the 100 ms control tick

The candidate population is generated HOST-side from a fixed seed and reused verbatim by every
level, including across git branches, so nothing in the comparison depends on device RNG order.
It mirrors the sampler in control/mppi.py: candidate 0 nominal, 25% from the wide uniform prior,
the rest a spline around the nominal plus per-step jitter.

Run once per branch/level set; each run writes a JSON keyed by the level names it could build.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import warp as wp

from helhest import dynamics
from helhest import worlds as W
from helhest.control.mppi import _cost_kernel
from helhest.control.mppi import CostParams
from helhest.control.mppi import MppiGpu
from helhest.control.mppi import SamplingConfig
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planning.costtogo import CostToGo


def candidates(B: int, T: int, seed: int, cfg: SamplingConfig, nominal: float) -> np.ndarray:
    """[T, B, 3] wheel-speed sequences, the host mirror of _sample_target_wheel_omega_kernel."""
    rng = np.random.default_rng(seed)
    n_wide = int(cfg.wide_frac * B)
    # knot index per step and the interpolation fraction between them, as _knot_bracket does
    pos = np.linspace(0.0, cfg.n_knots - 1, T)
    lo = np.clip(np.floor(pos).astype(int), 0, cfg.n_knots - 1)
    hi = np.clip(lo + 1, 0, cfg.n_knots - 1)
    frac = (pos - lo)[:, None]

    u = np.empty((T, B, 2), np.float32)
    # wide prior: knots uniform over the whole forward-arc box, independent of the nominal
    kw = rng.uniform(cfg.wmin, cfg.wmax, size=(cfg.n_knots, n_wide, 2))
    u[:, :n_wide] = (1.0 - frac[..., None]) * kw[lo] + frac[..., None] * kw[hi]
    # narrow prior: a smooth spline offset around the nominal, plus light per-step jitter
    n_nar = B - n_wide
    kn = rng.normal(0.0, cfg.sigma_knot, size=(cfg.n_knots, n_nar, 2))
    u[:, n_wide:] = nominal + (1.0 - frac[..., None]) * kn[lo] + frac[..., None] * kn[hi]
    u[:, n_wide:] += rng.normal(0.0, cfg.sigma, size=(T, n_nar, 1))  # shared L/R, as the kernel
    u[:, 0] = nominal  # candidate 0 keeps the nominal
    np.clip(u, cfg.wmin, cfg.wmax, out=u)
    return np.concatenate([u, np.zeros((T, B, 1), np.float32)], axis=2).astype(np.float32)


def kendall_tau_b(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> float:
    """Exact tau-b, accumulated in row chunks (n=4096 is 8.4M pairs; the full sign matrix is not
    worth allocating). Ties are counted, which matters: saturated costs produce many."""
    n = len(a)
    conc = disc = ta = tb = 0
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        da = np.sign(a[sl, None] - a[None, :]).astype(np.int8)
        db = np.sign(b[sl, None] - b[None, :]).astype(np.int8)
        upper = np.arange(i, min(i + chunk, n))[:, None] < np.arange(n)[None, :]
        p = (da * db)[upper]
        conc += int((p > 0).sum())
        disc += int((p < 0).sum())
        ta += int(((da == 0) & (db != 0))[upper].sum())
        tb += int(((db == 0) & (da != 0))[upper].sum())
    denom = np.sqrt((conc + disc + ta) * (conc + disc + tb))
    return float((conc - disc) / denom) if denom > 0 else float("nan")


# Each rung is one change on top of the previous. `needs` names the knob that must exist for the
# rung to be meaningful; a branch without it simply does not have that level, and is skipped rather
# than silently scored as its own baseline.
TAU = getattr(dynamics, "MOTOR_TAU", 0.19)
LADDER = {
    "base": (dict(tau_motor=0.0), {}, None),
    "lag": (dict(tau_motor=TAU), {}, "tau_motor"),
    "cylinder": (dict(tau_motor=TAU), dict(wheel_width=0.10), "wheel_width"),
    "traction": (
        dict(tau_motor=TAU, shear_lk=12.0, body_momentum=True, rolling_resistance=0.09),
        {},
        "shear_lk",
    ),
    "all": (
        dict(tau_motor=TAU, shear_lk=12.0, body_momentum=True, rolling_resistance=0.09),
        dict(wheel_width=0.10),
        "shear_lk",
    ),
}


def _supported(cls, kw: dict) -> dict:
    have = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in kw.items() if k in have}



def rock_world(cell: float = 0.06, seed: int = 1):
    """Isolated rocks scattered off the wheel tracks -- the geometry section 4 is aimed at.

    A sphere of radius R dilates a rock 0.35 m SIDEWAYS, so a rock beside the track lifts and tilts
    the robot; a cylinder of half-tread 0.05 m reaches only 5 cm and the robot straddles it. Rocks
    are 0.15-0.35 m tall (comparable to the wheel radius, so they matter) and 0.2-0.4 m across, with
    clear lanes between them -- unlike the stress worlds, whose obstacles are EXTENDED walls that
    both structuring elements meet identically.
    """
    from helhest.heightmap import _grid
    from helhest.heightmap import Heightmap

    xlim, ylim = (-2.0, 14.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    rng = np.random.default_rng(seed)
    for _ in range(60):
        cx, cy = rng.uniform(0.0, 12.0), rng.uniform(-4.0, 4.0)
        hx, hy = rng.uniform(0.10, 0.20), rng.uniform(0.10, 0.20)
        H[(np.abs(XX - cx) <= hx) & (np.abs(YY - cy) <= hy)] = rng.uniform(0.15, 0.35)
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def scenarios(n_start: int = 4):
    """(name, scene, mu, goal, [start poses]) -- starts placed IN the terrain, not 10 m short of it.

    The stress worlds put their obstacles 4-14 m from the nominal start, but a T=25 rollout at
    dt=0.1 covers only 1.2-3.5 m, so a rollout launched from there never touches them and every
    model level scores identical flat ground. Starts are therefore spread along the route on
    ground the robot can actually stand on.
    """
    out = []
    for world in list(W.WORLDS) + ["rocks"]:
        if world == "rocks":
            scene, goal = rock_world(), np.array([12.0, 0.0])
        else:
            builder, _s, goal = W.WORLDS[world]
            scene, goal = builder(), np.asarray(goal, np.float64)
        mu = W.matching_friction(scene)
        H = scene.H
        # a start is usable if the robot's footprint has no obstacle in it: the settle must have a
        # solution, or the rollout scores an artefact rather than the terrain
        rad = int(round(0.8 / scene.cell))
        ny, nx = H.shape
        ys = np.arange(rad, ny - rad, max(rad // 2, 1))
        xs = np.arange(rad, nx - rad, max(rad // 2, 1))
        cand = []
        for iy in ys:
            for ix in xs:
                patch = H[iy - rad : iy + rad, ix - rad : ix + rad]
                if patch.max() < 0.08:
                    wx = scene.x0 + ix * scene.cell
                    wy = scene.y0 + iy * scene.cell
                    cand.append((wx, wy))
        cand = np.array(cand)
        if not len(cand):
            continue
        # spread along the route: bucket by distance-to-goal and take one from each bucket
        d = np.linalg.norm(cand - goal[None, :], axis=1)
        order = np.argsort(-d)
        picks = order[np.linspace(0, len(order) - 1, n_start + 2).astype(int)[1:-1]]
        starts = []
        for i in picks:
            wx, wy = cand[i]
            starts.append((float(wx), float(wy), float(np.arctan2(goal[1] - wy, goal[0] - wx))))
        out.append((world, scene, mu, goal, starts))
    return out


def build_level(name: str, scene, mu, B: int, T: int, device: str):
    """A ForwardSimulator for one rung, or None if this branch lacks the knob that defines it."""
    sp_kw, rp_kw, needs = LADDER[name]
    fields = {f.name for f in dataclasses.fields(SolverParams)} | {
        f.name for f in dataclasses.fields(RobotParams)
    }
    if needs is not None and needs not in fields:
        return None
    rp = RobotParams(**_supported(RobotParams, rp_kw))
    sp = SolverParams(
        dt=dynamics.DT,
        k_turn=dynamics.K_TURN,
        newton_iters=6,
        atol=1e-4,
        **_supported(SolverParams, sp_kw),
    )
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    sim = ForwardSimulator(rp, sp, grid, B, T, device)
    sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    sim.set_friction(mu)
    return sim


def score(sim, planner, U: np.ndarray, start) -> tuple[np.ndarray, np.ndarray]:
    """Per-candidate MPPI cost and the rollout endpoints, for one fixed control population."""
    controlled, _derived, _c, _r = sim.rollout(U, tuple(float(v) for v in start))
    wp.launch(
        _cost_kernel,
        sim.batch_size,
        inputs=[
            sim.controlled,
            sim.derived,
            sim.clearance,
            sim.residual,
            sim.target_wheel_omega,
            planner.goal,
            planner.lattice_grid,
            planner.lattice_field,
            planner.n_theta,
            planner.cw,
            sim.robot,
            sim.n_steps,
        ],
        outputs=[planner.J],
        device=sim.device,
    )
    return planner.J.numpy().astype(np.float64), controlled[-1, :, :2].astype(np.float64)


def elite_u0(J: np.ndarray, U: np.ndarray, frac: float) -> np.ndarray:
    """The first command CEM would execute: the elite mean of u[0] -- what reaches the wheels."""
    k = max(int(frac * len(J)), 1)
    elite = np.argpartition(J, k - 1)[:k]
    return U[0, elite, :2].mean(axis=0)


def timed_ms(sim, reps: int = 20) -> float:
    sim.rollout_launch()
    wp.synchronize_device(sim.device)
    t0 = time.perf_counter()
    for _ in range(reps):
        sim.rollout_launch()
    wp.synchronize_device(sim.device)
    return (time.perf_counter() - t0) / reps * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--n-theta", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--levels", default="base,lag,cylinder,traction,all")
    args = ap.parse_args()

    wp.init()
    cfg = SamplingConfig()
    U = candidates(args.batch, args.horizon, args.seed, cfg, nominal=1.5)
    blobs = {}
    out = {"levels": {}, "meta": {"B": args.batch, "T": args.horizon, "seed": args.seed}}

    for world, scene, mu, goal, starts in scenarios(args.starts):
        cgrid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
        ctg = CostToGo(
            cgrid,
            dynamics.robot_params(),
            dynamics.planning_solver(),
            n_theta=args.n_theta,
            device=args.device,
        )
        Hc = wp.array(
            np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=args.device
        )
        V = ctg.compute(Hc, (float(goal[0]), float(goal[1])))

        for name in args.levels.split(","):
            sim = build_level(name, scene, mu, args.batch, args.horizon, args.device)
            if sim is None:
                continue
            planner = MppiGpu(sim, CostParams(), cfg, n_theta=args.n_theta)
            planner.set_lattice(V, cgrid.build())
            planner.goal.assign(np.asarray(goal, np.float32))
            ms = timed_ms(sim)
            for i, start in enumerate(starts):
                J, end = score(sim, planner, U, start)
                key = f"{name}|{world}|{i}"
                blobs[key + "|J"] = J.astype(np.float32)
                blobs[key + "|end"] = end.astype(np.float32)
                out["levels"].setdefault(name, {})[f"{world}|{i}"] = {
                    "u0": elite_u0(J, U, cfg.elite_frac).tolist(),
                    "ms": ms,
                    "start": list(start),
                }
            print(f"  {world:>8} {name:>10}  {len(starts)} starts  {ms:6.3f} ms")
            del planner, sim

    np.savez_compressed(args.out.replace(".json", ".npz"), **blobs)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
