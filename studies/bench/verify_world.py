"""Phase-1 gates: is the benchmark world actually a benchmark?

    .venv/bin/python -m studies.bench.verify_world

Seven things must hold or the comparison is vacuous. Each can invalidate the experiment on its
own, so all are checked before a single policy runs.

  1. GAP HIDDEN BY DEFAULT. Default sensing must not reveal the opening until the robot has
     already committed to an approach. If it is visible from the start, every policy routes
     straight to it and no sensing decision is being tested.
  2. BARRIER IMPASSABLE. Far above the wheel radius, so it is a routing problem, not a bump.
  3. BARRIER BLOCKS THE ROUTE. The straight line to the goal must actually be closed.
  4. A LOOK CAN FIND THE GAP. The sensing action must be capable of revealing it from far away,
     or no policy could ever win and the comparison is between equals.
  5. DECOY UNREACHABLE. Beyond the widest possible wheel-envelope contact from the TRUE route
     (start -> gap -> goal), so its elevation cannot influence the plan's cost physically.
  6. DECOY ATTRACTS ENTROPY. An entropy-directed sensor must genuinely prefer it. Measured the
     way the POLICY must actually decide -- expected gain under its own BELIEF, not under
     ground truth, which it does not have. That distinction is not pedantic: under the belief
     unknown ground is assumed flat, so no self-occlusion is predicted and the decoy direction
     scores its full cone, while the partially-observed WALL is known and does occlude. Scored
     against ground truth instead, the gap direction wins for a reason the policy could not
     possibly know, and the baseline would be a straw man.
  7. LOOKS ARE EXCLUSIVE. One look cone must not cover both gap and decoy, or the policies
     never actually have to choose.
"""

from __future__ import annotations

import numpy as np

from helhest.perception.lidar import lidar_scan  # noqa: F401

from . import world as W

MOUNT = 0.4
WHEEL_RADIUS = 0.35
N_SEEDS = 12
COMMIT_FRAC = 0.45  # the gap must stay hidden past this fraction of the approach
LOOK_VANTAGE_X = 3.0  # [m] where gate 4 tests whether a look can reach the gap


def scan(bw: W.BenchWorld, pose, fov: float, rng_m: float) -> np.ndarray:
    _, known = lidar_scan(
        bw.scene.H,
        bw.scene.x0,
        bw.scene.y0,
        bw.scene.cell,
        pose,
        fov_deg=fov,
        max_range=rng_m,
        mount_height=MOUNT,
    )
    return known


def _hit(known: np.ndarray, mask: np.ndarray, frac: float) -> bool:
    return bool((known & mask).sum() > frac * mask.sum())


def believed_map(bw: W.BenchWorld, upto_x: float, n: int = 24):
    """What the robot believes after driving straight to `upto_x` with default sensing only.

    Returns (elev, known). Unknown cells read as flat 0.0 -- the same optimistic assumption the
    planner makes, so the policy's expected-gain estimate is consistent with its own map.
    """
    known = np.zeros(bw.scene.H.shape, bool)
    for x in np.linspace(bw.start[0], upto_x, n):
        known |= scan(bw, (float(x), 0.0, bw.approach_yaw), W.DEFAULT_FOV, W.DEFAULT_RANGE)
    return np.where(known, bw.scene.H, 0.0), known


def expected_gain(elev, known, bw: W.BenchWorld, frm, bearing: float) -> int:
    """Unknown cells a look at `bearing` is EXPECTED to reveal, ray-cast on the believed map."""
    _, vis = lidar_scan(
        elev,
        bw.scene.x0,
        bw.scene.y0,
        bw.scene.cell,
        (frm[0], frm[1], bearing),
        fov_deg=W.LOOK_FOV,
        max_range=W.LOOK_RANGE,
        mount_height=MOUNT,
    )
    return int((vis & ~known).sum())


def first_default_sighting(bw: W.BenchWorld, n: int = 80) -> float:
    """Along-route x at which DEFAULT sensing first sees the gap, driving straight."""
    for x in np.linspace(bw.start[0], W.WALL_X, n):
        if _hit(
            scan(bw, (float(x), 0.0, bw.approach_yaw), W.DEFAULT_FOV, W.DEFAULT_RANGE),
            bw.gap_mask,
            0.10,
        ):
            return float(x)
    return float("inf")


def main() -> None:
    rows = []
    print(
        f"{'seed':>5}{'gap y':>7}{'seen x':>8}{'commit':>8}{'blocked':>9}{'look finds':>12}"
        f"{'decoy d':>9}{'unk fwd':>9}{'unk decoy':>11}{'sep deg':>9}{'cross':>7}"
    )
    for seed in range(N_SEEDS):
        bw = W.build(seed)
        seen_x = first_default_sighting(bw)
        frac = min(seen_x / W.WALL_X, 1.0) if np.isfinite(seen_x) else 1.0

        # (3) is the straight line to the goal actually closed?
        XX, YY = W.cell_centres(bw)
        corridor = np.abs(YY) <= 0.8
        blocked = float(
            (bw.wall_mask & corridor).sum()
            / max((corridor & (np.abs(XX - W.WALL_X) <= W.WALL_HALF_DEPTH)).sum(), 1)
        )

        # From a vantage a little way along the approach, not from the start. The robot uses
        # its looks while driving, and with a 9 m look range the gap is simply out of reach at
        # frame 0 -- testing there measures the range, not the capability.
        here = (LOOK_VANTAGE_X, 0.0)
        b_gap = W.bearing_to(bw, bw.gap_mask, here)
        b_decoy = W.bearing_to(bw, bw.decoy_mask, here)
        look_gap = scan(bw, (here[0], here[1], b_gap), W.LOOK_FOV, W.LOOK_RANGE)
        look_dec = scan(bw, (here[0], here[1], b_decoy), W.LOOK_FOV, W.LOOK_RANGE)
        finds = _hit(look_gap, bw.gap_mask, 0.10)

        # Gate 6, belief-based: stand where the robot has already met the wall, and ask which
        # bearing its own map says is most informative.
        vantage = (W.WALL_X - 2.0, 0.0)
        b_elev, b_known = believed_map(bw, vantage[0])
        bg = W.bearing_to(bw, bw.gap_mask, vantage)
        bd = W.bearing_to(bw, bw.decoy_mask, vantage)
        unk_gap = expected_gain(b_elev, b_known, bw, vantage, bg)
        unk_dec = expected_gain(b_elev, b_known, bw, vantage, bd)
        sep = float(np.degrees(abs(b_decoy - b_gap)))
        cross = _hit(look_dec, bw.gap_mask, 0.10)

        dist = W.route_distance(bw)
        decoy_d = float(dist[bw.decoy_mask].min())

        print(
            f"{seed:>5}{bw.gap_y:>7.2f}{seen_x:>8.2f}{frac:>8.0%}{blocked:>9.0%}"
            f"{str(finds):>12}{decoy_d:>9.2f}{unk_gap:>9d}{unk_dec:>11d}{sep:>9.1f}{str(cross):>7}"
        )
        rows.append((frac, blocked, finds, decoy_d, unk_gap, unk_dec, sep, cross))

    frac, blocked, finds, dd, ug, ud, sep, cross = (np.array(c) for c in zip(*rows))
    print("\ngates")
    checks = [
        (
            "1 gap hidden by default",
            f"median first sighting at {np.median(frac):.0%} of the "
            f"approach (want > {COMMIT_FRAC:.0%})",
            np.median(frac) > COMMIT_FRAC,
        ),
        (
            "2 barrier impassable",
            f"{W.WALL_HEIGHT} m vs {WHEEL_RADIUS} m wheel radius",
            W.WALL_HEIGHT > 2 * WHEEL_RADIUS,
        ),
        (
            "3 straight route blocked",
            f"{np.median(blocked):.0%} of the on-axis barrier is solid",
            np.median(blocked) > 0.95,
        ),
        ("4 a look finds the gap", f"{finds.sum()}/{len(finds)} seeds", bool(finds.all())),
        (
            "5 decoy unreachable",
            f"min {dd.min():.2f} m from the true route = "
            f"{dd.min()/W.ENVELOPE_REACH:.1f}x the {W.ENVELOPE_REACH} m envelope reach",
            dd.min() > 2.0 * W.ENVELOPE_REACH,
        ),
        (
            "6 decoy attracts entropy",
            f"under its OWN belief the decoy bearing scores "
            f"{np.median(ud/np.maximum(ug,1)):.2f}x the gap bearing",
            np.median(ud / np.maximum(ug, 1)) > 1.1,
        ),
        (
            "7 looks are exclusive",
            f"separation {np.median(sep):.0f} deg vs {W.LOOK_FOV} deg cone; "
            f"decoy look finds the gap in {int(cross.sum())}/{len(cross)} seeds",
            not bool(cross.any()),
        ),
    ]
    ok = True
    for name, detail, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<26} {detail}")
        ok &= bool(passed)
    print("\n" + ("WORLD IS A VALID BENCHMARK" if ok else "WORLD NOT USABLE -- fix before Phase 2"))


if __name__ == "__main__":
    main()
