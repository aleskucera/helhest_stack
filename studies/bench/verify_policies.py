"""Phase-3 gates: does each policy actually do what its name says?

    .venv/bin/python -m studies.bench.verify_policies

A comparison between policies is only meaningful if each one behaves as advertised. Two ways
this silently fails, both fatal to the result:

  A STRAW-MAN BASELINE. If `entropy` does not in fact chase the decoy, then beating it proves
  nothing about decision-focused sensing -- it just proves the baseline was badly implemented.
  Section 8 of the plan says to contrast explicitly with information-theoretic active
  perception, which obliges us to give it its best shot.
  A CHEATING POLICY. If `attribution` aims at the gap because something leaked ground truth
  rather than because the plan's sensitivity points there, the whole benchmark is circular.

So this measures, at a mid-approach vantage where the robot has already met the barrier:

  1. entropy aims AWAY from the plan corridor, toward the open decoy side.
  2. attribution aims ALONG the plan corridor.
  3. attribution puts ~no weight on the decoy -- the FOSM contribution there must be
     negligible, which is the quantitative form of "high entropy is not high relevance".
  4. the two policies genuinely DISAGREE, by more than a look cone.
  5. cvar (the strong baseline) lands near attribution -- it reaches the same place by
     sampling, so if it did not agree, one of the two would be wrong.
"""

from __future__ import annotations

import numpy as np

from . import policies as P
from . import world as W
from .loop import Belief
from .verify_world import scan

N_SEEDS = 8
VANTAGE_X = W.WALL_X - 2.0  # where the robot has met the barrier but not found the gap


def believe_at(bw: W.BenchWorld, upto_x: float, n: int = 24) -> Belief:
    known = np.zeros(bw.scene.H.shape, bool)
    for x in np.linspace(bw.start[0], upto_x, n):
        known |= scan(bw, (float(x), 0.0, bw.approach_yaw), W.DEFAULT_FOV, W.DEFAULT_RANGE)
    return Belief(np.where(known, bw.scene.H, 0.0), known, bw.scene.x0, bw.scene.y0, bw.scene.cell)


def main() -> None:
    print(
        f"{'seed':>5}{'gap brg':>9}{'decoy brg':>11}{'entropy':>9}{'attrib':>9}{'cvar':>8}"
        f"{'sigma':>8}{'attr-gap':>10}{'ent-decoy':>11}{'decoy share':>13}"
    )
    rows = []
    for seed in range(N_SEEDS):
        bw = W.build(seed)
        belief = believe_at(bw, VANTAGE_X)
        pose = (VANTAGE_X, 0.0, bw.approach_yaw)

        b_gap = W.bearing_to(bw, bw.gap_mask, pose[:2])
        b_dec = W.bearing_to(bw, bw.decoy_mask, pose[:2])
        chosen = {
            k: P.POLICIES[k](belief, pose, bw) for k in ("entropy", "attribution", "cvar", "sigma")
        }

        def err(b, ref):
            return float(np.degrees(abs(np.arctan2(np.sin(b - ref), np.cos(b - ref)))))

        attr_gap = err(chosen["attribution"], b_gap)
        ent_decoy = err(chosen["entropy"], b_dec)

        # (3) how much of the plan's FOSM variance does attribution place on the decoy?
        contrib = (P._sensitivity(belief, bw, pose) * belief.sigma()) ** 2
        share = float(contrib[bw.decoy_mask].sum() / max(contrib.sum(), 1e-30))

        print(
            f"{seed:>5}{np.degrees(b_gap):>9.0f}{np.degrees(b_dec):>11.0f}"
            f"{np.degrees(chosen['entropy']):>9.0f}{np.degrees(chosen['attribution']):>9.0f}"
            f"{np.degrees(chosen['cvar']):>8.0f}{np.degrees(chosen['sigma']):>8.0f}"
            f"{attr_gap:>10.0f}{ent_decoy:>11.0f}{share:>12.4%}"
        )
        rows.append(
            (
                attr_gap,
                ent_decoy,
                share,
                err(chosen["entropy"], chosen["attribution"]),
                err(chosen["cvar"], chosen["attribution"]),
            )
        )

    ag, ed, share, disagree, cvar_gap = (np.array(c) for c in zip(*rows))
    print("\ngates")
    checks = [
        (
            "1 entropy chases the decoy",
            f"median {np.median(ed):.0f} deg from the decoy bearing "
            f"(within a {W.LOOK_FOV:.0f} deg cone)",
            np.median(ed) < W.LOOK_FOV / 2,
        ),
        (
            "2 attribution aims at the gap",
            f"median {np.median(ag):.0f} deg from the gap bearing",
            np.median(ag) < W.LOOK_FOV / 2,
        ),
        (
            "3 decoy is decision-irrelevant",
            f"it holds {np.median(share):.4%} of the plan's FOSM " f"variance",
            np.median(share) < 0.01,
        ),
        (
            "4 the policies disagree",
            f"median {np.median(disagree):.0f} deg apart, vs a " f"{W.LOOK_FOV:.0f} deg cone",
            np.median(disagree) > W.LOOK_FOV,
        ),
        (
            "5 cvar agrees with attribution",
            f"median {np.median(cvar_gap):.0f} deg apart",
            np.median(cvar_gap) < W.LOOK_FOV / 2,
        ),
    ]
    ok = True
    for name, detail, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<31} {detail}")
        ok &= bool(passed)
    print("\n" + ("POLICIES BEHAVE AS ADVERTISED" if ok else "POLICIES NOT USABLE -- fix first"))


if __name__ == "__main__":
    main()
