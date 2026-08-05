"""Phase-3 gates: does each policy actually do what its name says, IN THE LOOP?

    .venv/bin/python -m studies.bench.verify_policies

An earlier version scored the policies at a single static vantage. That was wrong twice over.
It could not exercise the traced-route sensitivity the real policies use, and worse, it passed
only because of an RNG imbalance in which side the gap fell on -- once the gap side was
stratified, both policies turned out to be near-indifferent between the two lateral extremes
at that vantage, and the "passing" gates had been luck. A behavioural claim has to be measured
where the behaviour happens.

So this runs SHORT real episodes and records what the looks actually revealed:

  1. entropy spends its looks on the decoy more than on the gap -- it must genuinely fall for
     the bait, or beating it proves nothing about decision-focused sensing.
  2. attribution spends its looks on the gap more than on the decoy -- the whole claim.
  3. attribution finds the gap EARLIER than the null baseline, which is the mechanism by which
     any time saving must arrive. If it does not, a time win would be coming from somewhere
     else and the explanation would be wrong.
  4. no policy reads ground truth: `world` is passed for grid geometry only. Checked by
     construction (see policies.py) and evidenced by entropy aiming at the decoy -- a cheating
     policy would not.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from . import loop as L
from . import policies as P
from . import world as W

N_SEEDS = 8
FRAMES = 220  # long enough to spend the look budget and meet the barrier


def main() -> None:
    wp.init()
    arms = ("none", "entropy", "sigma", "cvar", "attribution")
    stats = {a: {"gap": [], "decoy": [], "found": []} for a in arms}

    print(f"{'seed':>5}{'gap y':>7}  " + "".join(f"{a[:5]:>18}" for a in arms))
    print(f"{'':>5}{'':>7}  " + "".join(f"{'looks@gap/decoy':>18}" for _ in arms))
    for seed in range(N_SEEDS):
        bw = W.build(seed)
        line = f"{seed:>5}{bw.gap_y:>7.1f}  "
        for a in arms:
            tr = L.run(bw, policy=P.POLICIES[a], max_frames=FRAMES)
            stats[a]["gap"].append(tr.look_at_gap)
            stats[a]["decoy"].append(tr.look_at_decoy)
            stats[a]["found"].append(tr.gap_known_frame if tr.gap_known_frame >= 0 else FRAMES)
            line += f"{tr.look_at_gap:>8d}/{tr.look_at_decoy:<9d}"
        print(line, flush=True)

    print(f"\n{'policy':<13}{'mean @gap':>11}{'mean @decoy':>13}{'median gap-found frame':>24}")
    for a in arms:
        print(
            f"{a:<13}{np.mean(stats[a]['gap']):>11.2f}{np.mean(stats[a]['decoy']):>13.2f}"
            f"{np.median(stats[a]['found']):>24.0f}"
        )

    ent, att, nul = stats["entropy"], stats["attribution"], stats["none"]
    print("\ngates")
    checks = [
        (
            "1 entropy falls for the decoy",
            f"{np.mean(ent['decoy']):.2f} decoy looks vs {np.mean(ent['gap']):.2f} at the gap",
            np.mean(ent["decoy"]) > np.mean(ent["gap"]),
        ),
        (
            "2 attribution targets the gap",
            f"{np.mean(att['gap']):.2f} gap looks vs {np.mean(att['decoy']):.2f} at the decoy",
            np.mean(att["gap"]) > np.mean(att["decoy"]),
        ),
        (
            "3 attribution finds it earlier",
            f"median frame {np.median(att['found']):.0f} vs {np.median(nul['found']):.0f} "
            f"for the null baseline",
            np.median(att["found"]) < np.median(nul["found"]),
        ),
    ]
    ok = True
    for name, detail, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<31} {detail}")
        ok &= bool(passed)
    print("\n" + ("POLICIES BEHAVE AS ADVERTISED" if ok else "POLICIES NOT USABLE -- fix first"))


if __name__ == "__main__":
    main()
