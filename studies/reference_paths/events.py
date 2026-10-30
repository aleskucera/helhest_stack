"""Group weird frames (scores.json) into events: runs of frames with the same complaint, merged
over gaps < 0.5 s. Prints each event with its bag time and where the robot was."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CHECKS = {
    "loop": lambda f: not f["empty"] and f["turn_abs_deg"] > 300,
    "s_bend": lambda f: not f["empty"] and f["s_bends"] >= 1,
    "empty": lambda f: f["empty"],
    "jump": lambda f: f.get("flick_max", 0) > 0.75,
    "pivot": lambda f: abs(f.get("start_err_deg", 0)) > 45,
    "kmax": lambda f: not f["empty"] and f["kmax"] > 1.3,
}


def main() -> None:
    res = json.loads(Path(sys.argv[1]).read_text())
    for r in res:
        print(f"== {r['bag']}")
        fr = r["frames"]
        for name, cond in CHECKS.items():
            ev = []
            for f in fr:
                if cond(f):
                    if ev and f["t"] - ev[-1][1] < 0.5:
                        ev[-1][1] = f["t"]
                        ev[-1][2] += 1
                    else:
                        ev.append([f["t"], f["t"], 1, f["x"], f["y"], f["v"]])
            ev = [e for e in ev if e[2] >= 2]
            if ev:
                s = ", ".join(f"{e[0]:.1f}-{e[1]:.1f}s({e[2]}) v{e[5]:.1f}" for e in ev[:12])
                print(f"  {name:7s} {len(ev):3d}: {s}")


if __name__ == "__main__":
    main()
