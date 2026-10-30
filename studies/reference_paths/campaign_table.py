"""Compare sim campaigns (campaign.sh output dirs): per world and configuration, how often the goal
was reached, the frames it took, wall contact, and the reference curve's own metrics.

    python campaign_table.py camp/C0_base camp/C1_div_taper ... [--refs] [--json out.json]
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

RE_REACH = re.compile(r"REACHED in (\d+) frames")
RE_FAIL = re.compile(r"did not reach in (\d+) frames.*closest ([\d.]+) m")
RE_WALL = re.compile(r"WALL CLEARANCE min ([-+\d.]+) m\s+frames touching a wall (\d+)")


def runs(d: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for log in sorted(d.glob("*_r*.log")):
        world, rep = log.stem.rsplit("_r", 1)
        txt = log.read_text(errors="ignore")
        r: dict = {"rep": rep}
        m = RE_REACH.search(txt)
        f = RE_FAIL.search(txt)
        if m:
            r.update(reached=True, frames=int(m.group(1)))
        elif f:
            r.update(reached=False, frames=int(f.group(1)), closest=float(f.group(2)))
        else:
            continue  # still running or crashed
        w = RE_WALL.search(txt)
        if w:
            r.update(clear_min=float(w.group(1)), touch=int(w.group(2)))
        out[world].append(r)
    return out


def ref_metrics(d: Path, world: str) -> dict:
    from replay_ref import score_recorded
    from replay_ref import summarize

    fr = []
    for rec in sorted(d.glob(f"{world}_r*_ref.npz")):
        fr += score_recorded(rec)
    return summarize(fr) if fr else {}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_refs = "--refs" in sys.argv
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out_json in args:
        args.remove(out_json)
    dirs = [Path(a) for a in args]
    table = {d.name: runs(d) for d in dirs}
    worlds = sorted({w for t in table.values() for w in t})
    names = [d.name for d in dirs]
    print(f"{'world':16s}" + "".join(f"{n[:22]:>24s}" for n in names))
    tot = {n: [0, 0, []] for n in names}
    for w in worlds:
        row = f"{w:16s}"
        for n in names:
            rs = table[n].get(w, [])
            if not rs:
                row += f"{'-':>24s}"
                continue
            ok = [r for r in rs if r["reached"]]
            fr = np.mean([r["frames"] for r in ok]) if ok else float("nan")
            touch = sum(r.get("touch", 0) for r in rs)
            cell = f"{len(ok)}/{len(rs)} {fr:6.0f}f t{touch}"
            row += f"{cell:>24s}"
            tot[n][0] += len(ok)
            tot[n][1] += len(rs)
            tot[n][2] += [r["frames"] for r in rs]
        print(row)
    print(
        f"{'TOTAL':16s}"
        + "".join(f"{f'{tot[n][0]}/{tot[n][1]} sum{sum(tot[n][2]):6d}f':>24s}" for n in names)
    )
    res = {"runs": table}
    if do_refs:
        keys = ("empty", "wall_neg", "loop", "jump", "excess", "stray", "detour_p95", "dev_p95")
        print("\nreference metrics (all frames of all reps)")
        print(f"{'world':16s}{'config':16s}" + "".join(f"{k:>11s}" for k in keys))
        res["refs"] = {}
        for w in worlds:
            for d in dirs:
                m = ref_metrics(d, w)
                res["refs"][f"{d.name}/{w}"] = m
                if m:
                    print(
                        f"{w:16s}{d.name[:15]:16s}"
                        + "".join(f"{m.get(k, float('nan')):11.2f}" for k in keys)
                    )
    if out_json:
        Path(out_json).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
