#!/usr/bin/env python3
"""Compare elevation_node replay logs produced by replay_ab.sh.

Reads the per-frame `F<n> carved=<c>/<m> map points` lines that `debug_frames` emits and
reports how much of the accumulated map each config carved away -- the map-erosion metric.
Frame counts are printed too: a config that processed fewer frames saw less data, so the
comparison is only valid when they match.

Usage:  python3 replay_report.py /tmp/odin_replay/default.log /tmp/odin_replay/matched.log
"""

from __future__ import annotations

import re
import sys

import numpy as np

_CARVE = re.compile(r"F(\d+) carved=(\d+)/(\d+) map points")


def parse(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    frames, carved, mapsz = [], [], []
    warnings = 0
    with open(path) as fh:
        for line in fh:
            m = _CARVE.search(line)
            if m:
                frames.append(int(m.group(1)))
                carved.append(int(m.group(2)))
                mapsz.append(int(m.group(3)))
            elif "WARN" in line or "warning" in line.lower():
                warnings += 1
    return np.array(frames), np.array(carved), np.array(mapsz), warnings


def report(path: str) -> dict[str, float]:
    f, c, m, warns = parse(path)
    name = path.rstrip("/").split("/")[-1].removesuffix(".log")
    if len(f) == 0:
        print(f"{name:<14} NO carve frames -- did the node process any clouds?")
        return {}
    # Fraction of the standing map removed per frame, and how big the map got.
    frac = np.divide(c, np.maximum(m, 1), dtype=np.float64)
    row = {
        "frames": len(f),
        "peak_map": int(m.max()),
        "final_map": int(m[-1]),
        "carved_total": int(c.sum()),
        "carve_frac_p50": float(np.percentile(frac, 50)),
        "carve_frac_p95": float(np.percentile(frac, 95)),
        "warnings": warns,
    }
    print(
        f"{name:<14} frames {row['frames']:>5}  peak_map {row['peak_map']:>7}  "
        f"final_map {row['final_map']:>7}  carved {row['carved_total']:>9}  "
        f"carve/frame p50 {row['carve_frac_p50'] * 100:5.2f}%  p95 {row['carve_frac_p95'] * 100:5.2f}%  "
        f"warns {row['warnings']}"
    )
    return row


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    rows = {}
    for p in sys.argv[1:]:
        r = report(p)
        if r:
            rows[p.rstrip("/").split("/")[-1].removesuffix(".log")] = r
    if len(rows) >= 2:
        names = list(rows)
        base = rows[names[0]]
        print(f"\nvs {names[0]}:")
        for n in names[1:]:
            r = rows[n]
            if r["frames"] != base["frames"]:
                print(f"  {n}: FRAME COUNT DIFFERS ({r['frames']} vs {base['frames']}) "
                      "-- rerun slower; the comparison is not valid")
            d_map = (r["final_map"] - base["final_map"]) / max(base["final_map"], 1) * 100
            d_carve = (r["carved_total"] - base["carved_total"]) / max(base["carved_total"], 1) * 100
            print(f"  {n}: final map {d_map:+.1f}%   total carved {d_carve:+.1f}%"
                  "   (bigger map + less carving = less erosion)")


if __name__ == "__main__":
    main()
