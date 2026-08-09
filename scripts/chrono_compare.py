"""Put the engine's statics next to Chrono's and score the PREREG_chrono.md predictions.

    python scripts/chrono_compare.py /tmp/engine_statics.json /tmp/chrono_sphere.json

The interesting quantity is not agreement. Predictions 3 and 4 say the two MUST disagree; what is
being tested is whether the disagreement has the size and the SIGN that the omitted tangential
contact reaction implies.

Pose is compared as perpendicular distance to the ground plane, not as z. Chrono's body is free to
translate along the plane while it settles, and on a slope a shift along x changes z without
changing the geometry at all -- so z is not a pose-invariant quantity here and comparing it
directly would report a disagreement that does not exist.
"""

from __future__ import annotations

import json
import math
import sys

WHEELS = ("left", "right", "rear")
RADIUS = 0.35


def load(path: str) -> dict:
    with open(path) as f:
        return {(r["axis"], r["tilt_deg"]): r for r in json.load(f)}


def normal(axis: str, deg: float) -> tuple[float, float, float]:
    """Ground normal for this case, matching both scripts' construction."""
    p = math.radians(deg if axis == "pitch" else 0.0)
    r = math.radians(deg if axis == "roll" else 0.0)
    return (math.sin(p) * math.cos(r), -math.sin(r), math.cos(p) * math.cos(r))


def perp(row: dict, axis: str, deg: float) -> float:
    n = normal(axis, deg)
    x, y = row.get("ref_xy", [0.0, 0.0])
    return x * n[0] + y * n[1] + row["z"] * n[2]


def main() -> None:
    eng, chr_ = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(k for k in eng if k in chr_)

    print("=== (1) pose: perpendicular distance to the plane, and tilt ===")
    print(f"{'case':>14}{'d_perp eng':>12}{'d_perp chr':>12}{'d (cm)':>9}"
          f"{'tilt eng':>10}{'tilt chr':>10}{'d (deg)':>9}")
    worst_d = worst_a = 0.0
    for k in keys:
        e, c = eng[k], chr_[k]
        de, dc = perp(e, *k), perp(c, *k)
        te = e["pitch_deg"] if k[0] == "pitch" else e["roll_deg"]
        tc = c["pitch_deg"] if k[0] == "pitch" else c["roll_deg"]
        worst_d, worst_a = max(worst_d, abs(de - dc)), max(worst_a, abs(te - tc))
        print(f"{f'{k[0]} {k[1]:.0f}':>14}{de:>12.4f}{dc:>12.4f}{(de - dc) * 100:>9.2f}"
              f"{te:>10.2f}{tc:>10.2f}{te - tc:>9.2f}")
    ok1 = worst_a < 0.5 and worst_d < 0.01
    print(f"  worst {worst_d * 100:.2f} cm, {worst_a:.3f} deg   (both should also equal the wheel "
          f"radius {RADIUS})  -> {'PASS' if ok1 else 'FAIL'}")

    print("\n=== (2,3,4) normal loads as a fraction of m g ===")
    print(f"{'case':>14}{'left e/c':>18}{'right e/c':>18}{'rear e/c':>18}"
          f"{'sum e/c':>18}{'cos(t)':>9}")
    for k in keys:
        e, c = eng[k], chr_[k]
        cells = "".join(f"{e['loads'][w]:>9.3f}{'/' + format(c['loads'][w], '.3f'):>9}"
                        for w in WHEELS)
        sums = f"{e['sum_loads']:>9.3f}{'/' + format(c['sum_loads'], '.3f'):>9}"
        print(f"{f'{k[0]} {k[1]:.0f}':>14}{cells}{sums}"
              f"{math.cos(math.radians(k[1])):>9.3f}")

    f_e, f_c = eng[("pitch", 0.0)], chr_[("pitch", 0.0)]
    d_flat = max(abs(f_e["loads"][w] - f_c["loads"][w]) for w in WHEELS)
    print(f"\n  (2) flat ground, worst wheel: {d_flat:.4f} of m g  -> "
          f"{'PASS' if d_flat < 0.02 else 'FAIL'}")

    e25, c25 = eng[("pitch", 25.0)], chr_[("pitch", 25.0)]
    emin, cmin = min(e25["loads"].values()), min(c25["loads"].values())
    ok3 = abs(emin - cmin) > 0.05 and cmin < emin
    print(f"  (3) least-loaded contact at 25 deg pitch: engine {emin:.4f} vs Chrono {cmin:.4f}, "
          f"gap {abs(emin - cmin):.4f} of m g  -> {'PASS' if ok3 else 'FAIL'}")
    ident = 0.2637 / math.cos(math.radians(25))
    print(f"      the engine's identity 0.2637/cos(t) predicts {ident:.4f} and it reports "
          f"{emin:.4f}: its margin RISES with tilt while the real one FALLS")

    ct = math.cos(math.radians(25.0))
    ok4 = abs(c25["sum_loads"] - ct) < 0.02
    print(f"  (4) load sum at 25 deg: engine {e25['sum_loads']:.4f} (its identity 1/cos = "
          f"{1 / ct:.4f}), Chrono {c25['sum_loads']:.4f} (m g cos t = {ct:.4f})  -> "
          f"{'PASS' if ok4 else 'FAIL'}")

    print("\n=== lateral transfer on a side slope: the sharpest form of (3) ===")
    print("  with a uniform plane every contact normal is parallel, so a normals-only balance")
    print("  splits the load by the CoM's barycentric weight -- and the CoM is on the centreline.")
    print("  The engine therefore CANNOT produce any left/right transfer, at any bank angle.")
    print(f"{'roll':>7}{'engine L-R':>13}{'Chrono L-R':>13}{'Chrono left':>13}")
    for k in [k for k in keys if k[0] == "roll"]:
        e, c = eng[k], chr_[k]
        print(f"{k[1]:>7.0f}{e['loads']['left'] - e['loads']['right']:>13.4f}"
              f"{c['loads']['left'] - c['loads']['right']:>13.4f}{c['loads']['left']:>13.4f}")


if __name__ == "__main__":
    main()
