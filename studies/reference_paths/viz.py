"""Contact sheet of reference-stage frames: V's heading-minimum (shaded), the cells forbidden at
every heading (black), the descent (grey dots), the taut string (blue), the curve (orange), the
robot (green arrow) and the goal (red star), in the window frame.

    python viz.py <rec.npz> <out.png> <frame> [<frame> ...]
    python viz.py <rec.npz> <out.png> --worst <scored.json> <key> [n]   # the n worst by a metric

`--worst` reads replay_ref.py's --out json (the entry named after the recording) and picks the
frames with the largest value of `key`, at most one per 1 s so one event does not fill the sheet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_ref import load  # noqa: E402
from replay_ref import ragged  # noqa: E402


def pick_worst(scored: list[dict], key: str, n: int) -> list[int]:
    cand = sorted(
        (f for f in scored if key in f and np.isfinite(f[key])), key=lambda f: -abs(f[key])
    )
    out: list[dict] = []
    for f in cand:
        if all(abs(f["t"] - o["t"]) > 1.0 for o in out):
            out.append(f)
        if len(out) >= n:
            break
    return [f["i"] for f in out]


def draw(ax, d: dict, i: int, rag: dict, title: str, alt: list | None = None) -> None:
    g = d["sgrid"][i]
    vmin = d["V"][i].min(2) if "V" in d else d["vmin"][i].astype(np.float32)
    forbid = d["blocked"][i].all(2) if "blocked" in d else d["forbid"][i] > 0
    ny, nx = vmin.shape
    c = g[2]
    ext = (g[0] - c / 2, g[0] + (nx - 0.5) * c, g[1] - c / 2, g[1] + (ny - 0.5) * c)
    show = np.where(vmin < 80.0, vmin, np.nan)
    ax.imshow(show, origin="lower", extent=ext, cmap="viridis", alpha=0.55)
    ax.imshow(np.where(forbid, 1.0, np.nan), origin="lower", extent=ext, cmap="gray_r", vmin=0)
    ds, ta, cu = rag["desc"][i], rag["taut"][i], rag["curve"][i]
    if len(ds):
        ax.plot(ds[:, 0], ds[:, 1], ".", color="0.5", ms=2)
    if len(ta):
        ax.plot(ta[:, 0], ta[:, 1], "-o", color="tab:blue", ms=3, lw=1.2)
    if len(cu):
        ax.plot(cu[:, 0], cu[:, 1], "-", color="tab:orange", lw=2.2)
    if alt is not None and len(alt[i]):
        ax.plot(alt[i][:, 0], alt[i][:, 1], "-", color="magenta", lw=1.6)
    st, go = d["state"][i], d["goal"][i]
    ax.arrow(st[0], st[1], 0.8 * np.cos(st[2]), 0.8 * np.sin(st[2]), color="lime", width=0.08)
    ax.plot(go[0], go[1], "r*", ms=12)
    # zoom on what matters: the robot, the curve, the taut string
    pts = np.vstack([st[None, :2]] + [a for a in (ta, cu) if len(a)])
    lo, hi = pts.min(0) - 2.0, pts.max(0) + 2.0
    span = max(hi - lo)
    mid = 0.5 * (lo + hi)
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def main() -> None:
    alt = None
    if "--alt" in sys.argv:  # a variant's curves (replay_ref.py --curves), drawn in magenta
        k = sys.argv.index("--alt")
        alt = ragged(load(Path(sys.argv[k + 1])), "curve")
        del sys.argv[k : k + 2]
    rec, out = Path(sys.argv[1]), Path(sys.argv[2])
    d = load(rec)
    if sys.argv[3] == "--worst":
        scored = json.loads(Path(sys.argv[4]).read_text())[rec.stem]
        key = sys.argv[5]
        n = int(sys.argv[6]) if len(sys.argv) > 6 else 12
        frames = pick_worst(scored, key, n)
        info = {f["i"]: f for f in scored}
    else:
        frames = [int(x) for x in sys.argv[3:]]
        info = {}
    rag = {k: ragged(d, k) for k in ("desc", "taut", "curve")}
    n = len(frames)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    for ax, i in zip(axs.ravel(), frames):
        t = float(d["t"][i] - d["t"][0])
        f = info.get(i, {})
        extra = " ".join(
            f"{k}={f[k]:.2f}" for k in ("wall_m", "taut_dev", "excess_bends", "flick") if k in f
        )
        draw(
            ax,
            d,
            i,
            rag,
            f"#{i} t={t:.1f}s cold={int(d['cold'][i])} n={int(d['n_ref'][i])}\n{extra}",
            alt,
        )
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    print(f"wrote {out} ({n} frames)")


if __name__ == "__main__":
    main()
