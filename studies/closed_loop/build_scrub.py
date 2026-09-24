"""Turn the sweep's frame histories into the compact files the scrub page fetches.

One grayscale PNG per world per layer -- every frame of that layer stacked into a vertical strip
-- plus one small `manifest.json`. Each layer is quantised to uint8 against its own per-frame
range, which is what makes this fit at all: the raw float histories are ~6 MB a world.

PNG rather than a raw buffer for two reasons. Artifacts only serve a fixed set of file types and
`.bin` is not among them; and a uint8 grid IS an image, so the browser decodes it natively and
PNG's filtering squeezes the large uniform regions -- unmeasured map, unreachable field -- far
harder than anything the page could do after the fact.

Quantising per FRAME rather than per world is deliberate. The map's height range grows as the
robot discovers things, and a single global scale would render the first frames -- where the only
relief is centimetres of ground -- as flat grey. The cost is that a colour means something
slightly different from frame to frame, which the page states rather than hides.

  python studies/closed_loop/build_scrub.py --dir studies/closed_loop/out/sweep2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import numpy as np
from PIL import Image

# `pocket_wall` is optional: a pocket run that touched a wall, kept beside a clean one to compare
ORDER = ["gap", "slalom", "pillars", "pocket", "pocket_wall", "ridge", "bumpy"]
LAYERS = ("h", "seen", "blk", "v", "route", "cv")


def _quant(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map [lo, hi] onto 1..255. 0 is reserved: the page draws it as "nothing here"."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.ones(a.shape, np.uint8)
    q = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return (1 + q * 254).astype(np.uint8)


def build(world: str, npz: pathlib.Path, out_dir: pathlib.Path) -> dict:
    d = np.load(npz)
    if "hist_meta" not in d:
        raise SystemExit(f"{npz} has no frame history -- rerun with --history N")
    meta = d["hist_meta"]
    h, seen, blk, v = d["hist_h"], d["hist_seen"], d["hist_blk"], d["hist_v"]
    cv = d["hist_cv"]
    # Runs recorded before the heading-route layer existed have no `hist_route`; fall back to an
    # all-ones plane so the older npz still build rather than failing on a missing key.
    route = d["hist_route"] if "hist_route" in d.files else np.ones_like(v)
    nf = len(meta)
    # [m] per SIMULATED frame, indexed by frame number (not by recorded frame); inf in a world
    # without solids. Older runs have none.
    clear = d["clearance"] if "clearance" in d.files else None
    cap = float(v[np.isfinite(v)].max()) if np.isfinite(v).any() else 1.0

    strips = {k: [] for k in LAYERS}
    frames = []
    for i in range(nf):
        # the value field's unreachable sentinel is a huge finite number, not an inf; it must not
        # set the scale or every reachable cell collapses into one bucket
        vi = v[i]
        reach = vi < cap * 0.9
        vlo = float(vi[reach].min()) if reach.any() else 0.0
        vhi = float(vi[reach].max()) if reach.any() else 1.0
        ci = cv[i]
        creach = ci < 1.0e29
        clo = float(ci[creach].min()) if creach.any() else 0.0
        chi = float(ci[creach].max()) if creach.any() else 1.0
        sm = seen[i] > 0
        hlo = float(h[i][sm].min()) if sm.any() else 0.0
        hhi = float(h[i][sm].max()) if sm.any() else 1.0

        planes = {
            "h": np.where(sm, _quant(h[i], hlo, hhi), 0),  # 0 = never measured
            "seen": (seen[i] > 0).astype(np.uint8),
            "blk": _quant(blk[i], 0.0, 1.0),
            "v": np.where(reach, _quant(vi, vlo, vhi), 0),  # 0 = no route
            "cv": np.where(creach, _quant(ci, clo, chi), 0),
            "route": _quant(route[i], 0.0, 1.0),
        }
        for k in LAYERS:
            # flipped here, once, rather than in the page: grid row 0 is the LOW y edge and a
            # PNG's row 0 is its top, so storing them already flipped means the viewer can blit
            # a band straight to the canvas with no transform
            strips[k].append(planes[k].astype(np.uint8)[::-1])
        row = [float(x) for x in meta[i]]
        f, rx, ry, yaw, cl, cr, dist, bx, by, roll, pitch = row[:11]
        vh = row[11] if len(row) > 11 else float("nan")
        frames.append(
            dict(
                f=int(f),
                x=round(rx, 3),
                y=round(ry, 3),
                yaw=round(yaw, 4),
                cmd=[round(cl, 2), round(cr, 2)],
                dist=round(dist, 2),
                bx=round(bx, 3),
                by=round(by, 3),
                h=[round(hlo, 3), round(hhi, 3)],
                v=[round(vlo, 2), round(vhi, 2)],
                cv=[round(clo, 2), round(chi, 2)],
                seen=round(float(sm.mean()), 4),
                blk=round(float(blk[i].mean()), 4),
                # [deg] the attitude the robot was actually at. Nose-up is NEGATIVE pitch, kept
                # in the settle's sign convention rather than flipped for display, so this reads
                # the same as everything else that talks about the envelope.
                rp=[round(np.degrees(roll), 1), round(np.degrees(pitch), 1)],
                # V at the robot's own cell AND own heading, against `v`'s best-over-headings.
                # None where the pose left the routing window, or for runs recorded before it.
                vh=(None if not np.isfinite(vh) else round(vh, 2)),
                # what fraction of the routing window's headings have a route at all
                route=round(float(route[i].mean()), 4),
                # [m] bare-footprint distance to the nearest wall, < 0 = touching; None = no walls
                cl=(
                    None
                    if clear is None or int(f) >= len(clear) or not np.isfinite(clear[int(f)])
                    else round(float(clear[int(f)]), 3)
                ),
            )
        )
    nbytes = 0
    for k in LAYERS:
        # frames stacked top to bottom; the page slices row bands out of one decoded image
        strip = np.concatenate(strips[k], axis=0)
        f = out_dir / f"{world}_{k}.png"
        Image.fromarray(strip, mode="L").save(f, optimize=True)
        nbytes += f.stat().st_size
    return dict(
        n=int(h.shape[1]),
        nr=int(v.shape[1]),
        nc=int(cv.shape[1]),
        cell=float(d["cell"]),
        # the unreachable sentinel, so the page can say "the heading it is ON has no route"
        # without inventing a threshold for it
        cap=round(cap, 2),
        # which controller produced this run (drive_sim saves the resolved plan_* values since
        # 1f9a71d). None for older runs -- the page then says it does not know rather than guess.
        controller=(json.loads(str(d["plan_config"])) if "plan_config" in d.files else None),
        coarse_cell=float(d["coarse_cell"]),
        off_r=(int(h.shape[1]) // 2 - int(v.shape[1]) // 2),
        goal=[float(x) for x in d["goal"]],
        reached=bool(d["reached"]),
        wall=(
            None
            if clear is None or not np.isfinite(clear).any()
            else dict(min=round(float(clear.min()), 3), touching=int((clear < 0).sum()))
        ),
        trail=[[round(float(x), 2), round(float(y), 2)] for x, y in d["trail"]],
        bytes=nbytes,
        frames=frames,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="studies/closed_loop/out/sweep2")
    p.add_argument("--out", default="studies/closed_loop/out/scrub")
    a = p.parse_args()
    src, out = pathlib.Path(a.dir), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # the envelope the settle vetoes against, so the page can mark an attitude that is past it
    # without restating three numbers that were measured once and may move again
    from helhest.engine.robot import RobotParams

    rp = RobotParams()
    manifest = {
        "layers": list(LAYERS),
        "limits": {
            "roll": round(float(np.degrees(rp.max_roll)), 1),
            "pitch_up": round(float(np.degrees(rp.max_pitch_up)), 1),
            "pitch_down": round(float(np.degrees(rp.max_pitch_down)), 1),
        },
        "worlds": {},
    }
    total = 0
    for w in ORDER:
        f = src / f"{w}.npz"
        if not f.exists():
            print(f"  {w:<8s} missing, skipped")
            continue
        manifest["worlds"][w] = build(w, f, out)
        size = manifest["worlds"][w]["bytes"]
        total += size
        print(f"  {w:<8s} {len(manifest['worlds'][w]['frames']):>4d} frames  {size/1e6:>6.2f} MB")
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    # the page itself is source and lives beside this script; the output directory is generated
    shutil.copy(pathlib.Path(__file__).parent / "scrub_page.html", out / "index.html")
    mj = (out / "manifest.json").stat().st_size
    print(
        f"\n  manifest {mj/1e6:.2f} MB    binaries {total/1e6:.2f} MB    total {(total+mj)/1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
