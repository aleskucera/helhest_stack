"""Why does looking make the route LONGER on the seeds the null baseline handles well?

    .venv/bin/python -m studies.bench.diagnose_confirmatory

See RESULTS.md section 6d. Attribution aimed along the CURRENT plan's route chases its own
commitment: look 1 reveals the wall ahead, the route bends to the nearest apparently-free edge
(an artifact of what has been observed), the next look follows the new route and confirms more
wall on that side. Confirmatory, not exploratory. On seed 27 the gap is at -42 deg and the
looks go [-3, +13, +63, -86].
"""

import numpy as np, warp as wp

wp.init()
from studies.bench import world as W, loop as L, policies as P

for seed in (27, 3):
    bw = W.build(seed)
    print(f"\n=== seed {seed}: gap at y={bw.gap_y:+.2f} ===")
    for tag, pol in (("none", None), ("attribution", P.POLICIES["attribution"])):
        tr = L.run(bw, policy=pol)
        t = np.array(tr.trail)
        # excursion to the side AWAY from the gap
        wrong = t[:, 1].max() if bw.gap_y < 0 else t[:, 1].min()
        near_wall = t[np.abs(t[:, 0] - W.WALL_X) < 1.5]
        print(
            f"  {tag:<12} t={tr.total_time:>4} path={tr.path_len:>5.1f}m gap@f{tr.gap_known_frame:<4}"
            f" wrong-side excursion {wrong:+.1f} m"
        )
        if tag == "attribution":
            print(f"      look bearings (deg): {[round(np.degrees(b)) for b in tr.look_bearings]}")
            print(
                f"      gap bearing from start: {np.degrees(W.bearing_to(bw, bw.gap_mask, (0.,0.))):.0f} deg"
            )
        for i in range(0, min(len(t), 200), 40):
            print(f"      f{i:>3} ({t[i,0]:5.2f},{t[i,1]:+5.2f})")
