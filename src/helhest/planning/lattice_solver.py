"""Walk the lattice's own policy, for drawing the route it would actually drive.

The value iteration this used to hold now lives in `terrain_value_field` -- one copy, serving
more than this robot, and carrying a pile of fixes this one never had: arcs that close on the
lattice, a cost that follows the ground a move really covers, swept cells checked at the heading
they are crossed at, and a fused feasibility field. What stays here is the policy walk, because
it exists to draw a picture and nothing else reads it.
"""

from __future__ import annotations

import numpy as np


def trace_optimal(ctg, start, n_theta, cnx, cny, x0, y0, cell, max_steps=500):
    """Follow the lattice's OWN policy from the start pose: at each pose pick the primitive the value
    iteration would (min over feasible forward arcs of arc_cost + V[successor]) and integrate it as a
    smooth arc. Mirrors _relax_lattice_pose_kernel's selection -> the optimal forward-only trajectory.
    """
    V = ctg.V.numpy()
    blocked = ctg.blocked.numpy()
    tiltf = ctg.graded_tilt.numpy()
    s = ctg.solver
    pdr, pdc = s._prim_dr.numpy(), s._prim_dc.numpy()
    pheading, pcost = s._prim_heading.numpy(), s._prim_cost.numpy()
    sdr, sdc, sn = s._sweep_dr.numpy(), s._sweep_dc.numpy(), s._sweep_n.numpy()
    sdt = s._sweep_dt.numpy()  # heading-bin offset at each swept cell
    n_prim, tw = s.n_prim, float(ctg.flatness_weight)
    gr, gc = (int(v) for v in ctg._goal_rc.numpy())
    dth = 2.0 * np.pi / n_theta

    # walk the lattice state (r, c, t) along primitive successors -- this is the optimal policy and
    # descends V monotonically to the goal. Draw the path through the cell centers it visits (the
    # cells the heatmap/arrows are drawn on), so there's no continuous-integration drift off the grid.
    r = int(round((float(start[1]) - y0) / cell))
    c = int(round((float(start[0]) - x0) / cell))
    t = int(round((float(start[2]) % (2.0 * np.pi)) / dth)) % n_theta  # NEAREST bin
    pts = [(float(start[0]), float(start[1]))]
    for _ in range(max_steps):
        if not (0 <= r < cny and 0 <= c < cnx) or (abs(r - gr) <= 1 and abs(c - gc) <= 1):
            break  # reached (the goal cell or an immediate neighbor)
        best_p, best_val = -1, np.inf
        for p in range(n_prim):
            ns, ok, tsum = int(sn[t, p]), True, 0.0
            for si in range(ns):
                sr, sc = r + int(sdr[t, p, si]), c + int(sdc[t, p, si])
                st = (t + int(sdt[t, p, si])) % n_theta  # the heading it is FACING there
                if not (0 <= sr < cny and 0 <= sc < cnx) or blocked[sr, sc, st] > 0.5:
                    ok = False
                    break
                tsum += tiltf[sr, sc, st]
            if not ok:
                continue
            nr, nc, nt = r + int(pdr[t, p]), c + int(pdc[t, p]), int(pheading[t, p])
            if not (0 <= nr < cny and 0 <= nc < cnx):
                continue
            arc = float(pcost[t, p]) * (1.0 + tw * tsum / ns if ns > 0 else 1.0)
            val = arc + V[nr, nc, nt]
            if val < best_val:
                best_val, best_p = val, p
        if best_p < 0 or best_val >= ctg._vcap * 0.9:
            break
        r, c, t = r + int(pdr[t, best_p]), c + int(pdc[t, best_p]), int(pheading[t, best_p])
        # lattice cell center (drift-free)
        pts.append((x0 + (c + 0.5) * cell, y0 + (r + 0.5) * cell))
    return np.asarray(pts)
