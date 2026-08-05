"""Where does the curvature come from -- smooth dynamics, or contact switching?

    .venv/bin/python -m studies.adjoint.hessian_split

Decides whether the second-derivative term c_i can be computed ANALYTICALLY instead of by
finite differences. It cannot, for a structural reason worth recording.

c_full   : perturb a cell, recompute the arg-max, re-run.  (what curvature_eval measured)
c_frozen : identical, but the contact is held at the UNPERTURBED arg-max, which is exactly
           the model an analytic second-order adjoint through today's code would see --
           with the contact fixed the envelope is LINEAR in h, so the dilation contributes
           zero second derivative by construction.
The gap between them is the part no frozen-contact analytic Hessian can ever produce.

MEASURED: c_frozen is 0.3-3% of c_full on flat/slope/curb, 35% on rock. So an analytic
Hessian through today's model would be ~2 orders too small nearly everywhere. Rock is the
cross-check -- it is the one region with genuine geometric terrain curvature, and it is the
one region where the frozen number is not negligible.

An analytic route would need all three of: a continuum contact model (so the contact POSITION
is differentiable), second-order envelope-theorem terms (at first order you may freeze the
maximiser; at second order you may not), and a hand-written second-order settle adjoint --
each needing its own Study-A-style validation. Batched finite differences capture the
switching for free because they re-run the true forward. Note the reversal: sub-cell
refinement failed to improve the FIRST derivative but is a prerequisite for computing the
SECOND one analytically.
"""

import numpy as np, warp as wp

wp.init()
from studies.adjoint.scene import build_scene, rollouts, REGION_NAMES
from studies.adjoint import sigma as sm
from studies.adjoint.scene import LANE_Y
from studies.adjoint.harness import _perturb_cell
from studies.adjoint.study_b import MonteCarlo, _cost, _cost_gradient

sc = build_scene()
poses, omega, labels = rollouts()
sig = sm.sigma_field(sc, inpainted=sm.decoy_mask(sc, LANE_Y[0] + 1.4))


def cost_at(mc, iy, ix, delta, refresh_contact):
    mc.h._reset_terrain(dilate=True)
    if delta != 0.0:
        wp.launch(
            _perturb_cell,
            mc.batch,
            inputs=[mc.h.sim.elevation, int(iy), int(ix), float(delta)],
            device=mc.h.device,
        )
    if refresh_contact:
        mc.h.sim._contact()  # true forward: the arg-max may move
    wp.copy(mc.h.sim.current_wheel_omega[0], mc.h.sim.init_current_wheel_omega)
    mc.h.terms.zero_()
    mc.h._launches()  # gather + init + T steps + terms
    return float(_cost(mc.h.terms.numpy())[0])


print(
    f"{'rollout':<15}{'region':<8}{'n':>5}{'|c_full| med':>14}{'|c_frozen| med':>16}{'frozen/full':>13}"
)
for b in (0, 2, 4, 6):
    mc = MonteCarlo(sc, poses[b], omega[:, b, :], 8)
    grads, _ = mc.h.adjoint(dilate=True, leaf="elevation")
    g = _cost_gradient(grads)[0]
    strength = np.abs(g)
    idx = np.argwhere(strength > 0.05 * strength.max())
    cells = idx[np.argsort(-strength[idx[:, 0], idx[:, 1]])][:40]
    mc.h._reset_terrain(dilate=True)
    mc.h.sim._contact()  # the frozen reference arg-max
    rows = {}
    for iy, ix in cells:
        sd = float(sig[iy, ix])
        vals = {}
        for tag, refresh in (("full", True), ("frozen", False)):
            if not refresh:
                mc.h._reset_terrain(dilate=True)
                mc.h.sim._contact()  # freeze on clean terrain
            j0 = cost_at(mc, iy, ix, 0.0, refresh)
            jp = cost_at(mc, iy, ix, +sd, refresh)
            jm = cost_at(mc, iy, ix, -sd, refresh)
            vals[tag] = (jp - 2 * j0 + jm) / (sd * sd)
        rows.setdefault(REGION_NAMES[int(sc.region[iy, ix])], []).append(vals)
    for reg, vs in rows.items():
        f = np.array([abs(v["full"]) for v in vs])
        z = np.array([abs(v["frozen"]) for v in vs])
        print(
            f"{labels[b]:<15}{reg:<8}{len(vs):>5}{np.median(f):>14.2f}{np.median(z):>16.2f}"
            f"{np.median(z)/max(np.median(f),1e-12):>12.1%}"
        )
    del mc
