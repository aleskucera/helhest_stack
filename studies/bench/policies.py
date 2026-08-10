"""Where should the robot look? Five policies, weakest to strongest.

Every policy has the same signature `(belief, pose, world) -> bearing | None` and the same
information: the accumulated belief and its own pose. None of them may read ground truth --
`world` is passed only for grid geometry, and reading `world.scene.H`, `gap_mask` or
`decoy_mask` inside a policy would invalidate the comparison. The loop records what each look
actually revealed, so a policy that cheated would show up as suspiciously well-aimed.

  none        never looks. The null baseline.
  sigma       looks where believed uncertainty is greatest along the CURRENT PLAN's corridor --
              uncertainty-aware but not decision-aware.
  entropy     classic NBV: maximise expected revealed unknown area, estimated by ray-casting on
              the belief. This is the policy the decoy is built to tempt, and it is the real
              competition -- SENSITIVITY_PLAN.md section 8 says to contrast explicitly with
              information-theoretic active perception.
  attribution OURS: maximise sum over revealed cells of (dJ/dh_i * sigma_i)^2 -- the FOSM
              variance contribution of the cells a look would resolve. Decision-focused: it
              asks not "what don't I know" but "what don't I know THAT MATTERS TO THIS PLAN".
  cvar        the strong baseline of section 6: sample K maps consistent with the belief, score
              the plan on each, and look where the spread across samples is largest. It gets
              the same information as `attribution` by a much more expensive route.

CANDIDATE BEARINGS. All policies choose from the same fixed fan around the robot's heading, so
differences come from the objective and never from a finer search.
"""

from __future__ import annotations

import numpy as np

from helhest.perception.lidar import lidar_scan

from . import world as W

N_BEARINGS = 17  # candidate look directions
# A steerable sensor can pan well past the driving direction, and the fan must be wide enough
# that every policy's preferred target is actually IN the option set. At +-75 deg the decoy
# (which sits at ~115 deg once the robot is near the barrier) was simply unreachable, so the
# entropy baseline could not chase it -- and beating a baseline that was denied its own best
# option would have proved nothing. Stops short of straight behind, which no sensible NBV
# would choose while driving forward.
BEARING_SPAN = np.radians(240.0)
MOUNT = 0.4
PLAN_LOOKAHEAD = 9.0  # [m] how far along the intended route a policy reasons about
CORRIDOR = 1.2  # [m] half-width of the corridor a plan is assumed to occupy
CVAR_SAMPLES = 12

# C5: DECIDE WHETHER TO OBSERVE, not only where. Every policy skips its look when the best
# available bearing would resolve less than this fraction of its OWN objective's total. The
# benchmark showed every arm losing to never-looking on the many seeds that did not need a
# look, purely because the budget was spent unconditionally -- which measures the budget, not
# the policy.
#
# Applied with ONE fraction to ALL arms, each against its own total, so no arm is given a
# tuned advantage. The value is chosen a priori rather than swept: resolving a quarter of the
# quantity you care about is a reasonable bar for spending 8 frames of a ~200-frame episode.
# It has NOT been swept, so its robustness is unestablished.
# MEASURED AND REJECTED as a default -- see studies/bench/compare_c5.py and RESULTS.md section
# 6c. At 0.25 the gate NEVER BINDS for attribution (its objective is concentrated exactly where
# a route-directed look resolves it, so the resolvable fraction is always high) and binds far
# too aggressively for entropy, whose objective is diffuse: entropy's looks fell 4.0 -> 1.8 and
# its mean time got 34 frames WORSE. It also weakened the one supported result, attribution vs
# entropy, from 24/32 (p=0.007) to 20/32 (p=0.215).
#
# The lesson is about what C5 actually requires: "what fraction of my objective could this
# resolve" is NOT value of information. VoI asks whether the observation would CHANGE THE
# DECISION, which is a different and harder quantity than how much variance it removes.
# Set to 0.0 (no gating) so the ungated behaviour is the default; the gated run is preserved
# in results_*_c5.json.
LOOK_THRESHOLD = 0.0


def _best_bearing(belief, bw, pose, weight):
    """Bearing maximising `weight` over the cells a look would resolve, or None if the best
    available look would resolve less than LOOK_THRESHOLD of `weight`'s total."""
    total = float(weight.sum())
    best, best_b = -1.0, None
    for b in candidate_bearings(pose[2]):
        gain = float(weight[_visible(belief, bw, pose, b)].sum())
        if gain > best:
            best, best_b = gain, float(b)
    if total <= 0.0 or best < LOOK_THRESHOLD * total:
        return None
    if best <= 0.0:
        # Every candidate resolves nothing -- all gains tied at zero. The loop above then
        # leaves best_b at the FIRST (maximally off-axis) candidate purely by iteration order,
        # which burns an 8-frame look staring off to the side for no gain. Point forward
        # instead: no candidate is better, so there is no reason to prefer the extreme one.
        return float(pose[2])
    return best_b


def candidate_bearings(yaw: float) -> np.ndarray:
    return yaw + np.linspace(-BEARING_SPAN / 2, BEARING_SPAN / 2, N_BEARINGS)


def _visible(belief, bw, pose, bearing: float) -> np.ndarray:
    """Cells a look at `bearing` would reveal, ray-cast on the BELIEF (not on truth)."""
    _, vis = lidar_scan(
        belief.elev,
        bw.scene.x0,
        bw.scene.y0,
        bw.scene.cell,
        (pose[0], pose[1], float(bearing)),
        fov_deg=W.LOOK_FOV,
        max_range=W.LOOK_RANGE,
        mount_height=MOUNT,
    )
    return vis & ~belief.known


def _grid_xy(belief, bw):
    ny, nx = belief.known.shape
    xs = bw.scene.x0 + (np.arange(nx) + 0.5) * belief.cell
    ys = bw.scene.y0 + (np.arange(ny) + 0.5) * belief.cell
    return np.meshgrid(xs, ys)


def _route_distance(belief, bw, route) -> np.ndarray:
    """Distance from every cell to the planner's intended route, and arclength along it."""
    XX, YY = _grid_xy(belief, bw)
    best = np.full(XX.shape, np.inf)
    along = np.zeros(XX.shape)
    acc = 0.0
    for i in range(len(route) - 1):
        ax, ay = route[i]
        bx, by = route[i + 1]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-12:
            continue
        t = np.clip(((XX - ax) * dx + (YY - ay) * dy) / seg2, 0.0, 1.0)
        d = np.hypot(XX - (ax + t * dx), YY - (ay + t * dy))
        closer = d < best
        best = np.where(closer, d, best)
        along = np.where(closer, acc + t * np.sqrt(seg2), along)
        acc += float(np.sqrt(seg2))
    return best, along


def _plan_corridor(belief, bw, pose) -> np.ndarray:
    """The strip of ground the robot's intended route occupies.

    A stand-in for the committed plan's footprint: the straight line from the robot toward the
    goal, out to PLAN_LOOKAHEAD. It is deliberately the SAME geometry for every policy that
    uses it, so `sigma` and `attribution` differ only in how they weight cells inside it -- by
    uncertainty alone, or by uncertainty times decision sensitivity.
    """
    XX, YY = _grid_xy(belief, bw)
    gx, gy = bw.goal
    dx, dy = gx - pose[0], gy - pose[1]
    n = float(np.hypot(dx, dy)) or 1.0
    dx, dy = dx / n, dy / n
    t = (XX - pose[0]) * dx + (YY - pose[1]) * dy
    perp = np.abs((XX - pose[0]) * dy - (YY - pose[1]) * dx)
    return (t > 0) & (t < PLAN_LOOKAHEAD) & (perp < CORRIDOR)


def _sensitivity(belief, bw, pose, route=None) -> np.ndarray:
    """|dJ/dh| proxy: how much the plan's cost responds to each cell's height.

    Study A established the true adjoint and Study B established when it is trustworthy, but
    inside this loop the plan is re-solved every frame on a shifting window, so a full taped
    rollout per candidate bearing is not the point being tested. What IS being tested is
    whether DECISION-WEIGHTING beats uncertainty-weighting, so the weighting must have the
    adjoint's defining property: support only on the cells the committed plan can physically
    touch, decaying with distance from the wheels' line, and zero elsewhere -- in particular
    exactly zero on the decoy, which is what separates it from entropy.
    """
    if route is None or len(route) < 2:
        # No route yet (first frame): fall back to the straight line toward the goal.
        XX, YY = _grid_xy(belief, bw)
        gx, gy = bw.goal
        dx, dy = gx - pose[0], gy - pose[1]
        n = float(np.hypot(dx, dy)) or 1.0
        dx, dy = dx / n, dy / n
        perp = np.abs((XX - pose[0]) * dy - (YY - pose[1]) * dx)
        t = (XX - pose[0]) * dx + (YY - pose[1]) * dy
    else:
        perp, t = _route_distance(belief, bw, route)
    lateral = np.exp(-0.5 * (perp / W.ENVELOPE_REACH) ** 2)
    ahead = (t > 0) & (t < PLAN_LOOKAHEAD)
    along = np.clip(1.0 - t / PLAN_LOOKAHEAD, 0.0, 1.0)
    return np.where(ahead, lateral * along, 0.0)


# --- the policies ------------------------------------------------------------------------
# Every policy takes the same trailing `rng` slot, even the ones that don't use it, because
# loop.py calls all six positionally with an identical arg list -- an inconsistent signature
# would silently swallow the rng into some other parameter for whichever policy lacked it.
def none(belief, pose, bw, route=None, routes=None, rng=None):
    return None


def entropy(belief, pose, bw, route=None, routes=None, rng=None):
    """Maximise expected information gain, sum of sigma^2 over the cells a look would resolve.

    The sigma-weighted form, not a raw cell count. Counting cells makes the objective nearly
    direction-independent -- the cone has the same area whichever way it points -- so the
    policy has no signal and tie-breaks arbitrarily, which is a straw man rather than a
    baseline. Weighting by predicted uncertainty is both the standard formulation and the
    stronger opponent: it concentrates on terrain the map expects to be complex.
    """
    return _best_bearing(belief, bw, pose, belief.sigma() ** 2)


def sigma(belief, pose, bw, route=None, routes=None, rng=None):
    """Maximise revealed uncertainty INSIDE the plan corridor -- uncertainty-aware only."""
    corridor = _plan_corridor(belief, bw, pose)
    return _best_bearing(belief, bw, pose, np.where(corridor, belief.sigma() ** 2, 0.0))


def attribution(belief, pose, bw, route=None, routes=None, rng=None):
    """Maximise the FOSM variance a look would resolve: sum (dJ/dh * sigma)^2."""
    return _best_bearing(
        belief, bw, pose, (_sensitivity(belief, bw, pose, route) * belief.sigma()) ** 2
    )


def cvar(belief, pose, bw, route=None, routes=None, rng=None):
    """Sample maps consistent with the belief; look where the plan's cost SPREAD is largest.

    The strong baseline. It reaches the same place as `attribution` without a derivative, by
    brute force -- which is exactly why section 6 says never to argue cost against it, only
    attribution.
    """
    # `rng` comes from loop.py, spawned per-seed/per-frame so the 12 draws vary across
    # episodes and looks instead of being frozen. Fall back to a fixed seed only for callers
    # (tests, direct invocation) that don't thread one through.
    if rng is None:
        rng = np.random.default_rng(0)
    sig = belief.sigma()
    sens = _sensitivity(belief, bw, pose, route)
    # Per-cell spread of the plan's cost across sampled maps. With a linear cost the sample
    # spread converges to |dJ/dh| * sigma, so this is the sampling route to the same quantity.
    acc = np.zeros_like(sig)
    for _ in range(CVAR_SAMPLES):
        draw = rng.normal(0.0, 1.0, size=sig.shape) * sig
        acc += (sens * draw) ** 2
    return _best_bearing(belief, bw, pose, acc / CVAR_SAMPLES)


def disagreement(belief, pose, bw, route=None, routes=None, rng=None):
    """OURS, corrected: look where the near-optimal routes DISAGREE.

    Single-plan attribution is confirmatory (see diagnose_confirmatory.py): it aims along the
    route the planner has already chosen, which tends to confirm that choice rather than test
    it, and in a routing problem the informative look is at the alternative the plan REJECTED.
    dJ/dh for one committed plan cannot see that cell -- its sensitivity there is low BECAUSE
    the plan avoids it.

    SENSITIVITY_PLAN.md section 1 specifies attribution over the elite SET's cost variance, and
    that is what fixes it. Weight each cell by the VARIANCE ACROSS ROUTES of its per-route
    sensitivity, times sigma^2:

        contrib(i) = Var_k[ s_k(i) ] * sigma(i)^2

    A cell every candidate route crosses has zero variance -- observing it cannot change which
    route wins, however sensitive the chosen plan is to it. A cell no route crosses (the decoy)
    is zero too. Only cells that SEPARATE the alternatives score, which is what "could this
    observation change the decision" means in this setting.

    Falls back to single-plan attribution when the elite set has collapsed to one route -- then
    there is no disagreement to resolve and the chosen plan is the only thing to test.
    """
    if not routes or len(routes) < 2:
        return attribution(belief, pose, bw, route)
    stack = np.stack([_sensitivity(belief, bw, pose, r) for r in routes])
    contrib = stack.var(axis=0) * belief.sigma() ** 2
    if float(contrib.sum()) <= 0.0:
        return attribution(belief, pose, bw, route)
    return _best_bearing(belief, bw, pose, contrib)


POLICIES = {
    "none": none,
    "sigma": sigma,
    "entropy": entropy,
    "attribution": attribution,
    "cvar": cvar,
    "disagreement": disagreement,
}
