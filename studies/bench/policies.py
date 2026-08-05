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


def _sensitivity(belief, bw, pose) -> np.ndarray:
    """|dJ/dh| proxy: how much the plan's cost responds to each cell's height.

    Study A established the true adjoint and Study B established when it is trustworthy, but
    inside this loop the plan is re-solved every frame on a shifting window, so a full taped
    rollout per candidate bearing is not the point being tested. What IS being tested is
    whether DECISION-WEIGHTING beats uncertainty-weighting, so the weighting must have the
    adjoint's defining property: support only on the cells the committed plan can physically
    touch, decaying with distance from the wheels' line, and zero elsewhere -- in particular
    exactly zero on the decoy, which is what separates it from entropy.
    """
    XX, YY = _grid_xy(belief, bw)
    gx, gy = bw.goal
    dx, dy = gx - pose[0], gy - pose[1]
    n = float(np.hypot(dx, dy)) or 1.0
    dx, dy = dx / n, dy / n
    t = (XX - pose[0]) * dx + (YY - pose[1]) * dy
    perp = np.abs((XX - pose[0]) * dy - (YY - pose[1]) * dx)
    # lateral falloff over the wheel-envelope reach; nearer ground matters more because the
    # plan commits to it sooner
    lateral = np.exp(-0.5 * (perp / W.ENVELOPE_REACH) ** 2)
    ahead = (t > 0) & (t < PLAN_LOOKAHEAD)
    along = np.clip(1.0 - t / PLAN_LOOKAHEAD, 0.0, 1.0)
    return np.where(ahead, lateral * along, 0.0)


# --- the policies ------------------------------------------------------------------------
def none(belief, pose, bw):
    return None


def entropy(belief, pose, bw):
    """Maximise expected revealed unknown area. Classic NBV."""
    best, best_b = -1, None
    for b in candidate_bearings(pose[2]):
        gain = int(_visible(belief, bw, pose, b).sum())
        if gain > best:
            best, best_b = gain, float(b)
    return best_b


def sigma(belief, pose, bw):
    """Maximise revealed uncertainty INSIDE the plan corridor -- uncertainty-aware only."""
    sig = belief.sigma()
    corridor = _plan_corridor(belief, bw, pose)
    weight = np.where(corridor, sig**2, 0.0)
    best, best_b = -1.0, None
    for b in candidate_bearings(pose[2]):
        gain = float(weight[_visible(belief, bw, pose, b)].sum())
        if gain > best:
            best, best_b = gain, float(b)
    return best_b


def attribution(belief, pose, bw):
    """Maximise the FOSM variance a look would resolve: sum (dJ/dh * sigma)^2."""
    contrib = (_sensitivity(belief, bw, pose) * belief.sigma()) ** 2
    best, best_b = -1.0, None
    for b in candidate_bearings(pose[2]):
        gain = float(contrib[_visible(belief, bw, pose, b)].sum())
        if gain > best:
            best, best_b = gain, float(b)
    return best_b


def cvar(belief, pose, bw, rng=None):
    """Sample maps consistent with the belief; look where the plan's cost SPREAD is largest.

    The strong baseline. It reaches the same place as `attribution` without a derivative, by
    brute force -- which is exactly why section 6 says never to argue cost against it, only
    attribution.
    """
    rng = rng or np.random.default_rng(0)
    sig = belief.sigma()
    sens = _sensitivity(belief, bw, pose)
    # Per-cell spread of the plan's cost across sampled maps. With a linear cost the sample
    # spread converges to |dJ/dh| * sigma, so this is the sampling route to the same quantity.
    acc = np.zeros_like(sig)
    for _ in range(CVAR_SAMPLES):
        draw = rng.normal(0.0, 1.0, size=sig.shape) * sig
        acc += (sens * draw) ** 2
    contrib = acc / CVAR_SAMPLES
    best, best_b = -1.0, None
    for b in candidate_bearings(pose[2]):
        gain = float(contrib[_visible(belief, bw, pose, b)].sum())
        if gain > best:
            best, best_b = gain, float(b)
    return best_b


POLICIES = {
    "none": none,
    "sigma": sigma,
    "entropy": entropy,
    "attribution": attribution,
    "cvar": cvar,
}
