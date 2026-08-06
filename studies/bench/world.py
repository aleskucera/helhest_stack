"""The section-6 benchmark world: a barrier whose gap you must find, and a decoy that pulls
an entropy-directed sensor the wrong way.

SENSITIVITY_PLAN.md section 6 asks for one specific scenario, and it is about DISAGREEMENT,
not cost reduction:

    a case where the highest-entropy cell and the highest-dJ/dh cell are different, and
    sensing the latter improves the task outcome while sensing the former does not.

Two earlier designs failed their gates, and both failures are worth recording because they
constrain what can work:

  A CREST occluding a hazard behind it. A crest gentle enough for the robot to CLIMB (25 deg
  pitch limit) is necessarily wide, so the sensor rides up its flank and sees over the top from
  ~4 m out. Drivability and occlusion are in direct conflict for that mechanism.

  A BARE STEP hidden by sensor range. The warning distance then simply EQUALS the sensor range
  -- nothing is learned about sensing policy, only about range. Worse, a forward look into open
  ground reveals MORE unknown area than a look at a rough decoy (rough terrain self-occludes),
  so an entropy-directed sensor prefers looking forward and the decoy never tempts it.

The barrier-with-a-gap fixes both structurally:

  WALL + GAP   an impassable wall square across the route, with a gap at a randomised lateral
               offset. The question is no longer "is something there" but WHERE THE OPENING IS,
               which is a routing decision and cannot be answered by stopping. The wall also
               BLOCKS a forward look's reach, so a forward look reveals little -- which is what
               finally makes the open decoy side genuinely more attractive to entropy.
  DECOY        a large, rough, unobserved region on the opposite side, beyond any possible
               wheel-envelope contact (0.715 m = half-track 0.365 + wheel radius 0.35), and
               angularly separated by more than one look cone so the policies must choose.

Failure is not a crash -- the robot can always stop. It is committing to a route that dead-ends
and then having to search along a wall. So the metric is time-to-goal, which is what section 6
asks for ("time-to-goal at equal safety").

Ground truth here is what the WarpDriver drives on. What the planner believes is accumulated
from occluded scans (loop.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helhest.heightmap import _grid
from helhest.heightmap import Heightmap

CELL = 0.06
XLIM = (-2.0, 14.0)
# Wide enough that a LOOK cone in either direction fits inside the map. At +-8.5 m the
# decoy-directed cone ran off the edge, so most of its reward simply did not exist and entropy
# preferred the gap for a pure map-size artifact -- which would have been a fair reviewer
# objection to the baseline rather than a property of the scenario.
YLIM = (-11.0, 11.0)

START = (0.0, 0.0, 0.0)
GOAL = (12.0, 0.0)

# --- sensing model ---------------------------------------------------------------------
DEFAULT_RANGE = 3.5  # [m] free every frame: the ground you are about to drive on
DEFAULT_FOV = 180.0
# 9 m, not 12. At 12 m a LATERAL look from the route ran past the map edge, so the decoy
# direction was scored on a truncated cone while the forward direction was not -- an artifact
# that made the critical direction look more informative than it is. 9 m keeps both cones
# inside the free space.
LOOK_RANGE = 9.0  # [m] a dwell buys range
LOOK_FOV = 40.0  # [deg] and costs field of view, so the bearing is a real choice

# --- the barrier -------------------------------------------------------------------------
WALL_X = 8.0  # [m] across the route
WALL_HALF_DEPTH = 0.3
WALL_HEIGHT = 0.8  # [m] far above the 0.35 m wheel radius: impassable, unambiguously
# Far enough off the approach line that finding it is a real search. At |y| in [2.4, 4.0] the
# robot met the wall and stumbled onto the gap almost immediately -- an ignorant run came
# within 6% of the omniscient one, leaving no headroom for any sensing policy to recover.
# Out here the robot must sweep along the barrier and guess which way first.
GAP_Y = (5.0, 8.0)
# Wide enough to actually DRIVE through. At 0.85 m the clearance either side was only
# 0.85 - 0.715 = 0.135 m against the wheel-envelope reach, and the settle-based feasibility
# would not commit: an omniscient robot parked in front of the gap and oscillated for 900
# frames while an ignorant one got through by accident. A benchmark whose ORACLE cannot solve
# it measures nothing.
GAP_HALF_WIDTH = 1.5

# --- the decoy -------------------------------------------------------------------------
# LATERAL, and near enough that its whole look cone stays in front of the barrier. The
# barrier spans the full map width, so ANY cone that reaches x = WALL_X is cut off there --
# while a gap-directed cone sees straight THROUGH the gap and collects the open ground beyond.
# That asymmetry, not the decoy's roughness, is what kept entropy preferring the gap through
# three attempted fixes (moving it, flattening it, enlarging the map). Placing the decoy off to
# the side at ~65 deg means its cone never reaches the wall within LOOK_RANGE.
DECOY_CX = 3.0
# Near enough that the robot's DEFAULT sensing clips its near edge while driving past. That
# matters: a context-predicted sigma can only mark the decoy as rough if some of its roughness
# has actually been observed. At 6.5 m the whole region stayed unseen, no roughness was ever
# measured, and the predicted sigma collapsed back to uniform -- leaving the entropy baseline
# with nothing to be tempted by. Still 3.5x the wheel-envelope reach from the route.
DECOY_ABS_CY = 5.0  # [m] always OPPOSITE the gap -- see build()
DECOY_HALF_X = 2.5
DECOY_HALF_Y = 2.5
# Gentle relief, not rugged. It must read as genuinely uncertain terrain, but rough ground
# SELF-OCCLUDES, which suppresses how much area a look actually reveals -- and an
# entropy-directed sensor counts revealed area. With sharp relief the decoy stopped being
# tempting at all and gate 6 failed, which would have quietly voided the whole comparison.
# Nearly flat. Roughness is incidental to the decoy's job -- what makes it tempting is being
# a large UNOBSERVED region -- and rough ground self-occludes, which suppresses the area a look
# reveals and so weakens the very baseline the decoy exists to tempt. Flat is the strongest
# version of the temptation, and therefore the fairest test.
DECOY_RELIEF = 0.04

BORDER = 0.5  # [m] impassable perimeter -- see build()

ENVELOPE_REACH = 0.715  # [m] half_track + wheel_radius: the widest a contact can ever be


@dataclass(frozen=True)
class BenchWorld:
    """Ground-truth terrain plus the masks the analysis needs to know what is what."""

    scene: Heightmap
    wall_mask: np.ndarray  # the impassable barrier
    gap_mask: np.ndarray  # the opening -- the cells the decision actually rests on
    decoy_mask: np.ndarray  # high-uncertainty, decision-irrelevant
    gap_y: float
    approach_yaw: float
    start: tuple[float, float, float]
    goal: tuple[float, float]
    seed: int


def build(seed: int = 0, cell: float = CELL) -> BenchWorld:
    """One randomised instance. `seed` varies the gap position and the approach angle."""
    rng = np.random.default_rng(seed)
    # The SIDE is randomised, but gap and decoy are always OPPOSITE. Randomising them
    # independently put the gap on the decoy's side in half the seeds, which collapsed the
    # angular separation to ~5 deg, let one look cone cover both, and ran the true route
    # straight through the decoy -- destroying four gates at once. Variety is worth having;
    # variety that breaks the design is not.
    # STRATIFIED, not sampled: alternating by seed guarantees an equal split of gap sides.
    # Drawing it at random gave 6 of one side and 2 of the other in the first 8 seeds, and the
    # two sides are NOT equally hard -- the planner breaks its search direction consistently,
    # so which side the gap is on dominates the null baseline's time. Balancing removes that
    # confound instead of hoping it averages out at the sample sizes a closed-loop study affords.
    side = 1.0 if seed % 2 == 0 else -1.0
    gap_y = side * float(rng.uniform(*GAP_Y))
    decoy_cy = -side * DECOY_ABS_CY
    # Randomised so no result is an artifact of a perfectly square approach.
    yaw0 = float(rng.uniform(-0.15, 0.15))

    XX, YY = _grid(XLIM, YLIM, cell)
    H = np.zeros_like(XX)

    band = np.abs(XX - WALL_X) <= WALL_HALF_DEPTH
    gap = band & (np.abs(YY - gap_y) <= GAP_HALF_WIDTH)
    wall = band & ~gap
    H[wall] = WALL_HEIGHT

    # Impassable perimeter. Without it the robot simply drove OFF the grid and around the end
    # of the barrier on the flat ground that edge-clamped sampling implies -- reaching the goal
    # without ever finding the gap, which would have made every policy look identical.
    edge = (
        (XX <= XLIM[0] + BORDER)
        | (XX >= XLIM[1] - BORDER)
        | (YY <= YLIM[0] + BORDER)
        | (YY >= YLIM[1] - BORDER)
    )
    H[edge] = WALL_HEIGHT

    decoy = (np.abs(XX - DECOY_CX) <= DECOY_HALF_X) & (np.abs(YY - decoy_cy) <= DECOY_HALF_Y)
    decoy &= ~band & ~edge  # the decoy is terrain, not a hole in the barrier or wall
    bumps = DECOY_RELIEF * (
        np.sin(3.1 * XX + 1.7 * seed) * np.cos(2.7 * YY) + 0.6 * np.cos(5.3 * XX - 2.1 * YY)
    )
    H[decoy] += bumps[decoy]

    return BenchWorld(
        scene=Heightmap(H, (XLIM[0], YLIM[0]), cell),
        wall_mask=wall,
        gap_mask=gap,
        decoy_mask=decoy,
        gap_y=gap_y,
        approach_yaw=yaw0,
        start=(START[0], START[1], yaw0),
        goal=GOAL,
        seed=seed,
    )


def cell_centres(world: BenchWorld) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = world.scene.H.shape
    xs = world.scene.x0 + (np.arange(nx) + 0.5) * world.scene.cell
    ys = world.scene.y0 + (np.arange(ny) + 0.5) * world.scene.cell
    return np.meshgrid(xs, ys)


def route_distance(world: BenchWorld) -> np.ndarray:
    """Perpendicular distance [m] of every cell from the TRUE route start -> gap -> goal.

    The straight line is the wrong reference here: the robot must pass through the gap, so the
    route bends. This is what certifies the decoy is beyond any possible wheel contact.
    """
    XX, YY = cell_centres(world)
    legs = [
        ((world.start[0], world.start[1]), (WALL_X, world.gap_y)),
        ((WALL_X, world.gap_y), world.goal),
    ]
    best = np.full(XX.shape, np.inf)
    for (ax, ay), (bx, by) in legs:
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        t = np.clip(((XX - ax) * dx + (YY - ay) * dy) / seg2, 0.0, 1.0)
        best = np.minimum(best, np.hypot(XX - (ax + t * dx), YY - (ay + t * dy)))
    return best


def bearing_to(world: BenchWorld, mask: np.ndarray, frm: tuple[float, float]) -> float:
    """Mean bearing [rad] from a point to the cells of `mask` -- where a policy would aim."""
    XX, YY = cell_centres(world)
    return float(np.arctan2((YY[mask] - frm[1]).mean(), (XX[mask] - frm[0]).mean()))


# =========================================================================================
# VARIANT B: an OPAQUE decision-critical feature.
#
# Variant A (the gap) produced a clean negative result: entropy finds the gap immediately,
# because an APERTURE is simultaneously the most decision-relevant and the most
# information-rich thing to look at -- sight passes through it, so a gap-directed look reveals
# the most area. Where the critical feature is a hole, the two objectives coincide by
# construction and no amount of tuning separates them.
#
# So variant B makes the critical feature OPAQUE and SMALL: a walled corridor on the direct
# route whose FLOOR is either drivable or blocked by a step, decided per seed. Looking at it
# reveals almost no new area -- the corridor walls see to that -- so an entropy-directed sensor
# has no reason to prefer it, while the plan drives straight through it and is maximally
# sensitive to its height. The decoy stays a large open unobserved region off to the side.
#
# If attribution beats entropy anywhere, it is here; if it does not beat entropy here either,
# that is a much stronger negative result than variant A alone.
# =========================================================================================

CORR_X = (6.6, 9.4)  # [m] the walled corridor on the direct route
CORR_HALF_WIDTH = 1.3  # [m] narrow enough that its walls occlude a look into it
CORR_WALL_H = 1.0
STEP_X_B = 8.6  # [m] where the blocking step sits, at the FAR end of the corridor
STEP_DEPTH_B = 0.6
STEP_H_B = (0.55, 0.75)  # [m] impassable when present
DETOUR_Y = 5.5  # [m] the way around, if the corridor turns out to be blocked


def build_corridor(seed: int = 0, cell: float = CELL) -> BenchWorld:
    """A walled corridor on the route that may or may not be blocked, plus the same decoy.

    Half the seeds are blocked and half are clear (stratified, not sampled). A policy that
    learns the corridor's state early can commit to the corridor or to the detour immediately;
    one that does not must drive in, discover, and back out.
    """
    rng = np.random.default_rng(1000 + seed)
    blocked = seed % 2 == 0
    step_h = float(rng.uniform(*STEP_H_B))
    side = 1.0 if (seed // 2) % 2 == 0 else -1.0  # which side the detour and decoy sit on
    yaw0 = float(rng.uniform(-0.10, 0.10))

    XX, YY = _grid(XLIM, YLIM, cell)
    H = np.zeros_like(XX)

    edge = (
        (XX <= XLIM[0] + BORDER)
        | (XX >= XLIM[1] - BORDER)
        | (YY <= YLIM[0] + BORDER)
        | (YY >= YLIM[1] - BORDER)
    )

    # A long barrier with ONE corridor through it, plus a detour opening far to `side`.
    band = (XX >= CORR_X[0]) & (XX <= CORR_X[1])
    inside = band & (np.abs(YY) <= CORR_HALF_WIDTH)
    detour = band & (np.abs(YY - side * DETOUR_Y) <= 1.4)
    H[band & ~inside & ~detour] = CORR_WALL_H

    critical = (np.abs(XX - STEP_X_B) <= STEP_DEPTH_B / 2) & (np.abs(YY) <= CORR_HALF_WIDTH)
    if blocked:
        H[critical] = step_h

    decoy_cy = -side * DECOY_ABS_CY
    decoy = (np.abs(XX - DECOY_CX) <= DECOY_HALF_X) & (np.abs(YY - decoy_cy) <= DECOY_HALF_Y)
    decoy &= ~band & ~edge
    bumps = DECOY_RELIEF * (
        np.sin(3.1 * XX + 1.7 * seed) * np.cos(2.7 * YY) + 0.6 * np.cos(5.3 * XX - 2.1 * YY)
    )
    H[decoy] += bumps[decoy]
    H[edge] = WALL_HEIGHT

    return BenchWorld(
        scene=Heightmap(H, (XLIM[0], YLIM[0]), cell),
        wall_mask=(H >= CORR_WALL_H - 1e-6) & ~edge,
        gap_mask=critical,  # "the cells the decision rests on" -- here the corridor floor
        decoy_mask=decoy,
        gap_y=side * DETOUR_Y if blocked else 0.0,
        approach_yaw=yaw0,
        start=(START[0], START[1], yaw0),
        goal=GOAL,
        seed=seed,
    )


# =========================================================================================
# VARIANT C: a COST decision, not a TOPOLOGY decision.
#
# Variants A and B both ask "is there a way through", which is a question about FEASIBILITY.
# That turned out to be the wrong question for this method, and the reason is structural:
# attribution weights cells by (dJ/dh * sigma)^2, and a per-cell Gaussian sigma cannot
# represent "there might be a wall here". Measured directly (see RESULTS.md section 6e): at
# sigma = 0.12 m sampled maps never contain a barrier and every sampled plan goes straight; at
# 0.3-0.8 m they contain rubble everywhere and NO route exists at all. A wall is a coherent
# 10 m object; smoothed per-cell noise is gravel.
#
# So variant C asks a question sigma CAN answer. Two routes around a central block, BOTH open,
# so the topology is never in doubt. One of them crosses rough ground that makes it slow and
# tilted; which one is randomised and hidden beyond default sensing range. The decision is
# purely "which open route is cheaper", and that is exactly what dJ/dh * sigma measures.
#
# The decoy is unchanged in role: a large unobserved region off both routes.
# =========================================================================================

# Two PARALLEL CHANNELS separated by a long wall, so the robot must COMMIT to one before it
# can see what is in it. An earlier layout put the rough patch beside a short block: the robot
# could see the lane entrance as it arrived at the fork, so the oracle saved only ~20 frames --
# less than the 32-frame look budget, leaving nothing for sensing to win. Commitment is what
# creates the stakes: enter the wrong channel and you either push through slowly or reverse out.
CHANNEL_X = (5.0, 12.0)  # [m] length of the divided section
DIVIDER_HALF_Y = 1.2  # central wall
OUTER_Y = 4.0  # outer walls run from here to the map edge -- the channels are the ONLY way
ROUGH_X = (8.0, 11.5)  # rough section, BEYOND default sensing range from the fork
ROUGH_AMPLITUDE = 0.38  # [m] passable but slow and tilted -- NOT a barrier
ROUGH_WAVELENGTH = 0.6  # short wavelength is what actually costs time
LANE_MID_Y = 2.6


def build_lanes(seed: int = 0, cell: float = CELL) -> BenchWorld:
    """Two open channels; one is rough beyond the point of commitment."""
    rng = np.random.default_rng(2000 + seed)
    rough_side = 1.0 if seed % 2 == 0 else -1.0  # stratified, as elsewhere
    yaw0 = float(rng.uniform(-0.08, 0.08))

    XX, YY = _grid(XLIM, YLIM, cell)
    H = np.zeros_like(XX)
    edge = (
        (XX <= XLIM[0] + BORDER)
        | (XX >= XLIM[1] - BORDER)
        | (YY <= YLIM[0] + BORDER)
        | (YY >= YLIM[1] - BORDER)
    )

    span = (XX >= CHANNEL_X[0]) & (XX <= CHANNEL_X[1])
    H[span & (np.abs(YY) <= DIVIDER_HALF_Y)] = WALL_HEIGHT
    # To the map edge. Bounded outer walls let the robot bypass the whole structure through
    # open ground beyond them, so the choice was never forced -- the null baseline went around
    # in 165 frames while the omniscient one entered a channel and took 372.
    H[span & (np.abs(YY) >= OUTER_Y)] = WALL_HEIGHT

    rough = (
        (XX >= ROUGH_X[0])
        & (XX <= ROUGH_X[1])
        & (np.abs(YY - rough_side * LANE_MID_Y) <= DIVIDER_HALF_Y)
    )
    ripple = (
        ROUGH_AMPLITUDE
        * np.sin(2 * np.pi * XX / ROUGH_WAVELENGTH)
        * np.cos(2 * np.pi * YY / (1.6 * ROUGH_WAVELENGTH))
    )
    H[rough] += ripple[rough]

    decoy_cy = -rough_side * (OUTER_Y + 2.6)
    decoy = (np.abs(XX - DECOY_CX) <= DECOY_HALF_X) & (np.abs(YY - decoy_cy) <= DECOY_HALF_Y)
    decoy &= ~span & ~edge
    bumps = DECOY_RELIEF * np.sin(3.1 * XX + 1.7 * seed) * np.cos(2.7 * YY)
    H[decoy] += bumps[decoy]
    H[edge] = WALL_HEIGHT

    return BenchWorld(
        scene=Heightmap(H, (XLIM[0], YLIM[0]), cell),
        wall_mask=span & (np.abs(YY) <= DIVIDER_HALF_Y),
        gap_mask=rough,  # "the cells the decision rests on" -- the rough channel
        decoy_mask=decoy,
        gap_y=-rough_side * LANE_MID_Y,  # the GOOD channel
        approach_yaw=yaw0,
        start=(START[0], START[1], yaw0),
        goal=GOAL,
        seed=seed,
    )
