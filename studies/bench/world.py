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
LOOK_RANGE = 12.0  # [m] a dwell buys range
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
DECOY_ABS_CY = 6.5  # [m] always OPPOSITE the gap -- see build()
DECOY_HALF_X = 2.5
DECOY_HALF_Y = 2.5
# Gentle relief, not rugged. It must read as genuinely uncertain terrain, but rough ground
# SELF-OCCLUDES, which suppresses how much area a look actually reveals -- and an
# entropy-directed sensor counts revealed area. With sharp relief the decoy stopped being
# tempting at all and gate 6 failed, which would have quietly voided the whole comparison.
DECOY_RELIEF = 0.12

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
