"""The ONE place `plan_*` parameters become planner objects.

Before this there were four configurations and they disagreed: `mppi.py`'s dataclass defaults,
the ROS node's declared defaults, the deployed params file, and `studies/closed_loop/drive_sim.py`,
which built its planner from library defaults. So every stress-world result validated a controller
the robot does not run -- `turn` 0.0 against the robot's 0.2, no spin prior, one friction replica
instead of three -- and the bug that stopped the robot turning around was invisible in simulation
because it lived in a value the simulation never set.

Now the node declares its parameters with these defaults and builds its planner through
`planner_config`, and so does `drive_sim`, reading the same params file the robot does. Rationale
for individual values stays beside their declarations in `elevation_node.py`; only the numbers live
here, so there is one of each.

No ROS and no YAML here: the node passes its cached values, `drive_sim` parses the file itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from helhest.control.mppi import CostParams
from helhest.control.mppi import SamplingConfig

# Everything the planner reads that the node and the simulator must agree on. Grid geometry and the
# command chain after MPPI (turn boost, goal brake, slew, yaw loop, consistency EMA) are NOT here:
# the simulator deliberately uses its own windows, and it does not run the command chain at all.
PLAN_DEFAULTS: dict[str, Any] = {
    # cost weights -> CostParams
    "plan_goal_running": 0.3,
    "plan_effort": 1e-3,
    "plan_turn": 0.03,
    "plan_smooth": 0.04,
    "plan_saturation": 300.0,
    # MPPI's hard veto on the cost-to-go's HAZARD field: actual wall contact, without the router's
    # margin, and never tilt. 1e6 is ten times the rollout-infeasibility weight. 0 = off.
    "plan_wall_veto": 1e6,
    # sampler -> SamplingConfig
    "plan_wmax": 4.0,
    "plan_wmin": 0.0,
    # Reverse, when plan_wmin < 0. The shaping is per metre reversed: 75 (the library default)
    # let the robot back 6-7 m out of a room it could have turned in (false_door 2/3), 200
    # left it unable to back 3 m out of a 2.2 m dead end (narrow_corridor 1/3); 120 does both
    # (3/3 each, corridor 3/3). The pivot prior that used to come on with reverse is OFF: it
    # took pocket from 3/3 to 0/3 (the robot froze in a 92%-blocked pose beside a corner).
    "plan_reverse_cost": 120.0,
    "plan_straight_frac": 0.2,
    "plan_spin_frac": 0.12,
    "plan_spin_min": 2.0,
    "plan_elite_frac": 0.01,
    "plan_n_mu": 1,
    # MPPI run
    "plan_n_theta": 24,
    "plan_horizon": 25,
    "plan_batch": 4096,
    "plan_nominal_reset": 1.5,
    "plan_mu_span": 0.25,
    # cost-to-go
    "plan_robust_margin_m": 0.3,
    "plan_robust_margin_deg": 0.0,
    "plan_obstacle_step_m": 0.0,
    "plan_pivot_cost": 0.0,
    # robot
    "plan_wheel_width": 0.10,
    # the coarse "which way" layer (planning/coarse.py) and the turn-first brake
    # (control/command.turn_first). Block size 0 = no coarse layer; memory 0 = a layer bound to the
    # window it is pooled from, which forgets what scrolls out of it; turn-first 0 = off.
    "plan_coarse_block_m": 0.6,
    "plan_coarse_memory_m": 60.0,
    "plan_bridge_m": 1.2,
    "plan_turn_first_deg": 45.0,
    "plan_turn_first_reach_m": 1.5,
}


@dataclass(frozen=True)
class PlannerConfig:
    cost: CostParams
    sampling: SamplingConfig
    costtogo: dict[str, Any]  # CostToGo kwargs beyond grid / robot / solver / device
    n_theta: int
    horizon: int
    batch: int
    nominal_reset: float
    mu_span: float  # the band MppiGpu.set_mu_band gets: 0 with a single friction replica
    wheel_width: float
    coarse: dict[str, float]  # block_m, memory_m, bridge_m -- CoarseRouter, sized by the caller
    turn_first: dict[str, float]  # start_deg, reach_m -- control.command.turn_first


def resolve(params: Mapping[str, Any]) -> dict[str, Any]:
    """Defaults overlaid with `params`, restricted to the keys above -- the flat dict worth saving
    next to a result, so the result can say which controller produced it."""
    return {**PLAN_DEFAULTS, **{k: v for k, v in params.items() if k in PLAN_DEFAULTS}}


def planner_config(params: Mapping[str, Any]) -> PlannerConfig:
    """`plan_*` values -> the objects the node and the simulator both build their planner from."""
    p = resolve(params)
    wmin = float(p["plan_wmin"])
    n_mu = max(1, int(p["plan_n_mu"]))
    # MppiGpu refuses a batch its friction replicas do not divide, and the deployed params set
    # plan_n_mu 3 over the 4096 default -- so that file, read as written, could not start the node
    # (sim-demo carried a hand-written `-p plan_batch:=4095` to get round it). Round down instead:
    # it only changes configurations that could not start at all.
    batch = int(p["plan_batch"])
    batch -= batch % n_mu
    return PlannerConfig(
        cost=CostParams(
            goal_running=float(p["plan_goal_running"]),
            effort=float(p["plan_effort"]),
            turn=float(p["plan_turn"]),
            smoothness=float(p["plan_smooth"]),
            saturation=float(p["plan_saturation"]),
            veto=float(p["plan_wall_veto"]),
            reverse=float(p["plan_reverse_cost"]),
        ),
        sampling=SamplingConfig(
            wmax=float(p["plan_wmax"]),
            # only the box the planner MAY use; the node gates the effective floor per frame on
            # map coverage behind the robot
            wmin=min(0.0, wmin),
            straight_frac=float(p["plan_straight_frac"]),
            # not conditioned on wmin: the spin band is exempt from the wmin clamp on purpose
            spin_frac=float(p["plan_spin_frac"]),
            spin_min=float(p["plan_spin_min"]),
            # no pivot prior with reverse: measured to freeze pocket (see plan_reverse_cost)
            pivot_frac=0.0,
            elite_frac=float(p["plan_elite_frac"]),
            n_mu=n_mu,
        ),
        costtogo=dict(
            n_theta=int(p["plan_n_theta"]),
            robust_margin_m=float(p["plan_robust_margin_m"]),
            robust_margin_deg=float(p["plan_robust_margin_deg"]),
            obstacle_step_m=float(p["plan_obstacle_step_m"]),
            pivot_cost=float(p["plan_pivot_cost"]),
        ),
        n_theta=int(p["plan_n_theta"]),
        horizon=int(p["plan_horizon"]),
        batch=batch,
        nominal_reset=float(p["plan_nominal_reset"]),
        mu_span=float(p["plan_mu_span"]) if n_mu > 1 else 0.0,
        wheel_width=float(p["plan_wheel_width"]),
        coarse=dict(
            block_m=float(p["plan_coarse_block_m"]),
            memory_m=float(p["plan_coarse_memory_m"]),
            bridge_m=float(p["plan_bridge_m"]),
        ),
        turn_first=dict(
            start_deg=float(p["plan_turn_first_deg"]),
            reach_m=float(p["plan_turn_first_reach_m"]),
        ),
    )
