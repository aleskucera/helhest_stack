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
    # SLOW IN NARROW PLACES. 0 = the spatial tube vetoes (plan_robust_margin_m removes poses).
    # > 0 = it only marks them narrow: the router charges plan_narrow_cost per unit penalty
    # (x flatness_weight 2 = extra cost per metre) and MPPI charges plan_narrow_weight *
    # (|v| - plan_narrow_speed)^2 per rollout step held there.
    "plan_narrow_speed": 0.0,
    "plan_narrow_weight": 100.0,
    "plan_narrow_cost": 0.15,
    # [m] how far from a wall the narrow route charge reaches, grading down to zero -- the pull
    # toward the middle of a passage (CostToGo._narrow_kernel)
    "plan_narrow_reach_m": 0.6,
    # CAREFUL WHERE IT IS TIGHT (control/governor.py). plan_clear_t_react > 0 replaces the spatial
    # tube's veto with the clearance speed law v = clearance / t_react (floored at v_min): the route
    # is priced in travel time under it, capped at v_cruise, and a governor after MPPI enforces it
    # on the next plan_clear_lookahead_s of the plan. 0 = off (the spatial tube vetoes).
    # t_react is seconds of error at the body's fastest-point speed (~0.125 s: see the module).
    "plan_clear_t_react": 0.0,
    "plan_clear_v_cruise": 1.5,
    "plan_clear_v_min": 0.15,
    "plan_clear_lookahead_s": 1.0,
    # [cost per second] MPPI's price for the time the governor would add to a manoeuvre, so it
    # picks one with room instead of one that has to be braked (CostWeights.clear_time)
    "plan_clear_mppi_weight": 100.0,
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
    governor: (
        dict[str, float] | None
    )  # t_react, v_min, lookahead_s -- ClearanceGovernor; None = off


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
    # slow-in-narrow: absent entirely when off, so the off state IS the configuration before it
    narrow_on = float(p["plan_narrow_speed"]) > 0.0
    clear_on = float(p["plan_clear_t_react"]) > 0.0
    if clear_on:
        narrow_on = False  # the governor supersedes narrow mode
    time_ctg_kw = (
        dict(
            time_cost=(
                float(p["plan_clear_v_cruise"]),
                float(p["plan_clear_t_react"]),
                float(p["plan_clear_v_min"]),
            )
        )
        if clear_on
        else {}
    )
    narrow_ctg_kw = (
        dict(
            narrow_cost=float(p["plan_narrow_cost"]),
            narrow_reach_m=float(p["plan_narrow_reach_m"]),
        )
        if narrow_on
        else {}
    )
    narrow_cost_kw = (
        dict(narrow=float(p["plan_narrow_weight"]), narrow_speed=float(p["plan_narrow_speed"]))
        if narrow_on
        else {}
    )
    if clear_on:
        narrow_cost_kw = dict(
            clear_time=float(p["plan_clear_mppi_weight"]),
            clear_t_react=float(p["plan_clear_t_react"]),
            clear_v_min=float(p["plan_clear_v_min"]),
        )
    return PlannerConfig(
        cost=CostParams(
            goal_running=float(p["plan_goal_running"]),
            effort=float(p["plan_effort"]),
            turn=float(p["plan_turn"]),
            smoothness=float(p["plan_smooth"]),
            saturation=float(p["plan_saturation"]),
            veto=float(p["plan_wall_veto"]),
            **narrow_cost_kw,
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
            pivot_frac=0.05 if wmin < 0.0 else 0.0,
            elite_frac=float(p["plan_elite_frac"]),
            n_mu=n_mu,
        ),
        costtogo=dict(
            n_theta=int(p["plan_n_theta"]),
            robust_margin_m=float(p["plan_robust_margin_m"]),
            robust_margin_deg=float(p["plan_robust_margin_deg"]),
            obstacle_step_m=float(p["plan_obstacle_step_m"]),
            pivot_cost=float(p["plan_pivot_cost"]),
            **narrow_ctg_kw,
            **time_ctg_kw,
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
        governor=(
            dict(
                t_react=float(p["plan_clear_t_react"]),
                v_min=float(p["plan_clear_v_min"]),
                lookahead_s=float(p["plan_clear_lookahead_s"]),
            )
            if clear_on
            else None
        ),
    )
