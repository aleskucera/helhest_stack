"""`planner_config` must reproduce what the node built inline, field for field.

The golden below is a transcription of `elevation_node._build_planner` as it stood before the
mapping moved into `helhest.planner_config`. If the two ever disagree on a field, the node and the
simulator have started running different controllers again -- which is the bug this module exists
to end: four configurations, and a simulator validating one the robot never ran.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re

import pytest

from helhest.control.mppi import CostParams
from helhest.control.mppi import SamplingConfig
from helhest.planner_config import PLAN_DEFAULTS
from helhest.planner_config import planner_config

REPO = pathlib.Path(__file__).resolve().parents[1]
NODE = REPO / "ros/helhest_stack_ros/helhest_stack_ros/elevation_node.py"
PARAMS = REPO / "ros/odin/odin_elevation.params.yaml"


def _golden(p: dict) -> tuple[CostParams, SamplingConfig, dict]:
    """The node's construction, verbatim in substance, from before the move."""
    cost = CostParams(
        goal_running=p["plan_goal_running"],
        effort=p["plan_effort"],
        turn=p["plan_turn"],
        smoothness=p["plan_smooth"],
        saturation=p["plan_saturation"],
        # added after the move: the wall veto, which the node now sets from the table
        veto=p["plan_wall_veto"],
        # added after the move: the clearance governor's MPPI term, present only when it is on
        **(
            dict(
                clear_time=p["plan_clear_mppi_weight"],
                clear_t_react=p["plan_clear_t_react"],
                clear_v_min=p["plan_clear_v_min"],
                clear_c0=p["plan_clear_c0"],
                clear_v_cruise=p["plan_clear_v_cruise"],
            )
            if p["plan_clear_t_react"] > 0.0
            else {}
        ),
    )
    sampling = SamplingConfig(
        wmax=p["plan_wmax"],
        wmin=min(0.0, p["plan_wmin"]),
        straight_frac=p["plan_straight_frac"],
        spin_frac=p["plan_spin_frac"],
        spin_min=p["plan_spin_min"],
        pivot_frac=0.05 if p["plan_wmin"] < 0.0 else 0.0,
        elite_frac=p["plan_elite_frac"],
        n_mu=max(1, int(p["plan_n_mu"])),
    )
    ctg = dict(
        n_theta=int(p["plan_n_theta"]),
        robust_margin_m=p["plan_robust_margin_m"],
        robust_margin_deg=p["plan_robust_margin_deg"],
        obstacle_step_m=p["plan_obstacle_step_m"],
        pivot_cost=p["plan_pivot_cost"],
    )
    if p["plan_clear_t_react"] > 0.0:  # added after the move: the route priced in travel time
        ctg["time_cost"] = (
            p["plan_clear_v_cruise"],
            p["plan_clear_t_react"],
            p["plan_clear_v_min"],
            p["plan_clear_c0"],
        )
    return cost, sampling, ctg


def _deployed() -> dict:
    """The robot's params file, read without PyYAML (not a helhest dependency): it is flat."""
    vals = {}
    for m in re.finditer(r"^\s+(plan_\w+):\s*([^\s#]+)", PARAMS.read_text(), re.M):
        raw = m.group(2)
        vals[m.group(1)] = {"true": True, "false": False}.get(raw.lower(), None)
        if vals[m.group(1)] is None:
            vals[m.group(1)] = float(raw) if any(c in raw for c in ".eE") else int(raw)
    return {**PLAN_DEFAULTS, **{k: v for k, v in vals.items() if k in PLAN_DEFAULTS}}


CASES = {
    "defaults": dict(PLAN_DEFAULTS),
    "deployed params file": _deployed(),
    "reverse enabled": {**PLAN_DEFAULTS, "plan_wmin": -2.0, "plan_n_mu": 3},
}


@pytest.mark.parametrize("name", list(CASES))
def test_planner_config_reproduces_the_node(name):
    p = CASES[name]
    cost, sampling, ctg = _golden(p)
    cfg = planner_config(p)
    assert dataclasses.asdict(cfg.cost) == dataclasses.asdict(cost)
    assert dataclasses.asdict(cfg.sampling) == dataclasses.asdict(sampling)
    assert cfg.costtogo == ctg
    assert cfg.mu_span == (p["plan_mu_span"] if int(p["plan_n_mu"]) > 1 else 0.0)


def test_the_deployed_file_is_actually_read():
    """Guards the test above against passing vacuously on a file nobody parsed."""
    d = _deployed()
    assert d["plan_turn"] == 0.2 and d["plan_wmax"] == 6.0 and d["plan_n_mu"] == 3


def test_the_node_takes_every_default_from_the_table():
    """Each shared parameter must be declared with PLAN_DEFAULTS[...] rather than a literal, or a
    bare launch and the simulator can quietly disagree again."""
    src = NODE.read_text()
    missing = [k for k in PLAN_DEFAULTS if f'PLAN_DEFAULTS["{k}"]' not in src]
    assert not missing, f"declared with a literal, not the table: {missing}"


def test_the_batch_is_one_the_friction_replicas_divide():
    """The deployed file sets plan_n_mu 3 and no plan_batch, over a 4096 default that 3 does not
    divide; MppiGpu raises on that. Read as written, the robot's own params could not start the
    planner."""
    cfg = planner_config(_deployed())
    assert cfg.sampling.n_mu == 3 and cfg.batch % 3 == 0 and cfg.batch == 4095
    assert planner_config(PLAN_DEFAULTS).batch == 4096  # n_mu 1: untouched


def test_the_coarse_layer_and_the_turn_first_brake_come_from_the_table():
    """Both were built in the simulator first, as flags; the node reads them from the same table
    the simulator now falls back to, so a false_door result in the sim is a result for the node."""
    cfg = planner_config(PLAN_DEFAULTS)
    assert cfg.coarse == dict(block_m=0.6, memory_m=60.0, bridge_m=1.2)
    assert cfg.turn_first == dict(start_deg=45.0, reach_m=1.5)
    off = planner_config({**PLAN_DEFAULTS, "plan_coarse_block_m": 0.0, "plan_turn_first_deg": 0.0})
    assert off.coarse["block_m"] == 0.0 and off.turn_first["start_deg"] == 0.0
