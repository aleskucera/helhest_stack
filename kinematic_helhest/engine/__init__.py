"""The Warp/CUDA runtime engine — the differentiable kinematic simulator.

This is the main code path (the numpy `reference/` package is the verification
oracle only). Public API:

    from kinematic_helhest.engine import (
        RobotParams, SolverParams, to_terrain, init_state, step,
    )

`init_state`/`step`/`settle`/`clearances` are Warp kernels/funcs launched with
`wp.launch`; `to_terrain` uploads a numpy `Heightmap` to the device. The implicit
settle adjoint (`@wp.func_grad(settle)`) lives in `engine.step` and registers on
import, so gradients work automatically. The oracle/FD verification harness lives
in the top-level `tests/engine/` package (run e.g. `python -m tests.engine.step`).
"""
from .step import clearances
from .step import init_state
from .step import settle
from .step import step
from .terrain import GridParams
from .simulator import Simulator
from .robot import Robot
from .robot import RobotParams
from .step import Solver
from .step import SolverParams
from .terrain import bounds_to_origin
from .terrain import Grid
from .terrain import sample_height
from .terrain import sample_height_grad
from .terrain import sample_normal
from .terrain import Terrain
from .terrain import terrain_from_device
from .terrain import to_terrain
