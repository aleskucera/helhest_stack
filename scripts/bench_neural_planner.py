"""Re-plan latency of the neural planner (helhest/control/neural.py) on this machine's GPU, without ROS.

One robot on a 9 m window with a 0.25 m ridge 3 m ahead, a straight-line cost-to-go to a goal 8 m ahead.
Times the whole captured re-plan (shoot, 20 Adam steps, pick) for cuBLAS GEMMs (if torch imports) and Warp tile
GEMMs, and prints the first command so a broken install shows up as nonsense rather than a fast number.

  PYTHONPATH=<helhest_stack>/src python scripts/bench_neural_planner.py WEIGHTS.npz [--reps 30]

Run it while navigation_node is stopped: the two share the GPU and the numbers would mix.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from helhest.control import neural


def bench(weights: str, gemm: str, steps: int, reps: int) -> None:
    n = int(round(9.0 / neural.MODEL_CELL))
    planner = neural.NeuralPlanner(weights, n, steps=steps, gemm=gemm)
    g = np.arange(n) * neural.MODEL_CELL
    X, Y = np.meshgrid(g, g)
    x0, y0 = 1.5, 4.5  # the robot, facing +x
    H = np.where(np.abs(X - (x0 + 3.0)) < 0.3, 0.25, 0.0)
    planner.H.assign(H)
    planner.VA.assign(np.hypot(X - (x0 + 8.0), Y - y0))
    planner.DA.assign(np.full((n, n), 10.0))
    S = np.zeros((1, 16))
    S[0, :2] = x0, y0
    S[0, 6] = 1.0
    planner.S.assign(S)
    planner.U0.assign(np.full((neural.N_SEG, 1, 2), 0.5, np.float32))
    t0 = time.perf_counter()
    planner.launch()
    first = time.perf_counter() - t0
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        planner.launch()
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    u0 = planner.OUT.numpy()[0, 0] * neural.WHEEL_MAX
    end = planner.KNOTS.numpy()[-1, 0]
    print(
        f"{gemm:6s} {steps:2d} steps: {1e3 * np.median(times):6.1f} ms per re-plan (median of {reps}, min "
        f"{1e3 * min(times):.1f}); first call {first:.1f} s; first command (wL, wR) "
        f"{u0.round(2)} rad/s; mean path end ({end[0] - x0:.2f}, {end[1] - y0:.2f}) m"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--steps", type=int, nargs="+", default=[0, 5, 10, 20], help="Adam steps")
    a = p.parse_args()
    wp.init()
    print("device", wp.get_device().name)
    gemms = ["cublas", "tile"] if neural._import_torch() else ["tile"]
    for gemm in gemms:
        for steps in a.steps:
            bench(a.weights, gemm, steps, a.reps)


if __name__ == "__main__":
    main()
