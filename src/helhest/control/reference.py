"""Numpy helper for the MPPI control packing (the GPU planner lives in mppi).

`_to_target_wheel_omega` packs the [B, T, 2] wheel-speed controls into the engine's [T, B, 3]
layout. (The GPU cost kernel is no longer differential-tested against a numpy twin: a hand-written
copy only proves the two AGREE, not that either is correct, and it taxes the hottest code with a
sync burden. tests/control/test_mppi.py validates the kernel by CONTRACT instead -- analytic
cost/sample/fallback checks against hand-computed values from the real Robot envelope.)
"""

import numpy as np


def _to_target_wheel_omega(Ub):
    """Controls [B, T, 2] (wL, wR) -> target_wheel_omega [T, B, 3] (rear = mean)."""
    bt = np.transpose(Ub, (1, 0, 2))  # [T, B, 2]
    rear = bt.mean(2, keepdims=True)
    return np.concatenate([bt, rear], axis=2).astype(np.float32)
