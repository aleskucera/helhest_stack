"""Small shared helpers for driving the sim from plain control arrays in tests."""
import numpy as np


def _to_target_wheel_omega(Ub):
    """Controls [B, T, 2] (wL, wR) -> target_wheel_omega [T, B, 3] (rear = mean). The host-side
    equivalent of the packing the GPU sampling kernel does on-device; used only to feed the
    ForwardSimulator from a numpy control array in tests."""
    bt = np.transpose(Ub, (1, 0, 2))  # [T, B, 2]
    rear = bt.mean(2, keepdims=True)
    return np.concatenate([bt, rear], axis=2).astype(np.float32)
