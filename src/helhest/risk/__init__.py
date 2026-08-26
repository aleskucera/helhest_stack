"""Uncertainty propagation over the height map: what the belief's error does to the robot's pose.

`WarpContactEstimator` turns a belief + a per-cell sigma into E[J] and Var[J] for a whole batch
of candidate trajectories at once, on device, in six kernel launches. `map_sigma` builds the
sigma the estimator consumes. `MppiGpu` is the consumer: `Var[J]` becomes the risk term in its
cost, so a plan that is merely UNCERTAIN can be ranked below one that is equally good in
expectation but stands on ground the robot has actually seen.
"""

from .contact import DERIV_W_DEFAULT
from .contact import MAX_K
from .contact import WarpContactEstimator
from .settle import settle_map
from .sigma import gauss_kernel
from .sigma import map_sigma
from .sigma import rho1_table
from .sigma import SigmaParams

__all__ = [
    "DERIV_W_DEFAULT",
    "MAX_K",
    "WarpContactEstimator",
    "gauss_kernel",
    "map_sigma",
    "rho1_table",
    "settle_map",
    "SigmaParams",
]
