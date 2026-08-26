"""The Warp contact estimator MOVED to `helhest.risk.contact`; this is a re-export.

The estimator was wired into `helhest.control.mppi.MppiGpu`, and `src/` imports nothing from
`studies/` -- so leaving it here would have made a deployed robot need the research tree on its
path. The kernels are unchanged; the three settle-cost weights are now a constructor parameter
(`deriv_w`, defaulting to the study's `(1.0, 0.7, 0.5)`) instead of an import from
`studies/adjoint/harness.py`, and `MAX_K` is 16 rather than 8 so the same build covers the
perception node's 0.08 m cell, where the cylinder needs K up to 13.

This shim exists so the committed measurement scripts (`clark_warp_verify.py`,
`clark_gpu_vs_mc_gpu.py`) keep running unedited against ONE definition of the estimator.
"""

from helhest.risk.contact import DERIV_W_DEFAULT  # noqa: F401
from helhest.risk.contact import k_conv_x  # noqa: F401
from helhest.risk.contact import k_conv_y  # noqa: F401
from helhest.risk.contact import k_finish  # noqa: F401
from helhest.risk.contact import k_fold_scatter  # noqa: F401
from helhest.risk.contact import k_nodes  # noqa: F401
from helhest.risk.contact import k_quad  # noqa: F401
from helhest.risk.contact import MAX_K  # noqa: F401
from helhest.risk.contact import VECF  # noqa: F401
from helhest.risk.contact import VECI  # noqa: F401
from helhest.risk.contact import WarpContactEstimator  # noqa: F401
