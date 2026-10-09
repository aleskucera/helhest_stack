"""Perception: the scan entry path (transform, z / self / range gates), the outlier filter, and the
inpainting the belief map is planned on (`belief_frame.BeliefFrame`). The mapping itself is the
`elevation_belief` package; pose helpers live in `helhest.localization.pose_math`.
"""

from .cloud_ops import ScanPreprocessor
from .cloud_ops import transform_points
from .heightmap import diffuse_inpaint
from .heightmap import gaussian_smooth
from .heightmap import multigrid_inpaint
from .outlier import OutlierFilterConfig
from .outlier import StatisticalOutlierFilter

__all__ = [
    "OutlierFilterConfig",
    "ScanPreprocessor",
    "StatisticalOutlierFilter",
    "diffuse_inpaint",
    "gaussian_smooth",
    "multigrid_inpaint",
    "transform_points",
]
