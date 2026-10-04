"""Camera model: lens calibration, distortion, projection and GCP solve.

``find_uv(beta, xyz, lcp)`` is the world-to-image projection: it returns the
distorted pixel coordinates ``(N, 2)`` of world points ``xyz`` ``(N, 3)``.
"""

from .distortion import distort, undistort
from .geometry import (P2m, angles2R, expand_beta, find_uv, find_xyz, find_xyz_6dof,
                       lcp_beta2P)
from .lens import LensCalibration, make_lcp
from .solve import Geometry, PoseFit, fit_pose, focal_lengths_for_fov, solve_geometry

__all__ = [
    "LensCalibration", "make_lcp", "distort", "undistort", "angles2R", "lcp_beta2P",
    "P2m", "expand_beta", "find_uv", "find_xyz", "find_xyz_6dof", "Geometry", "PoseFit",
    "fit_pose", "focal_lengths_for_fov", "solve_geometry",
]
