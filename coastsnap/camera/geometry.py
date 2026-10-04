"""Camera projection geometry (port of angles2R, lcpBeta2P, P2m, findUVnDOF,
findXYZ and findXYZ6dof).

``beta`` is always the full six degree-of-freedom extrinsic vector
``[x_cam, y_cam, z_cam, azimuth, tilt, roll]`` with angles in radians, in the
same local coordinate frame as the world points.  Use :func:`expand_beta` to
build it from a partial set of unknowns plus knowns, as MATLAB's
``knownFlags``/``knowns`` globals did.
"""

from __future__ import annotations

import numpy as np

from .distortion import distort, undistort
from .lens import LensCalibration


def angles2R(a: float, t: float, s: float) -> np.ndarray:
    """Rotation matrix from azimuth, tilt and swing/roll (Wolf 1983, p. 612)."""
    ca, sa, ct, st, cs, ss = np.cos(a), np.sin(a), np.cos(t), np.sin(t), np.cos(s), np.sin(s)
    return np.array([
        [ca * cs + sa * ct * ss, -cs * sa + ss * ct * ca, ss * st],
        [-ss * ca + cs * ct * sa, ss * sa + cs * ct * ca, cs * st],
        [st * sa, st * ca, -ct],
    ])


def _K(lcp: LensCalibration) -> np.ndarray:
    return np.array([[lcp.fx, 0, lcp.c0u], [0, -lcp.fy, lcp.c0v], [0, 0, 1.0]])


def lcp_beta2P(lcp: LensCalibration, beta) -> np.ndarray:
    """3x4 projection matrix from intrinsics and 6-DOF extrinsics."""
    beta = np.asarray(beta, dtype=float).ravel()
    R = angles2R(beta[3], beta[4], beta[5])
    IC = np.hstack([np.eye(3), -beta[:3, None]])
    P = _K(lcp) @ R @ IC
    return P / P[2, 3]


def P2m(P: np.ndarray) -> np.ndarray:
    """Projection matrix to the 11-element DLT ``m`` vector (Walton notation)."""
    return np.concatenate([P[0, :], P[2, :3], P[1, :]])


def expand_beta(unknowns, known_flags, knowns) -> np.ndarray:
    """Fill a 6-vector from the unknowns, as findUVnDOF does with its globals."""
    flags = np.asarray(known_flags, dtype=bool).ravel()
    unknowns = np.asarray(unknowns, dtype=float).ravel()
    if unknowns.size != (~flags).sum():
        raise ValueError("Length of beta0 must equal length of unknowns in knownFlags")
    b = np.empty(6)
    b[flags] = np.asarray(knowns, dtype=float).ravel()
    b[~flags] = unknowns
    return b


def find_uv(beta, xyz, lcp: LensCalibration, cull_behind: bool = True) -> np.ndarray:
    """Distorted image coordinates ``(N, 2)`` of world points ``xyz`` ``(N, 3)``.

    Port of ``findUVnDOF`` (which returns the same values stacked as
    ``[U; V]``).  With ``cull_behind`` (the default) points behind the camera
    give NaN; MATLAB projects them through the camera centre and they can land
    on the image as mirrored pixels.
    """
    beta = np.asarray(beta, dtype=float).ravel()
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    P = lcp_beta2P(lcp, beta)
    h = P @ np.vstack([xyz.T, np.ones(xyz.shape[0])])
    with np.errstate(invalid="ignore", divide="ignore"):
        U = h[0] / h[2]
        V = h[1] / h[2]
    if cull_behind:
        R = angles2R(beta[3], beta[4], beta[5])
        depth = (xyz - beta[:3]) @ R[2]
        U = np.where(depth > 0, U, np.nan)
        V = np.where(depth > 0, V, np.nan)
    Ud, Vd = distort(U, V, lcp)
    return np.column_stack([Ud, Vd])


def find_xyz(m, uv, val, flag: int) -> np.ndarray:
    """World ``(N, 3)`` points from image coordinates with one coordinate known.

    Port of ``findXYZ``: ``flag`` is 1, 2 or 3 for known x, y or z, and ``val``
    is a scalar or one value per point.  ``uv`` must be undistorted.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    if uv.shape[1] != 2:
        raise ValueError("[U V] values must be entered as rows")
    n = uv.shape[0]
    val = np.broadcast_to(np.asarray(val, dtype=float).ravel(), (n,)) if np.size(val) in (1, n) else None
    if val is None:
        raise ValueError("array of val not same length as UV list")
    U, V = uv[:, 0], uv[:, 1]
    A, B, C, D, E, F, G, H, J, K, L = np.asarray(m, dtype=float).ravel()
    M = E * U - A
    N = F * U - B
    O = G * U - C
    P = D - U
    Q = E * V - H
    R = F * V - J
    S = G * V - K
    T = L - V
    with np.errstate(invalid="ignore", divide="ignore"):
        if flag == 1:
            X = val
            Y = ((O * Q - S * M) * X + (S * P - O * T)) / (S * N - O * R)
            Z = ((N * Q - R * M) * X + (R * P - N * T)) / (R * O - N * S)
        elif flag == 2:
            Y = val
            X = ((O * R - S * N) * Y + (S * P - O * T)) / (S * M - O * Q)
            Z = ((M * R - Q * N) * Y + (Q * P - M * T)) / (Q * O - M * S)
        elif flag == 3:
            Z = val
            X = ((N * S - R * O) * Z + (R * P - N * T)) / (R * M - N * Q)
            Y = ((M * S - Q * O) * Z + (Q * P - M * T)) / (Q * N - M * R)
        else:
            raise ValueError("flag must be 1, 2 or 3")
    return np.column_stack([X, Y, Z])


def find_xyz_6dof(u, v, z, beta, lcp: LensCalibration) -> np.ndarray:
    """World points at elevation ``z`` seen at distorted pixels ``(u, v)``.

    Port of ``findXYZ6dof``; used for virtual GCPs.
    """
    uu, vu = undistort(u, v, lcp)
    m = P2m(lcp_beta2P(lcp, beta))
    return find_xyz(m, np.column_stack([uu, vu]), z, 3)


def make_uv_projector(geom: dict, model: str = "analytic"):
    """A ``project_uv(xyz) -> (N, 2)`` callable from a saved ``metadata["geom"]``.

    ``geom`` is the struct CSPGrectifyImage saves (``betas`` and ``lcp``), as
    loaded with ``scipy.io.loadmat(..., simplify_cells=True)``.
    """
    beta = np.asarray(geom["betas"], dtype=float).ravel()
    lcp = LensCalibration.from_matlab(geom["lcp"], model=model)
    return lambda xyz: find_uv(beta, xyz, lcp)
