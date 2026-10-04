"""Lens distortion (port of distortCaltech.m and undistortCaltech.m).

Both functions take and return pixel coordinates.  Points that cannot be
mapped (outside the calibration tables in ``"table"`` mode, or beyond the
fold-back radius in ``"analytic"`` mode) come back as NaN, which the
rectification treats as off-screen.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .lens import LensCalibration


def _interp1(xp, fp, x):
    """MATLAB ``interp1(xp, fp, x)``: linear, NaN outside ``[xp[0], xp[-1]]``."""
    x = np.asarray(x, dtype=float)
    out = np.interp(x, xp, fp)
    out[(x < xp[0]) | (x > xp[-1]) | np.isnan(x)] = np.nan
    return out


def _interp2(gx, gy, Z, x, y):
    """MATLAB ``interp2(gx, gy, Z, x, y)``: linear, NaN outside the grid."""
    f = RegularGridInterpolator((gy, gx), Z, bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([np.ravel(y), np.ravel(x)])
    good = ~np.isnan(pts).any(axis=1)
    out = np.full(pts.shape[0], np.nan)
    out[good] = f(pts[good])
    return out.reshape(np.shape(x))


def _tangential(lcp: LensCalibration, x, y, r2):
    dx = 2 * lcp.t1 * x * y + lcp.t2 * (r2 + 2 * x * x)
    dy = lcp.t1 * (r2 + 2 * y * y) + 2 * lcp.t2 * x * y
    return dx, dy


def distort(u, v, lcp: LensCalibration):
    """Undistorted pixel coordinates to distorted (what the camera recorded)."""
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    x = (u - lcp.c0u) / lcp.fx
    y = (v - lcp.c0v) / lcp.fy
    r2 = x * x + y * y
    if lcp.model == "table":
        r_tab, fr_tab = lcp.rad_table
        gx, gy, tdx, tdy = lcp.tang_table
        fr = _interp1(r_tab, fr_tab, np.sqrt(r2))
        dx = _interp2(gx, gy, tdx, x, y)
        dy = _interp2(gx, gy, tdy, x, y)
    else:
        fr = lcp.radial_factor(r2)
        dx, dy = _tangential(lcp, x, y, r2)
        fold = r2 > lcp.r_max**2
        fr = np.where(fold, np.nan, fr)
    x2 = x * fr + dx
    y2 = y * fr + dy
    return x2 * lcp.fx + lcp.c0u, y2 * lcp.fy + lcp.c0v


def undistort(ud, vd, lcp: LensCalibration, iterations: int = 20, tol: float = 1e-12):
    """Distorted pixel coordinates to undistorted (ideal pinhole)."""
    ud = np.asarray(ud, dtype=float).ravel()
    vd = np.asarray(vd, dtype=float).ravel()
    if lcp.model == "analytic" and not lcp.has_distortion:
        return ud.copy(), vd.copy()
    xd = (ud - lcp.c0u) / lcp.fx
    yd = (vd - lcp.c0v) / lcp.fy
    if lcp.model == "table":
        x, y = _undistort_table(xd, yd, lcp)
    else:
        x, y = _undistort_newton(xd, yd, lcp, iterations, tol)
    return x * lcp.fx + lcp.c0u, y * lcp.fy + lcp.c0v


def _undistort_table(x, y, lcp):
    # Same steps as undistortCaltech.m: invert the radial table, then remove a
    # first-order tangential correction.  MATLAB's ``if r~=0`` skips the whole
    # calculation if any point sits exactly on the principal point; here only
    # that point is passed through unchanged.
    r_tab, fr_tab = lcp.rad_table
    gx, gy, tdx, tdy = lcp.tang_table
    r = np.sqrt(x * x + y * y)
    r2 = _interp1(fr_tab * r_tab, r_tab, r)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = r2 / r
        x2 = x * s
        y2 = y * s
        x3 = x2 - _interp2(gx, gy, tdx, x2, y2) * s
        y3 = y2 - _interp2(gx, gy, tdy, x2, y2) * s
    centre = r == 0
    x3[centre] = x[centre]
    y3[centre] = y[centre]
    return x3, y3


def _undistort_newton(xd, yd, lcp, iterations, tol):
    d1, d2, d3, t1, t2 = lcp.d1, lcp.d2, lcp.d3, lcp.t1, lcp.t2
    x = xd.copy()
    y = yd.copy()
    for _ in range(iterations):
        s = x * x + y * y
        fr = 1 + d1 * s + d2 * s**2 + d3 * s**3
        dfr = d1 + 2 * d2 * s + 3 * d3 * s**2
        tx, ty = _tangential(lcp, x, y, s)
        fx_ = x * fr + tx - xd
        fy_ = y * fr + ty - yd
        a = fr + 2 * x * x * dfr + 2 * t1 * y + 6 * t2 * x
        b = 2 * x * y * dfr + 2 * t1 * x + 2 * t2 * y
        c = 2 * x * y * dfr + 2 * t1 * x + 2 * t2 * y
        d = fr + 2 * y * y * dfr + 6 * t1 * y + 2 * t2 * x
        det = a * d - b * c
        with np.errstate(invalid="ignore", divide="ignore"):
            step_x = (d * fx_ - b * fy_) / det
            step_y = (a * fy_ - c * fx_) / det
        x = x - step_x
        y = y - step_y
        if np.nanmax(np.abs(np.concatenate([step_x, step_y])), initial=0) < tol:
            break
    # Reject points that converged past the fold-back radius or did not converge.
    s = x * x + y * y
    tx, ty = _tangential(lcp, x, y, s)
    fr = lcp.radial_factor(s)
    err = np.hypot(x * fr + tx - xd, y * fr + ty - yd)
    bad = (s > lcp.r_max**2) | ~(err < 1e-9)
    x[bad] = np.nan
    y[bad] = np.nan
    return x, y
