"""Camera geometry from ground control points (the solve in CSPGrectifyImage.m).

MATLAB fits azimuth, tilt and roll with ``nlinfit`` for every trial focal
length between the site's FOV limits (5 pixel steps) and keeps the one with
the lowest mean squared error.  :func:`solve_geometry` does the same with
``scipy.optimize.least_squares`` (Levenberg-Marquardt), and can optionally
also fit the focal length continuously and radial distortion terms, which
matters for wide-angle phone lenses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

from .geometry import expand_beta, find_uv
from .lens import LensCalibration

# Intrinsic parameters that solve_geometry can fit.  "f" ties fx = fy;
# "c0u"/"c0v" free the principal point, which CoastSnap assumes is the image centre.
FREE_INTRINSICS = ("f", "c0u", "c0v", "d1", "d2", "d3", "t1", "t2")


@dataclass
class PoseFit:
    """Result of one nonlinear fit (the outputs of nlinfit + nlparci)."""

    beta: np.ndarray            # 6-DOF [x y z azimuth tilt roll]
    ci: np.ndarray              # 2x6 95% confidence bounds (zero width for knowns)
    mse: float                  # sum of squared residuals / (2N - p)
    residuals: np.ndarray       # (N, 2) picked minus modelled UV
    lcp: LensCalibration        # intrinsics used (fitted ones included)
    intrinsics_ci: dict = field(default_factory=dict)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.mse))


@dataclass
class Geometry(PoseFit):
    """Final geometry plus the FOV sweep that chose the focal length."""

    known_flags: np.ndarray = None
    knowns: np.ndarray = None
    sweep_fx: np.ndarray = None
    sweep_mse: np.ndarray = None

    @property
    def fov_deg(self) -> float:
        return self.lcp.hfov_deg


def _apply_intrinsics(lcp: LensCalibration, names, values) -> LensCalibration:
    changes = {}
    for n, v in zip(names, values):
        if n == "f":
            changes["fx"] = changes["fy"] = float(v)
        else:
            changes[n] = float(v)
    return lcp.replace(**changes)


def _initial_intrinsics(lcp, names):
    return [lcp.fx if n == "f" else getattr(lcp, n) for n in names]


def fit_pose(xyz, uv, lcp: LensCalibration, beta0, known_flags=(1, 1, 1, 0, 0, 0),
             free_intrinsics=(), tol=1e-12, max_nfev=None) -> PoseFit:
    """One nonlinear fit of the unknown extrinsics (and optional intrinsics).

    ``beta0`` is the full 6-vector; entries flagged known are held fixed.
    ``uv`` are the picked (distorted) pixel coordinates, ``(N, 2)``.
    """
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    flags = np.asarray(known_flags, dtype=bool)
    beta0 = np.asarray(beta0, dtype=float).ravel()
    knowns = beta0[flags]
    free_intrinsics = tuple(free_intrinsics)
    for n in free_intrinsics:
        if n not in FREE_INTRINSICS:
            raise ValueError(f"cannot fit {n!r}; choose from {FREE_INTRINSICS}")
    if free_intrinsics and lcp.model == "table" and set(free_intrinsics) - {"f"}:
        raise ValueError('fitting distortion needs lcp.model == "analytic"')
    nb = int((~flags).sum())
    y = np.concatenate([uv[:, 0], uv[:, 1]])     # nlinfit's [U; V] ordering

    def model(p):
        cam = _apply_intrinsics(lcp, free_intrinsics, p[nb:]) if free_intrinsics else lcp
        UV = find_uv(expand_beta(p[:nb], flags, knowns), xyz, cam)
        return np.concatenate([UV[:, 0], UV[:, 1]]), cam

    penalty = 10.0 * max(lcp.nu, lcp.nv)

    def resid(p):
        r = model(p)[0] - y
        # Steps that push a GCP behind the camera or past the distortion limit
        # get a large residual so Levenberg-Marquardt rejects them.
        return np.where(np.isfinite(r), r, penalty)

    p0 = np.concatenate([beta0[~flags], _initial_intrinsics(lcp, free_intrinsics)])
    if not np.all(np.isfinite(model(p0)[0])):
        raise ValueError(
            "A GCP projects outside the lens model (behind the camera or past the "
            "distortion limit) at the initial guess. Check the initial azimuth/tilt/roll."
        )
    if y.size < p0.size:
        raise ValueError(f"need at least {int(np.ceil(p0.size / 2))} GCPs to fit {p0.size} parameters")
    sol = least_squares(resid, p0, method="lm", xtol=tol, ftol=tol, gtol=tol,
                        x_scale="jac", max_nfev=max_nfev)
    p = sol.x
    f, cam = model(p)
    if not np.all(np.isfinite(f)):
        raise ValueError("fit converged with GCPs outside the lens model")
    r = y - f
    dof = y.size - p.size
    mse = float(r @ r / dof) if dof > 0 else np.nan

    # nlparci: beta +/- t(0.975, dof) * sqrt(diag(inv(J'J) * mse))
    J = sol.jac
    with np.errstate(invalid="ignore"):
        cov = np.linalg.pinv(J.T @ J) * mse
        half = stats.t.ppf(0.975, dof) * np.sqrt(np.diag(cov)) if dof > 0 else np.full(p.size, np.nan)
    ci = np.zeros((2, 6))
    ci[:, flags] = 0.0
    ci[0, ~flags] = p[:nb] - half[:nb]
    ci[1, ~flags] = p[:nb] + half[:nb]
    intr_ci = {n: (p[nb + i] - half[nb + i], p[nb + i] + half[nb + i])
               for i, n in enumerate(free_intrinsics)}
    n = uv.shape[0]
    return PoseFit(
        beta=expand_beta(p[:nb], flags, knowns), ci=ci, mse=mse,
        residuals=np.column_stack([r[:n], r[n:]]), lcp=cam, intrinsics_ci=intr_ci,
    )


def focal_lengths_for_fov(nu: int, fov_limits, step: float = 5.0) -> np.ndarray:
    """Trial focal lengths for a horizontal FOV range, as in CSPGrectifyImage."""
    hfov_min, hfov_max = fov_limits
    fx_max = 0.5 * nu / np.tan(hfov_min * np.pi / 360)   # Harley et al. (2019) Eq. 4
    fx_min = 0.5 * nu / np.tan(hfov_max * np.pi / 360)
    snap = lambda f: np.floor(f / step + 0.5) * step     # interp1(..., 'nearest')
    lo, hi = snap(fx_min), snap(fx_max)
    return lo + step * np.arange(int(np.floor((hi - lo) / step + 1e-9)) + 1)


def solve_geometry(xyz, uv, lcp: LensCalibration, beta0, known_flags=(1, 1, 1, 0, 0, 0),
                   fov_limits=None, fov_step: float = 5.0, refine_focal: bool = False,
                   free_distortion=(), tol: float = 1e-12) -> Geometry:
    """Solve the camera geometry from GCPs.

    With ``fov_limits=(min_deg, max_deg)`` this reproduces MATLAB: a sweep of
    focal lengths in ``fov_step`` pixel steps, each fitted from ``beta0``, and a
    final fit at the focal length with the lowest MSE.  Without it the focal
    length in ``lcp`` is used as is.

    ``refine_focal`` additionally fits ``fx = fy`` continuously after the sweep.
    ``free_distortion`` names further intrinsics to fit (any of
    ``FREE_INTRINSICS`` except ``"f"``), for example ``("d1", "d2")`` for barrel
    distortion on a wide-angle phone lens, plus ``("c0u", "c0v")`` if the
    principal point is off-centre.  Start ``lcp`` from a calibration if you
    have one.  Each extra term needs more GCPs and good coverage of the image,
    especially towards its edges, to be meaningful.
    """
    flags = np.asarray(known_flags, dtype=bool)
    beta0 = np.asarray(beta0, dtype=float).ravel()
    sweep_fx = sweep_mse = None
    if fov_limits is not None:
        sweep_fx = focal_lengths_for_fov(lcp.nu, fov_limits, fov_step)
        sweep_mse = np.full(sweep_fx.size, np.nan)
        for i, f in enumerate(sweep_fx):
            try:
                sweep_mse[i] = fit_pose(xyz, uv, lcp.replace(fx=f, fy=f), beta0, flags, tol=tol).mse
            except ValueError:
                pass
        if np.all(np.isnan(sweep_mse)):
            raise ValueError("no trial focal length gave a valid fit")
        best = sweep_fx[int(np.nanargmin(sweep_mse))]
        lcp = lcp.replace(fx=best, fy=best)

    free = (("f",) if refine_focal else ()) + tuple(free_distortion)
    fit = fit_pose(xyz, uv, lcp, beta0, flags, free_intrinsics=free, tol=tol)
    return Geometry(
        beta=fit.beta, ci=fit.ci, mse=fit.mse, residuals=fit.residuals, lcp=fit.lcp,
        intrinsics_ci=fit.intrinsics_ci, known_flags=flags.astype(int),
        knowns=beta0[flags], sweep_fx=sweep_fx, sweep_mse=sweep_mse,
    )
