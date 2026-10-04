"""Lens calibration profiles (port of makeLCPP3, makeRadDist, makeTangDist).

The distortion model is the Caltech / Brown model used by the CIRN toolbox,
which is the same as OpenCV's ``k1, k2, p1, p2, k3`` model:

    x_d = x * (1 + d1 r^2 + d2 r^4 + d3 r^6) + 2 t1 x y + t2 (r^2 + 2 x^2)
    y_d = y * (1 + d1 r^2 + d2 r^4 + d3 r^6) + t1 (r^2 + 2 y^2) + 2 t2 x y

where ``x, y`` are normalised image coordinates (tan of the view angle).

Two evaluation modes are supported:

``"table"``
    Reproduces MATLAB exactly: the radial factor and tangential terms are
    tabulated (``r`` in 0..2, tangential grid |x| <= 1.5, |y| <= 1.3) and
    linearly interpolated.  Anything outside the tables becomes NaN, so very
    wide fields of view (HFOV above roughly 110 degrees) lose their image
    edges.
``"analytic"`` (default)
    Evaluates the polynomial directly and inverts it with Newton iterations.
    It is valid out to the radius where the radial polynomial folds back on
    itself, so it handles wide-angle and ultra-wide phone lenses.  Inside the
    MATLAB table domain the two modes agree to a small fraction of a pixel.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property

import numpy as np

MODELS = ("analytic", "table")


@dataclass(frozen=True)
class LensCalibration:
    """Camera intrinsics and distortion, equivalent to the MATLAB ``lcp`` struct."""

    nu: int
    nv: int
    c0u: float
    c0v: float
    fx: float
    fy: float
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    t1: float = 0.0
    t2: float = 0.0
    model: str = "analytic"

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")

    def replace(self, **changes) -> "LensCalibration":
        return dataclasses.replace(self, **changes)

    @property
    def has_distortion(self) -> bool:
        return any((self.d1, self.d2, self.d3, self.t1, self.t2))

    @property
    def hfov_deg(self) -> float:
        """Horizontal field of view implied by ``fx`` (Harley et al. 2019, Eq. 4)."""
        return float(np.degrees(2 * np.arctan(self.nu / (2 * self.fx))))

    # ---- MATLAB lookup tables (makeRadDist / makeTangDist) -------------------

    @cached_property
    def rad_table(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lcp.r, lcp.fr)`` exactly as makeRadDist builds them."""
        r = np.linspace(0.0, 2.0, 201)
        r2 = r * r
        fr = 1 + self.d1 * r2 + self.d2 * r2 * r2 + self.d3 * r2 * r2 * r2
        good = np.diff(r * fr) > 0
        # MATLAB indexes the 201-long r with a 200-long mask, so the last
        # sample is always dropped.
        return r[:-1][good], fr[:-1][good]

    @cached_property
    def tang_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(lcp.x, lcp.y, lcp.dx, lcp.dy)`` exactly as makeTangDist builds them."""
        x = np.linspace(-1.5, 1.5, 31)
        y = np.linspace(-1.3, 1.3, 27)
        X, Y = np.meshgrid(x, y)
        r2 = X * X + Y * Y
        dx = 2 * self.t1 * X * Y + self.t2 * (r2 + 2 * X * X)
        dy = self.t1 * (r2 + 2 * Y * Y) + 2 * self.t2 * X * Y
        return x, y, dx, dy

    # ---- analytic model limits ----------------------------------------------

    @cached_property
    def r_max(self) -> float:
        """Largest undistorted radius before ``r * fr(r)`` stops increasing.

        Beyond this the distortion polynomial folds back, so points there
        could be wrongly mapped back onto the image.
        """
        # d/dr [r fr(r)] = 1 + 3 d1 s + 5 d2 s^2 + 7 d3 s^3 with s = r^2
        roots = np.roots([7 * self.d3, 5 * self.d2, 3 * self.d1, 1.0])
        s = [z.real for z in roots if abs(z.imag) < 1e-12 and z.real > 0]
        return float(np.sqrt(min(s))) if s else np.inf

    def radial_factor(self, r2):
        return 1 + self.d1 * r2 + self.d2 * r2**2 + self.d3 * r2**3

    # ---- conversion ---------------------------------------------------------

    @classmethod
    def from_opencv(cls, camera_matrix, dist_coeffs, image_size, model="analytic"):
        """Build from an OpenCV calibration (e.g. a phone checkerboard calibration).

        ``dist_coeffs`` is ``(k1, k2, p1, p2[, k3])``; ``image_size`` is
        ``(width, height)``.  OpenCV's model is identical to this one with
        ``d1=k1, d2=k2, d3=k3, t1=p1, t2=p2``.
        """
        K = np.asarray(camera_matrix, dtype=float)
        d = list(np.asarray(dist_coeffs, dtype=float).ravel()) + [0.0] * 5
        k1, k2, p1, p2, k3 = d[:5]
        return cls(
            nu=int(image_size[0]), nv=int(image_size[1]),
            c0u=K[0, 2], c0v=K[1, 2], fx=K[0, 0], fy=K[1, 1],
            d1=k1, d2=k2, d3=k3, t1=p1, t2=p2, model=model,
        )

    def to_matlab(self) -> dict:
        """The ``lcp`` struct as MATLAB stores it in rectification ``.mat`` files."""
        r, fr = self.rad_table
        x, y, dx, dy = self.tang_table
        return {
            "NU": float(self.nu), "NV": float(self.nv),
            "c0U": self.c0u, "c0V": self.c0v, "fx": self.fx, "fy": self.fy,
            "d1": self.d1, "d2": self.d2, "d3": self.d3, "t1": self.t1, "t2": self.t2,
            "r": r, "fr": fr, "x": x, "y": y, "dx": dx, "dy": dy,
        }

    @classmethod
    def from_matlab(cls, lcp: dict, model="analytic") -> "LensCalibration":
        """Read an ``lcp`` struct loaded with ``scipy.io.loadmat(..., simplify_cells=True)``."""
        g = lambda k, default=0.0: float(np.asarray(lcp.get(k, default)).ravel()[0])
        return cls(
            nu=int(g("NU")), nv=int(g("NV")), c0u=g("c0U"), c0v=g("c0V"),
            fx=g("fx"), fy=g("fy"), d1=g("d1"), d2=g("d2"), d3=g("d3"),
            t1=g("t1"), t2=g("t2"), model=model,
        )


# Calibrations hard-coded in makeLCPP3.m, keyed by camera then (NU, NV).
_KNOWN = {
    "Aerielle": {
        (4000, 3000): dict(c0u=2016.23, c0v=1421.23, fx=2327.5, fy=2323.56,
                           d1=-0.1326, d2=0.09946, d3=0.0, t1=-0.0003082, t2=0.0005749),
        (4000, 2250): dict(c0u=2006.000, c0v=1156.703, fx=2309.081, fy=2306.763,
                           d1=-0.02036, d2=0.01238, d3=0.0, t1=0.00750, t2=0.00125),
        (3840, 2160): dict(c0u=1957.13, c0v=1088.21, fx=2298.59, fy=2310.87,
                           d1=-0.14185, d2=0.11168, d3=0.0, t1=0.00369, t2=0.002314),
    },
    "Mavic": {
        (4000, 3000): dict(c0u=2004, c0v=1122, fx=3008, fy=2994,
                           d1=0.00660, d2=-0.09099, d3=0.0, t1=0.0, t2=0.0),
        (3840, 2160): dict(c0u=1758, c0v=1090, fx=3320, fy=3319,
                           d1=0.01638, d2=0.02744, d3=0.00105, t1=-0.01109, t2=0.0),
    },
}


def make_lcp(camera: str, nu: int, nv: int, model: str = "analytic") -> LensCalibration:
    """Port of ``makeLCPP3(whichStr, NU, NV)``.

    ``"CoastSnap"`` is the generic smartphone profile: principal point at the
    image centre, ``fx = fy = (815/960) * NU`` and no distortion.  The focal
    length is normally replaced by the FOV sweep in :func:`solve_geometry`.
    """
    if camera == "CoastSnap":
        f = (815 / 960) * nu
        return LensCalibration(nu=nu, nv=nv, c0u=nu / 2, c0v=nv / 2, fx=f, fy=f, model=model)
    if camera not in _KNOWN:
        raise ValueError(f"No lens calibration for camera {camera!r}")
    try:
        params = _KNOWN[camera][(nu, nv)]
    except KeyError:
        raise ValueError(f"No lens calibration yet for {camera} at {nu}x{nv}") from None
    return LensCalibration(nu=nu, nv=nv, model=model, **params)
