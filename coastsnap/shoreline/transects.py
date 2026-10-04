"""Cross-shore transects and beach-width analysis.

Covers the transect handling in ``CSPGmakeTrendPlot`` and
``CSPGmakeShorelineChangePlot``: intersecting shorelines with transects
(``polyxpoly`` in MATLAB), tidal correction with a characteristic beach slope,
and the beach-width trend in metres per week.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat


@dataclass
class Transects:
    """Cross-shore transects, landward start first.

    ``x`` and ``y`` are (2, M) arrays: row 0 is each transect's start
    (landward), row 1 its end (seaward). This is the ``SLtransects`` struct
    stored in the site's transect ``.mat`` file.
    """

    x: np.ndarray
    y: np.ndarray
    alongshore_distances: np.ndarray | None = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).reshape(2, -1)
        self.y = np.asarray(self.y, dtype=float).reshape(2, -1)
        if self.x.shape != self.y.shape:
            raise ValueError("transect x and y must have the same shape")
        if self.alongshore_distances is not None:
            self.alongshore_distances = np.asarray(self.alongshore_distances, dtype=float).ravel()

    @property
    def count(self) -> int:
        return self.x.shape[1]

    def subset(self, index) -> "Transects":
        index = np.atleast_1d(index)
        ad = None if self.alongshore_distances is None else self.alongshore_distances[index]
        return Transects(self.x[:, index], self.y[:, index], ad)


def load_transects(path) -> Transects:
    """Load ``SLtransects`` from a site transect ``.mat`` file."""
    data = loadmat(path, simplify_cells=True)
    if "SLtransects" not in data:
        raise KeyError(f"{path} has no SLtransects variable")
    t = data["SLtransects"]
    return Transects(t["x"], t["y"], t.get("alongshore_distances"))


def transect_chainage(shoreline_xy, transects: Transects, extend: float = 1.0) -> np.ndarray:
    """Distance from each transect's start to where the shoreline crosses it.

    Parameters
    ----------
    shoreline_xy : (K, 2) shoreline polyline in local coordinates.
    extend : scale each transect by this factor before intersecting. The
        MATLAB change plot retries with 1.5 when the shoreline misses a
        transect; pass ``extend=1.5`` for the same behaviour.

    Returns an (M,) array with NaN where the shoreline does not cross a
    transect. If it crosses more than once the most landward crossing is used.
    """
    pts = np.asarray(shoreline_xy, dtype=float)[:, :2]
    pts = pts[~np.isnan(pts).any(axis=1)]
    out = np.full(transects.count, np.nan)
    if len(pts) < 2:
        return out
    a = pts[:-1]
    b = pts[1:]
    for i in range(transects.count):
        p = np.array([transects.x[0, i], transects.y[0, i]])
        q = np.array([transects.x[1, i], transects.y[1, i]])
        q = p + extend * (q - p)
        t = _segment_hits(p, q, a, b)
        if t.size:
            out[i] = t.min() * np.linalg.norm(q - p)
    return out


def tidal_correction(shoreline_z, reference_level, beach_slope):
    """Horizontal shift (m) moving a shoreline at ``shoreline_z`` to ``reference_level``.

    Subtract it from a beach width, as the MATLAB plots do:
    ``width - (reference_level - shoreline_z) / beach_slope``.
    """
    return (reference_level - shoreline_z) / beach_slope


def beach_width(shoreline_xyz, transects: Transects, reference_level=None, beach_slope=None, extend=1.0):
    """Tidally corrected beach width along each transect.

    ``shoreline_xyz`` is the saved ``sl.xyz`` (K, 3) array; its z column is the
    shoreline elevation. Without ``reference_level``/``beach_slope`` the raw
    chainage is returned.
    """
    xyz = np.asarray(shoreline_xyz, dtype=float)
    width = transect_chainage(xyz[:, :2], transects, extend=extend)
    if reference_level is not None and beach_slope is not None:
        width = width - tidal_correction(xyz[0, 2], reference_level, beach_slope)
    return width


def beach_width_trend(dates, mean_widths):
    """Linear trend of alongshore-averaged beach width in metres per week.

    ``dates`` are datetimes (or days as floats). Returns ``(m_per_week, coeffs)``
    where ``coeffs`` is the ``np.polyfit`` result against days.
    """
    days = np.array([_to_days(d) for d in dates], dtype=float)
    widths = np.asarray(mean_widths, dtype=float)
    ok = ~np.isnan(widths)
    coeffs = np.polyfit(days[ok], widths[ok], 1)
    return coeffs[0] * 7, coeffs


def _to_days(d):
    if hasattr(d, "timestamp"):
        return d.timestamp() / 86400
    if isinstance(d, np.datetime64):
        return d.astype("datetime64[s]").astype(float) / 86400
    return float(d)


def _segment_hits(p, q, a, b):
    """Parameters t in [0, 1] along p->q where it crosses segments a[k]->b[k]."""
    r = q - p
    s = b - a
    denom = r[0] * s[:, 1] - r[1] * s[:, 0]
    ap = a - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ap[:, 0] * s[:, 1] - ap[:, 1] * s[:, 0]) / denom
        u = (ap[:, 0] * r[1] - ap[:, 1] * r[0]) / denom
    ok = (denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return t[ok]
