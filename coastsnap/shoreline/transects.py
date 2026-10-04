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


def shift_shoreline(shoreline_xyz, transects: Transects, shift, z=None, extend: float = 1.0) -> np.ndarray:
    """Move a shoreline along the transects (port of ``CSPGshiftSLxshore``).

    Returns one point per transect the shoreline crosses, moved ``shift``
    metres seaward (towards each transect's end; negative is landward).
    ``shift`` may be a scalar or one value per transect. The points take
    elevation ``z`` (default: the shoreline's own).

    MATLAB moves points towards +x along a line fitted to the transect, so
    the sign of the shift depends on which way the beach faces and vertical
    transects fail; this works along the transect direction instead.
    """
    xyz = np.asarray(shoreline_xyz, dtype=float)
    chain = transect_chainage(xyz[:, :2], transects, extend=extend)
    start = np.vstack([transects.x[0], transects.y[0]]).T
    vec = np.vstack([transects.x[1] - transects.x[0], transects.y[1] - transects.y[0]]).T
    unit = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    pts = start + (chain + np.broadcast_to(np.asarray(shift, dtype=float), chain.shape))[:, None] * unit
    z = xyz[0, 2] if z is None else z
    out = np.column_stack([pts, np.full(len(pts), float(z))])
    return out[~np.isnan(chain)]


def water_level_shift(shoreline_z, water_level, beach_slope):
    """Seaward shift (m) that moves a shoreline at ``shoreline_z`` to ``water_level``.

    A higher water level moves the shoreline landward, so this is
    ``-(water_level - shoreline_z) / beach_slope``.
    """
    return -tidal_correction(shoreline_z, water_level, beach_slope)


def make_transects(roi, coastline, spacing: float = 5.0, half_length: float = 500.0,
                   step: float = 0.1, flip: bool = False) -> Transects:
    """Build shore-normal transects (the Make Transect File tool, CSPGmakeTransectFiles).

    Parameters
    ----------
    roi : (P, 2) polygon spanning sand and water, where shorelines may be found.
    coastline : (Q, 2) polyline roughly along the shore, drawn from the end
        nearest the camera.
    spacing : distance between transects along ``coastline`` (m).
    flip : swap start and end. Transects run towards +x (towards +y when the
        coastline runs east-west), so flip when that points landward: the
        start must be the landward end.

    Transects are clipped to the part of each normal line inside ``roi``;
    normals that miss it are dropped. ``alongshore_distances`` are the
    distances of the kept transects along the coastline.
    """
    from matplotlib.path import Path as MplPath

    pts = np.asarray(coastline, dtype=float)[:, :2]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    locs = _colon(0.0, spacing, dist[-1])
    mx = np.interp(locs, dist, pts[:, 0])
    my = np.interp(locs, dist, pts[:, 1])
    d = np.arange(-half_length, half_length + step / 2, step)
    roi_path = MplPath(np.asarray(roi, dtype=float)[:, :2])
    starts, ends, along = [], [], []
    for i in range(len(locs) - 1):
        tx, ty = mx[i + 1] - mx[i], my[i + 1] - my[i]
        n = np.array([ty, -tx]) / np.hypot(tx, ty)
        if n[0] < 0 or (n[0] == 0 and n[1] < 0):
            n = -n
        line = np.column_stack([mx[i] + d * n[0], my[i] + d * n[1]])
        inside = np.flatnonzero(roi_path.contains_points(line))
        if inside.size:
            starts.append(line[inside[0]])
            ends.append(line[inside[-1]])
            along.append(locs[i])
    if not starts:
        raise ValueError("no transect crosses the region of interest")
    a, b = np.array(starts), np.array(ends)
    if flip:
        a, b = b, a
    return Transects(np.vstack([a[:, 0], b[:, 0]]), np.vstack([a[:, 1], b[:, 1]]), np.array(along))


def _colon(lo, step, hi):
    """MATLAB ``lo:step:hi``."""
    n = int(np.floor((hi - lo) / step + 1e-10)) + 1
    return lo + step * np.arange(max(n, 0))


def save_transects(path, transects: Transects):
    """Save as the ``SLtransects`` struct the MATLAB toolbox reads."""
    from scipy.io import savemat

    sl = {"x": transects.x, "y": transects.y}
    if transects.alongshore_distances is not None:
        sl["alongshore_distances"] = transects.alongshore_distances[None, :]
    savemat(path, {"SLtransects": sl})
    return path
