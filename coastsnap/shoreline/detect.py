"""Automatic shoreline detection on rectified plan-view images.

Python port of ``mapShorelineCCD.m`` and ``mapShorelineHUE.m`` from the
Shoreline-Mapping-Toolbox (Mitch Harley, 2018), which the MATLAB CoastSnap
GUI calls from ``CSPGmapShoreline`` and ``CSPGbulkRectAndMap``.

Both methods work the same way:

1. Sample the image along the cross-shore transects and build a smoothed
   histogram (kernel density) of a colour index: red minus blue for CCD, the
   HSV hue channel for HUE.
2. Split the histogram with Otsu's method, take the strongest peak on each
   side (water and sand) and set the threshold one third of the way from the
   water peak to the sand peak.
3. Contour the colour index at that threshold inside the region bounded by
   the transects and keep the longest contour.
4. Along each transect, keep the most landward contour point lying within
   ``search_halfwidth`` metres of the transect line.

Plotting and interactive editing are not done here: see
:mod:`coastsnap.shoreline.review` for the human-in-the-loop step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import contourpy
import numpy as np
from matplotlib.colors import rgb_to_hsv
from matplotlib.path import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks

from .transects import Transects

#: Weights applied to the (water, sand) histogram peaks to get the threshold.
#: Skews the threshold towards the sand peak, which the original authors found
#: works well in SE Australia.
DEFAULT_THRESH_WEIGHTINGS = (1 / 3, 2 / 3)


@dataclass
class ThresholdInfo:
    """Diagnostics from the automatic threshold, for plotting and review."""

    pdf_locs: np.ndarray
    pdf_values: np.ndarray
    otsu: float
    peak_locs: tuple[float, float]
    threshold: float


@dataclass
class ShorelineResult:
    """A detected shoreline in local (site origin) coordinates.

    ``x``/``y`` hold one point per transect with NaN where nothing was found;
    use :attr:`xy` for the NaN-free points the MATLAB code returns.
    """

    x: np.ndarray
    y: np.ndarray
    method: str
    threshold: float | None
    contour: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    threshold_info: ThresholdInfo | None = None

    @property
    def found(self) -> np.ndarray:
        return ~np.isnan(self.x)

    @property
    def xy(self) -> np.ndarray:
        """(K, 2) array of detected points, NaNs removed."""
        return np.column_stack([self.x, self.y])[self.found]

    @property
    def is_empty(self) -> bool:
        return not self.found.any()


def map_shoreline_ccd(
    xgrid,
    ygrid,
    iplan,
    transects: Transects,
    threshold: float | None = None,
    thresh_weightings=DEFAULT_THRESH_WEIGHTINGS,
    search_halfwidth: float = 1.0,
    blacktest_tol: float = 2.0,
) -> ShorelineResult:
    """Map a shoreline with the Colour Channel Divergence (red minus blue) method.

    Parameters
    ----------
    xgrid, ygrid : 1-D arrays of local eastings/northings of the image columns/rows.
    iplan : (len(ygrid), len(xgrid), 3) RGB plan image (uint8 or float).
    transects : cross-shore transects seeding the search and bounding the ROI.
    threshold : skip the automatic threshold and use this value instead
        (used by the review step when the user picks a threshold by hand).
    search_halfwidth : contour points within this distance (m) of a transect
        line count as lying on it.
    blacktest_tol : a point this far (m) seaward of the shoreline must not be a
        black (no-data) pixel, otherwise the point is dropped.
    """
    xgrid, ygrid, iplan = _orient(xgrid, ygrid, iplan)
    rgb = iplan.astype(float)
    ccd = rgb[..., 0] - rgb[..., 2]

    samples = sample_transects(xgrid, ygrid, ccd, transects)
    samples = samples[samples != 0]  # drop black (no-data) pixels at the image edge
    if samples.size == 0:
        # Matches MATLAB: greyscale images have R == B everywhere.
        return ShorelineResult(np.array([]), np.array([]), "CCD", None)

    info = None
    if threshold is None:
        info = auto_threshold(samples, thresh_weightings, use_findpeaks=False)
        threshold = info.threshold

    masked = _mask_outside_roi(xgrid, ygrid, ccd, transects)
    contour = longest_contour(xgrid, ygrid, masked, threshold)
    blackcheck = RegularGridInterpolator(
        (ygrid, xgrid), masked, bounds_error=False, fill_value=np.nan
    )
    x, y = points_on_transects(
        contour, transects, search_halfwidth, blackcheck=blackcheck, blacktest_tol=blacktest_tol
    )
    return ShorelineResult(x, y, "CCD", float(threshold), contour, info)


def map_shoreline_hue(
    xgrid,
    ygrid,
    iplan,
    transects: Transects,
    threshold: float | None = None,
    thresh_weightings=DEFAULT_THRESH_WEIGHTINGS,
    search_halfwidth: float = 1.0,
) -> ShorelineResult:
    """Map a shoreline using the HSV hue channel (values 0 to 1).

    Same parameters as :func:`map_shoreline_ccd`. Unlike CCD there is no
    black-pixel filtering, as in the MATLAB original.
    """
    xgrid, ygrid, iplan = _orient(xgrid, ygrid, iplan)
    rgb = iplan.astype(float)
    if np.issubdtype(iplan.dtype, np.integer):
        rgb /= np.iinfo(iplan.dtype).max
    hue = rgb_to_hsv(np.clip(rgb, 0, 1))[..., 0]

    info = None
    if threshold is None:
        samples = sample_transects(xgrid, ygrid, hue, transects)
        info = auto_threshold(samples, thresh_weightings, use_findpeaks=True)
        threshold = info.threshold

    masked = _mask_outside_roi(xgrid, ygrid, hue, transects)
    contour = longest_contour(xgrid, ygrid, masked, threshold)
    x, y = points_on_transects(contour, transects, search_halfwidth)
    return ShorelineResult(x, y, "HUE", float(threshold), contour, info)


METHODS = {"CCD": map_shoreline_ccd, "HUE": map_shoreline_hue}


def map_shoreline(xgrid, ygrid, iplan, transects, method="CCD", **kwargs) -> ShorelineResult:
    """Dispatch to :func:`map_shoreline_ccd` or :func:`map_shoreline_hue`."""
    try:
        func = METHODS[method.upper()]
    except KeyError:
        raise ValueError(f"Unknown shoreline method {method!r}; expected one of {list(METHODS)}")
    return func(xgrid, ygrid, iplan, transects, **kwargs)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def sample_transects(xgrid, ygrid, values, transects: Transects) -> np.ndarray:
    """Nearest-pixel samples of ``values`` along every transect (like ``improfile``).

    Each transect is sampled at roughly one point per pixel. Samples falling
    outside the image are dropped.
    """
    dx = abs(xgrid[1] - xgrid[0]) if len(xgrid) > 1 else 1.0
    dy = abs(ygrid[1] - ygrid[0]) if len(ygrid) > 1 else 1.0
    step = min(dx, dy)
    out = []
    for (x0, x1), (y0, y1) in zip(transects.x.T, transects.y.T):
        n = max(int(np.ceil(np.hypot(x1 - x0, y1 - y0) / step)) + 1, 2)
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        cols = np.rint((xs - xgrid[0]) / (xgrid[1] - xgrid[0])).astype(int)
        rows = np.rint((ys - ygrid[0]) / (ygrid[1] - ygrid[0])).astype(int)
        ok = (cols >= 0) & (cols < len(xgrid)) & (rows >= 0) & (rows < len(ygrid))
        out.append(values[rows[ok], cols[ok]])
    return np.concatenate(out) if out else np.array([])


def ksdensity(samples, npoints: int = 100):
    """Gaussian kernel density estimate like MATLAB's default ``ksdensity``.

    Uses the normal-approximation bandwidth with a robust (MAD) spread and
    evaluates on ``npoints`` points spanning the data range padded by three
    bandwidths.
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = x.size
    sig = np.median(np.abs(x - np.median(x))) / 0.6745
    if sig <= 0:
        sig = np.std(x) or 1.0
    bw = sig * (4 / (3 * n)) ** 0.2
    locs = np.linspace(x.min() - 3 * bw, x.max() + 3 * bw, npoints)
    # Bin the samples first so this stays fast for large images.
    counts, edges = np.histogram(x, bins=min(n, 2048))
    centres = 0.5 * (edges[:-1] + edges[1:])
    z = (locs[:, None] - centres[None, :]) / bw
    values = (np.exp(-0.5 * z**2) * counts).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
    return values, locs


def otsu_threshold(samples, nbins: int = 256) -> float:
    """Single-level Otsu threshold over ``nbins`` bins (like ``multithresh``)."""
    x = np.asarray(samples, dtype=float).ravel()
    lo, hi = x.min(), x.max()
    if hi == lo:
        return float(lo)
    counts, edges = np.histogram(x, bins=nbins, range=(lo, hi))
    p = counts / counts.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    sigma_b[~np.isfinite(sigma_b)] = -1
    k = int(np.argmax(sigma_b))
    return float(edges[k + 1])


def auto_threshold(samples, thresh_weightings=DEFAULT_THRESH_WEIGHTINGS, use_findpeaks=False):
    """Weighted average of the strongest histogram peak either side of Otsu.

    CCD (``use_findpeaks=False``) takes the PDF maximum on each side of the
    Otsu threshold; HUE (``use_findpeaks=True``) only considers local peaks,
    as in the MATLAB versions.
    """
    values, locs = ksdensity(samples)
    otsu = otsu_threshold(samples)
    if use_findpeaks:
        idx, _ = find_peaks(values)
        cand_locs, cand_vals = locs[idx], values[idx]
    else:
        cand_locs, cand_vals = locs, values
    below = cand_locs < otsu
    above = cand_locs > otsu
    if not below.any() or not above.any():
        raise ValueError("Could not find histogram peaks on both sides of the Otsu threshold")
    water = cand_locs[below][np.argmax(cand_vals[below])]
    sand = cand_locs[above][np.argmax(cand_vals[above])]
    thresh = thresh_weightings[0] * water + thresh_weightings[1] * sand
    return ThresholdInfo(locs, values, otsu, (float(water), float(sand)), float(thresh))


def roi_polygon(transects: Transects) -> np.ndarray:
    """Region of interest: transect starts followed by transect ends in reverse."""
    xs = np.concatenate([transects.x[0], transects.x[1][::-1]])
    ys = np.concatenate([transects.y[0], transects.y[1][::-1]])
    return np.column_stack([xs, ys])


def longest_contour(xgrid, ygrid, values, level) -> np.ndarray:
    """Vertices (K, 2) of the longest contour of ``values`` at ``level``.

    "Longest" means most vertices, as in the MATLAB code. NaNs are treated as
    gaps. Returns an empty (0, 2) array when there is no contour.
    """
    gen = contourpy.contour_generator(x=xgrid, y=ygrid, z=values)
    lines = gen.lines(level)
    if not lines:
        return np.empty((0, 2))
    return max(lines, key=len)


def points_on_transects(
    contour,
    transects: Transects,
    search_halfwidth=1.0,
    blackcheck=None,
    blacktest_tol=2.0,
):
    """Pick, for each transect, the most landward contour point lying on it.

    Points are rotated into each transect's frame (along-transect distance,
    offset). Those with ``|offset| < search_halfwidth`` and between the
    transect start and end are candidates; the one closest to the start wins.
    When ``blackcheck`` (an interpolator on the masked colour index) is given,
    the point is dropped if the pixel ``blacktest_tol`` metres seaward is
    exactly zero, i.e. black no-data.
    """
    m = transects.count
    x = np.full(m, np.nan)
    y = np.full(m, np.nan)
    if len(contour) == 0:
        return x, y
    for i in range(m):
        x0, y0 = transects.x[0, i], transects.y[0, i]
        dx = transects.x[1, i] - x0
        dy = transects.y[1, i] - y0
        length = np.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        rel = contour - (x0, y0)
        along = rel[:, 0] * ux + rel[:, 1] * uy
        offset = -rel[:, 0] * uy + rel[:, 1] * ux
        cand = np.flatnonzero((np.abs(offset) < search_halfwidth) & (along > 0) & (along < length))
        if cand.size == 0:
            continue
        j = cand[np.argmin(along[cand])]
        if blackcheck is not None:
            # Point blacktest_tol seaward along the transect, at the same offset.
            a, o = along[j] + blacktest_tol, offset[j]
            px = x0 + a * ux - o * uy
            py = y0 + a * uy + o * ux
            if blackcheck([[py, px]])[0] == 0:  # NaN (outside ROI) passes, as in MATLAB
                continue
        x[i], y[i] = contour[j]
    return x, y


def _mask_outside_roi(xgrid, ygrid, values, transects):
    X, Y = np.meshgrid(xgrid, ygrid)
    inside = Path(roi_polygon(transects)).contains_points(np.column_stack([X.ravel(), Y.ravel()]))
    masked = values.astype(float, copy=True)
    masked[~inside.reshape(masked.shape)] = np.nan
    return masked


def _orient(xgrid, ygrid, iplan):
    """Return ascending grids and the image flipped to match."""
    xgrid = np.asarray(xgrid, dtype=float).ravel()
    ygrid = np.asarray(ygrid, dtype=float).ravel()
    iplan = np.asarray(iplan)
    if iplan.ndim != 3 or iplan.shape[:2] != (len(ygrid), len(xgrid)):
        raise ValueError(
            f"iplan shape {iplan.shape} does not match (len(ygrid), len(xgrid), 3) = "
            f"({len(ygrid)}, {len(xgrid)}, 3)"
        )
    if len(xgrid) > 1 and xgrid[1] < xgrid[0]:
        xgrid, iplan = xgrid[::-1], iplan[:, ::-1]
    if len(ygrid) > 1 and ygrid[1] < ygrid[0]:
        ygrid, iplan = ygrid[::-1], iplan[::-1]
    return xgrid, ygrid, iplan
