"""Beach-width figures for a site (ports of the GUI's plotting tools).

* :func:`trend_plot` - CSPGmakeTrendPlot: shorelines over a time window drawn
  on the current image, and the alongshore-averaged beach width with its
  linear trend.
* :func:`change_plot` - CSPGmakeShorelineChangePlot: change between the
  current shoreline and one N shorelines earlier, along the beach.
* :func:`forecast_plot` - CSPGplotShorelineForecast: a forecast shoreline
  and its uncertainty band on the current image.
* :func:`beach_width_animation` - CSPGmakeBeachWidthAnimation: one frame per
  image plus a CSV of the beach-width time series.
* :func:`participation_plot` - CSPplotParticipationStatistics.

Beach widths are measured from each transect's landward end to the
shoreline, along the DB's "Transect averaging region", and corrected to
mean sea level (z = 0) with the site's characteristic beach slope, as the
MATLAB plots do.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from .camera.geometry import make_uv_projector
from .shoreline.io import ShorelineRecord, load_shoreline
from .shoreline.transects import Transects, beach_width, shift_shoreline, water_level_shift
from .site import ImageEntry, Site
from .timeutils import epoch_to_datetime

FIG_WIDTH_CM = 20
#: Above this many shorelines the trend plot only draws first, min, max and latest.
MAX_TREND_LINES = 8
#: Uncertainty (m) CSPGplotShorelineForecast puts around a forecast.
FORECAST_ERROR = 6.31
#: MATLAB's ``fit(..., 'smoothingspline', 'SmoothingParam', 0.1)``.
SMOOTHING_PARAM = 0.1

_LOGO = Path(__file__).resolve().parents[1] / "GUI" / "CoastSnap Logo Portrait.png"
_INCH = 2.54


# ---- the image the GUI has open ------------------------------------------------


@dataclass
class CurrentImage:
    """An oblique image with its rectification, the GUI's "current image"."""

    entry: ImageEntry
    image: np.ndarray
    metadata: dict      # the plan .mat's metadata (geom, rectz, ...)

    @classmethod
    def load(cls, site: Site, image) -> "CurrentImage":
        from PIL import Image
        from scipy.io import loadmat

        entry = image if isinstance(image, ImageEntry) else ImageEntry(int(Path(image).name.split(".")[0]), Path(image))
        _, mat = site.plan_paths(entry.path)
        if not mat.is_file():
            raise FileNotFoundError(f"{entry.name} has not been rectified (no {mat})")
        md = loadmat(mat, simplify_cells=True)["metadata"]
        return cls(entry, np.asarray(Image.open(entry.path).convert("RGB")), md)

    @property
    def kind(self) -> str:
        """Processed or Registered: the folder the image came from."""
        parts = self.entry.path.parts
        return "Registered" if "Registered" in parts else "Processed"

    @property
    def rectz(self) -> float:
        return float(self.metadata["rectz"])

    def project(self, xyz) -> np.ndarray:
        return make_uv_projector(self.metadata["geom"])(np.asarray(xyz, dtype=float))


@dataclass
class Survey:
    """A shoreline with the image it was mapped on."""

    entry: ImageEntry
    shoreline: ShorelineRecord
    local_time: datetime    # in the site's default time zone


def surveys(site: Site, entries) -> list[Survey]:
    """Images in ``entries`` that have a saved shoreline, oldest first."""
    out = []
    for e in entries:
        p = site.find_shoreline(e.name)
        if p is not None:
            out.append(Survey(e, load_shoreline(p), epoch_to_datetime(e.epoch, site.gmt_offset)))
    return out


def window(site: Site, current: CurrentImage, days: float) -> list[ImageEntry]:
    """Images from ``days`` before the current one up to it, inclusive."""
    lo = current.entry.epoch - days * 86400
    return [e for e in site.images(current.kind) if lo <= e.epoch <= current.entry.epoch]


def msl_widths(records, transects: Transects, index, slope, extend=1.0) -> np.ndarray:
    """(n surveys, m transects) beach widths corrected to mean sea level."""
    sub = transects.subset(index)
    return np.array([beach_width(r.xyz, sub, reference_level=0.0, beach_slope=slope, extend=extend)
                     for r in records]).reshape(len(records), sub.count)


def _mean(a, axis=None):
    """nanmean without the all-NaN warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _colors(n):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(max(n, 1))]


def _layout(image, bottom_cm=3.0):
    """Figure with the oblique image on top and a time-series strip below.

    Mirrors the geomplot layout: image full width, plot on the left half of
    the strip, title text and logo on the right.
    """
    import matplotlib.pyplot as plt

    h_img = FIG_WIDTH_CM * image.shape[0] / image.shape[1]
    gap, bot = 0.5, 1.2
    total = 0.2 + h_img + gap + bottom_cm + bot
    fig = plt.figure(figsize=(FIG_WIDTH_CM / _INCH, total / _INCH), facecolor="w")
    ax_img = fig.add_axes([0.01, (total - 0.2 - h_img) / total, 0.98, h_img / total])
    ax_img.imshow(image, extent=(0.5, image.shape[1] + 0.5, image.shape[0] + 0.5, 0.5))
    ax_img.set_axis_off()
    y0 = bot / total
    ax_ts = fig.add_axes([1.5 / FIG_WIDTH_CM, y0, 0.5 - 1.5 / FIG_WIDTH_CM, bottom_cm / total])
    ax_txt = fig.add_axes([0.55, y0, 0.3, bottom_cm / total])
    ax_txt.set_axis_off()
    if _LOGO.is_file():
        from PIL import Image

        logo = np.asarray(Image.open(_LOGO))
        h = 0.7 * bottom_cm
        w = h * logo.shape[1] / logo.shape[0]
        ax_logo = fig.add_axes([(FIG_WIDTH_CM - 0.2 - w) / FIG_WIDTH_CM, y0 + (bottom_cm - h) / total / 2,
                                w / FIG_WIDTH_CM, h / total])
        ax_logo.imshow(logo)
        ax_logo.set_axis_off()
    return fig, ax_img, ax_ts, ax_txt


def _headline(ax_txt, title, value_text, color):
    ax_txt.text(0.05, 0.7, title, fontsize=16, color="b", transform=ax_txt.transAxes)
    return ax_txt.text(0.05, 0.4, value_text, fontsize=16, color=color, transform=ax_txt.transAxes)


def _round5(v, up):
    return 5 * (np.ceil(v / 5) if up else np.floor(v / 5))


# ---- trend -------------------------------------------------------------------


@dataclass
class TrendResult:
    figure: object
    surveys: list
    widths: np.ndarray          # (n, m) MSL-corrected widths
    mean_width: np.ndarray      # (n,) alongshore average
    rate_m_per_year: float
    imin: int
    imax: int

    def summary(self) -> str:
        d = [s.local_time.strftime("%d/%m/%Y") for s in self.surveys]
        return (f"Minimum average beach width over time period is {self.mean_width[self.imin]:.1f}m ({d[self.imin]})\n"
                f"Maximum average beach width over time period is {self.mean_width[self.imax]:.1f}m ({d[self.imax]})\n"
                f"Beach width trend {self.rate_m_per_year:+.2f} metres/year")


def trend_plot(site: Site, current: CurrentImage, days: float = 42, max_lines: int = MAX_TREND_LINES) -> TrendResult:
    """Beach-width trend over the ``days`` before the current image.

    Each shoreline is drawn on the current image after moving it to the
    current water level, so shorelines from different tides line up.
    """
    import matplotlib.dates as mdates

    svs = surveys(site, window(site, current, days))
    if len(svs) < 2:
        raise ValueError(f"need at least two shorelines in the {days:g} days before {current.entry.name}")
    idx = site.averaging_transects
    slope = site.beach_slope
    widths = msl_widths([s.shoreline for s in svs], site.transects, idx, slope)
    mean = _mean(widths, axis=1)
    imin, imax = int(np.nanargmin(mean)), int(np.nanargmax(mean))
    labels = [s.local_time.strftime("%d/%m/%Y") for s in svs]

    fig, ax_img, ax_ts, ax_txt = _layout(current.image)
    few = len(svs) < max_lines
    shown = list(range(len(svs))) if few else [0, imin, imax, len(svs) - 1]
    colors = _colors(len(shown))
    for c, i in zip(colors, shown):
        sl = svs[i].shoreline
        moved = shift_shoreline(sl.xyz, site.transects, water_level_shift(sl.xyz[0, 2], current.rectz, slope),
                                z=current.rectz)
        if len(moved):
            uv = current.project(moved)
            ax_img.plot(uv[:, 0], uv[:, 1], lw=1, color=c)
    if few:
        ax_img.legend(labels, loc="upper right", fontsize=8)
    else:
        ax_img.legend([f"Initial width = {mean[0]:.1f} m ({labels[0]})",
                       f"Min. width = {mean[imin]:.1f} m ({labels[imin]})",
                       f"Max. width = {mean[imax]:.1f} m ({labels[imax]})",
                       f"Latest width = {mean[-1]:.1f} m ({labels[-1]})"], loc="upper right", fontsize=8)

    days_x = np.array([mdates.date2num(s.local_time.date()) for s in svs])
    if few:
        for c, x, y in zip(colors, days_x, mean):
            ax_ts.plot(x, y, ".", ms=12, color=c)
    else:
        ax_ts.plot(days_x, mean, ".", color="0.7", ms=10)
        for c, i in zip(colors, shown):
            ax_ts.plot(days_x[i], mean[i], ".", color=c, ms=10)
    ok = ~np.isnan(mean)
    coeffs = np.polyfit(days_x[ok], mean[ok], 1)
    xs = np.array([days_x.min() - 5, days_x.max() + 5])
    ax_ts.plot(xs, np.polyval(coeffs, xs), "k", lw=2)
    ax_ts.set_xlim(days_x.min() - 7, days_x.max() + 7)
    ax_ts.set_ylim(_round5(np.nanmin(mean) - 5, False), _round5(np.nanmax(mean) + 5, True))
    span = days_x.max() - days_x.min() + 14
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m" if span < 180 else "%m/%y"))
    ax_ts.set_ylabel("Beach width (m)")
    ax_ts.set_xlabel("Date")
    ax_ts.grid(True)
    rate = coeffs[0] * 365.25
    _headline(ax_txt, "Beach width trend", f"{rate:+.2f} metres/year", "g" if rate > 0 else "r")
    return TrendResult(fig, svs, widths, mean, rate, imin, imax)


def survey_figure(site: Site, survey: Survey, text: str, color):
    """One oblique image with its own shoreline (the min/max export figures)."""
    import matplotlib.pyplot as plt

    cur = CurrentImage.load(site, survey.entry)
    img = cur.image
    h = FIG_WIDTH_CM * img.shape[0] / img.shape[1]
    fig = plt.figure(figsize=(FIG_WIDTH_CM / _INCH, (h + 0.4) / _INCH), facecolor="w")
    ax = fig.add_axes([0.01, 0.2 / (h + 0.4), 0.98, h / (h + 0.4)])
    ax.imshow(img, extent=(0.5, img.shape[1] + 0.5, img.shape[0] + 0.5, 0.5))
    uv = cur.project(survey.shoreline.xyz)
    ax.plot(uv[:, 0], uv[:, 1], lw=2, color=color, label=text)
    ax.legend(loc="upper left", fontsize=12)
    ax.set_axis_off()
    return fig


def export_trend(site: Site, result: TrendResult, out_dir, dpi: int = 300, extremes: bool = True) -> list[Path]:
    """Save the trend figure and, optionally, the min and max width images."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"beachwidth_trend_figure_{site.name}.jpg"]
    result.figure.savefig(paths[0], dpi=dpi)
    if extremes:
        colors = _colors(4)
        for i, word, c in ((result.imin, "Minimum", colors[1]), (result.imax, "Maximum", colors[2])):
            s = result.surveys[i]
            fig = survey_figure(site, s, f"{word} beach width = {result.mean_width[i]:.1f} m "
                                         f"({s.local_time:%d/%m/%Y})", c)
            p = out_dir / f"{word[:3].lower()}_width_figure_{site.name}.jpg"
            fig.savefig(p, dpi=dpi)
            plt.close(fig)
            paths.append(p)
    return paths


# ---- change ------------------------------------------------------------------


@dataclass
class ChangeResult:
    figure: object
    previous: Survey
    present: Survey
    alongshore: np.ndarray
    change: np.ndarray          # present minus previous width per transect
    mean_change: float


def change_plot(site: Site, current: CurrentImage, steps_back: int = 1) -> ChangeResult:
    """Beach-width change since the shoreline ``steps_back`` surveys earlier."""
    svs = surveys(site, site.images(current.kind))
    try:
        now = next(i for i, s in enumerate(svs) if s.entry.epoch == current.entry.epoch)
    except StopIteration:
        raise ValueError(f"{current.entry.name} has no saved shoreline") from None
    prev = max(0, now - steps_back)
    idx = site.averaging_transects
    sub = site.transects.subset(idx)
    slope = site.beach_slope
    w_prev = msl_widths([svs[prev].shoreline], site.transects, idx, slope)[0]
    w_now = msl_widths([svs[now].shoreline], site.transects, idx, slope)[0]
    # As in MATLAB, retry the present shoreline on transects 1.5x as long.
    retry = msl_widths([svs[now].shoreline], site.transects, idx, slope, extend=1.5)[0]
    w_now = np.where(np.isnan(w_now), retry, w_now)
    change = w_now - w_prev

    fig, ax_img, ax_ts, ax_txt = _layout(current.image)
    colors = _colors(2)
    for c, s in zip(colors, (svs[prev], svs[now])):
        uv = current.project(s.shoreline.xyz)
        ax_img.plot(uv[:, 0], uv[:, 1], lw=1.5, color=c, label=s.local_time.strftime("%d/%m/%Y"))
    ax_img.legend(loc="upper right", fontsize=10)

    along = sub.alongshore_distances if sub.alongshore_distances is not None else np.arange(sub.count, dtype=float)
    ax_ts.fill_between(along, np.nan_to_num(change), step=None, alpha=0.8)
    ax_ts.set_xlim(np.min(along), np.max(along))
    ax_ts.set_ylim(_round5(np.nanmin(change) - 5, False), _round5(np.nanmax(change) + 5, True))
    ax_ts.set_ylabel("Beach change (m)")
    ax_ts.set_xlabel("Alongshore distance (m)")
    ax_ts.grid(True)
    mean = float(_mean(change))
    color = "r" if mean < 0 else "g"
    ax_ts.axhline(mean, ls="--", lw=2, color=color)
    _headline(ax_txt, "Beach width change", f"{mean:+.0f} metres (average)", color)
    return ChangeResult(fig, svs[prev], svs[now], along, change, mean)


# ---- forecast ----------------------------------------------------------------


def forecast_plot(site: Site, current: CurrentImage, width_change: float, error: float = FORECAST_ERROR):
    """Forecast shoreline after a beach-width change of ``width_change`` metres.

    Negative is erosion (a narrower beach). The grey band is ``± error``.
    MATLAB's prompt takes the change with the opposite sign convention on
    east-facing beaches; here the sign always means wider/narrower.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    p = site.find_shoreline(current.entry.name)
    if p is None:
        raise ValueError(f"{current.entry.name} has no saved shoreline")
    sl = load_shoreline(p)
    sub = site.transects.subset(site.averaging_transects)
    new = shift_shoreline(sl.xyz, sub, width_change)
    sea = shift_shoreline(sl.xyz, sub, width_change + error)
    land = shift_shoreline(sl.xyz, sub, width_change - error)

    img = current.image
    h = FIG_WIDTH_CM * img.shape[0] / img.shape[1]
    fig = plt.figure(figsize=(FIG_WIDTH_CM / _INCH, (h + 0.7) / _INCH), facecolor="w")
    ax = fig.add_axes([0.01, 0.5 / (h + 0.7), 0.98, h / (h + 0.7)])
    ax.imshow(img, extent=(0.5, img.shape[1] + 0.5, img.shape[0] + 0.5, 0.5))
    ax.set_axis_off()
    colors = _colors(2)
    uv = current.project(sl.xyz)
    ax.plot(uv[:, 0], uv[:, 1], lw=1.5, color=colors[1],
            label=f"Current shoreline ({epoch_to_datetime(current.entry.epoch, site.gmt_offset):%d/%m/%Y})")
    if len(new):
        band = current.project(np.vstack([sea, land[::-1]]))
        ax.fill(band[:, 0], band[:, 1], color="0.5", lw=0)
        uvn = current.project(new)
        ax.plot(uvn[:, 0], uvn[:, 1], "k", lw=2, label="Forecast post-storm shoreline")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color="0.5"))
    labels.append("Forecast uncertainty")
    ax.legend(handles, labels, loc="upper right", fontsize=10)
    return fig


# ---- animation ---------------------------------------------------------------


def smooth_series(days, values, p: float = SMOOTHING_PARAM):
    """Smoothing spline like MATLAB ``fit(..., 'smoothingspline', 'SmoothingParam', p)``.

    MATLAB minimises ``p Σ(y - s)² + (1 - p) ∫ s''²``; scipy's
    ``make_smoothing_spline`` takes ``lam = (1 - p) / p``. Repeated x values
    are averaged first.
    """
    from scipy.interpolate import make_smoothing_spline

    days = np.asarray(days, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = ~np.isnan(values)
    ux, inv = np.unique(days[ok], return_inverse=True)
    uy = np.bincount(inv, values[ok]) / np.bincount(inv)
    if len(ux) < 5:
        coeffs = np.polyfit(ux, uy, min(1, len(ux) - 1))
        return lambda x: np.polyval(coeffs, x)
    return make_smoothing_spline(ux, uy, lam=(1 - p) / p)


def beach_width_animation(site: Site, current: CurrentImage, out_dir, days: float = 42, dpi: int = 150) -> list[Path]:
    """Frames ``frame_001.jpg ...`` stepping through every image in the window,
    with the smoothed beach width below, and ``beachwidth_timeseries_<site>.csv``.

    The CSV has MATLAB's 8 columns: year, month, day, hour, minute, second
    (local time), alongshore-average width and smoothed width.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = window(site, current, days)
    svs = surveys(site, frames)
    if len(svs) < 2:
        raise ValueError("need at least two shorelines to make an animation")
    widths = msl_widths([s.shoreline for s in svs], site.transects, site.averaging_transects, site.beach_slope)
    mean = _mean(widths, axis=1)
    when = [s.local_time for s in svs]
    x = mdates.date2num(when)
    spline = smooth_series(x, mean)
    smooth = spline(x)

    csv_path = out_dir / f"beachwidth_timeseries_{site.name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        for t, a, b in zip(when, mean, smooth):
            w.writerow([t.year, t.month, t.day, t.hour, t.minute, t.second, f"{a:.4f}", f"{b:.4f}"])

    have_sl = {s.entry.epoch: s for s in svs}
    first = np.asarray(Image.open(frames[0].path).convert("RGB"))
    fig, ax_img, ax_ts, ax_txt = _layout(first)
    ax_ts.plot(x, mean, "x", color="0.7")
    xs = np.linspace(x.min(), x.max(), 200)
    ax_ts.plot(xs, spline(xs), "k", lw=1)
    ax_ts.set_xlim(x.min(), x.max())
    ax_ts.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_ts.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_ts.xaxis.get_major_locator()))
    ax_ts.grid(True)
    ax_ts.set_ylabel("Beach width (m)")
    ylim = ax_ts.get_ylim()
    ax_ts.set_ylim(ylim)
    _headline(ax_txt, "Beach width", "", "r")
    paths = []
    for n, e in enumerate(frames, 1):
        for artist in list(ax_img.images) + list(ax_img.lines):
            artist.remove()
        img = first if n == 1 else np.asarray(Image.open(e.path).convert("RGB"))
        ax_img.imshow(img, extent=(0.5, img.shape[1] + 0.5, img.shape[0] + 0.5, 0.5))
        s = have_sl.get(e.epoch)
        if s is not None and np.isfinite(s.shoreline.uv).any():
            ax_img.plot(s.shoreline.uv[:, 0], s.shoreline.uv[:, 1], "r", lw=2)
        t = mdates.date2num(epoch_to_datetime(e.epoch, site.gmt_offset))
        for artist in list(ax_ts.lines[2:]) + list(ax_txt.texts[1:]):
            artist.remove()
        ax_ts.plot([t, t], ylim, "r", lw=2)
        ax_txt.text(0.05, 0.4, f"{float(spline(t)):.1f} metres", fontsize=16, color="r", transform=ax_txt.transAxes)
        p = out_dir / f"frame_{n:03d}.jpg"
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return [csv_path] + paths


# ---- participation -----------------------------------------------------------

SUBMISSION_TYPES = ("Email", "Facebook", "Twitter", "Instagram", "App")


@dataclass
class Participation:
    days: list              # each date from start to end
    daily: np.ndarray       # submissions per day
    by_type: dict           # submission type -> count
    local_times: list       # every submission time in the site's default zone

    @property
    def cumulative(self) -> np.ndarray:
        return np.cumsum(self.daily)

    def summary(self) -> str:
        from collections import Counter

        n = int(self.cumulative[-1]) if len(self.days) else 0
        weeks = len(self.days) / 7
        day = Counter(t.strftime("%A") for t in self.local_times).most_common(1)
        hour = Counter(t.hour for t in self.local_times).most_common(1)
        return "\n".join([
            f"Most popular day of week is {day[0][0] if day else '-'}",
            f"Most popular time of day is {hour[0][0] if hour else '-'}",
            f"Total number of submissions is {n}",
            f"Average number of submissions per week is {n / weeks if weeks else 0:.1f}",
        ])


def participation(records, site_db: dict, start: date, end: date, app_times=()) -> Participation:
    """Count submissions per day (``CSPgetParticipationStatistics``).

    ``records`` are :class:`coastsnap.db.ImageRecord` rows for the site;
    ``app_times`` are local times of app submissions that are not in the DB
    (from a Spotteron export, see :func:`read_spotteron_times`).
    """
    offset = site_db["timezone"]["gmt_offset"]
    local = [r.time + timedelta(hours=offset - r.gmt_offset(site_db)) for r in records]
    local += list(app_times)
    ndays = (end - start).days + 1
    days = [start + timedelta(days=i) for i in range(ndays)]
    daily = np.zeros(ndays, dtype=int)
    for t in local:
        i = (t.date() - start).days
        if 0 <= i < ndays:
            daily[i] += 1
    by_type = {k: sum(r.source == k for r in records) for k in SUBMISSION_TYPES[:-1]}
    by_type["App"] = len(app_times)
    in_range = [t for t in local if start <= t.date() <= end]
    return Participation(days, daily, by_type, in_range)


def read_spotteron_times(path, root_id, root_col: int = 2, time_col: int = 17) -> list[datetime]:
    """App submission times for one site from a Spotteron export (as .xlsx).

    The MATLAB scripts disagree on the time column (12 and 17, 1-based) and
    note the root ID may be in column 1, so both are parameters.
    """
    from openpyxl import load_workbook

    from .db import parse_db_time

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(wb.worksheets[0].iter_rows(min_row=2, values_only=True))
    finally:
        wb.close()
    out = []
    for r in rows:
        if len(r) >= max(root_col, time_col) and r[root_col - 1] is not None and str(r[root_col - 1]).strip() == str(root_id):
            out.append(parse_db_time(r[time_col - 1]))
    return out


def participation_plot(stats: Participation, start: date, end: date):
    """Cumulative submissions and a pie chart of submission types."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(30 / _INCH, 9 / _INCH), facecolor="w")
    ax = fig.add_axes([1.4 / 30, 1.2 / 9, (30 - 1.4 - 8) / 30, (9 - 1.6) / 9])
    ax.plot(stats.days, stats.cumulative, "b", lw=1)
    ax.set_xlim(start, end)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative images")
    ax.grid(True, axis="y")
    ax.text(0.1, 0.8, f"Cumulative number of images:\n{start:%d/%m/%Y} - {end:%d/%m/%Y}",
            transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    pie = fig.add_axes([(30 - 8) / 30, 0.25, 8 / 30, 0.7])
    labels = [k for k, v in stats.by_type.items() if v]
    counts = [stats.by_type[k] for k in labels]
    if counts:
        pie.pie(counts, autopct="%1.0f%%", textprops={"fontsize": 10})
        pie.legend(labels, title="Submission type", loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=10)
    pie.set_aspect("equal")
    return fig
