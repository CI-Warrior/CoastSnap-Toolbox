"""Human-in-the-loop review of detected shorelines.

Replaces the MATLAB GUI steps ``CSPGmapShoreline`` (detect and show),
``CSPGeditShorelinePoints`` (drag points with ``impoly``),
``CSPGcropShorelinePoints`` (lasso points to delete), ``CSPGsaveShoreline``
(confirm and save) and the QA flag.

Two layers:

* :class:`ShorelineEditor` holds the editable state and every operation
  (move, insert, delete, crop, re-threshold, switch method, undo, accept or
  reject). It has no plotting, so batch tools and tests can drive it.
* :class:`ShorelineReviewer` is a matplotlib window on top of it. Run
  :func:`review_shoreline` for one image or :func:`review_queue` to step
  through every unreviewed shoreline from a bulk run.

Reviewer controls (also shown in the window):

=================  ==========================================================
drag a point       move it
right-click        delete the nearest point
``i``              insert a point at the cursor
``c``              crop: draw a polygon, points inside are removed
click histogram    re-detect with that threshold
``m``              switch between CCD and HUE and re-detect
``t``              re-detect with the automatic threshold
``u``              undo
``a`` / ``enter``  accept (marks the shoreline as reviewed and saves it)
``r``              reject (nothing is saved; an existing file is left alone)
``n`` / ``esc``    skip for now
=================  ==========================================================

Use the toolbar's zoom and pan freely; edits are ignored while either is on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from matplotlib.path import Path as MplPath

from .detect import ShorelineResult, map_shoreline
from .io import (
    PlanImage,
    ShorelineRecord,
    UVProjector,
    load_shoreline,
    make_record,
    save_shoreline,
)
from .transects import Transects

ACCEPTED = "accepted"
REJECTED = "rejected"
SKIPPED = "skipped"


class ShorelineEditor:
    """Editable shoreline for one plan image.

    Parameters
    ----------
    plan : the rectified plan image the shoreline is mapped on.
    transects : site transects.
    method : ``"CCD"`` or ``"HUE"``.
    points : start from these (K, 2) points (e.g. a saved shoreline) instead
        of running detection. Detection still runs once so the threshold
        histogram can be shown.
    """

    def __init__(self, plan: PlanImage, transects: Transects, method="CCD", points=None):
        self.plan = plan
        self.transects = transects
        self.method = method.upper()
        self.result: ShorelineResult = self._detect(None)
        self.points = self.result.xy if points is None else np.asarray(points, float).reshape(-1, 2)
        self.decision: str | None = None
        self._history: list[tuple] = []

    # -- detection ---------------------------------------------------------

    @property
    def threshold(self):
        return self.result.threshold

    def rethreshold(self, threshold: float | None):
        """Re-run detection at ``threshold`` (None: automatic) and replace the points."""
        self._push()
        self.result = self._detect(threshold)
        self.points = self.result.xy

    def set_method(self, method: str):
        """Switch detection method and re-detect with its automatic threshold."""
        self._push()
        self.method = method.upper()
        self.result = self._detect(None)
        self.points = self.result.xy

    def _detect(self, threshold):
        return map_shoreline(
            self.plan.xgrid, self.plan.ygrid, self.plan.iplan, self.transects,
            method=self.method, threshold=threshold,
        )

    # -- point edits -------------------------------------------------------

    def nearest(self, xy, tol=np.inf) -> int | None:
        """Index of the point nearest ``xy`` within ``tol`` metres, else None."""
        if len(self.points) == 0:
            return None
        d = np.hypot(*(self.points - np.asarray(xy, float)).T)
        i = int(np.argmin(d))
        return i if d[i] <= tol else None

    def move(self, i: int, xy):
        self._push()
        self.points = self.points.copy()
        self.points[i] = xy

    def delete(self, i: int):
        self._push()
        self.points = np.delete(self.points, i, axis=0)

    def insert(self, xy) -> int:
        """Insert a point where it best fits the line; returns its index."""
        self._push()
        xy = np.asarray(xy, float)
        pts = self.points
        if len(pts) < 2:
            i = len(pts)
        else:
            # Insert into the segment it is closest to, or at an end.
            a, b = pts[:-1], pts[1:]
            ab = b - a
            t = np.clip(((xy - a) * ab).sum(1) / np.maximum((ab**2).sum(1), 1e-12), 0, 1)
            seg_d = np.hypot(*(a + t[:, None] * ab - xy).T)
            k = int(np.argmin(seg_d))
            end_d = np.hypot(*(pts[[0, -1]] - xy).T)
            if end_d[0] < seg_d[k] and t[0] == 0:
                i = 0
            elif end_d[1] < seg_d[k] and t[-1] == 1:
                i = len(pts)
            else:
                i = k + 1
        self.points = np.insert(pts, i, xy, axis=0)
        return i

    def crop(self, polygon) -> int:
        """Remove points inside ``polygon`` ((P, 2) vertices); returns how many."""
        if len(self.points) == 0:
            return 0
        inside = MplPath(np.asarray(polygon, float)).contains_points(self.points)
        if inside.any():
            self._push()
            self.points = self.points[~inside]
        return int(inside.sum())

    def undo(self) -> bool:
        if not self._history:
            return False
        self.points, self.result, self.method = self._history.pop()
        return True

    def _push(self):
        self._history.append((self.points.copy(), self.result, self.method))

    # -- decision ----------------------------------------------------------

    def accept(self):
        self.decision = ACCEPTED

    def reject(self):
        self.decision = REJECTED

    def skip(self):
        self.decision = SKIPPED

    def to_record(
        self,
        origin,
        utm_zone: str,
        project_uv: UVProjector | None = None,
    ) -> ShorelineRecord:
        """Record for the current points; ``qa`` is True only once accepted."""
        return make_record(
            self.points, self.plan.rectz, origin, utm_zone, self.method,
            self.threshold, project_uv, qa=self.decision == ACCEPTED,
        )


@dataclass
class ReviewOutcome:
    decision: str
    record: ShorelineRecord | None
    path: Path | None = None


class ShorelineReviewer:
    """Matplotlib window for reviewing one shoreline. See the module docstring."""

    PICK_TOL_PX = 10

    def __init__(
        self,
        editor: ShorelineEditor,
        oblique_image=None,
        project_uv: UVProjector | None = None,
        title: str = "",
    ):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import PolygonSelector

        self._plt = plt
        self._PolygonSelector = PolygonSelector
        self.editor = editor
        self.project_uv = project_uv
        self.oblique_image = oblique_image if project_uv is not None else None
        self._drag = None
        self._selector = None
        self._message = ""

        self.fig = plt.figure(figsize=(14, 8) if self.oblique_image is not None else (9, 8))
        if self.oblique_image is not None:
            gs = self.fig.add_gridspec(2, 2, height_ratios=[4, 1])
            self.ax_obl = self.fig.add_subplot(gs[0, 0])
            self.ax_plan = self.fig.add_subplot(gs[0, 1])
            self.ax_hist = self.fig.add_subplot(gs[1, :])
            self.ax_obl.imshow(self.oblique_image)
            self.ax_obl.set_axis_off()
            (self.obl_line,) = self.ax_obl.plot([], [], "y-", lw=2)
        else:
            gs = self.fig.add_gridspec(2, 1, height_ratios=[4, 1])
            self.ax_plan = self.fig.add_subplot(gs[0])
            self.ax_hist = self.fig.add_subplot(gs[1])
            self.ax_obl = self.obl_line = None

        plan = editor.plan
        self.ax_plan.imshow(
            plan.iplan, origin="lower" if plan.ygrid[-1] > plan.ygrid[0] else "upper",
            extent=_extent(plan.xgrid, plan.ygrid),
        )
        t = editor.transects
        self.ax_plan.plot(t.x, t.y, color="w", lw=0.5, alpha=0.4)
        (self.contour_line,) = self.ax_plan.plot([], [], color="c", lw=0.8, alpha=0.6)
        (self.plan_line,) = self.ax_plan.plot([], [], "y.-", lw=2, ms=6, picker=True)
        self.ax_plan.set_xlabel("Eastings [m]")
        self.ax_plan.set_ylabel("Northings [m]")
        self.ax_plan.set_aspect("equal")
        self.fig.suptitle(title, fontsize=10)
        self.help = self.fig.text(
            0.01, 0.005,
            "drag: move  right-click: delete  i: insert  c: crop  click histogram: threshold  "
            "m: CCD/HUE  t: auto  u: undo  a: accept  r: reject  n: skip",
            fontsize=8, family="monospace",
        )

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("button_release_event", self._on_release)
        c.mpl_connect("key_press_event", self._on_key)
        self.redraw()

    # -- drawing -----------------------------------------------------------

    def redraw(self):
        ed = self.editor
        pts = ed.points
        self.plan_line.set_data(pts[:, 0], pts[:, 1])
        cont = ed.result.contour
        self.contour_line.set_data(cont[:, 0], cont[:, 1]) if len(cont) else self.contour_line.set_data([], [])
        if self.obl_line is not None:
            uv = self._uv(pts)
            self.obl_line.set_data(uv[:, 0], uv[:, 1])
        self._draw_hist()
        thr = "none" if ed.threshold is None else f"{ed.threshold:.3g}"
        self.ax_plan.set_title(
            f"{ed.method}  threshold {thr}  {len(pts)} points"
            + (f"  ·  {self._message}" if self._message else ""),
            fontsize=9,
        )
        self.fig.canvas.draw_idle()

    def _draw_hist(self):
        ax = self.ax_hist
        ax.clear()
        info = self.editor.result.threshold_info
        if info is not None:
            ax.plot(info.pdf_locs, info.pdf_values, "k-", lw=1)
            ax.plot(info.peak_locs, np.interp(info.peak_locs, info.pdf_locs, info.pdf_values), "ro")
        if self.editor.threshold is not None:
            ax.axvline(self.editor.threshold, color="r", ls=":", lw=2)
        ax.set_xlabel(("Red minus blue" if self.editor.method == "CCD" else "Hue") + "  (click to set threshold)", fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=8)

    def _uv(self, pts):
        if len(pts) == 0:
            return np.empty((0, 2))
        xyz = np.column_stack([pts, np.full(len(pts), self.editor.plan.rectz)])
        return np.asarray(self.project_uv(xyz), float).reshape(-1, 2)

    # -- events ------------------------------------------------------------

    def _toolbar_busy(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        return bool(tb is not None and getattr(tb, "mode", ""))

    def _pick_tol_m(self):
        # Convert the pixel pick tolerance to metres on the plan axes.
        x0, x1 = self.ax_plan.get_xlim()
        width_px = self.ax_plan.get_window_extent().width or 1
        return abs(x1 - x0) / width_px * self.PICK_TOL_PX

    def _on_press(self, ev):
        if self._toolbar_busy() or self._selector is not None or ev.xdata is None:
            return
        if ev.inaxes is self.ax_hist and ev.button == 1:
            self.editor.rethreshold(ev.xdata)
            self._message = "re-detected"
            self.redraw()
        elif ev.inaxes is self.ax_plan:
            i = self.editor.nearest((ev.xdata, ev.ydata), self._pick_tol_m())
            if i is None:
                return
            if ev.button == 1:
                self._drag = i
            elif ev.button == 3:
                self.editor.delete(i)
                self._message = "point deleted"
                self.redraw()

    def _on_motion(self, ev):
        if self._drag is None or ev.inaxes is not self.ax_plan or ev.xdata is None:
            return
        pts = self.editor.points.copy()
        pts[self._drag] = (ev.xdata, ev.ydata)
        self.plan_line.set_data(pts[:, 0], pts[:, 1])
        self.fig.canvas.draw_idle()

    def _on_release(self, ev):
        if self._drag is None:
            return
        i, self._drag = self._drag, None
        if ev.inaxes is self.ax_plan and ev.xdata is not None:
            self.editor.move(i, (ev.xdata, ev.ydata))
            self._message = "point moved"
        self.redraw()

    def _on_key(self, ev):
        ed = self.editor
        k = ev.key
        if self._selector is not None:
            if k == "escape":
                self._end_crop()
                self._message = "crop cancelled"
                self.redraw()
            return
        if k == "i" and ev.inaxes is self.ax_plan and ev.xdata is not None:
            ed.insert((ev.xdata, ev.ydata))
            self._message = "point inserted"
        elif k == "c":
            self._start_crop()
            return
        elif k == "m":
            ed.set_method("HUE" if ed.method == "CCD" else "CCD")
            self._message = f"switched to {ed.method}"
        elif k == "t":
            ed.rethreshold(None)
            self._message = "automatic threshold"
        elif k == "u":
            self._message = "undone" if ed.undo() else "nothing to undo"
        elif k in ("a", "enter"):
            ed.accept()
        elif k == "r":
            ed.reject()
        elif k in ("n", "escape"):
            ed.skip()
        else:
            return
        if ed.decision is not None:
            self._plt.close(self.fig)
        else:
            self.redraw()

    def _start_crop(self):
        self._message = "crop: click a polygon around points to remove, close it on the first vertex (esc cancels)"
        self.redraw()
        self._selector = self._PolygonSelector(
            self.ax_plan, self._on_crop, props=dict(color="r", lw=1.5)
        )

    def _on_crop(self, verts):
        n = self.editor.crop(verts)
        self._end_crop()
        self._message = f"{n} point{'s' if n != 1 else ''} cropped"
        self.redraw()

    def _end_crop(self):
        if self._selector is not None:
            self._selector.disconnect_events()
            self._selector.set_visible(False)
            self._selector = None

    def run(self) -> str:
        """Show the window until the user decides; returns the decision."""
        self._plt.show(block=True)
        if self.editor.decision is None:  # window closed without a key
            self.editor.skip()
        return self.editor.decision


def review_shoreline(
    plan: PlanImage,
    transects: Transects,
    origin,
    utm_zone: str,
    save_to=None,
    existing: ShorelineRecord | None = None,
    method: str = "CCD",
    oblique_image=None,
    project_uv: UVProjector | None = None,
    title: str = "",
) -> ReviewOutcome:
    """Detect (or load) a shoreline, let a person check and edit it, then save.

    On accept the shoreline is saved to ``save_to`` (if given) with QA set.
    Reject and skip save nothing.
    """
    points = None
    if existing is not None:
        points = existing.xyz[:, :2]
        method = existing.method or method
    editor = ShorelineEditor(plan, transects, method=method, points=points)
    decision = ShorelineReviewer(editor, oblique_image, project_uv, title).run()
    if decision != ACCEPTED:
        return ReviewOutcome(decision, None)
    record = editor.to_record(origin, utm_zone, project_uv)
    path = save_shoreline(save_to, record) if save_to is not None else None
    return ReviewOutcome(decision, record, path)


@dataclass
class QueueItem:
    """One shoreline awaiting review and the plan image it was mapped on."""

    shoreline_path: Path
    plan_path: Path
    oblique_image: np.ndarray | None = None


def unreviewed(shoreline_paths: Iterable[Path]) -> list[Path]:
    """Shoreline files whose QA flag is not set."""
    return [p for p in shoreline_paths if not load_shoreline(p).qa]


def review_queue(
    items: Iterable[QueueItem],
    transects: Transects,
    origin,
    utm_zone: str,
    project_uv_for=None,
    on_reject: str = "keep",
) -> list[tuple[QueueItem, str]]:
    """Step through shorelines one window at a time.

    ``project_uv_for(item)`` may return a UV projector for that image (from
    its rectification geometry) to show the oblique view. ``on_reject`` is
    ``"keep"`` (leave the unreviewed file in place) or ``"delete"`` (remove
    it, so it does not appear in analyses).
    """
    from .io import load_plan

    results = []
    for n, item in enumerate(items, 1):
        existing = load_shoreline(item.shoreline_path)
        project_uv = project_uv_for(item) if project_uv_for else None
        out = review_shoreline(
            load_plan(item.plan_path), transects, origin, utm_zone,
            save_to=item.shoreline_path, existing=existing,
            oblique_image=item.oblique_image, project_uv=project_uv,
            title=f"[{n}] {item.shoreline_path.name}",
        )
        if out.decision == REJECTED and on_reject == "delete":
            item.shoreline_path.unlink()
            item.shoreline_path.with_suffix(".csv").unlink(missing_ok=True)
        results.append((item, out.decision))
    return results


def _extent(xgrid, ygrid):
    dx = (xgrid[-1] - xgrid[0]) / max(len(xgrid) - 1, 1) / 2
    dy = (ygrid[-1] - ygrid[0]) / max(len(ygrid) - 1, 1) / 2
    lo_y, hi_y = sorted((ygrid[0] - dy, ygrid[-1] + dy))
    return (xgrid[0] - dx, xgrid[-1] + dx, lo_y, hi_y)
