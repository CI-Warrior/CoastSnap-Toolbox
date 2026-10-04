"""Bulk rectify and map (port of CSPGbulkRectAndMap).

Reuses the camera geometry of one rectified image for a run of later images
from the same (registered) viewpoint, rectifies each at its own tide level,
maps its shoreline and saves both. Shorelines are saved unreviewed (QA = 0)
for the reviewer to check later.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .camera.geometry import make_uv_projector
from .rectify import RectificationResult, RectificationSettings, load_geometry, rectify_image, save_rectified
from .shoreline import load_transects, make_record, map_shoreline, save_shoreline
from .site import ImageEntry, Site

#: Outcomes reported per image.
DONE, EXISTS, WRONG_SIZE, NO_TIDE, NO_SHORELINE = "done", "exists", "wrong size", "no tide", "no shoreline"


@dataclass
class BulkResult:
    image: Path
    status: str
    plan: Path | None = None
    shoreline: Path | None = None
    points: int = 0


def images_between(site: Site, start: ImageEntry | int, end: ImageEntry | int,
                   kind: str = "Processed") -> list[ImageEntry]:
    """Images after ``start`` up to and including ``end`` (epochs or entries).

    MATLAB walks the GUI's navigation list, which is the folder of the start
    image; this uses the site's ``kind`` folder (Processed or Registered).
    """
    s = start.epoch if isinstance(start, ImageEntry) else int(start)
    e = end.epoch if isinstance(end, ImageEntry) else int(end)
    if e < s:
        raise ValueError("end image is before the start image")
    return [im for im in site.images(kind) if s < im.epoch <= e]


def bulk_rectify_and_map(site: Site, images: Iterable[ImageEntry], geometry_mat,
                         method: str = "CCD", overwrite: bool = False,
                         world_file: bool = True,
                         on_image: Callable[[int, int, BulkResult], None] | None = None,
                         plot: Callable | None = None) -> list[BulkResult]:
    """Rectify ``images`` with the geometry in ``geometry_mat`` and map shorelines.

    ``geometry_mat`` is a rectified ``...plan....mat`` (normally that of the
    image the run starts from). Images that are already rectified are
    skipped unless ``overwrite``; so are images whose size differs from the
    first one, which usually means a registration went wrong. ``plot``, if
    given, is called as ``plot(I, plan, record)`` for each image.
    """
    from PIL import Image

    images = list(images)
    settings = RectificationSettings.from_site_db(site.db)
    beta, lcp, base_meta = load_geometry(geometry_mat)
    project_uv = make_uv_projector(base_meta["geom"])
    transects = load_transects(site.transect_file)
    utm_zone = site.db["UTMzone"]
    first_size = None
    results = []
    for n, im in enumerate(images, 1):
        I = np.asarray(Image.open(im.path).convert("RGB"))
        size = I.shape[:2]
        first_size = first_size or size
        jpg, mat = site.plan_paths(im.path)
        if jpg.exists() and not overwrite:
            res = BulkResult(im.path, EXISTS, jpg)
        elif size != first_size:
            res = BulkResult(im.path, WRONG_SIZE)
        else:
            res = _rectify_and_map(site, settings, im, I, beta, lcp, base_meta, project_uv,
                                   transects, utm_zone, method, world_file, jpg, mat, plot)
        results.append(res)
        if on_image:
            on_image(n, len(images), res)
    return results


def _rectify_and_map(site, settings, im, I, beta, lcp, base_meta, project_uv, transects,
                     utm_zone, method, world_file, jpg, mat, plot) -> BulkResult:
    tide = site.tide_level(im.epoch)
    if np.isnan(tide):
        return BulkResult(im.path, NO_TIDE)
    z = tide + settings.tidal_offset
    plan = rectify_image(I, beta, lcp, settings.rect_xy, z)
    metadata = copy.deepcopy(base_meta)
    metadata.update(whenDone=float(round(time.time())), rectz=float(z))
    metadata.setdefault("gcps", {})
    metadata["gcps"]["UVpicked"] = np.nan    # marks a bulk rectification
    save_rectified(RectificationResult(plan, None, metadata), str(jpg), str(mat),
                   world=(settings.res, site.origin) if world_file else None)
    found = map_shoreline(plan.x, plan.y, plan.timex, transects, method=method)
    if found.is_empty:
        return BulkResult(im.path, NO_SHORELINE, jpg)
    record = make_record(found.xy, z, site.origin, utm_zone, found.method, found.threshold,
                         project_uv, qa=False)
    sl_path = save_shoreline(site.shoreline_path(im.name), record)
    if plot is not None:
        plot(I, plan, record)
    return BulkResult(im.path, DONE, jpg, sl_path, len(found.xy))
