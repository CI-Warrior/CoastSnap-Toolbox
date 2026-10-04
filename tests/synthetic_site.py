"""A small synthetic CoastSnap database for workflow tests.

One site, "beach", seen by a camera 25 m up looking east over a flat beach.
Weekly snaps show the shoreline at x = 180 + 6k m (k = 0, 1, ...), so the
beach widens by exactly 6 m a week. The tide is constant at 0 m and the
tidal offset 0.3 m, so every image is rectified at z = 0.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.io import savemat

from coastsnap.camera import find_uv, find_xyz_6dof, make_lcp
from coastsnap.config import Paths
from coastsnap.naming import argus_filename
from coastsnap.shoreline import Transects, save_transects
from coastsnap.site import ImageEntry, Site
from coastsnap.timeutils import datetime_to_datenum, datetime_to_epoch

SITE = "beach"
E0, N0, CAM_Z = 342000.0, 6266000.0, 25.0
GCPS = [("Rock", 150, -120, 1.2), ("Pole", 220, -40, 2.5), ("Bin", 300, 60, 0.8),
        ("Sign", 180, 130, 4.0), ("Groyne", 400, -200, 0.5), ("Lifeguard", 500, 10, 1.5)]
TZ, TZ_OFFSET, ALT_TZ, ALT_OFFSET = "AEST", 10, "AEDT", 11
TIDAL_OFFSET, SLOPE = 0.3, 0.1
NU, NV = 800, 600
BETA = np.r_[0, 0, CAM_Z, np.radians([97, 78, 0.7])]
SAND, WATER, SKY = (200, 180, 120), (50, 90, 140), (180, 200, 230)
TRANSECT_START, TRANSECT_END = 140.0, 420.0
WEEKLY_CHANGE = 6.0
TRANSECT_YS = np.arange(-60, 61, 10.0)   # inside the camera's view
FIRST = datetime(2024, 1, 1, 9, 0)


def truth_lcp():
    f = 0.5 * NU / np.tan(np.radians(31))
    return make_lcp("CoastSnap", NU, NV).replace(fx=f, fy=f)


def shoreline_x(k: int) -> float:
    return 180.0 + WEEKLY_CHANGE * k


def msl_width(k: int) -> float:
    """Beach width from the transect start, corrected from z = 0.3 to MSL."""
    return shoreline_x(k) - TRANSECT_START + TIDAL_OFFSET / SLOPE


def render(k: int, seed: int = 0, size=(NV, NU)) -> np.ndarray:
    nv, nu = size
    lcp = truth_lcp() if size == (NV, NU) else make_lcp("CoastSnap", nu, nv)
    v, u = np.mgrid[1:nv + 1, 1:nu + 1]
    w = find_xyz_6dof(u.ravel(), v.ravel(), TIDAL_OFFSET, BETA, lcp)
    x = w[:, 0]
    img = np.where((x < shoreline_x(k))[:, None], SAND, WATER).astype(float)
    img[np.isnan(x)] = SKY
    img += np.random.default_rng(seed + k).normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8).reshape(nv, nu, 3)


def write_db(path: Path, rows=()):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = SITE
    sheet = [
        ("Station Data", None), ("Eastings", E0), ("Northings", N0), ("Height", CAM_Z),
        ("UTM Zone", "56 H"), ("Default Timezone", TZ), ("Default Timezone Offset From GMT", TZ_OFFSET),
        ("Alternative Timezone", ALT_TZ), ("Alternative Timezone Offset From GMT", ALT_OFFSET),
        ("Xlimit left", 100), ("Xlimit right", 450), ("Ylimit lower", -200), ("Ylimit upper", 200),
        ("Resolution", 2), ("Initial Azimuth Estimate", 100), ("Initial Tilt Estimate", 75),
        ("Initial Roll Estimate", 0), ("Tidal offset", TIDAL_OFFSET), ("Min FOV", 50), ("Max FOV", 75),
        ("Acceptable Accuracy", 5), ("Tide file", "tide_beach.mat"),
        ("Transect file", "SLtransects_beach.mat"), ("Transect averaging region", "[3:11]"),
        ("Characteristic beach slope", SLOPE),
    ]
    for name, x, y, z in GCPS:
        sheet += [("GCP name", name), ("Eastings", E0 + x), ("Northings", N0 + y), ("Elevation", z)]
    sheet += [("GCP combo", "[1 2 3 4 5 6]")]
    for r in sheet:
        ws.append(r)
    db = wb.create_sheet("database")
    db.append(["Site", "User", "Time", "Timezone", "Filename", "Source", "Type", "Time quality"])
    for r in rows:
        db.append(list(r))
    wb.save(path)


@dataclass
class SyntheticSite:
    site: Site
    images: list[ImageEntry]

    @property
    def paths(self) -> Paths:
        return self.site.paths


def image_name(t: datetime, user="Tester", kind="snap") -> tuple[int, str]:
    epoch = datetime_to_epoch(t, TZ_OFFSET)
    return epoch, argus_filename(epoch, SITE, kind, user, "jpg", TZ, TZ_OFFSET)


def save_image(path: Path, img: np.ndarray):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, quality=95)


def build(base: Path, n_images: int = 6, rectify_first: bool = True) -> SyntheticSite:
    from coastsnap.rectify import RectificationSettings, rectify_from_gcps, save_rectified

    paths = Paths.from_base(base)
    for d in (paths.database, paths.tide, paths.transects):
        d.mkdir(parents=True, exist_ok=True)
    write_db(paths.db_file)
    t = np.arange(datetime_to_datenum(FIRST - timedelta(days=30)),
                  datetime_to_datenum(FIRST + timedelta(days=400)), 1 / 24)
    savemat(paths.tide / "tide_beach.mat", {"tide": {"time": t, "level": np.zeros_like(t)}})
    n = len(TRANSECT_YS)
    save_transects(paths.transects / "SLtransects_beach.mat",
                   Transects(np.vstack([np.full(n, TRANSECT_START), np.full(n, TRANSECT_END)]),
                             np.vstack([TRANSECT_YS, TRANSECT_YS]), TRANSECT_YS - TRANSECT_YS[0]))
    site = Site(SITE, paths)
    entries = []
    for k in range(n_images):
        when = FIRST + timedelta(weeks=k)
        epoch, name = image_name(when)
        path = paths.site_images(SITE, "Processed") / str(when.year) / name
        save_image(path, render(k))
        entries.append(ImageEntry(epoch, path))
    if rectify_first:
        settings = RectificationSettings.from_site_db(site.db)
        uv = find_uv(BETA, settings.gcp_xyz(), truth_lcp())
        res = rectify_from_gcps(render(0), uv, settings, tide_level=site.tide_level(entries[0].epoch))
        jpg, mat = site.plan_paths(entries[0].path)
        save_rectified(res, str(jpg), str(mat), world=(settings.res, site.origin))
    return SyntheticSite(site, entries)
