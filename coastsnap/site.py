"""One CoastSnap site: its DB settings and its files.

Bundles what the MATLAB code gets from ``CSPloadPaths`` and
``CSPreadSiteDB``, and ports ``CSPgetImageList`` / ``CSPgetShorelineList``.

    >>> site = Site.open("manly", "D:/CoastSnap")
    >>> for img in site.images():
    ...     print(img.epoch, img.path, site.tide_level(img.epoch))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .config import Paths, load_paths
from .db import read_site_db
from .naming import epoch_of, parse_filename, plan_name, rectified_dir, shoreline_name


@dataclass(frozen=True, order=True)
class ImageEntry:
    """An image (or other product) in the database, ordered by time."""

    epoch: int
    path: Path = field(compare=True)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def parts(self) -> dict:
        return parse_filename(self.path.name)


def _year_dirs(folder: Path):
    """Sub-folders that look like years, as the MATLAB listings assume."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_dir() and len(p.name) == 4)


def _entries(folder: Path, patterns) -> list[ImageEntry]:
    out = []
    for year in _year_dirs(folder):
        for pattern in patterns:
            for f in year.glob(pattern):
                try:
                    out.append(ImageEntry(epoch_of(f.name), f))
                except ValueError:
                    continue   # not a CoastSnap name (e.g. a stray export)
    return sorted(set(out))


class Site:
    """A site sheet of CoastSnapDB.xlsx plus the folders that hold its data."""

    def __init__(self, name: str, paths: Paths, db: dict | None = None):
        self.name = name
        self.paths = paths
        self._db = db

    @classmethod
    def open(cls, name: str, base=None) -> "Site":
        """Site ``name`` in the database at ``base`` (or the configured one)."""
        return cls(name, load_paths(base))

    @property
    def db(self) -> dict:
        """The site sheet as :func:`coastsnap.db.read_site_db` returns it."""
        if self._db is None:
            self._db = read_site_db(self.paths.db_file, self.name)
        return self._db

    @property
    def gmt_offset(self) -> float:
        return float(self.db["timezone"]["gmt_offset"])

    @property
    def origin(self) -> tuple[float, float]:
        o = self.db["origin"]
        return float(o["eastings"]), float(o["northings"])

    # ---- tide and transects ------------------------------------------------

    @cached_property
    def tide(self):
        from .tide import load_tide

        return load_tide(self.paths.tide / self.db["tide"]["file"])

    def tide_level(self, epoch):
        """Tide at image time(s) in epoch seconds (``CSPgetTideLevel``)."""
        return self.tide.at_epoch(epoch, self.gmt_offset)

    @property
    def transect_file(self) -> Path:
        return self.paths.transects / self.db["sl_settings"]["transect_file"]

    @cached_property
    def transects(self):
        from .shoreline.transects import load_transects

        return load_transects(self.transect_file)

    @property
    def averaging_transects(self) -> np.ndarray:
        """0-based indices of the DB's "Transect averaging region" (1-based there)."""
        region = self.db["sl_settings"].get("transect_averaging_region")
        if not region:
            return np.arange(self.transects.count)
        return np.asarray(region, dtype=int) - 1

    @property
    def beach_slope(self) -> float:
        return float(self.db["sl_settings"]["beach_slope"])

    # ---- files -------------------------------------------------------------

    def images(self, kind: str = "Processed", types=None) -> list[ImageEntry]:
        """Images in ``Images/<site>/<kind>/<year>``, oldest first (``CSPgetImageList``).

        ``types`` restricts the image type field, e.g. ``("snap", "timex")``.
        """
        out = _entries(self.paths.site_images(self.name, kind), ("*.jpg", "*.JPG"))
        if types is not None:
            types = {t.lower() for t in types}
            out = [e for e in out if e.parts["type"].lower() in types]
        return out

    def shorelines(self) -> list[ImageEntry]:
        """Saved shoreline ``.mat`` files, oldest first (``CSPgetShorelineList``)."""
        return _entries(self.paths.site_shorelines(self.name), ("*.mat",))

    def plan_paths(self, image_path) -> tuple[Path, Path]:
        """Rectified ``.jpg`` and ``.mat`` for an image in Processed/Registered."""
        image_path = Path(image_path)
        folder = rectified_dir(image_path.parent)
        return folder / plan_name(image_path.name, ".jpg"), folder / plan_name(image_path.name, ".mat")

    def shoreline_path(self, image_name) -> Path:
        """Where the shoreline for an image is (or would be) saved."""
        p = parse_filename(Path(image_name).name)
        return self.paths.shorelines / p["site"] / p["year"] / shoreline_name(Path(image_name).name)

    def find_shoreline(self, image_name) -> Path | None:
        """Saved shoreline for an image, including ``..._registered.mat`` ones."""
        p = self.shoreline_path(image_name)
        if p.is_file():
            return p
        reg = p.with_name(p.stem + "_registered.mat")
        return reg if reg.is_file() else None
