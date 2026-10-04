"""Bring new images into the database.

* :func:`raw_to_processed` - CSPraw2Processed: rename the images in
  ``Images/<site>/Raw`` using their rows in the ``database`` sheet and move
  them to ``Processed/<year>``.
* :func:`raw_to_processed_no_db` - CSPraw2ProcessedNoDB: the same for
  cameras without DB rows, taking the time from EXIF or a Spotteron file name.
* :func:`sort_shared_images` - CSPprocessSharePath / CSPprocessEZDetach: sort
  emailed or shared images into each site's Raw folder by their GPS tag and
  add their DB rows.

Every function plans all moves first and raises before touching any file if
an image cannot be placed, where MATLAB stops part way through.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .db import ImageRecord, append_image_db, read_image_db
from .naming import argus_filename, clean_user
from .site import Site
from .timeutils import datetime_to_epoch, epoch_to_datetime

IMAGE_PATTERNS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG")


@dataclass
class Move:
    source: Path
    target: Path
    record: ImageRecord | None = None   # a new DB row to append, if any

    def __str__(self):
        return f"{self.source.name} -> {self.target}"


def _images(folder: Path) -> list[Path]:
    files = set()
    for pattern in IMAGE_PATTERNS:
        files.update(folder.glob(pattern))
    return sorted(files)


def _target(site: Site, epoch: int, image_type: str, user: str) -> Path:
    tz = site.db["timezone"]
    name = argus_filename(epoch, site.name, image_type.lower(), clean_user(user), "jpg",
                          tz["name"], tz["gmt_offset"])
    year = epoch_to_datetime(epoch, tz["gmt_offset"]).year
    return site.paths.site_images(site.name, "Processed") / str(year) / name


def apply(moves: list[Move], db_file=None) -> list[Move]:
    """Carry out planned moves, appending any new DB rows first."""
    new_rows = [m.record for m in moves if m.record is not None]
    if new_rows:
        append_image_db(db_file, new_rows)
    for m in moves:
        m.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(m.source), str(m.target))
    return moves


# ---- Raw -> Processed using the database sheet ------------------------------------

#: 4K Stogram (Instagram downloader) names start with a 39-character token.
STOGRAM_TOKEN_LENGTH = 39


def is_stogram(name: str) -> bool:
    return len(name.split("_", 1)[0]) == STOGRAM_TOKEN_LENGTH


def stogram_time(name: str) -> datetime:
    """Upload time in a 4K Stogram name (``yyyy-mm-dd HH.MM...``)."""
    return datetime.strptime(name[:16], "%Y-%m-%d %H.%M")


#: ``ask(path, upload_time, site_db) -> ImageRecord`` fills in the DB row for
#: an Instagram image (user, time zone, time quality and time), as the
#: MATLAB dialogs do.
AskStogram = Callable[[Path, datetime, dict], ImageRecord]


def plan_raw_to_processed(site: Site, ask: AskStogram | None = None) -> list[Move]:
    """Work out where every image in Raw goes, without moving anything."""
    raw = site.paths.site_images(site.name, "Raw")
    rows = read_image_db(site.paths.db_file)
    by_name = {}
    for r in rows:
        by_name.setdefault(r.filename, []).append(r)
    moves, problems = [], []
    for f in _images(raw):
        new = None
        if is_stogram(f.name):
            if ask is None:
                problems.append(f"{f.name}: Instagram image needs its user and time (run interactively)")
                continue
            rec = new = ask(f, stogram_time(f.name), site.db)
        else:
            found = by_name.get(f.name, [])
            if len(found) > 1:
                problems.append(f"{f.name}: more than one DB row has this file name")
                continue
            if not found:
                problems.append(f"{f.name}: no DB row has this file name")
                continue
            rec = found[0]
        try:
            epoch = rec.epoch(site.db)
        except ValueError as err:
            problems.append(str(err))
            continue
        moves.append(Move(f, _target(site, epoch, re.sub(r"\W", "", rec.type), rec.user), new))
    if problems:
        raise ValueError("cannot process these Raw images; nothing was moved:\n  " + "\n  ".join(problems))
    return moves


def raw_to_processed(site: Site, ask: AskStogram | None = None, dry_run: bool = False) -> list[Move]:
    moves = plan_raw_to_processed(site, ask)
    return moves if dry_run else apply(moves, site.paths.db_file)


# ---- Raw -> Processed without the database sheet ----------------------------------


def exif_time(path) -> datetime | None:
    """EXIF capture time (``DateTimeOriginal``, else ``DateTime``)."""
    from PIL import Image

    with Image.open(path) as im:
        exif = im.getexif()
        value = exif.get_ifd(0x8769).get(36867) or exif.get(306)
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def exif_latlon(path) -> tuple[float, float] | None:
    """Signed GPS latitude and longitude in degrees, if tagged."""
    from PIL import Image

    with Image.open(path) as im:
        gps = im.getexif().get_ifd(0x8825)
    if 2 not in gps or 4 not in gps:
        return None

    def deg(dms, ref):
        d, m, s = (float(v) for v in dms)
        value = d + m / 60 + s / 3600
        return -value if str(ref).upper() in ("S", "W") else value

    return deg(gps[2], gps.get(1, "N")), deg(gps[4], gps.get(3, "E"))


def spotteron_time(name: str) -> datetime | None:
    """Time from a Spotteron "Create Spot Package" name (``yyyymmddHHMM...``)."""
    try:
        return datetime.strptime(name[:12], "%Y%m%d%H%M")
    except ValueError:
        return None


def plan_raw_to_processed_no_db(site: Site, user: str, image_type: str = "snap",
                                timezone: str | None = None) -> list[Move]:
    raw = site.paths.site_images(site.name, "Raw")
    tz = site.db["timezone"]
    timezone = timezone or tz["name"]
    zones = {tz["name"]: tz["gmt_offset"], tz["alternative"]["name"]: tz["alternative"]["gmt_offset"]}
    if timezone not in zones:
        raise ValueError(f"time zone {timezone!r} is not one of the site's zones {sorted(z for z in zones if z)}")
    moves, problems = [], []
    for f in _images(raw):
        t = exif_time(f) or spotteron_time(f.name)
        if t is None:
            problems.append(f"{f.name}: no EXIF time and not a Spotteron file name")
            continue
        moves.append(Move(f, _target(site, datetime_to_epoch(t, zones[timezone]), image_type, user)))
    if problems:
        raise ValueError("cannot process these Raw images; nothing was moved:\n  " + "\n  ".join(problems))
    return moves


def raw_to_processed_no_db(site: Site, user: str, image_type: str = "snap", timezone: str | None = None,
                           dry_run: bool = False) -> list[Move]:
    moves = plan_raw_to_processed_no_db(site, user, image_type, timezone)
    return moves if dry_run else apply(moves)


# ---- shared folders ---------------------------------------------------------------

#: Sites within this many degrees of an image's GPS tag claim it.
GPS_THRESHOLD_DEG = 0.05
UNCLASSIFIED = "unclassified"


def site_locations(sites: list[Site]) -> dict[str, tuple[float, float]]:
    from .utm import utm2deg

    out = {}
    for s in sites:
        lat, lon = utm2deg(*s.origin, s.db["UTMzone"])
        out[s.name] = (float(lat), float(lon))
    return out


def classify_by_gps(latlon, locations: dict, threshold: float = GPS_THRESHOLD_DEG) -> str:
    """The one site within ``threshold`` degrees, else ``"unclassified"``."""
    if latlon is None:
        return UNCLASSIFIED
    near = [name for name, (lat, lon) in locations.items()
            if np.hypot(latlon[0] - lat, latlon[1] - lon) < threshold]
    return near[0] if len(near) == 1 else UNCLASSIFIED


def plan_shared_images(folder, sites: list[Site], user: str, timezone: str, source: str = "Email",
                       image_type: str = "Snap") -> list[Move]:
    """Sort a folder of shared images into ``Images/<site>/Raw`` by GPS.

    Images with no GPS tag, or near none or several sites, go to
    ``Images/unclassified``. Each image gets a DB row; the time comes from
    EXIF (time quality 1) or, for EZDetach names ``<x>_<yyyymmddHHMM>_...``,
    the time the email was sent (quality 2).
    """
    if not sites:
        raise ValueError("no sites given")
    paths = sites[0].paths
    locations = site_locations(sites)
    moves, problems = [], []
    for f in _images(Path(folder)):
        t, quality = exif_time(f), 1
        if t is None:
            parts = f.name.split("_")
            t, quality = (spotteron_time(parts[1]) if len(parts) > 1 else None), 2
        if t is None:
            problems.append(f"{f.name}: no EXIF time and no time in the file name")
            continue
        name = re.sub(r"\.jpeg$", ".jpg", f.name, flags=re.I)
        site = classify_by_gps(exif_latlon(f), locations)
        target = paths.images / UNCLASSIFIED / name if site == UNCLASSIFIED else paths.site_images(site, "Raw") / name
        rec = ImageRecord(site, user, t, timezone, name, source, image_type, quality)
        moves.append(Move(f, target, rec))
    if problems:
        raise ValueError("cannot sort these images; nothing was moved:\n  " + "\n  ".join(problems))
    return moves


def sort_shared_images(folder, sites: list[Site], user: str, timezone: str, source: str = "Email",
                       dry_run: bool = False) -> list[Move]:
    moves = plan_shared_images(folder, sites, user, timezone, source)
    return moves if dry_run else apply(moves, sites[0].paths.db_file)
