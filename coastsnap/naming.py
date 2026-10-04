"""CoastSnap (Argus-style) file names.

``1528322400.Thu.Jun.07_08_00_00.AEST.2018.manly.snap.Mitch.jpg`` is

    epoch . day . month . DD_hh_mm_ss . tz . year . site . type . user . format

with the date written in the site's default time zone. Ports
``CSPargusFilename`` and ``CSPparseFilename``, plus the ``strrep`` renames the
GUI uses to go from an image to its rectified product or shoreline.
"""

from __future__ import annotations

import re
from pathlib import Path

from .timeutils import epoch_to_datetime

_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

#: Image types in the Processed/Registered folders and what replaces them in
#: product names.
IMAGE_TYPES = ("snap", "timex", "daytimex")


def clean_user(user: str) -> str:
    """User name as it goes in a file name: no underscores, dots or spaces.

    ``CSPraw2Processed`` strips non-word characters and then ``_`` and ``.``,
    since both would break :func:`parse_filename`.
    """
    return re.sub(r"[^\w']|_", "", str(user))


def argus_filename(epoch, site: str, image_type: str, user: str = "", fmt: str = "jpg",
                   tz_name: str = "GMT", tz_offset: float = 0.0, camera=-1,
                   short: bool = False) -> str:
    """Build a CoastSnap file name (port of ``CSPargusFilename``).

    MATLAB looks the time zone up from the site DB; pass the site's default
    ``timezone.name`` and ``timezone.gmt_offset`` (hours) here. ``camera`` is
    -1 for CoastSnap (no camera field), an int or str for ``.c<n>``, or None
    for the ``.c*`` wildcard.
    """
    t = int(float(epoch) // 1)
    if camera is None:
        cam = ".c*"
    elif isinstance(camera, str):
        cam = ".c" + camera
    elif camera == -1:
        cam = ""
    else:
        cam = f".c{int(camera)}"
    d = epoch_to_datetime(t, tz_offset)
    if short:
        name = f"{t}.{d:%H%M%S}.{site}{cam}.{image_type}.{fmt}"
    else:
        name = (f"{t}.{_DAYS[d.weekday()]}.{_MONTHS[d.month - 1]}."
                f"{d.day:02d}_{d.hour:02d}_{d.minute:02d}_{d.second:02d}."
                f"{tz_name}.{d.year:4d}.{site}{cam}.{image_type}.{user}.{fmt}")
    # Empty type, user or format leave doubled or trailing dots.
    name = re.sub(r"\.{2,}", ".", name)
    return name.rstrip(".")


def parse_filename(fname) -> dict:
    """Split a CoastSnap file name into its string parts (``CSPparseFilename``)."""
    c = Path(fname).name.split(".")
    if len(c) < 10:
        raise ValueError(f"{fname!r} is not a CoastSnap filename")
    return {
        "epochtime": c[0],
        "dayname": c[1],
        "month": c[2],
        "day": c[3][0:2],
        "hour": c[3][3:5],
        "min": c[3][6:8],
        "sec": c[3][9:11],
        "timezone": c[4],
        "year": c[5],
        "site": c[6],
        "type": c[7],
        "user": c[8],
        "format": c[9],
    }


def epoch_of(fname) -> int:
    """Epoch time encoded in a CoastSnap file name."""
    return int(Path(fname).name.split(".", 1)[0])


def _retype(name: str, new_type: str) -> str:
    # MATLAB uses strrep(name, 'timex', 'plan'), which turns "daytimex" into
    # "dayplan"; keep that so both toolboxes find each other's files. Only the
    # type field is replaced, so a user called "snapper" is left alone.
    c = name.split(".")
    if len(c) >= 10:
        kind = c[7]
        c[7] = "day" + new_type if kind in ("daytimex", "dayplan", "dayshoreline") else new_type
        return ".".join(c)
    for kind in IMAGE_TYPES + ("plan",):
        name = name.replace(f".{kind}.", f".{new_type}.")
    return name


def plan_name(image_name: str, ext: str = ".jpg") -> str:
    """Rectified product name for an image (``snap``/``timex`` become ``plan``)."""
    return str(Path(_retype(Path(image_name).name, "plan")).with_suffix(ext))


def shoreline_name(image_name: str, ext: str = ".mat") -> str:
    """Shoreline file name for an image or plan product."""
    return str(Path(_retype(Path(image_name).name, "shoreline")).with_suffix(ext))


def rectified_dir(image_dir) -> Path:
    """``.../<site>/Processed/<year>`` (or ``Registered``) to ``.../Rectified/<year>``."""
    parts = ["Rectified" if p in ("Processed", "Registered") else p for p in Path(image_dir).parts]
    return Path(*parts)
