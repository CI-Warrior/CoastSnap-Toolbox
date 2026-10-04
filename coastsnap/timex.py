"""Daily time-exposure images (port of CSPmakeDayTimex).

Averages the snaps taken between 07:00 and 18:59 local time on each day that
has more than two of them, and saves a ``daytimex`` image beside them.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .naming import argus_filename, parse_filename
from .site import Site
from .timeutils import datetime_to_epoch, epoch_to_datetime

DAY_HOURS = range(7, 19)
MIN_IMAGES = 3


def make_day_timex(site: Site, first_day: date, last_day: date, kind: str = "Processed",
                   on_day=None) -> list[Path]:
    """Write one ``daytimex`` image per qualifying day; returns their paths.

    The file is named for local midnight of the day. MATLAB adds the GMT
    offset where it should subtract it, so its daytimex names are two
    offsets later than midnight; this uses midnight. Images whose size
    differs from the day's first image are left out.
    """
    from PIL import Image

    tz = site.db["timezone"]
    by_day = defaultdict(list)
    for e in site.images(kind, types=("snap", "timex")):
        t = epoch_to_datetime(e.epoch, tz["gmt_offset"])
        if t.hour in DAY_HOURS and first_day <= t.date() <= last_day:
            by_day[t.date()].append(e)
    out = []
    for day in sorted(by_day):
        entries = by_day[day]
        if len(entries) < MIN_IMAGES:
            continue
        total, n, size = None, 0, None
        for e in entries:
            I = np.asarray(Image.open(e.path).convert("RGB"), dtype=np.uint32)
            if size is None:
                size, total = I.shape, np.zeros(I.shape, dtype=np.uint32)
            if I.shape != size:
                continue
            total += I
            n += 1
        if n < MIN_IMAGES:
            continue
        mean = np.floor(total / n + 0.5).astype(np.uint8)    # imdivide rounds half up
        last = entries[-1]
        epoch = datetime_to_epoch(datetime(day.year, day.month, day.day), tz["gmt_offset"])
        name = argus_filename(epoch, site.name, "daytimex", parse_filename(last.name)["user"], "jpg",
                              tz["name"], tz["gmt_offset"])
        path = last.path.parent / name
        Image.fromarray(mean).save(path, quality=100)
        out.append(path)
        if on_day:
            on_day(day, n, path)
    return out
