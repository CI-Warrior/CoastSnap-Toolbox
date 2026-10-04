"""Time conversions used across the toolbox.

Ports epoch2Matlab, matlab2Epoch, matlab2Julian, argusDay and
CSPepoch2LocalMatlab. CoastSnap stores image times as UNIX epoch seconds
(GMT) in the filename; tide files and plots use MATLAB datenums in the
site's default (local, non-DST) time zone.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

#: ``datenum(1970, 1, 1)``
EPOCH_DATENUM = 719529.0
_SECONDS_PER_DAY = 86400.0


def epoch_to_datenum(epoch):
    """UNIX epoch seconds to MATLAB datenum (both GMT). Port of ``epoch2Matlab``."""
    return EPOCH_DATENUM + np.asarray(epoch, dtype=float) / _SECONDS_PER_DAY


def datenum_to_epoch(dn):
    """MATLAB datenum (GMT) to whole epoch seconds. Port of ``matlab2Epoch(datenum)``."""
    e = np.round((np.asarray(dn, dtype=float) - EPOCH_DATENUM) * _SECONDS_PER_DAY)
    return e.astype(np.int64) if np.ndim(e) else int(e)


def epoch_to_local_datenum(epoch, gmt_offset: float):
    """Port of ``CSPepoch2LocalMatlab``: epoch to a datenum in a fixed-offset zone."""
    return epoch_to_datenum(epoch) + gmt_offset / 24.0


def datenum_to_datetime(dn: float) -> datetime:
    """Naive datetime for a datenum, to the nearest millisecond.

    A datenum double only resolves about 10 microseconds, so finer digits
    would be noise.
    """
    days = int(np.floor(dn))
    frac = float(dn) - days
    return datetime.fromordinal(days - 366) + timedelta(milliseconds=round(frac * 86400e3))


def datetime_to_datenum(d: datetime) -> float:
    """Datenum of a naive datetime (an aware one is converted to GMT first)."""
    if d.tzinfo is not None:
        d = d.astimezone(timezone.utc).replace(tzinfo=None)
    midnight = datetime(d.year, d.month, d.day)
    return d.toordinal() + 366 + (d - midnight).total_seconds() / _SECONDS_PER_DAY


def epoch_to_datetime(epoch, gmt_offset: float = 0.0) -> datetime:
    """Naive datetime of an epoch in a fixed-offset zone (GMT by default)."""
    return datetime(1970, 1, 1) + timedelta(seconds=float(epoch) + gmt_offset * 3600)


def datetime_to_epoch(d: datetime, gmt_offset: float = 0.0) -> int:
    """Epoch seconds for a naive local datetime ``gmt_offset`` hours ahead of GMT."""
    if d.tzinfo is not None:
        return round(d.timestamp())
    return round((d - datetime(1970, 1, 1)).total_seconds() - gmt_offset * 3600)


def julian_day(dn: float) -> int:
    """Day of year (1 = 1 Jan) of a datenum. Port of ``matlab2Julian``."""
    d = datenum_to_datetime(dn)
    return int(np.fix(dn - datetime_to_datenum(datetime(d.year, 1, 1)))) + 1


def argus_day(epoch) -> str:
    """Argus day folder name such as ``'190_Jul.09'``. Port of ``argusDay``."""
    d = epoch_to_datetime(int(epoch))
    return f"{julian_day(epoch_to_datenum(int(epoch))):03d}_{d:%b}.{d.day:02d}"
