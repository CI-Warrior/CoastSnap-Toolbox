"""Tide levels for image times (port of CSPgetTideLevel).

A site's tide file (``Tide Data/<site tide file>``, named in the site DB) is a
``.mat`` holding a ``tide`` struct with ``time`` as MATLAB datenums in the
site's default time zone and ``level`` in metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .timeutils import epoch_to_local_datenum


@dataclass(frozen=True)
class TideRecord:
    time: np.ndarray    # datenum, site default time zone
    level: np.ndarray   # metres

    def at_datenum(self, dn):
        """Linear interpolation; NaN outside the record, as MATLAB's interp1."""
        out = np.interp(dn, self.time, self.level, left=np.nan, right=np.nan)
        return float(out) if np.ndim(out) == 0 else out

    def at_epoch(self, epoch, gmt_offset: float):
        """Tide at UNIX epoch times (GMT) for a tide file in GMT+``gmt_offset``."""
        return self.at_datenum(epoch_to_local_datenum(epoch, gmt_offset))


def load_tide(path) -> TideRecord:
    return _load_tide(str(Path(path).resolve()))


@lru_cache(maxsize=8)
def _load_tide(path: str) -> TideRecord:
    tide = loadmat(path, simplify_cells=True)["tide"]
    t = np.asarray(tide["time"], dtype=float).ravel()
    lev = np.asarray(tide["level"], dtype=float).ravel()
    order = np.argsort(t, kind="stable")
    return TideRecord(t[order], lev[order])


def get_tide_level(epoch, site_db: dict, tide_dir):
    """Tide level at each epoch time for a site (``CSPgetTideLevel``).

    ``site_db`` is the dict from :func:`coastsnap.db.read_site_db` and
    ``tide_dir`` the ``Tide Data`` folder (``Paths.tide``).
    """
    tide = load_tide(Path(tide_dir) / site_db["tide"]["file"])
    return tide.at_epoch(epoch, site_db["timezone"]["gmt_offset"])
