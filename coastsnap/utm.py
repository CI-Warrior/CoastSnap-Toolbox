"""UTM to latitude/longitude (port of tools/utm2deg.m, WGS84).

Used to place sites when sorting emailed images by their GPS tags; accurate
to well under a metre, which is plenty for that.
"""

from __future__ import annotations

import numpy as np

_SA = 6378137.0
_SB = 6356752.314245


def parse_zone(zone: str) -> tuple[int, bool]:
    """``"56 H"`` or ``"56H"`` to (zone number, northern hemisphere?)."""
    z = str(zone).replace(" ", "").upper()
    number, letter = int(z[:-1]), z[-1]
    if not "C" <= letter <= "X":
        raise ValueError(f"UTM zone {zone!r} should look like '56 H'")
    return number, letter > "M"


def utm2deg(x, y, zone: str):
    """Latitude and longitude in degrees for UTM eastings/northings in ``zone``."""
    number, north = parse_zone(zone)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    e2 = np.sqrt(_SA**2 - _SB**2) / _SB
    e2sq = e2**2
    c = _SA**2 / _SB
    X = x - 500000
    Y = y if north else y - 10000000
    S = number * 6 - 183
    lat = Y / (6366197.724 * 0.9996)
    cos2 = np.cos(lat) ** 2
    v = c / np.sqrt(1 + e2sq * cos2) * 0.9996
    a = X / v
    a1 = np.sin(2 * lat)
    a2 = a1 * cos2
    j2 = lat + a1 / 2
    j4 = (3 * j2 + a2) / 4
    j6 = (5 * j4 + a2 * cos2) / 3
    alfa = 0.75 * e2sq
    beta = 5 / 3 * alfa**2
    gama = 35 / 27 * alfa**3
    Bm = 0.9996 * c * (lat - alfa * j2 + beta * j4 - gama * j6)
    b = (Y - Bm) / v
    epsi = e2sq * a**2 / 2 * cos2
    eps = a * (1 - epsi / 3)
    nab = b * (1 - epsi) + lat
    delt = np.arctan(np.sinh(eps) / np.cos(nab))
    tao = np.arctan(np.cos(delt) * np.tan(nab))
    lon = np.degrees(delt) + S
    latitude = (lat + (1 + e2sq * cos2 - 1.5 * e2sq * np.sin(lat) * np.cos(lat) * (tao - lat)) * (tao - lat)) * 180 / np.pi
    return latitude, lon
