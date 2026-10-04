"""Site metadata from ``CoastSnapDB.xlsx`` (port of CSPreadSiteDB.m).

Each site sheet is a key/value layout: labels in column A, values in column B.
The ``database`` sheet has one row per submitted image (read by
CSPraw2Processed and the participation statistics).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


def _rows(path, site):
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[site]
        return [(r[0], r[1] if len(r) > 1 else None)
                for r in ws.iter_rows(min_col=1, max_col=2, values_only=True)]
    finally:
        wb.close()


def _num_list(v):
    """MATLAB ``str2num`` on strings such as ``"[1 2 3]"``, ``"1,2,3"`` or ``"[10:30]"``."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return [v]
    s = str(v).strip().strip("[]").replace(",", " ").replace(";", " ")
    out = []
    for tok in s.split():
        if ":" in tok:
            parts = [float(p) for p in tok.split(":")]
            lo, step, hi = (parts[0], 1.0, parts[1]) if len(parts) == 2 else parts
            n = int(np.floor((hi - lo) / step + 1e-10)) + 1
            vals = lo + step * np.arange(max(n, 0))
            out += [int(x) if float(x).is_integer() else float(x) for x in vals]
        else:
            out.append(float(tok) if "." in tok or "e" in tok.lower() else int(tok))
    return out


def read_site_db(path: str, site: str) -> dict:
    """Read one site sheet into the same nested structure as CSPreadSiteDB."""
    rows = _rows(path, site)
    labels = [str(a).strip() if a is not None else None for a, _ in rows]

    def val(label, offset=0):
        try:
            i = labels.index(label)
        except ValueError:
            return None
        return rows[i + offset][1]

    out = {
        "origin": {"eastings": val("Station Data", 1), "northings": val("Station Data", 2),
                   "z": val("Station Data", 3)},
        "UTMzone": val("UTM Zone"),
        "timezone": {
            "name": val("Default Timezone"),
            "gmt_offset": val("Default Timezone Offset From GMT"),
            "alternative": {"name": val("Alternative Timezone"),
                            "gmt_offset": val("Alternative Timezone Offset From GMT")},
        },
        "rect": {
            "xlim": [val("Xlimit left"), val("Xlimit right")],
            "ylim": [val("Ylimit lower"), val("Ylimit upper")],
            "res": val("Resolution"),
            "initial": {"azimuth": val("Initial Azimuth Estimate"),
                        "tilt": val("Initial Tilt Estimate"),
                        "roll": val("Initial Roll Estimate")},
            "tidal_offset": val("Tidal offset"),
            "FOVlims": [val("Min FOV"), val("Max FOV")],
            "accuracylim": val("Acceptable Accuracy"),
        },
        "tide": {"file": val("Tide file")},
        "sl_settings": {
            "transect_file": val("Transect file"),
            "transect_averaging_region": _num_list(val("Transect averaging region")),
            "beach_slope": val("Characteristic beach slope"),
        },
        "gcp": [
            {"name": str(rows[i][1]), "eastings": rows[i + 1][1],
             "northings": rows[i + 2][1], "z": rows[i + 3][1]}
            for i, lab in enumerate(labels) if lab == "GCP name"
        ],
        "gcp_combo": _num_list(val("GCP combo")),
    }
    if None in out["rect"]["FOVlims"]:
        out["rect"]["FOVlims"] = None
    return out


# ---- the "database" sheet: one row per submitted image ---------------------

#: Columns of the ``database`` sheet, in order.
IMAGE_DB_COLUMNS = ("site", "user", "time", "timezone", "filename", "source", "type", "time_quality")

_TIME_FORMATS = ("%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %I:%M %p", "%d/%m/%Y %H:%M")


@dataclass
class ImageRecord:
    """One row of the ``database`` sheet. ``time`` is local to ``timezone``."""

    site: str
    user: str
    time: datetime
    timezone: str
    filename: str
    source: str = ""
    type: str = "Snap"
    time_quality: int = 2

    def gmt_offset(self, site_db: dict) -> float:
        """Offset of this row's time zone, from the site's default or alternative zone."""
        tz = site_db["timezone"]
        if self.timezone == tz["name"]:
            return tz["gmt_offset"]
        if self.timezone == tz["alternative"]["name"]:
            return tz["alternative"]["gmt_offset"]
        raise ValueError(f"{self.filename}: time zone {self.timezone!r} is not one of the "
                         f"site's zones ({tz['name']}, {tz['alternative']['name']})")

    def epoch(self, site_db: dict) -> int:
        from .timeutils import datetime_to_epoch

        return datetime_to_epoch(self.time, self.gmt_offset(site_db))


def parse_db_time(value) -> datetime:
    """A ``database`` sheet time: an Excel date cell or a dd/mm/yyyy string."""
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"cannot read image time {value!r} (expected dd/mm/yyyy HH:MM[:SS] [AM/PM])")


def read_image_db(path, site: str | None = None) -> list[ImageRecord]:
    """Rows of the ``database`` sheet, optionally for one site only."""
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(wb["database"].iter_rows(min_row=2, max_col=8, values_only=True))
    finally:
        wb.close()
    out = []
    for r in rows:
        r = tuple(r) + (None,) * (8 - len(r))
        if not r[0] or r[2] is None:
            continue
        rec = ImageRecord(
            site=str(r[0]).strip(), user=str(r[1] or "").strip(), time=parse_db_time(r[2]),
            timezone=str(r[3] or "").strip(), filename=str(r[4] or "").strip(),
            source=str(r[5] or "").strip(), type=str(r[6] or "Snap").strip(),
            time_quality=int(r[7]) if r[7] not in (None, "") else 2,
        )
        if site is None or rec.site == site:
            out.append(rec)
    return out


def append_image_db(path, records) -> None:
    """Append rows to the ``database`` sheet (``xlswrite`` in CSPraw2Processed).

    Times are written as dd/mm/yyyy HH:MM text, as the MATLAB code writes them.
    """
    from openpyxl import load_workbook

    wb = load_workbook(path)
    try:
        ws = wb["database"]
        for rec in records:
            ws.append([rec.site, rec.user, rec.time.strftime("%d/%m/%Y %H:%M"), rec.timezone,
                       rec.filename, rec.source, rec.type, rec.time_quality])
        wb.save(path)
    finally:
        wb.close()
