"""Shoreline records and the files they live in.

Reads and writes the same ``.mat`` layout as the MATLAB toolbox so both can
share one CoastSnap database during the transition:

* Rectified products: ``Rectified/<site>/<year>/<...plan...>.mat`` holding
  ``xgrid``, ``ygrid``, ``Iplan`` and ``metadata`` (``CSPGrectifyImage``).
* Shorelines: ``Shorelines/<site>/<year>/<...shoreline...>.mat`` holding one
  ``sl`` struct (``CSPGmapShoreline`` / ``CSPGsaveShoreline``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.io import loadmat, savemat

#: Maps local (x, y, z) points (K, 3) to image (U, V) pixels (K, 2). The
#: rectification module provides this from the camera geometry
#: (``findUVnDOF`` in MATLAB).
UVProjector = Callable[[np.ndarray], np.ndarray]


@dataclass
class ShorelineRecord:
    """The ``sl`` struct saved for each mapped shoreline.

    ``qa`` is True once a person has reviewed the shoreline. Shorelines from
    bulk mapping start unreviewed, as ``out.QA = 0`` does in MATLAB.
    """

    xyz: np.ndarray
    utm: np.ndarray
    utm_zone: str
    uv: np.ndarray
    method: str
    threshold: float | None
    qa: bool = False
    when_done: float = field(default_factory=time.time)

    def to_mat_struct(self) -> dict:
        return {
            "whenDone": float(self.when_done),
            "xyz": np.asarray(self.xyz, dtype=float),
            "UTM": np.asarray(self.utm, dtype=float),
            "UTMzone": str(self.utm_zone),
            "UV": np.asarray(self.uv, dtype=float),
            "method": self.method,
            "threshold": np.nan if self.threshold is None else float(self.threshold),
            "QA": float(bool(self.qa)),
        }

    @classmethod
    def from_mat_struct(cls, sl: dict) -> "ShorelineRecord":
        xyz = np.atleast_2d(np.asarray(sl["xyz"], dtype=float)).reshape(-1, 3)
        threshold = sl.get("threshold")
        threshold = None if threshold is None or np.size(threshold) == 0 else float(threshold)
        if threshold is not None and np.isnan(threshold):
            threshold = None
        uv = sl.get("UV")
        uv = np.full((len(xyz), 2), np.nan) if uv is None else np.asarray(uv, dtype=float).reshape(-1, 2)
        return cls(
            xyz=xyz,
            utm=np.asarray(sl.get("UTM", np.full_like(xyz, np.nan)), dtype=float).reshape(-1, 3),
            utm_zone=str(sl.get("UTMzone", "")),
            uv=uv,
            method=str(sl.get("method", "")),
            threshold=threshold,
            # Shorelines saved before the QA field existed were mapped by hand.
            qa=bool(sl.get("QA", 1)),
            when_done=float(sl.get("whenDone", np.nan)),
        )

    def with_points(self, xy, origin, project_uv: UVProjector | None = None) -> "ShorelineRecord":
        """Copy with new (K, 2) local points; z, UTM and UV are recomputed."""
        z = self.xyz[0, 2] if len(self.xyz) else np.nan
        return make_record(
            xy, z, origin, self.utm_zone, self.method, self.threshold, project_uv, self.qa
        )


def make_record(
    xy,
    rectz: float,
    origin: tuple[float, float],
    utm_zone: str,
    method: str,
    threshold: float | None,
    project_uv: UVProjector | None = None,
    qa: bool = False,
) -> ShorelineRecord:
    """Build a record from local shoreline points, as ``CSPGmapShoreline`` does.

    ``origin`` is the site origin (eastings, northings) from the site DB;
    ``rectz`` is the rectification elevation (tide + offset) the plan image
    was made at. Without ``project_uv`` the UV column is NaN.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    xyz = np.column_stack([xy, np.full(len(xy), rectz)])
    utm = xyz + (origin[0], origin[1], 0.0)
    uv = np.full((len(xy), 2), np.nan)
    if project_uv is not None and len(xy):
        uv = np.asarray(project_uv(xyz), dtype=float).reshape(len(xy), 2)
    return ShorelineRecord(xyz, utm, utm_zone, uv, method, threshold, qa)


def save_shoreline(path, record: ShorelineRecord) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {"sl": record.to_mat_struct()})
    return path


def load_shoreline(path) -> ShorelineRecord:
    return ShorelineRecord.from_mat_struct(loadmat(path, simplify_cells=True)["sl"])


@dataclass
class PlanImage:
    """A rectified plan image and its geometry metadata."""

    xgrid: np.ndarray
    ygrid: np.ndarray
    iplan: np.ndarray
    metadata: dict

    @property
    def rectz(self) -> float:
        return float(self.metadata["rectz"])


def load_plan(path) -> PlanImage:
    """Load a rectified ``...plan....mat`` written by the MATLAB or Python rectifier."""
    d = loadmat(path, simplify_cells=True)
    return PlanImage(
        np.asarray(d["xgrid"], dtype=float).ravel(),
        np.asarray(d["ygrid"], dtype=float).ravel(),
        np.asarray(d["Iplan"]),
        d.get("metadata", {}),
    )


# --------------------------------------------------------------------------
# File naming
# --------------------------------------------------------------------------


def parse_filename(fname: str) -> dict:
    """Split a CoastSnap/Argus filename into parts (port of ``CSPparseFilename``).

    ``1528322400.Thu.Jun.07_08_00_00.AEST.2018.manly.snap.Mitch.jpg``
    """
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


def shoreline_filename(image_name: str) -> str:
    """Shoreline ``.mat`` name for an image (snap/timex/plan, .jpg or .mat)."""
    name = Path(image_name).name
    for kind in ("snap", "timex", "plan"):
        name = name.replace(f".{kind}.", ".shoreline.")
    return str(Path(name).with_suffix(".mat"))


def shoreline_path_for(shoreline_root, image_name: str) -> Path:
    """``<shoreline_root>/<site>/<year>/<name>.mat`` for an image."""
    parts = parse_filename(image_name)
    return Path(shoreline_root) / parts["site"] / parts["year"] / shoreline_filename(image_name)


def list_shorelines(shoreline_root, site: str) -> list[tuple[int, Path]]:
    """All saved shorelines for a site as ``(epochtime, path)``, oldest first.

    Port of ``CSPgetShorelineList`` without the tide lookup.
    """
    out = []
    site_dir = Path(shoreline_root) / site
    if not site_dir.is_dir():
        return out
    for year_dir in site_dir.iterdir():
        if year_dir.is_dir() and len(year_dir.name) == 4:
            for f in year_dir.glob("*.mat"):
                out.append((int(parse_filename(f.name)["epochtime"]), f))
    return sorted(out)
