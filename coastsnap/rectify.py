"""Plan-view rectification (port of buildRectProducts, makeFinalImages and the
rectification steps of CSPGrectifyImage / CSPGloadExistingGeometry).

Images are ``(rows, cols, 3)`` arrays, row 0 at the top, as read by Pillow or
imageio.  Plan products are ``(ny, nx, 3)`` with row 0 at ``y[0]`` (the
southern edge), matching MATLAB's ``Iplan``; flip vertically to display or
save it the way MATLAB's ``imwrite(flipud(Iplan))`` does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .camera.geometry import find_uv, find_xyz_6dof
from .camera.lens import LensCalibration, make_lcp
from .camera.solve import Geometry, solve_geometry
from .naming import plan_name, rectified_dir

# MATLAB rgb2gray weights.
_GRAY = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])


def matlab_colon(lo: float, step: float, hi: float) -> np.ndarray:
    """``lo:step:hi`` with MATLAB's end-point tolerance."""
    n = int(np.floor((hi - lo) / step + 1e-10)) + 1
    return lo + step * np.arange(max(n, 0))


def matlab_round(a):
    """Round half away from zero, as MATLAB's ``round`` does (NaN stays NaN)."""
    a = np.asarray(a, dtype=float)
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def to_uint8(a) -> np.ndarray:
    """MATLAB ``uint8(double)``: round half away, saturate, NaN to 0."""
    a = matlab_round(a)
    return np.clip(np.nan_to_num(a, nan=0.0), 0, 255).astype(np.uint8)


def rgb2gray(I) -> np.ndarray:
    return np.asarray(I, dtype=float) @ _GRAY


@dataclass
class PlanImage:
    """Output of makeFinalImages."""

    x: np.ndarray
    y: np.ndarray
    timex: np.ndarray       # uint8 (ny, nx, 3); this is MATLAB's Iplan
    bright: np.ndarray
    dark: np.ndarray
    N: np.ndarray           # number of frames contributing to each cell
    dn: float = None

    @property
    def extent(self):
        """``(xmin, xmax, ymin, ymax)`` for ``imshow(..., origin="lower")``."""
        return (self.x[0], self.x[-1], self.y[0], self.y[-1])


@dataclass
class PlanProducts:
    """Accumulator for buildRectProducts (timex, bright and dark over frames).

    ``rect_xy`` is ``[xmin, dx, xmax, ymin, dy, ymax]`` and ``z`` the
    rectification elevation, both in the local (origin-shifted) frame.
    """

    rect_xy: tuple
    z: float
    dn: list = field(default_factory=list)

    def __post_init__(self):
        xy = self.rect_xy
        self.x = matlab_colon(xy[0], xy[1], xy[2])
        self.y = matlab_colon(xy[3], xy[4], xy[5])
        X, Y = np.meshgrid(self.x, self.y)
        self.xyz = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, float(self.z))])
        shape = (self.y.size, self.x.size)
        self.sumI = np.zeros(shape + (3,))
        self.bright = np.zeros(shape + (3,))
        self.dark = np.zeros(shape + (3,))
        self.N = np.zeros(shape)

    def _sample(self, I, beta, lcp, cull_behind):
        NV, NU = I.shape[:2]
        UV = matlab_round(find_uv(beta, self.xyz, lcp, cull_behind=cull_behind))
        good = (UV[:, 0] >= 1) & (UV[:, 0] <= NU) & (UV[:, 1] >= 1) & (UV[:, 1] <= NV)
        rows, cols = np.unravel_index(np.flatnonzero(good), self.N.shape)
        pix = I[UV[good, 1].astype(int) - 1, UV[good, 0].astype(int) - 1, :3]
        return rows, cols, pix

    def add(self, dn, I, beta, lcp: LensCalibration, cull_behind: bool = True):
        """Add one oblique frame taken with geometry ``beta`` and lens ``lcp``."""
        I = np.asarray(I, dtype=float)
        rows, cols, pix = self._sample(I, beta, lcp, cull_behind)
        if not self.dn:
            for a in (self.sumI, self.bright, self.dark):
                a[rows, cols] = pix
            self.N[rows, cols] = 1
        else:
            g = rgb2gray(pix / 255)
            darker = g < rgb2gray(self.dark[rows, cols] / 255)
            brighter = g > rgb2gray(self.bright[rows, cols] / 255)
            self.sumI[rows, cols] += pix
            self.bright[rows[brighter], cols[brighter]] = pix[brighter]
            self.dark[rows[darker], cols[darker]] = pix[darker]
            self.N[rows, cols] += 1
        self.dn.append(dn)
        return self

    def final(self) -> PlanImage:
        """makeFinalImages: timex is the per-cell mean over contributing frames."""
        with np.errstate(invalid="ignore", divide="ignore"):
            timex = self.sumI / self.N[..., None]
        return PlanImage(
            x=self.x, y=self.y, timex=to_uint8(timex), bright=to_uint8(self.bright),
            dark=to_uint8(self.dark), N=self.N.copy(), dn=self.dn[0] if self.dn else None,
        )


def rectify_image(I, beta, lcp: LensCalibration, rect_xy, z, dn=1,
                  cull_behind: bool = True) -> PlanImage:
    """Rectify one oblique image onto the plane at elevation ``z``."""
    return PlanProducts(tuple(rect_xy), z).add(dn, I, beta, lcp, cull_behind).final()


# ---- site workflow ----------------------------------------------------------


@dataclass
class GCP:
    name: str
    eastings: float
    northings: float
    z: float


@dataclass
class RectificationSettings:
    """The rectification fields of a site's CoastSnapDB sheet (CSPreadSiteDB)."""

    origin_eastings: float
    origin_northings: float
    origin_z: float
    xlim: tuple
    ylim: tuple
    res: float
    initial_azimuth: float      # degrees
    initial_tilt: float
    initial_roll: float
    tidal_offset: float = 0.0
    fov_limits: tuple = None    # (min, max) horizontal FOV in degrees
    gcps: list = field(default_factory=list)
    gcp_combo: list = None      # 1-based indices into gcps, as in the DB
    accuracy_limit: float = None  # largest acceptable GCP RMSE in pixels

    @classmethod
    def from_site_db(cls, site: dict) -> "RectificationSettings":
        """From the nested dict returned by :func:`coastsnap.db.read_site_db`."""
        rect = site["rect"]
        return cls(
            origin_eastings=site["origin"]["eastings"],
            origin_northings=site["origin"]["northings"],
            origin_z=site["origin"]["z"],
            xlim=tuple(rect["xlim"]), ylim=tuple(rect["ylim"]), res=rect["res"],
            initial_azimuth=rect["initial"]["azimuth"],
            initial_tilt=rect["initial"]["tilt"],
            initial_roll=rect["initial"]["roll"],
            tidal_offset=rect.get("tidal_offset", 0.0) or 0.0,
            fov_limits=tuple(rect["FOVlims"]) if rect.get("FOVlims") is not None else None,
            gcps=[GCP(**g) for g in site.get("gcp", [])],
            gcp_combo=site.get("gcp_combo"),
            accuracy_limit=rect.get("accuracylim"),
        )

    @property
    def rect_xy(self):
        return (self.xlim[0], self.res, self.xlim[1], self.ylim[0], self.res, self.ylim[1])

    @property
    def beta0(self) -> np.ndarray:
        """Camera at the local origin with the initial angle estimates (radians)."""
        angles = np.radians([self.initial_azimuth, self.initial_tilt, self.initial_roll])
        return np.concatenate([[0.0, 0.0, self.origin_z], angles])

    def selected_gcps(self) -> list:
        if not self.gcp_combo:
            return list(self.gcps)
        return [self.gcps[i - 1] for i in np.atleast_1d(self.gcp_combo)]

    def unused_gcps(self) -> list:
        """GCPs not in the combo, which CSPGrectifyImage projects as an accuracy check."""
        used = {id(g) for g in self.selected_gcps()}
        return [g for g in self.gcps if id(g) not in used]

    def gcp_xyz(self, gcps=None) -> np.ndarray:
        """GCP coordinates in the local frame (origin subtracted)."""
        gcps = self.selected_gcps() if gcps is None else gcps
        return np.array([[g.eastings - self.origin_eastings,
                          g.northings - self.origin_northings, g.z] for g in gcps])


@dataclass
class RectificationResult:
    plan: PlanImage
    geometry: Geometry
    metadata: dict


def rectify_from_gcps(I, uv_picked, settings: RectificationSettings, tide_level: float,
                      camera: str = "CoastSnap", lcp: LensCalibration = None,
                      gcps=None, refine_focal: bool = False, free_distortion=(),
                      when_done: float = None) -> RectificationResult:
    """The non-interactive part of CSPGrectifyImage.

    ``uv_picked`` is ``(N, 2)`` pixel coordinates of the GCPs in the order of
    ``gcps`` (default: ``settings.selected_gcps()``), using MATLAB's 1-based
    pixel convention (pixel centres at 1..NU, 1..NV).  Pass ``lcp`` to use a
    lens calibration (for example a wide-angle phone calibrated with OpenCV)
    instead of the generic distortion-free CoastSnap profile.
    """
    I = np.asarray(I)
    NV, NU = I.shape[:2]
    if lcp is None:
        lcp = make_lcp(camera, NU, NV)
    xyz = settings.gcp_xyz(gcps)
    geom = solve_geometry(
        xyz, uv_picked, lcp, settings.beta0, known_flags=(1, 1, 1, 0, 0, 0),
        fov_limits=settings.fov_limits, refine_focal=refine_focal,
        free_distortion=free_distortion,
    )
    z = tide_level + settings.tidal_offset
    plan = rectify_image(I, geom.beta, geom.lcp, settings.rect_xy, z)
    metadata = {
        "rectz": z,
        "gcps": {"xyzMeas": xyz, "UVpicked": np.asarray(uv_picked, dtype=float)},
        "geom": geometry_to_matlab(geom),
    }
    if when_done is not None:
        metadata["whenDone"] = when_done
    return RectificationResult(plan=plan, geometry=geom, metadata=metadata)


def geometry_to_matlab(geom: Geometry) -> dict:
    """``metadata.geom`` as CSPGrectifyImage saves it."""
    return {
        "betas": geom.beta, "CI": geom.ci, "MSE": geom.mse, "lcp": geom.lcp.to_matlab(),
        "knownFlags": np.asarray(geom.known_flags, dtype=float),
        "knowns": np.asarray(geom.knowns, dtype=float),
    }


def rectified_paths(image_path: str) -> tuple[str, str]:
    """Plan ``.jpg`` and ``.mat`` paths for an oblique image, following the
    MATLAB naming (``Processed``/``Registered`` to ``Rectified``, ``snap``/``timex``
    to ``plan``)."""
    folder = rectified_dir(os.path.dirname(image_path) or ".")
    jpg = os.path.join(str(folder), plan_name(os.path.basename(image_path), ".jpg"))
    return jpg, os.path.splitext(jpg)[0] + ".mat"


def world_file(x, y, res: float, origin=(0.0, 0.0)) -> np.ndarray:
    """The six world-file values for a plan image saved north-up.

    ``[res, 0, 0, -res, west, north]`` where (west, north) is the centre of
    the top-left pixel in real-world (UTM) coordinates, as CSPGrectifyImage
    writes it.
    """
    return np.array([res, 0.0, 0.0, -res, float(np.min(x)) + origin[0], float(np.max(y)) + origin[1]])


def write_world_file(path, values) -> str:
    """Write a ``.jpw`` world file.

    MATLAB's ``save -ascii`` keeps 8 significant digits, which rounds UTM
    northings to about 0.1 m; this writes full precision instead.
    """
    with open(path, "w") as f:
        f.write("".join(f"{v:.6f}\n" for v in values))
    return str(path)


def save_rectified(result: RectificationResult, jpg_path: str, mat_path: str = None,
                   world: tuple = None):
    """Write the plan image and a MATLAB-readable ``.mat`` (xgrid, ygrid, Iplan, metadata).

    Pass ``world=(res, (origin_eastings, origin_northings))`` to also write the
    ``.jpw`` world file next to the image, so GIS tools can place it.
    """
    from PIL import Image
    from scipy.io import savemat

    mat_path = mat_path or os.path.splitext(jpg_path)[0] + ".mat"
    os.makedirs(os.path.dirname(os.path.abspath(jpg_path)), exist_ok=True)
    Image.fromarray(np.flipud(result.plan.timex)).save(jpg_path, quality=95)
    savemat(mat_path, {
        "xgrid": result.plan.x[None, :], "ygrid": result.plan.y[None, :],
        "Iplan": result.plan.timex, "metadata": result.metadata,
    }, do_compression=True)
    if world is not None:
        res, origin = world
        write_world_file(os.path.splitext(jpg_path)[0] + ".jpw",
                         world_file(result.plan.x, result.plan.y, res, origin))
    return jpg_path, mat_path


class AccuracyError(ValueError):
    """Raised when a GCP fit is worse than the site's "Acceptable Accuracy"."""


def check_accuracy(geometry: Geometry, limit) -> None:
    """Refuse a fit whose RMSE (pixels) exceeds the site DB ``rect.accuracylim``.

    CSPGrectifyImage shows the RMSE and does not save the result in that case.
    ``limit=None`` (not set in the DB) accepts any fit.
    """
    if limit is not None and geometry.rmse > float(limit):
        raise AccuracyError(
            f"RMSE {geometry.rmse:.1f} px is above the site limit of {float(limit):g} px; "
            "the rectification was not saved. Re-pick the GCPs to reduce the error.")


def load_geometry(mat_path: str, model: str = "analytic") -> tuple[np.ndarray, LensCalibration, dict]:
    """Read ``betas`` and ``lcp`` from a rectified ``.mat`` written by MATLAB or
    :func:`save_rectified` (as CSPGloadExistingGeometry does)."""
    from scipy.io import loadmat

    md = loadmat(mat_path, simplify_cells=True)["metadata"]
    beta = np.asarray(md["geom"]["betas"], dtype=float).ravel()
    return beta, LensCalibration.from_matlab(md["geom"]["lcp"], model=model), md


def virtual_gcp(u: float, v: float, geom: dict, origin, z: float = 0.0) -> np.ndarray:
    """Real-world (UTM) point under image pixel (u, v) at elevation ``z``.

    Port of CSPGgetVirtualGCP: ``geom`` is a rectified image's
    ``metadata["geom"]`` and ``origin`` the site's (eastings, northings).
    """
    beta = np.asarray(geom["betas"], dtype=float).ravel()
    lcp = LensCalibration.from_matlab(geom["lcp"])
    xyz = find_xyz_6dof(np.atleast_1d(float(u)), np.atleast_1d(float(v)), z, beta, lcp)[0]
    return np.array([xyz[0] + origin[0], xyz[1] + origin[1], z])


def rectify_with_existing_geometry(I, mat_path: str, settings: RectificationSettings,
                                   tide_level: float) -> PlanImage:
    """CSPGloadExistingGeometry: reuse a previous image's geometry on a new image."""
    beta, lcp, _ = load_geometry(mat_path)
    return rectify_image(I, beta, lcp, settings.rect_xy, tide_level + settings.tidal_offset)
