"""Site workflow: site DB -> GCP solve -> plan image -> .mat round trip."""

import os
import shutil
import subprocess

import numpy as np
import pytest
from scipy.io import loadmat

from coastsnap.camera import find_uv, find_xyz_6dof, make_lcp
from coastsnap.db import read_site_db
from coastsnap.rectify import (RectificationSettings, load_geometry, matlab_colon,
                               matlab_round, rectified_paths, rectify_from_gcps,
                               rectify_with_existing_geometry, save_rectified, to_uint8)

E0, N0 = 342000.0, 6266000.0
GCPS = [("Rock", 150, -120, 1.2), ("Pole", 220, -40, 2.5), ("Bin", 300, 60, 0.8),
        ("Sign", 180, 130, 4.0), ("Groyne", 400, -200, 0.5), ("Lifeguard", 500, 10, 1.5)]


def write_site_sheet(path, site="testbeach"):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = site
    rows = [
        ("Station Data", None), ("Eastings", E0), ("Northings", N0), ("Height", 25.0),
        ("UTM Zone", "56 H"), ("Default Timezone", "AEST"),
        ("Default Timezone Offset From GMT", 10), ("Alternative Timezone", "AEDT"),
        ("Alternative Timezone Offset From GMT", 11),
        ("Xlimit left", 100), ("Xlimit right", 450), ("Ylimit lower", -200), ("Ylimit upper", 200),
        ("Resolution", 2), ("Initial Azimuth Estimate", 100), ("Initial Tilt Estimate", 75),
        ("Initial Roll Estimate", 0), ("Tidal offset", 0.3), ("Min FOV", 50), ("Max FOV", 75),
        ("Acceptable Accuracy", 10), ("Tide file", "tide.mat"), ("Transect file", "transects.mat"),
        ("Transect averaging region", "[-5 5]"), ("Characteristic beach slope", 0.1),
    ]
    for name, x, y, z in GCPS:
        rows += [("GCP name", name), ("Eastings", E0 + x), ("Northings", N0 + y), ("Elevation", z)]
    rows += [("GCP combo", "[1 2 3 4 5 6]")]
    for r in rows:
        ws.append(r)
    wb.save(path)


@pytest.fixture
def site(tmp_path):
    path = tmp_path / "CoastSnapDB.xlsx"
    write_site_sheet(path)
    return read_site_db(str(path), "testbeach")


def test_read_site_db(site):
    assert site["origin"] == {"eastings": E0, "northings": N0, "z": 25.0}
    assert site["rect"]["FOVlims"] == [50, 75]
    assert site["rect"]["initial"]["azimuth"] == 100
    assert site["sl_settings"]["transect_averaging_region"] == [-5, 5]
    assert [g["name"] for g in site["gcp"]] == [g[0] for g in GCPS]
    assert site["gcp"][2]["northings"] == N0 + 60
    assert site["gcp_combo"] == [1, 2, 3, 4, 5, 6]


def test_settings_from_site_db(site):
    s = RectificationSettings.from_site_db(site)
    assert s.rect_xy == (100, 2, 450, -200, 2, 200)
    np.testing.assert_allclose(s.beta0, [0, 0, 25, np.radians(100), np.radians(75), 0])
    np.testing.assert_allclose(s.gcp_xyz(), [g[1:] for g in GCPS])


def render(beta, lcp):
    """An oblique frame of a plane textured by its own coordinates."""
    v, u = np.mgrid[1:lcp.nv + 1, 1:lcp.nu + 1]
    w = find_xyz_6dof(u.ravel(), v.ravel(), 0.0, beta, lcp)
    img = np.stack([np.mod(w[:, 0], 256), np.mod(w[:, 1] + 1000, 256), np.full(len(w), 90.0)], -1)
    return np.clip(np.nan_to_num(img), 0, 255).astype(np.uint8).reshape(lcp.nv, lcp.nu, 3)


def test_full_workflow_and_mat_round_trip(site, tmp_path):
    s = RectificationSettings.from_site_db(site)
    nu, nv = 800, 600
    truth = make_lcp("CoastSnap", nu, nv).replace(fx=0.5 * nu / np.tan(np.radians(31)),
                                                    fy=0.5 * nu / np.tan(np.radians(31)))
    beta = np.r_[0, 0, 25, np.radians([97, 78, 0.7])]
    img = render(beta, truth)
    uv = find_uv(beta, s.gcp_xyz(), truth) + 0.3
    tide = 0.45

    res = rectify_from_gcps(img, uv, s, tide_level=tide)
    assert res.geometry.rmse < 1
    assert res.geometry.fov_deg == pytest.approx(62, abs=0.3)
    assert res.metadata["rectz"] == pytest.approx(tide + 0.3)
    assert res.plan.timex.shape == (201, 176, 3)
    assert res.plan.N.mean() > 0.5

    jpg, mat = rectified_paths(str(tmp_path / "Processed" / "2020" /
                                   "1593820800.Sat.Jul.04_00_00_00.GMT.2020.testbeach.snap.CS.jpg"))
    assert "Rectified" in jpg and jpg.endswith(".plan.CS.jpg") and mat.endswith(".plan.CS.mat")
    save_rectified(res, jpg, mat)
    assert os.path.exists(jpg)

    m = loadmat(mat, simplify_cells=True)
    np.testing.assert_array_equal(m["Iplan"], res.plan.timex)
    np.testing.assert_allclose(m["xgrid"], res.plan.x)
    geom = m["metadata"]["geom"]
    np.testing.assert_allclose(geom["betas"], res.geometry.beta)
    assert geom["lcp"]["fx"] == res.geometry.lcp.fx
    np.testing.assert_allclose(geom["knownFlags"], [1, 1, 1, 0, 0, 0])

    b, lcp, _ = load_geometry(mat)
    np.testing.assert_allclose(b, res.geometry.beta)
    assert lcp == res.geometry.lcp
    again = rectify_with_existing_geometry(img, mat, s, tide)
    np.testing.assert_array_equal(again.timex, res.plan.timex)


@pytest.mark.skipif(shutil.which("octave") is None, reason="needs GNU Octave")
def test_matlab_can_reuse_saved_geometry(site, tmp_path):
    """CSPGloadExistingGeometry's calls work on a .mat written from Python."""
    s = RectificationSettings.from_site_db(site)
    lcp0 = make_lcp("CoastSnap", 400, 300)
    beta = np.r_[0, 0, 25, np.radians([97, 78, 0.7])]
    uv = find_uv(beta, s.gcp_xyz(), lcp0.replace(fx=330.0, fy=330.0))
    res = rectify_from_gcps(render(beta, lcp0), uv, s, tide_level=0.0)
    _, mat = save_rectified(res, str(tmp_path / "x.plan.jpg"))
    rect_code = os.path.join(os.path.dirname(__file__), "..", "rectifyCode")
    script = (f"addpath('{os.path.abspath(rect_code)}'); load('{mat}');"
              "uv = findUVnDOF(metadata.geom.betas, metadata.gcps.xyzMeas, metadata.geom);"
              "save('-v7', '" + str(tmp_path / "uv.mat") + "', 'uv');")
    subprocess.run(["octave", "--no-gui", "-q", "--eval", script], check=True,
                   capture_output=True)
    uv_m = loadmat(str(tmp_path / "uv.mat"))["uv"].reshape(2, -1).T
    np.testing.assert_allclose(uv_m, find_uv(res.geometry.beta, s.gcp_xyz(), res.geometry.lcp),
                               atol=1e-6)


def test_matlab_rounding_helpers():
    np.testing.assert_array_equal(matlab_round([0.5, 1.5, 2.5, -0.5, -2.5]), [1, 2, 3, -1, -3])
    np.testing.assert_array_equal(to_uint8([np.nan, -3, 254.5, 300]), [0, 0, 255, 255])
    np.testing.assert_allclose(matlab_colon(-1, 0.1, 1)[-1], 1.0)
    assert matlab_colon(0, 2, 5).tolist() == [0, 2, 4]
