"""Wide-angle phone lenses: synthetic camera with strong barrel distortion.

The truth is a 105 degree HFOV lens with d1=-0.18, d2=0.04 and an off-centre
principal point, roughly what an ultra-wide phone camera looks like.
"""

import numpy as np
import pytest

from coastsnap.camera import (LensCalibration, distort, find_uv, find_xyz_6dof, make_lcp,
                              solve_geometry, undistort)
from coastsnap.rectify import rectify_image


def wide_lens(nu=4000, nv=3000, model="analytic"):
    f = (nu / 2) / np.tan(np.radians(52.5))
    return LensCalibration(nu, nv, nu / 2 + 0.003 * nu, nv / 2 - 0.0027 * nv, f, f,
                           d1=-0.18, d2=0.04, model=model)


BETA = np.array([0, 0, 30, np.radians(90), np.radians(75), np.radians(0.5)])
BETA0 = np.r_[0, 0, 30, np.radians([95, 72, 0])]


def synthetic_gcps(lcp, noise=0.5, seed=3):
    """GCPs spread across the lower half of the frame, out to its edges."""
    rng = np.random.default_rng(seed)
    U, V = np.meshgrid(np.linspace(0.04, 0.96, 5) * lcp.nu, np.linspace(0.5, 0.96, 4) * lcp.nv)
    z = rng.uniform(0, 5, U.size)
    xyz = np.vstack([find_xyz_6dof([u], [v], zz, BETA, lcp)
                     for u, v, zz in zip(U.ravel(), V.ravel(), z)])
    uv = find_uv(BETA, xyz, lcp) + rng.normal(0, noise, (U.size, 2))
    return xyz, uv


def test_round_trip_to_the_corners():
    lcp = wide_lens()
    u, v = np.meshgrid(np.linspace(1, lcp.nu, 41), np.linspace(1, lcp.nv, 31))
    uu, vu = undistort(u, v, lcp)
    assert np.all(np.isfinite(uu))
    ud, vd = distort(uu, vu, lcp)
    np.testing.assert_allclose(ud, u.ravel(), atol=1e-6)
    np.testing.assert_allclose(vd, v.ravel(), atol=1e-6)


def test_matlab_tables_lose_the_corners():
    corners_u, corners_v = [1, 4000, 1, 4000], [1, 1, 3000, 3000]
    uu, _ = undistort(corners_u, corners_v, wide_lens(model="table"))
    assert np.isnan(uu).any()
    uu, _ = undistort(corners_u, corners_v, wide_lens())
    assert np.isfinite(uu).all()


def test_fold_back_points_are_rejected():
    lcp = LensCalibration(1000, 800, 500, 400, 400, 400, d1=-0.3)
    assert lcp.r_max == pytest.approx(np.sqrt(1 / 0.9))
    u = 500 + 400 * np.array([0.5, 1.0, 1.2])
    ud, _ = distort(u, [400, 400, 400], lcp)
    assert np.isfinite(ud[:2]).all() and np.isnan(ud[2])


def test_points_behind_camera_are_culled():
    lcp = make_lcp("CoastSnap", 1600, 1200)
    beta = np.r_[0, 0, 20, np.radians([90, 80, 0])]
    uv = find_uv(beta, [[100, 0, 0], [-100, 0, 0]], lcp)
    assert np.isfinite(uv[0]).all() and np.isnan(uv[1]).all()
    uv = find_uv(beta, [[-100, 0, 0]], lcp, cull_behind=False)
    assert np.isfinite(uv).all()     # MATLAB behaviour: mirrored through the camera


def test_opencv_calibration_maps_directly():
    K = [[1500.0, 0, 2010], [0, 1498, 1490], [0, 0, 1]]
    lcp = LensCalibration.from_opencv(K, [-0.18, 0.04, 0.001, -0.002, 0.003], (4000, 3000))
    assert (lcp.fx, lcp.fy, lcp.c0u, lcp.c0v) == (1500, 1498, 2010, 1490)
    assert (lcp.d1, lcp.d2, lcp.t1, lcp.t2, lcp.d3) == (-0.18, 0.04, 0.001, -0.002, 0.003)
    # same equations as cv2.projectPoints for a normalised point
    x, y = 0.4, -0.3
    r2 = x * x + y * y
    k = 1 - 0.18 * r2 + 0.04 * r2**2 + 0.003 * r2**3
    xd = x * k + 2 * 0.001 * x * y + -0.002 * (r2 + 2 * x * x)
    yd = y * k + 0.001 * (r2 + 2 * y * y) + 2 * -0.002 * x * y
    ud, vd = distort([2010 + 1500 * x], [1490 + 1498 * y], lcp)
    assert ud[0] == pytest.approx(2010 + 1500 * xd)
    assert vd[0] == pytest.approx(1490 + 1498 * yd)


def test_distortion_free_profile_cannot_fit_a_wide_lens():
    xyz, uv = synthetic_gcps(wide_lens())
    g = solve_geometry(xyz, uv, make_lcp("CoastSnap", 4000, 3000), BETA0, fov_limits=(90, 125))
    assert g.rmse > 20


def test_solving_distortion_recovers_the_camera():
    truth = wide_lens()
    xyz, uv = synthetic_gcps(truth)
    g = solve_geometry(xyz, uv, make_lcp("CoastSnap", 4000, 3000), BETA0, fov_limits=(90, 125),
                       refine_focal=True, free_distortion=("c0u", "c0v", "d1", "d2"))
    assert g.rmse < 0.7                                  # the 0.5 px picking noise
    np.testing.assert_allclose(np.degrees(g.beta[3:]), np.degrees(BETA[3:]), atol=0.05)
    assert g.fov_deg == pytest.approx(105, abs=0.2)
    assert g.lcp.d1 == pytest.approx(truth.d1, abs=0.01)
    assert g.lcp.c0u == pytest.approx(truth.c0u, abs=3)


def test_known_calibration_needs_only_the_pose():
    truth = wide_lens()
    xyz, uv = synthetic_gcps(truth)
    g = solve_geometry(xyz, uv, truth, BETA0)
    assert g.rmse < 0.7
    np.testing.assert_allclose(np.degrees(g.beta[3:]), np.degrees(BETA[3:]), atol=0.02)


def _ground(x, y):
    return np.stack([127.5 + 100 * np.sin(x / 9.0) * np.cos(y / 7.0),
                     127.5 + 100 * np.cos(x / 13.0),
                     127.5 + 100 * np.sin(y / 11.0)], -1)


def test_end_to_end_rectification_of_a_wide_angle_image():
    """Render an oblique image of a known beach texture through the distorted
    lens, solve the geometry from GCPs and check the plan view reproduces it."""
    truth = wide_lens(1200, 900)
    v, u = np.mgrid[1:901, 1:1201]
    world = find_xyz_6dof(u.ravel(), v.ravel(), 0.0, BETA, truth)
    depth = (world - BETA[:3]) @ np.array([np.sin(BETA[4]) * np.sin(BETA[3]),
                                           np.sin(BETA[4]) * np.cos(BETA[3]), -np.cos(BETA[4])])
    img = _ground(world[:, 0], world[:, 1])
    img[~(depth > 0)] = 0
    img = np.clip(np.round(img), 0, 255).astype(np.uint8).reshape(900, 1200, 3)

    xyz, uv = synthetic_gcps(truth, noise=0.3)
    rect = (20, 1, 120, -60, 1, 60)
    good = solve_geometry(xyz, uv, make_lcp("CoastSnap", 1200, 900), BETA0, fov_limits=(90, 125),
                          refine_focal=True, free_distortion=("c0u", "c0v", "d1", "d2"))
    bad = solve_geometry(xyz, uv, make_lcp("CoastSnap", 1200, 900), BETA0, fov_limits=(90, 125))
    X, Y = np.meshgrid(np.arange(20, 121), np.arange(-60, 61))
    expected = _ground(X, Y)

    def error(geom):
        plan = rectify_image(img, geom.beta, geom.lcp, rect, 0.0)
        ok = plan.N > 0
        assert ok.mean() > 0.9
        return np.abs(plan.timex[ok].astype(float) - expected[ok]).mean()

    # the solved geometry is as good as the true one (about 0.7 grey levels of
    # nearest-pixel sampling error); ignoring the distortion is far worse
    assert error(good) < 1.5
    assert error(bad) > 10 * error(good)
