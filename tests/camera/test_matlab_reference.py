"""The port against outputs of the original MATLAB code (run in GNU Octave).

``model="table"`` is the MATLAB-compatible lens model and must match to
floating point; ``model="analytic"`` (the default) is checked to stay within
a small fraction of a pixel inside the region MATLAB covers.
"""

import numpy as np
import pytest

from coastsnap.camera import (P2m, angles2R, distort, find_uv, find_xyz_6dof, lcp_beta2P,
                              make_lcp, solve_geometry, undistort)
from coastsnap.camera.solve import focal_lengths_for_fov
from coastsnap.rectify import PlanProducts, rectify_image


def test_lens_tables_match(ref):
    lcp = make_lcp("Aerielle", 4000, 3000).to_matlab()
    m = ref["A"]["lcp"]
    for k in ("c0U", "c0V", "fx", "fy", "d1", "d2", "t1", "t2"):
        assert lcp[k] == pytest.approx(m[k])
    for k in ("r", "fr", "x", "y", "dx", "dy"):
        np.testing.assert_allclose(lcp[k], m[k], atol=1e-12, err_msg=k)


def test_distortion_table_mode_matches(ref):
    A = ref["A"]
    lcp = make_lcp("Aerielle", 4000, 3000, model="table")
    ud, vd = distort(A["u"], A["v"], lcp)
    np.testing.assert_allclose(ud, A["ud"], atol=1e-9)
    np.testing.assert_allclose(vd, A["vd"], atol=1e-9)
    uu, vu = undistort(A["ud"], A["vd"], lcp)
    np.testing.assert_allclose(uu, A["uu"], atol=1e-9)
    np.testing.assert_allclose(vu, A["vu"], atol=1e-9)


def test_distortion_analytic_mode_close_and_more_accurate(ref):
    A = ref["A"]
    lcp = make_lcp("Aerielle", 4000, 3000)
    ud, vd = distort(A["u"], A["v"], lcp)
    # MATLAB interpolates the polynomial from tables; the exact value differs by < 0.1 px
    assert np.max(np.abs(ud - A["ud"])) < 0.1
    assert np.max(np.abs(vd - A["vd"])) < 0.1
    # MATLAB's single-step inverse is several pixels off its own forward model;
    # the Newton inverse recovers the original points.
    uu, vu = undistort(ud, vd, lcp)
    np.testing.assert_allclose(uu, A["u"], atol=1e-6)
    np.testing.assert_allclose(vu, A["v"], atol=1e-6)
    matlab_err = np.max(np.hypot(A["uu"] - A["u"], A["vu"] - A["v"]))
    assert matlab_err > 1.0


def test_projection_matches(ref):
    B = ref["B"]
    lcp = make_lcp("CoastSnap", 1600, 1200, model="table")
    np.testing.assert_allclose(angles2R(0.3, 1.2, -0.05), B["R"], atol=1e-15)
    np.testing.assert_allclose(lcp_beta2P(lcp, B["beta"]), B["P"], rtol=1e-12)
    np.testing.assert_allclose(P2m(B["P"]), B["m"], rtol=1e-12)
    uv = find_uv(B["beta"], B["xyz"], lcp, cull_behind=False)
    np.testing.assert_array_equal(np.isnan(uv), np.isnan(B["uv"]))
    np.testing.assert_allclose(uv, B["uv"], atol=1e-8)
    xyz = find_xyz_6dof(B["uv"][:, 0], B["uv"][:, 1], 0.7, B["beta"], lcp)
    np.testing.assert_allclose(xyz, B["xyz6"], atol=1e-8)


def test_projection_with_distortion_matches(ref):
    B = ref["B"]
    lcp = make_lcp("Aerielle", 4000, 3000, model="table")
    uv = find_uv(B["beta"], B["xyz"], lcp, cull_behind=False)
    np.testing.assert_array_equal(np.isnan(uv), np.isnan(B["uvA"]))
    np.testing.assert_allclose(uv, B["uvA"], atol=1e-8)
    xyz = find_xyz_6dof(B["uvA"][:, 0], B["uvA"][:, 1], 0.7, B["beta"], lcp)
    np.testing.assert_allclose(xyz, B["xyz6A"], atol=1e-8)


def test_analytic_projection_covers_points_matlab_loses(ref):
    B = ref["B"]
    uv = find_uv(B["beta"], B["xyz"], make_lcp("Aerielle", 4000, 3000), cull_behind=False)
    lost = np.isnan(B["uvA"][:, 0])
    assert lost.any() and not np.isnan(uv).any()
    assert np.max(np.abs(uv[~lost] - B["uvA"][~lost])) < 0.1


def test_fov_sweep_and_solve_match(ref):
    C = ref["C"]
    fx = focal_lengths_for_fov(1600, C["fov_lims"])
    np.testing.assert_array_equal(fx, C["fx"])
    g = solve_geometry(C["xyz"], C["uv"], make_lcp("CoastSnap", 1600, 1200), C["beta0"],
                       fov_limits=C["fov_lims"])
    np.testing.assert_allclose(g.sweep_mse, C["mse_all"], rtol=1e-8)
    assert g.lcp.fx == C["fx_best"]
    np.testing.assert_allclose(g.beta, C["betas"], atol=1e-9)
    assert g.mse == pytest.approx(C["mse"], rel=1e-9)
    np.testing.assert_allclose(g.ci, C["ci"], atol=1e-8)


def test_refine_focal_beats_sweep(ref):
    C = ref["C"]
    lcp = make_lcp("CoastSnap", 1600, 1200)
    g = solve_geometry(C["xyz"], C["uv"], lcp, C["beta0"], fov_limits=C["fov_lims"],
                       refine_focal=True)
    true_fx = 0.5 * 1600 / np.tan(62 * np.pi / 360)
    lo, hi = g.intrinsics_ci["f"]
    assert lo < true_fx < hi
    assert g.mse < C["mse"]


@pytest.mark.parametrize("model", ["table", "analytic"])
def test_rect_products_two_frames_match(ref, model):
    D = ref["D"]
    lcp = make_lcp("CoastSnap", 320, 240, model=model)
    pp = PlanProducts(tuple(D["rectxy"]), D["z"])
    f1 = pp.add(1, D["I1"], D["beta1"], lcp, cull_behind=False).final()
    np.testing.assert_array_equal(f1.x, D["x"])
    np.testing.assert_array_equal(f1.y, D["y"])
    np.testing.assert_array_equal(f1.timex, D["timex1"])
    np.testing.assert_array_equal(f1.N, D["N1"])
    f2 = pp.add(2, D["I2"], D["beta2"], lcp, cull_behind=False).final()
    for k in ("timex", "bright", "dark"):
        np.testing.assert_array_equal(getattr(f2, k), D[k + "2"], err_msg=k)
    np.testing.assert_array_equal(f2.N, D["N2"])


def _phantom_frame():
    uu, vv = np.meshgrid(np.arange(1, 4001), np.arange(1, 3001))
    return np.stack([np.mod(uu + 2 * vv, 256), np.mod(3 * uu - vv, 256),
                     np.mod(uu * vv, 256)], -1).astype(float)


def test_rectify_with_distorted_lens_matches(ref):
    D = ref["D"]
    I = _phantom_frame()
    plan = rectify_image(I, D["betaE"], make_lcp("Aerielle", 4000, 3000, model="table"),
                         D["rectxyE"], 0)
    np.testing.assert_array_equal(plan.timex, D["timexE"])
    np.testing.assert_array_equal(plan.N, D["NE"])
    # the analytic model only moves a handful of cells to a neighbouring pixel
    plan = rectify_image(I, D["betaE"], make_lcp("Aerielle", 4000, 3000), D["rectxyE"], 0)
    np.testing.assert_array_equal(plan.N, D["NE"])
    changed = (plan.timex != D["timexE"]).any(-1).mean()
    assert changed < 0.03
