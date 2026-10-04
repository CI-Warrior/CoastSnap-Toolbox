from datetime import datetime, timedelta
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from coastsnap.shoreline import (
    PlanImage,
    ShorelineEditor,
    Transects,
    beach_width,
    beach_width_trend,
    load_shoreline,
    load_transects,
    make_record,
    map_shoreline_ccd,
    map_shoreline_hue,
    save_shoreline,
    transect_chainage,
)
from coastsnap.shoreline import cli
from coastsnap.shoreline.detect import auto_threshold, otsu_threshold
from coastsnap.shoreline.io import list_shorelines, parse_filename, shoreline_filename, shoreline_path_for
from coastsnap.shoreline.review import ACCEPTED, REJECTED, ShorelineReviewer

SAND = (200, 180, 120)
WATER = (50, 90, 140)
FNAME = "1528322400.Thu.Jun.07_08_00_00.AEST.2018.manly.snap.Mitch.jpg"


def true_x(y):
    return 40 + 0.1 * y


def synthetic_plan(seed=0, sand_east=False, black_from=None):
    """Sand west of x = 40 + 0.1 y, water east (or mirrored), with noise."""
    rng = np.random.default_rng(seed)
    xgrid = np.arange(0, 100.01, 0.5)
    ygrid = np.arange(0, 200.01, 0.5)
    X, Y = np.meshgrid(xgrid, ygrid)
    sand = X > true_x(Y) if sand_east else X < true_x(Y)
    img = np.where(sand[..., None], SAND, WATER).astype(float)
    img += rng.normal(0, 8, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if black_from is not None:
        img[Y > black_from] = 0
    return xgrid, ygrid, img


def transects(sand_east=False, spacing=10):
    ys = np.arange(5, 196, spacing, dtype=float)
    start, end = (95.0, 5.0) if sand_east else (5.0, 95.0)
    return Transects(
        x=np.vstack([np.full_like(ys, start), np.full_like(ys, end)]),
        y=np.vstack([ys, ys]),
        alongshore_distances=ys,
    )


# -- detection -------------------------------------------------------------


def test_ccd_finds_shoreline():
    x, y, img = synthetic_plan()
    t = transects()
    res = map_shoreline_ccd(x, y, img, t)
    assert res.method == "CCD"
    assert res.found.all()
    np.testing.assert_allclose(res.x, true_x(res.y), atol=0.75)
    np.testing.assert_allclose(res.y, t.y[0], atol=1.0)
    # Weighted 1/3 water + 2/3 sand peak in red-minus-blue units.
    expected = (WATER[0] - WATER[2]) / 3 + 2 * (SAND[0] - SAND[2]) / 3
    assert res.threshold == pytest.approx(expected, abs=8)


def test_ccd_transects_pointing_west():
    x, y, img = synthetic_plan(sand_east=True)
    res = map_shoreline_ccd(x, y, img, transects(sand_east=True))
    assert res.found.all()
    np.testing.assert_allclose(res.x, true_x(res.y), atol=0.75)


def test_ccd_descending_ygrid():
    x, y, img = synthetic_plan()
    res = map_shoreline_ccd(x, y[::-1], img[::-1], transects())
    assert res.found.all()
    np.testing.assert_allclose(res.x, true_x(res.y), atol=0.75)


def test_ccd_drops_points_next_to_black_pixels():
    # Image is black (no data) north of y = 150; points there must go.
    x, y, img = synthetic_plan(black_from=150)
    res = map_shoreline_ccd(x, y, img, transects())
    assert not np.isnan(res.x[transects().y[0] < 140]).any()
    assert np.isnan(res.x[transects().y[0] > 152]).all()


def test_ccd_greyscale_returns_empty():
    x, y, img = synthetic_plan()
    grey = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2).astype(np.uint8)
    res = map_shoreline_ccd(x, y, grey, transects())
    assert res.is_empty and res.threshold is None


def test_manual_threshold_is_used():
    x, y, img = synthetic_plan()
    res = map_shoreline_ccd(x, y, img, transects(), threshold=0.0)
    assert res.threshold == 0.0 and res.threshold_info is None
    assert res.found.all()


def test_hue_finds_shoreline():
    x, y, img = synthetic_plan()
    res = map_shoreline_hue(x, y, img, transects())
    assert res.method == "HUE"
    assert 0 < res.threshold < 1
    assert res.found.all()
    np.testing.assert_allclose(res.x, true_x(res.y), atol=0.75)


def test_otsu_and_auto_threshold_split_bimodal():
    rng = np.random.default_rng(1)
    s = np.concatenate([rng.normal(-90, 10, 4000), rng.normal(80, 10, 6000)])
    assert -60 < otsu_threshold(s) < 50
    info = auto_threshold(s)
    assert info.peak_locs[0] == pytest.approx(-90, abs=6)
    assert info.peak_locs[1] == pytest.approx(80, abs=6)


def test_shape_mismatch_raises():
    x, y, img = synthetic_plan()
    with pytest.raises(ValueError):
        map_shoreline_ccd(x, y, img[:, :-1], transects())


# -- transects / analysis --------------------------------------------------


def test_chainage_and_tidal_correction():
    t = transects()
    ys = np.linspace(0, 200, 50)
    sl = np.column_stack([true_x(ys), ys, np.full_like(ys, 0.5)])
    chain = transect_chainage(sl[:, :2], t)
    np.testing.assert_allclose(chain, true_x(t.y[0]) - 5, atol=1e-9)
    # Shoreline at z = 0.5 corrected to z = 0 on a 1:10 slope moves 5 m seaward.
    bw = beach_width(sl, t, reference_level=0.0, beach_slope=0.1)
    np.testing.assert_allclose(bw, chain + 5)


def test_chainage_missing_and_extended():
    t = transects()
    sl = np.array([[99.0, 0.0], [99.0, 200.0]])  # beyond transect ends at x = 95
    assert np.isnan(transect_chainage(sl, t)).all()
    np.testing.assert_allclose(transect_chainage(sl, t, extend=1.5), 94.0)


def test_trend_metres_per_week():
    d0 = datetime(2024, 1, 1)
    dates = [d0 + timedelta(days=7 * i) for i in range(5)]
    rate, _ = beach_width_trend(dates, [50, 52, 54, 56, 58])
    assert rate == pytest.approx(2.0)


def test_load_transects(tmp_path):
    t = transects()
    p = tmp_path / "SLtransects.mat"
    savemat(p, {"SLtransects": {"x": t.x, "y": t.y, "alongshore_distances": t.alongshore_distances}})
    loaded = load_transects(p)
    np.testing.assert_array_equal(loaded.x, t.x)
    assert loaded.count == t.count


# -- records and files -----------------------------------------------------


def test_record_round_trip(tmp_path):
    xy = np.array([[1.0, 2.0], [3.0, 4.0]])
    rec = make_record(xy, 0.7, (342000.0, 6265000.0), "56H", "CCD", 23.0,
                      project_uv=lambda xyz: xyz[:, :2] * 10)
    np.testing.assert_allclose(rec.utm[:, :2], xy + (342000, 6265000))
    np.testing.assert_allclose(rec.xyz[:, 2], 0.7)
    np.testing.assert_allclose(rec.uv, xy * 10)
    assert rec.qa is False
    p = save_shoreline(tmp_path / "a" / "x.mat", rec)
    raw = loadmat(p, simplify_cells=True)["sl"]
    assert set(raw) >= {"xyz", "UTM", "UTMzone", "UV", "method", "threshold", "QA", "whenDone"}
    back = load_shoreline(p)
    np.testing.assert_allclose(back.xyz, rec.xyz)
    assert (back.method, back.threshold, back.qa, back.utm_zone) == ("CCD", 23.0, False, "56H")


def test_filenames(tmp_path):
    assert parse_filename(FNAME)["site"] == "manly"
    assert shoreline_filename(FNAME) == FNAME.replace(".snap.", ".shoreline.").replace(".jpg", ".mat")
    plan_mat = FNAME.replace(".snap.", ".plan.").replace(".jpg", ".mat")
    assert shoreline_filename(plan_mat) == shoreline_filename(FNAME)
    p = shoreline_path_for(tmp_path, FNAME)
    assert p.parent == tmp_path / "manly" / "2018"
    save_shoreline(p, make_record([[0, 0]], 0, (0, 0), "56H", "CCD", 1.0))
    assert list_shorelines(tmp_path, "manly") == [(1528322400, p)]


# -- human-in-the-loop -----------------------------------------------------


@pytest.fixture
def editor():
    x, y, img = synthetic_plan()
    plan = PlanImage(x, y, img, {"rectz": 0.4})
    return ShorelineEditor(plan, transects())


def test_editor_operations(editor):
    n = len(editor.points)
    editor.move(0, (10.0, 5.0))
    assert tuple(editor.points[0]) == (10.0, 5.0)
    editor.delete(1)
    assert len(editor.points) == n - 1
    i = editor.insert((41.5, 15.0))  # between the y=5 and y=25 points
    assert i == 1 and len(editor.points) == n
    north = int((editor.points[:, 1] > 100).sum())
    assert editor.crop([(0, 100), (100, 100), (100, 200), (0, 200)]) == north
    assert (editor.points[:, 1] < 100).all()
    for _ in range(4):
        assert editor.undo()
    assert len(editor.points) == n and not editor.undo()


def test_editor_rethreshold_and_method(editor):
    editor.rethreshold(-200.0)  # below everything: no contour
    assert len(editor.points) == 0
    editor.undo()
    assert len(editor.points) == transects().count
    editor.set_method("HUE")
    assert editor.method == "HUE" and 0 < editor.threshold < 1


def test_editor_record_qa_follows_decision(editor):
    assert editor.to_record((0, 0), "56H").qa is False
    editor.accept()
    rec = editor.to_record((0, 0), "56H")
    assert rec.qa is True and rec.method == "CCD"
    np.testing.assert_allclose(rec.xyz[:, 2], 0.4)


def _key(rv, key, ax=None, xy=(None, None)):
    rv._on_key(SimpleNamespace(key=key, inaxes=ax, xdata=xy[0], ydata=xy[1]))


def test_reviewer_window_drives_editor(editor):
    rv = ShorelineReviewer(editor, oblique_image=np.zeros((50, 80, 3)),
                           project_uv=lambda xyz: xyz[:, :2])
    n = len(editor.points)
    _key(rv, "i", rv.ax_plan, (41.5, 15.0))
    assert len(editor.points) == n + 1
    _key(rv, "u")
    assert len(editor.points) == n
    # Click on the histogram sets the threshold.
    rv._on_press(SimpleNamespace(inaxes=rv.ax_hist, button=1, xdata=10.0, ydata=0.0))
    assert editor.threshold == 10.0
    # Right-click deletes the nearest point.
    p = editor.points[3]
    rv._on_press(SimpleNamespace(inaxes=rv.ax_plan, button=3, xdata=p[0], ydata=p[1]))
    assert len(editor.points) == n - 1
    rv._on_crop([(0, 100), (100, 100), (100, 200), (0, 200)])
    assert (editor.points[:, 1] < 100).all()
    np.testing.assert_allclose(rv.obl_line.get_xydata(), editor.points)
    _key(rv, "a")
    assert editor.decision == ACCEPTED


def test_cli_map_then_review_marks_qa(tmp_path, monkeypatch):
    x, y, img = synthetic_plan()
    t = transects()
    plan_name = FNAME.replace(".snap.", ".plan.").replace(".jpg", ".mat")
    rect = tmp_path / "Rectified" / "manly" / "2018"
    rect.mkdir(parents=True)
    savemat(rect / plan_name, {"xgrid": x, "ygrid": y, "Iplan": img, "metadata": {"rectz": 0.4}})
    tfile = tmp_path / "SLtransects.mat"
    savemat(tfile, {"SLtransects": {"x": t.x, "y": t.y}})
    common = ["--transects", str(tfile), "--shoreline-root", str(tmp_path / "Shorelines"),
              "--origin", "342000", "6265000", "--utm-zone", "56H"]

    assert cli.main(["map", str(rect / plan_name), *common]) == 0
    out = shoreline_path_for(tmp_path / "Shorelines", plan_name)
    rec = load_shoreline(out)
    assert rec.qa is False and len(rec.xyz) == t.count

    # Reviewer: delete a point then accept.
    def fake_run(self):
        self.editor.delete(0)
        self.editor.accept()
        return self.editor.decision

    monkeypatch.setattr(ShorelineReviewer, "run", fake_run)
    review = ["review", "--site", "manly", "--rectified-root", str(tmp_path / "Rectified"), *common]
    assert cli.main(review) == 0
    rec = load_shoreline(out)
    assert rec.qa is True and len(rec.xyz) == t.count - 1

    # Nothing left to review; a rejected item would be left unchanged.
    monkeypatch.setattr(ShorelineReviewer, "run", lambda self: (self.editor.reject(), REJECTED)[1])
    assert cli.main([*review, "--all"]) == 0
    assert load_shoreline(out).qa is True


def test_cli_uses_camera_projector_when_available(tmp_path, monkeypatch):
    import sys
    import types

    fake = types.ModuleType("coastsnap.camera")
    fake.make_uv_projector = lambda geom: (lambda xyz: xyz[:, :2] * geom["scale"])
    monkeypatch.setitem(sys.modules, "coastsnap.camera", fake)

    x, y, img = synthetic_plan()
    t = transects()
    plan_name = FNAME.replace(".snap.", ".plan.").replace(".jpg", ".mat")
    savemat(tmp_path / plan_name, {"xgrid": x, "ygrid": y, "Iplan": img,
                                   "metadata": {"rectz": 0.4, "geom": {"scale": 2.0}}})
    tfile = tmp_path / "SLtransects.mat"
    savemat(tfile, {"SLtransects": {"x": t.x, "y": t.y}})
    cli.main(["map", str(tmp_path / plan_name), "--transects", str(tfile),
              "--shoreline-root", str(tmp_path / "S"), "--origin", "0", "0", "--utm-zone", "56H"])
    rec = load_shoreline(shoreline_path_for(tmp_path / "S", plan_name))
    np.testing.assert_allclose(rec.uv, rec.xyz[:, :2] * 2)
