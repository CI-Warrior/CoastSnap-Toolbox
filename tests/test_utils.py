"""File naming, time, UTM, paths and shoreline shifts, checked against the
MATLAB code (tests/data/utils_reference.mat, made by
tests/matlab_reference/make_utils_reference.m)."""

import os
from datetime import datetime

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from coastsnap import timeutils as tu
from coastsnap.config import Paths, load_paths
from coastsnap.naming import (argus_filename, clean_user, epoch_of, parse_filename, plan_name,
                              rectified_dir, shoreline_name)
from coastsnap.shoreline import Transects, make_transects, shift_shoreline
from coastsnap.shoreline.transects import load_transects, save_transects
from coastsnap.tide import TideRecord, get_tide_level
from coastsnap.utm import utm2deg

from conftest import DATA


@pytest.fixture(scope="module")
def ref():
    return loadmat(os.path.join(DATA, "utils_reference.mat"), simplify_cells=True)


def test_argus_filename_matches_matlab(ref):
    N = ref["N"]
    expected = iter(N["names"])
    for epoch in np.atleast_1d(N["epochs"]):
        for name, offset in zip(N["tz_names"], np.atleast_1d(N["tz_offsets"])):
            for user in N["users"]:
                user = user if isinstance(user, str) else ""
                assert argus_filename(epoch, "manly", "snap", user, "jpg", name, offset) == next(expected)
    assert argus_filename(1528322400, "manly", "timex", "Mitch", "jpg", "PST", -8, camera=2) == N["cam2"]


def test_parse_filename_round_trip():
    name = argus_filename(1528322400, "manly", "snap", "Mitch", "jpg", "AEST", 10)
    p = parse_filename(name)
    assert p == {"epochtime": "1528322400", "dayname": "Thu", "month": "Jun", "day": "07", "hour": "08",
                 "min": "00", "sec": "00", "timezone": "AEST", "year": "2018", "site": "manly",
                 "type": "snap", "user": "Mitch", "format": "jpg"}
    assert epoch_of("/a/b/" + name) == 1528322400
    with pytest.raises(ValueError):
        parse_filename("IMG_0001.jpg")


def test_product_names():
    snap = "1528322400.Thu.Jun.07_08_00_00.AEST.2018.manly.snap.snapper.jpg"
    assert plan_name(snap) == "1528322400.Thu.Jun.07_08_00_00.AEST.2018.manly.plan.snapper.jpg"
    assert shoreline_name(snap) == "1528322400.Thu.Jun.07_08_00_00.AEST.2018.manly.shoreline.snapper.mat"
    day = snap.replace(".snap.", ".daytimex.")
    # MATLAB's strrep(name, 'timex', 'plan') gives "dayplan"; keep the same names.
    assert plan_name(day, ".mat").split(".")[7] == "dayplan"
    assert shoreline_name(plan_name(day)).split(".")[7] == "dayshoreline"
    assert rectified_dir("/x/Images/manly/Registered/2018").as_posix() == "/x/Images/manly/Rectified/2018"
    assert clean_user("Jo_Smith. Jr") == "JoSmithJr"


def test_time_conversions_match_matlab(ref):
    T = ref["T"]
    epochs = np.atleast_1d(T["epochs"])
    np.testing.assert_allclose(tu.epoch_to_datenum(epochs), T["datenums"], rtol=0, atol=1e-9)
    assert np.array_equal(tu.datenum_to_epoch(np.atleast_1d(T["datenums"])), np.atleast_1d(T["back"]).astype(int))
    assert [tu.julian_day(d) for d in np.atleast_1d(T["datenums"])] == list(np.atleast_1d(T["julian"]).astype(int))
    assert [tu.argus_day(e) for e in epochs] == list(T["argusday"])
    np.testing.assert_allclose(tu.epoch_to_local_datenum(epochs, 10), T["local"], atol=1e-9)


def test_datetime_helpers():
    d = datetime(2024, 2, 29, 13, 45, 30)
    assert tu.datenum_to_datetime(tu.datetime_to_datenum(d)) == d
    assert tu.datetime_to_datenum(datetime(1970, 1, 1)) == tu.EPOCH_DATENUM
    e = tu.datetime_to_epoch(d, 10)
    assert tu.epoch_to_datetime(e, 10) == d
    assert e == tu.datetime_to_epoch(datetime(2024, 2, 29, 3, 45, 30))


def test_utm2deg_matches_matlab(ref):
    U = ref["U"]
    for x, y, z, lat, lon in zip(U["x"], U["y"], U["zones"], U["lat"], U["lon"]):
        got = utm2deg(x, y, z)
        assert got == pytest.approx((lat, lon), abs=1e-9)


def test_shift_shoreline_matches_matlab(ref):
    S = ref["S"]
    t = Transects(S["transects"]["x"], S["transects"]["y"])
    got = shift_shoreline(S["sl"]["xyz"], t, S["shift"], z=1.2)
    np.testing.assert_allclose(got, S["new"], atol=1e-9)
    # Negative shifts move landward; the default elevation is the shoreline's own.
    on = shift_shoreline(S["sl"]["xyz"], t, 0)
    back = shift_shoreline(S["sl"]["xyz"], t, -S["shift"])
    np.testing.assert_allclose(np.hypot(*(on - back)[:, :2].T), S["shift"])
    assert (back[:, 0] < on[:, 0]).all() and (back[:, 2] == 0.4).all()


def test_shift_shoreline_any_orientation():
    # West-facing beach with vertical transects: MATLAB's line fit cannot do this.
    t = Transects([[0], [0]], [[10], [-10]])
    sl = np.array([[-5, 2, 0.5], [15, 2, 0.5]])
    np.testing.assert_allclose(shift_shoreline(sl, t, 3), [[0, -1, 0.5]])


def test_make_transects(tmp_path):
    roi = [[0, -50], [100, -50], [100, 250], [0, 250]]
    t = make_transects(roi, [[50, 0], [50, 200]], spacing=10)
    assert t.count == 20
    np.testing.assert_allclose(t.alongshore_distances, np.arange(0, 200, 10))
    np.testing.assert_allclose(t.x[0], 0, atol=0.11)      # start at -x
    np.testing.assert_allclose(t.x[1], 100, atol=0.11)
    np.testing.assert_allclose(t.y[0], t.y[1])
    flipped = make_transects(roi, [[50, 0], [50, 200]], spacing=10, flip=True)
    np.testing.assert_allclose(flipped.x[0], t.x[1])
    # Coastline running east-west: normals are vertical (MATLAB divides by 0).
    t2 = make_transects(roi, [[10, 100], [90, 100]], spacing=20)
    np.testing.assert_allclose(t2.x[0], [10, 30, 50, 70])
    assert (t2.y[0] < t2.y[1]).all()
    p = save_transects(tmp_path / "SLtransects_x.mat", t)
    back = load_transects(p)
    np.testing.assert_allclose(back.x, t.x)
    np.testing.assert_allclose(back.alongshore_distances, t.alongshore_distances)


def test_tide_interpolation(tmp_path):
    t = np.array([738000.0, 738000.5, 738001.0])
    rec = TideRecord(t, np.array([0.0, 1.0, 0.0]))
    assert rec.at_datenum(738000.25) == pytest.approx(0.5)
    assert np.isnan(rec.at_datenum(737999.0))
    savemat(tmp_path / "tide.mat", {"tide": {"time": t, "level": [0.0, 1.0, 0.0]}})
    site = {"tide": {"file": "tide.mat"}, "timezone": {"gmt_offset": 10}}
    epoch = tu.datenum_to_epoch(738000.25 - 10 / 24)
    assert get_tide_level(epoch, site, tmp_path) == pytest.approx(0.5)


def test_paths(tmp_path, monkeypatch):
    p = Paths.from_base(tmp_path, tide="elsewhere/tides")
    assert p.db_file == tmp_path / "Database" / "CoastSnapDB.xlsx"
    assert p.transects == tmp_path / "Shorelines" / "Transect Files"
    assert p.tide == tmp_path / "elsewhere" / "tides"
    assert p.site_images("manly", "Raw") == tmp_path / "Images" / "manly" / "Raw"
    monkeypatch.setenv("COASTSNAP_BASE", str(tmp_path / "env"))
    assert load_paths().base == tmp_path / "env"
    assert load_paths(tmp_path).base == tmp_path
    monkeypatch.delenv("COASTSNAP_BASE")
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[paths]\nbase = "{tmp_path.as_posix()}"\nimages = "/data/imgs"\n')
    assert load_paths(config_file=cfg).images.as_posix() == "/data/imgs"
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_paths(config_file=tmp_path / "missing.toml")
