"""End-to-end workflow on a synthetic database: list -> bulk rectify and map ->
trend, change, forecast and animation -> day timex -> ingest."""

from datetime import date, datetime, timedelta

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from scipy.io import loadmat

import synthetic_site as syn
from coastsnap import batch, cli, ingest, plots
from coastsnap.db import ImageRecord, append_image_db, read_image_db
from coastsnap.naming import parse_filename
from coastsnap.shoreline import load_plan, load_shoreline, make_record, map_shoreline, save_shoreline
from coastsnap.site import ImageEntry
from coastsnap.timex import make_day_timex


@pytest.fixture(scope="module")
def mapped(tmp_path_factory):
    """Six weekly images, all rectified and mapped (the first by hand, the rest in bulk)."""
    s = syn.build(tmp_path_factory.mktemp("db"))
    site = s.site
    plan = load_plan(site.plan_paths(s.images[0].path)[1])
    found = map_shoreline(plan.xgrid, plan.ygrid, plan.iplan, site.transects)
    rec = make_record(found.xy, plan.rectz, site.origin, "56 H", found.method, found.threshold, qa=True)
    save_shoreline(site.shoreline_path(s.images[0].name), rec)
    s.results = batch.bulk_rectify_and_map(site, batch.images_between(site, s.images[0], s.images[-1]),
                                           site.plan_paths(s.images[0].path)[1])
    return s


def test_site_listing_and_tide(mapped):
    site = mapped.site
    assert site.images() == mapped.images
    assert site.tide_level(mapped.images[2].epoch) == 0.0
    assert site.averaging_transects.tolist() == list(range(2, 11))
    assert len(site.shorelines()) == len(mapped.images)
    assert site.find_shoreline(mapped.images[3].name) == site.shoreline_path(mapped.images[3].name)


def test_bulk_rectify_and_map(mapped):
    site = mapped.site
    assert [r.status for r in mapped.results] == [batch.DONE] * 5
    for k, im in enumerate(mapped.images[1:], 1):
        jpg, mat = site.plan_paths(im.path)
        assert jpg.is_file() and jpg.with_suffix(".jpw").is_file()
        md = loadmat(mat, simplify_cells=True)["metadata"]
        assert md["rectz"] == pytest.approx(syn.TIDAL_OFFSET)
        assert np.isnan(md["gcps"]["UVpicked"])
        sl = load_shoreline(site.shoreline_path(im.name))
        assert not sl.qa
        assert np.isfinite(sl.uv).all()
        inside = np.abs(sl.xyz[:, 1]) < 70
        assert np.median(sl.xyz[inside, 0]) == pytest.approx(syn.shoreline_x(k), abs=2)
    world = np.loadtxt(site.plan_paths(mapped.images[1].path)[0].with_suffix(".jpw"))
    np.testing.assert_allclose(world, [2, 0, 0, -2, syn.E0 + 100, syn.N0 + 200])


def test_bulk_skips_done_and_wrong_size(mapped, tmp_path):
    site = mapped.site
    again = batch.bulk_rectify_and_map(site, mapped.images[1:3], site.plan_paths(mapped.images[0].path)[1])
    assert [r.status for r in again] == [batch.EXISTS] * 2
    _, name = syn.image_name(syn.FIRST + timedelta(days=1))
    odd = tmp_path / "Processed" / "2024" / name
    syn.save_image(odd, syn.render(0, size=(300, 400)))
    res = batch.bulk_rectify_and_map(site, [mapped.images[1], ImageEntry(0, odd)],
                                     site.plan_paths(mapped.images[0].path)[1])
    assert [r.status for r in res] == [batch.EXISTS, batch.WRONG_SIZE]


def test_images_between_rejects_reversed_range(mapped):
    with pytest.raises(ValueError):
        batch.images_between(mapped.site, mapped.images[3], mapped.images[1])


def test_trend_plot(mapped):
    site = mapped.site
    cur = plots.CurrentImage.load(site, mapped.images[-1])
    res = plots.trend_plot(site, cur, days=60)
    assert len(res.surveys) == 6
    np.testing.assert_allclose(res.mean_width, [syn.msl_width(k) for k in range(6)], atol=1.5)
    assert res.rate_m_per_year == pytest.approx(syn.WEEKLY_CHANGE * 365.25 / 7, rel=0.05)
    assert (res.imin, res.imax) == (0, 5)
    assert "Beach width trend +" in res.summary()
    # A shorter window only sees the last few shorelines.
    assert len(plots.trend_plot(site, cur, days=15).surveys) == 3


def test_trend_export(mapped, tmp_path):
    site = mapped.site
    res = plots.trend_plot(site, plots.CurrentImage.load(site, mapped.images[-1]), days=60)
    paths = plots.export_trend(site, res, tmp_path, dpi=50)
    assert [p.name for p in paths] == ["beachwidth_trend_figure_beach.jpg", "min_width_figure_beach.jpg",
                                       "max_width_figure_beach.jpg"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_change_plot(mapped):
    site = mapped.site
    cur = plots.CurrentImage.load(site, mapped.images[-1])
    res = plots.change_plot(site, cur, steps_back=2)
    assert res.previous.entry == mapped.images[3]
    assert res.mean_change == pytest.approx(2 * syn.WEEKLY_CHANGE, abs=1.5)
    assert len(res.change) == 9
    # Stepping back past the first shoreline uses the first one.
    assert plots.change_plot(site, cur, steps_back=99).previous.entry == mapped.images[0]


def test_forecast_plot(mapped):
    site = mapped.site
    fig = plots.forecast_plot(site, plots.CurrentImage.load(site, mapped.images[2]), -10)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[1:] == ["Forecast post-storm shoreline", "Forecast uncertainty"]


def test_beach_width_animation(mapped, tmp_path):
    site = mapped.site
    paths = plots.beach_width_animation(site, plots.CurrentImage.load(site, mapped.images[-1]), tmp_path,
                                        days=60, dpi=40)
    assert paths[0].name == "beachwidth_timeseries_beach.csv"
    rows = np.loadtxt(paths[0], delimiter=",")
    assert rows.shape == (6, 8)
    assert rows[0, :6].tolist() == [2024, 1, 1, 9, 0, 0]
    np.testing.assert_allclose(rows[:, 6], [syn.msl_width(k) for k in range(6)], atol=1.5)
    assert [p.name for p in paths[1:]] == [f"frame_{i:03d}.jpg" for i in range(1, 7)]


def test_smooth_series_matches_matlab_weighting():
    x = np.arange(20.0)
    y = 2 * x + np.sin(x)
    s = plots.smooth_series(x, y, p=1.0 - 1e-9)    # p -> 1 interpolates
    np.testing.assert_allclose(s(x), y, atol=1e-3)
    flat = plots.smooth_series(x, y, p=1e-9)        # p -> 0 is the least-squares line
    np.testing.assert_allclose(flat(x), np.polyval(np.polyfit(x, y, 1), x), atol=1e-2)


def test_day_timex(tmp_path):
    from PIL import Image

    s = syn.build(tmp_path, n_images=1, rectify_first=False)
    day = syn.FIRST.date() + timedelta(days=3)
    imgs = []
    for hour, k in ((8, 0), (12, 1), (16, 2), (20, 3)):    # 20:00 is outside the day window
        when = datetime(day.year, day.month, day.day, hour)
        _, name = syn.image_name(when, user=f"u{hour}")
        path = s.paths.site_images("beach") / "2024" / name
        syn.save_image(path, syn.render(k))
        if hour < 19:
            imgs.append(np.asarray(Image.open(path), dtype=float))
    out = make_day_timex(s.site, day, day)
    assert len(out) == 1
    parts = parse_filename(out[0].name)
    assert parts["type"] == "daytimex" and parts["user"] == "u16"
    assert (parts["day"], parts["hour"], parts["min"]) == (f"{day.day:02d}", "00", "00")
    got = np.asarray(Image.open(out[0]), dtype=float)
    assert np.abs(got - np.mean(imgs, axis=0)).mean() < 2    # JPEG round trip
    assert make_day_timex(s.site, day + timedelta(days=1), day + timedelta(days=9)) == []


# ---- ingest --------------------------------------------------------------------


def _raw(s, name, k=0, exif=None):
    from PIL import Image

    path = s.paths.site_images("beach", "Raw") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(syn.render(k, size=(60, 80)))
    if exif is not None:
        im.save(path, exif=exif)
    else:
        im.save(path)
    return path


def test_raw_to_processed(tmp_path):
    s = syn.build(tmp_path, n_images=0, rectify_first=False)
    _raw(s, "IMG_0001.jpg")
    _raw(s, "IMG_0002.jpg")
    append_image_db(s.paths.db_file, [
        ImageRecord("beach", "Jo_Smith.", datetime(2024, 3, 2, 7, 15), "AEDT", "IMG_0001.jpg", "Email", "Snap", 1),
        ImageRecord("beach", "Al", datetime(2024, 12, 31, 23, 30), "AEST", "IMG_0002.jpg", "Facebook", "Timex", 2),
    ])
    moves = ingest.raw_to_processed(s.site)
    names = sorted(m.target.relative_to(s.paths.images / "beach").as_posix() for m in moves)
    # 07:15 AEDT is 06:15 AEST; names are written in the default zone.
    assert names == [
        "Processed/2024/1709324100.Sat.Mar.02_06_15_00.AEST.2024.beach.snap.JoSmith.jpg",
        "Processed/2024/1735651800.Tue.Dec.31_23_30_00.AEST.2024.beach.timex.Al.jpg",
    ]
    assert all(m.target.is_file() and not m.source.exists() for m in moves)


def test_raw_to_processed_moves_nothing_on_error(tmp_path):
    s = syn.build(tmp_path, n_images=0, rectify_first=False)
    ok = _raw(s, "IMG_0001.jpg")
    _raw(s, "IMG_9999.jpg")
    append_image_db(s.paths.db_file, [ImageRecord("beach", "Jo", datetime(2024, 3, 2, 7, 15), "AEST",
                                                  "IMG_0001.jpg")])
    with pytest.raises(ValueError, match="IMG_9999.jpg: no DB row"):
        ingest.raw_to_processed(s.site)
    assert ok.exists()


def test_raw_to_processed_instagram(tmp_path):
    s = syn.build(tmp_path, n_images=0, rectify_first=False)
    name = "2024-05-06 10.20.33 " + "x" * 19 + "_abc.jpg"   # 39-character first token
    assert ingest.is_stogram(name)
    _raw(s, name)

    def ask(path, upload, site_db):
        assert upload == datetime(2024, 5, 6, 10, 20)
        return ImageRecord("beach", "insta.user", upload, "AEST", path.name, "Instagram", "Snap", 2)

    moves = ingest.raw_to_processed(s.site, ask=ask)
    assert moves[0].target.name.endswith(".AEST.2024.beach.snap.instauser.jpg")
    assert [r.user for r in read_image_db(s.paths.db_file)] == ["insta.user"]
    with pytest.raises(ValueError, match="Instagram"):
        _raw(s, name)
        ingest.raw_to_processed(s.site)


def test_raw_to_processed_no_db(tmp_path):
    from PIL import Image

    s = syn.build(tmp_path, n_images=0, rectify_first=False)
    exif = Image.Exif()
    exif[306] = "2024:02:03 04:05:06"
    _raw(s, "camera.jpg", exif=exif)
    _raw(s, "202402031530_spot.jpg")
    moves = ingest.raw_to_processed_no_db(s.site, "cam_1", "timex", "AEDT", dry_run=True)
    assert sorted(m.target.name for m in moves) == [
        "1706893506.Sat.Feb.03_03_05_06.AEST.2024.beach.timex.cam1.jpg",
        "1706934600.Sat.Feb.03_14_30_00.AEST.2024.beach.timex.cam1.jpg",
    ]
    assert all(m.source.exists() for m in moves)    # dry run


def test_sort_shared_images(tmp_path):
    from PIL import Image

    s = syn.build(tmp_path, n_images=0, rectify_first=False)
    lat, lon = ingest.site_locations([s.site])["beach"]
    shared = tmp_path / "shared"
    shared.mkdir()

    def dms(v):
        v = abs(v)
        d = int(v)
        m = int((v - d) * 60)
        return (float(d), float(m), round(((v - d) * 60 - m) * 60, 4))

    exif = Image.Exif()
    exif[306] = "2024:02:03 04:05:06"
    exif[0x8825] = {1: "S" if lat < 0 else "N", 2: dms(lat), 3: "E", 4: dms(lon)}
    Image.fromarray(syn.render(0, size=(60, 80))).save(shared / "near.jpeg", exif=exif)
    far = Image.Exif()
    far[306] = "2024:02:03 04:05:06"
    Image.fromarray(syn.render(0, size=(60, 80))).save(shared / "nogps.jpg", exif=far)

    assert ingest.exif_latlon(shared / "near.jpeg") == pytest.approx((lat, lon), abs=1e-5)
    moves = ingest.sort_shared_images(shared, [s.site], "Sam", "AEST")
    targets = {m.source.name: m.target.relative_to(s.paths.images).as_posix() for m in moves}
    assert targets == {"near.jpeg": "beach/Raw/near.jpg", "nogps.jpg": "unclassified/nogps.jpg"}
    rows = read_image_db(s.paths.db_file)
    assert [(r.site, r.filename, r.time_quality) for r in rows] == [
        ("beach", "near.jpg", 1), ("unclassified", "nogps.jpg", 1)]


def test_participation():
    db = {"timezone": {"name": "AEST", "gmt_offset": 10, "alternative": {"name": "AEDT", "gmt_offset": 11}}}
    recs = [
        ImageRecord("b", "u", datetime(2024, 1, 1, 0, 30), "AEDT", "a", "Email"),   # 31 Dec AEST
        ImageRecord("b", "u", datetime(2024, 1, 2, 9), "AEST", "b", "Instagram"),
        ImageRecord("b", "u", datetime(2024, 1, 2, 17), "AEST", "c", "Instagram"),
    ]
    st = plots.participation(recs, db, date(2023, 12, 31), date(2024, 1, 3), [datetime(2024, 1, 3, 8)])
    assert st.daily.tolist() == [1, 0, 2, 1]
    assert st.cumulative[-1] == 4
    assert st.by_type == {"Email": 1, "Facebook": 0, "Twitter": 0, "Instagram": 2, "App": 1}
    assert "Total number of submissions is 4" in st.summary()
    plots.participation_plot(st, date(2023, 12, 31), date(2024, 1, 3))


# ---- command line ----------------------------------------------------------------


def test_cli_figures(mapped, tmp_path, capsys):
    base = str(mapped.paths.base)
    img = mapped.images[-1].name
    assert cli.main(["--base", base, "trend", "beach", img, "--days", "60", "--out", str(tmp_path),
                     "--dpi", "40", "--no-extremes"]) == 0
    assert "Beach width trend +" in capsys.readouterr().out
    assert cli.main(["--base", base, "change", "beach", img, "--out", str(tmp_path), "--dpi", "40"]) == 0
    assert (tmp_path / "shoreline_change_figure_beach.jpg").is_file()
    with pytest.raises(SystemExit, match="cannot find image"):
        cli.main(["--base", base, "trend", "beach", "nope.jpg"])


def test_cli_bulk(tmp_path, capsys):
    s = syn.build(tmp_path, n_images=3)
    assert cli.main(["--base", str(tmp_path), "bulk", "beach", "--from", s.images[0].name,
                     "--to", s.images[-1].name]) == 0
    out = capsys.readouterr().out
    assert out.count(": done") == 2


def test_accuracy_limit(mapped):
    from coastsnap.rectify import AccuracyError, check_accuracy
    from coastsnap.camera.solve import PoseFit

    fit = PoseFit(beta=np.zeros(6), ci=np.zeros((2, 6)), mse=36.0, residuals=np.zeros((1, 2)), lcp=None)
    check_accuracy(fit, None)
    check_accuracy(fit, 6)
    with pytest.raises(AccuracyError, match="RMSE 6.0 px is above the site limit of 5 px"):
        check_accuracy(fit, mapped.site.db["rect"]["accuracylim"])


def test_reviewed_shorelines_get_a_csv(mapped):
    site = mapped.site
    first = site.shoreline_path(mapped.images[0].name)       # saved with qa=True
    assert first.with_suffix(".csv").is_file()
    assert not site.shoreline_path(mapped.images[1].name).with_suffix(".csv").exists()
    lines = first.with_suffix(".csv").read_bytes().split(b"\r\n")
    assert lines[0] == b"Eastings,Northings,Elevation"
    e, n, z = (float(v) for v in lines[1].split(b","))
    assert e > syn.E0 + 100 and z == 0.3


def test_virtual_gcp(mapped):
    from coastsnap.camera import find_uv
    from coastsnap.rectify import virtual_gcp

    md = loadmat(mapped.site.plan_paths(mapped.images[0].path)[1], simplify_cells=True)["metadata"]
    u, v = find_uv(syn.BETA, np.array([[200.0, 10.0, 0.0]]), syn.truth_lcp())[0]
    np.testing.assert_allclose(virtual_gcp(u, v, md["geom"], mapped.site.origin),
                               [syn.E0 + 200, syn.N0 + 10, 0], atol=0.5)


def test_tag_registered(mapped, tmp_path):
    from coastsnap.tag import caption, tag_registered

    im = mapped.images[1]
    assert caption(mapped.site, im.name) == ("Date: 2024/01/08 Time: 09:00 Contributor: Tester "
                                             "Tide level: 0.00m AHD")
    syn.save_image(tmp_path / ("_0001_" + im.name), syn.render(1, size=(60, 80)))
    out = tag_registered(mapped.site, tmp_path)
    assert [p.name for p in out] == [im.name.replace(".jpg", "_registered.jpg")]
