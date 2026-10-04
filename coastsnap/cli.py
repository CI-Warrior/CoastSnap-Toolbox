"""The ``coastsnap`` command.

The database folder is found as described in :mod:`coastsnap.config`
(``--base``, ``COASTSNAP_BASE`` or ``coastsnap.toml``). Images can be given
as a path or as a file name in the site's Processed (or Registered) folder.

    coastsnap ingest manly                       # Raw -> Processed via CoastSnapDB.xlsx
    coastsnap bulk manly --from IMG --to IMG     # rectify + map with IMG's geometry
    coastsnap shoreline review --site manly ...  # check unreviewed shorelines
    coastsnap trend manly IMG --days 42 --out figs
    coastsnap change manly IMG --back 1
    coastsnap forecast manly IMG -10
    coastsnap animate manly IMG --out frames
    coastsnap timex manly 2024-01-01 2024-01-31
    coastsnap transects manly IMG SLtransects_manly
    coastsnap participation manly 2024-01-01 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from .site import ImageEntry, Site


def _date(s: str) -> date:
    return date.fromisoformat(s)


def find_image(site: Site, image: str, kinds=("Processed", "Registered")) -> ImageEntry:
    p = Path(image)
    if p.is_file():
        return ImageEntry(int(p.name.split(".")[0]), p)
    for kind in kinds:
        for e in site.images(kind):
            if e.name == p.name:
                return e
    raise SystemExit(f"cannot find image {image!r} for site {site.name}")


def _show_or_save(fig, out: Path | None, name: str, dpi: int):
    import matplotlib.pyplot as plt

    if out is None:
        plt.show()
    else:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / name, dpi=dpi)
        print(f"saved {out / name}")


# ---- commands ----------------------------------------------------------------


def cmd_bulk(site, args):
    from .batch import bulk_rectify_and_map, images_between

    start = find_image(site, args.start)
    end = find_image(site, args.end)
    kind = "Registered" if "Registered" in start.path.parts else "Processed"
    geometry = Path(args.geometry) if args.geometry else site.plan_paths(start.path)[1]
    if not geometry.is_file():
        raise SystemExit(f"{start.name} has not been rectified (no {geometry}); rectify it first")
    images = images_between(site, start, end, kind)

    def report(n, total, res):
        extra = f", {res.points} shoreline points" if res.points else ""
        print(f"[{n}/{total}] {res.image.name}: {res.status}{extra}")

    bulk_rectify_and_map(site, images, geometry, method=args.method, overwrite=args.overwrite,
                         world_file=not args.no_world_file, on_image=report)


def cmd_trend(site, args):
    from .plots import CurrentImage, export_trend, trend_plot

    res = trend_plot(site, CurrentImage.load(site, find_image(site, args.image)), days=args.days)
    print(res.summary())
    if args.out:
        for p in export_trend(site, res, args.out, dpi=args.dpi, extremes=not args.no_extremes):
            print(f"saved {p}")
    else:
        _show_or_save(res.figure, None, "", args.dpi)


def cmd_change(site, args):
    from .plots import CurrentImage, change_plot

    res = change_plot(site, CurrentImage.load(site, find_image(site, args.image)), steps_back=args.back)
    print(f"Average beach width change {res.mean_change:+.1f} m since {res.previous.local_time:%d/%m/%Y}")
    _show_or_save(res.figure, args.out, f"shoreline_change_figure_{site.name}.jpg", args.dpi)


def cmd_forecast(site, args):
    from .plots import CurrentImage, forecast_plot

    fig = forecast_plot(site, CurrentImage.load(site, find_image(site, args.image)), args.change, args.error)
    _show_or_save(fig, args.out, f"shoreline_forecast_figure_{site.name}.jpg", args.dpi)


def cmd_animate(site, args):
    from .plots import CurrentImage, beach_width_animation

    paths = beach_width_animation(site, CurrentImage.load(site, find_image(site, args.image)), args.out,
                                  days=args.days, dpi=args.dpi)
    print(f"saved {paths[0]} and {len(paths) - 1} frames in {args.out}")


def cmd_timex(site, args):
    from .timex import make_day_timex

    out = make_day_timex(site, args.first, args.last, kind=args.kind,
                         on_day=lambda d, n, p: print(f"{d}: {n} images -> {p.name}"))
    print(f"{len(out)} day timex image(s) written")


def cmd_transects(site, args):
    import matplotlib.pyplot as plt

    from .interactive import draw_transect_inputs
    from .shoreline import load_plan, make_transects, save_transects

    image = find_image(site, args.image)
    _, mat = site.plan_paths(image.path)
    if not mat.is_file():
        raise SystemExit(f"{image.name} has not been rectified; rectify it first to make transects")
    plan = load_plan(mat)
    fig, ax, roi, coastline = draw_transect_inputs(plan.xgrid, plan.ygrid, plan.iplan)
    t = make_transects(roi, coastline, spacing=args.spacing)
    ax.plot(t.x, t.y, "r", lw=0.8)
    ax.plot(t.x[0], t.y[0], "ro", ms=3)
    ax.plot(t.x[1], t.y[1], "bo", ms=3)
    ax.set_title("Blue circles should be the seaward ends")
    plt.show(block=False)
    answer = input(f"{t.count} transects. Save? [y]es / [n]o / [f]lip and save: ").strip().lower()[:1]
    if answer not in ("y", "f"):
        print("not saved")
        return
    if answer == "f":
        t = make_transects(roi, coastline, spacing=args.spacing, flip=True)
    site.paths.transects.mkdir(parents=True, exist_ok=True)
    path = save_transects(site.paths.transects / f"{args.name}.mat", t)
    print(f"saved {path} ({t.count} transects); set the site's \"Transect file\" in CoastSnapDB.xlsx")


def _ask_stogram(path, upload, site_db):
    from datetime import datetime

    from .db import ImageRecord

    tz = site_db["timezone"]
    print(f"\n{path.name} (uploaded {upload:%d/%m/%Y %H:%M})")
    user = input("  Instagram username: ").strip()
    zone = input(f"  time zone [{tz['name']}] or {tz['alternative']['name']}: ").strip() or tz["name"]
    q = input("  time accuracy 1 = stated, 2 = good upload time, 3 = poor upload time [2]: ").strip() or "2"
    t = upload
    if q != "2":
        s = input(f"  time (dd/mm/yyyy HH:MM) [{upload:%d/%m/%Y %H:%M}]: ").strip()
        t = datetime.strptime(s, "%d/%m/%Y %H:%M") if s else upload
    return ImageRecord(path.parent.parent.name, user, t, zone, path.name, "Instagram", "Snap", int(q))


def _report_moves(moves, dry_run):
    for m in moves:
        print(("would move " if dry_run else "moved ") + str(m))
    print(f"{len(moves)} image(s)")


def cmd_ingest(site, args):
    from .ingest import raw_to_processed, raw_to_processed_no_db

    if args.user:
        moves = raw_to_processed_no_db(site, args.user, args.type, args.timezone, dry_run=args.dry_run)
    else:
        moves = raw_to_processed(site, ask=None if args.no_prompt else _ask_stogram, dry_run=args.dry_run)
    _report_moves(moves, args.dry_run)


def cmd_sort_shared(paths, args):
    from .ingest import sort_shared_images

    sites = [Site(name, paths) for name in args.sites]
    _report_moves(sort_shared_images(args.folder, sites, args.user, args.timezone, args.source,
                                     dry_run=args.dry_run), args.dry_run)


def cmd_participation(site, args):
    from .db import read_image_db
    from .plots import participation, participation_plot, read_spotteron_times

    app = []
    if args.spotteron:
        app = read_spotteron_times(args.spotteron, args.root_id, args.root_col, args.time_col)
    stats = participation(read_image_db(site.paths.db_file, site.name), site.db, args.first, args.last, app)
    print(stats.summary())
    _show_or_save(participation_plot(stats, args.first, args.last), args.out,
                  f"participation_{site.name}.jpg", args.dpi)


# ---- parser ------------------------------------------------------------------


def build_parser():
    ap = argparse.ArgumentParser(prog="coastsnap", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="CoastSnap database folder (default: COASTSNAP_BASE or coastsnap.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def site_cmd(name, func, help):
        p = sub.add_parser(name, help=help)
        p.add_argument("site")
        p.set_defaults(func=func, needs_site=True)
        return p

    def figure_opts(p, out_help="save the figure here instead of showing it"):
        p.add_argument("--out", type=Path, help=out_help)
        p.add_argument("--dpi", type=int, default=300)

    p = site_cmd("bulk", cmd_bulk, "rectify and map a run of images with one image's geometry")
    p.add_argument("--from", dest="start", required=True, help="rectified image to start from (not redone)")
    p.add_argument("--to", dest="end", required=True, help="last image to rectify and map")
    p.add_argument("--geometry", help="rectified .mat to take the geometry from (default: the --from image's)")
    p.add_argument("--method", default="CCD", choices=["CCD", "HUE"])
    p.add_argument("--overwrite", action="store_true", help="redo images that are already rectified")
    p.add_argument("--no-world-file", action="store_true", help="do not write .jpw world files")

    p = site_cmd("trend", cmd_trend, "beach-width trend up to an image")
    p.add_argument("image")
    p.add_argument("--days", type=float, default=42, help="length of the trend window (default 42)")
    p.add_argument("--no-extremes", action="store_true", help="with --out, skip the min/max width images")
    figure_opts(p, "save the trend and min/max figures in this folder")

    p = site_cmd("change", cmd_change, "beach-width change since an earlier shoreline")
    p.add_argument("image")
    p.add_argument("--back", type=int, default=1, help="how many shorelines to step back (default 1)")
    figure_opts(p)

    p = site_cmd("forecast", cmd_forecast, "draw a forecast shoreline on an image")
    p.add_argument("image")
    p.add_argument("change", type=float, help="forecast beach-width change in metres (negative = erosion)")
    p.add_argument("--error", type=float, default=6.31, help="forecast uncertainty in metres (default 6.31)")
    figure_opts(p)

    p = site_cmd("animate", cmd_animate, "beach-width animation frames and time-series CSV")
    p.add_argument("image", help="last image of the animation")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--days", type=float, default=42)
    p.add_argument("--dpi", type=int, default=150)

    p = site_cmd("timex", cmd_timex, "make daily time-exposure images")
    p.add_argument("first", type=_date, help="first day, YYYY-MM-DD")
    p.add_argument("last", type=_date, help="last day, YYYY-MM-DD")
    p.add_argument("--kind", default="Processed", choices=["Processed", "Registered"])

    p = site_cmd("transects", cmd_transects, "draw a transect file on a rectified image")
    p.add_argument("image")
    p.add_argument("name", help='file name without .mat, by convention "SLtransects_<site>"')
    p.add_argument("--spacing", type=float, default=5.0, help="transect spacing in metres (default 5)")

    p = site_cmd("ingest", cmd_ingest, "rename Raw images and move them to Processed")
    p.add_argument("--dry-run", action="store_true", help="only list the moves")
    p.add_argument("--no-prompt", action="store_true", help="fail on Instagram images instead of asking")
    p.add_argument("--user", help="no DB rows: use this user name and EXIF/Spotteron times")
    p.add_argument("--type", default="snap", help="with --user: image type (default snap)")
    p.add_argument("--timezone", help="with --user: time zone of the image times (default: site's)")

    p = sub.add_parser("sort-shared", help="sort shared/emailed images into sites by GPS")
    p.add_argument("folder", type=Path)
    p.add_argument("--sites", nargs="+", required=True, help="candidate sites")
    p.add_argument("--user", required=True)
    p.add_argument("--timezone", required=True)
    p.add_argument("--source", default="Email")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_sort_shared, needs_site=False)

    p = site_cmd("participation", cmd_participation, "submission statistics")
    p.add_argument("first", type=_date)
    p.add_argument("last", type=_date)
    p.add_argument("--spotteron", type=Path, help="Spotteron export (.xlsx) for app submissions")
    p.add_argument("--root-id", help="the site's Spotteron root ID")
    p.add_argument("--root-col", type=int, default=2)
    p.add_argument("--time-col", type=int, default=17)
    figure_opts(p)

    p = sub.add_parser("shoreline", help="map and review shorelines (see coastsnap shoreline -h)",
                       add_help=False)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=None, needs_site=False)
    return ap


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd == "shoreline":
        from .shoreline.cli import main as shoreline_main

        return shoreline_main(args.rest)
    from .config import load_paths

    paths = load_paths(args.base)
    if args.needs_site:
        args.func(Site(args.site, paths), args)
    else:
        args.func(paths, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
