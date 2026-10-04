"""Command line for mapping and reviewing shorelines.

Map rectified images, saving unreviewed shorelines (like ``CSPGbulkRectAndMap``)::

    python -m coastsnap.shoreline map Rectified/manly/2024/*.plan.*.mat \\
        --transects "Shorelines/Transect Files/SLtransects_manly.mat" \\
        --shoreline-root Shorelines --origin 342000 6265000 --utm-zone 56H

Add ``--review`` to check each one in the reviewer as it is mapped (like
``CSPGmapShoreline``). Later, step through everything still unreviewed::

    python -m coastsnap.shoreline review --site manly --shoreline-root Shorelines \\
        --rectified-root Rectified --transects ... --origin ... --utm-zone 56H
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .detect import map_shoreline
from .io import list_shorelines, load_plan, make_record, parse_filename, save_shoreline, shoreline_path_for
from .review import ACCEPTED, QueueItem, review_queue, review_shoreline, unreviewed
from .transects import load_transects


def _common(p):
    p.add_argument("--transects", required=True, type=Path, help="site SLtransects .mat file")
    p.add_argument("--shoreline-root", required=True, type=Path, help="the Shorelines folder")
    p.add_argument("--origin", required=True, nargs=2, type=float, metavar=("EASTINGS", "NORTHINGS"),
                   help="site origin from the site DB")
    p.add_argument("--utm-zone", required=True, help="site UTM zone, e.g. 56H")


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m coastsnap.shoreline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("map", help="detect shorelines on rectified plan images")
    m.add_argument("plans", nargs="+", type=Path, help="rectified ...plan....mat files")
    _common(m)
    m.add_argument("--method", default="CCD", choices=["CCD", "HUE"])
    m.add_argument("--review", action="store_true", help="review each shoreline before saving")
    m.add_argument("--overwrite", action="store_true", help="replace shorelines that already exist")

    r = sub.add_parser("review", help="review shorelines that have not been checked yet")
    r.add_argument("--site", required=True)
    r.add_argument("--rectified-root", required=True, type=Path, help="the Rectified folder")
    _common(r)
    r.add_argument("--all", action="store_true", help="include shorelines already reviewed")
    r.add_argument("--delete-rejected", action="store_true",
                   help="delete rejected shoreline files instead of leaving them unreviewed")
    return ap


def cmd_map(args) -> int:
    transects = load_transects(args.transects)
    origin = tuple(args.origin)
    for n, plan_path in enumerate(args.plans, 1):
        out_path = shoreline_path_for(args.shoreline_root, plan_path.name)
        prefix = f"[{n}/{len(args.plans)}] {plan_path.name}:"
        if out_path.exists() and not args.overwrite:
            print(prefix, "shoreline exists, skipping (use --overwrite)")
            continue
        plan = load_plan(plan_path)
        if args.review:
            out = review_shoreline(plan, transects, origin, args.utm_zone, save_to=out_path,
                                   method=args.method, title=plan_path.name)
            print(prefix, out.decision + (f" -> {out.path}" if out.path else ""))
            continue
        result = map_shoreline(plan.xgrid, plan.ygrid, plan.iplan, transects, method=args.method)
        if result.is_empty:
            print(prefix, "no shoreline found")
            continue
        record = make_record(result.xy, plan.rectz, origin, args.utm_zone,
                             result.method, result.threshold, qa=False)
        save_shoreline(out_path, record)
        print(prefix, f"{len(result.xy)} points, threshold {result.threshold:.3g} -> {out_path} (unreviewed)")
    return 0


def plan_path_for(rectified_root: Path, shoreline_path: Path) -> Path:
    parts = parse_filename(shoreline_path.name)
    return rectified_root / parts["site"] / parts["year"] / shoreline_path.name.replace(".shoreline.", ".plan.")


def cmd_review(args) -> int:
    paths = [p for _, p in list_shorelines(args.shoreline_root, args.site)]
    if not args.all:
        paths = unreviewed(paths)
    items = []
    for p in paths:
        plan = plan_path_for(args.rectified_root, p)
        if plan.exists():
            items.append(QueueItem(p, plan))
        else:
            print(f"{p.name}: no rectified image at {plan}, skipping")
    if not items:
        print("Nothing to review.")
        return 0
    print(f"Reviewing {len(items)} shoreline(s).")
    results = review_queue(items, load_transects(args.transects), tuple(args.origin), args.utm_zone,
                           on_reject="delete" if args.delete_rejected else "keep")
    accepted = sum(d == ACCEPTED for _, d in results)
    print(f"{accepted} accepted, {len(results) - accepted} rejected or skipped.")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"map": cmd_map, "review": cmd_review}[args.cmd](args)
