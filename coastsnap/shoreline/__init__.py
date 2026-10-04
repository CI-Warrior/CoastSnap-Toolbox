"""Shoreline detection, review and analysis on rectified CoastSnap images."""

from .detect import ShorelineResult, map_shoreline, map_shoreline_ccd, map_shoreline_hue
from .io import (
    PlanImage,
    ShorelineRecord,
    list_shorelines,
    load_plan,
    load_shoreline,
    make_record,
    save_shoreline,
    shoreline_path_for,
)
from .review import ShorelineEditor, review_queue, review_shoreline
from .transects import (
    Transects,
    beach_width,
    beach_width_trend,
    load_transects,
    make_transects,
    save_transects,
    shift_shoreline,
    transect_chainage,
    water_level_shift,
)

__all__ = [
    "PlanImage",
    "ShorelineEditor",
    "ShorelineRecord",
    "ShorelineResult",
    "Transects",
    "beach_width",
    "beach_width_trend",
    "list_shorelines",
    "load_plan",
    "load_shoreline",
    "load_transects",
    "make_record",
    "make_transects",
    "map_shoreline",
    "map_shoreline_ccd",
    "map_shoreline_hue",
    "review_queue",
    "review_shoreline",
    "save_shoreline",
    "save_transects",
    "shoreline_path_for",
    "shift_shoreline",
    "transect_chainage",
    "water_level_shift",
]
