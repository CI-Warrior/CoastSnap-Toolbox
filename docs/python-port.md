# MATLAB to Python map

Where each part of the MATLAB toolbox (upstream master, February 2025) lives in the `coastsnap` package, and where the Python version behaves differently on purpose. Both read and write the same database folders and `.mat` layouts.

## Root scripts

| MATLAB | Python |
|---|---|
| `CSPloadPaths`, `CSPsetPaths` | `coastsnap.config.load_paths` (`--base`, `COASTSNAP_BASE` or `coastsnap.toml`) |
| `CSPreadSiteDB` | `coastsnap.db.read_site_db`, or `Site.db` |
| `CSPargusFilename`, `CSPparseFilename` | `coastsnap.naming.argus_filename`, `parse_filename` |
| `CSPepoch2LocalMatlab` | `coastsnap.timeutils.epoch_to_local_datenum` |
| `CSPgetTideLevel` | `coastsnap.tide.get_tide_level`, or `Site.tide_level` |
| `CSPgetImageList`, `CSPgetShorelineList` | `Site.images(kind)`, `Site.shorelines()` |
| `CSPraw2Processed` | `coastsnap.ingest.raw_to_processed` (`coastsnap ingest`) |
| `CSPraw2ProcessedNoDB` | `coastsnap.ingest.raw_to_processed_no_db` (`coastsnap ingest --user`) |
| `CSPprocessSharePath`, `CSPprocessEZDetach` | `coastsnap.ingest.sort_shared_images` (`coastsnap sort-shared`) |
| `CSPmakeDayTimex` | `coastsnap.timex.make_day_timex` (`coastsnap timex`) |
| `CSPtagRegisteredImages(NoTide)` | `coastsnap.tag.tag_registered` |
| `CSPgetParticipationStatistics`, `CSPplotParticipationStatistics` | `coastsnap.plots.participation`, `participation_plot` (`coastsnap participation`) |
| `CSPcorrectAEDT_AEST` | not ported (a one-off fix for one site's 2019 files) |

## rectifyCode

| MATLAB | Python |
|---|---|
| `makeLCPP3`, `makeRadDist`, `makeTangDist` | `coastsnap.camera.lens` |
| `distort`, `undistort`, `distortCaltech`, `undistortCaltech` | `coastsnap.camera.distortion` |
| `angles2R`, `lcpBeta2P`, `P2m`, `findUVnDOF`, `findXYZ`, `findXYZ6dof` | `coastsnap.camera.geometry` |
| `buildRectProducts`, `makeFinalImages` | `coastsnap.rectify.PlanProducts`, `rectify_image` |
| `epoch2Matlab`, `matlab2Epoch`, `matlab2Julian`, `argusDay` | `coastsnap.timeutils` |
| `inImage`, `onScreen` | folded into `PlanProducts` |

## GUI

| MATLAB | Python |
|---|---|
| `CSPGrectifyImage` | `coastsnap.interactive.pick_gcps` + `coastsnap.rectify.rectify_from_gcps`, `check_accuracy`, `save_rectified(world=...)` |
| `CSPGloadExistingGeometry` | `coastsnap.rectify.rectify_with_existing_geometry` |
| `CSPGbulkRectAndMap` | `coastsnap.batch.bulk_rectify_and_map` (`coastsnap bulk`) |
| `CSPGmapShoreline`, `CSPGbulkShorelineMapper` | `coastsnap.shoreline` (`coastsnap shoreline map`) |
| `CSPGeditShorelinePoints`, `CSPGcropShorelinePoints`, `CSPGsaveShoreline`, `CSPGdeleteShoreline`, `CSPGqaShoreline` | `coastsnap.shoreline.review` (`coastsnap shoreline review`) |
| `CSPGmakeTrendPlot` | `coastsnap.plots.trend_plot`, `export_trend` (`coastsnap trend`) |
| `CSPGmakeShorelineChangePlot` | `coastsnap.plots.change_plot` (`coastsnap change`) |
| `CSPGplotShorelineForecast` | `coastsnap.plots.forecast_plot` (`coastsnap forecast`) |
| `CSPGmakeBeachWidthAnimation` | `coastsnap.plots.beach_width_animation` (`coastsnap animate`) |
| `CSPGshiftSLxshore` | `coastsnap.shoreline.shift_shoreline` |
| `CSPGmakeTransectFiles` | `coastsnap.shoreline.make_transects` + `coastsnap.interactive.draw_transect_inputs` (`coastsnap transects`) |
| `CSPGgetVirtualGCP` | `coastsnap.rectify.virtual_gcp` |
| `CSP.m`/`CSP.fig`, `CSPGloadImage`, `CSPGnextImage`, `CSPGpreviousImage`, `CSPGplusXdays`, `CSPGminusXdays` | not ported yet (the GUI shell) |

## tools

`utm2deg` is `coastsnap.utm.utm2deg`. `deg2utm`, `geomplot`, `distinguishable_colors` and `export_fig` are replaced by matplotlib.

## Deliberate differences

- **Paths** come from configuration, not hardcoded Windows paths.
- **Product names**: MATLAB renames with `strrep(name, 'snap', 'plan')` anywhere in the name, so a contributor called "snapper" breaks it. Python replaces only the type field. `daytimex` still becomes `dayplan`/`dayshoreline` so both find each other's files.
- **World files** are written with full precision. MATLAB's `save -ascii` keeps 8 significant digits, which rounds UTM northings to about 0.1 m.
- **Ingest** checks every image first and moves nothing if any cannot be placed. MATLAB stops part way through.
- **Shared images** use the GPS latitude/longitude reference tags. MATLAB negates the latitude, which only works in the southern hemisphere. The per-person site rules hardcoded in `CSPprocessSharePath` are not carried over.
- **Day timex** files are named for local midnight. MATLAB adds the GMT offset where it should subtract it.
- **Shoreline shifts** (trend plot, forecast) move points along each transect, seaward positive. MATLAB moves them along +x, so the direction depends on which way the beach faces, and north-south transects fail.
- **Forecast** takes a beach-width change (negative is erosion) on every beach.
- **Trend plot** draws each moved shoreline at the current image's rectification level (tide + offset); MATLAB shifts by tide + offset but projects at the tide alone.
- **Make Transect File**: MATLAB's "Flip and save!" flips the plotted ends but saves the unflipped transects, and a coastline drawn exactly east-west divides by zero. Both work here.
- **Beach-width animation** loops over every image in the window. MATLAB's loop bound is `2:navepochs` (an epoch time, not a count).
- **Smoothing spline**: MATLAB's Curve Fitting `smoothingspline` with `SmoothingParam` p is scipy's `make_smoothing_spline` with `lam = (1 - p) / p`.
- **Site DB** ranges such as `[10:30]` in "Transect averaging region" are expanded as MATLAB's `str2num` does.
- **Participation**: the Spotteron export column holding the time differs between the two MATLAB scripts (12 and 17), so it is a parameter.

See `coastsnap/shoreline/README.md` for the shoreline-detection differences and the main README for the lens model.
