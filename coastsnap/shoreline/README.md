# coastsnap.shoreline

Python port of the CoastSnap shoreline mapping step: automatic detection on a
rectified plan image, a review window where a person checks and fixes the
result, and the transect / beach-width tools behind the trend and change plots.

| MATLAB | Python |
|---|---|
| `mapShorelineCCD.m`, `mapShorelineHUE.m` (Shoreline-Mapping-Toolbox) | `detect.map_shoreline_ccd`, `detect.map_shoreline_hue` |
| `rotatePoints.m`, `unrotatePoints.m`, `inpoly.m` | inside `detect.points_on_transects`, `matplotlib.path.Path` |
| `CSPGmapShoreline.m` | `review.review_shoreline` (or `python -m coastsnap.shoreline map --review`) |
| `CSPGeditShorelinePoints.m`, `CSPGcropShorelinePoints.m` | drag / insert / delete / crop in `review.ShorelineReviewer` |
| `CSPGsaveShoreline.m`, `QA` flag | accept in the reviewer saves with `QA = 1` |
| mapping half of `CSPGbulkRectAndMap.m` | `python -m coastsnap.shoreline map` (saves with `QA = 0`) |
| (new) review everything with `QA = 0` | `review.review_queue`, `python -m coastsnap.shoreline review` |
| `CSPgetShorelineList.m`, `CSPparseFilename.m` | `io.list_shorelines`, `io.parse_filename` |
| `polyxpoly` + tidal correction in `CSPGmakeTrendPlot.m` / `CSPGmakeShorelineChangePlot.m` | `transects.transect_chainage`, `transects.beach_width`, `transects.beach_width_trend` |

Files stay compatible with the MATLAB toolbox: it reads the rectified
`...plan....mat` (`xgrid`, `ygrid`, `Iplan`, `metadata.rectz`) and the site
`SLtransects` `.mat`, and writes the same `sl` struct (`xyz`, `UTM`, `UTMzone`,
`UV`, `method`, `threshold`, `QA`, `whenDone`) to
`Shorelines/<site>/<year>/<...shoreline...>.mat`.

## Human-in-the-loop workflow

1. Map a batch unattended: `python -m coastsnap.shoreline map <plan .mat files> ...`.
   Every shoreline is saved unreviewed (`QA = 0`).
2. Review: `python -m coastsnap.shoreline review --site <site> ...` opens each
   unreviewed shoreline over its plan image with the threshold histogram
   below. Drag, insert (`i`), delete (right-click) or crop (`c`) points, click
   the histogram to re-detect at a different threshold, `m` to try HUE. `a`
   accepts and saves with `QA = 1`; `r` rejects; `n` skips.
3. Analyses should use only `QA = 1` shorelines.

Pass `--review` to `map` to check each image as it is mapped instead.

## Image (UV) coordinates

The saved `UV` column (shoreline in oblique-image pixels) needs the camera
geometry. Pass a `project_uv(xyz) -> uv` callable to `make_record`,
`review_shoreline` or `review_queue` (the review window then also draws the
shoreline on the oblique image). The CLI uses
`coastsnap.camera.make_uv_projector(metadata["geom"])` from the rectification
port when it is installed, and writes NaN for `UV` otherwise.

## Differences from the MATLAB code

- Transect sampling for the threshold samples each transect only; MATLAB's
  `improfile` call also sampled the zig-zag joining one transect's end to the
  next one's start.
- `ksdensity` and `multithresh` are reimplemented (`detect.ksdensity`,
  `detect.otsu_threshold`), so thresholds can differ slightly from MATLAB's.
- The longest contour is chosen over all contours. The MATLAB indexing never
  considered the last contour in the contour matrix.
- HUE corrects the transect angle for west-pointing transects, as CCD already
  did. MATLAB's HUE version did not, so it missed shorelines on those
  transects.
- The MATLAB GUI called `mapShorelineHUE` with six arguments though it takes
  five, so HUE never worked from the GUI. It works here.
- `CSPGqaShoreline.m` called the deleted `CSPgetGCPcombo` and was broken; QA is
  now the accept step of the reviewer.
- The trend, change, forecast and animation figures are in `coastsnap.plots`.

## Tests

`python -m pytest` (from the repo root) runs synthetic-image tests for detection, file round trips,
analysis, the editor and the review window (headless). Golden tests against
MATLAB output from the Starters Toolkit are still to do.

Dependencies beyond the package's core set: `matplotlib` and `contourpy`
(contourpy ships with matplotlib).
