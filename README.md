# CoastSnap Community Beach Monitoring ToolBox

This toolbox was written by Dr. Mitchell Harley at the University of New South Wales (Australia). Please refer to the [publication in Coastal Engineering](https://www.sciencedirect.com/science/article/abs/pii/S0378383918304551) for further information.

## Getting Started

1.  [Download the CoastSnap Starters Toolkit](https://unsw-my.sharepoint.com/:u:/g/personal/z2273773_ad_unsw_edu_au/EdgaW1h8u0NPh0u8lM1opi4Bb8gVSzVGAhtP4-Eg5jeNEQ?e=DIOUCG) and unzip contents to your local directory. This directory will be your BASE_PATH (needed for step 3)

2.  Download the MATLAB files of this GitHub site to the CoastSnap/Code directory

3.  Make sure you have the following Matlab toolkits installed: mapping toolbox, Statistics and ML toolbox, Image processing toolbox and Curve Fitting Toolbox

4.  Update the following two files to match your local setup: CSPsetPaths.m and CSPloadPaths.m

Thats all - you should now be up and running

For an indepth tutorial refer to the following video: https://www.youtube.com/watch?v=Al7vq26dlyk

## Demo using manly CoastSnap example

![](demo.gif)

## Setting up a new CoastSnap station

Setting up a new CoastSnap station requires two steps:

1.  In the **Images** directory, create a copy of **create_newsitename_here** and rename it to your sitename (e.g. *nthnarrabn*, *manly*).

2.  In the **CoastSnapDB.xlsx** (found in the **Database** folder), create a new tab with the *exact* name as your sitename above. It is best to copy an existing tab like **manly** as the structure of this needs to be identical.

## Inserting relevant station metadata

To be added

## Managing images using the CoastSnap Database

All new images from various the sources (e.g. Instagram, Facebook, Email) are to be saved to the **Raw** folder for each respective station.





## CoastSnap GUI



## Python port (work in progress)

The `coastsnap` Python package is a port of this toolbox that reads and writes the same database (CoastSnapDB.xlsx, the Images/Shorelines/Tide Data folders and the `.mat` files), so MATLAB and Python can be used side by side. Everything except the GUI is ported; [docs/python-port.md](docs/python-port.md) maps each MATLAB file to its Python equivalent and lists where the behaviour differs.

```bash
pip install -e ".[test]"
pytest
```

Point it at your database folder (the BASE_PATH above) with `--base`, the `COASTSNAP_BASE` environment variable, or a `coastsnap.toml` file containing `[paths]` and `base = "D:/CoastSnap"`. Then:

```bash
coastsnap ingest manly                                  # Raw -> Processed (CSPraw2Processed)
coastsnap bulk manly --from <rectified image> --to <last image>   # CSPGbulkRectAndMap
coastsnap shoreline review --site manly ...             # QA unreviewed shorelines
coastsnap trend manly <image> --days 42 --out figures   # CSPGmakeTrendPlot
coastsnap change manly <image> --back 1                 # CSPGmakeShorelineChangePlot
coastsnap forecast manly <image> -10                    # CSPGplotShorelineForecast
coastsnap animate manly <image> --out frames            # CSPGmakeBeachWidthAnimation
coastsnap timex manly 2024-01-01 2024-01-31             # CSPmakeDayTimex
coastsnap transects manly <image> SLtransects_manly     # Make Transect File
coastsnap participation manly 2024-01-01 2024-12-31     # CSPplotParticipationStatistics
```

Rectifying a new image by picking its GCPs:

```python
from PIL import Image
import numpy as np
from coastsnap.site import Site
from coastsnap.rectify import RectificationSettings, check_accuracy, rectify_from_gcps, save_rectified
from coastsnap.interactive import pick_gcps

site = Site.open("manly")
image = site.images()[-1]
settings = RectificationSettings.from_site_db(site.db)
img = np.asarray(Image.open(image.path))
uv = pick_gcps(img, [g.name for g in settings.selected_gcps()])
result = rectify_from_gcps(img, uv, settings, tide_level=site.tide_level(image.epoch))
check_accuracy(result.geometry, settings.accuracy_limit)       # the DB's "Acceptable Accuracy"
save_rectified(result, *site.plan_paths(image.path), world=(settings.res, site.origin))
```

**Wide-angle lenses.** The MATLAB code assumes a distortion-free phone camera and tabulates distortion only out to about 56° off-axis, so ultra-wide images lose their edges. The Python port evaluates the lens model directly (`LensCalibration(model="analytic")`, the default) and can either use a known calibration (`LensCalibration.from_opencv(K, dist, (width, height))`) or fit the distortion from the GCPs: `rectify_from_gcps(..., refine_focal=True, free_distortion=("d1", "d2"))`, adding `"c0u", "c0v"` if the principal point is off-centre. Fitting distortion needs more GCPs, spread out towards the image edges. `model="table"` reproduces MATLAB exactly.

The tests compare against outputs of the original MATLAB code, regenerated with `octave --no-gui -q tests/matlab_reference/make_reference.m` and `make_utils_reference.m`.
