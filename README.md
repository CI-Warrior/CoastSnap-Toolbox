# CoastSnap Community Beach Monitoring ToolBox

This toolbox was written by Dr. Mitchell Harley at the University of New South Wales (Australia). Please refer to the [publication in Coastal Engineering](https://www.sciencedirect.com/science/article/abs/pii/S0378383918304551) for further information.

## Getting Started

1.  [Download the CoastSnap Starters Toolkit](https://unsw-my.sharepoint.com/:u:/g/personal/z2273773_ad_unsw_edu_au/Ed2R-hnWdytDm-Ya94DCG0MBUBJbiaGZkR5em16g8weODg?e=V10fsx) and unzip contents to your local directory. This directory will be your BASE_PATH (needed for step 3)

2.  Download the MATLAB files of this GitHub site to the CoastSnap/Code directory

3.  Update the following two files to match your local setup: CSPsetPaths.m and CSPloadPaths.m

Thats all - you should now be up and running

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

The `coastsnap` Python package is a port of this toolbox. So far it covers image rectification: the camera model and lens distortion from `rectifyCode/`, the GCP solve with the FOV sweep from `CSPGrectifyImage`, and plan-view products from `buildRectProducts`/`makeFinalImages`.

```bash
pip install -e ".[db,test]"
pytest
```

```python
from PIL import Image
import numpy as np
from coastsnap.db import read_site_db
from coastsnap.rectify import RectificationSettings, rectify_from_gcps, rectified_paths, save_rectified
from coastsnap.interactive import pick_gcps

settings = RectificationSettings.from_site_db(read_site_db("Database/CoastSnapDB.xlsx", "manly"))
img = np.asarray(Image.open(image_path))
uv = pick_gcps(img, [g.name for g in settings.selected_gcps()])
result = rectify_from_gcps(img, uv, settings, tide_level=0.4)
print(result.geometry.rmse, result.geometry.fov_deg)
save_rectified(result, *rectified_paths(image_path))   # same .jpg/.mat layout as MATLAB
```

**Wide-angle lenses.** The MATLAB code assumes a distortion-free phone camera and tabulates distortion only out to about 56° off-axis, so ultra-wide images lose their edges. The Python port evaluates the lens model directly (`LensCalibration(model="analytic")`, the default) and can either use a known calibration (`LensCalibration.from_opencv(K, dist, (width, height))`) or fit the distortion from the GCPs: `rectify_from_gcps(..., refine_focal=True, free_distortion=("d1", "d2"))`, adding `"c0u", "c0v"` if the principal point is off-centre. Fitting distortion needs more GCPs, spread out towards the image edges. `model="table"` reproduces MATLAB exactly.

The tests compare against outputs of the original MATLAB code, regenerated with `octave --no-gui -q tests/matlab_reference/make_reference.m`.
