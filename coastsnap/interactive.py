"""Interactive GCP picking (the ginput loop of CSPGrectifyImage), using matplotlib."""

from __future__ import annotations

import numpy as np


def pick_gcps(image, names, title_prefix="Digitize"):
    """Click each named GCP on the image in turn.

    Zoom/pan with the toolbar, then press any key to pick the point with the
    next mouse click.  Returns ``(N, 2)`` pixel coordinates in MATLAB's
    1-based convention, ready for :func:`coastsnap.rectify.rectify_from_gcps`.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    nv, nu = np.shape(image)[:2]
    # extent puts pixel centres at 1..NU, 1..NV like MATLAB's image()
    ax.imshow(image, extent=(0.5, nu + 0.5, nv + 0.5, 0.5))
    uv = []
    for i, name in enumerate(names):
        ax.set_title(f"GCP {i + 1} of {len(names)}: {title_prefix} {name} (key to pick)")
        fig.canvas.draw()
        while not plt.waitforbuttonpress():   # mouse clicks are zoom/pan
            pass
        (u, v), = plt.ginput(1, timeout=0)
        ax.plot(u, v, "go", markersize=3)
        uv.append((u, v))
    return np.array(uv)
