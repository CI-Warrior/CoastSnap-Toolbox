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


def draw_polyline(ax, prompt: str, closed: bool = False):
    """Click points on ``ax``; Enter (or a right click) finishes. Returns (K, 2)."""
    import matplotlib.pyplot as plt

    ax.set_title(prompt + " (click points, Enter to finish)")
    ax.figure.canvas.draw()
    pts = np.array(plt.ginput(n=-1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=2))
    if len(pts):
        loop = np.vstack([pts, pts[:1]]) if closed else pts
        ax.plot(loop[:, 0], loop[:, 1], "g-" if not closed else "c-")
    return pts


def draw_transect_inputs(plan_x, plan_y, plan_image):
    """The two drawings the Make Transect File tool asks for, on a plan image.

    Returns ``(roi, coastline)`` in local coordinates for
    :func:`coastsnap.shoreline.make_transects`.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(plan_image, origin="lower", extent=(plan_x[0], plan_x[-1], plan_y[0], plan_y[-1]))
    ax.set_xlabel("Eastings [m]")
    ax.set_ylabel("Northings [m]")
    roi = draw_polyline(ax, "Draw region of interest, equally between sand/water", closed=True)
    coastline = draw_polyline(ax, "Draw representative shoreline (starting from nearest station)")
    return fig, ax, roi, coastline
