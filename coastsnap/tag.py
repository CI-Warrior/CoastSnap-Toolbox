"""Caption images with their date, contributor and tide (port of
CSPtagRegisteredImages / CSPtagRegisteredImagesNoTide)."""

from __future__ import annotations

from pathlib import Path

from .naming import parse_filename
from .site import Site
from .timeutils import epoch_to_datetime


def caption(site: Site, image_name: str, tide: bool = True) -> str:
    """``Date: 2024/01/08 Time: 09:00 Contributor: Jo Tide level: 0.42m AHD``."""
    p = parse_filename(Path(image_name).name)
    epoch = int(p["epochtime"])
    t = epoch_to_datetime(epoch, site.gmt_offset)
    text = f"Date: {t:%Y/%m/%d} Time: {t:%H:%M} Contributor: {p['user'].replace('_', '')}"
    if tide:
        text += f" Tide level: {float(site.tide_level(epoch)):0.2f}m AHD"
    return text


def tag_image(src, dst, text: str, font_size: int | None = None):
    """Write ``src`` to ``dst`` with ``text`` in dark grey at the bottom left."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(src).convert("RGB")
    size = font_size or max(12, im.height // 40)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:   # Pillow < 10.1
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(im)
    margin = (int(0.02 * im.width), int(0.02 * im.height))
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((margin[0], im.height - margin[1] - (box[3] - box[1])), text, fill=(51, 51, 51), font=font)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=95)
    return Path(dst)


def tag_registered(site: Site, folder, prefix_len: int = 6, tide: bool = True) -> list[Path]:
    """Caption every image exported from Photoshop into ``folder``.

    Photoshop prefixes exported names with ``prefix_len`` characters; the rest
    is the CoastSnap name. Writes ``<name>_registered.jpg`` beside each.
    """
    out = []
    for f in sorted(Path(folder).glob("*.jpg")):
        if f.stem.endswith("_registered"):
            continue
        name = f.name[prefix_len:]
        out.append(tag_image(f, f.with_name(Path(name).stem + "_registered.jpg"), caption(site, name, tide)))
    return out
