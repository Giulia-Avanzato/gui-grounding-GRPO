"""Controlled visual degradations for the robustness study (PIL only).

All functions take and return RGB PIL images. `occlude_nontarget` places a grey
patch that never overlaps the target box, so an accuracy drop cannot be explained
by the target simply being hidden.
"""
from __future__ import annotations

import io
import random

from PIL import Image, ImageEnhance, ImageFilter

LEVELS = {
    "blur": [0, 2, 4, 8],             # box-blur radius in pixels
    "darken": [0.0, 0.25, 0.5, 0.75],  # fraction of brightness removed
    "jpeg": [100, 30, 10, 3],          # JPEG quality (lower = worse)
    "occlude": [0.0, 0.2, 0.4, 0.6],   # fraction of image area covered
}


def blur(img: Image.Image, radius: float) -> Image.Image:
    return img if radius <= 0 else img.filter(ImageFilter.BoxBlur(radius))


def darken(img: Image.Image, frac: float) -> Image.Image:
    return img if frac <= 0 else ImageEnhance.Brightness(img).enhance(1.0 - frac)


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    if quality >= 100:
        return img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def occlude_nontarget(img: Image.Image, frac: float, bbox, seed: int = 0, tries: int = 200) -> Image.Image:
    """Cover ~frac of the image area with a grey rectangle disjoint from bbox.

    If no disjoint placement is found, the patch is shrunk until one is.
    """
    if frac <= 0:
        return img
    w, h = img.size
    rng = random.Random(seed)
    x1, y1, x2, y2 = bbox
    scale = 1.0
    for _ in range(8):
        pw = max(1, int(w * (frac * scale) ** 0.5))
        ph = max(1, int(h * (frac * scale) ** 0.5))
        for _ in range(tries):
            px = rng.randint(0, max(0, w - pw))
            py = rng.randint(0, max(0, h - ph))
            disjoint = px + pw <= x1 or px >= x2 or py + ph <= y1 or py >= y2
            if disjoint:
                out = img.copy()
                out.paste((128, 128, 128), (px, py, px + pw, py + ph))
                return out
        scale *= 0.7
    return img


def apply(img: Image.Image, kind: str, level, bbox=None, seed: int = 0) -> Image.Image:
    if kind in (None, "none"):
        return img
    if kind == "blur":
        return blur(img, level)
    if kind == "darken":
        return darken(img, level)
    if kind == "jpeg":
        return jpeg(img, level)
    if kind == "occlude":
        return occlude_nontarget(img, level, bbox, seed=seed)
    raise ValueError(f"unknown degradation: {kind}")
