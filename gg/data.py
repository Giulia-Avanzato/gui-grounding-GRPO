"""ScreenSpot loading and preprocessing.

Design choice: we resize every screenshot ourselves (multiple of 28 px, pixel budget
`max_pixels`) and rescale the target box into that same pixel space. The model
processor is then configured so that it does NOT resize again, which keeps the
coordinates the model emits and the ground-truth box in one consistent space.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

PROMPT_TEMPLATE = (
    'Click on the UI element described by: "{instruction}".\n'
    "Reply with exactly one click point (two numbers) as <answer>[x, y]</answer>, in pixel "
    "coordinates of the image shown. Do not output a bounding box and do not add any other text."
)

MOBILE = {"ios", "android", "mobile"}
DESKTOP = {"macos", "windows", "desktop"}


def platform_group(data_source: str) -> str:
    s = (data_source or "").strip().lower()
    if s in MOBILE:
        return "mobile"
    if s in DESKTOP:
        return "desktop"
    return "web"  # web, gitlab, shop, forum, tool, ...


def to_xyxy_pixels(bbox, w: int, h: int, fmt: str = "xyxy"):
    """Return (x1, y1, x2, y2) in pixels.

    fmt: 'xyxy' (x1,y1,x2,y2) or 'xywh' (x,y,w,h). Values all <= 1.0 are treated as
    relative to the image size.
    """
    b = [float(v) for v in bbox]
    if fmt == "xywh":
        b = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
    elif fmt != "xyxy":
        raise ValueError(f"bad bbox format {fmt}")
    if max(b) <= 1.0 + 1e-6:
        b = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
    return tuple(b)


def resize_to_budget(img: Image.Image, max_pixels: int, multiple: int = 28):
    """Downscale so w*h <= ~max_pixels and both sides are multiples of `multiple`.
    Never upscales beyond rounding. Returns (image, sx, sy)."""
    w, h = img.size
    scale = min(1.0, math.sqrt(max_pixels / float(w * h)))
    nw = max(multiple, int(round(w * scale / multiple)) * multiple)
    nh = max(multiple, int(round(h * scale / multiple)) * multiple)
    # rounding can overshoot the budget; step down until it fits
    while nw * nh > max_pixels * 1.05 and (nw > multiple or nh > multiple):
        if nw >= nh and nw > multiple:
            nw -= multiple
        elif nh > multiple:
            nh -= multiple
    out = img.resize((nw, nh), Image.BICUBIC)
    return out, nw / w, nh / h


def prepare_example(row, max_pixels: int, bbox_format: str = "xyxy"):
    """Turn a raw dataset row into the dict used by training and evaluation."""
    img = row["image"].convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = to_xyxy_pixels(row["bbox"], w, h, bbox_format)
    img2, sx, sy = resize_to_budget(img, max_pixels)
    bbox = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
    src = row.get("data_source", row.get("data_souce", ""))
    return {
        "image": img2,
        "bbox": [float(v) for v in bbox],
        "wh": [img2.size[0], img2.size[1]],
        "instruction": row["instruction"],
        "platform": platform_group(src),
        "data_source": src,
        "data_type": row.get("data_type", ""),
    }


def build_messages(instruction: str):
    """Conversational prompt with an image placeholder (the image is passed separately)."""
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT_TEMPLATE.format(instruction=instruction)},
        ],
    }]


def load_screenspot(name: str = "rootsautomation/ScreenSpot", split: str = "test"):
    from datasets import load_dataset  # lazy: not needed for unit tests
    return load_dataset(name, split=split)


def load_split(name: str, groups, max_pixels: int, bbox_format: str = "xyxy",
               n: int | None = None, seed: int = 0):
    """Load ScreenSpot, keep rows whose platform group is in `groups`, shuffle
    deterministically with `seed`, take the first `n`, and preprocess them."""
    import random
    ds = load_screenspot(name)
    col = next((c for c in ("data_source", "data_souce") if c in ds.column_names), None)
    if col is None:
        raise KeyError(f"no data_source column in {ds.column_names}")
    idx = [i for i, s in enumerate(ds[col]) if platform_group(s) in set(groups)]
    random.Random(seed).shuffle(idx)
    if n:
        idx = idx[:n]
    return [prepare_example(ds[i], max_pixels, bbox_format) for i in idx]


def check_bbox_overlays(examples, out_dir: str, n: int = 8):
    """Draw target boxes on a few screenshots. LOOK AT THESE before trusting any number:
    it is the only way to catch a wrong bbox-format assumption."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for i, ex in enumerate(examples[:n]):
        im = ex["image"].copy()
        ImageDraw.Draw(im).rectangle(ex["bbox"], outline=(255, 0, 0), width=4)
        im.save(os.path.join(out_dir, f"overlay_{i}_{ex['platform']}.png"))
