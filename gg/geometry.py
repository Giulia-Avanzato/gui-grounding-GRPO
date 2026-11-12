"""Point parsing and geometry helpers. Pure Python, no heavy dependencies."""
from __future__ import annotations

import math
import re

_NUM = r"(-?\d+(?:\.\d+)?)"
_PAIR = rf"[\[\(]\s*{_NUM}\s*,\s*{_NUM}\s*[\]\)]"
# a point [x, y] OR a box [x1, y1, x2, y2] (a box is read as its centre in lenient mode)
_PAIR_OR_BOX = rf"[\[\(]\s*{_NUM}\s*,\s*{_NUM}(?:\s*,\s*{_NUM}\s*,\s*{_NUM})?\s*[\]\)]"
_STRICT = re.compile(rf"^\s*<answer>\s*{_PAIR}\s*</answer>\s*$", re.DOTALL)
_TAGGED = re.compile(rf"<answer>\s*{_PAIR_OR_BOX}\s*</answer>", re.DOTALL)
_LENIENT = re.compile(_PAIR_OR_BOX)


def _point_from_match(m):
    a, b, c, d = m.groups() if m.re.groups == 4 else (*m.groups(), None, None)
    if c is not None:
        return (float(a) + float(c)) / 2.0, (float(b) + float(d)) / 2.0
    return float(a), float(b)


def parse_point(text: str, strict: bool = False):
    """Return (x, y) or None.

    strict=True : the whole completion must be exactly <answer>[x, y]</answer> (two numbers).
    strict=False: first tagged pair, else first bare pair; a 4-number box is read as its centre.
    """
    if not isinstance(text, str):
        return None
    if strict:
        m = _STRICT.match(text)
        return (float(m.group(1)), float(m.group(2))) if m else None
    m = _TAGGED.search(text) or _LENIENT.search(text)
    return _point_from_match(m) if m else None


def in_bbox(pt, bbox) -> bool:
    x, y = pt
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def norm_dist_to_center(pt, bbox, wh) -> float:
    """Distance from pt to the bbox centre, divided by the image diagonal."""
    cx, cy = bbox_center(bbox)
    diag = math.hypot(wh[0], wh[1])
    return math.hypot(pt[0] - cx, pt[1] - cy) / diag
