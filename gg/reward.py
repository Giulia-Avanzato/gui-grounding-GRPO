"""Verifiable rewards for single-step GUI grounding (TRL GRPOTrainer compatible).

TRL calls each reward function as  f(completions=..., **dataset_columns)  and
expects a list of floats, one per completion. Conversational completions arrive
as [{"role": "assistant", "content": "..."}], plain ones as strings.
"""
from __future__ import annotations

from .geometry import in_bbox, norm_dist_to_center, parse_point


def completion_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, (list, tuple)) and c:
        last = c[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            if isinstance(content, list):  # [{"type": "text", "text": ...}]
                return "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content)
        return str(last)
    return ""


def score_point(text: str, bbox, wh, shape_coef: float = 0.0, shape_scale: float = 0.25) -> float:
    """1.0 if the click lies inside bbox.

    Otherwise shape_coef * max(0, 1 - d / shape_scale), where d is the distance to
    the box centre divided by the image diagonal. shape_coef=0 gives a binary reward.
    Unparseable output scores 0.
    """
    pt = parse_point(text, strict=False)
    if pt is None:
        return 0.0
    if in_bbox(pt, bbox):
        return 1.0
    if shape_coef <= 0:
        return 0.0
    d = norm_dist_to_center(pt, bbox, wh)
    return shape_coef * max(0.0, 1.0 - d / shape_scale)


def format_reward(completions, **kwargs):
    """1.0 if the completion is exactly <answer>[x, y]</answer>."""
    return [1.0 if parse_point(completion_text(c), strict=True) is not None else 0.0
            for c in completions]


def make_grounding_reward(shape_coef: float = 0.3, shape_scale: float = 0.25):
    """Build the accuracy reward. Uses dataset columns `bbox` (x1,y1,x2,y2 in the
    model's pixel space) and `wh` ([width, height] of the image the model sees)."""

    def grounding_reward(completions, bbox, wh, **kwargs):
        return [score_point(completion_text(c), b, s, shape_coef, shape_scale)
                for c, b, s in zip(completions, bbox, wh)]

    grounding_reward.__name__ = f"grounding_reward_shape{shape_coef:g}"
    return grounding_reward
