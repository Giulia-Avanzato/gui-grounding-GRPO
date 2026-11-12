import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PIL import Image

from gg.geometry import parse_point, in_bbox
from gg.reward import format_reward, make_grounding_reward, score_point, completion_text
from gg.data import to_xyxy_pixels, resize_to_budget, prepare_example, platform_group
from gg.degrade import occlude_nontarget, blur, darken, jpeg
from gg.stats import bootstrap_ci, paired_bootstrap_diff


# ---------- geometry ----------
def test_parse_strict_and_lenient():
    assert parse_point("<answer>[10, 20]</answer>", strict=True) == (10.0, 20.0)
    assert parse_point("<answer>(10.5,20)</answer>", strict=True) == (10.5, 20.0)
    assert parse_point("sure! <answer>[1, 2]</answer>", strict=True) is None
    assert parse_point("sure! <answer>[1, 2]</answer>") == (1.0, 2.0)
    assert parse_point("click at (3, 4)") == (3.0, 4.0)
    assert parse_point("no point here") is None
    assert parse_point(None) is None


def test_box_is_read_as_centre_in_lenient_mode_only():
    assert parse_point("[347, 109, 362, 128]") == (354.5, 118.5)
    assert parse_point("<answer>[0, 0, 10, 20]</answer>") == (5.0, 10.0)
    assert parse_point("<answer>[0, 0, 10, 20]</answer>", strict=True) is None
    assert parse_point("[1, 2, 3]") is None            # three numbers: neither point nor box
    assert parse_point("[15,42]") == (15.0, 42.0)


def test_in_bbox_edges_inclusive():
    assert in_bbox((0, 0), (0, 0, 10, 10))
    assert in_bbox((10, 10), (0, 0, 10, 10))
    assert not in_bbox((10.1, 5), (0, 0, 10, 10))


# ---------- reward ----------
BOX, WH = (100, 100, 200, 150), (1000, 800)


def test_format_reward():
    comps = ["<answer>[1, 2]</answer>", "[1, 2]", [{"role": "assistant", "content": "<answer>[3,4]</answer>"}]]
    assert format_reward(comps) == [1.0, 0.0, 1.0]


def test_binary_reward():
    r = make_grounding_reward(shape_coef=0.0)
    out = r(["<answer>[150, 120]</answer>", "<answer>[900, 700]</answer>", "garbage"],
            bbox=[BOX] * 3, wh=[WH] * 3)
    assert out == [1.0, 0.0, 0.0]


def test_shaped_reward_is_monotone_and_below_hit():
    near = score_point("<answer>[210, 125]</answer>", BOX, WH, shape_coef=0.3)
    far = score_point("<answer>[600, 600]</answer>", BOX, WH, shape_coef=0.3)
    hit = score_point("<answer>[150, 120]</answer>", BOX, WH, shape_coef=0.3)
    assert hit == 1.0
    assert 0.0 < near < 0.3          # a near miss never beats a hit
    assert far == 0.0 or far < near
    assert near > far


def test_reward_function_has_name_for_logging():
    assert "grounding_reward" in make_grounding_reward(0.3).__name__


def test_completion_text_variants():
    assert completion_text("abc") == "abc"
    assert completion_text([{"role": "assistant", "content": "x"}]) == "x"
    assert completion_text([{"role": "assistant", "content": [{"type": "text", "text": "y"}]}]) == "y"


# ---------- data ----------
def test_bbox_formats():
    assert to_xyxy_pixels((10, 20, 30, 40), 100, 100, "xyxy") == (10, 20, 30, 40)
    assert to_xyxy_pixels((10, 20, 30, 40), 100, 100, "xywh") == (10, 20, 40, 60)
    assert to_xyxy_pixels((0.1, 0.2, 0.3, 0.4), 100, 200, "xyxy") == (10, 40, 30, 80)
    x1, y1, x2, y2 = to_xyxy_pixels((0.1, 0.2, 0.2, 0.1), 100, 100, "xywh")
    assert (round(x1), round(y1), round(x2), round(y2)) == (10, 20, 30, 30)


def test_resize_budget_and_multiples():
    img = Image.new("RGB", (2560, 1440))
    out, sx, sy = resize_to_budget(img, 400_000)
    assert out.size[0] % 28 == 0 and out.size[1] % 28 == 0
    assert out.size[0] * out.size[1] <= 400_000 * 1.05
    assert abs(sx - out.size[0] / 2560) < 1e-9


def test_small_image_not_upscaled_much():
    img = Image.new("RGB", (300, 200))
    out, _, _ = resize_to_budget(img, 400_000)
    assert abs(out.size[0] - 300) <= 28 and abs(out.size[1] - 200) <= 28


def test_prepare_example_keeps_geometry_consistent():
    img = Image.new("RGB", (1000, 500))
    row = {"image": img, "bbox": [0.4, 0.4, 0.6, 0.6], "instruction": "x", "data_source": "macos"}
    ex = prepare_example(row, max_pixels=100_000)
    w, h = ex["wh"]
    x1, y1, x2, y2 = ex["bbox"]
    assert abs(x1 / w - 0.4) < 0.02 and abs(y2 / h - 0.6) < 0.02   # still ~relative-consistent
    assert ex["platform"] == "desktop"
    # centre of box must be a hit
    assert in_bbox(((x1 + x2) / 2, (y1 + y2) / 2), ex["bbox"])


def test_platform_group():
    assert platform_group("iOS") == "mobile" and platform_group("android") == "mobile"
    assert platform_group("macOS") == "desktop" and platform_group("windows") == "desktop"
    assert platform_group("web") == "web" and platform_group("gitlab") == "web"


# ---------- degradations ----------
def _noisy(w=200, h=200):
    import numpy as np
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype="uint8"))


def test_degradations_preserve_size():
    img = _noisy()
    for out in (blur(img, 4), darken(img, 0.5), jpeg(img, 10)):
        assert out.size == img.size


def test_darken_reduces_brightness():
    import numpy as np
    img = _noisy()
    assert np.asarray(darken(img, 0.5)).mean() < np.asarray(img).mean()


def test_occlusion_never_touches_target():
    import numpy as np
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    bbox = (80, 80, 120, 120)
    for seed in range(30):
        out = np.asarray(occlude_nontarget(img, 0.3, bbox, seed=seed))
        target_patch = out[80:120, 80:120]
        assert (target_patch == 255).all(), f"target covered at seed {seed}"
    covered = (np.asarray(occlude_nontarget(img, 0.3, bbox, seed=1)) != 255).any(axis=2).mean()
    assert covered > 0.05  # something was actually occluded


# ---------- stats ----------
def test_bootstrap_ci_brackets_mean():
    vals = [1] * 60 + [0] * 40
    m, lo, hi = bootstrap_ci(vals, seed=1)
    assert abs(m - 0.6) < 1e-9 and lo < m < hi and hi - lo < 0.25


def test_paired_diff_detects_shift():
    a = [1] * 80 + [0] * 20          # 80% correct
    b = [1] * 50 + [0] * 50          # 50% correct, same 100 samples (nested hits)
    d, lo, hi = paired_bootstrap_diff(a, b, seed=0)
    assert abs(d - 0.3) < 1e-9 and lo > 0.15 and hi < 0.45
