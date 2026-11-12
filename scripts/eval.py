"""Evaluate a VLM (optionally + LoRA adapter) on ScreenSpot GUI grounding.

One call = one condition (degradation kind + level). Writes one JSONL file with one
row per sample, so statistics can be recomputed later (scripts/analyze.py).

Example (Kaggle T4):
  python scripts/eval.py --tag base --groups desktop --n 334 --out results
  python scripts/eval.py --tag base --groups desktop --n 334 --degrade blur --level 4 \
         --n_disp 4 --out results
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gg import data as D
from gg import degrade as G
from gg.geometry import in_bbox, norm_dist_to_center, parse_point


def load_model(model_name, adapter, load_4bit, max_pixels):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as AutoVLM

    kw = dict(torch_dtype=torch.float16, device_map={"": 0})
    if load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoVLM.from_pretrained(model_name, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    # min/max pixels chosen so the processor does NOT resize our pre-resized images
    processor = AutoProcessor.from_pretrained(
        model_name, min_pixels=28 * 28 * 4, max_pixels=int(max_pixels * 1.3))
    return model, processor


def generate(model, processor, image, instruction, n=1, sample=False,
             max_new_tokens=32, temperature=1.0):
    import torch
    text = processor.apply_chat_template(D.build_messages(instruction),
                                         add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=sample, num_return_sequences=n)
    if sample:
        gen_kw.update(temperature=temperature, top_p=1.0, top_k=0)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)


def dispersion(points, wh):
    """sqrt(var_x + var_y) / image diagonal over the valid sampled points (NaN if < 2)."""
    pts = [p for p in points if p is not None]
    if len(pts) < 2:
        return float("nan")
    a = np.asarray(pts, dtype=float)
    return float(math.sqrt(a[:, 0].var() + a[:, 1].var()) / math.hypot(*wh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter dir")
    ap.add_argument("--tag", required=True, help="name of the model variant, e.g. base, sft, grpo_s0")
    ap.add_argument("--dataset", default="rootsautomation/ScreenSpot")
    ap.add_argument("--bbox_format", default="xyxy", choices=["xyxy", "xywh"])
    ap.add_argument("--groups", nargs="+", default=["desktop"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="subset selection seed (keep fixed)")
    ap.add_argument("--max_pixels", type=int, default=400 * 28 * 28)
    ap.add_argument("--degrade", default="none", choices=["none", "blur", "darken", "jpeg", "occlude"])
    ap.add_argument("--level", type=float, default=0.0)
    ap.add_argument("--n_disp", type=int, default=0, help="# sampled completions for dispersion (0 = off)")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    examples = D.load_split(args.dataset, args.groups, args.max_pixels, args.bbox_format, args.n, args.seed)
    D.check_bbox_overlays(examples, os.path.join(args.out, "overlays"), n=6)
    model, processor = load_model(args.model, args.adapter, not args.no_4bit, args.max_pixels)

    level = int(args.level) if args.degrade in ("blur", "jpeg") else args.level
    path = os.path.join(args.out, f"{args.tag}_{args.degrade}_{level}.jsonl")
    hits = []
    with open(path, "w") as f:
        for i, ex in enumerate(examples):
            img = G.apply(ex["image"], args.degrade, level, bbox=ex["bbox"], seed=i)
            text = generate(model, processor, img, ex["instruction"])[0]
            pt = parse_point(text)
            hit = bool(pt is not None and in_bbox(pt, ex["bbox"]))
            row = dict(tag=args.tag, kind=args.degrade, level=level, idx=i,
                       platform=ex["platform"], data_type=ex["data_type"], text=text,
                       parsed=pt is not None, strict=parse_point(text, strict=True) is not None,
                       hit=hit, dist=(norm_dist_to_center(pt, ex["bbox"], ex["wh"]) if pt else None))
            if args.n_disp >= 2:
                samples = generate(model, processor, img, ex["instruction"], n=args.n_disp, sample=True)
                spts = [parse_point(s) for s in samples]
                row["dispersion"] = dispersion(spts, ex["wh"])
                row["sample_hit_rate"] = float(np.mean([p is not None and in_bbox(p, ex["bbox"]) for p in spts]))
            f.write(json.dumps(row) + "\n")
            hits.append(hit)
            if (i + 1) % 25 == 0:
                print(f"[{args.tag} {args.degrade}={level}] {i + 1}/{len(examples)}  acc={np.mean(hits):.3f}", flush=True)
    print(f"DONE {path}  acc={np.mean(hits):.3f}  n={len(hits)}")


if __name__ == "__main__":
    main()
