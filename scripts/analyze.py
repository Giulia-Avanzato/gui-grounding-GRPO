"""Summarise results/*.jsonl produced by scripts/eval.py.

  python scripts/analyze.py results/*.jsonl
  python scripts/analyze.py results/*.jsonl --compare grpo_s0 base      # paired, clean condition

Prints (1) accuracy with 95% bootstrap CI per file, (2) for each (tag, degradation) the
ratio-of-relative-changes table: error rate vs mean sampling dispersion, both divided by
their clean value (same idea as Fig. 1 of the diffusion-policy study).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gg.stats import bootstrap_ci, paired_bootstrap_diff


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    args = ap.parse_args()

    runs = defaultdict(dict)  # (tag, kind) -> {level: rows}
    for p in args.files:
        rows = load(p)
        if rows:
            runs[(rows[0]["tag"], rows[0]["kind"])][rows[0]["level"]] = rows

    print("\n== accuracy (click inside target box), 95% bootstrap CI ==")
    print(f"{'tag':14s}{'kind':9s}{'level':>7s}{'n':>6s}{'acc':>8s}{'  95% CI':>16s}{'strict-fmt':>12s}")
    for (tag, kind), by_level in sorted(runs.items()):
        for lvl in sorted(by_level):
            rows = by_level[lvl]
            m, lo, hi = bootstrap_ci([r["hit"] for r in rows])
            fmt = np.mean([r["strict"] for r in rows])
            print(f"{tag:14s}{kind:9s}{lvl:>7}{len(rows):>6d}{m:>8.3f}   [{lo:.3f},{hi:.3f}]{fmt:>10.2f}")

    print("\n== robustness: error vs sampling dispersion, relative to clean ==")
    for (tag, kind), by_level in sorted(runs.items()):
        if kind == "none" or not any("dispersion" in r for rows in by_level.values() for r in rows):
            continue
        clean = runs.get((tag, "none"), {}).get(0.0) or runs.get((tag, "none"), {}).get(0)
        if clean is None:
            continue

        def stats(rows):
            err = 1 - np.mean([r["hit"] for r in rows])
            disp = np.nanmean([r.get("dispersion", np.nan) for r in rows])
            return err, disp

        e0, d0 = stats(clean)
        print(f"\n{tag} / {kind}   (clean: error={e0:.3f}, dispersion={d0:.4f})")
        print(f"{'level':>7s}{'error':>9s}{'x clean':>9s}{'disp':>9s}{'x clean':>9s}")
        for lvl in sorted(by_level):
            e, d = stats(by_level[lvl])
            print(f"{lvl:>7}{e:>9.3f}{e / max(e0, 1e-9):>9.2f}{d:>9.4f}{d / max(d0, 1e-9):>9.2f}")

    if args.compare:
        a, b = args.compare
        ra = (runs.get((a, "none")) or {}).get(0.0) or (runs.get((a, "none")) or {}).get(0)
        rb = (runs.get((b, "none")) or {}).get(0.0) or (runs.get((b, "none")) or {}).get(0)
        if ra and rb:
            ia = {r["idx"]: r["hit"] for r in ra}
            ib = {r["idx"]: r["hit"] for r in rb}
            common = sorted(set(ia) & set(ib))
            d, lo, hi = paired_bootstrap_diff([ia[i] for i in common], [ib[i] for i in common])
            print(f"\n== paired accuracy difference {a} - {b} on {len(common)} shared samples ==")
            print(f"{d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  ({'significant' if lo > 0 or hi < 0 else 'CI includes 0'})")
        else:
            print("\n[compare] clean runs for both tags are required")


if __name__ == "__main__":
    main()
