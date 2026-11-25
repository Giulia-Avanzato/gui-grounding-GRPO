# GUI grounding with GRPO: does RL on a verifiable reward help on a held-out platform, and does sampling dispersion track input degradation?

A small, reproducible study on a single consumer GPU: RL fine-tuning (GRPO, LoRA, 4-bit) of a small
vision-language model on single-step GUI grounding, compared with its zero-shot baseline,
followed by a robustness evaluation under blur.

**Status: core experiments complete (single seed).** Not run: additional seeds, and corruptions other than blur
(darkening, JPEG and occlusion are implemented in `gg/degrade.py` but were not evaluated).

## Questions

1. **RL with a verifiable reward.** Does GRPO with a rule-based reward (click inside the target box) improve
   grounding on a *held-out platform group*? (train on mobile + web, test on desktop)
2. **Reward design.** Binary reward vs. a shaped reward that gives partial credit to near misses.
3. **Robustness and uncertainty.** Under blur, how much does error grow, and does the model's sampling
   dispersion (spread of clicks over N sampled completions) track it? Same protocol as the diffusion-policy
   study (error and dispersion both divided by their clean value).

## Setup

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`, 4-bit NF4, LoRA on the language model only.
- Data: ScreenSpot (`rootsautomation/ScreenSpot`, 1272 samples, test split only, evaluation benchmark).
  Split by platform to avoid screenshot leakage: **train** = mobile + web, **test** = desktop.
- Output format: `<answer>[x, y]</answer>`, direct (no chain of thought), pixel coordinates of the image the model sees.
  Images are resized by us to a pixel budget (multiple of 28) and the target box is rescaled to the same space.
- Reward: `0.2 * format + 1.0 * grounding`; grounding = 1 if the click is inside the box, else
  `shape_coef * max(0, 1 - d/0.25)` with `d` = distance to the box centre / image diagonal (`shape_coef=0` = binary).
- Training: 300 training examples, 100 steps, seed 0; shaped reward uses `shape_coef=0.3`.
- Statistics: 95% percentile bootstrap CIs; paired bootstrap for model-vs-model differences.
- Hardware: development and smoke tests on a Kaggle T4. All reported results (baseline, both GRPO trainings,
  evaluations) were produced on a server with NVIDIA RTX 2080 Ti GPUs (11 GB each), fp16 (no bf16), using one GPU
  per run (`CUDA_VISIBLE_DEVICES=0`). Each 100-step training took about 50 minutes.

## Run

Use a single GPU. If several are visible, prefix every command with `CUDA_VISIBLE_DEVICES=0` (TRL refuses `DataParallel`).
`run_all.sh` is the script used to produce the reported runs.

```bash
pip install -q -U transformers trl peft bitsandbytes accelerate datasets
python -m pytest -q tests                      # CPU tests, ~1 s

# 0. Look at the overlays in results/overlays/*.png. If the red boxes are not on the target
#    elements, the bbox format is wrong: rerun with --bbox_format xywh.
python scripts/eval.py --tag base --groups desktop --n 100 --out results

# 1. Baseline on the full desktop test set
python scripts/eval.py --tag base --groups desktop --out results

# 2. Smoke test of training (measure seconds/step here, then choose --steps)
python scripts/train_grpo.py --out runs/smoke --n_train 32 --steps 4

# 3. Real run
python scripts/train_grpo.py --out runs/grpo_shape0.3_s0 --n_train 300 --steps 100 --seed 0

# 4. Evaluate the trained adapter on the same test set
python scripts/eval.py --tag grpo_s0 --adapter runs/grpo_shape0.3_s0 --groups desktop --out results

# 5. Compare (paired bootstrap, same samples)
python scripts/analyze.py results/*.jsonl --compare grpo_s0 base
```

Ablation and robustness (after step 5):

```bash
# binary reward
python scripts/train_grpo.py --out runs/grpo_binary_s0 --shape_coef 0 --n_train 300 --steps 100 --seed 0
# more seeds: --seed 1, --seed 2  (the training run is the unit of replication, not the test sample)

# degradations, with dispersion (4 sampled completions per example)
for L in 2 4 8; do python scripts/eval.py --tag grpo_s0 --adapter runs/grpo_shape0.3_s0 --groups desktop \
    --degrade blur --level $L --n_disp 4 --out results; done
python scripts/eval.py --tag grpo_s0 --adapter runs/grpo_shape0.3_s0 --groups desktop --n_disp 4 --out results  # clean + dispersion
python scripts/analyze.py results/*.jsonl

# per-example check: does dispersion flag wrong clicks? (AUROC within each condition)
python scripts/auroc_disp.py results/grpo_s0_*.jsonl > results/auroc_dispersion.txt
```

## Development notes (observed while building this; base model, 4-bit, desktop, n=100, Kaggle T4)

| prompt / parsing | accuracy | answers parsed | strict `<answer>` format |
|---|---|---|---|
| first prompt, only 2-number points counted | 0.25 | 0.46 | 0.02 |
| revised prompt, 4-number box answers read as their centre | 0.68 | 0.98 | 0.03 |

The base model often answers with a 4-number box instead of a point. Counting those as unparseable
underestimated accuracy by a wide margin, so a trained-vs-base comparison against the first number would
have been inflated. Accuracy and strict-format rate are therefore always reported separately. On the
revised protocol: text elements 0.75 (n=59), icons 0.59 (n=41), with wide intervals at this sample size.

Two environment fixes were needed with recent library versions: expose a single GPU, and cast the LoRA
parameters to fp32 so the fp16 gradient scaler can unscale them (done in `scripts/train_grpo.py`).

## Results (single seed, n=334 desktop examples
