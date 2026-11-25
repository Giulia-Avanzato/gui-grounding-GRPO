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
- Training: 300 training examples, 100 steps, seed 0, 4 sampled completions per prompt, learning rate 2e-5, LoRA rank 16, `max_pixels` 313600; shaped reward uses `shape_coef=0.3`.
- Statistics: 95% percentile bootstrap CIs; paired bootstrap for model-vs-model differences.
- Hardware: development and smoke tests on a Kaggle T4. All reported results (baseline, both GRPO trainings,
  evaluations) were produced on a server with NVIDIA RTX 2080 Ti GPUs (11 GB each), fp16 (no bf16), using one GPU
  per run (`CUDA_VISIBLE_DEVICES=0`). Each 100-step training took about 50 minutes.

## Run

Use a single GPU. If several are visible, prefix every command with `CUDA_VISIBLE_DEVICES=0` (TRL refuses `DataParallel`).

`run_all.sh` is the script used for the reported runs. Its final step (clean evaluation with dispersion) stopped early and was rerun separately after adding `remove_invalid_values=True` to the sampling call in `eval.py` (it only affects sampled generations, not the greedy answer used for accuracy); the AUROC analysis (`scripts/auroc_disp.py`) was run afterwards.

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

## Results (single seed, n=334 desktop examples from ScreenSpot)

Accuracy = click falls inside the target box. 95% bootstrap CIs.

| Model | Accuracy | 95% CI |
|---|---|---|
| Base (Qwen2.5-VL-3B) | 0.665 | [0.614, 0.713] |
| GRPO, binary reward (seed 0) | 0.680 | [0.629, 0.728] |
| GRPO, shaped reward (seed 0) | 0.710 | [0.659, 0.757] |

Paired differences on the same 334 examples:

| Comparison | Difference | 95% CI |
|---|---|---|
| shaped - base | +4.5 pts | [+0.9, +8.1] |
| binary - base | +1.5 pts | [-1.8, +4.5] (includes 0) |
| shaped - binary | +3.0 pts | [+0.9, +5.4] |

The base model rarely uses the strict answer format (0.04 vs 0.99 after GRPO), but accuracy is computed with a
lenient parser, so the gain is not explained by parsing failures of the base model.

### Robustness to blur (shaped model)

Accuracy: 0.710 (clean), 0.497 (blur 2), 0.210 (blur 4), 0.090 (blur 8).

Can sampling dispersion (4 samples, no ground truth needed) flag wrong clicks?
AUROC of dispersion for predicting an error, computed within each condition:

| Condition | AUROC | 95% CI |
|---|---|---|
| Clean | 0.76 | [0.71, 0.82] |
| Blur 2 | 0.77 | [0.72, 0.82] |
| Blur 4 | 0.69 | [0.62, 0.76] |
| Blur 8 | 0.53 | [0.42, 0.64] (chance) |

Dispersion is a moderate error signal on clean and lightly blurred inputs and carries no information at the
strongest blur. Full output: `results/auroc_dispersion.txt`.

### Summary of answers (all single seed)

1. Shaped-reward GRPO improved accuracy on the held-out desktop platform by +4.5 points (CI lower bound +0.9).
   The binary-reward run showed no detectable gain.
2. Shaped beat binary by +3.0 points, but with one seed per run this needs replication.
3. Error grows steeply under blur; dispersion is a moderate error signal up to medium blur and uninformative at blur 8.

## Known limitations (read before quoting any number)

- One training run per configuration: the CIs capture test-set sampling noise, not training variance.
  The shaped vs binary difference needs more seeds to be trusted.
- One model (3B, 4-bit + LoRA), one dataset split, one corruption type (blur). Small training budget (100 steps, 300 examples).
- 4 of 334 clean examples had no valid dispersion and were excluded from the AUROC.
- An AUROC around 0.76 is a useful but not reliable error detector.
- ScreenSpot is an evaluation benchmark. Training on a platform subset of it and testing on another is a
  pragmatic protocol for a small study, not a standard one. Screenshots of the same app can appear in different
  platform groups; the base model may have seen similar data in pre-training.
- Single-step grounding only; no multi-step agent evaluation.
- Direct answers without reasoning; a thinking variant is out of scope.
- 4-bit 3B model, fp16 on Turing GPUs (no bf16): numerical instability is possible.

## If it breaks (smoke-tested on Kaggle T4 with transformers 5.22, trl 1.17, peft 0.21, bitsandbytes 0.50)

| Symptom | Likely cause / fix |
|---|---|
| `GRPOConfig` warns about dropped kwargs | Different TRL version; the script filters kwargs by signature. Check which ones were dropped (e.g. `reward_weights`, `max_prompt_length`). |
| Image/token count mismatch error | A prompt is being truncated. Set `max_prompt_length` to a large int (e.g. 1024) in `train_grpo.py`. |
| KeyError/ValueError about `image` | Newer TRL expects a list under `images`. Run with `--image_key images` and wrap images in a list in `train_grpo.py`. |
| CUDA OOM | Lower `--max_pixels` (e.g. `300*28*28`), `--num_generations 4`→`2` (keep batch divisible), or use `Qwen/Qwen2-VL-2B-Instruct`. |
| NaN loss / all-zero reward | fp16 instability, or the model never outputs a parseable point: check the logged completions and the baseline's `strict-fmt`. |
| `trainable params: 0` | LoRA regex matched nothing in this transformers version; inspect `model.named_modules()` and adjust `LLM_ONLY_LORA`. |
| Baseline accuracy ~0 | Coordinate-space or bbox-format mismatch. Check the overlays first, then print a few raw completions. |

## Layout

```
gg/geometry.py   point parsing, box tests          gg/degrade.py  blur / darken / jpeg / non-target occlusion
gg/reward.py     format + grounding rewards        gg/stats.py    bootstrap CIs
gg/data.py       loading, resizing, prompts        scripts/{eval,train_grpo,analyze,auroc_disp}.py
tests/test_core.py  17 CPU tests (no GPU needed)   run_all.sh     script used for the reported runs
```

## References

- Lu et al., *UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning* (2025), arXiv:2503.21620
- Cheng et al., *SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents* (2024), arXiv:2401.10935
- Shao et al., *DeepSeekMath* (2024), arXiv:2402.03300 (GRPO)
