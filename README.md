# GUI grounding with GRPO: does RL on a verifiable reward transfer across platforms, and does the model notice when its input degrades?

A small, reproducible study on a single T4: RL fine-tuning (GRPO, LoRA, 4-bit) of a small
vision-language model on single-step GUI grounding, compared with its zero-shot baseline,
followed by a robustness evaluation under controlled visual degradations.

**Status: in progress.** The pipeline runs end-to-end on a Kaggle T4 (smoke-tested). Final results are added below only once produced by these scripts.

## Questions

1. **RL with a verifiable reward.** Does GRPO with a rule-based reward (click inside the target box) improve
   grounding on a *held-out platform group*? (train on mobile + web, test on desktop)
2. **Reward design.** Binary reward vs. a shaped reward that gives partial credit to near misses.
3. **Robustness and uncertainty.** Under blur / darkening / JPEG / non-target occlusion, how much does error
   grow, and does the model's sampling dispersion (spread of clicks over N sampled completions) track it?
   Same protocol as the diffusion-policy study (error and dispersion both divided by their clean value).

## Setup

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`, 4-bit NF4, LoRA on the language model only.
- Data: ScreenSpot (`rootsautomation/ScreenSpot`, 1272 samples, test split only, evaluation benchmark).
  Split by platform to avoid screenshot leakage: **train** = mobile + web, **test** = desktop.
- Output format: `<answer>[x, y]</answer>`, direct (no chain of thought), pixel coordinates of the image the model sees.
  Images are resized by us to a pixel budget (multiple of 28) and the target box is rescaled to the same space.
- Reward: `0.2 * format + 1.0 * grounding`; grounding = 1 if the click is inside the box, else
  `shape_coef * max(0, 1 - d/0.25)` with `d` = distance to the box centre / image diagonal (`shape_coef=0` = binary).
- Statistics: 95% percentile bootstrap CIs; paired bootstrap for model-vs-model differences.

## Run (Kaggle, GPU T4)

Select a single **T4**. If two GPUs are visible, prefix every command with `CUDA_VISIBLE_DEVICES=0` (TRL refuses `DataParallel`).

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
```

## Development notes (observed while building this; base model, 4-bit, desktop, n=100)

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

## Results

_Pending: full desktop baseline (n=334), GRPO adapter, paired comparison with 95% CIs._

## Known limitations (read before quoting any number)

- ScreenSpot is an evaluation benchmark. Training on a platform subset of it and testing on another is a
  pragmatic protocol for a small study, not a standard one. Screenshots of the same app can appear in different
  platform groups; the base model may have seen similar data in pre-training.
- One training run per configuration unless more seeds are run. With ~330 test samples, differences of a few
  points are within noise; look at the CIs.
- Single-step grounding only; no multi-step agent evaluation.
- Direct answers without reasoning; a thinking variant is out of scope.
- 4-bit 3B model, fp16 on T4 (no bf16): numerical instability is possible.

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
gg/data.py       loading, resizing, prompts        scripts/{eval,train_grpo,analyze}.py
tests/test_core.py  17 CPU tests (no GPU needed)
```

## References

- Lu et al., *UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning* (2025), arXiv:2503.21620
- Cheng et al., *SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents* (2024), arXiv:2401.10935
- Shao et al., *DeepSeekMath* (2024), arXiv:2402.03300 (GRPO)
