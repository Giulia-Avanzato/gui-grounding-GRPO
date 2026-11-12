"""GRPO fine-tuning (LoRA, 4-bit) of a small VLM for GUI grounding with a verifiable reward.

NOT TESTED on GPU by the author of this scaffold: TRL's VLM-GRPO API changes between
versions. The script filters GRPOConfig kwargs by the installed signature and prints
what it dropped. See README "If it breaks".

Example (Kaggle T4, first smoke test, ~10 min):
  python scripts/train_grpo.py --out runs/smoke --n_train 32 --steps 4
Real run:
  python scripts/train_grpo.py --out runs/grpo_shape0.3_s0 --n_train 300 --steps 100 --seed 0
"""
import argparse
import inspect
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gg import data as D
from gg.reward import format_reward, make_grounding_reward

LLM_ONLY_LORA = (r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                 r"|mlp\.(gate_proj|up_proj|down_proj))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="rootsautomation/ScreenSpot")
    ap.add_argument("--bbox_format", default="xyxy", choices=["xyxy", "xywh"])
    ap.add_argument("--train_groups", nargs="+", default=["mobile", "web"])
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=400 * 28 * 28)
    ap.add_argument("--shape_coef", type=float, default=0.3, help="0 = binary reward")
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--image_key", default="image", help="dataset column TRL reads images from: image or images")
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, set_seed
    from trl import GRPOConfig, GRPOTrainer
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    set_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "run_args.json"), "w"), indent=2)

    # ---- data -------------------------------------------------------------
    examples = D.load_split(args.dataset, args.train_groups, args.max_pixels,
                            args.bbox_format, args.n_train, args.seed)
    D.check_bbox_overlays(examples, os.path.join(args.out, "overlays"), n=6)
    rows = [{"prompt": D.build_messages(e["instruction"]), args.image_key: e["image"],
             "bbox": e["bbox"], "wh": e["wh"]} for e in examples]
    train_ds = Dataset.from_list(rows)
    print(f"train examples: {len(train_ds)}  groups={args.train_groups}")

    # ---- model ------------------------------------------------------------
    kw = dict(dtype=torch.float16, device_map={"": 0})
    if not args.no_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoVLM.from_pretrained(args.model, **kw)
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=LLM_ONLY_LORA, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()  # must be > 0, otherwise the regex matched nothing
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=28 * 28 * 4, max_pixels=int(args.max_pixels * 1.3), padding_side="left")

    # ---- trainer ----------------------------------------------------------
    grounding_reward = make_grounding_reward(shape_coef=args.shape_coef)
    wanted = dict(
        output_dir=args.out,
        learning_rate=args.lr,
        max_steps=args.steps,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_completion_length=32,
        max_prompt_length=None,          # never truncate: it would cut image tokens
        temperature=1.0,
        beta=0.0,                        # no KL term
        reward_weights=[0.2, 1.0],       # [format, grounding]
        fp16=True, bf16=False,           # T4 has no bf16
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=2,
        save_steps=max(10, args.steps // 4),
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        log_completions=True,
        seed=args.seed,
    )
    accepted = set(inspect.signature(GRPOConfig.__init__).parameters)
    cfg_kwargs = {k: v for k, v in wanted.items() if k in accepted}
    dropped = sorted(set(wanted) - set(cfg_kwargs))
    if dropped:
        print(f"[warn] GRPOConfig in this TRL version does not accept: {dropped}")
    cfg = GRPOConfig(**cfg_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, grounding_reward],
        args=cfg,
        train_dataset=train_ds,
        processing_class=processor,
    )
    # fp16 GradScaler cannot unscale bf16 gradients: force trainable (LoRA) params to fp32
    n_cast = 0
    for prm in trainer.model.parameters():
        if prm.requires_grad and prm.dtype != torch.float32:
            prm.data = prm.data.float()
            n_cast += 1
    print(f"cast {n_cast} trainable tensors to fp32")
    trainer.train()
    trainer.save_model(args.out)
    print("saved adapter to", args.out)


if __name__ == "__main__":
    main()
