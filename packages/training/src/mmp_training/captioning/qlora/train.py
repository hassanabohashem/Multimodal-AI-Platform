"""QLoRA fine-tuning of Qwen3-VL-4B for captioning.

Run:
    make train-caption
    # or
    uv run python -m mmp_training.captioning.qlora.train --config configs/caption_qlora.yaml

The LoRA targets LLM-tower projections only; the vision tower stays frozen so
the adapter is servable by vLLM (docs/adr/001). Loss is masked to the
assistant turn. Checkpoints are adapter-only (~50 MB) and resume-safe for
Colab/Kaggle sessions.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str) -> dict[str, Any]:
    """Read a YAML experiment config."""
    return yaml.safe_load(Path(path).read_text())


def build_conversation(example: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Map a (image, caption) pair to the chat format TRL expects for VLMs."""
    return {
        "images": [example["image"]],
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": example["caption"]}]},
        ],
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    quant = BitsAndBytesConfig(
        load_in_4bit=cfg["quant"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quant"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model_id"], quantization_config=quant, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(cfg["model_id"])

    peft_config = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM",
    )

    # Dataset: COCO Karpathy split prepared by data/scripts (image path + caption columns).
    ds = load_dataset("yerevann/coco-karpathy", split="train").shuffle(seed=cfg["seed"])
    if cfg["data"].get("subset"):
        ds = ds.select(range(cfg["data"]["subset"]))
    ds = ds.map(lambda ex: build_conversation(ex, cfg["data"]["prompt"]))

    sft = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["train"]["epochs"],
        learning_rate=cfg["train"]["lr"],
        lr_scheduler_type=cfg["train"]["scheduler"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        per_device_train_batch_size=cfg["train"]["per_device_batch"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        bf16=cfg["train"]["bf16"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        max_length=cfg["train"]["max_seq_len"],
        save_steps=cfg["train"]["save_steps"],
        logging_steps=20,
        report_to="wandb",
        run_name=f"caption-qlora-r{cfg['lora']['r']}",
        seed=cfg["seed"],
        assistant_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model, args=sft, train_dataset=ds, processing_class=processor, peft_config=peft_config
    )
    trainer.train(resume_from_checkpoint=any(Path(cfg["output_dir"]).glob("checkpoint-*")))
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
