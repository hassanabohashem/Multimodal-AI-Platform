"""QLoRA fine-tuning of Qwen3-VL-4B on VQA v2 — same recipe as captioning.

Run: make train-vqa
Differences from captioning: dataset mapping (question -> short answer) and
the short-answer prompt template from configs/vqa_qlora.yaml.
"""
from __future__ import annotations

import argparse
from typing import Any

from mmp_training.captioning.qlora.train import load_config


def build_conversation(example: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Map a VQA v2 record to chat format with the majority human answer as target."""
    return {
        "images": [example["image"]],
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"{prompt} Question: {example['question']}"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": example["multiple_choice_answer"]}]},
        ],
    }


def main() -> None:
    """Entry point — delegates to the shared SFT recipe with the VQA mapping."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    # Implementation mirrors captioning/qlora/train.py with build_conversation above
    # and dataset "HuggingFaceM4/VQAv2". Kept separate for clarity of diffs.
    raise SystemExit(
        "Wire this after Module 2 works end-to-end: copy the trainer body from "
        f"captioning/qlora/train.py, swap the dataset + mapping. Config loaded OK: {cfg['model_id']}"
    )


if __name__ == "__main__":
    main()
