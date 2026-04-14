"""Stage A training pipeline: ModernBERT + LoRA.

Run locally for smoke tests; full training runs on Kaggle T4 on Day 6.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.logger import get_logger

logger = get_logger(__name__)

ADAPTER_DIR = Path("models/stage_a_adapter")
MERGED_DIR = Path("models/stage_a_merged")


def _compute_class_weights(labels: np.ndarray, num_labels: int) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    weights = labels.shape[0] / (num_labels * counts)
    return weights.astype(np.float32)  # type: ignore[no-any-return]


def _tokenize_dataset(
    tokenizer: Any,
    texts: list[str],
    labels: list[int],
    max_length: int,
) -> Any:
    import torch
    from torch.utils.data import Dataset

    over_limit = 0
    for t in texts:
        enc = tokenizer(t, truncation=False, add_special_tokens=True)
        if len(enc["input_ids"]) > max_length:
            over_limit += 1
    if over_limit:
        logger.warning(
            "truncating samples exceeding max_length",
            extra={"n_over": over_limit, "max_length": max_length},
        )

    class _DS(Dataset):  # type: ignore[type-arg,misc]
        def __init__(self) -> None:
            self.enc = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return int(self.labels.shape[0])

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return {
                "input_ids": self.enc["input_ids"][idx],
                "attention_mask": self.enc["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    return _DS()


def train_stage_a(config: dict[str, Any]) -> None:
    """Train ModernBERT + LoRA Stage A classifier.

    Loads train/val parquet, fits LoRA adapters, logs to MLflow,
    and saves adapter + merged checkpoint.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    stage_a = config["model"]["stage_a"]
    model_name = stage_a["model_name"]
    num_labels = int(stage_a.get("num_labels", 3))
    max_length = int(stage_a.get("max_length", 2048))

    data_dir = Path(config.get("data", {}).get("processed_dir", "data/processed"))
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    logger.info(
        "loaded splits",
        extra={"n_train": len(train_df), "n_val": len(val_df)},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _train_ds = _tokenize_dataset(
        tokenizer,
        train_df["text"].tolist(),
        train_df["label"].astype(int).tolist(),
        max_length,
    )
    _val_ds = _tokenize_dataset(
        tokenizer,
        val_df["text"].tolist(),
        val_df["label"].astype(int).tolist(),
        max_length,
    )

    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(stage_a["lora_r"]),
        lora_alpha=int(stage_a["lora_alpha"]),
        lora_dropout=float(stage_a.get("lora_dropout", 0.1)),
        target_modules=list(stage_a["target_modules"]),
        bias="none",
    )
    model = get_peft_model(base, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "trainable parameters",
        extra={"trainable": trainable, "total": total, "pct": trainable / total},
    )
    _ = (os, _train_ds, _val_ds, num_labels, model)
    raise NotImplementedError("training loop wired in next commit")


if __name__ == "__main__":
    train_stage_a(load_config())
