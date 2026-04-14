"""Stage A training pipeline: ModernBERT + LoRA.

Run locally for smoke tests; full training runs on Kaggle T4 on Day 6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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
    """Train ModernBERT + LoRA Stage A classifier (stub)."""
    raise NotImplementedError


if __name__ == "__main__":
    train_stage_a(load_config())
