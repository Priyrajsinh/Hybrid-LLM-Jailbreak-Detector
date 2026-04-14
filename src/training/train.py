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


def train_stage_a(config: dict[str, Any]) -> None:
    """Train ModernBERT + LoRA Stage A classifier (stub)."""
    raise NotImplementedError


if __name__ == "__main__":
    train_stage_a(load_config())
