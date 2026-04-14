"""Calibrate Stage A confidence thresholds for the policy gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)

THRESHOLDS_PATH = Path("models/manifests/thresholds.json")
SAFE_LABEL = 0
JAILBREAK_LABEL = 1


class _ProbaModel(Protocol):
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...


def _max_threshold_with_fpr(
    scores: np.ndarray, is_safe: np.ndarray, max_fpr: float
) -> float:
    """Lowest threshold T s.t. FPR(safe flagged) < max_fpr when score >= T."""
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_is_safe = is_safe[order]
    n_safe = max(int(is_safe.sum()), 1)
    fp = 0
    best_t = 1.0
    for i, score in enumerate(sorted_scores):
        if sorted_is_safe[i]:
            fp += 1
            if fp / n_safe >= max_fpr:
                return float(score) + 1e-6
        best_t = float(score)
    return best_t


def calibrate_thresholds(
    model: _ProbaModel, val_data: Any, config: dict[str, Any]
) -> dict[str, Any]:
    raise NotImplementedError
