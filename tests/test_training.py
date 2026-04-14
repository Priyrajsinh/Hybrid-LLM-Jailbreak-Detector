"""Tests for Stage A training and threshold calibration.

Training is mocked end-to-end — we exercise orchestration, not real fits.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.training.train import ADAPTER_DIR, MERGED_DIR, train_stage_a


def _tiny_config() -> dict:
    return {
        "model": {
            "stage_a": {
                "model_name": "answerdotai/ModernBERT-base",
                "max_length": 32,
                "num_labels": 3,
                "lora_r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
            }
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1e-4,
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "seed": 0,
            "early_stopping_patience": 1,
        },
        "data": {"processed_dir": "data/processed_tiny_test"},
        "mlflow": {"experiment_name": "p1-test"},
    }


@pytest.fixture
def tiny_split(tmp_path: Path) -> Path:
    out = tmp_path / "processed"
    out.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(6)],
            "text": ["hi"] * 6,
            "label": [0, 1, 2, 0, 1, 2],
            "source_dataset": ["x"] * 6,
            "source_type": ["user"] * 6,
            "language": ["en"] * 6,
            "is_multiturn": [False] * 6,
        }
    )
    df.to_parquet(out / "train.parquet")
    df.to_parquet(out / "val.parquet")
    return out


def _fake_model_factory() -> MagicMock:
    import torch

    model = MagicMock()
    model.parameters.return_value = [torch.zeros(1, requires_grad=True)]

    def fwd(**kwargs: object) -> MagicMock:
        bs = kwargs["input_ids"].shape[0]  # type: ignore[union-attr]
        out = MagicMock()
        out.logits = torch.zeros(bs, 3, requires_grad=True)
        return out

    model.side_effect = fwd
    merged = MagicMock()
    model.merge_and_unload.return_value = merged
    model.save_pretrained = MagicMock()
    merged.save_pretrained = MagicMock()
    return model


def test_training_creates_adapter(
    tiny_split: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _tiny_config()
    cfg["data"]["processed_dir"] = str(tiny_split)

    tok = MagicMock()
    import torch

    def _tok_call(texts: object, **kwargs: object) -> dict[str, object] | MagicMock:
        if isinstance(texts, list):
            n = len(texts)
            return {
                "input_ids": torch.zeros(n, 8, dtype=torch.long),
                "attention_mask": torch.ones(n, 8, dtype=torch.long),
            }
        return {"input_ids": [0, 1, 2]}

    tok.side_effect = _tok_call

    fake_model = _fake_model_factory()
    monkeypatch.setattr(
        "src.training.train.ADAPTER_DIR", tmp_path / "adapter", raising=True
    )
    monkeypatch.setattr(
        "src.training.train.MERGED_DIR", tmp_path / "merged", raising=True
    )

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tok),
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=MagicMock(),
        ),
        patch("peft.get_peft_model", return_value=fake_model),
        patch("transformers.get_linear_schedule_with_warmup"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run"),
        patch("mlflow.log_params"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_artifacts"),
    ):
        train_stage_a(cfg)

    fake_model.save_pretrained.assert_called_once()
    fake_model.merge_and_unload.assert_called_once()
    assert ADAPTER_DIR.name == "stage_a_adapter"
    assert MERGED_DIR.name == "stage_a_merged"


def test_training_logs_mlflow(
    tiny_split: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _tiny_config()
    cfg["data"]["processed_dir"] = str(tiny_split)
    monkeypatch.setattr("src.training.train.ADAPTER_DIR", tmp_path / "a")
    monkeypatch.setattr("src.training.train.MERGED_DIR", tmp_path / "m")

    tok = MagicMock()
    import torch

    def _tok_call(texts: object, **kwargs: object) -> dict[str, object]:
        if isinstance(texts, list):
            n = len(texts)
            return {
                "input_ids": torch.zeros(n, 8, dtype=torch.long),
                "attention_mask": torch.ones(n, 8, dtype=torch.long),
            }
        return {"input_ids": [0, 1]}

    tok.side_effect = _tok_call

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tok),
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=MagicMock(),
        ),
        patch("peft.get_peft_model", return_value=_fake_model_factory()),
        patch("transformers.get_linear_schedule_with_warmup"),
        patch("mlflow.set_experiment") as m_exp,
        patch("mlflow.start_run"),
        patch("mlflow.log_params") as m_params,
        patch("mlflow.log_metric") as m_metric,
        patch("mlflow.log_artifacts"),
    ):
        train_stage_a(cfg)

    m_exp.assert_called_once()
    m_params.assert_called_once()
    assert m_metric.called
