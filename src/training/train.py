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
    import mlflow
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    stage_a = config["model"]["stage_a"]
    tcfg = config["training"]
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)  # nosec B615
    train_ds = _tokenize_dataset(
        tokenizer,
        train_df["text"].tolist(),
        train_df["label"].astype(int).tolist(),
        max_length,
    )
    val_ds = _tokenize_dataset(
        tokenizer,
        val_df["text"].tolist(),
        val_df["label"].astype(int).tolist(),
        max_length,
    )

    base = AutoModelForSequenceClassification.from_pretrained(  # nosec B615
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

    class_weights_np = _compute_class_weights(
        train_df["label"].astype(int).to_numpy(), num_labels
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    batch_size = int(tcfg["batch_size"])
    epochs = int(tcfg["epochs"])
    lr = float(tcfg["learning_rate"])
    weight_decay = float(tcfg["weight_decay"])
    warmup_ratio = float(tcfg["warmup_ratio"])
    patience = int(tcfg["early_stopping_patience"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "p1"))
    with mlflow.start_run(run_name="stage_a_lora"):
        mlflow.log_params(
            {
                "model_name": model_name,
                "lora_r": stage_a["lora_r"],
                "lora_alpha": stage_a["lora_alpha"],
                "max_length": max_length,
                **{f"train_{k}": v for k, v in tcfg.items()},
            }
        )
        mlflow.log_metric("trainable_params", trainable)

        best_val_loss = float("inf")
        bad_epochs = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = loss_fn(out.logits, batch["labels"])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += float(loss.item())

            model.eval()
            val_loss = 0.0
            correct = 0
            seen = 0
            with torch.no_grad():
                for batch in val_loader:
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    val_loss += float(loss_fn(out.logits, batch["labels"]).item())
                    preds = out.logits.argmax(dim=-1)
                    correct += int((preds == batch["labels"]).sum().item())
                    seen += int(batch["labels"].shape[0])

            val_acc = correct / max(seen, 1)
            logger.info(
                "epoch complete",
                extra={
                    "epoch": epoch,
                    "train_loss": epoch_loss / max(len(train_loader), 1),
                    "val_loss": val_loss / max(len(val_loader), 1),
                    "val_acc": val_acc,
                },
            )
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            if os.environ.get("ENABLE_WEAVE") == "1":
                try:
                    import weave

                    weave.log(
                        {
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                        }
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("weave log failed", extra={"err": str(exc)})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    logger.info(
                        "early stopping", extra={"epoch": epoch, "patience": patience}
                    )
                    break

        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ADAPTER_DIR))
        logger.info("saved adapter", extra={"path": str(ADAPTER_DIR)})

        merged = model.merge_and_unload()
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(MERGED_DIR))
        tokenizer.save_pretrained(str(MERGED_DIR))
        logger.info("saved merged model", extra={"path": str(MERGED_DIR)})

        mlflow.log_artifacts(str(ADAPTER_DIR), artifact_path="stage_a_adapter")


if __name__ == "__main__":
    train_stage_a(load_config())
