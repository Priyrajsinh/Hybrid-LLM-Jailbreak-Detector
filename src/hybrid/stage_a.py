"""Stage A: fast ModernBERT-base + LoRA classifier."""

from typing import Any, Optional

from src.logger import get_logger

_LABEL_NAMES = ["safe", "jailbreak", "indirect_injection"]


class StageAClassifier:
    """ModernBERT-base + LoRA adapter for 3-class classification.

    Model + tokenizer are lazy-loaded on first classify() so tests can
    inject fakes without touching the Hugging Face cache.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        stage_cfg: dict[str, Any] = config["model"]["stage_a"]
        self._model_name: str = str(stage_cfg["model_name"])
        self._max_length: int = int(stage_cfg["max_length"])
        self._num_labels: int = int(stage_cfg.get("num_labels", 3))
        self._model = model
        self._tokenizer = tokenizer
        self._logger = get_logger(__name__)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        # pragma: no cover - heavy model download path
        from transformers import (  # pragma: no cover
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(  # pragma: no cover
            self._model_name
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(  # noqa
            self._model_name, num_labels=self._num_labels
        )

    def classify(self, text: str) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        # C32: log a warning if we had to truncate a long prompt.
        if self._rough_token_estimate(text) > self._max_length:
            self._logger.warning(
                "stage_a_truncation",
                extra={
                    "text_len_chars": len(text),
                    "max_length": self._max_length,
                },
            )

        import torch

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits
        probs_tensor = torch.softmax(logits, dim=-1).squeeze(0)
        probs = probs_tensor.tolist()
        label = int(torch.argmax(logits, dim=-1).item())

        probabilities: dict[str, float] = {
            _LABEL_NAMES[i]: float(probs[i]) for i in range(self._num_labels)
        }
        return {
            "label": label,
            "confidence": float(max(probs)),
            "probabilities": probabilities,
        }

    def classify_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        return [self.classify(t) for t in texts]

    @staticmethod
    def _rough_token_estimate(text: str) -> int:
        # ModernBERT uses ~4 chars/token on English — close enough to pre-warn
        # without paying for the real tokenizer.
        return max(1, len(text) // 4)
