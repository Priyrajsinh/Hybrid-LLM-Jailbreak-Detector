"""Hybrid detection pipeline + GPT-2 perplexity gate."""

from typing import Any, Optional

from src.logger import get_logger

# Lazy-loaded GPT-2 globals. Tests monkeypatch these module-level names
# with fakes so no network download happens during CI.
_PERPLEXITY_MODEL: Optional[Any] = None
_PERPLEXITY_TOKENIZER: Optional[Any] = None


def _load_perplexity_model() -> None:  # pragma: no cover - heavy download path
    global _PERPLEXITY_MODEL, _PERPLEXITY_TOKENIZER
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    _PERPLEXITY_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
    _PERPLEXITY_MODEL = GPT2LMHeadModel.from_pretrained("gpt2")


def perplexity_gate(text: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run BEFORE Stage A on EVERY request including batch (C35).

    Returns {perplexity, blocked, reason_tag}. The caller (HybridPipeline)
    is responsible for attaching reason_tag to the final ClassifyResponse.
    Score is logged on every call regardless of decision.
    """
    logger = get_logger(__name__)
    threshold = float(config["model"].get("perplexity_threshold", 500.0))

    if _PERPLEXITY_MODEL is None or _PERPLEXITY_TOKENIZER is None:
        _load_perplexity_model()

    tokenizer = _PERPLEXITY_TOKENIZER
    model = _PERPLEXITY_MODEL
    assert tokenizer is not None
    assert model is not None

    import torch

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_ids = encoded["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    ppl = float(torch.exp(loss).item())
    blocked = ppl > threshold
    reason_tag: Optional[str] = "perplexity_anomaly" if blocked else None

    logger.info(
        "perplexity_gate",
        extra={
            "perplexity": ppl,
            "threshold": threshold,
            "blocked": blocked,
        },
    )
    return {"perplexity": ppl, "blocked": blocked, "reason_tag": reason_tag}


class HybridPipeline:
    """Full 5-layer pipeline orchestration. Placeholder until commit 7."""

    pass
