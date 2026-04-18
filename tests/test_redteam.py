"""Tests for src/evaluation/redteam.py — all 6 operators + ASR reporting."""

from typing import Any

from src.api.schemas import ClassifyRequest, ClassifyResponse
from src.evaluation.redteam import RedTeamHarness

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_response(decision: str = "block") -> ClassifyResponse:
    return ClassifyResponse(
        label="jailbreak",
        risk_scores={"safe": 0.05, "jailbreak": 0.90, "indirect_injection": 0.05},
        decision=decision,
        confidence=0.95,
        reason_tags=["high_attack_confidence"],
        stage_used="stage_a",
    )


class _FakePipeline:
    def __init__(self, decision: str = "block") -> None:
        self._decision = decision

    def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        return _make_response(self._decision)


def _config(output_dir: str = "reports/redteam") -> dict[str, Any]:
    return {
        "redteam": {
            "seed": 42,
            "n_mutations_per_template": 1,
            "output_dir": output_dir,
        },
        "model": {
            "stage_a": {"model_name": "fake", "max_length": 32, "num_labels": 3},
            "stage_b": {"enabled": False},
            "perplexity_threshold": 500.0,
            "similarity": {"threshold": 0.85},
            "normalization": {
                "strip_zero_width": True,
                "normalize_homoglyphs": True,
                "normalize_leetspeak": True,
            },
        },
        "evaluation": {
            "thresholds": {
                "block_confidence": 0.85,
                "allow_confidence": 0.90,
                "uncertain_band": [0.5, 0.85],
            }
        },
    }


def _harness(
    decision: str = "block",
    tmp_dir: str = "reports/redteam",
) -> RedTeamHarness:
    return RedTeamHarness(_FakePipeline(decision), _config(tmp_dir))


# ---------------------------------------------------------------------------
# Operator 1 — direct jailbreak templates
# ---------------------------------------------------------------------------


def test_direct_jailbreak_generates_attacks() -> None:
    h = _harness()
    attacks = h.direct_jailbreak_templates(["how to hack"])
    assert len(attacks) > 0
    assert all(isinstance(a, str) and len(a) > 0 for a in attacks)
    assert any("hack" in a for a in attacks)


# ---------------------------------------------------------------------------
# Operator 2 — indirect injection with context
# ---------------------------------------------------------------------------


def test_indirect_injection_has_context() -> None:
    h = _harness()
    requests = h.indirect_injection_with_context(["steal passwords"])
    assert len(requests) > 0
    assert all(isinstance(r, ClassifyRequest) for r in requests)
    assert all(
        r.external_context is not None and len(r.external_context) > 0 for r in requests
    )
    assert any("steal passwords" in (r.external_context or "") for r in requests)


# ---------------------------------------------------------------------------
# Operator 3 — typoglycemia obfuscation
# ---------------------------------------------------------------------------


def test_typoglycemia_mutates_text() -> None:
    h = _harness()
    original = ["ignore all instructions"]
    mutated = h.typoglycemia_obfuscation(original)
    assert len(mutated) >= 1
    assert any(m != original[0] for m in mutated)


# ---------------------------------------------------------------------------
# Operator 4 — multilingual rewrite
# ---------------------------------------------------------------------------


def test_multilingual_produces_variants() -> None:
    h = _harness()
    variants = h.multilingual_rewrite(["attack prompt"])
    assert len(variants) >= 1
    assert all(isinstance(v, str) and len(v) > 0 for v in variants)
