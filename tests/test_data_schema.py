import logging

import pytest

from src.api.schemas import ClassifyRequest, ClassifyResponse
from src.config import load_config
from src.logger import get_logger


def test_config_loads():
    config = load_config("config/config.yaml")
    assert isinstance(config, dict)
    assert "model" in config
    assert config["model"]["perplexity_threshold"] == 500.0


def test_get_logger_returns_logger():
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)


def test_get_logger_no_duplicate_handlers():
    logger1 = get_logger("dedup_test")
    handler_count_1 = len(logger1.handlers)
    logger2 = get_logger("dedup_test")
    assert len(logger2.handlers) == handler_count_1


def test_classify_request_valid():
    req = ClassifyRequest(user_prompt="Hello, how are you?")
    assert req.user_prompt == "Hello, how are you?"
    assert req.source_type == "user_input"
    assert req.external_context is None
    assert req.conversation_history is None


def test_classify_request_rejects_empty_prompt():
    with pytest.raises(Exception):
        ClassifyRequest(user_prompt="   ")


def test_classify_request_rejects_missing_prompt():
    with pytest.raises(Exception):
        ClassifyRequest()  # type: ignore[call-arg]


def test_classify_request_with_all_fields():
    req = ClassifyRequest(
        user_prompt="Ignore previous instructions",
        external_context="Some external doc",
        source_type="external_doc",
        conversation_history=["Hello", "Hi there"],
    )
    assert req.source_type == "external_doc"
    assert len(req.conversation_history) == 2  # type: ignore[arg-type]


def test_classify_response_has_required_fields():
    resp = ClassifyResponse(
        label="safe",
        risk_scores={"safe": 0.95, "jailbreak": 0.03, "indirect_injection": 0.02},
        decision="allow",
        confidence=0.95,
        reason_tags=[],
        stage_used="stage_a",
    )
    assert resp.label == "safe"
    assert resp.attack_type is None
    assert resp.similarity_score is None
    assert resp.perplexity_score is None
    assert resp.token_attributions is None


def test_classify_response_with_optional_fields():
    resp = ClassifyResponse(
        label="jailbreak",
        risk_scores={"safe": 0.05, "jailbreak": 0.90, "indirect_injection": 0.05},
        decision="block",
        confidence=0.90,
        reason_tags=["role_play", "ignore_instructions"],
        attack_type="prompt_injection",
        stage_used="stage_b",
        similarity_score=0.92,
        perplexity_score=620.5,
        token_attributions=[{"ignore": 0.8, "instructions": 0.6}],
    )
    assert resp.attack_type == "prompt_injection"
    assert resp.similarity_score == 0.92
    assert resp.perplexity_score == 620.5
    assert resp.token_attributions is not None


def test_exceptions_hierarchy():
    from src.exceptions import (
        ClassificationError,
        DataLoadError,
        ModelNotFoundError,
        PolicyViolationError,
        ProjectBaseError,
    )

    assert issubclass(DataLoadError, ProjectBaseError)
    assert issubclass(ModelNotFoundError, ProjectBaseError)
    assert issubclass(ClassificationError, ProjectBaseError)
    assert issubclass(PolicyViolationError, ProjectBaseError)


def test_sample_record_valid():
    from src.data.schema import SampleRecord

    record = SampleRecord(
        sample_id="test_001",
        text="Hello world",
        label=0,
        source_dataset="safe_corpus",
        source_type="user_input",
        language="en",
        is_multiturn=False,
    )
    assert record.label == 0


def test_sample_record_invalid_label():
    from src.data.schema import SampleRecord

    with pytest.raises(Exception):
        SampleRecord(
            sample_id="test_002",
            text="Some text",
            label=99,
            source_dataset="test",
            source_type="user_input",
            language="en",
            is_multiturn=False,
        )


def test_sample_record_empty_text():
    from src.data.schema import SampleRecord

    with pytest.raises(Exception):
        SampleRecord(
            sample_id="test_003",
            text="   ",
            label=0,
            source_dataset="test",
            source_type="user_input",
            language="en",
            is_multiturn=False,
        )


def test_hybrid_stubs_importable():
    from src.hybrid.explain import TokenExplainer
    from src.hybrid.normalize import InputNormalizer
    from src.hybrid.pipeline import HybridPipeline
    from src.hybrid.policy_gate import PolicyGate
    from src.hybrid.similarity import SimilarityGate
    from src.hybrid.stage_a import StageAClassifier
    from src.hybrid.stage_b import StageBJudge

    assert StageAClassifier is not None
    assert StageBJudge is not None
    assert PolicyGate is not None
    assert HybridPipeline is not None
    assert InputNormalizer is not None
    assert SimilarityGate is not None
    assert TokenExplainer is not None


def test_api_app_health_endpoint():
    from src.api.app import health

    result = health()
    assert result == {"status": "ok"}


def test_ui_theme_get_css():
    from src.ui.theme import get_css

    css = get_css()
    assert isinstance(css, str)


def test_ui_theme_get_theme():
    from src.ui.theme import get_theme

    result = get_theme()
    assert result is None


def test_feedback_store_instantiable():
    from src.api.feedback import FeedbackStore

    store = FeedbackStore()
    assert store is not None
