from typing import Optional

from pydantic import BaseModel, field_validator


class ClassifyRequest(BaseModel):
    user_prompt: str
    external_context: Optional[str] = None
    source_type: str = "user_input"
    conversation_history: Optional[list[str]] = None

    @field_validator("user_prompt")
    @classmethod
    def user_prompt_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_prompt must not be empty")
        return v

    @field_validator("source_type")
    @classmethod
    def source_type_must_be_valid(cls, v: str) -> str:
        allowed = {"user_input", "external_doc", "api_call", "system_prompt"}
        if v not in allowed:
            raise ValueError(f"source_type must be one of {allowed}")
        return v


class ClassifyResponse(BaseModel):
    label: str
    risk_scores: dict[str, float]
    decision: str
    confidence: float
    reason_tags: list[str]
    attack_type: Optional[str] = None
    stage_used: str
    similarity_score: Optional[float] = None
    perplexity_score: Optional[float] = None
    token_attributions: Optional[list[dict[str, float]]] = None
