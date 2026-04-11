import pandera
from pandera.pandas import Column, DataFrameSchema
from pydantic import BaseModel, field_validator

DATASET_SCHEMA = DataFrameSchema(
    {
        "sample_id": Column(str),
        "text": Column(str),
        "label": Column(int, pandera.Check.isin([0, 1, 2])),
        "source_dataset": Column(str),
        "source_type": Column(str),
        "language": Column(str),
        "is_multiturn": Column(bool),
    }
)


class SampleRecord(BaseModel):
    sample_id: str
    text: str
    label: int
    source_dataset: str
    source_type: str
    language: str
    is_multiturn: bool

    @field_validator("label")
    @classmethod
    def label_must_be_valid(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError(
                "label must be 0 (safe), 1 (jailbreak), or 2 (indirect_injection)"
            )
        return v

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v
