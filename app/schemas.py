"""Request and response shapes. Pydantic validates every request against these."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TicketIn(BaseModel):
    text: str = Field(..., description="Raw support-ticket text", examples=["I was charged twice for my subscription"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Several tickets to classify at once")

    @field_validator("texts")
    @classmethod
    def none_blank(cls, v: list[str]) -> list[str]:
        if any(not t.strip() for t in v):
            raise ValueError("texts must not contain empty strings")
        return v


class PredictionOut(BaseModel):
    id: int
    label: str
    confidence: float
    needs_review: bool
    scores: dict[str, float]
    model_version: str
    latency_ms: float


class BatchOut(BaseModel):
    predictions: list[PredictionOut]
    count: int


class StoredPrediction(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    input_text: str
    predicted_label: str
    confidence: float
    needs_review: bool
    model_version: str
    latency_ms: float


class PredictionList(BaseModel):
    items: list[StoredPrediction]
    total: int
    limit: int
    offset: int


class Stats(BaseModel):
    total_predictions: int
    by_label: dict[str, int]
    average_confidence: float | None
    needs_review: int


class ModelInfo(BaseModel):
    loaded: bool
    version: str
    labels: list[str]
    test_metrics: dict | None = None
