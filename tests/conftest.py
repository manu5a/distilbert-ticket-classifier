"""Shared test fixtures.

The API tests use a tiny keyword-based FakePredictor and an in-memory SQLite
database, so they run in seconds and don't need the trained model, a GPU or
PostgreSQL. The real model is tested separately (see test_predictor.py).
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import Base
from app.main import create_app
from app.predictor import Prediction
from ml.preprocess import clean_text

LABELS = ["account_access", "billing", "general_inquiry", "refund_request", "shipping_delivery", "technical_issue"]
KEYWORDS = {
    "billing": ["charged", "invoice", "payment"],
    "refund_request": ["refund"],
    "account_access": ["log in", "password", "locked"],
    "shipping_delivery": ["parcel", "delivery", "order hasn't arrived"],
    "technical_issue": ["crash", "error", "slow"],
}


class FakePredictor:
    labels = LABELS
    version = "test-fake"
    metrics = {"accuracy": 0.9, "macro": {"precision": 0.9, "recall": 0.9, "f1": 0.9}}

    def predict(self, texts):
        out = []
        for raw in texts:
            text = clean_text(raw).lower()
            label = next((lbl for lbl, words in KEYWORDS.items() if any(w in text for w in words)), "general_inquiry")
            confidence = 0.95 if label != "general_inquiry" else 0.4  # low confidence -> needs_review
            scores = {lbl: (confidence if lbl == label else round((1 - confidence) / 5, 4)) for lbl in LABELS}
            out.append(Prediction(label=label, confidence=confidence, scores=scores, cleaned_text=clean_text(raw)))
        return out


@pytest.fixture
def settings() -> Settings:
    # CI sets TEST_DATABASE_URL to a real PostgreSQL; locally we default to in-memory SQLite.
    db_url = os.environ.get("TEST_DATABASE_URL", "sqlite://")
    return Settings(database_url=db_url, model_dir="does-not-exist", max_text_length=500, max_batch_size=5, review_threshold=0.6)


def _fresh_tables(app) -> None:
    """Start every test with empty tables (matters when using a shared PostgreSQL)."""
    Base.metadata.drop_all(app.state.engine)
    Base.metadata.create_all(app.state.engine)


@pytest.fixture
def client(settings):
    app = create_app(settings=settings, predictor=FakePredictor())
    with TestClient(app) as c:  # "with" runs the startup code (DB tables etc.)
        _fresh_tables(app)
        yield c


@pytest.fixture
def client_without_model(settings):
    app = create_app(settings=settings)  # model_dir doesn't exist -> model fails to load
    with TestClient(app) as c:
        _fresh_tables(app)
        yield c
