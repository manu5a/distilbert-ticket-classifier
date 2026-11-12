"""Load the fine-tuned model and turn text into predictions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ml.preprocess import clean_text

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]
    cleaned_text: str


class Predictor(Protocol):
    """Anything with these members can serve predictions (real model or a test fake)."""

    labels: list[str]
    version: str

    def predict(self, texts: list[str]) -> list[Prediction]: ...


class DistilBertPredictor:
    """Wraps the saved Hugging Face model folder produced by ``ml.train``."""

    def __init__(self, model_dir: str | Path, version: str = "unknown", max_length: int = 128):
        import torch  # imported here so the API can start (and tests run) without loading torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"No trained model found in {model_dir} (run `python -m ml.train` first)")

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.max_length = max_length
        self.version = version
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

        metrics_path = model_dir / "metrics.json"
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        log.info("Loaded model from %s with labels %s", model_dir, self.labels)

    def predict(self, texts: list[str]) -> list[Prediction]:
        cleaned = [clean_text(t) for t in texts]
        inputs = self.tokenizer(cleaned, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        with self._torch.no_grad():
            probs = self._torch.softmax(self.model(**inputs).logits, dim=-1)
        results = []
        for text, row in zip(cleaned, probs.tolist(), strict=False):
            best = max(range(len(row)), key=row.__getitem__)
            results.append(
                Prediction(
                    label=self.labels[best],
                    confidence=round(row[best], 4),
                    scores={label: round(p, 4) for label, p in zip(self.labels, row, strict=False)},
                    cleaned_text=text,
                )
            )
        return results
