"""End-to-end tests of the real training code and the real predictor.

Downloading DistilBERT (~260 MB) in every test run would be slow, so these
tests build a *tiny* randomly-initialised DistilBERT with the same
architecture. It's useless as a classifier, but it exercises exactly the same
code paths: tokenising, the training loop, saving, reloading and predicting.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from app.predictor import DistilBertPredictor  # noqa: E402
from ml.train import parse_args, train  # noqa: E402
from scripts.generate_sample_data import generate  # noqa: E402


@pytest.fixture(scope="module")
def tiny_base_model(tmp_path_factory):
    """A 2-layer DistilBERT + WordPiece tokenizer saved to a local folder."""
    path = tmp_path_factory.mktemp("tiny-distilbert")
    words = set()
    for row in generate(300, seed=3):
        words.update(re.findall(r"\w+|[^\w\s]", row["text"].lower()))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[email]", "[url]", "[ref]", "[num]"] + sorted(words)
    (path / "vocab.txt").write_text("\n".join(vocab))

    tokenizer = transformers.DistilBertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(path)
    config = transformers.DistilBertConfig(vocab_size=len(vocab), dim=32, hidden_dim=64, n_layers=2, n_heads=2, max_position_embeddings=128)
    transformers.DistilBertModel(config).save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def trained_dir(tiny_base_model, tmp_path_factory):
    data = tmp_path_factory.mktemp("data") / "tickets.csv"
    pd.DataFrame(generate(240, seed=5)).to_csv(data, index=False)
    out = tmp_path_factory.mktemp("model")
    args = parse_args(
        ["--data", str(data), "--model-name", str(tiny_base_model), "--output-dir", str(out),
         "--epochs", "2", "--batch-size", "16", "--max-length", "48", "--lr", "1e-3"]
    )
    train(args)
    return out


def test_training_writes_model_and_reports(trained_dir):
    for name in ("config.json", "tokenizer.json", "metrics.json", "classification_report.txt", "confusion_matrix.csv", "labels.json"):
        assert (trained_dir / name).exists(), name
    metrics = json.loads((trained_dir / "metrics.json").read_text())
    assert metrics["split"] == "test"
    assert {"precision", "recall", "f1"} <= set(metrics["macro"])
    assert len(metrics["history"]) >= 1
    assert len(metrics["per_class"]) == 6


def test_predictor_loads_and_predicts(trained_dir):
    p = DistilBertPredictor(trained_dir, version="tiny")
    results = p.predict(["I was charged twice", "my parcel never arrived!!!"])
    assert len(results) == 2
    for r in results:
        assert r.label in p.labels
        assert 0 <= r.confidence <= 1
        assert abs(sum(r.scores.values()) - 1) < 0.01
    assert p.metrics is not None


def test_predictor_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DistilBertPredictor(tmp_path)
