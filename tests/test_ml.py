"""Tests for text cleaning, dataset preparation and metrics."""

import json

import pandas as pd
import pytest

from ml.data import describe, load_dataset, split_dataset
from ml.evaluate import compute_metrics, most_confused_pairs, write_reports
from ml.preprocess import clean_text
from scripts.generate_sample_data import generate


# --- preprocessing -----------------------------------------------------------
def test_clean_text_masks_personal_data():
    out = clean_text("Email me at jo.bloggs@mail.ie or see https://x.com/a ref #12345, order 98765432")
    assert "[EMAIL]" in out and "jo.bloggs" not in out
    assert "[URL]" in out and "https" not in out
    assert "[REF]" in out
    assert "[NUM]" in out


def test_clean_text_normalises_whitespace_and_punctuation():
    assert clean_text("  Help!!!   now??  ") == "Help! now?"


def test_clean_text_rejects_non_strings():
    with pytest.raises(TypeError):
        clean_text(None)  # type: ignore[arg-type]


# --- dataset -----------------------------------------------------------------
@pytest.fixture
def csv_path(tmp_path):
    rows = generate(600, seed=1)
    rows += [{"ticket_id": "X1", "text": None, "label": "billing"}, {"ticket_id": "X2", "text": "ok", "label": "billing"}]
    rows += [rows[0]]  # exact duplicate
    path = tmp_path / "tickets.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_dataset_cleans_and_dedupes(csv_path):
    df = load_dataset(csv_path)
    assert list(df.columns) == ["text", "label"]
    assert df["text"].isna().sum() == 0
    assert (df["text"].str.len() >= 5).all()
    assert not df["text"].duplicated().any()


def test_load_dataset_requires_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"body": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_label_names_are_normalised(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"text": ["first ticket text", "second ticket text"], "label": ["Technical Issue", "billing"]}).to_csv(path, index=False)
    assert set(load_dataset(path)["label"]) == {"technical_issue", "billing"}


def test_split_is_stratified_and_disjoint(csv_path):
    df = load_dataset(csv_path)
    s = split_dataset(df, val_size=0.1, test_size=0.1, seed=0)
    assert len(s.train) + len(s.val) + len(s.test) == len(df)
    assert set(s.train["text"]).isdisjoint(s.test["text"])
    for part in (s.train, s.val, s.test):
        assert set(part["label"]) == set(df["label"])  # every label in every split
    assert s.id2label[s.label2id["billing"]] == "billing"


def test_describe(csv_path):
    d = describe(load_dataset(csv_path))
    assert d["rows"] > 0 and d["avg_words"] > 0 and len(d["label_counts"]) == 6


# --- metrics -----------------------------------------------------------------
def test_compute_metrics_perfect():
    m = compute_metrics([0, 1, 2, 1], [0, 1, 2, 1], ["a", "b", "c"])
    assert m["accuracy"] == 1.0 and m["macro"]["f1"] == 1.0
    assert m["per_class"]["b"]["support"] == 2


def test_compute_metrics_known_values():
    # class a: TP=1, FP=1, FN=1 -> precision 0.5, recall 0.5
    m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 0], ["a", "b"])
    assert m["per_class"]["a"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    assert m["accuracy"] == 0.5


def test_most_confused_pairs():
    pairs = most_confused_pairs([0, 0, 0, 1], [1, 1, 0, 1], ["a", "b"])
    assert pairs[0] == {"true": "a", "predicted": "b", "count": 2}


def test_write_reports(tmp_path):
    write_reports(tmp_path, [0, 1, 1], [0, 1, 0], ["a", "b"], extra={"note": "x"})
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["note"] == "x" and "most_confused" in metrics
    assert (tmp_path / "classification_report.txt").exists()
    assert (tmp_path / "confusion_matrix.csv").read_text().startswith("true\\predicted,a,b")
