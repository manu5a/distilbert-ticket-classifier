"""Load, validate, clean and split the labelled ticket dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.preprocess import clean_text

REQUIRED_COLUMNS = {"text", "label"}
MIN_TEXT_LENGTH = 5


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    label2id: dict[str, int]

    @property
    def id2label(self) -> dict[int, str]:
        return {i: label for label, i in self.label2id.items()}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read a CSV and return a cleaned DataFrame with ``text`` and ``label``.

    Cleaning steps:
    1. check required columns exist
    2. drop rows with missing text/label
    3. apply ``clean_text`` and drop texts that are too short
    4. normalise label names (lowercase, underscores)
    5. drop exact duplicate texts (keeps the first)
    """
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df["label"] = df["label"].astype(str).str.strip().str.lower().str.replace(r"[\s-]+", "_", regex=True)
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable rows left after cleaning")
    return df[["text", "label"]]


def split_dataset(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1, seed: int = 42) -> DatasetSplits:
    """Stratified train/validation/test split so every label appears in each part."""
    labels = sorted(df["label"].unique())
    if len(labels) < 2:
        raise ValueError("Need at least two labels to train a classifier")
    label2id = {label: i for i, label in enumerate(labels)}

    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val, stratify=train_val["label"], random_state=seed)
    return DatasetSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        label2id=label2id,
    )


def describe(df: pd.DataFrame) -> dict:
    """Small summary used in the training report."""
    return {
        "rows": int(len(df)),
        "label_counts": {k: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "avg_words": round(float(df["text"].str.split().str.len().mean()), 1),
    }


def save_label_map(label2id: dict[str, int], path: str | Path) -> None:
    Path(path).write_text(json.dumps(label2id, indent=2))
