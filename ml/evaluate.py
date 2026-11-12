"""Classification metrics: precision, recall, F1 and a confusion matrix."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], label_names: Sequence[str]) -> dict:
    """Return overall and per-class metrics.

    * precision - of the tickets we put in class X, how many really were X
    * recall    - of the tickets that really were X, how many we found
    * F1        - harmonic mean of the two (one number to compare models)
    * macro avg - every class counts equally (good for unbalanced data)
    * weighted  - classes count by how many tickets they have
    """
    labels = list(range(len(label_names)))
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro": {"precision": round(float(macro[0]), 4), "recall": round(float(macro[1]), 4), "f1": round(float(macro[2]), 4)},
        "weighted": {"precision": round(float(weighted[0]), 4), "recall": round(float(weighted[1]), 4), "f1": round(float(weighted[2]), 4)},
        "per_class": {
            name: {
                "precision": round(float(per_class[0][i]), 4),
                "recall": round(float(per_class[1][i]), 4),
                "f1": round(float(per_class[2][i]), 4),
                "support": int(per_class[3][i]),
            }
            for i, name in enumerate(label_names)
        },
    }


def most_confused_pairs(y_true: Sequence[int], y_pred: Sequence[int], label_names: Sequence[str], top_k: int = 5) -> list[dict]:
    """Which classes does the model mix up most? Useful for error analysis."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    pairs = [
        {"true": label_names[i], "predicted": label_names[j], "count": int(cm[i][j])}
        for i in range(len(label_names))
        for j in range(len(label_names))
        if i != j and cm[i][j] > 0
    ]
    return sorted(pairs, key=lambda p: p["count"], reverse=True)[:top_k]


def write_reports(out_dir: str | Path, y_true: Sequence[int], y_pred: Sequence[int], label_names: Sequence[str], extra: dict | None = None) -> dict:
    """Save metrics.json, classification_report.txt and confusion_matrix.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred, label_names)
    metrics["most_confused"] = most_confused_pairs(y_true, y_pred, label_names)
    if extra:
        metrics.update(extra)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    report = classification_report(
        y_true, y_pred, labels=list(range(len(label_names))), target_names=list(label_names), digits=4, zero_division=0
    )
    (out / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    lines = ["true\\predicted," + ",".join(label_names)]
    lines += [f"{label_names[i]}," + ",".join(str(int(v)) for v in row) for i, row in enumerate(cm)]
    (out / "confusion_matrix.csv").write_text("\n".join(lines) + "\n")
    return metrics
