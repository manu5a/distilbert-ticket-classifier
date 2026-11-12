"""Fine-tune DistilBERT to classify support tickets.

A plain PyTorch training loop (no ``Trainer`` magic) so every step is visible:
tokenise -> batches -> forward pass -> loss -> backward -> optimiser step.

Usage:
    python -m ml.train --data data/sample_tickets.csv --output-dir models/ticket-classifier

Runs on CPU (slow, ~15-30 min for the sample data) or a GPU (a few minutes,
e.g. free Google Colab). The output folder is what the API loads.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ml.data import describe, load_dataset, split_dataset
from ml.evaluate import compute_metrics, write_reports

log = logging.getLogger("train")


class TicketDataset(Dataset):
    """Wraps tokenised tickets so a DataLoader can batch them."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_loader(model, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int], float]:
    """Run the model over a DataLoader; return (true labels, predictions, mean loss)."""
    model.eval()
    y_true, y_pred, losses = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(out.loss.item())
        y_pred.extend(out.logits.argmax(dim=-1).tolist())
        y_true.extend(batch["labels"].tolist())
    return y_true, y_pred, float(np.mean(losses)) if losses else 0.0


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = pick_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Device: %s", device)

    # 1. Data -------------------------------------------------------------
    df = load_dataset(args.data)
    splits = split_dataset(df, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    label_names = [splits.id2label[i] for i in range(len(splits.label2id))]
    log.info("Dataset: %s", describe(df))
    log.info("Split sizes: train=%d val=%d test=%d", len(splits.train), len(splits.val), len(splits.test))

    # 2. Tokeniser + model --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label=splits.id2label,
        label2id=splits.label2id,
    ).to(device)

    def make_loader(frame, shuffle: bool) -> DataLoader:
        ds = TicketDataset(frame["text"].tolist(), frame["label"].map(splits.label2id).tolist(), tokenizer, args.max_length)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(splits.train, shuffle=True)
    val_loader = make_loader(splits.val, shuffle=False)
    test_loader = make_loader(splits.test, shuffle=False)

    # 3. Optimiser + learning-rate schedule ---------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    # 4. Training loop with early stopping on validation macro-F1 ----------
    best_f1, epochs_without_improvement, history = -1.0, 0, []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step % args.log_every == 0:
                log.info("epoch %d step %d/%d loss %.4f", epoch, step, len(train_loader), running_loss / step)

        y_true, y_pred, val_loss = predict_loader(model, val_loader, device)
        val_metrics = compute_metrics(y_true, y_pred, label_names)
        record = {
            "epoch": epoch,
            "train_loss": round(running_loss / len(train_loader), 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro"]["f1"],
        }
        history.append(record)
        log.info("Epoch %d: %s", epoch, record)

        if val_metrics["macro"]["f1"] > best_f1:
            best_f1, epochs_without_improvement = val_metrics["macro"]["f1"], 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            log.info("New best model saved (val macro-F1 %.4f)", best_f1)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                log.info("Early stopping: no improvement for %d epoch(s)", args.patience)
                break

    # 5. Final evaluation of the *best* checkpoint on the held-out test set --
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    y_true, y_pred, test_loss = predict_loader(model, test_loader, device)
    extra = {
        "split": "test",
        "test_loss": round(test_loss, 4),
        "base_model": args.model_name,
        "dataset": describe(df),
        "history": history,
        "training_seconds": round(time.time() - start, 1),
        "hyperparameters": {
            k: getattr(args, k) for k in ("epochs", "batch_size", "lr", "max_length", "weight_decay", "warmup_ratio", "seed")
        },
    }
    metrics = write_reports(output_dir, y_true, y_pred, label_names, extra=extra)
    (output_dir / "labels.json").write_text(json.dumps(label_names, indent=2))
    log.info("Test accuracy %.4f | macro-F1 %.4f", metrics["accuracy"], metrics["macro"]["f1"])
    log.info("Model and reports saved to %s", output_dir)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT for support-ticket classification")
    p.add_argument("--data", default="data/sample_tickets.csv", help="CSV with 'text' and 'label' columns")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--output-dir", default="models/ticket-classifier")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2, help="epochs without val improvement before stopping")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(parse_args())
