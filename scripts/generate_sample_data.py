"""Generate a synthetic, labelled support-ticket dataset.

The real project idea is to train on a company's own support tickets. Those
can't be published, so this script creates a realistic stand-in dataset that
anyone can reproduce. Swap in your own CSV (columns: ``text``, ``label``) at
any time — nothing else in the project needs to change.

Usage:
    python scripts/generate_sample_data.py --rows 1800 --out data/sample_tickets.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

PRODUCTS = ["Pro plan", "Basic plan", "mobile app", "web dashboard", "API", "desktop client", "team workspace"]
DEVICES = ["iPhone", "Android phone", "Windows laptop", "MacBook", "Chrome", "Firefox", "iPad"]
AMOUNTS = ["€9.99", "€19.99", "€49", "€120", "€14.50", "€240"]
COURIERS = ["An Post", "DPD", "DHL", "FedEx", "UPS"]
DAYS = ["2 days", "a week", "10 days", "3 weeks", "since Monday", "since last Friday"]
ERRORS = ["error 500", "a blank white screen", "'connection timed out'", "error code E-4031", "a spinning wheel", "'unexpected token'"]

TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "I was charged {amount} twice this month for the {product}. Please fix this.",
        "Why does my invoice show {amount} when my {product} should cost less?",
        "My card was billed {amount} but I cancelled the {product} last month.",
        "Can you send me a VAT invoice for my {product} subscription?",
        "The payment for {amount} failed but the money left my bank account.",
        "How do I change the credit card used for the {product}?",
        "I upgraded to the {product} and got charged {amount} straight away, is that right?",
        "Need a copy of all invoices for last year for our accounts team.",
    ],
    "technical_issue": [
        "The {product} crashes on my {device} every time I open it.",
        "I keep getting {error} when I try to save in the {product}.",
        "Since the last update the {product} is very slow on {device}.",
        "Sync between the {product} and my {device} stopped working {days} ago.",
        "Uploading files fails with {error} on the {product}.",
        "The {product} won't load on {device}, I just see {error}.",
        "Notifications stopped working on my {device} after the update.",
        "Export to PDF in the {product} produces an empty file.",
    ],
    "account_access": [
        "I can't log in to my account, it says my password is wrong.",
        "The password reset email never arrives. I've waited {days}.",
        "My account got locked after too many login attempts on my {device}.",
        "Two-factor codes are not being accepted on the {product}.",
        "I lost my phone and can't get my 2FA code, how do I get back in?",
        "How do I change the email address on my account?",
        "Someone else may have logged into my account, please help secure it.",
        "I need to add a new admin user to our {product}.",
    ],
    "shipping_delivery": [
        "My order hasn't arrived, it's been {days} and tracking shows nothing.",
        "The {courier} tracking link says delivered but I never got the parcel.",
        "Can I change the delivery address for my order before it ships?",
        "My package from {courier} arrived damaged.",
        "When will my order be dispatched? I ordered {days} ago.",
        "Do you ship to Northern Ireland and how long does it take?",
        "{courier} returned my parcel to sender, can you resend it?",
        "I received the wrong item in my delivery.",
    ],
    "refund_request": [
        "I'd like a refund for the {product}, it doesn't do what I need.",
        "Please refund the {amount} I paid, I cancelled within 14 days.",
        "I returned the item {days} ago and still have no refund.",
        "Can I get my money back for the unused months of the {product}?",
        "The product was faulty and I want a full refund of {amount}.",
        "How long does a refund take to reach my card?",
        "I was promised a refund of {amount} by your agent but nothing came.",
        "Cancel my order and refund me please.",
    ],
    "general_inquiry": [
        "Does the {product} support exporting data to Excel?",
        "What are your customer support opening hours?",
        "Is there a student discount for the {product}?",
        "Do you have a roadmap for new features in the {product}?",
        "Can the {product} be used offline on {device}?",
        "Where can I find documentation for the {product}?",
        "Do you offer training sessions for new teams?",
        "Is my data stored in the EU? We need to check GDPR compliance.",
    ],
}

OPENERS = ["", "", "Hi, ", "Hello team, ", "Hi there - ", "URGENT: ", "Good morning, "]
CLOSERS = ["", "", " Thanks.", " Thanks in advance!", " Regards, Customer", " Please reply asap!!", " Ticket ref #{ref}.", " My email is user{ref}@example.com"]


def _noise(text: str, rng: random.Random) -> str:
    """Add the kind of mess real tickets have: casing, extra spaces, typos."""
    if rng.random() < 0.15:
        text = text.lower()
    if rng.random() < 0.08:
        text = text.upper()
    if rng.random() < 0.15:
        text = text.replace(" ", "  ", 1)
    if rng.random() < 0.10 and len(text) > 10:
        i = rng.randrange(1, len(text) - 1)
        text = text[:i] + text[i + 1] + text[i] + text[i + 2 :]  # swap two characters
    return text


def make_ticket(label: str, rng: random.Random) -> str:
    body = rng.choice(TEMPLATES[label]).format(
        product=rng.choice(PRODUCTS),
        device=rng.choice(DEVICES),
        amount=rng.choice(AMOUNTS),
        courier=rng.choice(COURIERS),
        days=rng.choice(DAYS),
        error=rng.choice(ERRORS),
    )
    closer = rng.choice(CLOSERS).format(ref=rng.randint(1000, 99999))
    return _noise(rng.choice(OPENERS) + body + closer, rng)


def generate(rows: int, seed: int = 42) -> list[dict[str, str]]:
    rng = random.Random(seed)
    labels = list(TEMPLATES)
    # Slightly unbalanced on purpose, like real support queues.
    weights = [0.22, 0.24, 0.16, 0.14, 0.12, 0.12]
    records = []
    for i in range(rows):
        label = rng.choices(labels, weights=weights)[0]
        records.append({"ticket_id": f"T{i + 1:05d}", "text": make_ticket(label, rng), "label": label})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/sample_tickets.csv"))
    args = parser.parse_args()

    records = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket_id", "text", "label"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} tickets to {args.out}")


if __name__ == "__main__":
    main()
