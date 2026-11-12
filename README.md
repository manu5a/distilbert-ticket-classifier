# DistilBERT Support-Ticket Classifier

[![CI](https://github.com/manu5a/distilbert-ticket-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/manu5a/distilbert-ticket-classifier/actions/workflows/ci.yml)

An end-to-end machine-learning service that reads a customer-support ticket and
routes it to the right category (billing, technical issue, refund, …).

It covers the whole lifecycle: **data preparation → fine-tuning DistilBERT →
evaluation (precision / recall / F1) → a FastAPI inference service → storing
predictions in PostgreSQL → tests, Docker, CI/CD with GitHub Actions and
deployment to AWS.**

**Stack:** Python · Hugging Face Transformers · PyTorch · scikit-learn · FastAPI · PostgreSQL · SQLAlchemy · Docker · GitHub Actions · AWS (ECR, ECS Fargate, RDS, S3, CloudWatch)

```
                ┌──────────── training (ml/) ────────────┐
 tickets.csv ─► clean & split ─► fine-tune DistilBERT ─► evaluate ─► model + metrics
                                                                        │
                ┌──────────── serving (app/) ────────────┐              ▼
 POST /predict ─► validate ─► clean_text ─► model ─► response ─► PostgreSQL
                                                    │
                      GET /predictions, /predictions/stats, /model, /health, /ready
```

## Dataset

Real company support tickets can't be published, so the repo ships a
**synthetic dataset** (`data/sample_tickets.csv`, 1,800 tickets, 6 labels)
produced by `scripts/generate_sample_data.py`. It deliberately includes the
mess real tickets have — inconsistent casing, typos, email addresses, ticket
references, extra punctuation — and an unbalanced label distribution.

To use your own data, provide any CSV with `text` and `label` columns:

```bash
python -m ml.train --data path/to/your_tickets.csv
```

| Label | Example |
|---|---|
| `billing` | "I was charged €49 twice this month for the Pro plan." |
| `technical_issue` | "The mobile app crashes on my Android phone every time I open it." |
| `account_access` | "The password reset email never arrives." |
| `shipping_delivery` | "The DPD tracking link says delivered but I never got the parcel." |
| `refund_request` | "I returned the item a week ago and still have no refund." |
| `general_inquiry` | "Is my data stored in the EU? We need to check GDPR compliance." |

## How it works

### 1. Data preparation — `ml/data.py`, `ml/preprocess.py`
- Validates the CSV has `text` and `label` columns
- Drops empty rows, too-short texts and exact duplicates
- Normalises labels (`"Technical Issue"` → `technical_issue`)
- `clean_text()` masks personal/noisy tokens (`[EMAIL]`, `[URL]`, `[REF]`, `[NUM]`), normalises unicode, whitespace and repeated punctuation. **The same function runs at training and prediction time** to avoid training/serving skew.
- Stratified 80 / 10 / 10 train / validation / test split

### 2. Fine-tuning — `ml/train.py`
- `distilbert-base-uncased` with a classification head
- A plain PyTorch loop: AdamW, linear warm-up/decay schedule, gradient clipping
- Evaluates on the validation set every epoch, keeps the best checkpoint by **macro-F1**, early stopping
- Final, untouched **test set** evaluation of the best checkpoint

### 3. Evaluation — `ml/evaluate.py`
Writes to the model folder:
- `metrics.json` — accuracy, macro & weighted precision/recall/F1, per-class scores, most-confused label pairs, training history
- `classification_report.txt` — the scikit-learn report
- `confusion_matrix.csv`

Macro-F1 is the headline metric because the classes are unbalanced: it gives the small classes (e.g. refunds) the same weight as the big ones.

### 4. Inference API — `app/`
| Endpoint | What it does |
|---|---|
| `POST /predict` | Classify one ticket → label, confidence, all class scores, `needs_review` flag |
| `POST /predict/batch` | Classify up to 32 tickets |
| `GET /predictions` | Stored predictions, filter by `label` / `needs_review`, paginated |
| `GET /predictions/stats` | Counts per label, average confidence, how many need review |
| `GET /model` | Loaded model version, labels and its test-set metrics |
| `GET /health` | Liveness check (process is up) |
| `GET /ready` | Readiness check (model loaded **and** database reachable) |

- **Input validation** with Pydantic: empty text, missing fields, over-long text and over-sized batches return `422`
- Predictions below a confidence threshold (default 0.6) are flagged `needs_review` for a human
- Every prediction is stored in PostgreSQL with its input, cleaned text, confidence, model version and latency
- **Structured JSON logs** with a request id on every request (also returned as the `x-request-id` header)
- If the model fails to load the API still starts, `/ready` reports `503`, and `/predict` returns `503` instead of crashing

Example:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"text": "I was charged twice for my subscription!!"}'
```
```json
{
  "id": 1,
  "label": "billing",
  "confidence": 0.9871,
  "needs_review": false,
  "scores": {"account_access": 0.0021, "billing": 0.9871, "general_inquiry": 0.0032, "refund_request": 0.0049, "shipping_delivery": 0.0011, "technical_issue": 0.0016},
  "model_version": "local-dev",
  "latency_ms": 18.4
}
```
(Values above are illustrative.)

## Running it

### Locally

```bash
python -m venv .venv && source .venv/bin/activate
make install                 # CPU PyTorch + dependencies
make train                   # ~15-30 min on a laptop CPU; minutes on a GPU
docker compose up -d db      # PostgreSQL
make run                     # API on http://localhost:8000/docs
```

### Everything in Docker

```bash
make train                   # once, writes models/ticket-classifier
docker compose up --build    # PostgreSQL + API, model mounted from ./models
```

### Training on a free GPU
Open a Google Colab notebook with a GPU runtime and run:
```bash
!git clone https://github.com/manu5a/distilbert-ticket-classifier && cd distilbert-ticket-classifier && pip install -r requirements.txt && python -m ml.train
```

## Testing

```bash
make test
```

- **API tests** (`tests/test_api.py`) use a fake predictor and an in-memory database, so they run in seconds. In CI they run against a **real PostgreSQL** service container.
- **ML tests** (`tests/test_ml.py`) cover text cleaning, dataset validation, the stratified split and the metric calculations against hand-checked values.
- **Pipeline tests** (`tests/test_model_pipeline.py`) build a tiny randomly-initialised DistilBERT and run the *real* training loop, saving, reloading and prediction code end to end.

## CI/CD — `.github/workflows/`

| Workflow | Trigger | Steps |
|---|---|---|
| `ci.yml` | every push / PR | ruff lint → pytest against PostgreSQL with coverage → build Docker image → start the container and check `/health` |
| `train.yml` | manual | fine-tune on a GitHub runner → publish the test report → upload the model as an artifact and to S3 |
| `deploy.yml` | after green CI on `main`, or manual | pull model from S3 → build image → push to ECR → deploy to ECS Fargate and wait for it to be healthy |

AWS login uses GitHub OIDC (no long-lived AWS keys stored in GitHub). Deployment
stays switched off until AWS is configured — see **[docs/aws-setup.md](docs/aws-setup.md)**.

## Project structure

```
├── app/                    FastAPI service
│   ├── main.py             endpoints, middleware, health checks
│   ├── predictor.py        loads the model, turns text into predictions
│   ├── db.py               SQLAlchemy engine + predictions table
│   ├── schemas.py          request/response validation (Pydantic)
│   ├── config.py           settings from environment variables
│   └── logging_config.py   JSON logging
├── ml/                     training pipeline
│   ├── preprocess.py       clean_text() shared with the API
│   ├── data.py             load, validate, clean, split
│   ├── train.py            DistilBERT fine-tuning loop
│   └── evaluate.py         precision / recall / F1, confusion matrix
├── scripts/generate_sample_data.py
├── data/sample_tickets.csv
├── tests/
├── deploy/aws/task-definition.json
├── docs/aws-setup.md
├── Dockerfile, docker-compose.yml, Makefile
└── .github/workflows/      ci.yml, train.yml, deploy.yml
```

## Possible next steps
- Database migrations with Alembic
- Feedback endpoint so agents can correct a label, and use corrections to retrain
- Track experiments (MLflow) and compare against a TF-IDF + logistic regression baseline
- Load balancer with HTTPS and autoscaling on ECS
