"""FastAPI inference service for the support-ticket classifier.

Run locally:
    uvicorn app.main:app --reload
Then open http://localhost:8000/docs for the interactive API page.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import PredictionRecord, init_db, make_engine, make_session_factory
from app.logging_config import setup_logging
from app.predictor import DistilBertPredictor, Predictor
from app.schemas import BatchIn, BatchOut, ModelInfo, PredictionList, PredictionOut, Stats, StoredPrediction, TicketIn

log = logging.getLogger("app")


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    """Build the app. Tests pass in their own settings and a fake predictor."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        app.state.engine = make_engine(settings.database_url)
        app.state.session_factory = make_session_factory(app.state.engine)
        init_db(app.state.engine)

        app.state.predictor = predictor
        if app.state.predictor is None:
            try:
                app.state.predictor = DistilBertPredictor(settings.model_dir, version=settings.model_version)
            except Exception:  # keep the API up so /health and /ready can report the problem
                log.exception("Model failed to load from %s", settings.model_dir)
        log.info("Startup complete", extra={"model_loaded": app.state.predictor is not None, "env": settings.environment})
        yield
        app.state.engine.dispose()

    app = FastAPI(
        title="Support Ticket Classifier",
        description="Classifies customer-support tickets with a fine-tuned DistilBERT model and stores every prediction.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # --- middleware: request id + one log line per request ---------------
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("Unhandled error", extra={"request_id": request_id, "path": request.url.path})
            response = JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": request_id})
        duration = round((time.perf_counter() - start) * 1000, 2)
        response.headers["x-request-id"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration,
            },
        )
        return response

    # --- dependencies --------------------------------------------------------
    def get_db(request: Request) -> Iterator[Session]:
        with request.app.state.session_factory() as session:
            yield session

    def get_predictor(request: Request) -> Predictor:
        p = request.app.state.predictor
        if p is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not loaded")
        return p

    def check_length(text_value: str) -> None:
        if len(text_value) > settings.max_text_length:
            raise HTTPException(
                422,
                f"text is longer than the {settings.max_text_length}-character limit",
            )

    def classify_and_store(texts: list[str], request: Request, db: Session, predictor: Predictor) -> list[PredictionOut]:
        start = time.perf_counter()
        results = predictor.predict(texts)
        latency = round((time.perf_counter() - start) * 1000 / len(texts), 2)

        records = [
            PredictionRecord(
                request_id=request.state.request_id,
                input_text=raw,
                cleaned_text=r.cleaned_text,
                predicted_label=r.label,
                confidence=r.confidence,
                needs_review=r.confidence < settings.review_threshold,
                model_version=predictor.version,
                latency_ms=latency,
            )
            for raw, r in zip(texts, results, strict=False)
        ]
        db.add_all(records)
        db.commit()
        return [
            PredictionOut(
                id=rec.id,
                label=r.label,
                confidence=r.confidence,
                needs_review=rec.needs_review,
                scores=r.scores,
                model_version=rec.model_version,
                latency_ms=latency,
            )
            for rec, r in zip(records, results, strict=False)
        ]

    # --- health checks -------------------------------------------------------
    @app.get("/health", tags=["health"], summary="Liveness: is the process running?")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", tags=["health"], summary="Readiness: can we serve predictions?")
    def ready(request: Request):
        checks = {"model": request.app.state.predictor is not None, "database": False}
        try:
            with request.app.state.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["database"] = True
        except Exception:
            log.exception("Database check failed")
        ok = all(checks.values())
        return JSONResponse(status_code=200 if ok else 503, content={"status": "ready" if ok else "not_ready", "checks": checks})

    # --- prediction endpoints ------------------------------------------------
    @app.post("/predict", response_model=PredictionOut, tags=["predict"], summary="Classify one ticket")
    def predict(body: TicketIn, request: Request, db: Session = Depends(get_db), predictor: Predictor = Depends(get_predictor)):
        check_length(body.text)
        return classify_and_store([body.text], request, db, predictor)[0]

    @app.post("/predict/batch", response_model=BatchOut, tags=["predict"], summary="Classify several tickets")
    def predict_batch(body: BatchIn, request: Request, db: Session = Depends(get_db), predictor: Predictor = Depends(get_predictor)):
        if len(body.texts) > settings.max_batch_size:
            raise HTTPException(422, f"at most {settings.max_batch_size} texts per batch")
        for t in body.texts:
            check_length(t)
        preds = classify_and_store(body.texts, request, db, predictor)
        return BatchOut(predictions=preds, count=len(preds))

    # --- stored results ------------------------------------------------------
    @app.get("/predictions", response_model=PredictionList, tags=["results"], summary="List stored predictions")
    def list_predictions(
        db: Session = Depends(get_db),
        label: str | None = Query(None, description="Only this predicted label"),
        needs_review: bool | None = Query(None, description="Only low-confidence predictions"),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        query = select(PredictionRecord)
        if label:
            query = query.where(PredictionRecord.predicted_label == label)
        if needs_review is not None:
            query = query.where(PredictionRecord.needs_review == needs_review)
        total = db.scalar(select(func.count()).select_from(query.subquery()))
        rows = db.scalars(query.order_by(PredictionRecord.id.desc()).limit(limit).offset(offset)).all()
        return PredictionList(items=[StoredPrediction.model_validate(r) for r in rows], total=total, limit=limit, offset=offset)

    @app.get("/predictions/stats", response_model=Stats, tags=["results"], summary="Summary of stored predictions")
    def stats(db: Session = Depends(get_db)):
        by_label = dict(
            db.execute(
                select(PredictionRecord.predicted_label, func.count()).group_by(PredictionRecord.predicted_label)
            ).all()
        )
        avg = db.scalar(select(func.avg(PredictionRecord.confidence)))
        review = db.scalar(select(func.count()).where(PredictionRecord.needs_review.is_(True)))
        return Stats(
            total_predictions=sum(by_label.values()),
            by_label=by_label,
            average_confidence=round(float(avg), 4) if avg is not None else None,
            needs_review=review or 0,
        )

    @app.get("/model", response_model=ModelInfo, tags=["model"], summary="Loaded model details and test metrics")
    def model_info(request: Request):
        p = request.app.state.predictor
        if p is None:
            return ModelInfo(loaded=False, version=settings.model_version, labels=[])
        metrics = getattr(p, "metrics", None)
        summary = None
        if metrics:
            summary = {k: metrics[k] for k in ("accuracy", "macro", "weighted", "per_class") if k in metrics}
        return ModelInfo(loaded=True, version=p.version, labels=p.labels, test_metrics=summary)

    return app


app = create_app()
