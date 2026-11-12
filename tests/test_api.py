"""Tests for the FastAPI endpoints."""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_when_model_and_db_ok(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["checks"] == {"model": True, "database": True}


def test_ready_reports_missing_model(client_without_model):
    r = client_without_model.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["model"] is False


def test_predict_returns_structured_response(client):
    r = client.post("/predict", json={"text": "I was charged twice this month!!!"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "billing"
    assert 0 <= body["confidence"] <= 1
    assert set(body["scores"]) == {"account_access", "billing", "general_inquiry", "refund_request", "shipping_delivery", "technical_issue"}
    assert body["needs_review"] is False
    assert body["model_version"] == "test-fake"
    assert isinstance(body["id"], int)


def test_predict_adds_request_id_header(client):
    r = client.post("/predict", json={"text": "app keeps crashing"}, headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"


def test_low_confidence_is_flagged_for_review(client):
    r = client.post("/predict", json={"text": "Do you have a student discount?"})
    assert r.json()["needs_review"] is True


def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_rejects_missing_field(client):
    assert client.post("/predict", json={}).status_code == 422


def test_predict_rejects_too_long_text(client):
    assert client.post("/predict", json={"text": "a" * 501}).status_code == 422


def test_predict_returns_503_without_model(client_without_model):
    assert client_without_model.post("/predict", json={"text": "hello there"}).status_code == 503


def test_batch_predict(client):
    texts = ["refund please", "parcel never came, delivery late", "password reset not working"]
    r = client.post("/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    assert r.json()["count"] == 3
    assert [p["label"] for p in r.json()["predictions"]] == ["refund_request", "shipping_delivery", "account_access"]


def test_batch_rejects_too_many(client):
    assert client.post("/predict/batch", json={"texts": ["x y z"] * 6}).status_code == 422


def test_batch_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_predictions_are_stored_and_listed(client):
    client.post("/predict", json={"text": "I want a refund"})
    client.post("/predict", json={"text": "invoice is wrong, charged too much"})
    r = client.get("/predictions")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert data["items"][0]["predicted_label"] == "billing"  # newest first
    assert data["items"][1]["input_text"] == "I want a refund"


def test_predictions_filter_by_label_and_review(client):
    client.post("/predict/batch", json={"texts": ["refund me", "refund again", "what are your hours?"]})
    assert client.get("/predictions", params={"label": "refund_request"}).json()["total"] == 2
    assert client.get("/predictions", params={"needs_review": True}).json()["total"] == 1


def test_stats(client):
    client.post("/predict/batch", json={"texts": ["refund me", "refund again", "what are your hours?"]})
    s = client.get("/predictions/stats").json()
    assert s["total_predictions"] == 3
    assert s["by_label"] == {"refund_request": 2, "general_inquiry": 1}
    assert s["needs_review"] == 1
    assert 0 < s["average_confidence"] < 1


def test_stats_empty(client):
    s = client.get("/predictions/stats").json()
    assert s == {"total_predictions": 0, "by_label": {}, "average_confidence": None, "needs_review": 0}


def test_model_info(client):
    info = client.get("/model").json()
    assert info["loaded"] is True
    assert "billing" in info["labels"]
    assert info["test_metrics"]["accuracy"] == 0.9


def test_model_info_without_model(client_without_model):
    assert client_without_model.get("/model").json()["loaded"] is False
