"""
Smoke tests for the deployment mock-up. Runs the app in-process with
FastAPI's TestClient, so no server needs to be started.

    python part3_machine_learning/api/test_api.py
    # or: pytest part3_machine_learning/api/test_api.py -q
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import app  # noqa: E402

logger = logging.getLogger(__name__)


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["regressor_loaded"] is True
        assert body["classifier_loaded"] is True


def test_predict_volume_peak_vs_night():
    with TestClient(app) as client:
        peak = client.post("/predict/volume", json={
            "date": "2018-03-14", "hour": 17, "weather": "Clouds", "temp_celsius": 2})
        night = client.post("/predict/volume", json={
            "date": "2018-03-14", "hour": 3, "weather": "Clouds", "temp_celsius": 2})
        assert peak.status_code == 200 and night.status_code == 200
        assert peak.json()["predicted_volume"] > 4000
        assert night.json()["predicted_volume"] < 1500
        assert peak.json()["congestion_category"] in {"High", "Severe"}
        assert night.json()["congestion_category"] == "Low"


def test_predict_risk_rain_at_peak_is_riskier_than_clear_at_night():
    with TestClient(app) as client:
        risky = client.post("/predict/risk", json={
            "date": "2018-03-14", "hour": 17, "weather": "Snow", "temp_celsius": -3})
        safe = client.post("/predict/risk", json={
            "date": "2018-03-14", "hour": 3, "weather": "Clear", "temp_celsius": -3})
        assert risky.status_code == 200 and safe.status_code == 200
        assert risky.json()["high_risk_probability"] > safe.json()["high_risk_probability"]
        assert "proxy" in risky.json()["disclaimer"].lower()


def test_validation_errors_are_clean():
    with TestClient(app) as client:
        bad_hour = client.post("/predict/volume", json={"date": "2018-03-14", "hour": 25})
        assert bad_hour.status_code == 422
        bad_weather = client.post("/predict/volume", json={
            "date": "2018-03-14", "hour": 8, "weather": "Tornado"})
        assert bad_weather.status_code == 422
        bad_window = client.post("/recommend", json={"earliest": 20, "latest": 6})
        assert bad_window.status_code == 422


def test_recommend_returns_sentence():
    with TestClient(app) as client:
        response = client.post("/recommend", json={
            "date": "2018-07-04", "weather": "Rain", "earliest": 6, "latest": 20})
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "model"
        assert body["recommendation"].startswith("For your journey on")
        assert len(body["best_windows"]) == 3


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    raise SystemExit(1 if failures else 0)
