"""
Capstone Part 3, Task 6.3 — FastAPI deployment mock-up.

Serves the trained models over HTTP:

    GET  /health                      liveness and which models are loaded
    GET  /models                      model metadata and test-set metrics
    POST /predict/volume              hourly traffic volume (regressor)
    POST /predict/risk                proxy accident-risk probability (classifier)
    POST /recommend                   travel-timing recommendation

Run locally
-----------
    uvicorn part3_machine_learning.api.app:app --reload --port 8000
    # then open http://127.0.0.1:8000/docs for the interactive Swagger UI

Example
-------
    curl -X POST http://127.0.0.1:8000/predict/volume \\
         -H "Content-Type: application/json" \\
         -d '{"date": "2018-07-04", "hour": 17, "weather": "Rain", "temp_celsius": 24}'

This is a deployment *simulation*: a single process, models loaded from
local files at start-up, no authentication, no rate limiting. The final
report lists what a production deployment would add.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import date as Date, datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

API_DIR = Path(__file__).resolve().parent
PART3_DIR = API_DIR.parent
REPO_ROOT = PART3_DIR.parent
sys.path.insert(0, str(PART3_DIR / "src"))
sys.path.insert(0, str(REPO_ROOT / "part2_python"))

import common  # noqa: E402
import recommender  # noqa: E402
from logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

REGRESSOR_PATH = common.MODEL_DIR / "regressor_hist_gradient_boosting.joblib"
CLASSIFIER_PATH = common.MODEL_DIR / "classifier_random_forest.joblib"
RESULTS_PATH = common.OUTPUT_DIR / "supervised_results.json"
API_LOG = PART3_DIR / "logs" / "api.log"

WEATHER_OPTIONS = tuple(sorted(recommender.WEATHER_SEVERITY))
STATE: dict = {"regressor": None, "classifier": None, "metrics": {}, "profile": None}


# ---------------------------------------------------------------------------
# Start-up
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(log_file=API_LOG)
    logger.info("API starting — loading models")
    try:
        STATE["regressor"] = joblib.load(REGRESSOR_PATH)
        logger.info("Regressor loaded from %s", REGRESSOR_PATH.name)
    except (OSError, ValueError) as exc:
        logger.error("Regressor could not be loaded: %s", exc)
    try:
        STATE["classifier"] = joblib.load(CLASSIFIER_PATH)
        logger.info("Classifier loaded from %s", CLASSIFIER_PATH.name)
    except (OSError, ValueError) as exc:
        logger.error("Classifier could not be loaded: %s", exc)
    try:
        STATE["metrics"] = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Metrics file unavailable; /models will report none")
    try:
        STATE["profile"] = recommender.load_or_build_profile()
    except (OSError, ValueError) as exc:
        logger.error("Recommender profile unavailable: %s", exc)
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Smart City Traffic Intelligence API",
    description=(
        "Deployment simulation for the I-94 westbound traffic models. "
        "Predicts hourly volume, a proxy accident-risk score, and recommends "
        "travel windows. **The risk score is a proxy derived from congestion and "
        "weather; it is not a prediction of actual collisions.**"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ConditionsIn(BaseModel):
    date: Date = Field(..., description="Calendar date, YYYY-MM-DD", examples=["2018-07-04"])
    hour: int = Field(..., ge=0, le=23, description="Hour of day, 0-23", examples=[17])
    weather: str = Field("Clouds", description=f"One of {', '.join(WEATHER_OPTIONS)}")
    temp_celsius: float | None = Field(
        None, ge=-45, le=50, description="Air temperature; seasonal default if omitted")

    @field_validator("weather")
    @classmethod
    def check_weather(cls, value: str) -> str:
        match = [w for w in WEATHER_OPTIONS if w.lower() == value.lower()]
        if not match:
            raise ValueError(f"weather must be one of {', '.join(WEATHER_OPTIONS)}")
        return match[0]


class VolumeOut(BaseModel):
    date: Date
    hour: int
    weather: str
    temp_celsius: float
    predicted_volume: float
    congestion_category: Literal["Low", "Medium", "High", "Severe"]
    is_holiday: bool
    model: str


class RiskOut(BaseModel):
    date: Date
    hour: int
    weather: str
    temp_celsius: float
    high_risk_probability: float
    high_risk: bool
    model: str
    disclaimer: str


class RecommendIn(BaseModel):
    date: Date | None = Field(None, description="Specific date; enables model mode")
    day_type: Literal["weekday", "weekend", "all"] = "weekday"
    weather: str | None = None
    temp_celsius: float | None = None
    earliest: int = Field(0, ge=0, le=23)
    latest: int = Field(23, ge=0, le=23)
    duration: int = Field(1, ge=1, le=12, description="journey length in hours")

    @field_validator("weather")
    @classmethod
    def check_weather(cls, value: str | None) -> str | None:
        if value is None:
            return None
        match = [w for w in WEATHER_OPTIONS if w.lower() == value.lower()]
        if not match:
            raise ValueError(f"weather must be one of {', '.join(WEATHER_OPTIONS)}")
        return match[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_row(model, conditions: ConditionsIn) -> tuple[pd.DataFrame, float]:
    """One feature row for the requested hour, matching the model's inputs."""
    features = list(model.feature_names_in_)
    frame = recommender.features_for_date(
        conditions.date, conditions.weather, conditions.temp_celsius, features)
    row = frame.iloc[[conditions.hour]]
    temp_c = float(row["temp"].iloc[0] - 273.15)
    return row, temp_c


def require(model_key: str):
    model = STATE.get(model_key)
    if model is None:
        logger.error("Request refused — %s not loaded", model_key)
        raise HTTPException(status_code=503,
                            detail=f"{model_key} is not available; run supervised.py first")
    return model


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    status = {
        "status": "ok" if STATE["regressor"] is not None else "degraded",
        "regressor_loaded": STATE["regressor"] is not None,
        "classifier_loaded": STATE["classifier"] is not None,
        "recommender_ready": STATE["profile"] is not None,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    logger.info("GET /health -> %s", status["status"])
    return status


@app.get("/models")
def models():
    metrics = STATE.get("metrics", {})
    logger.info("GET /models")
    return {
        "regressor": {
            "file": REGRESSOR_PATH.name,
            "algorithm": "HistGradientBoostingRegressor",
            "target": "traffic_volume (vehicles per hour)",
            "test_metrics": metrics.get("regression", {}).get("models", {})
                                   .get("hist_gradient_boosting", {}).get("metrics"),
        },
        "classifier": {
            "file": CLASSIFIER_PATH.name,
            "algorithm": "RandomForestClassifier",
            "target": "high_risk (PROXY label: high/severe congestion AND adverse weather)",
            "test_metrics": metrics.get("classification", {}).get("models", {})
                                   .get("random_forest", {}).get("metrics"),
        },
        "split": metrics.get("split"),
        "feature_count": metrics.get("feature_count"),
    }


@app.post("/predict/volume", response_model=VolumeOut)
def predict_volume(conditions: ConditionsIn):
    model = require("regressor")
    logger.info("POST /predict/volume %s", conditions.model_dump())
    row, temp_c = build_row(model, conditions)
    predicted = float(max(model.predict(row)[0], 0.0))
    profile = STATE.get("profile") or {}
    quartiles = profile.get("quartiles", {"q1": 1249, "q2": 3428, "q3": 4952})
    return VolumeOut(
        date=conditions.date, hour=conditions.hour, weather=conditions.weather,
        temp_celsius=round(temp_c, 1), predicted_volume=round(predicted, 0),
        congestion_category=recommender.classify(predicted, quartiles),
        is_holiday=(conditions.date.month, conditions.date.day) in recommender.FIXED_HOLIDAYS,
        model=REGRESSOR_PATH.name,
    )


@app.post("/predict/risk", response_model=RiskOut)
def predict_risk(conditions: ConditionsIn):
    model = require("classifier")
    logger.info("POST /predict/risk %s", conditions.model_dump())
    row, temp_c = build_row(model, conditions)
    probability = float(model.predict_proba(row)[0, 1])
    return RiskOut(
        date=conditions.date, hour=conditions.hour, weather=conditions.weather,
        temp_celsius=round(temp_c, 1),
        high_risk_probability=round(probability, 4),
        high_risk=probability >= 0.5,
        model=CLASSIFIER_PATH.name,
        disclaimer=(
            "Proxy label: 'high risk' means high or severe congestion is expected to "
            "coincide with severe or low-visibility weather. No collision data was "
            "used. This is not a prediction that an accident will occur."
        ),
    )


@app.post("/recommend")
def recommend_route(request: RecommendIn):
    logger.info("POST /recommend %s", request.model_dump())
    try:
        rec = recommender.recommend(
            day_type=request.day_type, target_date=request.date,
            weather=request.weather, earliest=request.earliest,
            latest=request.latest, duration=request.duration,
            temp_c=request.temp_celsius, profile=STATE.get("profile"),
        )
    except recommender.RecommendationError as exc:
        logger.error("POST /recommend rejected: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return rec.to_dict()
