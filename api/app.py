"""
FastAPI service for Home Credit default risk scoring.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import ModelRegistry, predict, predict_batch

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
    stream=sys.stdout,
)
logger = logging.getLogger("credit_risk_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ModelRegistry.load()
        logger.info("Model ready.")
    except FileNotFoundError as exc:
        logger.warning("Model not yet trained: %s", exc)
    yield


app = FastAPI(
    title="Home Credit Default Risk API",
    description="Predicts default probability from Home Credit application features.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


class HomeCreditApplication(BaseModel):
    NAME_CONTRACT_TYPE: str = Field("Cash loans")
    CODE_GENDER: str = Field("F")
    FLAG_OWN_CAR: str = Field("N")
    FLAG_OWN_REALTY: str = Field("Y")
    CNT_CHILDREN: int = Field(0, ge=0, le=20)
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: Optional[float] = Field(None, ge=0)
    AMT_GOODS_PRICE: Optional[float] = Field(None, ge=0)
    NAME_INCOME_TYPE: str = Field("Working")
    NAME_EDUCATION_TYPE: str = Field("Secondary / secondary special")
    NAME_FAMILY_STATUS: str = Field("Married")
    NAME_HOUSING_TYPE: str = Field("House / apartment")
    DAYS_BIRTH: int = Field(..., le=-1)
    DAYS_EMPLOYED: int = Field(..., description="Home Credit uses 365243 for unknown employment.")
    OCCUPATION_TYPE: Optional[str] = None
    CNT_FAM_MEMBERS: Optional[float] = Field(None, ge=1)
    REGION_RATING_CLIENT: Optional[int] = Field(None, ge=1, le=3)
    REGION_RATING_CLIENT_W_CITY: Optional[int] = Field(None, ge=1, le=3)
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "F",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "CNT_CHILDREN": 0,
                "AMT_INCOME_TOTAL": 162000,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000,
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_HOUSING_TYPE": "House / apartment",
                "DAYS_BIRTH": -12005,
                "DAYS_EMPLOYED": -4542,
                "EXT_SOURCE_2": 0.262949,
                "EXT_SOURCE_3": 0.139376,
            }
        }
    }


class PredictionResponse(BaseModel):
    request_id: str
    default_probability: float
    risk_band: str
    model_version: str
    latency_ms: Optional[float] = None


class BatchRequest(BaseModel):
    applications: list[HomeCreditApplication] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    request_id: str
    count: int
    predictions: list[PredictionResponse]


@app.get("/", tags=["Meta"])
def root():
    return {"service": "Home Credit Default Risk API", "version": "2.0.0", "docs": "/docs"}


@app.get("/healthz", tags=["Ops"])
def health():
    return {"status": "ok"}


@app.get("/readyz", tags=["Ops"])
def ready():
    try:
        _, meta = ModelRegistry.load()
        return {"status": "ready", "model_sha": meta["sha256"][:12] if meta else "unknown"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not ready: {exc}",
        )


@app.get("/v1/model/info", tags=["Model"])
def model_info():
    try:
        _, meta = ModelRegistry.load()
        return meta
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/v1/predict", response_model=PredictionResponse, tags=["Scoring"])
def score_application(application: HomeCreditApplication):
    start = time.perf_counter()
    try:
        result = predict(application.model_dump())
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.")

    return PredictionResponse(
        request_id=str(uuid.uuid4()),
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        **result,
    )


@app.post("/v1/predict/batch", response_model=BatchResponse, tags=["Scoring"])
def score_batch(batch: BatchRequest):
    records = [application.model_dump() for application in batch.applications]
    try:
        results = predict_batch(records)
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Batch prediction failed.")

    request_id = str(uuid.uuid4())
    predictions = [
        PredictionResponse(request_id=request_id, latency_ms=None, **result)
        for result in results
    ]
    return BatchResponse(request_id=request_id, count=len(predictions), predictions=predictions)


@app.post("/v1/model/reload", tags=["Ops"])
def reload_model():
    try:
        _, meta = ModelRegistry.reload()
        return {"status": "reloaded", "model_sha": meta["sha256"][:12] if meta else "unknown"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


try:
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:
    pass
