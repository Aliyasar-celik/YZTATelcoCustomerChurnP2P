from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., examples=[70.35])
    TotalCharges: float | str = Field(..., examples=[1397.55])


class PredictRequest(BaseModel):
    customer: CustomerFeatures


class PredictBatchRequest(BaseModel):
    customers: List[CustomerFeatures]


class PredictResult(BaseModel):
    prediction_label: str
    churn_probability: float
    threshold: float
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    metadata_loaded: bool
    model_path: str
    metadata_path: str
    detail: str


@dataclass
class ModelArtifacts:
    model: Any
    metadata: Dict[str, Any]
    model_path: Path
    metadata_path: Path


class ModelService:
    def __init__(self) -> None:
        self._artifacts: Optional[ModelArtifacts] = None

    @staticmethod
    def _resolve_project_root() -> Path:
        cwd = Path.cwd().resolve()
        for candidate in [cwd, *cwd.parents]:
            if (candidate / "src").exists() and (candidate / "artifacts").exists():
                return candidate
        return cwd

    def load(self) -> None:
        root = self._resolve_project_root()
        model_path = root / "artifacts" / "churn_model.joblib"
        metadata_path = root / "artifacts" / "metadata.json"

        if not model_path.exists() or not metadata_path.exists():
            self._artifacts = ModelArtifacts(
                model=None,
                metadata={},
                model_path=model_path,
                metadata_path=metadata_path,
            )
            return

        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._artifacts = ModelArtifacts(
            model=model,
            metadata=metadata,
            model_path=model_path,
            metadata_path=metadata_path,
        )

    def health(self) -> HealthResponse:
        if self._artifacts is None:
            self.load()

        assert self._artifacts is not None
        loaded = self._artifacts.model is not None and bool(self._artifacts.metadata)
        detail = (
            "Model and metadata loaded successfully."
            if loaded
            else "Model artifacts are missing. Run notebook section 21 to export artifacts first."
        )
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=self._artifacts.model is not None,
            metadata_loaded=bool(self._artifacts.metadata),
            model_path=str(self._artifacts.model_path),
            metadata_path=str(self._artifacts.metadata_path),
            detail=detail,
        )

    def _ensure_loaded(self) -> ModelArtifacts:
        if self._artifacts is None:
            self.load()

        assert self._artifacts is not None
        if self._artifacts.model is None or not self._artifacts.metadata:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model is not loaded. Please run notebook section 21 (Phase 1 Export) "
                    "to generate artifacts/churn_model.joblib and artifacts/metadata.json."
                ),
            )

        return self._artifacts

    @staticmethod
    def _coerce_input(df: pd.DataFrame) -> pd.DataFrame:
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        return df

    def predict_one(self, payload: CustomerFeatures) -> PredictResult:
        artifacts = self._ensure_loaded()

        features = artifacts.metadata.get("feature_columns", [])
        threshold = float(artifacts.metadata.get("threshold", 0.5))
        model_name = str(artifacts.metadata.get("model_name", "unknown"))

        row = payload.model_dump()
        df = pd.DataFrame([row])
        df = self._coerce_input(df)

        if features:
            missing = [c for c in features if c not in df.columns]
            if missing:
                raise HTTPException(status_code=422, detail=f"Missing expected features: {missing}")
            df = df[features]

        if hasattr(artifacts.model, "predict_proba"):
            proba = float(artifacts.model.predict_proba(df)[:, 1][0])
        else:
            proba = float(artifacts.model.predict(df)[0])

        label = "Yes" if proba >= threshold else "No"

        return PredictResult(
            prediction_label=label,
            churn_probability=proba,
            threshold=threshold,
            model_name=model_name,
        )

    def predict_batch(self, payloads: List[CustomerFeatures]) -> List[PredictResult]:
        return [self.predict_one(p) for p in payloads]


service = ModelService()
app = FastAPI(title="Telco Churn Prediction API", version="0.2.0")


@app.on_event("startup")
async def startup_event() -> None:
    service.load()


@app.get("/", tags=["meta"])
async def root() -> Dict[str, str]:
    return {"message": "Telco churn API is running"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return service.health()


@app.post("/predict", response_model=PredictResult, tags=["prediction"])
async def predict(request: PredictRequest) -> PredictResult:
    return service.predict_one(request.customer)


@app.post("/predict-batch", response_model=List[PredictResult], tags=["prediction"])
async def predict_batch(request: PredictBatchRequest) -> List[PredictResult]:
    if not request.customers:
        raise HTTPException(status_code=422, detail="customers cannot be empty")
    return service.predict_batch(request.customers)
