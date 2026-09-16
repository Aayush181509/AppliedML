"""Loan default prediction service.

Run from the repository root:
    uvicorn app.api.main:app --reload

Then open http://127.0.0.1:8000/docs for the generated documentation.
"""

import json
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# Loaded ONCE, at startup — never inside the request handler.
model = joblib.load(MODEL_DIR / "loan_default_pipeline.joblib")
metadata = json.loads((MODEL_DIR / "loan_default_pipeline.json").read_text())

app = FastAPI(title="Loan default prediction", version=metadata["version"])


class Application(BaseModel):
    """One loan application, as it exists at the moment of submission."""

    loan_amount: float = Field(gt=0, le=50_000_000, examples=[800_000])
    tenure_months: int = Field(ge=1, le=120, examples=[24])
    interest_rate: float = Field(ge=0, le=50, examples=[14.5])
    previous_loans: int = Field(ge=0, le=50, examples=[0])
    days_past_due_history: int = Field(ge=0, le=3650, examples=[30])
    annual_income: Optional[float] = Field(default=None, ge=0)
    credit_score: Optional[float] = Field(default=None, ge=300, le=850)
    sector: str
    employment_type: str
    has_collateral: Literal["Yes", "No"]


class Prediction(BaseModel):
    probability_of_default: float
    decision: Literal["approve", "review"]
    threshold: float
    model_version: str


@app.get("/health")
def health():
    """Liveness check — what a load balancer calls every few seconds."""
    return {"status": "ok", "model_version": metadata["version"]}


@app.get("/metadata")
def model_metadata():
    """What is deployed right now. The first thing to check in an incident."""
    return metadata


@app.post("/predict", response_model=Prediction)
def predict(application: Application) -> Prediction:
    frame = pd.DataFrame([application.model_dump()])
    probability = float(model.predict_proba(frame)[0, 1])

    return Prediction(
        probability_of_default=round(probability, 4),
        decision="review" if probability >= metadata["threshold"] else "approve",
        threshold=metadata["threshold"],
        model_version=metadata["version"],
    )
