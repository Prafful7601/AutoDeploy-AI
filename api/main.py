"""
AutoDeploy AI — Stage 3, Layer 1: prediction API.

A minimal FastAPI service wrapping the Stage 2 model. Two endpoints:

  GET  /health   - is the model loaded, and what does it expect as input.
  POST /predict  - given one build's feature vector, return the predicted
                   failure probability, a risk tier, and the SHAP-ranked
                   top contributing features for that specific prediction.

No database, no auth beyond what's needed later for the GitHub Action
(handled in Layer 3). Run with:

    uvicorn api.main:app --reload

from the project root (with the venv activated).
"""

import logging

from fastapi import FastAPI, HTTPException

from .model import ModelNotLoadedError, PredictionService
from .schema import BuildFeatures, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AutoDeploy AI",
    description="Predicts whether a CI build will fail before it runs, and explains why.",
    version="0.1.0",
)

# Loaded once at import time, not per-request. If the artifact is missing,
# this does NOT crash the process — see ModelNotLoadedError handling below.
# That's a deliberate choice: it means `GET /health` stays reachable to
# report *why* the service is degraded, rather than uvicorn just refusing
# to start with a bare traceback. /predict still fails loudly (503, with
# the same actionable message) on every request while the model is absent.
service = PredictionService()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if service.is_loaded else "degraded",
        model_loaded=service.is_loaded,
        model_path=str(service.model_path),
        detail=None if service.is_loaded else service.load_error,
        feature_schema=service.feature_schema(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: BuildFeatures):
    try:
        result = service.predict(features)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result)
