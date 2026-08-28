"""
AutoDeploy AI — Stage 3 (prediction API) + Stage 4 live-mode glue.

Endpoints:

  GET  /health        - is the model loaded, what does it expect as input,
                         and how does cold-start routing behave. Also what
                         the Stage 4 frontend polls on load to decide
                         live-vs-demo mode.
  POST /predict        - given one build's feature vector, return either the
                         predicted failure probability + risk tier + SHAP-
                         ranked top contributors, OR the cold_start state
                         when this repo has no prior build history (Layer 1b).
  POST /predict-live    - Stage 4 glue: given {owner, repo, sha}, runs the
                         Stage 3 Layer 2 extractor then the same
                         PredictionService as /predict. Does NOT
                         reimplement extraction or model logic — this is a
                         thin wrapper, not a second code path. Reachable
                         only from the frontend origin (see CORS below).

No database. Run with:

    uvicorn api.main:app --reload

from the project root (with the venv activated). Requires GITHUB_TOKEN in
the environment (or .env) for /predict-live to have a reasonable GitHub
API rate limit — works unauthenticated at 60 requests/hour without it.
Never sent to or readable from the frontend; loaded server-side only.
"""

import logging
import os
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from extractor.extract import build_feature_vector
from extractor.github_client import GitHubClient, GitHubRateLimitError

from .model import ModelNotLoadedError, PredictionService
from .schema import BuildFeatures, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autodeploy_ai.api")

app = FastAPI(
    title="AutoDeploy AI",
    description="Predicts whether a CI build will fail before it runs, and explains why.",
    version="0.1.0",
)

# The Stage 4 dashboard runs on a different origin (Vite dev server, or
# wherever it's deployed) than this API — browsers block cross-origin
# fetches by default without this. Configurable via env rather than
# hardcoded so a deployed frontend origin can be added without a code
# change; defaults cover local dev only.
# `or` rather than `.get(..., default)` alone: .env.example ships CORS_ORIGINS=
# with no value, and if that's copied verbatim to .env, os.environ.get would
# return "" (a set-but-empty var, not unset) — silently breaking CORS instead
# of falling back to the local-dev default.
CORS_ORIGINS = (os.environ.get("CORS_ORIGINS") or "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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
        cold_start_behavior=service.cold_start_behavior(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: BuildFeatures):
    """Score one build.

    Returns HTTP 200 in two shapes: a normal tiered prediction, or the
    cold-start state (`status="cold_start"`, `risk_tier=null`) when this
    repo has no prior build history. cold_start is a valid answer, not an
    error — see api/schema.py for the rule and the reasoning.
    """
    try:
        result = service.predict(features)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result)


class LivePredictionRequest(BaseModel):
    owner: str
    repo: str
    sha: str
    branch: Optional[str] = None
    is_pr: Optional[bool] = None


@app.post("/predict-live")
def predict_live(req: LivePredictionRequest):
    """Stage 4 glue: real repo + commit in, real prediction out.

    Every failure mode gets its own honest response — never a fabricated
    or zeroed-out prediction:
      - repo/commit not found (GitHub 404)      -> HTTP 404
      - GitHub API rate limit exhausted          -> HTTP 429
      - any other GitHub API error                -> HTTP 502
      - model not loaded                          -> HTTP 503
      - extracted features fail schema validation -> HTTP 502

    On success, returns the same shape as /predict's response, plus the
    raw extracted `features` dict (so the frontend can show input_summary
    fields like language) and each feature's extraction provenance level
    (EXACT/APPROXIMATED/UNAVAILABLE — see the Stage 3 Layer 2 parity
    report) so the frontend isn't presenting live data as more certain
    than it is.
    """
    token = os.environ.get("GITHUB_TOKEN")
    client = GitHubClient(token=token)

    try:
        extraction = build_feature_vector(
            req.owner, req.repo, req.sha, branch=req.branch, is_pr=req.is_pr, client=client
        )
    except GitHubRateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail=(
                "GitHub API rate limit exhausted — couldn't fetch enough history to "
                f"score this commit reliably. {exc}"
            ),
        ) from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Could not find {req.owner}/{req.repo} at commit {req.sha[:12]} — check the owner/name and SHA.",
            ) from exc
        raise HTTPException(status_code=502, detail=f"GitHub API error while gathering data for this commit: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - any other extraction failure must be reported, not swallowed
        logger.warning("predict-live extraction failed for %s/%s@%s: %r", req.owner, req.repo, req.sha, exc)
        raise HTTPException(status_code=502, detail=f"Could not extract features for this commit: {exc!r}") from exc

    if not service.is_loaded:
        raise HTTPException(status_code=503, detail=service.load_error or "Model not loaded.")

    try:
        features_obj = BuildFeatures(**extraction.features)
    except ValidationError as exc:
        raise HTTPException(status_code=502, detail=f"Extracted features failed validation: {exc}") from exc

    result = service.predict(features_obj)
    return {
        **result,
        "features": extraction.features,
        "feature_provenance": extraction.provenance,
    }
