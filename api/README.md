# api/

Stage 3 deliverable (not yet built). Will hold a minimal FastAPI service that:

- loads the trained model from `outputs/models/`
- exposes `POST /predict` — accepts a build's feature payload, returns failure
  probability + top SHAP contributors
- exposes `GET /health` — basic liveness check
- validates input with pydantic

No database. Placeholder until Stage 3 begins.
