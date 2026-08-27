# api/

Stage 3, Layer 1: a minimal FastAPI service wrapping the Stage 2 model.

- `schema.py` — the input/output contract. Single source of truth for
  which of the 31 training features are required vs. allowed to be
  null (the 5 cold-start history features).
- `model.py` — loads the temporally-trained HistGradientBoostingClassifier
  (`outputs/models/hgb_split_temporal.joblib`, gitignored — see Stage 2),
  assembles a feature row in the exact training column order, predicts,
  and computes per-request SHAP contributions.
- `main.py` — the two routes.

## Run it

```bash
source venv/bin/activate
uvicorn api.main:app --reload
```

- `GET /health` — model-loaded status + the full feature schema (which
  fields are required, which allow null, and why).
- `POST /predict` — a build's feature vector in, `{failure_probability,
  risk_tier, top_contributors}` out. See `schema.py`'s `BuildFeatures`
  example for a full valid payload, or http://127.0.0.1:8000/docs once
  running (FastAPI's auto-generated interactive docs).

If the model artifact is missing, the service still starts (so `/health`
stays reachable to explain why), but `/health` reports `"status":
"degraded"` with an actionable message, and `/predict` returns `503` with
the same message: run `python scripts/05_train_and_evaluate.py` first.

## Tests

```bash
pytest tests/test_api.py -v
```

Covers: a valid prediction, missing/null required-field rejection (422),
every cold-start history field being individually and jointly omittable
(200), and the risk-tier boundary logic.

## A finding worth knowing before relying on this

Manually testing a synthetic "brand-new project, first build ever" vector
(all 5 history features null, `project_prior_build_count` /
`consecutive_failure_streak` at 0) returned **0.706 failure probability —
High risk** — not the reassuring "not enough data" middle ground one might
hope for. The model reads "no history" itself as a strong risk signal
(`previous_build_status=null` contributed +1.35 toward failure, more than
any single feature in the earlier failure-streak example). This is
consistent with Stage 2's SHAP finding that ~73% of this model's
attribution comes from history features that are undefined for new
projects — worth treating cold-start predictions from this model with real
skepticism until Layer 3's cold-start caveat section is read.
