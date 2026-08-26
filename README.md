# AutoDeploy AI

Predicts whether a CI build will **fail before it runs**, using only
information available at commit time (change size, author history, build
history, project context) — and explains *why* via SHAP feature attributions.

This is a scoped, defensible demo, not a production system. See
[Out of scope](#out-of-scope--future-work) for what was deliberately left out.

> **Status: Stage 0 (project scaffolding) complete.** Stages 1–4 below are
> not yet built. This README will be rewritten at the end with the real data
> source, feature list, and honest performance numbers — nothing below the
> status line is a claim of results yet.

## How the pieces fit together

```
data/          historical build records (raw + processed). Never committed.
scripts/       data collection, feature engineering, training, evaluation
outputs/       trained model, evaluation figures/reports, SHAP artifacts
api/           FastAPI service wrapping the trained model (Stage 3)
web/           single-page UI that calls the API (Stage 4)
```

## Project plan (built and reviewed in stages)

1. **Data** — historical CI build records with commit metadata. TravisTorrent
   first choice; GitHub Actions API (15–20 repos) as fallback if
   TravisTorrent isn't reasonably obtainable. Binary target: build fails (1)
   or passes (0).
2. **Features & model** — leakage-free features (change size, author
   history, build history, project context), class imbalance handled with
   class weights, logistic regression / random forest / gradient boosting
   compared via cross-validated precision/recall/F1/ROC-AUC (failure class
   emphasized), best model saved, SHAP explanations added.
3. **API** — FastAPI service, one `/predict` endpoint (probability + top
   SHAP contributors), `/health` check, input validation. No database.
4. **Web UI** — one plain page that submits features to the API and shows
   the risk + explanation.

Each stage is committed and reviewed before the next one starts.

## Out of scope / future work

Not built unless explicitly requested later:

- a GitHub Actions bot that comments predictions on PRs
- a full multi-page dashboard
- user accounts / auth
- deployment infrastructure (this runs locally)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # only needed if Stage 1 falls back to the GitHub API
```

A single fixed random seed (`SEED = 42` in [scripts/config.py](scripts/config.py))
is used everywhere randomness could enter — data splits, model training —
so results are reproducible run to run.

## Running each stage

Filled in as each stage lands:

- **Stage 1 (data):** _not yet built_
- **Stage 2 (train):** _not yet built_
- **Stage 3 (API):** _not yet built_
- **Stage 4 (web UI):** _not yet built_

## Data source

_To be filled in after Stage 1 — will state plainly which source was used
(TravisTorrent or GitHub Actions API), how many build records were
collected, and the class balance._

## Performance

_To be filled in after Stage 2 — real cross-validated numbers, including
if minority-class (failure) performance is weak. No inflated metrics._
