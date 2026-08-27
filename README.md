# AutoDeploy AI

Predicts whether a CI build will **fail before it runs**, using only
information available at commit time (change size, author history, build
history, project context) — and explains *why* via SHAP feature attributions.

This is a scoped, defensible demo, not a production system. See
[Out of scope](#out-of-scope--future-work) for what was deliberately left out.

> **Status: Stage 3 in progress.** Stages 1–2 complete (data, features,
> model, SHAP). Stage 3 Layer 1 (prediction API) built and tested; Layers 2
> (live feature extractor) and 3 (GitHub Action) not yet built. Stage 4
> (web UI) not started. Real numbers below, not projections.

## Architecture: train → extract → serve → annotate

```
data/          historical build records (raw + processed). Never committed.
scripts/       data collection, feature engineering, training, evaluation
outputs/       trained model (gitignored), evaluation figures/reports, SHAP artifacts
api/           FastAPI service wrapping the trained model (Stage 3, Layer 1)
web/           single-page UI that calls the API (Stage 4)
```

1. **Train** (`scripts/`) — TravisTorrent build history -> leakage-free
   features -> HistGradientBoostingClassifier, evaluated against real
   baselines, explained with SHAP. Offline, one-time (rerun to retrain).
2. **Extract** (Stage 3, Layer 2, in progress) — given a live repo + commit,
   pull the same 31 features from the GitHub API, causally (no future data).
3. **Serve** (`api/`, Stage 3 Layer 1, done) — FastAPI wraps the trained
   model: `POST /predict` returns failure probability, risk tier, and the
   SHAP-ranked features driving that specific prediction.
4. **Annotate** (Stage 3, Layer 3, planned) — a GitHub Action that runs the
   extractor + API on every push/PR and posts the result as a commit status
   and PR comment, in plain language.

Each stage/layer is committed and reviewed before the next one starts.

## Out of scope / future work

Not built unless explicitly requested later:

- a full multi-page dashboard
- user accounts / auth
- deployment infrastructure (this runs locally; the GitHub Action in Stage 3
  Layer 3 calls the API, it doesn't host it anywhere)

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

- **Stage 1 (data):**
  ```bash
  python scripts/01_fetch_data.py       # downloads + extracts TravisTorrent (~3.8 GB uncompressed)
  python scripts/02_prepare_dataset.py  # collapses to build level, labels, writes data/processed/builds_labeled.parquet
  ```
- **Stage 2 (features, training, SHAP):**
  ```bash
  python scripts/03_build_features.py     # 31 leakage-free features + temporal & held-out-projects splits
  python scripts/04_evaluate_baselines.py # majority-class and previous-build-failed baselines, alone
  python scripts/05_train_and_evaluate.py # trains HistGradientBoostingClassifier, saves outputs/models/*.joblib
  python scripts/06_shap_analysis.py      # SHAP interpretation of the temporally-trained model
  ```
- **Stage 3, Layer 1 (prediction API):**
  ```bash
  uvicorn api.main:app --reload   # requires outputs/models/hgb_split_temporal.joblib to already exist
  pytest tests/test_api.py -v
  ```
  See [api/README.md](api/README.md) for the endpoints and a notable finding
  about cold-start predictions.
- **Stage 3, Layers 2–3 (extractor, GitHub Action):** _not yet built_
- **Stage 4 (web UI):** _not yet built_

## Data source

**TravisTorrent** (Beller, Gousios & Zaidman, MSR 2017) — the standard public
research dataset for CI build-outcome prediction. Downloaded from its
permanent Figshare archive (its original site, travistorrent.testroots.org,
has since been taken over by an unrelated domain — do not use it). No
fallback to the GitHub Actions API was needed; TravisTorrent was obtainable
on the first attempt and is a stronger source than a fresh 15–20 repo sample
would be in the time available.

Author identity (needed for "author history" features) isn't in
TravisTorrent's main table, so it's joined in from a companion commit-metadata
dataset (Zenodo 829968). That dataset only covers a subset of TravisTorrent's
projects, so the final dataset is restricted to the 243 projects with real
commit-author coverage — full reasoning and numbers in
[outputs/reports/stage1_data_report.md](outputs/reports/stage1_data_report.md).

**Final dataset:** 261,139 labeled builds across 243 projects, 2011–2016.

**Class balance:** 71.3% passed / 28.7% failed-or-errored — imbalanced,
handled with class weights in Stage 2 rather than resampling.

## Model & performance (Stage 2, real numbers)

**Features:** 31, all leakage-free (computed only from information available
strictly before the build in question — see
[outputs/reports/stage2_feature_list.md](outputs/reports/stage2_feature_list.md)
for every feature's exact provenance). Covers change size, author history,
build/project history, and project context, per the original plan.

**Model:** `HistGradientBoostingClassifier` (scikit-learn), not XGBoost —
see [Deviations](#deviations-from-the-original-plan) below. Trained
separately on two splits: **temporal** (train on builds before
2015-11-01, test after) and **held-out-projects** (49 of 243 projects never
seen in training). `class_weight='balanced'`, chosen over `None` on a
validation slice (not the test set); no SMOTE, no resampling.

**The real baseline to beat wasn't majority class — it was "the previous
build in this project failed → predict fail."** That baseline is already
strong (0.795 ROC-AUC / 0.565 PR-AUC temporal).

| Method | Split | Precision (fail) | Recall (fail) | F1 (fail) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Majority class | temporal | 0.000 | 0.000 | 0.000 | n/a | n/a |
| Previous build failed → predict fail | temporal | 0.697 | 0.697 | 0.697 | 0.795 | 0.565 |
| **HGB (this model)** | **temporal** | **0.611** | **0.755** | **0.675** | **0.876** | **0.804** |
| Majority class | held-out-projects | 0.000 | 0.000 | 0.000 | n/a | n/a |
| Previous build failed → predict fail | held-out-projects | 0.608 | 0.608 | 0.608 | 0.738 | 0.468 |
| **HGB (this model)** | **held-out-projects** | **0.545** | **0.676** | **0.603** | **0.830** | **0.690** |

The model clearly beats the real baseline on both splits (+0.239 and +0.221
PR-AUC respectively) — but its temporal-vs-held-out PR-AUC gap (0.114) is
*wider* than the baseline's (0.096): it leans more on project-specific
history than the naive rule does, proportionally. Full numbers, class-weight
selection detail, and methodology notes:
[outputs/reports/stage2_model_results.md](outputs/reports/stage2_model_results.md).

**SHAP finding — the honest headline of this project:** three features
(`previous_build_status`, `project_prior_failure_rate`,
`consecutive_failure_streak`) account for **65.8%** of total model
attribution. Including author-history (which cold-starts the same way for
an unseen author), **~73% of what this model keys on is undefined for a
brand-new project or author.** This is fundamentally more an
"is this project currently in a failure streak" detector than a
"these code characteristics predict failure" detector — which is exactly
why the naive previous-build rule is already competitive. The remaining
~27% of signal (branch type, repo age, test density, PR-vs-push, change
size) is real but individually weak (each under ~4% of total attribution).
Full ranking and a flagged counterintuitive finding (larger teams predict
*higher* risk, concentrated in newer/less-mature projects):
[outputs/reports/stage2_shap_analysis.md](outputs/reports/stage2_shap_analysis.md).

This same finding shows up again at the API layer, not just in SHAP: a
synthetic "first build ever, all history null" test vector against the live
API returned **0.706 — High risk** (see [api/README.md](api/README.md)).
Layer 3's GitHub Action will need an explicit cold-start caveat in its own
README section for exactly this reason — not written yet, since Layer 3
isn't built.

## Deviations from the original plan

- **Gradient-boosting model is scikit-learn's `HistGradientBoostingClassifier`,
  not XGBoost.** XGBoost's macOS wheel requires Homebrew's `libomp`, not
  installed on this machine; installing system-level toolchains for a demo
  wasn't judged worth it. No SHAP or functional loss at this dataset's size.
- **Model comparison narrowed from "LR vs RF vs GBM" to "HGB vs. two
  explicit baselines"** (majority class, previous-build-failed) — a
  deliberate scope refinement made during Stage 2, not a silent cut.
- **`git_diff_test_churn` is 0 for all 261,139 rows** in TravisTorrent — a
  real gap in the dataset, not a bug here. Dropped rather than kept as a
  zero-variance feature; `test_file_ratio` covers the same distinction.
- Full list of smaller judgment calls (temporal cutoff choice, author-history
  scoping, missing-value handling, etc.) in each stage's report under
  `outputs/reports/`.
