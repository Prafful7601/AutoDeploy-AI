# AutoDeploy AI

Predicts whether a CI build will **fail before it runs**, using only
information available at commit time (change size, author history, build
history, project context) — and explains *why* via SHAP feature attributions.

This is a scoped, defensible demo, not a production system. See
[Out of scope](#out-of-scope--future-work) for what was deliberately left out.

> **Status: Stage 3 nearly complete — one push blocked.** Stages 1–2
> complete (data, features, model, SHAP). Stage 3 Layers 1 + 1b (prediction
> API + cold-start handling) and Layer 2 (live GitHub feature extractor +
> parity report) built, tested, and pushed. Layer 3 (advisory GitHub
> Action) is built and passes 99 local tests, but the workflow file itself
> (`.github/workflows/predict.yml`) has NOT yet been pushed or run live on
> GitHub — the git credential in use lacks the `workflow` OAuth scope
> GitHub requires for that specific file. See
> [Using the Action](#using-the-action) for what's verified so far and
> what's still pending. **Stage 4 (dashboard) complete** — demo mode and
> live mode both built, tested, and screenshotted (see
> [Stage 4: dashboard](#stage-4-dashboard)). Real numbers below, not
> projections.

## Architecture: train → extract → serve → annotate

```
data/          historical build records (raw + processed). Never committed.
scripts/       data collection, feature engineering, training, evaluation
outputs/       trained model (gitignored), evaluation figures/reports, SHAP artifacts
api/           FastAPI service wrapping the trained model (Stage 3, Layer 1)
web/           React dashboard (Stage 4) — demo mode built; live mode (calling the API) not yet wired
```

1. **Train** (`scripts/`) — TravisTorrent build history -> leakage-free
   features -> HistGradientBoostingClassifier, evaluated against real
   baselines, explained with SHAP. Offline, one-time (rerun to retrain).
2. **Extract** (`extractor/`, Stage 3 Layer 2, done) — given a live repo +
   commit, pulls the same 31 features from the GitHub API, causally (no
   future data). Every feature is rated EXACT/APPROXIMATED/UNAVAILABLE
   against what training actually used — see
   [outputs/reports/stage3_feature_parity.md](outputs/reports/stage3_feature_parity.md).
   Two structural caveats matter more than any single feature: the model
   trained on Travis CI outcomes but serves against GitHub Actions outcomes
   (a different CI system), and live project/author history is bounded by
   how long Actions has been enabled on a repo, not its full git history.
3. **Serve** (`api/`, Stage 3 Layers 1 + 1b, done) — FastAPI wraps the
   trained model: `POST /predict` returns failure probability, risk tier, and
   the SHAP-ranked features driving that specific prediction — or a
   `cold_start` state, with no tier, when the repo has no build history yet.
4. **Annotate** (`.github/workflows/predict.yml`, Stage 3 Layer 3, code
   done, **not yet pushed/run live** — see the status banner above) — a
   GitHub Action that runs on every push/PR, posts the result as a commit
   status (always advisory, never blocking) and, on PRs, a comment — plain
   language, an experimental banner leading every post, and no bare
   percentage as a headline. See [Using the Action](#using-the-action).

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
- **Stage 3, Layers 1 + 1b (prediction API, cold-start handling):**
  ```bash
  uvicorn api.main:app --reload   # requires outputs/models/hgb_split_temporal.joblib to already exist
  pytest tests/test_api.py -v
  ```
  See [api/README.md](api/README.md) for the endpoints and the cold-start
  rule.
- **Stage 3, Layer 2 (live extractor):**
  ```bash
  python -m extractor.cli OWNER/REPO SHA --branch main   # prints the 31-feature JSON vector
  python scripts/07_generate_parity_report.py            # regenerates the feature parity report
  pytest tests/test_extractor.py -v
  ```
  Set `GITHUB_TOKEN` in `.env` for a 5,000/hour rate limit (works
  unauthenticated at 60/hour for a quick check). Tested live against
  `spf13/cobra` — a real repo with real Actions history.
- **Stage 3, Layer 3 (GitHub Action):** `.github/workflows/predict.yml` —
  runs automatically on push/PR once added to a repo (see
  [Using the Action](#using-the-action)); `pytest tests/test_post_prediction.py -v`
  to test the comment-composition logic locally. **Not yet pushed to this
  repo** (blocked — see status banner); the workflow file exists locally
  and all 99 tests pass, but it hasn't run on GitHub's infrastructure yet.
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
API returned **0.706 — High risk**. See
[Cold start: the honest version](#cold-start-the-honest-version) below —
this one is load-bearing enough to have changed the API's behavior.

## Cold start: the honest version

**Without special handling, this tool over-flags essentially every new repo
as High risk until build history accrues.** Not "leans on weaker features
early" — over-flags. A brand-new repo, which is exactly the case where
someone would most want a build-failure predictor, gets a red flag on its
first clean commit.

The mechanism is unavoidable given what the model learned. In training,
`previous_build_status=null` almost always meant "first build of a young,
churning, not-yet-stable project", and those projects did fail more often —
so null was genuinely predictive, and the model learned *null itself* as a
failure signal (+1.35 SHAP on the test vector, its strongest single
driver). At serve time null also means "established, healthy repo where
someone just installed this tool." The model cannot tell those apart. Same
null, two opposite real-world situations, one learned response: flag it.
Textbook train-serve skew, caught before shipping.

**The fix (Stage 3, Layer 1b): an explicit `cold_start` state.** When a repo
has no prior build history, `/predict` does not return a risk tier at all.
It returns `status: "cold_start"` with a plain-language message, the raw
probability clearly labelled low-confidence rather than dressed as a tier,
and the top drivers among the features that *do* have values. The trigger is
the two project-history features specifically — a new *contributor* to an
established repo still gets a normal tier, because project history (the
dominant 65.8% signal block) is intact in that case. Full rule, the
two-groups reasoning, and both response shapes:
[api/README.md](api/README.md#cold-start-handling-layer-1b).

**What was rejected:** imputing null history to a neutral prior or the
global base rate. That's the cheap fix and it's the dishonest one — it hides
the uncertainty by pretending a new repo has average history, and moves the
miscalibration somewhere less visible. Nulls stay null.

**Planned upgrade:** a dedicated **history-free model** for cold-start
repos, trained only on the transferable change/process features and routed
behind this same `cold_start` check. Its expected performance is
approximately the held-out-projects number already measured in Stage 2,
**~0.69 PR-AUC** — versus 0.804 for the history-using model on the temporal
split. That's the honest trade: a calibrated-but-weaker prediction in place
of no tier at all. SHAP showed the transferable signal is thin, so it won't
be great; it will be *calibrated*, rather than stuck at "everything is
High."

## Using the Action

> **Current status: not yet run live.** `.github/workflows/predict.yml` and
> `.github/scripts/post_prediction.py` are written and pass 99 local tests
> (including structural checks on every hard requirement below), but the
> workflow file itself hasn't been pushed to this repo yet — the git
> credential in use here lacks the `workflow` OAuth scope GitHub requires
> to create/modify files under `.github/workflows/`. Everything described
> below is the designed and locally-verified behavior; it will be updated
> with the actual posted comment/status once the push goes through.

`.github/workflows/predict.yml` runs on every push and pull request, extracts
the triggering commit's features live, and posts the result as a commit
status plus (on PRs) a PR comment.

**Adding it to a repo:**
1. Copy `.github/workflows/predict.yml` and `.github/scripts/post_prediction.py`
   into the target repo (and `extractor/`, `api/`, `scripts/`, `requirements.txt`
   — the workflow trains the model itself on first run, see below, so it needs
   the full pipeline, not just the two Action files).
2. No secrets to configure — it uses the workflow's own built-in
   `secrets.GITHUB_TOKEN`, nothing to add manually. Required permissions
   (`contents: read`, `statuses: write`, `pull-requests: write`) are already
   declared in the workflow file.
3. First run on a given commit of the training scripts downloads
   TravisTorrent and trains the model (~5–10 minutes); every run after that
   restores the trained model from `actions/cache` in seconds, until the
   training scripts or `requirements.txt` change.

**Read this before adding it to a repo you actually merge code into:**
- **It is advisory only, by construction.** The commit status always posts
  `state: success` — there is no code path that posts a failing/blocking
  status, and the job has `continue-on-error: true` besides. It cannot be
  configured as a required check that blocks merges just by adding it; don't
  wire it up as one.
- **Cross-CI-system skew.** The model was trained on Travis CI outcomes
  (2011–2016) and is applied here to GitHub Actions outcomes on today's
  repos — a different CI system, a different era. See the
  [feature-parity report](outputs/reports/stage3_feature_parity.md) for what
  is and isn't comparable, feature by feature.
- **Cold-start over-flagging on established repos.** Live history is bounded
  by how long *Actions* has been enabled, not the repo's actual age — see
  [Cold start](#cold-start-the-honest-version). The Action flags this
  explicitly (`cold_start` state, and a "shallow history detected" note when
  a repo looks old but has thin recorded history) rather than papering over it.
- **No bare percentages.** Every comment leads with an experimental banner;
  the raw probability, when shown at all, is visually secondary and labeled
  low-confidence — never a headline verdict. The plain-language drivers are
  the actual content.
- **What "could not score" means.** If the extractor can't get enough data
  (rate limit, API error, or the model isn't available), the Action posts
  that plainly rather than a fabricated or zeroed prediction — see the
  [parity report](outputs/reports/stage3_feature_parity.md)'s framing of why
  that failure mode matters.

**The deployment reality, stated plainly:** this is a working demonstration
of a complete pipeline (train → extract → serve → annotate), not a
calibrated tool ready to gate real merges. Making it production-trustworthy
would require retraining on GitHub Actions outcomes directly (removing the
cross-CI skew) or building a proper cross-CI calibration layer — neither is
built here. Treat every number this Action posts as a demonstration that the
pipeline works, not as ground truth about your build.

## Stage 4: dashboard

> **Status: complete.** Demo mode and live mode both built, tested, and
> screenshotted — in that order deliberately, per instruction, so the
> honest prediction cards could be checked against real fixture data
> before any live path was wired in.

A React (Vite) dashboard at `web/`, backed by two new endpoints on the
existing FastAPI service — no second model or extraction code path.

**Run it (backend + frontend together):**
```bash
# Terminal 1 — backend
source venv/bin/activate
uvicorn api.main:app --reload          # http://localhost:8000

# Terminal 2 — frontend
cd web
npm install
npm run dev                             # http://localhost:5173
```
Set `GITHUB_TOKEN` in the root `.env` for live mode to have a reasonable
GitHub API rate limit (5,000/hour vs. 60/hour unauthenticated) — see
`.env.example`. Not required for demo mode.

**How the fallback behaves — this is the point, not a nice-to-have:** on
load, the frontend calls `GET /health` with a 2.5s timeout. Backend not
running (or unreachable, or slow) → the dashboard shows the *existing*
demo-mode experience with its "sample data" banner, exactly as before —
no error, no broken UI, no flash of blank content. A portfolio visitor
with no backend running still sees a fully working, honest dashboard.
Backend reachable → the same page upgrades to live mode: a repo/SHA input
replaces the example picker, calling the same `PredictionCard` component
demo mode uses — one set of cards, two data sources.

**Backend glue:** `POST /predict-live` (`api/main.py`) takes
`{owner, repo, sha}`, runs the Stage 3 Layer 2 extractor and the same
`PredictionService` `/predict` uses, and returns the same response shape.
Every real-world failure gets its own honest response, never a fabricated
or zeroed prediction:

| Situation | Response |
|---|---|
| Repo/commit doesn't exist | `404`, clear message |
| GitHub rate limit exhausted | `429`, "couldn't fetch enough history" |
| Model not loaded | `503` |
| Extracted features invalid | `502` |
| No prior build history found | `200`, the same `cold_start` card everywhere else uses |

CORS is enabled (`CORS_ORIGINS` env var, defaults to the Vite dev server's
origins) so the frontend's cross-origin requests aren't blocked by the
browser. The GitHub token never leaves the backend — it's read from env
server-side and is not referenced anywhere in `web/`'s code.

**What's real in demo mode:** every one of the 4 bundled examples is a
genuine inference from the actual trained model — nothing in
`web/src/data/demoFixtures.json` is hand-typed, and it's regenerated by
[scripts/09_generate_demo_fixtures.py](scripts/09_generate_demo_fixtures.py),
never hand-edited. Full provenance per example, including why the 4th
isn't backed by a second live pull, is in `web/README.md` and inside the
fixture JSON itself.

**Design honesty, enforced structurally, not just by eye:** the risk tier
is always the headline, never a bare percentage — the probability, when
shown, lives inside a visually de-emphasized secondary line, in both demo
and live mode alike (same component). The `cold_start` card uses a
completely different visual pattern (dashed border, neutral blue, its own
badge shape) from every risk-tier card, so it can never be mistaken for
one. The cross-CI-system caveat is a persistent banner at the top of the
page, not a tooltip.

**Deviations flagged:**
- No "frontend-design" skill exists in this environment (checked via the
  Skill tool). Styled by hand instead; full reasoning in `web/README.md`.
- No Node.js was installed on this machine at all — installed via `nvm`
  (user-level, no sudo, fully reversible) rather than blocking on that.
- Live-mode verification needed a real GitHub token partway through (this
  session's unauthenticated rate limit ran out mid-testing) — the user
  supplied one, stored only in the local, gitignored `.env`.

Screenshots and what to capture for a portfolio: see
[web/README.md](web/README.md#what-to-screenshot-for-a-portfolio).

## Deployment (Render + Vercel, both free tier)

Config is prepared; this section is what to do in each dashboard after
connecting the repo. Nothing here has been deployed by this project
itself — no accounts, no live URLs yet.

### Backend → Render

`render.yaml` at the repo root is a Render Blueprint — connecting this
repo in Render's dashboard should auto-detect it and offer to create the
service with everything below pre-filled.

- **Start command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
  — binds to `0.0.0.0` and Render's assigned `$PORT`, not `localhost`.
- **Python version:** pinned two ways for redundancy — `.python-version`
  (`3.9.6`) and a `PYTHON_VERSION` env var in `render.yaml`, since
  Render's respected mechanism has changed across its docs over time.
  This matters more than usual here: the committed model was pickled
  under this exact Python/scikit-learn combination.
- **`requirements.txt`** is the same fully-pinned file used everywhere
  else in this project (verified: every line has `==`, no loose
  specifiers) — one dependency list, not a slimmed-down deploy-only copy.
- **`GITHUB_TOKEN`**: marked `sync: false` in `render.yaml`, meaning
  Render will prompt for it once in the dashboard and never store it in
  the file or git. Read server-side only via `os.environ.get(...)` (see
  `api/main.py`, `extractor/github_client.py`) — confirmed nowhere in
  this codebase is a token hardcoded (checked with `git grep` across
  everything tracked). The API still runs without it, just at GitHub's
  unauthenticated 60-requests/hour limit instead of 5,000/hour.
- **`CORS_ORIGINS`**: defaults to `"*"` in `render.yaml` so the frontend
  works immediately without a chicken-and-egg URL problem (Vercel's URL
  isn't known until after its first deploy). Safe as a wildcard
  specifically because `api/main.py` never sets `allow_credentials=True`.
  Tighten it to the real Vercel URL afterward by editing the env var
  directly in Render's dashboard — no code change or redeploy needed.

**The model artifact — flagged explicitly, a real decision, not an
oversight:** `outputs/models/hgb_split_temporal.joblib` has been
gitignored since Stage 2 ("never commit model artifacts"). A gitignored
model does not exist on a fresh Render checkout, and Render's free tier
isn't a good place to retrain from scratch on every deploy (the full
pipeline downloads ~525 MB of TravisTorrent data first). Checked the
actual file size before deciding: **414,805 bytes — 405 KB.** That's not
"a large binary," it's smaller than several images already in this repo.
So it's now committed as a deliberate, narrow exception (see the
`.gitignore` comment at that line), rather than solved with a training
step or a separately-hosted release asset — both real options that were
considered and rejected here specifically because the file is small
enough that they'd be solving a problem this project doesn't actually
have. The sibling held-out-projects model stays gitignored; it's Stage 2
evaluation output, never served.

### Frontend → Vercel

`web/vercel.json` pins the build explicitly (`npm run build` →
`dist/`, `framework: "vite"`) rather than relying on Vercel's
auto-detection alone. **Set the project's Root Directory to `web/`** in
Vercel's dashboard — this repo isn't a single-app repo, and Vercel won't
know the frontend lives in a subdirectory otherwise.

- **`VITE_API_BASE_URL`**: set this in Vercel's Environment Variables to
  the Render service's URL once deployed (e.g.
  `https://autodeploy-ai-api.onrender.com`). Vite bakes it into the
  build at build time — there's no hardcoded `localhost` in the shipped
  bundle; unset, the frontend falls back to a `localhost:8000` default
  that simply won't resolve in a visitor's browser, which itself
  triggers the same honest demo-mode fallback (see below), not a broken
  page.

**The fallback — confirmed working, and there's a real nuance worth
knowing for Render specifically:** `web/src/lib/api.js`'s `checkHealth()`
gives the backend 2.5 seconds to respond to `GET /health` before falling
back to demo mode; already verified live (Stage 4) that an unreachable or
slow API results in the existing, honest demo-mode banner, not an error
page. **Render's free tier spins a service down after ~15 minutes idle
and takes up to ~50 seconds to wake back up** — well past that 2.5s
window. In practice this means: after any idle period, the dashboard will
correctly and honestly show demo mode on first load, even though the
backend would have answered if given another 45+ seconds. This is a
deliberate tradeoff, not a bug — a portfolio visitor should never wait
tens of seconds staring at a loading spinner before seeing a working
page. If a visitor does submit a live query while the backend is cold,
`predictLive()` has no artificial timeout and will simply wait through
the wake-up, showing the existing "Extracting features…" state the whole
time.

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
- **The GitHub Action trains the model itself (cached), rather than
  downloading a pre-published artifact.** The trained model is gitignored
  (never committed) and the brief didn't specify how a fresh CI checkout
  gets one. Considered publishing it as a GitHub Release asset instead —
  rejected because it would require a token with release-publish
  permissions and would only work for repos that could reach *this* demo
  repo's releases, whereas train-and-cache is self-contained and works
  identically for anyone who copies the Action to their own repo.
- **The Action runs the model in-process** (`api.model.PredictionService`
  imported directly) rather than over HTTP against a running `POST
  /predict` server — same underlying code, skips standing up a server
  process/port/health-check inside a CI job for no benefit.
- **No "frontend-design" skill exists in this environment** (Stage 4) —
  the brief named one to load before styling the dashboard. Styled by hand
  instead; see `web/README.md`.
- **The trained model is now committed** (`outputs/models/hgb_split_temporal.joblib`),
  reversing Stage 2's "never commit model artifacts" rule — deliberately,
  for deployment: it's 405 KB, and a gitignored model doesn't exist on a
  fresh Render checkout. See the Deployment section above for the full
  reasoning and the alternatives considered and rejected.
- Full list of smaller judgment calls (temporal cutoff choice, author-history
  scoping, missing-value handling, etc.) in each stage's report under
  `outputs/reports/`.

<!-- test commit: verifying the GitHub Action's PR-comment path -->
