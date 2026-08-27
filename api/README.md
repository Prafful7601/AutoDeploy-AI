# api/

Stage 3, Layers 1 and 1b: a minimal FastAPI service wrapping the Stage 2 model.

- `coldstart.py` — the cold-start rule, as executable code. Deliberately
  dependency-free (stdlib only) so the routing decision can be tested
  without loading the model or the web framework, and so Layer 3's Action
  can reuse it.
- `schema.py` — the input/output contract, and the **canonical
  documentation of the cold-start rule**. Single source of truth for which
  of the 31 training features are required vs. allowed to be null (the 5
  cold-start history features), and which of those 5 withhold a risk tier.
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

- `GET /health` — model-loaded status, the full feature schema (which
  fields are required, which allow null, which trigger cold-start), and a
  `cold_start_behavior` block documenting the rule and both response states.
- `POST /predict` — a build's feature vector in. Out: either a normal
  tiered prediction, or the cold-start state. See `schema.py`'s
  `BuildFeatures` example for a full valid payload, or
  http://127.0.0.1:8000/docs once running.

If the model artifact is missing, the service still starts (so `/health`
stays reachable to explain why), but `/health` reports `"status":
"degraded"` with an actionable message, and `/predict` returns `503` with
the same message: run `python scripts/05_train_and_evaluate.py` first.

## Cold-start handling (Layer 1b)

**The problem this exists to solve.** Layer 1 testing found that a synthetic
"brand-new repo, first build ever" vector scored **0.706 — High risk**,
before the repo had done anything wrong. This is not a bug and not an edge
case; it is mechanically inevitable. Stage 2's SHAP analysis showed that
`previous_build_status`, `project_prior_failure_rate` and
`consecutive_failure_streak` carry **65.8%** of total model attribution. In
the training data, `previous_build_status=null` almost always meant "first
build of a young, churning, not-yet-stable project" — and those projects
failed more often. So the model learned *null itself* as a failure signal
(+1.35 SHAP, its single strongest driver on that vector). At serve time,
null also means "established, healthy repo where someone just installed
this tool." The model cannot distinguish the two. That is textbook
train-serve skew.

**Without handling, this tool over-flags essentially every new repo as High
risk until build history accrues.** The `cold_start` state is the fix.

### The rule

A request is cold-start **if and only if at least one *project*-history
feature is absent** (key omitted, `null`, or NaN):

```
previous_build_status is null   OR   project_prior_failure_rate is null
```

Absence of the three *author*-history features alone is **not** cold-start.

Why the trigger is narrower than "any of the 5 nullable fields": those 5
are really two independent groups.

| Group | Null when | Consequence | Routes to |
|---|---|---|---|
| `previous_build_status`, `project_prior_failure_rate` | repo has no build before this one | the 65.8% dominant signal block is gone, *and* the model reads the absence as a failure signal | `cold_start` |
| the 3 `author_*` features | this contributor's first build in this repo | project history fully intact; ubiquitous in training data | normal tier |

A new contributor opening their first PR against a mature repo is routine,
and the model handles it well. Withholding a tier there would fire on
every first-time contributor while doing nothing for the case that actually
breaks. Those nulls are still reported — every response carries
`null_history_features` — so the information isn't lost, it just doesn't
suppress a prediction the model is equipped to make.

`0` is a real value, not an absence: `previous_build_status=0` means "the
previous build passed", a strong *negative* risk signal. Only
null/omitted/NaN counts as absent.

**Partial project history** (one of the two null but not the other) is not
reachable from a well-formed feature vector — both derive from the same
"is there a previous build?" question. If it arrives anyway we fail toward
`cold_start` rather than serve a tier off half-known history, and tag it
`reason: "partial_project_history_inconsistent"` so it's distinguishable
from a genuine first build.

### The two response states

`status: "ok"` — `risk_tier` is Low/Medium/High, `probability_confidence`
is `"normal"`, contributors ranked over all 31 features. Unchanged from
Layer 1.

`status: "cold_start"` — **no risk tier is issued** (`risk_tier: null`).
The raw `failure_probability` is still returned, but with
`probability_confidence: "low"` and a plain-language `message` explaining
it is not a risk tier. Contributors are ranked over **non-null features
only**: on a cold vector the null history features have the largest SHAP
magnitudes in the row, but that magnitude describes an *absence*, not a
property of the commit — listing it as the top "driver" of a prediction
we're declining to tier would be misleading. Excluding it surfaces the
change-size and project-context signal that genuinely does apply.

Both states return **HTTP 200**. `cold_start` is a valid, expected answer
about a real commit, not a client error.

### What we deliberately do not do

We do **not** impute null history to a neutral prior, the global base rate,
or any other filler. That would swap a visible "I don't know enough yet"
for an invisible miscalibration, scoring a brand-new repo as though it had
average history. Nulls stay null all the way into the model, which handles
them natively — exactly as in training. The `cold_start` state is how the
uncertainty is surfaced instead.

### Planned upgrade

A dedicated **history-free model** for cold-start repos, trained only on
the transferable change/process features and routed behind this same
`cold_start` check. Expected performance is approximately Stage 2's
held-out-projects result, **~0.69 PR-AUC** (vs. 0.804 for the
history-using model on the temporal split). It would replace "no tier"
with a calibrated-but-weaker tier. Not built; `cold_start` is the honest
interim answer, and it is the correct floor regardless — a system that
says "I don't know enough yet" beats one that confidently flags every new
repo red.

## Tests

```bash
pytest tests/test_api.py -v
```

Covers: the cold-start rule exhaustively (all 32 null combinations, NaN and
omitted-key equivalence, and a regression guard that `0` is not treated as
absent); cold-start routing end-to-end (fully cold → `cold_start`,
author-only nulls → normal tier, partial project history → `cold_start`
with the distinct reason code); that cold-start contributors exclude null
features and every listed driver has a value; that no imputation is
happening; `/health` documenting the trigger set and both states; the
Layer 1 behaviors unchanged (valid prediction, failure-streak vector still
High, 422 on missing/null required fields); and the risk-tier boundary
logic.

Most tests run against the real trained model, not a mock, and skip
cleanly if the artifact is absent. `TestColdStartRule` needs no model at
all — the routing rule decides whether a prediction is presented as
actionable or withheld, so it's verified independently of everything else.
