"""
The API's input/output contract: the single source of truth for which of
the 31 training features may be NaN, and — as of Layer 1b — the canonical
definition of the cold-start rule.

Only 5 features are allowed to arrive missing or explicitly `null`:
`previous_build_status`, `project_prior_failure_rate`, and the 3
author-history features. These are the features that are genuinely
undefined — not "unknown", *undefined* — for a project's first-ever build
or an author's first-ever build in a project (see
scripts/03_build_features.py). The model was trained on real NaNs in
exactly these columns (HistGradientBoostingClassifier handles missingness
natively), so passing them through as NaN reproduces training conditions
instead of guessing a fake value.

Every other feature is required. In particular `test_file_ratio` is
required even though ~0.5% of training rows had it as NaN (a genuine
"zero files changed" edge case) — that's a change-size feature, not a
history one, and the "history features may be missing" exception in the
brief was written specifically about cold-start history, not this edge
case. Flagging this narrow reading explicitly: a caller hitting a true
zero-file-diff commit should pass `test_file_ratio=0.0`, not omit it.


================================================================================
THE COLD-START RULE (canonical definition)
================================================================================

Executable form: `api/coldstart.py`. If the two disagree, this docstring is
the specification and that module is the bug.

    A request is COLD-START if and only if at least one PROJECT-history
    feature is absent:

        previous_build_status is null   OR
        project_prior_failure_rate is null

    Absence of author-history features alone is NOT cold-start.

"Absent" means: key omitted, value explicitly null, or value NaN. Note that
`0.0` is a real value, not an absence — `previous_build_status=0` means
"the previous build passed", which is a strong *negative* risk signal and
must never be confused with "there was no previous build".

--------------------------------------------------------------------------------
Why the trigger is project history and not "any of the 5"
--------------------------------------------------------------------------------

The 5 nullable features look like one group but are two, with different
real-world meanings and very different consequences for the prediction:

  PROJECT history (2 features) is null if and only if this repo has no
  build before the one being scored. This is the case that breaks the
  model. Stage 2's SHAP analysis found that `previous_build_status`,
  `project_prior_failure_rate` and `consecutive_failure_streak` together
  carry 65.8% of total attribution; when project history is null, that
  block is gone. Worse, the model does not treat the absence neutrally —
  in training, `previous_build_status=null` overwhelmingly meant "first
  build of a young, churning, not-yet-stable project", and young projects
  failed more often. So the model learned null as a *failure signal* in
  its own right (+1.35 SHAP on the Layer 1 test vector, its single
  strongest driver). A synthetic brand-new-repo vector scored 0.706 —
  High risk — before the repo had done anything wrong. At serve time null
  also means "established repo, someone just installed this tool", and the
  model cannot distinguish the two. That is textbook train-serve skew, and
  it is what the cold_start state exists to surface.

  AUTHOR history (3 features) is null every time a contributor makes their
  first build in a repo. On an established repo this is routine, extremely
  common in the training data, and — critically — the dominant project
  history block is still fully intact. Withholding a risk tier here would
  suppress a prediction the model is well equipped to make, and would fire
  on every first-time contributor to a healthy repo. So author-only nulls
  are served as a normal tier.

The nulls are still reported in that case: every response carries
`null_history_features`, so a caller can see that author history was
missing even when a tier was returned. That is transparency without
inventing a third confidence state.

--------------------------------------------------------------------------------
What we deliberately do NOT do
--------------------------------------------------------------------------------

We do not impute null history to a neutral prior, the global base rate, or
any other filler. That would replace a visible "I don't know enough yet"
with an invisible miscalibration — the model would confidently score a
brand-new repo as though it had average history. Nulls stay null all the
way into the model (which handles them natively); the cold_start response
state is how the uncertainty is surfaced instead.

--------------------------------------------------------------------------------
The two response states
--------------------------------------------------------------------------------

  status="ok"          risk_tier is "Low" | "Medium" | "High",
                       probability_confidence="normal",
                       contributors ranked over all 31 features.

  status="cold_start"  risk_tier is null — no tier is issued at all.
                       failure_probability is still returned, with
                       probability_confidence="low" and a plain-language
                       message explaining it is not a risk tier.
                       Contributors are ranked over non-null features only,
                       because the null history features would otherwise
                       dominate the SHAP ranking with values that describe
                       an absence rather than a property of this commit.

Both states are HTTP 200. cold_start is a valid, expected answer about a
real commit, not a client error.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from .coldstart import (  # noqa: F401  (re-exported: schema.py is the contract's front door)
    AUTHOR_HISTORY_FEATURES,
    COLD_START_MESSAGE,
    COLD_START_TRIGGER_FEATURES,
    COLD_START_UPGRADE_PATH,
    CONFIDENCE_LOW,
    CONFIDENCE_NORMAL,
    CONTRIBUTORS_SCOPE_ALL,
    CONTRIBUTORS_SCOPE_NON_NULL,
    NAN_ALLOWED_FEATURES,
    PROJECT_HISTORY_FEATURES,
    STATUS_COLD_START,
    STATUS_OK,
)

KNOWN_LANGUAGES = {"ruby", "python", "java", "go"}


class BuildFeatures(BaseModel):
    # --- Change size (13) --- all required; direct diff-at-commit-time facts.
    src_churn: float = Field(ge=0, description="Lines changed in source files")
    files_added: float = Field(ge=0)
    files_deleted: float = Field(ge=0)
    files_modified: float = Field(ge=0)
    total_files_changed: float = Field(ge=0)
    src_files_changed: float = Field(ge=0)
    doc_files_changed: float = Field(ge=0)
    other_files_changed: float = Field(ge=0)
    tests_added: float = Field(ge=0)
    tests_deleted: float = Field(ge=0)
    test_file_ratio: float = Field(ge=0, le=1, description="(tests_added+tests_deleted)/total_files_changed; pass 0.0 if total_files_changed is 0, not null")
    num_commits_in_build: float = Field(ge=1)
    commits_on_touched_files: float = Field(ge=0)

    # --- Build/project history (4) --- 2 may be null (cold-start), 2 are always defined (default 0).
    # NOTE: null in either of the next two TRIGGERS the cold_start response
    # state (no risk tier). See the cold-start rule in this module's docstring.
    previous_build_status: Optional[float] = Field(default=None, ge=0, le=1, description="1=prior build in this project failed, 0=passed, null=no prior build (first build in project). NULL TRIGGERS COLD_START. 0 is a real value ('passed'), not an absence.")
    project_prior_failure_rate: Optional[float] = Field(default=None, ge=0, le=1, description="null=first build in project. NULL TRIGGERS COLD_START.")
    project_prior_build_count: float = Field(ge=0, description="0 for a project's first build")
    consecutive_failure_streak: float = Field(ge=0, description="0 if the previous build passed or this is the first build")

    # --- Author history (3) --- all may be null (author never seen in this project before, or unidentifiable).
    # Null here does NOT trigger cold_start: a new contributor to an
    # established repo still has the dominant project-history signal intact.
    # The nulls are reported in the response's `null_history_features`.
    author_prior_builds_in_project: Optional[float] = Field(default=None, ge=0, description="null=author's first build in this project. Does NOT trigger cold_start.")
    author_prior_failure_rate_in_project: Optional[float] = Field(default=None, ge=0, le=1, description="null=author's first build in this project. Does NOT trigger cold_start.")
    author_days_since_last_build_in_project: Optional[float] = Field(default=None, ge=0, description="null=author's first build in this project. Does NOT trigger cold_start.")

    # --- Project/build context (11) --- all required.
    team_size: float = Field(ge=1)
    repo_age_days: float = Field(ge=0)
    repo_num_commits: float = Field(ge=0)
    sloc: float = Field(ge=0)
    test_lines_per_kloc: float = Field(ge=0)
    test_cases_per_kloc: float = Field(ge=0)
    asserts_per_kloc: float = Field(ge=0)
    is_pr: int = Field(ge=0, le=1, description="1 if this build is triggered by a pull request")
    by_core_team_member: int = Field(ge=0, le=1)
    language: str = Field(description=f"one of {sorted(KNOWN_LANGUAGES)}; anything else is accepted but treated as an unrecognized-language reference case (see stage3 API notes)")
    is_main_branch: int = Field(ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "src_churn": 42, "files_added": 1, "files_deleted": 0, "files_modified": 3,
                "total_files_changed": 4, "src_files_changed": 3, "doc_files_changed": 0,
                "other_files_changed": 1, "tests_added": 1, "tests_deleted": 0,
                "test_file_ratio": 0.25, "num_commits_in_build": 1, "commits_on_touched_files": 12,
                "previous_build_status": 0, "project_prior_failure_rate": 0.18,
                "project_prior_build_count": 340, "consecutive_failure_streak": 0,
                "author_prior_builds_in_project": 22, "author_prior_failure_rate_in_project": 0.09,
                "author_days_since_last_build_in_project": 3.5,
                "team_size": 8, "repo_age_days": 900, "repo_num_commits": 4500, "sloc": 32000,
                "test_lines_per_kloc": 180, "test_cases_per_kloc": 12, "asserts_per_kloc": 30,
                "is_pr": 0, "by_core_team_member": 1, "language": "ruby", "is_main_branch": 1,
            }
        }
    }


class Contributor(BaseModel):
    feature: str
    feature_value: Optional[float]
    shap_value: float
    direction: str  # "increases risk" | "decreases risk"


class ColdStartDetail(BaseModel):
    """Present only when status == "cold_start"."""

    reason: str = Field(description='"no_prior_build_in_repo" (expected) | "partial_project_history_inconsistent" (malformed vector — one project-history field null but not the other; we fail toward cold_start)')
    triggered_by: List[str] = Field(description="The project-history features that were absent. Non-empty by definition.")
    history_scoring_active: bool = Field(default=False, description="Always False here. True is represented by status='ok'.")
    upgrade_path: str = Field(description="The planned dedicated history-free model for this case.")


class PredictionResponse(BaseModel):
    status: str = Field(description='"ok" | "cold_start"')
    failure_probability: float = Field(description="Always returned. On cold_start this is NOT a risk score — see probability_confidence and message.")
    probability_confidence: str = Field(description='"normal" | "low". "low" means the model lacked the history it mostly relies on; do not gate anything on the probability.')

    # Null exactly when status == "cold_start". Withholding the tier is the
    # entire point of the cold-start state, so this is Optional by design and
    # not a "sometimes we forgot to set it".
    risk_tier: Optional[str] = Field(default=None, description='"Low" | "Medium" | "High", or null on cold_start (no tier is issued).')
    risk_tier_thresholds: Optional[dict] = Field(default=None, description="Null on cold_start, since no tier was applied.")

    message: Optional[str] = Field(default=None, description="Plain-language explanation. Populated on cold_start.")
    cold_start: Optional[ColdStartDetail] = Field(default=None, description="Populated on cold_start, null otherwise.")

    null_history_features: List[str] = Field(description="Which of the 5 nullable history features were absent — reported on EVERY response, including status='ok' (e.g. a new contributor to an established repo).")
    top_contributors: List[Contributor]
    contributors_scope: str = Field(description='"all_features" normally; "non_null_features_only" on cold_start, where null history features are excluded because their SHAP describes an absence, not this commit.')


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # "model_*" fields below are ours, not pydantic's

    status: str
    model_loaded: bool
    model_path: str
    detail: Optional[str] = None
    feature_schema: dict
    cold_start_behavior: dict = Field(default_factory=dict, description="The cold-start rule, its trigger set, and the two response states.")
