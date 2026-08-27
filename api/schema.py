"""
The API's input/output contract, and the single source of truth for which
of the 31 training features are allowed to be NaN.

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
"""

from typing import List, Optional

from pydantic import BaseModel, Field

# Features allowed to be missing/null — cold-start-only, nothing else.
NAN_ALLOWED_FEATURES = {
    "previous_build_status",
    "project_prior_failure_rate",
    "author_prior_builds_in_project",
    "author_prior_failure_rate_in_project",
    "author_days_since_last_build_in_project",
}

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
    previous_build_status: Optional[float] = Field(default=None, ge=0, le=1, description="1=prior build in this project failed, 0=passed, null=no prior build (first build in project)")
    project_prior_failure_rate: Optional[float] = Field(default=None, ge=0, le=1, description="null=first build in project")
    project_prior_build_count: float = Field(ge=0, description="0 for a project's first build")
    consecutive_failure_streak: float = Field(ge=0, description="0 if the previous build passed or this is the first build")

    # --- Author history (3) --- all may be null (author never seen in this project before, or unidentifiable).
    author_prior_builds_in_project: Optional[float] = Field(default=None, ge=0)
    author_prior_failure_rate_in_project: Optional[float] = Field(default=None, ge=0, le=1)
    author_days_since_last_build_in_project: Optional[float] = Field(default=None, ge=0)

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


class PredictionResponse(BaseModel):
    failure_probability: float
    risk_tier: str
    risk_tier_thresholds: dict
    top_contributors: List[Contributor]


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # "model_*" fields below are ours, not pydantic's

    status: str
    model_loaded: bool
    model_path: str
    detail: Optional[str] = None
    feature_schema: dict
