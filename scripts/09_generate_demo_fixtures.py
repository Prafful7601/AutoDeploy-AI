"""
Stage 4: generate the web dashboard's demo-mode fixtures.

Every example here is a REAL inference from the actual trained model
(outputs/models/hgb_split_temporal.joblib) — nothing in the output JSON is
hand-typed or invented. What varies between examples is where the INPUT
feature vector came from:

  - "low_risk_real_repo" uses a real live GitHub extraction (spf13/cobra,
    captured in extractor/fixtures/example_extraction.json during Stage 3
    Layer 2 development) — real repo, real commit, real GitHub Actions
    history, real model output.
  - "high_risk_streak" and "cold_start_new_repo" use hand-specified
    feature vectors identical to existing test fixtures
    (tests/test_post_prediction.py's VALID_PAYLOAD / COLD_PAYLOAD) — the
    INPUT values were chosen to illustrate a scenario, but the OUTPUT is
    the real model's real inference on them, exactly as those tests
    already verify.
  - "cross_ci_caveat_medium" is a new hand-specified vector (PR build,
    unrecognized language, moderate/short history) built the same way —
    added here rather than pulled from a second live extraction because
    this session's unauthenticated GitHub rate limit (6/60 remaining, no
    token available) wasn't enough left to reliably complete one. Flagged
    explicitly: this is the one example not backed by a real repo,
    though its OUTPUT is still 100% real model inference on those inputs.

Every example's `_provenance` field states plainly which of these it is.
Run this whenever the model is retrained, to keep the dashboard's demo
data honest and current.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT  # noqa: E402

from api.model import PredictionService  # noqa: E402
from api.schema import BuildFeatures  # noqa: E402

OUT_PATH = PROJECT_ROOT / "web" / "src" / "data" / "demoFixtures.json"
EXTRACTOR_FIXTURE = PROJECT_ROOT / "extractor" / "fixtures" / "example_extraction.json"

# Identical to tests/test_post_prediction.py's VALID_PAYLOAD — an active
# failure streak in an established project.
HIGH_RISK_STREAK_FEATURES = {
    "src_churn": 42, "files_added": 1, "files_deleted": 0, "files_modified": 3,
    "total_files_changed": 4, "src_files_changed": 3, "doc_files_changed": 0,
    "other_files_changed": 1, "tests_added": 1, "tests_deleted": 0,
    "test_file_ratio": 0.25, "num_commits_in_build": 1, "commits_on_touched_files": 12,
    "previous_build_status": 1, "project_prior_failure_rate": 0.35,
    "project_prior_build_count": 340, "consecutive_failure_streak": 3,
    "author_prior_builds_in_project": 22, "author_prior_failure_rate_in_project": 0.09,
    "author_days_since_last_build_in_project": 3.5,
    "team_size": 8, "repo_age_days": 900, "repo_num_commits": 4500, "sloc": 32000,
    "test_lines_per_kloc": 180, "test_cases_per_kloc": 12, "asserts_per_kloc": 30,
    "is_pr": 0, "by_core_team_member": 1, "language": "ruby", "is_main_branch": 1,
}

# Identical to tests/test_post_prediction.py's COLD_PAYLOAD — a repo's
# first-ever recorded build.
COLD_START_FEATURES = {
    "src_churn": 5, "files_added": 0, "files_deleted": 0, "files_modified": 1,
    "total_files_changed": 1, "src_files_changed": 1, "doc_files_changed": 0,
    "other_files_changed": 0, "tests_added": 0, "tests_deleted": 0,
    "test_file_ratio": 0.0, "num_commits_in_build": 1, "commits_on_touched_files": 0,
    "project_prior_build_count": 0, "consecutive_failure_streak": 0,
    "team_size": 3, "repo_age_days": 400, "repo_num_commits": 50, "sloc": 3000,
    "test_lines_per_kloc": 100, "test_cases_per_kloc": 5, "asserts_per_kloc": 10,
    "is_pr": 0, "by_core_team_member": 1, "language": "go", "is_main_branch": 1,
}

# New: PR build, unrecognized language, short/moderate history. Tuned (by
# running it through the real model, not by picking a target output first)
# to land in Medium — see this script's module docstring for why this one
# isn't backed by a live extraction.
CROSS_CI_CAVEAT_FEATURES = {
    "src_churn": 180, "files_added": 2, "files_deleted": 1, "files_modified": 6,
    "total_files_changed": 9, "src_files_changed": 7, "doc_files_changed": 1,
    "other_files_changed": 1, "tests_added": 0, "tests_deleted": 0,
    "test_file_ratio": 0.0, "num_commits_in_build": 1, "commits_on_touched_files": 40,
    "previous_build_status": 0, "project_prior_failure_rate": 0.22,
    "project_prior_build_count": 55, "consecutive_failure_streak": 0,
    "author_prior_builds_in_project": 4, "author_prior_failure_rate_in_project": 0.25,
    "author_days_since_last_build_in_project": 45.0,
    "team_size": 15, "repo_age_days": 60, "repo_num_commits": 300, "sloc": 18000,
    "test_lines_per_kloc": 90, "test_cases_per_kloc": 6, "asserts_per_kloc": 15,
    "is_pr": 1, "by_core_team_member": 0, "language": "javascript", "is_main_branch": 0,
}


def run(features: dict) -> dict:
    service = PredictionService()
    if not service.is_loaded:
        raise RuntimeError(f"Model not loaded: {service.load_error}")
    return service.predict(BuildFeatures(**features))


def main():
    live_fixture = json.loads(EXTRACTOR_FIXTURE.read_text())
    low_risk_result = run(live_fixture["features"])

    examples = [
        {
            "id": "high_risk_streak",
            "label": "Active failure streak",
            "_provenance": (
                "Hand-specified feature vector (3 consecutive prior failures, elevated "
                "project failure rate), identical to tests/test_post_prediction.py's "
                "VALID_PAYLOAD. Output is real inference from the trained model on these "
                "inputs — not a hand-typed result."
            ),
            "input_summary": {"repo_description": "Established Ruby project, 340 prior builds", "language": "ruby"},
            "features": HIGH_RISK_STREAK_FEATURES,
            "result": run(HIGH_RISK_STREAK_FEATURES),
        },
        {
            "id": "low_risk_real_repo",
            "label": "Real repo, healthy history",
            "_provenance": (
                f"REAL live GitHub extraction — repo `{live_fixture['_meta']['repo']}`, commit "
                f"`{live_fixture['_meta']['sha'][:12]}`, captured {live_fixture['_meta']['captured']}. "
                "Every input feature came from the live GitHub API via the Stage 3 Layer 2 "
                "extractor; this is the most end-to-end-real example of the four."
            ),
            "input_summary": {
                "repo_description": f"{live_fixture['_meta']['repo']} (real, public)",
                "language": live_fixture["features"]["language"],
            },
            "features": live_fixture["features"],
            "result": low_risk_result,
        },
        {
            "id": "cold_start_new_repo",
            "label": "Brand-new repo, first build",
            "_provenance": (
                "Hand-specified feature vector (all 5 cold-start history fields absent, "
                "0 prior builds), identical to tests/test_post_prediction.py's COLD_PAYLOAD. "
                "Output is real inference showing the cold_start branch of the actual "
                "trained service — not a mocked or invented state."
            ),
            "input_summary": {"repo_description": "New Go project, no Actions history yet", "language": "go"},
            "features": COLD_START_FEATURES,
            "result": run(COLD_START_FEATURES),
        },
        {
            "id": "cross_ci_caveat_medium",
            "label": "PR build, unrecognized language",
            "_provenance": (
                "Hand-specified feature vector (PR build, 'javascript' — not one of this "
                "model's 3 known languages, so handled as the unrecognized-language "
                "reference case; short 60-day history) — NOT backed by a live extraction: "
                "this session's unauthenticated GitHub rate limit (6/60 remaining, no token) "
                "wasn't enough left to reliably complete a second one. Output is real "
                "inference from the trained model on these inputs. Chosen to illustrate the "
                "language-coverage gap and give tier variety (Medium), not to hide the "
                "absence of a second live example."
            ),
            "input_summary": {"repo_description": "Newer JS project, PR from a non-core contributor", "language": "javascript"},
            "features": CROSS_CI_CAVEAT_FEATURES,
            "result": run(CROSS_CI_CAVEAT_FEATURES),
        },
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"generated_by": "scripts/09_generate_demo_fixtures.py", "examples": examples}, indent=2))
    print(f"Wrote {len(examples)} examples -> {OUT_PATH}")
    for ex in examples:
        r = ex["result"]
        print(f"  {ex['id']}: status={r['status']} tier={r.get('risk_tier')} p={r['failure_probability']:.3f}")


if __name__ == "__main__":
    main()
