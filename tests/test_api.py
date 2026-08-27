"""
Tests for Stage 3 Layer 1 (the prediction API). Requires the temporal model
to already exist at outputs/models/hgb_split_temporal.joblib (Stage 2's
scripts/05_train_and_evaluate.py) — these are integration tests against the
real trained model, not a mock, since the whole point is to catch schema
mismatches between training and serving.

Run with: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app  # noqa: E402
from api.model import PredictionService  # noqa: E402

client = TestClient(app)

VALID_PAYLOAD = {
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


def _requires_model():
    if not PredictionService().is_loaded:
        pytest.skip("Model artifact not found — run scripts/05_train_and_evaluate.py first.")


class TestHealth:
    def test_health_reports_something(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "model_loaded" in body
        assert "feature_schema" in body
        # All 31 features should be documented regardless of load state.
        assert len(body["feature_schema"]) == 31

    def test_health_schema_marks_nan_allowed_features(self):
        r = client.get("/health")
        schema = r.json()["feature_schema"]
        nan_allowed = {k for k, v in schema.items() if v["nan_allowed"]}
        assert nan_allowed == {
            "previous_build_status", "project_prior_failure_rate",
            "author_prior_builds_in_project", "author_prior_failure_rate_in_project",
            "author_days_since_last_build_in_project",
        }
        for name in nan_allowed:
            assert schema[name]["required"] is False
        # Spot check a required, non-nan-allowed feature.
        assert schema["src_churn"]["required"] is True


class TestPredictValid:
    def test_valid_vector_returns_200_and_sane_shape(self):
        _requires_model()
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["failure_probability"] <= 1.0
        assert body["risk_tier"] in {"Low", "Medium", "High"}
        assert 1 <= len(body["top_contributors"]) <= 5
        for c in body["top_contributors"]:
            assert c["direction"] in {"increases risk", "decreases risk"}


class TestMissingFeatureRejection:
    def test_missing_required_feature_is_422(self):
        payload = dict(VALID_PAYLOAD)
        del payload["src_churn"]
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_null_required_feature_is_422(self):
        payload = dict(VALID_PAYLOAD)
        payload["src_churn"] = None
        r = client.post("/predict", json=payload)
        assert r.status_code == 422


class TestAllowedNaNHistoryFeatures:
    @pytest.mark.parametrize("field", [
        "previous_build_status", "project_prior_failure_rate",
        "author_prior_builds_in_project", "author_prior_failure_rate_in_project",
        "author_days_since_last_build_in_project",
    ])
    def test_omitting_history_feature_still_succeeds(self, field):
        _requires_model()
        payload = dict(VALID_PAYLOAD)
        del payload[field]
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, r.text

    @pytest.mark.parametrize("field", [
        "previous_build_status", "project_prior_failure_rate",
        "author_prior_builds_in_project", "author_prior_failure_rate_in_project",
        "author_days_since_last_build_in_project",
    ])
    def test_explicit_null_history_feature_still_succeeds(self, field):
        _requires_model()
        payload = dict(VALID_PAYLOAD)
        payload[field] = None
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, r.text

    def test_first_build_in_project_all_five_missing_at_once(self):
        _requires_model()
        payload = dict(VALID_PAYLOAD)
        for field in [
            "previous_build_status", "project_prior_failure_rate",
            "author_prior_builds_in_project", "author_prior_failure_rate_in_project",
            "author_days_since_last_build_in_project",
        ]:
            del payload[field]
        payload["project_prior_build_count"] = 0
        payload["consecutive_failure_streak"] = 0
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, r.text


class TestTierBoundaries:
    """Unit tests on the pure classify_risk function — not round-tripped
    through real model probabilities, since forcing an exact probability
    out of a real trained model isn't reliable, but the boundary logic
    itself needs to be exactly right."""

    def setup_method(self):
        self.service = PredictionService.__new__(PredictionService)  # skip model loading

    @pytest.mark.parametrize("prob,expected", [
        (0.0, "Low"),
        (0.29, "Low"),
        (0.2999999, "Low"),
        (0.3, "Medium"),
        (0.45, "Medium"),
        (0.5999999, "Medium"),
        (0.6, "High"),
        (0.75, "High"),
        (1.0, "High"),
    ])
    def test_boundaries(self, prob, expected):
        assert self.service.classify_risk(prob) == expected
