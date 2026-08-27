"""
Tests for Stage 3 Layer 1 + 1b (the prediction API and cold-start routing).

Most of these are integration tests against the REAL trained model at
outputs/models/hgb_split_temporal.joblib (Stage 2's
scripts/05_train_and_evaluate.py) — not a mock — since the whole point is to
catch schema mismatches between training and serving. They skip cleanly if
the artifact is absent.

`TestColdStartRule` is the exception and is deliberately dependency-light:
it tests api/coldstart.py directly, so the routing rule is verified even
with no model present. That rule decides whether a prediction is presented
as actionable or withheld, so it gets exhaustive coverage rather than
sampled coverage.

Run with: pytest tests/test_api.py -v
"""

import itertools
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.coldstart import (  # noqa: E402
    AUTHOR_HISTORY_FEATURES,
    PROJECT_HISTORY_FEATURES,
    cold_start_reason,
    is_cold_start,
    null_history_features,
)
from api.main import app  # noqa: E402
from api.model import PredictionService  # noqa: E402

client = TestClient(app)

ALL_HISTORY_FEATURES = tuple(PROJECT_HISTORY_FEATURES) + tuple(AUTHOR_HISTORY_FEATURES)

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


def _payload_without(*fields):
    p = dict(VALID_PAYLOAD)
    for f in fields:
        p.pop(f, None)
    return p


def _brand_new_repo_payload():
    """The Layer 1 finding's vector: first build ever in a fresh repo."""
    p = _payload_without(*ALL_HISTORY_FEATURES)
    p["project_prior_build_count"] = 0
    p["consecutive_failure_streak"] = 0
    p["repo_age_days"] = 1
    return p


# =============================================================================
# The cold-start rule, tested directly. No model needed.
# =============================================================================

class TestColdStartRule:
    PRESENT = {f: VALID_PAYLOAD[f] for f in ALL_HISTORY_FEATURES}

    def test_exhaustive_null_matrix(self):
        """All 32 null combinations route by project history alone."""
        seen_cold = seen_normal = 0
        for mask in itertools.product([False, True], repeat=5):
            values = {
                f: (None if null else self.PRESENT[f])
                for f, null in zip(ALL_HISTORY_FEATURES, mask)
            }
            expected_cold = any(mask[:len(PROJECT_HISTORY_FEATURES)])
            assert is_cold_start(values) is expected_cold, values
            seen_cold += expected_cold
            seen_normal += not expected_cold
        assert (seen_cold, seen_normal) == (24, 8)

    @pytest.mark.parametrize("field", PROJECT_HISTORY_FEATURES)
    def test_project_history_null_triggers(self, field):
        assert is_cold_start({**self.PRESENT, field: None}) is True

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_author_history_null_alone_never_triggers(self, n):
        for combo in itertools.combinations(AUTHOR_HISTORY_FEATURES, n):
            values = {**self.PRESENT, **{f: None for f in combo}}
            assert is_cold_start(values) is False, combo
            assert cold_start_reason(values) is None

    def test_nan_counts_as_absent(self):
        """By the time values reach the model, None has become np.nan."""
        nan = float("nan")
        for field in PROJECT_HISTORY_FEATURES:
            assert is_cold_start({**self.PRESENT, field: nan}) is True

    def test_missing_key_counts_as_absent(self):
        for field in PROJECT_HISTORY_FEATURES:
            assert is_cold_start({k: v for k, v in self.PRESENT.items() if k != field}) is True

    @pytest.mark.parametrize("field", PROJECT_HISTORY_FEATURES)
    def test_zero_is_a_real_value_not_an_absence(self, field):
        """Regression guard against a falsy check.

        previous_build_status=0 means "the previous build passed" — a strong
        negative risk signal. Treating it as absent would withhold a tier
        from exactly the healthiest builds.
        """
        assert is_cold_start({**self.PRESENT, field: 0.0}) is False
        assert is_cold_start({**self.PRESENT, field: 0}) is False

    def test_reason_codes(self):
        both_null = {**self.PRESENT, **{f: None for f in PROJECT_HISTORY_FEATURES}}
        assert cold_start_reason(both_null) == "no_prior_build_in_repo"
        one_null = {**self.PRESENT, PROJECT_HISTORY_FEATURES[0]: None}
        assert cold_start_reason(one_null) == "partial_project_history_inconsistent"

    def test_null_history_features_reports_all_five_in_order(self):
        assert null_history_features({f: None for f in ALL_HISTORY_FEATURES}) == list(ALL_HISTORY_FEATURES)
        assert null_history_features(self.PRESENT) == []

    def test_project_and_author_groups_are_disjoint_and_total_five(self):
        assert not set(PROJECT_HISTORY_FEATURES) & set(AUTHOR_HISTORY_FEATURES)
        assert len(ALL_HISTORY_FEATURES) == 5


# =============================================================================
# /health
# =============================================================================

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
        assert nan_allowed == set(ALL_HISTORY_FEATURES)
        for name in nan_allowed:
            assert schema[name]["required"] is False
        # Spot check a required, non-nan-allowed feature.
        assert schema["src_churn"]["required"] is True

    def test_health_schema_marks_cold_start_triggers(self):
        """Only the 2 project-history features trigger; the 3 author ones don't."""
        schema = client.get("/health").json()["feature_schema"]
        triggers = {k for k, v in schema.items() if v["cold_start_trigger"]}
        assert triggers == set(PROJECT_HISTORY_FEATURES)
        for f in AUTHOR_HISTORY_FEATURES:
            assert schema[f]["nan_allowed"] is True
            assert schema[f]["cold_start_trigger"] is False

    def test_health_documents_cold_start_behavior(self):
        body = client.get("/health").json()
        cs = body["cold_start_behavior"]
        assert sorted(cs["trigger_features"]) == sorted(PROJECT_HISTORY_FEATURES)
        assert sorted(cs["non_trigger_nullable_features"]) == sorted(AUTHOR_HISTORY_FEATURES)
        assert set(cs["states"]) == {"ok", "cold_start"}

    def test_health_language_known_values_excludes_python(self):
        """Regression guard for a real bug found while building Layer 2:
        TravisTorrent covers 4 languages, but Stage 1's project-coverage
        filter left zero python rows in the final 243-project dataset, so
        the trained model has no `language_python` column. `known_values`
        must reflect what the loaded model actually distinguishes (derived
        from its feature columns), not TravisTorrent's original 4."""
        known = client.get("/health").json()["feature_schema"]["language"]["known_values"]
        assert set(known) == {"go", "java", "ruby"}
        assert "python" not in known


class TestUnrecognizedLanguageHandling:
    """`language` accepts any string (never a validation error), but only
    values in known_values get their own model column. Anything else —
    including 'python', despite it being one of TravisTorrent's original
    4 languages — is handled identically to a wholly unrecognized value."""

    def test_python_and_a_made_up_language_predict_identically(self):
        _requires_model()
        p1 = {**VALID_PAYLOAD, "language": "python"}
        p2 = {**VALID_PAYLOAD, "language": "some-made-up-language"}
        r1 = client.post("/predict", json=p1)
        r2 = client.post("/predict", json=p2)
        assert r1.status_code == r2.status_code == 200
        assert r1.json()["failure_probability"] == r2.json()["failure_probability"]

    def test_known_language_predicts_differently_from_unrecognized(self):
        _requires_model()
        p_known = {**VALID_PAYLOAD, "language": "ruby"}
        p_unknown = {**VALID_PAYLOAD, "language": "python"}
        r_known = client.post("/predict", json=p_known)
        r_unknown = client.post("/predict", json=p_unknown)
        assert r_known.status_code == r_unknown.status_code == 200
        # Not required to differ by any specific amount, just confirms the
        # two payloads aren't silently collapsed to the exact same features
        # for a language the model actually has a column for.
        assert r_known.json()["failure_probability"] != r_unknown.json()["failure_probability"]


# =============================================================================
# Normal (history present) predictions — unchanged behavior
# =============================================================================

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

    def test_full_history_is_status_ok_with_no_cold_start_block(self):
        _requires_model()
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["status"] == "ok"
        assert body["probability_confidence"] == "normal"
        assert body["risk_tier"] in {"Low", "Medium", "High"}
        assert body["risk_tier_thresholds"] == {"low_max": 0.3, "high_min": 0.6}
        assert body["cold_start"] is None
        assert body["message"] is None
        assert body["null_history_features"] == []
        assert body["contributors_scope"] == "all_features"

    def test_failure_streak_vector_still_scores_high(self):
        """The Layer 1 demo scenario must be untouched by Layer 1b."""
        _requires_model()
        payload = dict(VALID_PAYLOAD)
        payload.update({
            "previous_build_status": 1,
            "consecutive_failure_streak": 3,
            "project_prior_failure_rate": 0.35,
        })
        body = client.post("/predict", json=payload).json()
        assert body["status"] == "ok"
        assert body["risk_tier"] == "High"
        assert body["failure_probability"] > 0.6


# =============================================================================
# Cold-start routing through the live API
# =============================================================================

class TestColdStartRouting:
    def test_fully_cold_vector_returns_cold_start(self):
        _requires_model()
        r = client.post("/predict", json=_brand_new_repo_payload())
        assert r.status_code == 200, r.text  # cold_start is an answer, not an error
        body = r.json()
        assert body["status"] == "cold_start"
        assert body["risk_tier"] is None
        assert body["risk_tier_thresholds"] is None
        assert body["probability_confidence"] == "low"
        assert body["cold_start"]["reason"] == "no_prior_build_in_repo"
        assert sorted(body["cold_start"]["triggered_by"]) == sorted(PROJECT_HISTORY_FEATURES)
        assert body["cold_start"]["history_scoring_active"] is False
        assert set(body["null_history_features"]) == set(ALL_HISTORY_FEATURES)

    def test_cold_start_message_is_plain_language_and_present(self):
        _requires_model()
        body = client.post("/predict", json=_brand_new_repo_payload()).json()
        msg = body["message"]
        assert msg and "insufficient build history" in msg.lower()
        assert "not" in msg.lower()  # explicitly disclaims being a risk tier

    def test_cold_start_still_reports_probability_but_labels_it(self):
        """The raw probability is retained for transparency — the Layer 1
        finding (0.706 on a new repo) should remain visible, not hidden."""
        _requires_model()
        body = client.post("/predict", json=_brand_new_repo_payload()).json()
        assert 0.0 <= body["failure_probability"] <= 1.0
        assert body["probability_confidence"] == "low"
        assert body["risk_tier"] is None  # ...but never dressed as a tier

    def test_cold_start_contributors_exclude_null_features(self):
        _requires_model()
        body = client.post("/predict", json=_brand_new_repo_payload()).json()
        assert body["contributors_scope"] == "non_null_features_only"
        assert body["top_contributors"], "cold_start must still show some signal"
        names = {c["feature"] for c in body["top_contributors"]}
        # None of the null history features may appear as a "driver"...
        assert not names & set(ALL_HISTORY_FEATURES)
        # ...and every listed driver must have an actual value.
        for c in body["top_contributors"]:
            assert c["feature_value"] is not None

    @pytest.mark.parametrize("field", PROJECT_HISTORY_FEATURES)
    def test_single_project_history_null_routes_to_cold_start(self, field):
        """Partial project history: one of the two null. Not reachable from a
        well-formed extractor, so we fail toward cold_start rather than serve
        a tier off half-known history — and flag it with a distinct reason."""
        _requires_model()
        body = client.post("/predict", json=_payload_without(field)).json()
        assert body["status"] == "cold_start"
        assert body["risk_tier"] is None
        assert body["cold_start"]["reason"] == "partial_project_history_inconsistent"
        assert body["cold_start"]["triggered_by"] == [field]

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_author_only_nulls_route_to_a_normal_tier(self, n):
        """DOCUMENTED DECISION: a new contributor to an established repo gets
        a normal tier. Project history — 65.8% of attribution — is intact,
        and this case was ubiquitous in training. The nulls are still
        surfaced via null_history_features."""
        _requires_model()
        for combo in itertools.combinations(AUTHOR_HISTORY_FEATURES, n):
            body = client.post("/predict", json=_payload_without(*combo)).json()
            assert body["status"] == "ok", combo
            assert body["risk_tier"] in {"Low", "Medium", "High"}, combo
            assert body["probability_confidence"] == "normal"
            assert body["cold_start"] is None
            assert set(body["null_history_features"]) == set(combo)
            assert body["contributors_scope"] == "all_features"

    def test_explicit_null_routes_same_as_omitted(self):
        """`{"previous_build_status": null}` and omitting the key must be
        indistinguishable — an extractor may legitimately send either."""
        _requires_model()
        omitted = client.post("/predict", json=_payload_without("previous_build_status")).json()
        explicit = dict(VALID_PAYLOAD)
        explicit["previous_build_status"] = None
        explicit_body = client.post("/predict", json=explicit).json()
        assert omitted["status"] == explicit_body["status"] == "cold_start"
        assert omitted["failure_probability"] == pytest.approx(explicit_body["failure_probability"])

    def test_previous_build_passed_is_not_cold_start_end_to_end(self):
        """previous_build_status=0 must produce a real tier, not cold_start."""
        _requires_model()
        payload = dict(VALID_PAYLOAD)
        payload["previous_build_status"] = 0
        body = client.post("/predict", json=payload).json()
        assert body["status"] == "ok"
        assert body["risk_tier"] is not None

    def test_no_imputation_happens(self):
        """Guard the explicit instruction not to impute. If a null history
        feature were being filled with a prior/base rate, the cold vector's
        probability would converge toward the full-history vector's. It
        should not: nulls stay null into the model."""
        _requires_model()
        cold = client.post("/predict", json=_brand_new_repo_payload()).json()
        warm = client.post("/predict", json=VALID_PAYLOAD).json()
        assert cold["failure_probability"] != pytest.approx(warm["failure_probability"], abs=1e-6)


# =============================================================================
# Input validation (unchanged from Layer 1)
# =============================================================================

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
    @pytest.mark.parametrize("field", ALL_HISTORY_FEATURES)
    def test_omitting_history_feature_still_succeeds(self, field):
        _requires_model()
        r = client.post("/predict", json=_payload_without(field))
        assert r.status_code == 200, r.text

    @pytest.mark.parametrize("field", ALL_HISTORY_FEATURES)
    def test_explicit_null_history_feature_still_succeeds(self, field):
        _requires_model()
        payload = dict(VALID_PAYLOAD)
        payload[field] = None
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, r.text

    def test_first_build_in_project_all_five_missing_at_once(self):
        _requires_model()
        r = client.post("/predict", json=_brand_new_repo_payload())
        assert r.status_code == 200, r.text
        # As of Layer 1b this specific case is the cold_start state.
        assert r.json()["status"] == "cold_start"


# =============================================================================
# Tier boundaries
# =============================================================================

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
