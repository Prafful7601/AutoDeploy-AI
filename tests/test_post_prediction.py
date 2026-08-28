"""
Tests for the Stage 3 Layer 3 GitHub Action script (.github/scripts/post_prediction.py).

These enforce the brief's hard requirements structurally, not just by
inspection: the experimental banner must lead every output, a probability
must never appear before it, cold_start must never show a tier, and every
one of the model's actual feature columns must have a plain-language
explanation (regression guard for the language_* one-hot bug found while
building this).
"""

import json
import re
import sys
from pathlib import Path

import joblib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

import post_prediction as pp  # noqa: E402
from api.model import MODEL_PATH, PredictionService  # noqa: E402
from api.schema import BuildFeatures  # noqa: E402

VALID_PAYLOAD = {
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

COLD_PAYLOAD = {
    "src_churn": 5, "files_added": 0, "files_deleted": 0, "files_modified": 1,
    "total_files_changed": 1, "src_files_changed": 1, "doc_files_changed": 0,
    "other_files_changed": 0, "tests_added": 0, "tests_deleted": 0,
    "test_file_ratio": 0.0, "num_commits_in_build": 1, "commits_on_touched_files": 0,
    "project_prior_build_count": 0, "consecutive_failure_streak": 0,
    "team_size": 3, "repo_age_days": 400, "repo_num_commits": 50, "sloc": 3000,
    "test_lines_per_kloc": 100, "test_cases_per_kloc": 5, "asserts_per_kloc": 10,
    "is_pr": 0, "by_core_team_member": 1, "language": "go", "is_main_branch": 1,
}


def _requires_model():
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not found — run scripts/05_train_and_evaluate.py first.")


class TestExplainDriverCoverage:
    def test_every_actual_model_feature_has_a_mapping(self):
        """Regression guard for the language_* one-hot bug: SHAP attributes
        importance to the model's ACTUAL columns (e.g. language_go), not
        the pre-encoding 'language' field. Every real column must resolve
        to something other than the raw-name fallback."""
        _requires_model()
        bundle = joblib.load(MODEL_PATH)
        for feature in bundle["feature_cols"]:
            text = pp.explain_driver({"feature": feature, "feature_value": 5.0, "direction": "increases risk"})
            assert "no plain-language mapping" not in text, f"{feature} has no plain-language mapping"

    def test_unmapped_feature_falls_back_safely_not_crashes(self):
        text = pp.explain_driver({"feature": "totally_made_up_feature", "feature_value": 1.0, "direction": "increases risk"})
        assert "totally_made_up_feature" in text

    def test_language_dummy_pattern_match(self):
        text = pp.explain_driver({"feature": "language_go", "feature_value": 1.0, "direction": "increases risk"})
        assert "Go" in text and "higher" in text


class TestBannerAndHeadlineOrdering:
    """Structural enforcement, not just inspection: the banner must lead,
    and a bare percentage must never appear before it (i.e. never as a
    headline)."""

    def _first_percentage_index(self, text):
        m = re.search(r"\d+%", text)
        return m.start() if m else None

    def test_ok_comment_banner_precedes_any_percentage(self):
        _requires_model()
        service = PredictionService()
        prediction = service.predict(BuildFeatures(**VALID_PAYLOAD))
        body = pp.compose_ok_comment(prediction, VALID_PAYLOAD)
        assert body.startswith("> ⚠️ **Experimental")
        pct_index = self._first_percentage_index(body)
        assert pct_index is not None, "expected the probability to appear somewhere"
        assert pct_index > body.index("###"), "a percentage appeared before the explanation section — looks like a headline"

    def test_ok_comment_predicted_probability_is_wrapped_in_sub_tag(self):
        """The MODEL'S OWN predicted probability must be visually
        de-emphasized — not the same claim as 'no percentage appears
        anywhere': a driver explanation legitimately citing a historical
        failure rate (e.g. 'elevated, 35%') is supporting detail, not the
        prediction itself, and is fine to state plainly."""
        _requires_model()
        service = PredictionService()
        prediction = service.predict(BuildFeatures(**VALID_PAYLOAD))
        body = pp.compose_ok_comment(prediction, VALID_PAYLOAD)
        predicted_pct = f"{prediction['failure_probability']:.0%}"
        sub_start = body.index("<sub>")
        sub_end = body.index("</sub>")
        pct_index = body.index(predicted_pct)
        assert sub_start < pct_index < sub_end, "the predicted probability itself must be inside the de-emphasized <sub> block"

    def test_ok_comment_never_exposes_internal_confidence_label(self):
        """probability_confidence='normal' must not appear verbatim next to
        'experimental' — would read as reassurance it's trustworthy."""
        _requires_model()
        service = PredictionService()
        prediction = service.predict(BuildFeatures(**VALID_PAYLOAD))
        assert prediction["probability_confidence"] == "normal"  # sanity: this IS the internal state
        body = pp.compose_ok_comment(prediction, VALID_PAYLOAD)
        assert "confidence: normal" not in body
        assert "low-confidence" in body  # the OUTWARD label is always low-confidence

    def test_cold_start_comment_has_no_risk_tier_language(self):
        service = PredictionService()
        prediction = service.predict(BuildFeatures(**COLD_PAYLOAD))
        assert prediction["status"] == "cold_start"
        body = pp.compose_cold_start_comment(prediction, COLD_PAYLOAD)
        assert body.startswith("> ⚠️ **Experimental")
        assert "No risk tier is being shown" in body
        for tier_word in ("**Low**", "**Medium**", "**High**"):
            assert tier_word not in body
        assert self._first_percentage_index(body) is None, "cold_start must never show a probability"

    def test_could_not_score_has_no_tier_or_probability(self):
        body = pp.compose_could_not_score("rate_limit", "some detail")
        assert body.startswith("> ⚠️ **Experimental")
        assert self._first_percentage_index(body) is None
        for tier_word in ("Low", "Medium", "High"):
            assert tier_word not in body


class TestShallowHistoryNote:
    def test_detects_old_repo_thin_history(self):
        note = pp.shallow_history_note({"project_prior_build_count": 2, "repo_age_days": 1000})
        assert "Shallow history detected" in note
        assert "1000" in note

    def test_no_alarm_for_genuinely_new_repo(self):
        note = pp.shallow_history_note({"project_prior_build_count": 0, "repo_age_days": 5})
        assert "Shallow history detected" not in note
        assert "Live build history" in note  # standing caveat still present

    def test_no_alarm_for_established_repo_with_real_history(self):
        note = pp.shallow_history_note({"project_prior_build_count": 300, "repo_age_days": 2000})
        assert "Shallow history detected" not in note


class TestEventContext:
    def test_push_event_uses_env_vars(self, monkeypatch):
        monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
        monkeypatch.setenv("GITHUB_REPOSITORY", "acme/widgets")
        monkeypatch.setenv("GITHUB_SHA", "abc123")
        monkeypatch.setenv("GITHUB_REF_NAME", "main")
        monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
        ctx = pp.get_event_context()
        assert ctx == {"owner": "acme", "repo": "widgets", "sha": "abc123", "branch": "main", "is_pr": False, "pr_number": None}

    def test_pull_request_event_uses_head_sha_not_merge_sha(self, monkeypatch, tmp_path):
        event = {"pull_request": {"head": {"sha": "headsha123", "ref": "feature-branch"}, "number": 42}}
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps(event))
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        monkeypatch.setenv("GITHUB_REPOSITORY", "acme/widgets")
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
        # Deliberately NOT setting GITHUB_SHA to a merge commit — the point
        # is that pull_request events must use the PR head SHA, not it.
        ctx = pp.get_event_context()
        assert ctx["sha"] == "headsha123"
        assert ctx["branch"] == "feature-branch"
        assert ctx["is_pr"] is True
        assert ctx["pr_number"] == 42


class TestStatusDescription:
    def test_all_kinds_fit_within_github_140_char_limit(self):
        service = PredictionService()
        ok_prediction = service.predict(BuildFeatures(**VALID_PAYLOAD)) if MODEL_PATH.exists() else {"risk_tier": "High"}
        for kind, result in [("ok", ok_prediction), ("cold_start", None), ("could_not_score", None)]:
            desc = pp.status_description(kind, result)
            assert len(desc) <= 140
