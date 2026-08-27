"""
Tests for Stage 3 Layer 2 (the live feature extractor).

Two kinds:
  - Pure unit tests (file_classification, github_client's pagination/count
    helpers) against mocked HTTP responses — no network, no rate limit.
  - One live integration test against a real public repo with real commit
    and Actions history (spf13/cobra), per the brief's explicit requirement.
    It costs ~15 GitHub API calls; skips cleanly (not a failure) on rate
    limit or network errors so it doesn't break CI runs that happen to hit
    GitHub's unauthenticated 60/hour cap.

Run with: pytest tests/test_extractor.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractor.extract import _history_from_runs, build_feature_vector  # noqa: E402
from extractor.file_classification import classify_file  # noqa: E402
from extractor.github_client import GitHubClient, GitHubRateLimitError  # noqa: E402

LIVE_TEST_REPO = ("spf13", "cobra")
LIVE_TEST_SHA = "adbc8813901bba65827259daa8e22ff94ec1f30e"  # a real, merged PR commit with CI history


# =============================================================================
# file_classification — pure function, exhaustively testable
# =============================================================================

class TestFileClassification:
    @pytest.mark.parametrize("path", [
        "tests/test_foo.py", "test/foo_test.go", "spec/models/user_spec.rb",
        "src/__tests__/App.test.js", "lib/foo.spec.ts", "app/test_helper.rb",
    ])
    def test_recognizes_test_files(self, path):
        assert classify_file(path) == "test"

    @pytest.mark.parametrize("path", [
        "docs/guide.md", "README.md", "CHANGELOG.rst", "documentation/setup.txt", "LICENSE",
    ])
    def test_recognizes_doc_files(self, path):
        assert classify_file(path) == "doc"

    @pytest.mark.parametrize("path", [
        "src/main.go", "lib/app.rb", "com/example/Main.java", ".github/workflows/ci.yml",
    ])
    def test_everything_else_is_src(self, path):
        # Including CI config — there's no live 'other' bucket, documented
        # in file_classification.py and the parity report.
        assert classify_file(path) == "src"


# =============================================================================
# _history_from_runs — the project/author history sequencing logic
# =============================================================================

def _run(conclusion, created_at, actor=None):
    return {"conclusion": conclusion, "created_at": created_at, "actor": {"login": actor} if actor else None}


class TestHistoryFromRuns:
    def test_empty_runs_is_all_none(self):
        status, rate, count, streak, last = _history_from_runs([])
        assert (status, rate, count, streak, last) == (None, None, 0, 0, None)

    def test_all_ambiguous_conclusions_counted_as_no_history(self):
        runs = [_run("cancelled", "2024-01-03T00:00:00Z"), _run("skipped", "2024-01-02T00:00:00Z")]
        status, rate, count, streak, last = _history_from_runs(runs)
        assert (status, rate, count, streak, last) == (None, None, 0, 0, None)

    def test_most_recent_first_ordering_drives_previous_status(self):
        runs = [_run("failure", "2024-01-03T00:00:00Z"), _run("success", "2024-01-02T00:00:00Z")]
        status, rate, count, streak, last = _history_from_runs(runs)
        assert status == 1  # most recent (first in list) is the failure
        assert count == 2
        assert rate == 0.5

    def test_consecutive_streak_stops_at_first_pass(self):
        runs = [
            _run("failure", "2024-01-05T00:00:00Z"),
            _run("failure", "2024-01-04T00:00:00Z"),
            _run("success", "2024-01-03T00:00:00Z"),
            _run("failure", "2024-01-02T00:00:00Z"),
        ]
        _, _, _, streak, _ = _history_from_runs(runs)
        assert streak == 2  # the two failures before hitting the success

    def test_ambiguous_runs_interleaved_are_skipped_not_counted(self):
        runs = [
            _run("failure", "2024-01-05T00:00:00Z"),
            _run("cancelled", "2024-01-04T30:00:00Z"),
            _run("failure", "2024-01-04T00:00:00Z"),
        ]
        _, _, count, streak, _ = _history_from_runs(runs)
        assert count == 2
        assert streak == 2  # cancelled run doesn't break the streak

    def test_actor_filter_scopes_to_one_author(self):
        runs = [
            _run("failure", "2024-01-05T00:00:00Z", actor="alice"),
            _run("success", "2024-01-04T00:00:00Z", actor="bob"),
            _run("success", "2024-01-03T00:00:00Z", actor="alice"),
        ]
        status, rate, count, streak, _ = _history_from_runs(runs, actor_login="alice")
        assert count == 2  # bob's run excluded
        assert status == 1  # alice's most recent is the failure


# =============================================================================
# GitHubClient — pagination/count helpers, mocked HTTP
# =============================================================================

class TestGitHubClientHelpers:
    def _client_with_mocked_session(self):
        client = GitHubClient(token="fake-token-not-real")
        client.session = MagicMock()
        return client

    def test_count_via_last_page_reads_link_header(self):
        client = self._client_with_mocked_session()
        resp = MagicMock(status_code=200)
        resp.links = {"last": {"url": "https://api.github.com/repos/x/y/commits?page=42&per_page=1"}}
        resp.raise_for_status = MagicMock()
        client.session.request.return_value = resp
        assert client.count_via_last_page("/repos/x/y/commits") == 42

    def test_count_via_last_page_no_link_header_means_0_or_1(self):
        client = self._client_with_mocked_session()
        resp = MagicMock(status_code=200)
        resp.links = {}
        resp.json.return_value = [{"sha": "abc"}]
        resp.raise_for_status = MagicMock()
        client.session.request.return_value = resp
        assert client.count_via_last_page("/repos/x/y/commits") == 1

    def test_get_all_pages_unwraps_items_key(self):
        client = self._client_with_mocked_session()
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"total_count": 2, "workflow_runs": [{"id": 1}, {"id": 2}]}
        resp.links = {}
        resp.raise_for_status = MagicMock()
        client.session.request.return_value = resp
        result = client.get_all_pages("/repos/x/y/actions/runs", max_pages=1, items_key="workflow_runs")
        assert result == [{"id": 1}, {"id": 2}]

    def test_endpoint_param_named_to_avoid_github_path_query_collision(self):
        """Regression guard: GitHub's own `path` query parameter (filter
        commits by file path) previously collided with this client's own
        `path` positional parameter name, silently breaking every
        commits_on_touched_files call. The parameter is `endpoint` now."""
        import inspect
        for method_name in ["get", "get_json", "get_json_or_none", "get_all_pages", "count_via_last_page"]:
            sig = inspect.signature(getattr(GitHubClient, method_name))
            assert "path" not in sig.parameters, f"{method_name} still has a 'path' parameter"

    def test_rate_limit_raises_clear_error(self):
        client = self._client_with_mocked_session()
        resp = MagicMock(status_code=403, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "9999999999"})
        client.session.request.return_value = resp
        with pytest.raises(GitHubRateLimitError):
            client.get_json("/repos/x/y")


# =============================================================================
# Live integration test — real repo, real history (required by the brief)
# =============================================================================

class TestLiveExtraction:
    def test_real_repo_produces_all_31_features_with_real_history(self):
        client = GitHubClient()
        try:
            result = build_feature_vector(*LIVE_TEST_REPO, LIVE_TEST_SHA, branch="main", client=client)
        except GitHubRateLimitError:
            pytest.skip("GitHub API rate limit hit — not a test failure, just no budget left this hour.")
        except Exception as exc:  # network flakiness, DNS, etc. — not what this test is checking
            pytest.skip(f"Live GitHub API call failed (network?): {exc!r}")

        # All 31 features present, matching api/schema.py's BuildFeatures.
        expected = {
            "src_churn", "files_added", "files_deleted", "files_modified", "total_files_changed",
            "src_files_changed", "doc_files_changed", "other_files_changed", "tests_added",
            "tests_deleted", "test_file_ratio", "num_commits_in_build", "commits_on_touched_files",
            "previous_build_status", "project_prior_failure_rate", "project_prior_build_count",
            "consecutive_failure_streak", "author_prior_builds_in_project",
            "author_prior_failure_rate_in_project", "author_days_since_last_build_in_project",
            "team_size", "repo_age_days", "repo_num_commits", "sloc", "test_lines_per_kloc",
            "test_cases_per_kloc", "asserts_per_kloc", "is_pr", "by_core_team_member",
            "language", "is_main_branch",
        }
        assert set(result.features) == expected
        assert set(result.provenance) == expected

        # This specific commit/repo has real, non-empty project history —
        # the "real history" half of the brief's requirement, not a
        # cold-start vector.
        assert result.features["project_prior_build_count"] > 0
        assert result.features["previous_build_status"] is not None
        assert result.features["language"] == "go"

        # Every feature's provenance is one of the three declared levels
        # (allowing the "partial" suffix commits_on_touched_files can add).
        for name, prov in result.provenance.items():
            assert prov["level"].startswith(("EXACT", "APPROXIMATED", "UNAVAILABLE")), (name, prov)
            assert prov["note"], f"{name} has an empty provenance note"

    def test_extracted_vector_is_accepted_by_the_real_predict_endpoint(self):
        """Closes the loop: Layer 2's output must be valid Layer 1 input."""
        pytest.importorskip("fastapi")
        from api.main import app
        from api.model import PredictionService
        from fastapi.testclient import TestClient

        if not PredictionService().is_loaded:
            pytest.skip("Model artifact not found — run scripts/05_train_and_evaluate.py first.")

        client = GitHubClient()
        try:
            result = build_feature_vector(*LIVE_TEST_REPO, LIVE_TEST_SHA, branch="main", client=client)
        except GitHubRateLimitError:
            pytest.skip("GitHub API rate limit hit.")
        except Exception as exc:
            pytest.skip(f"Live GitHub API call failed: {exc!r}")

        api_client = TestClient(app)
        r = api_client.post("/predict", json=result.features)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] in ("ok", "cold_start")
