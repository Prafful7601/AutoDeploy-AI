"""
Stage 3, Layer 2: given a live repo + commit, compute the same 31-feature
vector api/schema.py expects — causally, using only information available
at (or before) that commit's timestamp, from the GitHub REST API.

This is NOT a re-implementation of TravisTorrent's own feature computation
(that pipeline no longer exists and its exact per-feature logic was never
fully published). Every feature here is a best-effort live proxy, rated
EXACT / APPROXIMATED / UNAVAILABLE against what training actually used —
see build_feature_vector()'s returned `provenance` dict and
outputs/reports/stage3_feature_parity.md (generated from this module's own
PROVENANCE constants, not hand-maintained prose that could drift from the
code).

Two identities matter for the history features:
  - PROJECT history (previous_build_status, project_prior_failure_rate,
    consecutive_failure_streak): built from GitHub Actions workflow run
    conclusions for this repo, most-recent-first, filtered to before this
    commit's timestamp.
  - AUTHOR history: same run list, filtered to runs whose triggering actor
    matches this commit's GitHub-linked author login. If the commit has no
    linked GitHub account (common for commits authored with a non-GitHub
    email), author history is None — exactly like TravisTorrent's own
    9.4%-unmatched-author gap from Stage 1, not a new problem introduced
    here.

Nulls are never imputed here, on purpose — see api/coldstart.py. A feature
this module can't determine is either None (for the 5 features the API
allows null) or a clearly-labeled constant fallback (for required features
with no live source), never a silent 0.
"""

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .file_classification import classify_file
from .github_client import GitHubClient

logger = logging.getLogger("autodeploy_ai.extractor")

# --- Training-data constant fallbacks -----------------------------------
# Frozen medians from data/processed/builds_labeled.parquet at the time
# Stage 2 was trained (243 projects, ruby/java/go only — see the KNOWN
# LANGUAGE COVERAGE note below). Computed once, hardcoded here rather than
# read from the parquet at runtime, because Layer 3's GitHub Action runs
# in a fresh checkout that will not have data/processed/ available.
# Used ONLY for the 3 features with no live source at all (test density
# metrics) — see PROVENANCE for why these specific 3 and not others.
TRAINING_MEDIAN_BY_LANGUAGE = {
    "ruby": {"sloc": 5316, "test_lines_per_kloc": 972.96, "test_cases_per_kloc": 78.45, "asserts_per_kloc": 163.29},
    "java": {"sloc": 48541, "test_lines_per_kloc": 388.36, "test_cases_per_kloc": 11.68, "asserts_per_kloc": 48.43},
    "go": {"sloc": 4549, "test_lines_per_kloc": 274.59, "test_cases_per_kloc": 0.0, "asserts_per_kloc": 28.81},
}
TRAINING_MEDIAN_OVERALL = {"sloc": 11318, "test_lines_per_kloc": 670.85, "test_cases_per_kloc": 33.99, "asserts_per_kloc": 82.47}

# KNOWN LANGUAGE COVERAGE: the trained model has language_{ruby,java,go}
# columns only — zero python rows survived Stage 1's project-coverage
# filter (see api/schema.py's KNOWN_LANGUAGES and the bug that constant
# fixed). A repo whose GitHub-reported language isn't one of these 3 is
# passed through as-is (the API treats it as an unrecognized-language
# reference case, not an error).

CONCLUSION_TO_LABEL = {"success": 0, "failure": 1, "timed_out": 1}
# Everything else (cancelled, skipped, neutral, action_required, stale,
# None/in-progress) is ambiguous and excluded from the history sequence —
# the same policy Stage 1 applied to TravisTorrent's canceled/started rows.

MAX_HISTORY_PAGES = 5          # 500 workflow runs of lookback, per project and per author
MAX_TOUCHED_FILES_FOR_HOTNESS = 5  # commits_on_touched_files: capped, see PROVENANCE


@dataclass
class FeatureResult:
    features: Dict[str, Any]
    provenance: Dict[str, Dict[str, str]]  # feature -> {level, note}
    raw: Dict[str, Any]  # debugging: fetched commit/repo/run data


def _iso(d: dt.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_gh_datetime(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def _label_run(run: dict) -> Optional[int]:
    return CONCLUSION_TO_LABEL.get(run.get("conclusion"))


def _history_from_runs(runs: List[dict], actor_login: Optional[str] = None):
    """runs: most-recent-first, already filtered to before the target
    commit's timestamp. If actor_login is given, further filters to that
    actor before computing. Returns (prior_status, prior_failure_rate,
    prior_build_count, consecutive_streak, days_since_last)."""
    if actor_login is not None:
        runs = [r for r in runs if (r.get("actor") or {}).get("login") == actor_login]

    labeled = [(r, _label_run(r)) for r in runs]
    resolved = [(r, lab) for r, lab in labeled if lab is not None]

    if not resolved:
        return None, None, 0, 0, None

    prior_status = resolved[0][1]
    prior_failure_rate = sum(lab for _, lab in resolved) / len(resolved)
    prior_build_count = len(resolved)

    streak = 0
    for _, lab in resolved:  # most-recent-first: count leading failures
        if lab == 1:
            streak += 1
        else:
            break

    last_run_time = _parse_gh_datetime(resolved[0][0]["created_at"])
    return prior_status, prior_failure_rate, prior_build_count, streak, last_run_time


def build_feature_vector(
    owner: str,
    repo: str,
    sha: str,
    branch: Optional[str] = None,
    is_pr: Optional[bool] = None,
    client: Optional[GitHubClient] = None,
) -> FeatureResult:
    client = client or GitHubClient()
    features: Dict[str, Any] = {}
    provenance: Dict[str, Dict[str, str]] = {}
    raw: Dict[str, Any] = {}

    def set_feature(name, value, level, note):
        features[name] = value
        provenance[name] = {"level": level, "note": note}

    # --- Fetch the commit and repo -----------------------------------
    commit = client.get_json(f"/repos/{owner}/{repo}/commits/{sha}")
    repo_meta = client.get_json(f"/repos/{owner}/{repo}")
    raw["commit"] = commit
    raw["repo"] = repo_meta

    commit_date = _parse_gh_datetime(commit["commit"]["author"]["date"])
    commit_date_iso = _iso(commit_date)
    author_login = (commit.get("author") or {}).get("login")  # None if not GitHub-linked

    # --- Change size: from the commit diff ----------------------------
    files = commit.get("files", [])
    classified = [(f, classify_file(f["filename"])) for f in files]
    src_files = [f for f, c in classified if c == "src"]
    doc_files = [f for f, c in classified if c == "doc"]
    test_files = [f for f, c in classified if c == "test"]

    src_churn = sum(f["additions"] + f["deletions"] for f in src_files)
    set_feature("src_churn", src_churn, "APPROXIMATED",
                 "Sum of additions+deletions over files classified 'src' by a path heuristic "
                 "(extractor/file_classification.py), not TravisTorrent's original per-language "
                 "diff tool. Config/CI/build files not caught by the doc/test heuristics are "
                 "counted as src here (TravisTorrent had a separate 'other' bucket) — see doc/other split below.")

    files_added = sum(1 for f in files if f["status"] == "added")
    files_deleted = sum(1 for f in files if f["status"] == "removed")
    files_modified = sum(1 for f in files if f["status"] in ("modified", "renamed", "changed"))
    set_feature("files_added", files_added, "EXACT", "Direct from the commit's file-status list.")
    set_feature("files_deleted", files_deleted, "EXACT", "Direct from the commit's file-status list.")
    set_feature("files_modified", files_modified, "EXACT", "Direct from the commit's file-status list.")
    set_feature("total_files_changed", len(files), "EXACT", "len(commit.files).")

    set_feature("src_files_changed", len(src_files), "APPROXIMATED", "Path-heuristic classification; see src_churn note.")
    set_feature("doc_files_changed", len(doc_files), "APPROXIMATED", "Path-heuristic classification (extension/dirname match).")
    # No live 'other' bucket (config/CI/generated files) — see file_classification.py.
    set_feature("other_files_changed", 0, "UNAVAILABLE",
                "The classifier has no 'other' bucket (build config, generated files, binary "
                "assets) — everything not test/doc is counted as src instead. Always 0 live; "
                "training median was low-single-digits, so this mostly shifts a little weight "
                "from other_files_changed onto src_files_changed rather than losing information "
                "outright. SHAP rank 20/33 (0.7% share) — low-impact.")

    tests_added_files = sum(1 for f, c in classified if c == "test" and f["status"] == "added")
    tests_deleted_files = sum(1 for f, c in classified if c == "test" and f["status"] == "removed")
    set_feature("tests_added", tests_added_files, "APPROXIMATED",
                "TravisTorrent counted test CASES added (test-level diff parsing). This counts "
                "whole NEW test FILES only — a commit that adds test cases inside an existing "
                "test file (a 'modified' status file) contributes 0 here. Systematically "
                "undercounts the common case; SHAP rank 30/33 (0.0% share) in training, so this "
                "gap costs little live signal despite being the least faithful approximation.")
    set_feature("tests_deleted", tests_deleted_files, "APPROXIMATED", "Same file-level-not-case-level gap as tests_added. SHAP rank 22/33 (0.4% share).")

    total_changed = len(files)
    test_touch = tests_added_files + tests_deleted_files
    test_file_ratio = round(test_touch / total_changed, 4) if total_changed > 0 else 0.0
    set_feature("test_file_ratio", test_file_ratio, "APPROXIMATED",
                "Derived from the approximated tests_added/tests_deleted above, so it inherits "
                "that gap. Note: 0.0 here (not null) when total_changed is 0 — matches the API's "
                "required, non-nullable schema for this field (see api/schema.py).")

    set_feature("num_commits_in_build", 1, "EXACT",
                "This extractor scores one specific commit SHA. Matches how GitHub Actions "
                "actually triggers builds (per push-head or per-PR-head commit) — unlike old "
                "Travis CI, which sometimes batched several commits into one build "
                "(git_num_all_built_commits could exceed 1 in training).")

    # commits_on_touched_files: sum, over up to MAX_TOUCHED_FILES_FOR_HOTNESS
    # changed files, of how many commits (on the default branch, up to this
    # commit's time) have touched that file before. Capped for API cost —
    # a commit touching 40 files would otherwise cost 40 extra requests.
    touched_for_hotness = [f["filename"] for f in files[:MAX_TOUCHED_FILES_FOR_HOTNESS]]
    hotness_total = 0
    hotness_failures = []
    for file_path in touched_for_hotness:
        try:
            hotness_total += client.count_via_last_page(
                f"/repos/{owner}/{repo}/commits", path=file_path, until=commit_date_iso
            )
        except Exception as exc:
            # One file's history failing shouldn't sink the whole extraction,
            # but it must not silently look identical to "this file has no
            # history" either — logged AND surfaced in the provenance note.
            logger.warning("commits_on_touched_files: failed for %s: %r", file_path, exc)
            hotness_failures.append(file_path)
    hotness_note = (
        f"Sum of per-file commit-history counts (via GitHub's Link-header pagination "
        f"trick), capped to the first {MAX_TOUCHED_FILES_FOR_HOTNESS} changed files for "
        f"API cost — this commit touched {len(files)}. TravisTorrent's exact aggregation "
        "method (sum vs. max vs. something else) was never published, so 'sum' here is "
        "a judgment call, not a replication."
    )
    if hotness_failures:
        hotness_note += (
            f" WARNING: the per-file lookup failed for {len(hotness_failures)} of "
            f"{len(touched_for_hotness)} files ({hotness_failures}) — the total above is "
            "a partial sum, undercounting. See logs for the underlying error."
        )
    set_feature("commits_on_touched_files", hotness_total,
                "APPROXIMATED" if not hotness_failures else "APPROXIMATED (partial — see note)",
                hotness_note)

    # --- Project/build context ----------------------------------------
    repo_created = _parse_gh_datetime(repo_meta["created_at"])
    repo_age_days = (commit_date - repo_created).days
    set_feature("repo_age_days", repo_age_days, "EXACT", "(commit timestamp - repo.created_at), both from the GitHub API.")

    repo_num_commits = client.count_via_last_page(f"/repos/{owner}/{repo}/commits", until=commit_date_iso)
    set_feature("repo_num_commits", repo_num_commits, "APPROXIMATED",
                "Count of commits on the repo's default branch up to this commit's timestamp, "
                "via the Link-header page-count trick. Default-branch-only; TravisTorrent's "
                "exact scope (all branches vs. default) isn't documented.")

    team_size = client.count_via_last_page(f"/repos/{owner}/{repo}/contributors", anon="true")
    set_feature("team_size", max(team_size, 1), "APPROXIMATED",
                "Count of all-time contributors as of NOW, not as of this commit's timestamp — "
                "the GitHub contributors endpoint has no 'until' parameter. For the actual "
                "Layer 3 use case (scoring the commit that just triggered the Action, i.e. "
                "'now') this is exact by construction, since there is no future yet. Extracting "
                "features for an older historical commit with this same call WOULD leak future "
                "contributors — noted, not hidden.")

    language = (repo_meta.get("language") or "").strip().lower()
    set_feature("language", language, "APPROXIMATED",
                "GitHub's single repo-wide 'primary language' (linguist byte-count detection), "
                "not a per-commit language tag. Passed through as-is even if it isn't one of "
                "this model's 3 known languages (ruby/java/go) — the API already handles that "
                "gracefully as the unrecognized-language reference case, not an error.")

    if is_pr is not None:
        resolved_is_pr = int(is_pr)
        is_pr_level, is_pr_note = "EXACT", "Provided directly by the caller (e.g. Layer 3's GitHub Actions event context knows this without an API call)."
    else:
        prs = client.get_json_or_none(f"/repos/{owner}/{repo}/commits/{sha}/pulls") or []
        resolved_is_pr = int(len(prs) > 0)
        is_pr_level, is_pr_note = "APPROXIMATED", "No branch/event context given, so derived from whether GitHub associates any pull request with this commit — can lag if the PR was opened after the commit, or reflect a since-merged PR rather than 'this build ran as a PR check'."
    set_feature("is_pr", resolved_is_pr, is_pr_level, is_pr_note)

    if branch is not None:
        resolved_main = int(branch in ("master", "main", "trunk"))
        branch_level, branch_note = "EXACT", "Branch provided directly by the caller (Layer 3 knows this from github.ref_name)."
    else:
        heads = client.get_json_or_none(f"/repos/{owner}/{repo}/commits/{sha}/branches-where-head") or []
        head_names = {b["name"] for b in heads}
        resolved_main = int(bool(head_names & {"master", "main", "trunk"}))
        branch_level = "APPROXIMATED"
        branch_note = ("No branch given, so derived from GitHub's 'branches where this commit is "
                       "the tip' — accurate for the current HEAD of a branch, but returns empty "
                       "for a commit that's since been superseded, even if it was on master once.")
    set_feature("is_main_branch", resolved_main, branch_level, branch_note)

    by_core_team_member = 0
    core_note = "UNAVAILABLE: no live source for TravisTorrent's exact 'core team member' definition."
    if author_login:
        try:
            contributors = client.get_all_pages(f"/repos/{owner}/{repo}/contributors", max_pages=2, anon="false")
            ranked = sorted(contributors, key=lambda c: -c.get("contributions", 0))
            top_logins = {c["login"] for c in ranked[: max(1, len(ranked) // 2)]}  # top half, by contribution count
            by_core_team_member = int(author_login in top_logins)
            core_note = ("APPROXIMATED: 'core' defined here as top half of all-time contributors "
                         "by commit count (current, not as-of-commit — same look-ahead caveat as "
                         "team_size). TravisTorrent's actual threshold was never published.")
        except Exception as exc:
            logger.warning("by_core_team_member: contributor lookup failed: %r", exc)
            core_note = f"UNAVAILABLE: contributor lookup failed ({exc!r}) — defaulted to 0, not a confirmed 'not core'."
    set_feature("by_core_team_member", by_core_team_member,
                "APPROXIMATED" if author_login else "UNAVAILABLE", core_note)

    for name in ("sloc", "test_lines_per_kloc", "test_cases_per_kloc", "asserts_per_kloc"):
        table = TRAINING_MEDIAN_BY_LANGUAGE.get(language, TRAINING_MEDIAN_OVERALL)
        value = table[name] if language in TRAINING_MEDIAN_BY_LANGUAGE else TRAINING_MEDIAN_OVERALL[name]
        set_feature(name, value, "UNAVAILABLE",
                    f"No live source without cloning the repo and running static analysis "
                    f"(lines-of-code / test-case / assertion counting). Fallback: Stage 2's "
                    f"training-data median for language='{language}' "
                    f"({'per-language table' if language in TRAINING_MEDIAN_BY_LANGUAGE else 'overall, language unrecognized'}) "
                    f"= {value}. This is a CONSTANT per language — it will not vary across live "
                    f"commits or repos of the same language, removing this feature's ability to "
                    f"differentiate predictions live. Acceptable given its modest training "
                    f"importance (SHAP shares: sloc 1.2%, test_lines_per_kloc 2.1%, "
                    f"test_cases_per_kloc 1.0%, asserts_per_kloc 0.8% — none in the top 6).")

    # --- History (project + author) ------------------------------------
    runs = client.get_all_pages(
        f"/repos/{owner}/{repo}/actions/runs", max_pages=MAX_HISTORY_PAGES,
        items_key="workflow_runs", created=f"<{commit_date_iso}"
    )
    raw["prior_runs_fetched"] = len(runs)

    p_status, p_rate, p_count, p_streak, _ = _history_from_runs(runs)
    proj_note = (
        f"From GitHub Actions workflow run conclusions (created < this commit's timestamp), "
        f"any branch — matching Stage 2's own judgment call to use wall-clock project order "
        f"rather than a branch-scoped or TravisTorrent-internal notion of 'previous build'. "
        f"Lookback capped at {MAX_HISTORY_PAGES * 100} most recent runs. Ambiguous conclusions "
        f"(cancelled/skipped/neutral/in-progress) excluded from the sequence, same policy Stage 1 "
        f"applied to TravisTorrent's canceled/started rows. NOTE: this is Actions run history, "
        f"which only exists from whenever Actions was enabled on this repo — NOT the repo's full "
        f"commit/build history the way TravisTorrent's per-project history was. A repo with a "
        f"10-year git history but Actions enabled last month will look history-cold-start here "
        f"even though it is an old, stable project. This is a real, structural gap, not a bug."
    )
    set_feature("previous_build_status", p_status,
                "APPROXIMATED" if p_status is not None else "UNAVAILABLE (correctly null -> triggers cold_start)",
                proj_note + " NULL HERE TRIGGERS THE API'S cold_start RESPONSE STATE (see api/coldstart.py) — this is the intended, honest behavior for a repo with no qualifying prior Actions run, not a failure of extraction.")
    set_feature("project_prior_failure_rate", p_rate,
                "APPROXIMATED" if p_rate is not None else "UNAVAILABLE (correctly null -> triggers cold_start)",
                proj_note + " NULL HERE TRIGGERS cold_start, same as previous_build_status.")
    set_feature("project_prior_build_count", p_count, "APPROXIMATED", proj_note)
    set_feature("consecutive_failure_streak", p_streak, "APPROXIMATED", proj_note)

    if author_login:
        a_status, a_rate, a_count, a_streak, a_last_time = _history_from_runs(runs, actor_login=author_login)
        author_note = (
            f"Author identity = the commit's GitHub-linked account ('{author_login}'), matched "
            f"against each prior run's triggering actor. Same lookback/exclusion caveats as "
            f"project history above. NULL HERE DOES NOT TRIGGER cold_start (see api/coldstart.py "
            "- author-history absence alone is routine and the model handles it well)."
        )
        a_days_since = (commit_date - a_last_time).total_seconds() / 86400.0 if a_last_time else None
        set_feature("author_prior_builds_in_project", a_count, "APPROXIMATED", author_note)
        set_feature("author_prior_failure_rate_in_project", a_rate, "APPROXIMATED", author_note)
        set_feature("author_days_since_last_build_in_project", a_days_since, "APPROXIMATED", author_note)
    else:
        no_author_note = (
            "This commit has no GitHub-linked author account (commit email doesn't match any "
            "GitHub user), so author identity can't be established at all — correctly left null, "
            "the same live-equivalent of TravisTorrent's own 9.4% unmatched-author gap from "
            "Stage 1, not a new problem introduced live. Does NOT trigger cold_start."
        )
        set_feature("author_prior_builds_in_project", None, "UNAVAILABLE (author unidentifiable)", no_author_note)
        set_feature("author_prior_failure_rate_in_project", None, "UNAVAILABLE (author unidentifiable)", no_author_note)
        set_feature("author_days_since_last_build_in_project", None, "UNAVAILABLE (author unidentifiable)", no_author_note)

    return FeatureResult(features=features, provenance=provenance, raw=raw)
