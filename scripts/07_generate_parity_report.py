"""
Stage 3, Layer 2 deliverable: the feature-parity report.

For every one of the 31 training features, classifies the live extractor's
source as EXACT / APPROXIMATED / UNAVAILABLE against what Stage 1/2
actually used, states plainly what happens when it can't be gotten live
(NaN, a documented constant, or a coarser proxy), and what that costs.

PARITY_TABLE below is the authoritative content — a static table, not
dependent on a live API call succeeding, because this report is a required
deliverable per the brief ("not optional") and must not silently degrade
if GitHub's rate limit is exhausted when this script runs. Every note is
copied verbatim from extractor/extract.py's own set_feature(...) calls at
the time this was written; if the two drift, extract.py is correct and
this table is stale — check there first.

The "worked example" section is supplementary evidence, not the
specification: it attempts a fresh live extraction, and falls back to a
frozen fixture (extractor/fixtures/example_extraction.json, captured
during this project's own development) if the environment has no
GITHUB_TOKEN or the unauthenticated 60/hour rate limit is exhausted.
Either way it's a REAL extraction, never fabricated numbers — only which
run produced it varies.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR  # noqa: E402

from extractor.extract import MAX_HISTORY_PAGES, MAX_TOUCHED_FILES_FOR_HOTNESS, build_feature_vector  # noqa: E402
from extractor.github_client import GitHubClient, GitHubRateLimitError  # noqa: E402

REPORT_PATH = REPORTS_DIR / "stage3_feature_parity.md"
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "extractor" / "fixtures" / "example_extraction.json"
EXAMPLE_OWNER, EXAMPLE_REPO, EXAMPLE_SHA, EXAMPLE_BRANCH = (
    "spf13", "cobra", "adbc8813901bba65827259daa8e22ff94ec1f30e", "main",
)

PROJECT_HISTORY_FEATURES = ("previous_build_status", "project_prior_failure_rate")
AUTHOR_HISTORY_FEATURES = (
    "author_prior_builds_in_project", "author_prior_failure_rate_in_project",
    "author_days_since_last_build_in_project",
)

# (feature, category, level, note) — verbatim from extractor/extract.py.
PARITY_TABLE = [
    ("src_churn", "Change size", "APPROXIMATED",
     "Sum of additions+deletions over files classified 'src' by a path heuristic "
     "(extractor/file_classification.py), not TravisTorrent's original per-language diff "
     "tool. Config/CI/build files not caught by doc/test heuristics are counted as src here "
     "(TravisTorrent had a separate 'other' bucket)."),
    ("files_added", "Change size", "EXACT", "Direct from the commit's file-status list."),
    ("files_deleted", "Change size", "EXACT", "Direct from the commit's file-status list."),
    ("files_modified", "Change size", "EXACT", "Direct from the commit's file-status list."),
    ("total_files_changed", "Change size", "EXACT", "len(commit.files)."),
    ("src_files_changed", "Change size", "APPROXIMATED", "Path-heuristic classification; see src_churn note."),
    ("doc_files_changed", "Change size", "APPROXIMATED", "Path-heuristic classification (extension/dirname match)."),
    ("other_files_changed", "Change size", "UNAVAILABLE",
     "The classifier has no 'other' bucket (build config, generated files, binary assets) — "
     "everything not test/doc is counted as src instead. Always 0 live. SHAP rank 20/33 "
     "(0.7% share) in training — low-impact."),
    ("tests_added", "Change size", "APPROXIMATED",
     "TravisTorrent counted test CASES added (test-level diff parsing). This counts whole NEW "
     "test FILES only — a commit adding test cases inside an existing test file contributes 0. "
     "Systematically undercounts the common case. SHAP rank 30/33 (0.0% share) in training."),
    ("tests_deleted", "Change size", "APPROXIMATED",
     "Same file-level-not-case-level gap as tests_added. SHAP rank 22/33 (0.4% share)."),
    ("test_file_ratio", "Change size", "APPROXIMATED",
     "Derived from the approximated tests_added/tests_deleted, inherits that gap. Returns 0.0 "
     "(not null) when no files changed, matching the API's required, non-nullable schema."),
    ("num_commits_in_build", "Change size", "EXACT",
     "This extractor scores one specific commit SHA — matches how GitHub Actions actually "
     "triggers builds (per push-head or per-PR-head commit), unlike old Travis CI, which "
     "sometimes batched several commits into one build."),
    ("commits_on_touched_files", "Change size", "APPROXIMATED",
     f"Sum of per-file commit-history counts (GitHub Link-header pagination trick), capped to "
     f"the first {MAX_TOUCHED_FILES_FOR_HOTNESS} changed files for API cost. TravisTorrent's "
     "exact aggregation method (sum vs. max) was never published — a judgment call, not a "
     "replication. Found and fixed a real bug here during development: a parameter-name "
     "collision between this client's endpoint-path argument and GitHub's own `path` query "
     "parameter silently zeroed this feature every time until fixed (see github_client.py)."),
    ("previous_build_status", "Build/project history", "APPROXIMATED (or correctly null)",
     "From GitHub Actions workflow run conclusions (created before this commit's timestamp), "
     "any branch — matching Stage 2's judgment call to use wall-clock project order rather "
     f"than a branch-scoped notion of 'previous build'. Lookback capped at "
     f"{MAX_HISTORY_PAGES * 100} most recent runs. Ambiguous conclusions "
     "(cancelled/skipped/neutral/in-progress) excluded, same policy Stage 1 applied to "
     "TravisTorrent's canceled/started rows. Actions run history only exists from whenever "
     "Actions was enabled on the repo — NOT the repo's full history. **Null here triggers the "
     "API's `cold_start` response state** — the intended, honest behavior for a repo with no "
     "qualifying prior run, not an extraction failure."),
    ("project_prior_failure_rate", "Build/project history", "APPROXIMATED (or correctly null)",
     "Same source and caveats as previous_build_status. **Null here also triggers `cold_start`.**"),
    ("project_prior_build_count", "Build/project history", "APPROXIMATED",
     "Same source and caveats as previous_build_status. Does not itself gate cold_start "
     "(0 is a valid, expected value for a brand-new repo, not an absence)."),
    ("consecutive_failure_streak", "Build/project history", "APPROXIMATED",
     "Same source and caveats as previous_build_status."),
    ("author_prior_builds_in_project", "Author history", "APPROXIMATED (or UNAVAILABLE if author unidentifiable)",
     "Author identity = the commit's GitHub-linked account, matched against each prior run's "
     "triggering actor. Null if the commit has no GitHub-linked account (non-GitHub commit "
     "email) — the live equivalent of TravisTorrent's own 9.4% unmatched-author gap from Stage "
     "1, not a new problem. **Null here does NOT trigger `cold_start`** — routine, and the "
     "dominant project-history signal is unaffected."),
    ("author_prior_failure_rate_in_project", "Author history", "APPROXIMATED (or UNAVAILABLE if author unidentifiable)",
     "Same source and caveats as author_prior_builds_in_project. Does not trigger cold_start."),
    ("author_days_since_last_build_in_project", "Author history", "APPROXIMATED (or UNAVAILABLE if author unidentifiable)",
     "Same source and caveats as author_prior_builds_in_project. Does not trigger cold_start."),
    ("team_size", "Project/build context", "APPROXIMATED",
     "Count of all-time contributors as of NOW, not as of the commit's timestamp — GitHub's "
     "contributors endpoint has no 'until' parameter. Exact by construction for Layer 3's "
     "actual use case (scoring the commit that just triggered the Action — there is no future "
     "yet); would leak future contributors if used to backtest an older historical commit."),
    ("repo_age_days", "Project/build context", "EXACT", "(commit timestamp - repo.created_at), both from the GitHub API."),
    ("repo_num_commits", "Project/build context", "APPROXIMATED",
     "Count of commits on the default branch up to this commit's timestamp, via the "
     "Link-header page-count trick. Default-branch-only; TravisTorrent's exact scope isn't documented."),
    ("sloc", "Project/build context", "UNAVAILABLE",
     "No live source without cloning the repo and running static analysis. Fallback: Stage 2's "
     "training-data median SLOC for the repo's language (per-language table, or overall if "
     "unrecognized). A CONSTANT per language — doesn't vary across live commits/repos of the "
     "same language. SHAP share 1.2% in training — modest importance, so this is an acceptable trade."),
    ("test_lines_per_kloc", "Project/build context", "UNAVAILABLE",
     "Same as sloc: no live source, training-median-by-language constant fallback. SHAP share 2.1%."),
    ("test_cases_per_kloc", "Project/build context", "UNAVAILABLE",
     "Same as sloc: no live source, training-median-by-language constant fallback. SHAP share 1.0%."),
    ("asserts_per_kloc", "Project/build context", "UNAVAILABLE",
     "Same as sloc: no live source, training-median-by-language constant fallback. SHAP share 0.8%."),
    ("is_pr", "Project/build context", "EXACT (if caller provides it) or APPROXIMATED",
     "EXACT when the caller (Layer 3's GitHub Actions event context) passes it directly. "
     "Without that context, APPROXIMATED: derived from whether GitHub associates any pull "
     "request with the commit — can lag a PR opened after the commit, or reflect a "
     "since-merged PR rather than 'this build ran as a PR check'."),
    ("by_core_team_member", "Project/build context", "APPROXIMATED (or UNAVAILABLE if author unidentifiable)",
     "'Core' defined as top half of all-time contributors by commit count (current, not "
     "as-of-commit — same look-ahead caveat as team_size). TravisTorrent's actual threshold "
     "was never published, so this is a judgment call, not a replication."),
    ("language", "Project/build context", "APPROXIMATED",
     "GitHub's single repo-wide 'primary language' (linguist byte-count detection), not a "
     "per-commit tag. Passed through as-is even if not one of this model's 3 known languages "
     "(ruby/java/go) — the API treats that as the unrecognized-language reference case, not an "
     "error. NOTE: 'python' is NOT a known language despite being one of TravisTorrent's "
     "original 4 — see the bug this project found and fixed in api/schema.py's KNOWN_LANGUAGES."),
    ("is_main_branch", "Project/build context", "EXACT (if caller provides it) or APPROXIMATED",
     "EXACT when the caller passes the branch directly (Layer 3 knows this from "
     "github.ref_name). Without it, APPROXIMATED via GitHub's 'branches where this commit is "
     "the tip' — accurate for a branch's current HEAD, empty for a commit since superseded "
     "even if it was on master once."),
]

assert len(PARITY_TABLE) == 31, f"expected 31 features, got {len(PARITY_TABLE)}"

CATEGORY_ORDER = ["Change size", "Build/project history", "Author history", "Project/build context"]


def cold_start_note(feature: str) -> str:
    if feature in PROJECT_HISTORY_FEATURES:
        return "TRIGGERS cold_start"
    if feature in AUTHOR_HISTORY_FEATURES:
        return "does not trigger cold_start"
    return "—"


def get_worked_example():
    """Best-effort fresh live extraction; frozen fixture on failure. Purely
    illustrative — see module docstring for why this is not what makes the
    per-feature table above trustworthy."""
    client = GitHubClient()
    try:
        result = build_feature_vector(EXAMPLE_OWNER, EXAMPLE_REPO, EXAMPLE_SHA, branch=EXAMPLE_BRANCH, client=client)
        return result.features, "fresh live extraction (this report-generation run)"
    except (GitHubRateLimitError, Exception) as exc:
        fixture = json.loads(FIXTURE_PATH.read_text())
        return fixture["features"], f"frozen fixture captured {fixture['_meta']['captured']} (live call failed this run: {exc!r})"


def main():
    features, example_source = get_worked_example()

    counts = {lvl: 0 for lvl in ("EXACT", "APPROXIMATED", "UNAVAILABLE")}
    for _, _, level, _ in PARITY_TABLE:
        for lvl in counts:
            if level.startswith(lvl):
                counts[lvl] += 1
                break

    lines = ["# Stage 3, Layer 2: feature parity report", "",
             f"Example repo/commit used throughout: `{EXAMPLE_OWNER}/{EXAMPLE_REPO}` @ "
             f"`{EXAMPLE_SHA[:12]}`. Worked-example values below: {example_source}.", "",
             "## The two caveats that apply above any single feature", "",
             "**1. Train/serve domain shift.** The model was trained on Travis CI build "
             "outcomes (TravisTorrent, 2011-2016). This extractor pulls GitHub Actions "
             "outcomes for live serving. Different CI systems, different eras, different "
             "repos — a 'failed build' carries the same label but not necessarily the same "
             "underlying causes. Bigger than any individual feature approximation below.", "",
             "**2. History window mismatch.** TravisTorrent's per-project history was "
             "effectively that project's entire build history. Live project/author history "
             "here is bounded by (a) how long GitHub Actions has been enabled on the repo — a "
             "10-year-old project that enabled Actions last month looks history-cold-start here "
             f"despite being old and stable — and (b) this extractor's own lookback cap "
             f"({MAX_HISTORY_PAGES * 100} most recent runs). Both structural, not bugs.", "",
             f"## Summary: {counts['EXACT']} EXACT / {counts['APPROXIMATED']} APPROXIMATED / "
             f"{counts['UNAVAILABLE']} UNAVAILABLE (of 31)", "",
             "\"(or correctly null)\" / \"(or UNAVAILABLE if author unidentifiable)\" qualifiers "
             "on the history features mean the classification depends on whether this specific "
             "repo/commit/author actually has resolvable history — counted here by their normal "
             "APPROXIMATED case.", ""]

    for category in CATEGORY_ORDER:
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Feature | Level | Cold-start | Notes |")
        lines.append("|---|---|---|---|")
        for feature, cat, level, note in PARITY_TABLE:
            if cat != category:
                continue
            lines.append(f"| `{feature}` | {level} | {cold_start_note(feature)} | {note} |")
        lines.append("")

    lines += ["## Worked example: extracted feature values", "",
              f"```json\n{json.dumps(features, indent=2)}\n```", "",
              "Fed through `POST /predict` during development, this vector returned "
              "`status: \"ok\"` (real project/author history was found — not a cold-start "
              "case), `risk_tier: \"Low\"` (0.192 probability), with `consecutive_failure_streak`, "
              "`previous_build_status`, and `project_prior_failure_rate` as the top "
              "risk-decreasing contributors and `team_size`/`is_pr` as the top risk-increasing "
              "ones — consistent with Stage 2's SHAP findings that the history cluster "
              "dominates and that both of those context features carry real (if smaller) "
              "signal in the expected directions.",
              ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
