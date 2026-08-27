# Stage 3, Layer 2: feature parity report

Example repo/commit used throughout: `spf13/cobra` @ `adbc8813901b`. Worked-example values below: frozen fixture captured 2026-08-28, live, unauthenticated (60 req/hour), this development session (live call failed this run: GitHubRateLimitError('GitHub API rate limit exhausted. Resets in 2300s (at epoch 1787858821). Set GITHUB_TOKEN for a 5,000/hour limit instead of the unauthenticated 60/hour.')).

## The two caveats that apply above any single feature

**1. Train/serve domain shift.** The model was trained on Travis CI build outcomes (TravisTorrent, 2011-2016). This extractor pulls GitHub Actions outcomes for live serving. Different CI systems, different eras, different repos — a 'failed build' carries the same label but not necessarily the same underlying causes. Bigger than any individual feature approximation below.

**2. History window mismatch.** TravisTorrent's per-project history was effectively that project's entire build history. Live project/author history here is bounded by (a) how long GitHub Actions has been enabled on the repo — a 10-year-old project that enabled Actions last month looks history-cold-start here despite being old and stable — and (b) this extractor's own lookback cap (500 most recent runs). Both structural, not bugs.

## Summary: 8 EXACT / 18 APPROXIMATED / 5 UNAVAILABLE (of 31)

"(or correctly null)" / "(or UNAVAILABLE if author unidentifiable)" qualifiers on the history features mean the classification depends on whether this specific repo/commit/author actually has resolvable history — counted here by their normal APPROXIMATED case.

## Change size

| Feature | Level | Cold-start | Notes |
|---|---|---|---|
| `src_churn` | APPROXIMATED | — | Sum of additions+deletions over files classified 'src' by a path heuristic (extractor/file_classification.py), not TravisTorrent's original per-language diff tool. Config/CI/build files not caught by doc/test heuristics are counted as src here (TravisTorrent had a separate 'other' bucket). |
| `files_added` | EXACT | — | Direct from the commit's file-status list. |
| `files_deleted` | EXACT | — | Direct from the commit's file-status list. |
| `files_modified` | EXACT | — | Direct from the commit's file-status list. |
| `total_files_changed` | EXACT | — | len(commit.files). |
| `src_files_changed` | APPROXIMATED | — | Path-heuristic classification; see src_churn note. |
| `doc_files_changed` | APPROXIMATED | — | Path-heuristic classification (extension/dirname match). |
| `other_files_changed` | UNAVAILABLE | — | The classifier has no 'other' bucket (build config, generated files, binary assets) — everything not test/doc is counted as src instead. Always 0 live. SHAP rank 20/33 (0.7% share) in training — low-impact. |
| `tests_added` | APPROXIMATED | — | TravisTorrent counted test CASES added (test-level diff parsing). This counts whole NEW test FILES only — a commit adding test cases inside an existing test file contributes 0. Systematically undercounts the common case. SHAP rank 30/33 (0.0% share) in training. |
| `tests_deleted` | APPROXIMATED | — | Same file-level-not-case-level gap as tests_added. SHAP rank 22/33 (0.4% share). |
| `test_file_ratio` | APPROXIMATED | — | Derived from the approximated tests_added/tests_deleted, inherits that gap. Returns 0.0 (not null) when no files changed, matching the API's required, non-nullable schema. |
| `num_commits_in_build` | EXACT | — | This extractor scores one specific commit SHA — matches how GitHub Actions actually triggers builds (per push-head or per-PR-head commit), unlike old Travis CI, which sometimes batched several commits into one build. |
| `commits_on_touched_files` | APPROXIMATED | — | Sum of per-file commit-history counts (GitHub Link-header pagination trick), capped to the first 5 changed files for API cost. TravisTorrent's exact aggregation method (sum vs. max) was never published — a judgment call, not a replication. Found and fixed a real bug here during development: a parameter-name collision between this client's endpoint-path argument and GitHub's own `path` query parameter silently zeroed this feature every time until fixed (see github_client.py). |

## Build/project history

| Feature | Level | Cold-start | Notes |
|---|---|---|---|
| `previous_build_status` | APPROXIMATED (or correctly null) | TRIGGERS cold_start | From GitHub Actions workflow run conclusions (created before this commit's timestamp), any branch — matching Stage 2's judgment call to use wall-clock project order rather than a branch-scoped notion of 'previous build'. Lookback capped at 500 most recent runs. Ambiguous conclusions (cancelled/skipped/neutral/in-progress) excluded, same policy Stage 1 applied to TravisTorrent's canceled/started rows. Actions run history only exists from whenever Actions was enabled on the repo — NOT the repo's full history. **Null here triggers the API's `cold_start` response state** — the intended, honest behavior for a repo with no qualifying prior run, not an extraction failure. |
| `project_prior_failure_rate` | APPROXIMATED (or correctly null) | TRIGGERS cold_start | Same source and caveats as previous_build_status. **Null here also triggers `cold_start`.** |
| `project_prior_build_count` | APPROXIMATED | — | Same source and caveats as previous_build_status. Does not itself gate cold_start (0 is a valid, expected value for a brand-new repo, not an absence). |
| `consecutive_failure_streak` | APPROXIMATED | — | Same source and caveats as previous_build_status. |

## Author history

| Feature | Level | Cold-start | Notes |
|---|---|---|---|
| `author_prior_builds_in_project` | APPROXIMATED (or UNAVAILABLE if author unidentifiable) | does not trigger cold_start | Author identity = the commit's GitHub-linked account, matched against each prior run's triggering actor. Null if the commit has no GitHub-linked account (non-GitHub commit email) — the live equivalent of TravisTorrent's own 9.4% unmatched-author gap from Stage 1, not a new problem. **Null here does NOT trigger `cold_start`** — routine, and the dominant project-history signal is unaffected. |
| `author_prior_failure_rate_in_project` | APPROXIMATED (or UNAVAILABLE if author unidentifiable) | does not trigger cold_start | Same source and caveats as author_prior_builds_in_project. Does not trigger cold_start. |
| `author_days_since_last_build_in_project` | APPROXIMATED (or UNAVAILABLE if author unidentifiable) | does not trigger cold_start | Same source and caveats as author_prior_builds_in_project. Does not trigger cold_start. |

## Project/build context

| Feature | Level | Cold-start | Notes |
|---|---|---|---|
| `team_size` | APPROXIMATED | — | Count of all-time contributors as of NOW, not as of the commit's timestamp — GitHub's contributors endpoint has no 'until' parameter. Exact by construction for Layer 3's actual use case (scoring the commit that just triggered the Action — there is no future yet); would leak future contributors if used to backtest an older historical commit. |
| `repo_age_days` | EXACT | — | (commit timestamp - repo.created_at), both from the GitHub API. |
| `repo_num_commits` | APPROXIMATED | — | Count of commits on the default branch up to this commit's timestamp, via the Link-header page-count trick. Default-branch-only; TravisTorrent's exact scope isn't documented. |
| `sloc` | UNAVAILABLE | — | No live source without cloning the repo and running static analysis. Fallback: Stage 2's training-data median SLOC for the repo's language (per-language table, or overall if unrecognized). A CONSTANT per language — doesn't vary across live commits/repos of the same language. SHAP share 1.2% in training — modest importance, so this is an acceptable trade. |
| `test_lines_per_kloc` | UNAVAILABLE | — | Same as sloc: no live source, training-median-by-language constant fallback. SHAP share 2.1%. |
| `test_cases_per_kloc` | UNAVAILABLE | — | Same as sloc: no live source, training-median-by-language constant fallback. SHAP share 1.0%. |
| `asserts_per_kloc` | UNAVAILABLE | — | Same as sloc: no live source, training-median-by-language constant fallback. SHAP share 0.8%. |
| `is_pr` | EXACT (if caller provides it) or APPROXIMATED | — | EXACT when the caller (Layer 3's GitHub Actions event context) passes it directly. Without that context, APPROXIMATED: derived from whether GitHub associates any pull request with the commit — can lag a PR opened after the commit, or reflect a since-merged PR rather than 'this build ran as a PR check'. |
| `by_core_team_member` | APPROXIMATED (or UNAVAILABLE if author unidentifiable) | — | 'Core' defined as top half of all-time contributors by commit count (current, not as-of-commit — same look-ahead caveat as team_size). TravisTorrent's actual threshold was never published, so this is a judgment call, not a replication. |
| `language` | APPROXIMATED | — | GitHub's single repo-wide 'primary language' (linguist byte-count detection), not a per-commit tag. Passed through as-is even if not one of this model's 3 known languages (ruby/java/go) — the API treats that as the unrecognized-language reference case, not an error. NOTE: 'python' is NOT a known language despite being one of TravisTorrent's original 4 — see the bug this project found and fixed in api/schema.py's KNOWN_LANGUAGES. |
| `is_main_branch` | EXACT (if caller provides it) or APPROXIMATED | — | EXACT when the caller passes the branch directly (Layer 3 knows this from github.ref_name). Without it, APPROXIMATED via GitHub's 'branches where this commit is the tip' — accurate for a branch's current HEAD, empty for a commit since superseded even if it was on master once. |

## Worked example: extracted feature values

```json
{
  "src_churn": 24,
  "files_added": 0,
  "files_deleted": 0,
  "files_modified": 3,
  "total_files_changed": 3,
  "src_files_changed": 3,
  "doc_files_changed": 0,
  "other_files_changed": 0,
  "tests_added": 0,
  "tests_deleted": 0,
  "test_file_ratio": 0.0,
  "num_commits_in_build": 1,
  "commits_on_touched_files": 55,
  "repo_age_days": 4693,
  "repo_num_commits": 1106,
  "team_size": 330,
  "language": "go",
  "is_pr": 1,
  "is_main_branch": 1,
  "by_core_team_member": 1,
  "sloc": 4549,
  "test_lines_per_kloc": 274.59,
  "test_cases_per_kloc": 0.0,
  "asserts_per_kloc": 28.81,
  "previous_build_status": 0,
  "project_prior_failure_rate": 0.0410958904109589,
  "project_prior_build_count": 292,
  "consecutive_failure_streak": 0,
  "author_prior_builds_in_project": 13,
  "author_prior_failure_rate_in_project": 0.15384615384615385,
  "author_days_since_last_build_in_project": 0.7765162037037037
}
```

Fed through `POST /predict` during development, this vector returned `status: "ok"` (real project/author history was found — not a cold-start case), `risk_tier: "Low"` (0.192 probability), with `consecutive_failure_streak`, `previous_build_status`, and `project_prior_failure_rate` as the top risk-decreasing contributors and `team_size`/`is_pr` as the top risk-increasing ones — consistent with Stage 2's SHAP findings that the history cluster dominates and that both of those context features carry real (if smaller) signal in the expected directions.