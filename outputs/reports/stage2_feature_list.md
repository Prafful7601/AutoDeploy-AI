# Stage 2 feature list & split report

Reviewed and approved before training. See scripts/03_build_features.py for the exact computation of every feature below.

## Feature list (provenance)

| Feature | Description | Computed from | Time window (leakage boundary) |
|---|---|---|---|
| `src_churn` | Lines changed in source files (this build's commit range) | git_diff_src_churn, TravisTorrent's own diff-at-commit-time field | none (point-in-time) |
| `files_added` | Files added | gh_diff_files_added | none (point-in-time) |
| `files_deleted` | Files deleted | gh_diff_files_deleted | none (point-in-time) |
| `files_modified` | Files modified | gh_diff_files_modified | none (point-in-time) |
| `total_files_changed` | files_added + files_deleted + files_modified | derived (sum of the three above) | none (point-in-time) |
| `src_files_changed` | Source files touched | gh_diff_src_files | none (point-in-time) |
| `doc_files_changed` | Doc files touched | gh_diff_doc_files | none (point-in-time) |
| `other_files_changed` | Other files touched | gh_diff_other_files | none (point-in-time) |
| `tests_added` | Test cases added | gh_diff_tests_added | none (point-in-time) |
| `tests_deleted` | Test cases deleted | gh_diff_tests_deleted | none (point-in-time) |
| `test_file_ratio` | (tests_added + tests_deleted) / total_files_changed; NaN if 0 files changed | derived | none (point-in-time) |
| `num_commits_in_build` | Number of commits bundled into this one CI build | git_num_all_built_commits | none (point-in-time) |
| `commits_on_touched_files` | Historical commit count touching the files this build changes | gh_num_commits_on_files_touched (TravisTorrent-computed) | as of triggering commit (trust assumption, see report) |
| `previous_build_status` | Outcome (0/1) of the immediately preceding build in this project | derived: groupby(project)['failed'].shift(1) | strictly the 1 prior build, same project, any branch |
| `project_prior_failure_rate` | Failure rate over ALL earlier builds in this project | derived: expanding mean via cumsum/cumcount, shifted to exclude current row | expanding, all strictly-earlier builds, same project |
| `project_prior_build_count` | How many earlier builds this project has had | derived: groupby(project).cumcount() | expanding, same project |
| `consecutive_failure_streak` | Count of consecutive failed builds immediately preceding this one | derived: sequential streak counter over groupby(project)['failed'] | strictly earlier builds, same project, resets on a pass |
| `author_prior_builds_in_project` | Count of this author's earlier builds in this project | derived: groupby([project, author_email]).cumcount() | expanding, same project, same author |
| `author_prior_failure_rate_in_project` | This author's failure rate on earlier builds in this project | derived: expanding mean via cumsum/cumcount on the (project, author) group | expanding, same project, same author |
| `author_days_since_last_build_in_project` | Days since this author's previous build in this project | derived: groupby([project, author_email])['build_ts'].shift(1), differenced | strictly the 1 prior build, same project, same author |
| `team_size` | Number of contributors on the project | gh_team_size | as of build (TravisTorrent-computed) |
| `repo_age_days` | Age of the repository | gh_repo_age | as of build (TravisTorrent-computed) |
| `repo_num_commits` | Total commits in the repo | gh_repo_num_commits | as of build (TravisTorrent-computed) |
| `sloc` | Source lines of code in the project | gh_sloc | as of build (TravisTorrent-computed) |
| `test_lines_per_kloc` | Test code density | gh_test_lines_per_kloc | as of build (TravisTorrent-computed) |
| `test_cases_per_kloc` | Test case density | gh_test_cases_per_kloc | as of build (TravisTorrent-computed) |
| `asserts_per_kloc` | Assertion density | gh_asserts_cases_per_kloc | as of build (TravisTorrent-computed) |
| `is_pr` | Is this build triggered by a pull request (vs a direct push) | gh_is_pr | none (point-in-time) |
| `by_core_team_member` | Is the author a core team member | gh_by_core_team_member | as of build (TravisTorrent-computed) |
| `language` | Project's primary language (categorical: ruby/python/java/go) | gh_lang | static |
| `is_main_branch` | Is this build on master/main/trunk (heuristic) | derived from git_branch | none (point-in-time) |

**Total features: 31**, plus `previous_build_status` doubles as the rule-based baseline described below.

## Splits

### Temporal split (primary)

Global cutoff at the 80% quantile of build start time, rounded to a calendar month: **2015-11-01**. All builds before this date -> train; all builds on/after -> test. A single global cutoff means no project's builds straddle train and test by chance — every build in a given calendar week falls on the same side of the split for every project.

| Split | Builds | Failed | Failure rate |
|---|---|---|---|
| train | 207,165 | 60,767 | 29.3% |
| test | 53,974 | 14,052 | 26.0% |

### Held-out-projects split

49 of 243 projects (20%) selected at random (seed=42) and held out entirely — none of their builds appear in training, at any point in time. Not stratified by project size or failure rate.

| Split | Builds | Failed | Failure rate |
|---|---|---|---|
| train | 201,107 | 59,682 | 29.7% |
| test | 60,032 | 15,137 | 25.2% |

## Judgment calls in this stage

1. **Author/project history scoped within-project only** (not across a person's other projects), per the leakage rule as written, and because cross-project author history would leak project identity into the held-out-projects split.
2. **`previous_build_status` uses wall-clock build order**, not TravisTorrent's own `tr_prev_build` pointer — that field's exact scoping isn't documented clearly enough to trust without independent verification.
3. **`gh_num_commits_in_push`, `gh_commits_in_push` dropped** (100% null in this dataset) and **`gh_description_complexity` dropped** (84.7% null) rather than engineered into mostly-missing features. **`git_diff_test_churn` (and the test_churn_ratio derived from it) dropped too — it is exactly 0 for all 261,139 rows**, a data-quality gap in TravisTorrent itself, not a bug here; `test_file_ratio` (file-count based) still captures test-vs-source change size.
4. **Missing values left as NaN**, not imputed — `HistGradientBoostingClassifier` (the chosen model) handles missingness natively via its histogram splits, so an author's or project's first-ever build correctly looks distinct from '0 prior failures' rather than being conflated with it.
5. **`is_main_branch` is a name heuristic** (master/main/trunk) since no explicit default-branch field exists in this dataset.
6. **Project-context fields (team size, SLOC, repo age, etc.) are taken as-is from TravisTorrent's own per-build covariates**, trusting the original paper's methodology that these are computed as of the triggering commit rather than re-deriving them independently.
7. **Temporal cutoff (80th percentile) and project-holdout fraction (20%) are both configurable constants** at the top of the script, not tuned to any result.