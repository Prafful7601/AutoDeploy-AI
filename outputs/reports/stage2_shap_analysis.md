# Stage 2 SHAP interpretation

Model: HistGradientBoostingClassifier trained on the **temporal split** (class_weight='balanced'). SHAP values computed with `shap.TreeExplainer` over the full **temporal test set** (53,974 builds) — not train, and not the held-out-projects model. Values are in the model's raw (log-odds) output space, as returned by TreeExplainer by default.

## Global importance (mean |SHAP|), all features ranked

| Rank | Feature | Mean \|SHAP\| | Share of total |
|---|---|---|---|
| 1 | `consecutive_failure_streak` | 0.9279 | 30.8% |
| 2 | `previous_build_status` | 0.7106 | 23.6% |
| 3 | `project_prior_failure_rate` | 0.3416 | 11.3% |
| 4 | `author_prior_failure_rate_in_project` | 0.2220 | 7.4% |
| 5 | `is_main_branch` | 0.1136 | 3.8% |
| 6 | `author_days_since_last_build_in_project` | 0.0696 | 2.3% |
| 7 | `repo_age_days` | 0.0656 | 2.2% |
| 8 | `test_lines_per_kloc` | 0.0618 | 2.1% |
| 9 | `project_prior_build_count` | 0.0509 | 1.7% |
| 10 | `is_pr` | 0.0489 | 1.6% |
| 11 | `total_files_changed` | 0.0482 | 1.6% |
| 12 | `team_size` | 0.0414 | 1.4% |
| 13 | `repo_num_commits` | 0.0371 | 1.2% |
| 14 | `num_commits_in_build` | 0.0365 | 1.2% |
| 15 | `sloc` | 0.0347 | 1.2% |
| 16 | `files_added` | 0.0346 | 1.1% |
| 17 | `test_cases_per_kloc` | 0.0307 | 1.0% |
| 18 | `author_prior_builds_in_project` | 0.0290 | 1.0% |
| 19 | `asserts_per_kloc` | 0.0242 | 0.8% |
| 20 | `other_files_changed` | 0.0204 | 0.7% |
| 21 | `language_ruby` | 0.0164 | 0.5% |
| 22 | `tests_deleted` | 0.0120 | 0.4% |
| 23 | `commits_on_touched_files` | 0.0071 | 0.2% |
| 24 | `src_files_changed` | 0.0060 | 0.2% |
| 25 | `language_java` | 0.0049 | 0.2% |
| 26 | `files_modified` | 0.0044 | 0.1% |
| 27 | `src_churn` | 0.0039 | 0.1% |
| 28 | `language_go` | 0.0035 | 0.1% |
| 29 | `test_file_ratio` | 0.0018 | 0.1% |
| 30 | `tests_added` | 0.0014 | 0.0% |
| 31 | `files_deleted` | 0.0002 | 0.0% |
| 32 | `doc_files_changed` | 0.0001 | 0.0% |
| 33 | `by_core_team_member` | 0.0000 | 0.0% |

## Recent-history cluster: combined share

`previous_build_status` + `project_prior_failure_rate` + `consecutive_failure_streak`, combined: **1.9800 mean |SHAP|, 65.8% of total importance** across all 33 features. It dominates, as expected — three features out of 33 account for 65.8% of the model's total attribution.

## Everything else, ranked (change-size, context, and author-history features)

This is the part that matters for held-out-projects generalization, since these are the features not entirely reset to 'unknown' for a project the model has never seen (author-history features *are* reset per-project like the history cluster is, and land accordingly low below — called out explicitly rather than silently lumped in with the change-size/context story).

| Rank | Feature | Mean \|SHAP\| | Share of total |
|---|---|---|---|
| 1 | `author_prior_failure_rate_in_project` | 0.2220 | 7.4% |
| 2 | `is_main_branch` | 0.1136 | 3.8% |
| 3 | `author_days_since_last_build_in_project` | 0.0696 | 2.3% |
| 4 | `repo_age_days` | 0.0656 | 2.2% |
| 5 | `test_lines_per_kloc` | 0.0618 | 2.1% |
| 6 | `project_prior_build_count` | 0.0509 | 1.7% |
| 7 | `is_pr` | 0.0489 | 1.6% |
| 8 | `total_files_changed` | 0.0482 | 1.6% |
| 9 | `team_size` | 0.0414 | 1.4% |
| 10 | `repo_num_commits` | 0.0371 | 1.2% |
| 11 | `num_commits_in_build` | 0.0365 | 1.2% |
| 12 | `sloc` | 0.0347 | 1.2% |
| 13 | `files_added` | 0.0346 | 1.1% |
| 14 | `test_cases_per_kloc` | 0.0307 | 1.0% |
| 15 | `author_prior_builds_in_project` | 0.0290 | 1.0% |
| 16 | `asserts_per_kloc` | 0.0242 | 0.8% |
| 17 | `other_files_changed` | 0.0204 | 0.7% |
| 18 | `language_ruby` | 0.0164 | 0.5% |
| 19 | `tests_deleted` | 0.0120 | 0.4% |
| 20 | `commits_on_touched_files` | 0.0071 | 0.2% |
| 21 | `src_files_changed` | 0.0060 | 0.2% |
| 22 | `language_java` | 0.0049 | 0.2% |
| 23 | `files_modified` | 0.0044 | 0.1% |
| 24 | `src_churn` | 0.0039 | 0.1% |
| 25 | `language_go` | 0.0035 | 0.1% |
| 26 | `test_file_ratio` | 0.0018 | 0.1% |
| 27 | `tests_added` | 0.0014 | 0.0% |
| 28 | `files_deleted` | 0.0002 | 0.0% |
| 29 | `doc_files_changed` | 0.0001 | 0.0% |
| 30 | `by_core_team_member` | 0.0000 | 0.0% |

**Top change-size/context features (excluding all history-flavored features — the 3-feature cluster, author-history, and `project_prior_build_count`): `is_main_branch`, `repo_age_days`, `test_lines_per_kloc`, `is_pr`, `total_files_changed`.**

`author_prior_failure_rate_in_project` (rank 4 overall, 7.4% share) is excluded from that list on purpose — it's author-history, which cold-starts for a never-seen author the same way the 3-feature cluster cold-starts for a never-seen project, so it doesn't transfer to held-out projects either.

## Counterintuitive directions

- `team_size` (corr=+0.084): expected — larger teams often assumed to be more stable/tested -> LOWER failure risk. Observed — higher feature value -> HIGHER predicted failure risk. **FLAGGED — contradicts intuition**
- `repo_age_days` (corr=-0.447): expected — older, more mature projects assumed -> LOWER failure risk. Observed — higher feature value -> LOWER predicted failure risk. matches intuition
- `sloc` (corr=+0.387): expected — larger codebases sometimes assumed riskier -> HIGHER failure risk. Observed — higher feature value -> HIGHER predicted failure risk. matches intuition

Not a contradiction of any stated intuition, but worth flagging as a strong, clean effect: `is_pr` correlates at **+0.91** with predicted failure risk — PR-triggered builds are far more likely to be flagged than direct pushes. Plausible read: PRs disproportionately carry exploratory/WIP/first-draft commits versus a maintainer's own vetted push, rather than PR builds being inherently riskier work.

## Dependence plots

- `outputs/figures/shap_dependence_team_size.png`