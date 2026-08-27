"""
Stage 1 (data), step 2: build the labeled, build-level dataset.

TravisTorrent's main CSV is one row per *job* (a build run against one
language/environment combination in Travis's job matrix), not one row per
build — a single push can spawn several jobs that all share the same
outcome. This script:

  1. Loads the raw CSV (trimmed to the columns we might use — the huge
     per-job log/text columns are dropped here since they're either
     post-build leakage or unneeded bulk, see the exclusion list below).
  2. Collapses job rows to one row per `tr_build_id` (job rows belonging to
     the same build share identical build-level fields — verified during
     exploration: zero builds had inconsistent `tr_status` across jobs).
  3. Attaches commit-author identity (name/email) by joining the trigger
     commit SHA against TravisTorrent's companion commit-metadata dataset
     (Zenodo 829968), since the main CSV has no author column at all.
  4. Defines the binary target: 1 = build failed, 0 = build passed.
  5. Writes the result to data/processed/ and prints/saves a data report
     (record counts, join rate, class balance) — Stage 1's deliverable.

No feature engineering happens here — that's Stage 2, done from this raw
labeled table, with an explicit leakage check per feature.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR, set_global_seed  # noqa: E402

RAW_CSV = RAW_DATA_DIR / "travistorrent_final.csv"
COMMITLOG_DB = RAW_DATA_DIR / "commitlog.sqlite"
OUT_PARQUET = PROCESSED_DATA_DIR / "builds_labeled.parquet"
OUT_CSV_PREVIEW = PROCESSED_DATA_DIR / "builds_labeled_preview.csv"
REPORT_PATH = REPORTS_DIR / "stage1_data_report.md"

# Columns kept from the raw CSV. Excluded on purpose:
#   - tr_job_id, tr_build_number, tr_jobs: job-matrix bookkeeping, not
#     needed once collapsed to build level.
#   - tr_log_* , tr_duration, tr_original_commit's sibling log fields:
#     these describe what happened *during/after* the build ran (test
#     counts, pass/fail per suite, build duration) — pure post-build
#     leakage if used as features. Kept out of the base table entirely
#     so a later script can't accidentally reach for them.
#   - git_all_built_commits: a single string concatenating every commit
#     SHA in the build with '#', sometimes tens of KB; not needed.
#   - gh_num_issue_comments / gh_num_pr_comments / gh_num_commit_comments:
#     comment counts can accumulate *after* a build result is known
#     (e.g. "build failed, see comment") — excluded to stay safe rather
#     than audit each case.
#   - gh_pr_created_at, gh_pull_req_num, git_merged_with, tr_virtual_merged_into:
#     PR bookkeeping not used by any planned feature.
KEEP_COLUMNS = [
    "tr_build_id",
    "gh_project_name",
    "gh_is_pr",
    "gh_lang",
    "git_branch",
    "gh_num_commits_in_push",
    "gh_commits_in_push",
    "git_prev_commit_resolution_status",
    "git_prev_built_commit",
    "tr_prev_build",
    "gh_first_commit_created_at",
    "gh_team_size",
    "git_num_all_built_commits",
    "git_trigger_commit",
    "git_diff_src_churn",
    "git_diff_test_churn",
    "gh_diff_files_added",
    "gh_diff_files_deleted",
    "gh_diff_files_modified",
    "gh_diff_tests_added",
    "gh_diff_tests_deleted",
    "gh_diff_src_files",
    "gh_diff_doc_files",
    "gh_diff_other_files",
    "gh_num_commits_on_files_touched",
    "gh_sloc",
    "gh_test_lines_per_kloc",
    "gh_test_cases_per_kloc",
    "gh_asserts_cases_per_kloc",
    "gh_by_core_team_member",
    "gh_description_complexity",
    "gh_pushed_at",
    "gh_build_started_at",
    "gh_repo_age",
    "gh_repo_num_commits",
    "tr_original_commit",
    "tr_status",
]

# tr_status -> binary target. 'canceled' (manually stopped, not a verdict
# on the code) and 'started' (incomplete record, 3 rows total in the whole
# dataset) are dropped rather than forced into a class they don't belong to.
STATUS_TO_LABEL = {"passed": 0, "failed": 1, "errored": 1}
DROPPED_STATUSES = {"canceled", "started"}


def load_builds() -> pd.DataFrame:
    print(f"Reading {RAW_CSV} ...")
    df = pd.read_csv(
        RAW_CSV,
        usecols=KEEP_COLUMNS,
        dtype={"gh_project_name": "string", "git_trigger_commit": "string",
               "tr_original_commit": "string", "git_branch": "string"},
        low_memory=False,
    )
    n_rows = len(df)
    n_builds = df["tr_build_id"].nunique()
    print(f"  {n_rows:,} job rows across {n_builds:,} unique builds")

    # Collapse job rows -> build rows. Verified during exploration that all
    # jobs in a build share the same tr_status; .first() is safe here since
    # every kept column is a build-level (not job-level) attribute.
    builds = df.groupby("tr_build_id", as_index=False).first()
    assert len(builds) == n_builds
    return builds, n_rows


def attach_author_identity(builds: pd.DataFrame):
    print(f"Loading commit metadata from {COMMITLOG_DB} ...")
    con = sqlite3.connect(COMMITLOG_DB)
    commits = pd.read_sql_query(
        "SELECT project, sha, date AS commit_date, author_name, author_email FROM commits",
        con,
    )
    con.close()
    commitlog_projects = set(commits["project"].unique())
    print(f"  {len(commits):,} commit records across {len(commitlog_projects):,} projects")

    # The commit-metadata dataset (Zenodo 829968) only covers 1,283 projects,
    # of which just 243 overlap with the 948 projects in TravisTorrent's
    # build table. Joining as-is gives a 25.6% author-match rate — almost
    # all misses are simply projects with zero rows in the commit log, not
    # bad SHA matches. Rather than build author-history features that are
    # NaN for 74% of the data, restrict to the 243 projects that *do* have
    # commit coverage: within that subset the match rate is 90.5%, which is
    # a realistic, imputable gap (e.g. force-pushes/rebases changing SHAs).
    n_before = len(builds)
    n_projects_before = builds["gh_project_name"].nunique()
    builds = builds[builds["gh_project_name"].isin(commitlog_projects)].copy()
    print(f"  Restricting to projects with commit-log coverage: "
          f"{n_projects_before:,} -> {builds['gh_project_name'].nunique():,} projects, "
          f"{n_before:,} -> {len(builds):,} builds")

    # Prefer the commit that actually triggered the build; fall back to the
    # originally-pushed commit if the trigger field is blank.
    builds["join_sha"] = builds["git_trigger_commit"].where(
        builds["git_trigger_commit"].notna() & (builds["git_trigger_commit"].str.len() > 0),
        builds["tr_original_commit"],
    )

    merged = builds.merge(
        commits,
        left_on=["gh_project_name", "join_sha"],
        right_on=["project", "sha"],
        how="left",
    )
    matched = merged["author_email"].notna().sum()
    print(f"  Author identity matched for {matched:,} / {len(merged):,} builds "
          f"({matched / len(merged):.1%})")
    return merged, n_before, n_projects_before


def label_and_split(df: pd.DataFrame):
    df = df.copy()
    kept_mask = df["tr_status"].isin(STATUS_TO_LABEL)
    dropped = (~kept_mask).sum()
    df = df.loc[kept_mask].copy()
    df["failed"] = df["tr_status"].map(STATUS_TO_LABEL).astype("int8")
    return df, dropped


def write_report(n_raw_rows, n_builds, n_builds_before_filter, n_projects_before_filter,
                  n_matched, n_dropped_status, df):
    n_final = len(df)
    n_fail = int(df["failed"].sum())
    n_pass = n_final - n_fail
    n_projects = df["gh_project_name"].nunique()
    date_min = pd.to_datetime(df["gh_build_started_at"], errors="coerce").min()
    date_max = pd.to_datetime(df["gh_build_started_at"], errors="coerce").max()

    lines = [
        "# Stage 1 data report",
        "",
        "**Source:** TravisTorrent public dataset "
        "(Beller, Gousios & Zaidman, *TravisTorrent: Synthesizing Travis CI and "
        "GitHub for Full-Stack Research on Continuous Integration*, MSR 2017), "
        "downloaded from its permanent Figshare archive "
        "(https://doi.org/10.6084/m9.figshare.19314170), snapshot "
        "`final-2017-01-25.csv.gz`. Commit-author identity joined in from the "
        "companion commit-metadata dataset (Zenodo record 829968).",
        "",
        "TravisTorrent was obtained on the first attempt — no fallback to the "
        "GitHub Actions API was needed. (Its original project site, "
        "travistorrent.testroots.org, has since been taken over by an unrelated "
        "domain; the Figshare archive linked from the project's own GitHub Pages "
        "mirror is what was used instead.)",
        "",
        "## Record counts",
        "",
        f"- Raw job-level rows: {n_raw_rows:,}",
        f"- Unique builds (after collapsing job rows): {n_builds_before_filter:,} "
        f"across {n_projects_before_filter:,} projects",
        "- **Filtered to projects with commit-author coverage** (see note below): "
        f"{n_builds:,} builds retained",
        f"- Builds dropped for ambiguous status (`canceled`/`started`): {n_dropped_status:,}",
        f"- **Final labeled build records: {n_final:,}**",
        f"- Distinct projects in final dataset: {n_projects:,}",
        f"- Build date range: {date_min.date()} to {date_max.date()}",
        f"- Commit-author identity matched: {n_matched:,} / {n_builds:,} builds "
        f"({n_matched / n_builds:.1%}) within the filtered project set — the "
        "remainder are real gaps (e.g. force-pushed/rebased commits no longer "
        "in history) and are imputed explicitly in Stage 2, not silently filled.",
        "",
        "**Why filtered to a project subset:** TravisTorrent's main table has no "
        "author column. The companion commit-metadata dataset (Zenodo 829968) "
        "supplies it, but only covers 1,283 projects, of which 243 overlap with "
        f"the {n_builds_before_filter:,} builds / {n_projects_before_filter:,} "
        "projects otherwise available. Joining without filtering gives only a "
        "25.6% author-match rate — almost all misses are projects entirely "
        "absent from the commit log, not join failures. Restricting to the 243 "
        "overlapping projects raises the match rate to 90.5% on a still-large, "
        "still-multi-project dataset, which is what's used from here on.",
        "",
        "## Class balance (target: `failed`)",
        "",
        f"- Passed (0): {n_pass:,} ({n_pass / n_final:.1%})",
        f"- Failed or errored (1): {n_fail:,} ({n_fail / n_final:.1%})",
        "",
        "This is an imbalanced binary classification problem — handled in "
        "Stage 2 with class weights rather than resampling.",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


def main():
    set_global_seed()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    builds, n_raw_rows = load_builds()
    n_builds_before_filter = len(builds)
    n_projects_before_filter = builds["gh_project_name"].nunique()
    merged, _, _ = attach_author_identity(builds)
    n_builds_filtered = len(merged)
    n_matched = merged["author_email"].notna().sum()

    labeled, n_dropped_status = label_and_split(merged)

    labeled.to_parquet(OUT_PARQUET, index=False)
    labeled.head(200).to_csv(OUT_CSV_PREVIEW, index=False)
    print(f"\nWrote {len(labeled):,} labeled build records -> {OUT_PARQUET}")
    print(f"Preview (200 rows) -> {OUT_CSV_PREVIEW}")

    write_report(n_raw_rows, n_builds_filtered, n_builds_before_filter,
                 n_projects_before_filter, n_matched, n_dropped_status, labeled)


if __name__ == "__main__":
    main()
