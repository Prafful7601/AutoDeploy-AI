# Stage 1 data report

**Source:** TravisTorrent public dataset (Beller, Gousios & Zaidman, *TravisTorrent: Synthesizing Travis CI and GitHub for Full-Stack Research on Continuous Integration*, MSR 2017), downloaded from its permanent Figshare archive (https://doi.org/10.6084/m9.figshare.19314170), snapshot `final-2017-01-25.csv.gz`. Commit-author identity joined in from the companion commit-metadata dataset (Zenodo record 829968).

TravisTorrent was obtained on the first attempt — no fallback to the GitHub Actions API was needed. (Its original project site, travistorrent.testroots.org, has since been taken over by an unrelated domain; the Figshare archive linked from the project's own GitHub Pages mirror is what was used instead.)

## Record counts

- Raw job-level rows: 3,881,992
- Unique builds (after collapsing job rows): 925,897 across 948 projects
- **Filtered to projects with commit-author coverage** (see note below): 262,112 builds retained
- Builds dropped for ambiguous status (`canceled`/`started`): 973
- **Final labeled build records: 261,139**
- Distinct projects in final dataset: 243
- Build date range: 2011-04-16 to 2016-08-31
- Commit-author identity matched: 237,368 / 262,112 builds (90.6%) within the filtered project set — the remainder are real gaps (e.g. force-pushed/rebased commits no longer in history) and are imputed explicitly in Stage 2, not silently filled.

**Why filtered to a project subset:** TravisTorrent's main table has no author column. The companion commit-metadata dataset (Zenodo 829968) supplies it, but only covers 1,283 projects, of which 243 overlap with the 925,897 builds / 948 projects otherwise available. Joining without filtering gives only a 25.6% author-match rate — almost all misses are projects entirely absent from the commit log, not join failures. Restricting to the 243 overlapping projects raises the match rate to 90.5% on a still-large, still-multi-project dataset, which is what's used from here on.

## Class balance (target: `failed`)

- Passed (0): 186,320 (71.3%)
- Failed or errored (1): 74,819 (28.7%)

This is an imbalanced binary classification problem — handled in Stage 2 with class weights rather than resampling.