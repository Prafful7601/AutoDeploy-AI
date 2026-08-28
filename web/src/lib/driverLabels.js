// Plain-language explanations for each of the model's 31 features. Ported
// from .github/scripts/post_prediction.py's DRIVER_TEMPLATES — same
// mapping, same coverage requirement (every real model feature column
// must resolve to a sentence, including the language_{go,java,ruby}
// one-hot columns SHAP actually attributes importance to, not the
// pre-encoding "language" field). Kept in sync by hand; if the two drift,
// the Python version is authoritative since it's exercised by the Action.

const pct = (v) => (v === null || v === undefined ? "unknown" : `${Math.round(v * 100)}%`);

const TEMPLATES = {
  consecutive_failure_streak: (v) =>
    v && v > 0
      ? `recent builds here have been failing in a streak (${Math.round(v)} in a row)`
      : "no active failure streak in recent builds",
  previous_build_status: (v) =>
    v === 1
      ? "the immediately preceding build in this project failed"
      : "the immediately preceding build in this project passed",
  project_prior_failure_rate: (v, inc) =>
    `this project's historical failure rate is ${inc ? "elevated" : "on the low side"} (${pct(v)})`,
  project_prior_build_count: (v) =>
    `this project has ${Math.round(v)} prior recorded build(s) in its Actions history`,
  author_prior_failure_rate_in_project: (v, inc) =>
    `this author's past builds in this project have a ${inc ? "higher" : "lower"} failure rate (${pct(v)})`,
  author_prior_builds_in_project: (v) => `this author has ${Math.round(v)} prior build(s) in this project`,
  author_days_since_last_build_in_project: (v) => `this author last built here ${v.toFixed(1)} day(s) ago`,
  team_size: (v, inc) =>
    `this project's contributor count (${Math.round(v)}) is associated with ${inc ? "more" : "less"} risk here, historically`,
  repo_age_days: (v, inc) =>
    `this project's age (${Math.round(v)} days) is associated with ${inc ? "more" : "less"} risk here`,
  repo_num_commits: (v, inc) =>
    `this project's overall commit volume is associated with ${inc ? "more" : "less"} risk here`,
  is_pr: (v, inc) =>
    `this is a ${v === 1 ? "pull-request" : "direct push"} build, which this model associates with ${inc ? "somewhat higher" : "somewhat lower"} risk here`,
  is_main_branch: (v, inc) =>
    `this build is on ${v === 1 ? "the main" : "a non-main"} branch, associated with ${inc ? "higher" : "lower"} risk here`,
  by_core_team_member: (v, inc) =>
    v === 1
      ? `the author looks like a core contributor to this project, associated with ${inc ? "higher" : "lower"} risk here`
      : `the author doesn't look like one of this project's top contributors, associated with ${inc ? "higher" : "lower"} risk here`,
  language: (v, inc) => `this project's primary language (${v}) is associated with ${inc ? "more" : "less"} risk in the training data`,
  src_churn: (v, inc) =>
    `the size of this change (${Math.round(v)} lines changed in source files) is associated with ${inc ? "more" : "less"} risk`,
  total_files_changed: (v, inc) =>
    `the number of files touched (${Math.round(v)}) is associated with ${inc ? "more" : "less"} risk`,
  files_added: (v) => `${Math.round(v)} file(s) added in this change`,
  files_deleted: (v) => `${Math.round(v)} file(s) deleted in this change`,
  files_modified: (v) => `${Math.round(v)} file(s) modified in this change`,
  src_files_changed: (v) => `${Math.round(v)} source file(s) touched`,
  doc_files_changed: (v) => `${Math.round(v)} documentation file(s) touched`,
  other_files_changed: (v) => `${Math.round(v)} other file(s) touched`,
  tests_added: (v) => (v && v > 0 ? `${Math.round(v)} new test file(s) added` : "no new test files added"),
  tests_deleted: (v) => (v && v > 0 ? `${Math.round(v)} test file(s) removed` : "no test files removed"),
  test_file_ratio: (v) => `${pct(v)} of the changed files were test files`,
  num_commits_in_build: (v) => `${Math.round(v)} commit(s) in this build`,
  commits_on_touched_files: (v) =>
    `the files touched here have ${Math.round(v)} commits of prior history — a "how often does this code change" signal`,
  sloc: () => "the project's overall code size (approximate — no live source, see parity report)",
  test_lines_per_kloc: () => "the project's test-code density (approximate — no live source, see parity report)",
  test_cases_per_kloc: () => "the project's test-case density (approximate — no live source, see parity report)",
  asserts_per_kloc: () => "the project's assertion density (approximate — no live source, see parity report)",
};

export function explainDriver(contrib) {
  const { feature, feature_value: value, direction } = contrib;
  const increases = direction === "increases risk";

  if (feature.startsWith("language_")) {
    const lang = feature.slice("language_".length);
    const isThatLanguage = value === 1;
    const langLabel = lang.charAt(0).toUpperCase() + lang.slice(1);
    return `this project ${isThatLanguage ? "is" : "is not"} written in ${langLabel}, which this model associates with ${
      increases ? "higher" : "lower"
    } risk here`;
  }

  const template = TEMPLATES[feature];
  if (template) {
    try {
      return template(value, increases);
    } catch {
      // fall through to generic fallback below
    }
  }
  return `\`${feature}\` is ${increases ? "higher than typical" : "lower than typical"} for this build`;
}
