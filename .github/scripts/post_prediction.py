"""
Stage 3, Layer 3: the GitHub Action's glue script.

Reads the triggering push/PR event from GitHub Actions' own environment,
runs the Layer 2 extractor for that commit, runs the Stage 2 model
in-process (importing api.model.PredictionService directly rather than
spinning up an HTTP server in the CI job — same code the API wraps, just
without the network hop, which is simpler and more reliable in a CI
container), and posts the result as a commit status + PR comment.

DESIGN CHOICE, flagged explicitly: this script runs the model in-process
rather than calling POST /predict over HTTP. Functionally identical
(PredictionService is the same object the API uses), and avoids managing a
server process/port/health-check inside the CI job for no benefit.

Every hard requirement from the brief is enforced structurally here, not
just described:
  - EXPERIMENTAL_BANNER is prepended to every possible output.
  - Probability only ever appears inside `secondary_line()`, in an
    already-hedged sentence — there is no code path that emits it as a
    headline.
  - cold_start is handled by its own branch that never computes or shows a
    tier.
  - Any failure to extract or validate features routes to
    could_not_score() — never a fabricated or zeroed prediction.
  - post_commit_status() hardcodes state="success" — there is no other
    call site for the GitHub status API in this file, so the job can
    never post a failing/blocking check.
  - main() is wrapped so the process always exits 0, even on a totally
    unexpected error — the workflow run itself must never show red.
"""

import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pydantic import ValidationError  # noqa: E402

from api.coldstart import STATUS_COLD_START  # noqa: E402
from api.model import PredictionService  # noqa: E402
from api.schema import BuildFeatures  # noqa: E402
from extractor.extract import build_feature_vector  # noqa: E402
from extractor.github_client import GitHubClient, GitHubRateLimitError  # noqa: E402

# The demo repo where the parity report and cold-start docs actually live —
# hardcoded rather than derived from the calling repo, because an Action
# copied into someone else's repo won't have these files unless they copied
# them too. Linking back to the canonical source is correct either way.
CANONICAL_REPO = "Prafful7601/AutoDeploy-AI"
PARITY_REPORT_URL = f"https://github.com/{CANONICAL_REPO}/blob/main/outputs/reports/stage3_feature_parity.md"
COLDSTART_DOC_URL = f"https://github.com/{CANONICAL_REPO}/blob/main/api/README.md#cold-start-handling-layer-1b"

STATUS_CONTEXT = "autodeploy-ai/experimental-signal"
SHALLOW_BUILD_COUNT_THRESHOLD = 10
SHALLOW_REPO_AGE_DAYS_THRESHOLD = 180

EXPERIMENTAL_BANNER = (
    "> ⚠️ **Experimental, cross-CI-system signal — not validated. Do not treat this as a verdict.**\n"
    "> Trained on Travis CI build outcomes; applied here to GitHub Actions builds — a different "
    "CI system, with known miscalibration on real repos. See the "
    f"[feature-parity report]({PARITY_REPORT_URL}) for exactly what is and isn't comparable. "
    "This demonstrates an end-to-end pipeline, not a production risk assessment."
)

# --------------------------------------------------------------------------
# Plain-language driver explanations. Every one of the 31 features has an
# entry (or hits the generic fallback) so the comment never surfaces a raw
# feature name when it can help it — "the explanation is the point."
# --------------------------------------------------------------------------
def _pct(v):
    return f"{v:.0%}" if v is not None else "unknown"


DRIVER_TEMPLATES = {
    "consecutive_failure_streak": lambda v, inc: (
        f"recent builds here have been failing in a streak ({int(v)} in a row)" if v and v > 0
        else "no active failure streak in recent builds"
    ),
    "previous_build_status": lambda v, inc: "the immediately preceding build in this project failed" if v == 1 else "the immediately preceding build in this project passed",
    "project_prior_failure_rate": lambda v, inc: f"this project's historical failure rate is {'elevated' if inc else 'on the low side'} ({_pct(v)})",
    "project_prior_build_count": lambda v, inc: f"this project has {int(v)} prior recorded build(s) in its Actions history",
    "author_prior_failure_rate_in_project": lambda v, inc: f"this author's past builds in this project have a {'higher' if inc else 'lower'} failure rate ({_pct(v)})",
    "author_prior_builds_in_project": lambda v, inc: f"this author has {int(v)} prior build(s) in this project",
    "author_days_since_last_build_in_project": lambda v, inc: f"this author last built here {v:.1f} day(s) ago",
    "team_size": lambda v, inc: f"this project's contributor count ({int(v)}) is associated with {'more' if inc else 'less'} risk here, historically",
    "repo_age_days": lambda v, inc: f"this project's age ({int(v)} days) is associated with {'more' if inc else 'less'} risk here",
    "repo_num_commits": lambda v, inc: f"this project's overall commit volume is associated with {'more' if inc else 'less'} risk here",
    "is_pr": lambda v, inc: f"this is a {'pull-request' if v == 1 else 'direct push'} build, which this model associates with {'somewhat higher' if inc else 'somewhat lower'} risk here",
    "is_main_branch": lambda v, inc: f"this build is on {'the main' if v == 1 else 'a non-main'} branch, associated with {'higher' if inc else 'lower'} risk here",
    "by_core_team_member": lambda v, inc: (
        f"the author looks like a core contributor to this project, associated with {'higher' if inc else 'lower'} risk here"
        if v == 1 else
        f"the author doesn't look like one of this project's top contributors, associated with {'higher' if inc else 'lower'} risk here"
    ),
    "language": lambda v, inc: f"this project's primary language ({v}) is associated with {'more' if inc else 'less'} risk in the training data",
    "src_churn": lambda v, inc: f"the size of this change ({int(v)} lines changed in source files) is associated with {'more' if inc else 'less'} risk",
    "total_files_changed": lambda v, inc: f"the number of files touched ({int(v)}) is associated with {'more' if inc else 'less'} risk",
    "files_added": lambda v, inc: f"{int(v)} file(s) added in this change",
    "files_deleted": lambda v, inc: f"{int(v)} file(s) deleted in this change",
    "files_modified": lambda v, inc: f"{int(v)} file(s) modified in this change",
    "src_files_changed": lambda v, inc: f"{int(v)} source file(s) touched",
    "doc_files_changed": lambda v, inc: f"{int(v)} documentation file(s) touched",
    "other_files_changed": lambda v, inc: f"{int(v)} other file(s) touched",
    "tests_added": lambda v, inc: f"{int(v)} new test file(s) added" if v and v > 0 else "no new test files added",
    "tests_deleted": lambda v, inc: f"{int(v)} test file(s) removed" if v and v > 0 else "no test files removed",
    "test_file_ratio": lambda v, inc: f"{_pct(v)} of the changed files were test files",
    "num_commits_in_build": lambda v, inc: f"{int(v)} commit(s) in this build",
    "commits_on_touched_files": lambda v, inc: f"the files touched here have {int(v)} commits of prior history — a 'how often does this code change' signal",
    "sloc": lambda v, inc: "the project's overall code size (approximate — no live source, see parity report)",
    "test_lines_per_kloc": lambda v, inc: "the project's test-code density (approximate — no live source, see parity report)",
    "test_cases_per_kloc": lambda v, inc: "the project's test-case density (approximate — no live source, see parity report)",
    "asserts_per_kloc": lambda v, inc: "the project's assertion density (approximate — no live source, see parity report)",
}


def explain_driver(contrib: dict) -> str:
    feature = contrib["feature"]
    value = contrib.get("feature_value")
    increases = contrib["direction"] == "increases risk"

    # SHAP attributes importance to the model's actual (one-hot) columns,
    # e.g. `language_go`, not the pre-encoding `language` field the
    # DRIVER_TEMPLATES dict below covers — caught this by testing against
    # a real cold-start vector, where it fell through to the raw-name
    # fallback. Handled as a pattern rather than 3 near-duplicate entries.
    if feature.startswith("language_"):
        lang = feature[len("language_"):]
        is_that_language = "is" if value == 1 else "is not"
        return f"this project {is_that_language} written in {lang.capitalize()}, which this model associates with {'higher' if increases else 'lower'} risk here"

    template = DRIVER_TEMPLATES.get(feature)
    if template:
        try:
            return template(value, increases)
        except Exception:
            pass
    arrow = "higher than typical" if increases else "lower than typical"
    return f"`{feature}` is {arrow} for this build (raw feature name — no plain-language mapping written for it yet)"


def shallow_history_note(features: dict) -> str:
    build_count = features.get("project_prior_build_count") or 0
    repo_age = features.get("repo_age_days") or 0
    standing = (
        "Live build history here only goes back to whenever GitHub Actions was enabled on "
        "this repo, not the repo's actual age — an established project with Actions enabled "
        "recently can look history-thin and get over-flagged for that reason alone."
    )
    if build_count < SHALLOW_BUILD_COUNT_THRESHOLD and repo_age > SHALLOW_REPO_AGE_DAYS_THRESHOLD:
        return (
            f"⚠️ **Shallow history detected:** this repo is ~{int(repo_age)} days old but only "
            f"{int(build_count)} prior build(s) were found in its Actions history. {standing}"
        )
    return f"_{standing}_"


def secondary_line(probability: float, tier: str) -> str:
    # Deliberately does NOT surface the API's own probability_confidence
    # field ("normal"/"low") here: that label means "did this model have
    # history to work with", not "is this number trustworthy". Every
    # number shown outward from this Action is experimental/low-confidence
    # regardless of that internal state — the cross-CI-system miscalibration
    # applies whether or not history was present, and putting the word
    # "normal" next to "experimental" would read as reassurance it isn't.
    return (
        f"\n\n<sub>Reference only, not a calibrated risk assessment — experimental, "
        f"low-confidence signal leans **{tier}** (raw model output: {probability:.0%}). "
        f"See the banner above before acting on this.</sub>"
    )


def compose_ok_comment(result: dict, features: dict) -> str:
    drivers = "\n".join(f"- {explain_driver(c)}" for c in result["top_contributors"])
    body = (
        f"{EXPERIMENTAL_BANNER}\n\n"
        f"### What this build's history and change characteristics suggest\n\n{drivers}\n\n"
        f"{shallow_history_note(features)}"
        f"{secondary_line(result['failure_probability'], result['risk_tier'])}"
    )
    return body


def compose_cold_start_comment(result: dict, features: dict) -> str:
    non_null_drivers = [c for c in result["top_contributors"]]
    drivers_block = ""
    if non_null_drivers:
        drivers = "\n".join(f"- {explain_driver(c)}" for c in non_null_drivers)
        drivers_block = f"\n\n**Signal available from this commit's change characteristics** (history-independent):\n\n{drivers}"
    return (
        f"{EXPERIMENTAL_BANNER}\n\n"
        f"### Insufficient build history for a reliable signal\n\n"
        f"This repository has no qualifying prior build recorded in GitHub Actions history yet "
        f"(or a data inconsistency was detected — reason: `{result['cold_start']['reason']}`). "
        f"**No risk tier is being shown** — about two-thirds of what this model relies on is "
        f"prior-build history, and guessing at a tier here would be worse than saying nothing. "
        f"See [cold-start handling]({COLDSTART_DOC_URL}) for the full reasoning."
        f"{drivers_block}\n\n{shallow_history_note(features)}"
    )


def compose_could_not_score(reason_kind: str, detail: str) -> str:
    reasons = {
        "rate_limit": "the GitHub API rate limit was exhausted while gathering this commit's history",
        "extraction_error": "required data for this commit could not be retrieved from the GitHub API",
        "model_not_loaded": "the trained model artifact was not available in this run",
        "invalid_features": "the extracted feature vector failed validation against the model's expected schema",
        "unexpected": "an unexpected error occurred while generating a prediction",
    }
    reason_text = reasons.get(reason_kind, reasons["unexpected"])
    return (
        f"{EXPERIMENTAL_BANNER}\n\n"
        f"### Could not generate a signal for this commit\n\n"
        f"{reason_text.capitalize()}. **This is not a signal that anything is wrong with the "
        f"code** — the demo pipeline simply couldn't retrieve or validate what it needed this "
        f"run, so it is reporting that honestly rather than posting a fabricated or zeroed "
        f"prediction (see the [feature-parity report]({PARITY_REPORT_URL}) for why that "
        f"matters).\n\n<sub>Detail: `{detail[:300]}`</sub>"
    )


def status_description(kind: str, result: dict = None) -> str:
    """Commit status descriptions are capped at 140 chars by GitHub — keep
    these short; the full explanation lives in the PR comment."""
    if kind == "ok":
        return f"Experimental signal leans {result['risk_tier']} (low-confidence, cross-CI) — see PR/commit comment"[:140]
    if kind == "cold_start":
        return "Insufficient build history for a signal (cold start) — see comment"[:140]
    return "Could not generate a signal this run — see comment for why"[:140]


# --------------------------------------------------------------------------
# GitHub Actions event context
# --------------------------------------------------------------------------
def get_event_context():
    event_name = os.environ["GITHUB_EVENT_NAME"]
    owner, repo = os.environ["GITHUB_REPOSITORY"].split("/", 1)
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    event = json.loads(Path(event_path).read_text()) if event_path else {}

    if event_name == "pull_request":
        pr = event["pull_request"]
        return {
            "owner": owner, "repo": repo, "sha": pr["head"]["sha"], "branch": pr["head"]["ref"],
            "is_pr": True, "pr_number": pr["number"],
        }
    return {
        "owner": owner, "repo": repo, "sha": os.environ["GITHUB_SHA"],
        "branch": os.environ.get("GITHUB_REF_NAME"), "is_pr": False, "pr_number": None,
    }


# --------------------------------------------------------------------------
# Posting — commit status ALWAYS state="success" (advisory only, never
# blocks merges), PR comment only on pull_request events.
# --------------------------------------------------------------------------
def post_commit_status(ctx, description, token):
    url = f"https://api.github.com/repos/{ctx['owner']}/{ctx['repo']}/statuses/{ctx['sha']}"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={"state": "success", "context": STATUS_CONTEXT, "description": description},
        timeout=15,
    )
    print(f"POST status -> {resp.status_code}")
    if resp.status_code >= 300:
        print(resp.text, file=sys.stderr)


def post_pr_comment(ctx, body, token):
    url = f"https://api.github.com/repos/{ctx['owner']}/{ctx['repo']}/issues/{ctx['pr_number']}/comments"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={"body": body},
        timeout=15,
    )
    print(f"POST PR comment -> {resp.status_code}")
    if resp.status_code >= 300:
        print(resp.text, file=sys.stderr)


def publish(ctx, token, status_kind, result=None):
    body = {
        "ok": lambda: compose_ok_comment(result["prediction"], result["features"]),
        "cold_start": lambda: compose_cold_start_comment(result["prediction"], result["features"]),
    }.get(status_kind, lambda: compose_could_not_score(result["reason_kind"], result["detail"]))()

    post_commit_status(ctx, status_description(status_kind, result.get("prediction") if result else None), token)
    if ctx["is_pr"]:
        post_pr_comment(ctx, body, token)
    print("\n" + "=" * 80 + "\nPOSTED BODY:\n" + "=" * 80 + f"\n{body}\n")


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("No GITHUB_TOKEN available — cannot post anything. Exiting quietly.", file=sys.stderr)
        return
    ctx = get_event_context()
    print(f"Context: {ctx}")

    try:
        extraction = build_feature_vector(
            ctx["owner"], ctx["repo"], ctx["sha"], branch=ctx["branch"], is_pr=ctx["is_pr"],
            client=GitHubClient(token=token),
        )
    except GitHubRateLimitError as exc:
        publish(ctx, token, "could_not_score", {"reason_kind": "rate_limit", "detail": str(exc)})
        return
    except Exception as exc:
        publish(ctx, token, "could_not_score", {"reason_kind": "extraction_error", "detail": repr(exc)})
        return

    service = PredictionService()
    if not service.is_loaded:
        publish(ctx, token, "could_not_score", {"reason_kind": "model_not_loaded", "detail": service.load_error or ""})
        return

    try:
        features_obj = BuildFeatures(**extraction.features)
    except ValidationError as exc:
        publish(ctx, token, "could_not_score", {"reason_kind": "invalid_features", "detail": str(exc)})
        return

    prediction = service.predict(features_obj)
    kind = "cold_start" if prediction["status"] == STATUS_COLD_START else "ok"
    publish(ctx, token, kind, {"prediction": prediction, "features": extraction.features})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - last resort: this job must never fail the build
        print(f"UNEXPECTED ERROR (job still exits 0): {exc!r}", file=sys.stderr)
    sys.exit(0)
